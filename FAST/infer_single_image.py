#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
단일 이미지 FAST Detection 추론 스크립트 (train_fast_from_lmdb.py로 학습한 체크포인트용)

사용 예:
python FAST/infer_single_image.py \
  --checkpoint /home/mango/ocr_test/outputs/fast_lmdb_train/checkpoint_epoch_1.pth \
  --image /home/mango/ocr_test/FAST/5350034-2011-0001-0019.jpg \
  --output /home/mango/ocr_test/FAST/5350034-2011-0001-0019_det.png \
  --config FAST/config/fast/korean_ocr/multi_lmdb_config.py \
  --device cuda
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")  # GUI 백엔드 비활성화(블로킹 방지)
import argparse
import json
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 프로젝트 루트/FAST 모듈 경로 추가
ROOT = '/home/mango/ocr_test'
FAST_DIR = os.path.join(ROOT, 'FAST')
sys.path.insert(0, ROOT)
sys.path.insert(0, FAST_DIR)

from mmcv import Config, ConfigDict
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
try:
    from peft import PeftModel
except Exception:
    PeftModel = None
from models import build_model
from dataset.utils import scale_aligned_short
import torchvision.transforms as transforms


def _try_load_cfg_from_checkpoint(ckpt):
    # mmdet 계열 체크포인트에서 자주 쓰이는 키들을 탐색해 cfg를 추출
    for key in ['cfg', 'config']:
        if key in ckpt and ckpt[key] is not None:
            data = ckpt[key]
            try:
                if isinstance(data, str):
                    # 문자열이면 파일 경로 또는 Python dict 문자열일 수 있음
                    if os.path.exists(data):
                        return Config.fromfile(data)
                    try:
                        parsed = json.loads(data)
                        return Config(parsed)
                    except Exception:
                        pass
                if isinstance(data, dict):
                    return Config(data)
            except Exception:
                pass
    if 'meta' in ckpt and ckpt['meta'] is not None:
        meta = ckpt['meta']
        for key in ['cfg', 'config']:
            if key in meta and meta[key] is not None:
                try:
                    if isinstance(meta[key], dict):
                        return Config(meta[key])
                except Exception:
                    pass
    return None


def load_model(cfg_path, checkpoint_path, device_str='cuda'):
    device = torch.device(device_str if torch.cuda.is_available() and device_str == 'cuda' else 'cpu')

    ckpt = torch.load(checkpoint_path, map_location=device)

    # 구성 로드: 우선 인자 cfg, 없으면 ckpt에서 시도
    if cfg_path and os.path.exists(cfg_path):
        cfg = Config.fromfile(cfg_path)
    else:
        cfg = _try_load_cfg_from_checkpoint(ckpt)
        if cfg is None:
            raise RuntimeError('구성(cfg)을 찾을 수 없습니다. --config 경로를 제공해 주세요.')

    # 보호: cfg.test_cfg 기본값 주입
    try:
        if not hasattr(cfg, 'test_cfg') or cfg.test_cfg is None:
            cfg.test_cfg = ConfigDict(dict(min_area=5, min_score=0.3, bbox_type='rect'))
    except Exception:
        pass

    model = build_model(cfg.model)
    model = model.to(device)

    # state_dict 가져오기 (ema/state_dict/직접)
    state_dict = ckpt.get('state_dict', ckpt.get('ema', ckpt))
    # DataParallel 호환
    def _strip_prefix(k: str) -> str:
        # torch.compile 래퍼: '_orig_mod.' 제거, DDP: 'module.' 제거
        return k.replace('_orig_mod.', '').replace('module.', '')
    new_state = {_strip_prefix(k): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f'⚠️ missing keys: {len(missing)}')
    if unexpected:
        print(f'⚠️ unexpected keys: {len(unexpected)}')

    model.eval()
    return model, cfg, device


def load_trocr(trocr_path: str, device: torch.device):
    """
    TrOCR 모델과 프로세서를 체크포인트에서 불러옵니다.
    - trocr_path는 `checkpoint-60000` 같은 디렉터리를 기대합니다.
    - PEFT 어댑터(LoRA/DoRA)가 포함된 상태로 저장되어 있다면 자동으로 로드됩니다.
    """
    # 가능한 경우: 디렉터리에 전체 모델이 저장된 경우 그대로 로드
    try:
        processor = TrOCRProcessor.from_pretrained(trocr_path)
    except Exception:
        # processor가 디렉터리에 없으면 베이스에서 로드
        base_name = os.environ.get("TROCR_BASE", "ddobokki/ko-trocr")
        processor = TrOCRProcessor.from_pretrained(base_name)

    try:
        model = VisionEncoderDecoderModel.from_pretrained(trocr_path)
        model.to(device)
        model.eval()
        return processor, model
    except Exception as e:
        # 어댑터만 저장된 경우: 베이스 모델 + PEFT 어댑터 주입
        if PeftModel is None:
            raise e
        base_name = os.environ.get("TROCR_BASE", "ddobokki/ko-trocr")
        base_model = VisionEncoderDecoderModel.from_pretrained(base_name)
        # 어댑터 로드 전, 토크나이저 크기에 맞춰 디코더 임베딩 리사이즈
        try:
            target_vocab_size = len(getattr(processor, "tokenizer", {}))
            if isinstance(target_vocab_size, int) and target_vocab_size > 0:
                if hasattr(base_model, "decoder") and hasattr(base_model.decoder, "resize_token_embeddings"):
                    base_model.decoder.resize_token_embeddings(target_vocab_size)
                base_model.config.vocab_size = target_vocab_size
        except Exception:
            pass

        # 1) 루트에 단일 어댑터 형식(adapter_config.json 등) 존재 시
        adapter_cfg = os.path.join(trocr_path, "adapter_config.json")
        adapter_bin = os.path.join(trocr_path, "adapter_model.bin")
        adapter_safetensors = os.path.join(trocr_path, "adapter_model.safetensors")
        if os.path.exists(adapter_cfg) and (os.path.exists(adapter_bin) or os.path.exists(adapter_safetensors)):
            model = PeftModel.from_pretrained(base_model, trocr_path)
            model.to(device)
            model.eval()
            return processor, model

        # 2) 서브디렉터리(encoder_dora/decoder_lora)에 각 어댑터가 저장된 형식
        enc_dir = os.path.join(trocr_path, "encoder_dora")
        dec_dir = os.path.join(trocr_path, "decoder_lora")
        def _has_adapter(dir_path: str) -> bool:
            return os.path.isdir(dir_path) and os.path.exists(os.path.join(dir_path, "adapter_config.json")) and \
                   (os.path.exists(os.path.join(dir_path, "adapter_model.bin")) or os.path.exists(os.path.join(dir_path, "adapter_model.safetensors")))

        if _has_adapter(dec_dir) or _has_adapter(enc_dir):
            model = base_model
            # 디코더 어댑터 우선 로드
            if _has_adapter(dec_dir):
                model = PeftModel.from_pretrained(model, dec_dir, adapter_name="decoder_lora")
            else:
                model = PeftModel.from_pretrained(model, enc_dir, adapter_name="encoder_dora")
            # 나머지 어댑터 추가 로드
            if _has_adapter(enc_dir) and hasattr(model, "load_adapter"):
                model.load_adapter(enc_dir, adapter_name="encoder_dora")
            if _has_adapter(dec_dir) and hasattr(model, "load_adapter"):
                try:
                    model.load_adapter(dec_dir, adapter_name="decoder_lora")
                except Exception:
                    pass
            model.to(device)
            model.eval()
            return processor, model

        # 해당 형식 모두 아니면 원래 오류 전달
        raise e


def preprocess(image_path, short_size):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    org_h, org_w = img.shape[:2]
    img_resized = scale_aligned_short(img, short_size)
    proc_h, proc_w = img_resized.shape[:2]
    pil_img = Image.fromarray(img_resized).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(pil_img).unsqueeze(0)
    # 모델 head가 img_meta['org_img_size'][0] 형태를 기대하므로 리스트로 감싼다
    meta = {
        'org_img_size': [np.array([org_h, org_w])],
        'img_size': [np.array([proc_h, proc_w])],
        'filename': [os.path.basename(image_path)]
    }
    return tensor, meta, (org_h, org_w), (proc_h, proc_w), img


def visualize(image_rgb, dets, org_size, proc_size, output_path=None):
    # fast_head.get_results는 내부에서 원본 좌표계로 복원하므로 추가 스케일링은 하지 않는다
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_rgb)
    ax.axis('off')
    # dets는 head.get_results에서 반환된 outputs['results']
    # 형태: [ { 'bboxes': np.ndarray(M, K), 'scores': np.ndarray(M) } ]
    boxes_drawn = 0
    if dets:
        res = dets[0] if isinstance(dets, list) else dets
        bboxes = res.get('bboxes') if isinstance(res, dict) else None
        if bboxes is not None:
            try:
                import numpy as np
                for bb in bboxes:
                    bb = np.array(bb).reshape(-1)
                    if bb.size >= 8:
                        xs = bb[0::2]; ys = bb[1::2]
                        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                    elif bb.size >= 4:
                        x1, y1, x2, y2 = bb[:4]
                    else:
                        continue
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                             linewidth=2, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    boxes_drawn += 1
            except Exception:
                pass
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f'💾 저장: {output_path}')
        plt.close(fig)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='FAST 단일 이미지 Detection 추론')
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--image', required=True, type=str)
    parser.add_argument('--output', default=None, type=str)
    parser.add_argument('--config', default=None, type=str, help='선택: cfg 미포함 ckpt인 경우 필요')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--trocr_checkpoint', type=str, default=None, help='TrOCR 학습 체크포인트 디렉터리 (예: /home/.../checkpoint-60000)')
    args = parser.parse_args()

    model, cfg, device = load_model(args.config, args.checkpoint, args.device)
    trocr_processor = None
    trocr_model = None
    if args.trocr_checkpoint:
        trocr_processor, trocr_model = load_trocr(args.trocr_checkpoint, device)
    short_size = getattr(getattr(cfg, 'data', None), 'test', None)
    if short_size is not None:
        short_size = getattr(cfg.data.test, 'short_size', 736)
    else:
        short_size = 736

    img_tensor, meta, org_size, proc_size, img_rgb = preprocess(args.image, short_size)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        # get_results는 generate_bbox 내부에서 cfg.test_cfg를 참조하므로 전체 cfg를 전달해야 함
        outputs = model(img_tensor, img_metas=meta, cfg=cfg)
    # forward는 head.get_results의 반환을 outputs.update(...)로 합치므로 'results' 키를 읽는다
    dets = outputs.get('results', outputs)
    if isinstance(dets, dict) and 'results' in dets:
        dets = dets['results']

    # 결과 통계
    n = 0
    if dets:
        res0 = dets[0] if isinstance(dets, list) else dets
        bboxes = res0.get('bboxes') if isinstance(res0, dict) else None
        n = 0 if bboxes is None else len(bboxes)
    print(f'✅ 검출 수: {n}')

    visualize(img_rgb, dets, org_size, proc_size, args.output)

    # 선택: 검출된 영역에 대해 TrOCR로 간단히 인식 시연(영역 수가 많을 수 있으므로 상위 몇 개만)
    if trocr_model is not None and dets and isinstance(dets, list) and len(dets) > 0:
        res = dets[0]
        bboxes = res.get('bboxes')
        scores = res.get('scores')
        if bboxes is not None and scores is not None:
            top_k = min(5, len(bboxes))
            print(f"\n[TrOCR demo] Top-{top_k} boxes recognition from {args.trocr_checkpoint}")
            for i in range(top_k):
                bb = np.array(bboxes[i]).reshape(-1)
                if bb.size >= 8:
                    xs = bb[0::2]; ys = bb[1::2]
                    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                elif bb.size >= 4:
                    x1, y1, x2, y2 = map(int, bb[:4])
                else:
                    continue
                crop = img_rgb[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                pil = Image.fromarray(crop)
                pv = trocr_processor(pil, return_tensors="pt").pixel_values.to(device)
                gen = trocr_model.generate(pv, max_length=64)
                text = trocr_processor.batch_decode(gen, skip_special_tokens=True)[0]
                print(f"  [{i}] {text}")


if __name__ == '__main__':
    main()


