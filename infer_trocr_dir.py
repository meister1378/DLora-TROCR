#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
디렉터리 내 이미지들을 현재 TrOCR 체크포인트로 일괄 인퍼런스하여 CSV로 저장합니다.

사용 예:
python infer_trocr_dir.py \
  --images_dir /home/mango/ocr_test/debug_samples \
  --output_csv /home/mango/ocr_test/debug_infer.csv \
  --ckpt /home/mango/ocr_test/output/dlora_ko_trocr_multi/checkpoint-60000 \
  --base ddobokki/ko-trocr --device cuda
"""
import os
import sys
import csv
import argparse
from typing import List

from PIL import Image
from tqdm import tqdm

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


def _gather_images(root: str, patterns: List[str]) -> List[str]:
    exts = tuple(sum([p.split(',') for p in patterns], [])) if patterns else ()
    if not patterns:
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            low = fn.lower()
            if low.endswith(exts):
                files.append(os.path.join(dirpath, fn))
    files.sort()
    return files


def _load_trocr(ckpt: str, base: str, device: torch.device):
    # processor 우선 로드
    try:
        if ckpt:
            processor = TrOCRProcessor.from_pretrained(ckpt)
        else:
            raise ValueError("no-ckpt")
    except Exception:
        processor = TrOCRProcessor.from_pretrained(base)

    # 전체 저장 형식 우선 시도
    try:
        if ckpt:
            model = VisionEncoderDecoderModel.from_pretrained(ckpt)
            model.to(device)
            model.eval()
            return processor, model
    except Exception as e:
        # 어댑터만 저장된 형식 처리
        if PeftModel is None or not ckpt:
            # ckpt가 없거나 어댑터 경로가 아니면 베이스만 사용
            base_model = VisionEncoderDecoderModel.from_pretrained(base)
            base_model.to(device)
            base_model.eval()
            return processor, base_model
        base_model = VisionEncoderDecoderModel.from_pretrained(base)
        # 토크나이저 길이에 맞춰 임베딩 리사이즈
        try:
            vocab_size = len(getattr(processor, 'tokenizer', {}))
            if isinstance(vocab_size, int) and vocab_size > 0 and hasattr(base_model, 'decoder'):
                if hasattr(base_model.decoder, 'resize_token_embeddings'):
                    base_model.decoder.resize_token_embeddings(vocab_size)
                base_model.config.vocab_size = vocab_size
        except Exception:
            pass

        adapter_cfg = os.path.join(ckpt, 'adapter_config.json')
        adapter_bin = os.path.join(ckpt, 'adapter_model.bin')
        adapter_st = os.path.join(ckpt, 'adapter_model.safetensors')
        if os.path.exists(adapter_cfg) and (os.path.exists(adapter_bin) or os.path.exists(adapter_st)):
            model = PeftModel.from_pretrained(base_model, ckpt)
        else:
            # 서브 디렉터리 형식(encoder_dora/decoder_lora)
            enc_dir = os.path.join(ckpt, 'encoder_dora')
            dec_dir = os.path.join(ckpt, 'decoder_lora')
            def _has_adapter(d: str) -> bool:
                return os.path.isdir(d) and os.path.exists(os.path.join(d, 'adapter_config.json')) and (os.path.exists(os.path.join(d, 'adapter_model.bin')) or os.path.exists(os.path.join(d, 'adapter_model.safetensors')))
            if _has_adapter(dec_dir):
                model = PeftModel.from_pretrained(base_model, dec_dir, adapter_name='decoder_lora')
                if _has_adapter(enc_dir) and hasattr(model, 'load_adapter'):
                    model.load_adapter(enc_dir, adapter_name='encoder_dora')
            elif _has_adapter(enc_dir):
                model = PeftModel.from_pretrained(base_model, enc_dir, adapter_name='encoder_dora')
            else:
                raise e

        model.to(device)
        model.eval()
        return processor, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images_dir', type=str, default=os.environ.get('DEBUG_IMAGES', '/home/mango/ocr_test/debug_samples'))
    ap.add_argument('--output_csv', type=str, default='/home/mango/ocr_test/debug_infer.csv')
    ap.add_argument('--ckpt', type=str, default=os.environ.get('TROCR_CKPT', ''))
    ap.add_argument('--base', type=str, default=os.environ.get('TROCR_BASE', 'ddobokki/ko-trocr'))
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--num_beams', type=int, default=int(os.environ.get('TROCR_BEAMS', '1')))
    ap.add_argument('--max_len', type=int, default=int(os.environ.get('TROCR_MAX_LEN', '32')))
    ap.add_argument('--repetition_penalty', type=float, default=float(os.environ.get('TROCR_REP_PEN', '1.2')))
    ap.add_argument('--no_repeat_ngram', type=int, default=int(os.environ.get('TROCR_NO_REPEAT', '3')))
    ap.add_argument('--length_penalty', type=float, default=float(os.environ.get('TROCR_LEN_PEN', '0.8')))
    ap.add_argument('--clean_repeats', action='store_true', default=True)
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    ckpt = getattr(args, 'ckpt', '') or None
    if ckpt and not (os.path.isdir(ckpt) or '/' in ckpt or ckpt.count('-') or ckpt.count('/')):
        # 단순 문자열이면서 로컬 디렉터리도 아닌 경우 취소
        ckpt = None
    processor, model = _load_trocr(ckpt, args.base, device)

    files = _gather_images(args.images_dir, [])
    if not files:
        print(f"❌ No images found under: {args.images_dir}")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    import re
    def _clean(text: str) -> str:
        if not args.clean_repeats:
            return text
        t = re.sub(r'\s+', ' ', text).strip()
        # 과도 반복 완화: 동일 문자 3회 이상 연속 → 2회로 축소
        t = re.sub(r'(.)\1{2,}', r'\1\1', t)
        return t

    with open(args.output_csv, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'text'])
        for path in tqdm(files, desc='TrOCR infer'):
            try:
                pil = Image.open(path).convert('RGB')
                enc = processor(images=pil, return_tensors='pt')
                pv = enc.pixel_values.to(device)
                ids = model.generate(
                    pixel_values=pv,
                    max_length=args.max_len,
                    num_beams=args.num_beams,
                    early_stopping=True,
                    no_repeat_ngram_size=args.no_repeat_ngram,
                    repetition_penalty=args.repetition_penalty,
                    length_penalty=args.length_penalty,
                )
                text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
                text = _clean(text)
            except Exception as e:
                text = f"<ERROR: {e}>"
            writer.writerow([path, text])

    print(f"✅ Saved: {args.output_csv} ({len(files)} items)")


if __name__ == '__main__':
    main()


