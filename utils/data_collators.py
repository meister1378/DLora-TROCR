from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from transformers import TrOCRProcessor
from PIL import Image
import numpy as np
import torch
import os
from time import perf_counter as _pc

from literal import DatasetColumns


@dataclass
class DataCollatorForOCR:
    processor: TrOCRProcessor
    padding = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        _t0 = _pc()
        features = [f for f in features if f is not None]
        if not features:
            # 모든 샘플이 None이라 배치가 비게 되면, 빈 딕셔너리를 반환하여 Trainer가 이 배치를 건너뛰게 함
            print("[DataCollator] Warning: All samples in the batch were invalid (None). Skipping batch.")
            return {}
        
        def resolve_image(feat: Any):
            if not isinstance(feat, dict):
                return feat
            for k in [DatasetColumns.pixel_values, "image", "img", "imgs", "raw_img", "pixel", "pixel_value"]:
                if k in feat and feat[k] is not None:
                    v = feat[k]
                    # Accept only PIL or valid numpy/tensor image-like shapes
                    if isinstance(v, Image.Image):
                        return v
                    if isinstance(v, np.ndarray):
                        # 최소 크기 및 형태 검증
                        if v.size < 100:
                            return None
                        if v.ndim == 3 and v.shape[0] == 1 and v.shape[1] == 1:
                            return None
                        if v.ndim == 2 and (v.shape[0] < 5 or v.shape[1] < 5):
                            return None
                        if v.ndim == 3 and (v.shape[0] < 5 or v.shape[1] < 5):
                            return None
                        if (v.ndim == 2) or (v.ndim == 3 and v.shape[-1] in (1, 3)):
                            return v
                        return None
                    if isinstance(v, torch.Tensor):
                        if v.dim() == 2:
                            return v
                        if v.dim() == 3 and (v.shape[-1] in (1, 3) or v.shape[0] in (1, 3)):
                            return v
                        return None
                    return None
            for v in feat.values():
                if isinstance(v, Image.Image):
                    return v
                if isinstance(v, np.ndarray):
                    # 최소 크기 및 형태 검증
                    if v.size < 100:
                        continue
                    if v.ndim == 3 and v.shape[0] == 1 and v.shape[1] == 1:
                        continue
                    if v.ndim == 2 and (v.shape[0] < 5 or v.shape[1] < 5):
                        continue
                    if v.ndim == 3 and (v.shape[0] < 5 or v.shape[1] < 5):
                        continue
                    if (v.ndim == 2) or (v.ndim == 3 and v.shape[-1] in (1, 3)):
                        return v
                    continue
                if isinstance(v, torch.Tensor):
                    if v.dim() == 2:
                        return v
                    if v.dim() == 3 and (v.shape[-1] in (1, 3) or v.shape[0] in (1, 3)):
                        return v
                    continue
            return None

        images = []
        kept = []
        dropped = []
        for feature in features:
            img = resolve_image(feature)
            if img is None:
                dropped.append({k: type(v).__name__ for k, v in feature.items()}) if isinstance(feature, dict) else dropped.append(type(feature).__name__)
                continue
            images.append(img)
            kept.append(feature)
            
        if not images:
            # 스캔 이후 read_image 복구 누락 등으로 None이 들어오면 배치 스킵
            print("[BatchDebug] All invalid features:", dropped)
            # 여기서는 빈 딕셔너리를 반환하는 것이 더 안전할 수 있습니다.
            # raise KeyError("No valid images in batch; all samples missing/invalid image fields")
            print("[DataCollator] Warning: No valid images found in the batch after filtering. Skipping batch.")
            return {}
        
        # 이미지들을 PIL.Image로 강제 정규화하여 PIL이 처리 가능한 형태(HxW 또는 HxWxC, C∈{1,3})로 맞춘다
        def to_pil_safe(x):
            try:
                if isinstance(x, Image.Image):
                    return x
                if isinstance(x, torch.Tensor):
                    x = x.detach().cpu()
                    if x.dim() == 3:
                        if x.shape[0] in (1, 3):
                            x = x.permute(1, 2, 0)
                        # else assume HWC
                    elif x.dim() == 2:
                        pass
                    x = x.numpy()
                if isinstance(x, np.ndarray):
                    arr = x
                    
                    # 비정상적인 형태 필터링 (너무 작은 이미지)
                    if arr.size < 100:  # 최소 10x10 크기
                        return None
                    
                    # move to HWC if needed
                    if arr.ndim == 3:
                        # 비정상적인 형태: (1,1,N) 같은 경우 거부
                        if arr.shape[0] == 1 and arr.shape[1] == 1:
                            return None
                        
                        if arr.shape[-1] in (1, 3):
                            pass
                        elif arr.shape[0] in (1, 3):
                            arr = np.transpose(arr, (1, 2, 0))
                        else:
                            # try squeeze a singleton dim then recover
                            squeezed = arr.squeeze()
                            if squeezed.ndim == 2:
                                arr = squeezed
                            else:
                                # as a last resort, treat as (C,H,W) with C=1
                                arr = np.transpose(arr, (1, 2, 0))
                    
                    # 최소 크기 검증 (H, W 모두 5 이상)
                    if arr.ndim == 2 and (arr.shape[0] < 5 or arr.shape[1] < 5):
                        return None
                    if arr.ndim == 3 and (arr.shape[0] < 5 or arr.shape[1] < 5):
                        return None
                    # dtype to uint8
                    if arr.dtype != np.uint8:
                        a = arr.astype(np.float32)
                        a_min, a_max = float(a.min()), float(a.max())
                        if a_max <= 1.0:
                            a = a * 255.0
                        elif a_max > 255.0 or a_min < 0.0:
                            a = 255.0 * (a - a_min) / (a_max - a_min + 1e-8)
                        arr = np.clip(a, 0, 255).astype(np.uint8)
                    return Image.fromarray(arr)
            except Exception:
                return None

        pil_images = []
        new_kept = []
        for img, feat in zip(images, kept):
            pil = to_pil_safe(img)
            if pil is None:
                continue
            # 추가 검증: PIL 이미지의 크기 확인
            try:
                if pil.size[0] < 5 or pil.size[1] < 5:
                    continue
            except Exception:
                continue
            pil_images.append(pil)
            new_kept.append(feat)

        if not pil_images:
            print("[DataCollator] Warning: All images failed PIL normalization. Skipping batch.")
            return {}

        images = pil_images
        kept = new_kept

        # Processor로 전달하기 전 최종 검증
        try:
            batch = self.processor(images, return_tensors=self.return_tensors)
        except Exception as e:
            print(f"[DataCollator] Error in processor: {e}. Skipping batch.")
            return {}
        
        if kept and isinstance(kept[0], dict) and DatasetColumns.labels in kept[0]:
            texts = [f.get(DatasetColumns.labels, "") for f in kept]
            # 라벨 NFKD 정규화 (워커 토크나이즈 경로 포함)
            try:
                import unicodedata as _ud
                texts = [_ud.normalize("NFKD", (t or "")) for t in texts]
            except Exception:
                pass
            if os.environ.get("DISABLE_LABEL_TOKENIZE_IN_WORKER", "0") == "1":
                batch["labels_texts"] = texts
            else:
                # AddedVocabulary bad split 방지: 제어/제로폭/공백 문자 제거
                import unicodedata as _ud
                def _clean_text(t: str) -> str:
                    if not isinstance(t, str):
                        return ""
                    t = t.replace("\r", " ").replace("\n", " ").replace("\t", " ")
                    out = []
                    for ch in t:
                        try:
                            cat = _ud.category(ch)
                            if cat and cat[0] == 'C':
                                continue
                            if ch.isspace():
                                out.append(' ')
                                continue
                            if ord(ch) in (0x200B, 0x200C, 0x200D, 0xFEFF):
                                continue
                            out.append(ch)
                        except Exception:
                            continue
                    return "".join(out).strip()

                texts = [_clean_text(t) for t in texts]
                labels = self.processor.tokenizer(
                    texts, padding=self.padding, return_tensors=self.return_tensors
                ).input_ids.to(torch.long)
                pad_id = self.processor.tokenizer.pad_token_id
                labels[labels == pad_id] = -100
                # HF DataCollatorForSeq2Seq 정합: pad_to_multiple, DEC_MAX_LEN, DEC_PAD_TO_MAX 적용
                try:
                    multiple = int(os.environ.get("PAD_TO_MULTIPLE", "0"))
                except Exception:
                    multiple = 0
                try:
                    dec_max_len = int(os.environ.get("DEC_MAX_LEN", "0"))
                except Exception:
                    dec_max_len = 0
                dec_pad_to_max = os.environ.get("DEC_PAD_TO_MAX", "0") == "1"
                if labels.dim() == 2:
                    T = labels.size(1)
                    target_T = T
                    if dec_pad_to_max and dec_max_len and dec_max_len > 0 and target_T < dec_max_len:
                        target_T = dec_max_len
                    if multiple and multiple > 1:
                        target_T = ((target_T + multiple - 1) // multiple) * multiple
                    pad_len = max(0, target_T - T)
                    if pad_len > 0:
                        labels = torch.nn.functional.pad(labels, (0, pad_len), value=-100)
                batch["labels"] = labels
                try:
                    batch["decoder_attention_mask"] = (labels != -100).long()
                except Exception:
                    pass
        # Collate profiling (optional)
        if os.environ.get("PROFILE_COLLATE", "0") == "1":
            try:
                batch["collate_ms"] = int(((_pc() - _t0) * 1000))
            except Exception:
                pass
        
        return batch

@dataclass
class DataCollatorForGptOCR:
    processor: TrOCRProcessor
    padding = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        _t0 = _pc()

        def resolve_image(feat: Any):
            if not isinstance(feat, dict):
                return feat
            for k in [DatasetColumns.pixel_values, "image", "img", "imgs", "raw_img", "pixel", "pixel_value"]:
                if k in feat and feat[k] is not None:
                    return feat[k]
            for v in feat.values():
                if isinstance(v, Image.Image):
                    return v
                if isinstance(v, np.ndarray) and v.ndim >= 2:
                    return v
                if isinstance(v, torch.Tensor) and v.dim() >= 2:
                    return v
            return None

        images = []
        kept = []
        dropped = []
        for feature in features:
            img = resolve_image(feature)
            if img is None:
                dropped.append({k: type(v).__name__ for k, v in feature.items()}) if isinstance(feature, dict) else dropped.append(type(feature).__name__)
                continue
            images.append(img)
            kept.append(feature)
        if not images:
            print("[BatchDebug] All invalid features:", dropped)
            raise KeyError("No valid images in batch; all samples missing/invalid image fields")
        batch = self.processor(images, return_tensors=self.return_tensors, input_data_format="channels_first")
        if kept and isinstance(kept[0], dict):
            texts = [ (f.get(DatasetColumns.labels, "")) for f in kept ]
            if os.environ.get("DISABLE_LABEL_TOKENIZE_IN_WORKER", "0") == "1":
                batch["labels_texts"] = texts
            else:
                texts = [
                    self.processor.tokenizer.bos_token
                    + t
                    + self.processor.tokenizer.eos_token
                    for t in texts
                ]
                labels = self.processor.tokenizer(
                    texts, padding=self.padding, return_tensors=self.return_tensors
                ).input_ids
                pad_id = self.processor.tokenizer.pad_token_id
                labels[labels == pad_id] = -100
                # PAD_TO_MULTIPLE: 동적 길이로 인한 재컴파일/메모리 증가 완화
                try:
                    multiple = int(os.environ.get("PAD_TO_MULTIPLE", "0"))
                except Exception:
                    multiple = 0
                if multiple and multiple > 1 and labels.dim() == 2:
                    T = labels.size(1)
                    new_T = ((T + multiple - 1) // multiple) * multiple
                    pad_len = new_T - T
                    if pad_len > 0:
                        labels = torch.nn.functional.pad(labels, (0, pad_len), value=-100)
                batch["labels"] = labels
                try:
                    batch["decoder_attention_mask"] = (labels != -100).long()
                except Exception:
                    pass
        # Collate profiling (optional)
        if os.environ.get("PROFILE_COLLATE", "0") == "1":
            try:
                batch["collate_ms"] = int(((_pc() - _t0) * 1000))
            except Exception:
                pass
        return batch
