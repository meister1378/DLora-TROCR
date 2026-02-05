#202510161654
import logging
import re
import os
import random
import sys
from functools import partial
import sys
from io import BytesIO
import unicodedata
from PIL import Image
import warnings
import math
import time
import numpy as np

import torch
import torch.nn.functional as F
from setproctitle import setproctitle
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    GenerationConfig,
    TrOCRProcessor,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from transformers.trainer_utils import is_main_process

from arguments import DatasetsArguments, ModelArguments, MyTrainingArguments
from utils import DataCollatorForGptOCR, DataCollatorForOCR
from utils.augmentation import Augmentator
from utils.dataset_utils import get_dataset
from utils.training_utils import compute_metrics, has_unk_token, seed_everything

try:
    from peft import LoraConfig, inject_adapter_in_model, get_peft_model
    try:
        # QLoRA 4bit 훈련 준비 유틸 (입력 그라디언트/레이어 정밀도 구성 등)
        from peft import prepare_model_for_kbit_training
    except Exception:
        prepare_model_for_kbit_training = None
except ImportError:
    LoraConfig = None
    inject_adapter_in_model = None
    get_peft_model = None
    prepare_model_for_kbit_training = None
    
logger = logging.getLogger(__name__)

# QLoRA 적용 상태 진단 로그
def _log_qlora_diagnostics(model):
    try:
        logger.info("[QLoRA-Diag] ---- start ----")
        # 1) 학습 파라미터 그룹 요약
        enc_train, dec_train, other_train = 0, 0, 0
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "encoder_dora" in n:
                enc_train += int(p.numel())
            elif "decoder_lora" in n:
                dec_train += int(p.numel())
            else:
                other_train += int(p.numel())
        logger.info(f"[QLoRA-Diag] trainable encoder_dora={enc_train:,} decoder_lora={dec_train:,} other={other_train:,}")

        # 2) bitsandbytes 4bit 모듈 수
        try:
            import bitsandbytes as bnb  # type: ignore
            lin4_cnt = sum(1 for m in model.modules() if isinstance(m, bnb.nn.Linear4bit))
            logger.info(f"[QLoRA-Diag] bnb Linear4bit modules={lin4_cnt}")
        except Exception as _e:
            logger.info(f"[QLoRA-Diag] bitsandbytes not available: {_e}")

        # 3) quantization_config 요약
        try:
            qc = getattr(model, "quantization_config", None)
            if qc is not None:
                vals = {}
                for k in ["load_in_4bit", "bnb_4bit_quant_type", "bnb_4bit_use_double_quant", "bnb_4bit_compute_dtype"]:
                    try:
                        vals[k] = getattr(qc, k)
                    except Exception:
                        pass
                logger.info(f"[QLoRA-Diag] quantization_config={vals}")
            else:
                logger.info("[QLoRA-Diag] quantization_config=None")
        except Exception as _e2:
            logger.info(f"[QLoRA-Diag] quantization_config read failed: {_e2}")

        # 4) 어댑터 파라미터 dtype 샘플
        try:
            shown = 0
            for n, p in model.named_parameters():
                if ("lora_" in n or "dora_" in n) and shown < 4:
                    logger.info(f"[QLoRA-Diag] adapter_param {n} dtype={p.dtype} numel={p.numel()}")
                    shown += 1
        except Exception:
            pass

        logger.info("[QLoRA-Diag] ---- end ----")
    except Exception:
        pass

# 모든 로드된 어댑터를 활성화(이름과 상관없이 peft_config의 키 전부)
def _activate_all_adapters(model):
    try:
        names = []
        if hasattr(model, 'peft_config') and isinstance(model.peft_config, dict):
            names = list(model.peft_config.keys())
        if not names:
            return
        if hasattr(model, 'set_active_adapters') and callable(getattr(model, 'set_active_adapters')):
            try:
                model.set_active_adapters(names if len(names) > 1 else names[0])
            except Exception:
                pass
        elif hasattr(model, 'set_adapter') and callable(getattr(model, 'set_adapter')):
            try:
                model.set_adapter(names if len(names) > 1 else names[0])
            except Exception:
                pass
        try:
            logger.info(f"[PEFT] Active adapters set: {names}")
        except Exception:
            pass
    except Exception:
        pass

# 디버그 출력 전역 제한 (전체 실행 기준 최대 N개)
try:
    _DEBUG_PRINT_BUDGET = int(os.environ.get("DEBUG_SHOW_LIMIT", "5"))
except Exception:
    _DEBUG_PRINT_BUDGET = 5
_debug_shown_global = 0

# 어댑터 grad 보장 유틸
def _enforce_adapter_grads(model):
    try:
        for name, param in model.named_parameters():
            if ("lora_" in name) or ("dora_" in name):
                param.requires_grad = True
            else:
                param.requires_grad = False
    except Exception:
        pass

# 활성 어댑터 보장 유틸
def _ensure_active_adapters(model):
    try:
        names = []
        try:
            if hasattr(model, 'peft_config') and isinstance(model.peft_config, dict):
                names = list(model.peft_config.keys())
        except Exception:
            names = []
        # encoder_dora*, decoder_lora* 모두 활성화 대상으로 포함
        enc_list = [n for n in names if str(n).startswith('encoder_dora')]
        dec_list = [n for n in names if str(n).startswith('decoder_lora')]
        want = enc_list + dec_list
        # 폴백: 정확한 이름만 있는 경우
        if not want:
            if 'encoder_dora' in names:
                want.append('encoder_dora')
            if 'decoder_lora' in names:
                want.append('decoder_lora')
        # 추가 폴백: 일부 로더는 첫 어댑터를 'default'로 등록함
        if not want and 'default' in names:
            want.append('default')
        if want:
            if hasattr(model, 'set_active_adapters') and callable(getattr(model, 'set_active_adapters')):
                try:
                    model.set_active_adapters(want if len(want) > 1 else want[0])
                except Exception:
                    pass
            elif hasattr(model, 'set_adapter') and callable(getattr(model, 'set_adapter')):
                try:
                    model.set_adapter(want if len(want) > 1 else want[0])
                except Exception:
                    pass
            try:
                logger.info(f"[PEFT] Active adapters set: {want}")
            except Exception:
                pass
    except Exception:
        pass

# 어댑터 파라미터/그라디언트 진단 로그
def _log_adapter_param_stats(model, tag: str = "ADAPTER_PARAM"):
    try:
        stats = {"enc": {"num": 0, "elems": 0, "mean_abs": 0.0}, "dec": {"num": 0, "elems": 0, "mean_abs": 0.0}, "other": {"num": 0, "elems": 0, "mean_abs": 0.0}}
        for name, p in model.named_parameters():
            if ("lora_" not in name) and ("dora_" not in name):
                continue
            key = "enc" if name.startswith("encoder.") else ("dec" if name.startswith("decoder.") else "other")
            stats[key]["num"] += 1
            try:
                n_elem = int(p.numel())
                stats[key]["elems"] += n_elem
                with torch.no_grad():
                    stats[key]["mean_abs"] += float(p.detach().abs().mean().item())
            except Exception:
                pass
        for k in stats:
            try:
                if stats[k]["num"] > 0:
                    stats[k]["mean_abs"] = stats[k]["mean_abs"] / max(1, stats[k]["num"])
            except Exception:
                pass
        logger.info(f"[{tag}] enc: num={stats['enc']['num']} elems={stats['enc']['elems']} mean_abs={stats['enc']['mean_abs']:.6f} | dec: num={stats['dec']['num']} elems={stats['dec']['elems']} mean_abs={stats['dec']['mean_abs']:.6f} | other: num={stats['other']['num']}")
    except Exception:
        pass

def _log_adapter_grad_stats(model, tag: str = "ADAPTER_GRAD"):
    try:
        stats = {"enc": {"num": 0, "norm_sum": 0.0}, "dec": {"num": 0, "norm_sum": 0.0}, "other": {"num": 0, "norm_sum": 0.0}}
        for name, p in model.named_parameters():
            if ("lora_" not in name) and ("dora_" not in name):
                continue
            if p.grad is None:
                continue
            key = "enc" if name.startswith("encoder.") else ("dec" if name.startswith("decoder.") else "other")
            try:
                g = p.grad.detach()
                gnorm = float(g.float().norm().item())
                stats[key]["num"] += 1
                stats[key]["norm_sum"] += gnorm
            except Exception:
                pass
        logger.info(f"[{tag}] enc: count={stats['enc']['num']} norm_sum={stats['enc']['norm_sum']:.6f} | dec: count={stats['dec']['num']} norm_sum={stats['dec']['norm_sum']:.6f}")
    except Exception:
        pass

# 저장 유틸: safetensors 옵션 및 폴백 처리
def _save_with_opts(model_obj, out_dir: str, is_base: bool = False, logger_prefix: str = "Save"):
    wants_safe = os.environ.get("SAVE_SAFETENSORS", "0") == "1"
    max_shard = os.environ.get("SAVE_MAX_SHARD", "10MB")
    def _sanitize_generation_config_if_needed(m):
        try:
            gcfg = getattr(m, "generation_config", None)
            if gcfg is not None:
                nb = getattr(gcfg, "num_beams", None)
                if not isinstance(nb, int) or nb <= 1:
                    if hasattr(gcfg, "early_stopping") and getattr(gcfg, "early_stopping") is None:
                        setattr(gcfg, "early_stopping", False)
                    if hasattr(gcfg, "length_penalty") and getattr(gcfg, "length_penalty", None) is not None and nb in (None, 0, 1):
                        try:
                            setattr(gcfg, "length_penalty", None)
                        except Exception:
                            pass
        except Exception:
            pass
    try:
        if is_base:
            _sanitize_generation_config_if_needed(model_obj)
        model_obj.save_pretrained(out_dir, safe_serialization=wants_safe, max_shard_size=max_shard)
        return True
    except Exception as e1:
        try:
            # safetensors 저장 실패 시 bin으로 폴백
            model_obj.save_pretrained(out_dir, safe_serialization=False, max_shard_size=max_shard)
            logger.warning(f"[{logger_prefix}] safetensors save failed, fell back to bin: {e1}")
            return True
        except Exception as e2:
            logger.warning(f"[{logger_prefix}] save failed: {e2}")
            return False

# DLORA/QLORA 상태 로깅 유틸
def _log_dlora_status(model, tag: str = "DLORA_STATUS"):
    try:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        adapter_trainable = sum(p.numel() for n, p in model.named_parameters() if ("lora_" in n or "dora_" in n) and p.requires_grad)
        non_adapter_trainable = sum(p.numel() for n, p in model.named_parameters() if ("lora_" not in n and "dora_" not in n) and p.requires_grad)
        logger.info(f"[{tag}] trainable={trainable_params:,} / total={total_params:,}, adapter_trainable={adapter_trainable:,}, non_adapter_trainable={non_adapter_trainable:,}")
        if non_adapter_trainable > 0:
            sample = [n for n, p in model.named_parameters() if ("lora_" not in n and "dora_" not in n) and p.requires_grad][:8]
            logger.warning(f"[{tag}] Non-adapter parameters require grad: sample={sample}")
    except Exception:
        pass

# 과도한 경고 억제 (이미지 채널/peft_config 중복 경고 등)
try:
    warnings.filterwarnings("ignore", message=r"The channel dimension is ambiguous.*")
    warnings.filterwarnings("ignore", message=r"Already found a `peft_config` attribute.*")
except Exception:
    pass

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"


# ========================= DLORA DEBUG UTILS =========================
def _print_dlora_debug_info(model, vision_model_name: str, text_model_name: str) -> None:
    """DLORA 적용 가능성을 검증하기 위해 환경 및 모듈 정보를 출력합니다."""
    try:
        import peft
        peft_version = getattr(peft, "__version__", "unknown")
        peft_ok = True
    except ImportError:
        peft_version = "not installed or incompatible"
        peft_ok = False

    try:
        import transformers as _tf
        tf_version = getattr(_tf, "__version__", "unknown")
    except ImportError:
        tf_version = "unknown"

    print("==== DLORA DEBUG INFO ====")
    print(f"peft version: {peft_version} | peft ok: {peft_ok}")
    print(f"transformers version: {tf_version}")
    print(f"torch version: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()} | num devices: {torch.cuda.device_count()}")
    print(f"encoder model: {vision_model_name}")
    print(f"decoder model: {text_model_name}")

    encoder_hits, decoder_hits, other_hits = [], [], []
    attention_tokens = ["query", "key", "value", "q_proj", "k_proj", "v_proj", "out_proj"]

    for name, _ in model.named_modules():
        if not any(tok in name for tok in attention_tokens):
            continue

        # === 수정된 분류 로직 시작 ===
        # 우선순위 1: 모듈 이름의 시작 부분을 기준으로 명확하게 분류
        if name.startswith("encoder."):
            encoder_hits.append(name)
        elif name.startswith("decoder."):
            decoder_hits.append(name)
        # 우선순위 2: 시작 부분이 애매할 경우, 내부에 포함된 문자열로 분류
        elif ".encoder." in name:
            encoder_hits.append(name)
        elif ".decoder." in name:
            decoder_hits.append(name)
        # 모든 조건에 해당하지 않는 경우
        else:
            other_hits.append(name)
        # === 수정된 분류 로직 끝 ===

    def _preview(lst):
        return lst[:8]

    print("- encoder candidate modules (expect DoRA on ['query','key','value'] or ['q_proj',...]):")
    print(_preview(encoder_hits) if encoder_hits else "(none)")
    print("- decoder candidate modules (expect LoRA on ['q_proj','k_proj','v_proj','out_proj'] or ['query',...]):")
    print(_preview(decoder_hits) if decoder_hits else "(none)")
    if other_hits:
        print("- other hits (neither encoder nor decoder scoped):")
        print(_preview(other_hits))
    print(f"counts -> encoder: {len(encoder_hits)}, decoder: {len(decoder_hits)}, other: {len(other_hits)}")
    print("==== END DLORA DEBUG INFO ====")


def _discover_attention_modules(model):
    """인코더/디코더의 어텐션 모듈 이름을 탐색하여 DLORA 타겟을 추천합니다."""
    encoder_qkv_tokens, decoder_qkv_tokens = set(), set()
    encoder_examples, decoder_examples = [], []
    decoder_has_out_proj, decoder_has_crossattention = False, False

    for name, module in model.named_modules():
        if ".encoder." in name:
            if any(tok in name for tok in ["q_proj", "k_proj", "v_proj", "query", "key", "value"]):
                for tok in ["q_proj", "k_proj", "v_proj", "query", "key", "value"]:
                    if tok in name:
                        encoder_qkv_tokens.add(tok)
                        if len(encoder_examples) < 6:
                            encoder_examples.append(name)
        elif ".decoder." in name or name.startswith("decoder."):
            if "crossattention" in name:
                decoder_has_crossattention = True
            if any(tok in name for tok in ["q_proj", "k_proj", "v_proj", "out_proj", "query", "key", "value"]):
                for tok in ["q_proj", "k_proj", "v_proj", "out_proj", "query", "key", "value"]:
                    if tok in name:
                        if tok == "out_proj":
                            decoder_has_out_proj = True
                        else:
                            decoder_qkv_tokens.add(tok)
                        if len(decoder_examples) < 6:
                            decoder_examples.append(name)

    enc_style = "proj" if any(t.endswith("_proj") for t in encoder_qkv_tokens) else "named"
    dec_style = "proj" if any(t.endswith("_proj") for t in decoder_qkv_tokens) else "named"

    if enc_style == "proj":
        enc_regex = r"encoder\..*(q_proj|k_proj|v_proj)$"
    else:
        enc_regex = r"encoder\..*attention\.(self|attn|attention)\.(query|key|value)$"

    if dec_style == "proj":
        dec_regex = r"decoder\..*(q_proj|k_proj|v_proj|out_proj)$"
    else:
        dec_regex = r"decoder\..*(self\.(query|key|value)|crossattention\.self\.(query|key|value))$"

    return {
        "encoder": {"recommended_regex": enc_regex, "examples": encoder_examples},
        "decoder": {"recommended_regex": dec_regex, "examples": decoder_examples, "has_crossattention": decoder_has_crossattention},
    }


# ========================= LMDB OCR DATASET =========================
class OCRLmdbDataset(torch.utils.data.Dataset):
    """LMDB에서 (이미지, 텍스트) 샘플을 읽어 TrOCR 입력 dict를 반환합니다."""
    def __init__(self, lmdb_path: str, is_sub_char: bool = False):
        super().__init__()
        self.lmdb_path = lmdb_path
        self.is_sub_char = is_sub_char
        self._env = None
        self._transform = None
        try:
            import lmdb
        except ImportError as e:
            raise ImportError("lmdb 패키지가 필요합니다. `pip install lmdb`로 설치해주세요.") from e
        if not os.path.exists(self.lmdb_path):
            raise FileNotFoundError(f"LMDB 경로가 존재하지 않습니다: {self.lmdb_path}")
        
        _tmp_env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        try:
            with _tmp_env.begin(write=False) as txn:
                length_bytes = txn.get('num-samples'.encode())
                if length_bytes is None:
                    raise KeyError(f"'num-samples' 키를 찾을 수 없습니다: {self.lmdb_path}")
                self.length = int(length_bytes.decode())
        finally:
            _tmp_env.close()

    def set_transform(self, transform_fn):
        self._transform = transform_fn

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        # 워커 프로세스에서 필요한 모듈을 명시적으로 import
        from PIL import Image
        from io import BytesIO

        if self._env is None:
            import lmdb
            self._env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        
        try:
            real_index = index % self.length
            key_idx = (real_index + 1)
            with self._env.begin(write=False) as txn:
                img_key = f'image-{key_idx:09d}'.encode()
                lbl_key = f'label-{key_idx:09d}'.encode()

                img_bytes = txn.get(img_key)
                if not img_bytes or not txn.get(lbl_key):
                    return None

                pil_image = Image.open(BytesIO(img_bytes)).convert('RGB')
                
                label_bytes = txn.get(lbl_key)
                text = label_bytes.decode('utf-8', errors='ignore')
                # 입력 라벨은 항상 NFKD로 정규화 (sub-char 모델 전제)
                if text:
                    text = unicodedata.normalize("NFKD", text)

                sample = {'pixel_values': pil_image, 'labels': text}
                if self._transform is not None:
                    try:
                        sample = self._transform(sample)
                    except Exception:
                        pass
                return sample
        
        except Exception:
            return None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_env'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._env = None


class ConcatOCRLmdbDataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_paths, is_sub_char: bool = False):
        super().__init__()
        parts = [OCRLmdbDataset(p, is_sub_char=is_sub_char) for p in lmdb_paths]
        self.concat = torch.utils.data.ConcatDataset(parts)

    def __len__(self) -> int:
        return len(self.concat)

    def __getitem__(self, index: int):
        return self.concat[index]

    def set_transform(self, transform_fn):
        try:
            for ds in self.concat.datasets:
                if hasattr(ds, 'set_transform') and callable(getattr(ds, 'set_transform')):
                    ds.set_transform(transform_fn)
        except Exception:
            pass


def main(model_args: ModelArguments, dataset_args: DatasetsArguments, training_args: MyTrainingArguments):
    global _debug_shown_global
    setproctitle("kyowon_ocr")
    seed_everything(training_args.seed)
    torch.backends.cudnn.benchmark = True
    
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
        logger.info("[MP] Multiprocessing start_method set to 'spawn'")
    except RuntimeError as e:
        logger.warning(f"[MP] Could not set start_method to 'spawn': {e}")
    
    # --- 1. 모델, 토크나이저, 프로세서 로드 ---
    vision_model_name = model_args.encoder_model_name_or_path
    text_model_name = model_args.decoder_model_name_or_path
    single_model_name = model_args.model_name_or_path
    use_single_model = single_model_name is not None

    # 체크포인트 토크나이저/프로세서 경로 해석 (환경변수 및 output_dir 기반)
    tokenizer_ckpt_dir = None
    tokenizer_ckpt_candidate_dir = None
    tokenizer_loaded_from_ckpt = False
    def _dir_has_tokenizer_files(path: str) -> bool:
        try:
            if not path or not os.path.isdir(path):
                return False
            # 대표적인 토크나이저 파일 후보들
            candidate_files = [
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "spiece.model",
                "vocab.txt",
            ]
            names = set(os.listdir(path))
            for fname in candidate_files:
                if fname in names:
                    return True
            return False
        except Exception:
            return False
    try:
        _env_tok = os.environ.get("TOKENIZER_CKPT", "").strip()
        if _env_tok and os.path.isdir(_env_tok):
            tokenizer_ckpt_dir = _env_tok
            tokenizer_ckpt_candidate_dir = _env_tok
        else:
            _resume_env = os.environ.get("RESUME_FROM", "").strip()
            if _resume_env and os.path.isdir(_resume_env):
                tokenizer_ckpt_dir = _resume_env
                tokenizer_ckpt_candidate_dir = _resume_env
            else:
                from transformers.trainer_utils import get_last_checkpoint
                if hasattr(training_args, "output_dir") and training_args.output_dir and os.path.isdir(training_args.output_dir):
                    _last_ckpt_dir = get_last_checkpoint(training_args.output_dir)
                    if _last_ckpt_dir and os.path.isdir(_last_ckpt_dir):
                        tokenizer_ckpt_dir = _last_ckpt_dir
                        tokenizer_ckpt_candidate_dir = _last_ckpt_dir
    except Exception as _te:
        logger.warning(f"[Tokenizer] Failed to resolve checkpoint dir for tokenizer: {_te}")
    # 토크나이저 파일이 실제로 존재하는 디렉터리인지 검증
    if tokenizer_ckpt_dir and not _dir_has_tokenizer_files(tokenizer_ckpt_dir):
        logger.info(f"[Tokenizer] Skip checkpoint tokenizer: no tokenizer files in {tokenizer_ckpt_dir}")
        tokenizer_ckpt_dir = None

    # fast 토크나이저 사용 여부 (기본 비활성화: fast 추가 vocab 시 패닉 방지)
    _fast_tok = os.environ.get("TOKENIZER_FAST", "0").strip().lower() in ("1", "true", "yes")

    if use_single_model:
        try:
            # 체크포인트에 저장된 프로세서(토크나이저 포함)를 우선 로드
            if tokenizer_ckpt_dir and os.path.isdir(tokenizer_ckpt_dir):
                try:
                    ocr_processor = TrOCRProcessor.from_pretrained(tokenizer_ckpt_dir, use_fast=True)
                    logger.info(f"[Processor] Loaded processor from checkpoint: {tokenizer_ckpt_dir}")
                    tokenizer_loaded_from_ckpt = True
                except Exception as _pe:
                    logger.warning(f"[Processor] Failed to load processor from checkpoint '{tokenizer_ckpt_dir}': {_pe}")
                    ocr_processor = TrOCRProcessor.from_pretrained(single_model_name, use_fast=True)
                    logger.info(f"[Processor] Fallback to single model processor: {single_model_name}")
            else:
                ocr_processor = TrOCRProcessor.from_pretrained(single_model_name, use_fast=True)
                logger.info(f"[Processor] Loaded processor from base model: {single_model_name}")
            image_processor, tokenizer = ocr_processor.image_processor, ocr_processor.tokenizer

            # 체크포인트에 프로세서가 없더라도 토크나이저 파일이 있으면 별도로 토크나이저만 로드하여 덮어쓰기
            try:
                if tokenizer_ckpt_dir and os.path.isdir(tokenizer_ckpt_dir) and _dir_has_tokenizer_files(tokenizer_ckpt_dir):
                    _tok_override = AutoTokenizer.from_pretrained(tokenizer_ckpt_dir, use_fast=True)
                    tokenizer = _tok_override
                    ocr_processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
                    tokenizer.model_max_length = 512
                    ocr_processor.tokenizer.model_max_length = 512
                    logger.info(f"[Tokenizer] Overrode tokenizer from checkpoint: {tokenizer_ckpt_dir}")
                    tokenizer_loaded_from_ckpt = True
            except Exception as _toe:
                logger.warning(f"[Tokenizer] Failed to override tokenizer from checkpoint '{tokenizer_ckpt_dir}': {_toe}")
        except Exception as _e:
            logger.warning(f"[Processor] Failed to load single model processor from '{single_model_name}': {_e}")
            # 단일 모델 로드 실패 시 분리 로드로 폴백 시도
            if not vision_model_name or not text_model_name:
                raise OSError(
                    "--model_name_or_path 경로에 체크포인트가 없거나 불완전합니다.\n"
                    "다음 중 하나로 실행하세요:\n"
                    "1) 허깅페이스 단일 모델로 실행: --model_name_or_path microsoft/trocr-base-printed (또는 적절한 베이스 모델)\n"
                    "2) 분리 경로로 실행: --encoder_model_name_or_path <vision_model> --decoder_model_name_or_path <text_model>"
                ) from _e
            use_single_model = False

    if not use_single_model:
        image_processor = AutoImageProcessor.from_pretrained(vision_model_name, use_fast=True)

        # 우선순위: TOKENIZER_CKPT(환경변수) > RESUME_FROM(경로) > output_dir의 마지막 체크포인트
        tokenizer = None
        tokenizer_ckpt_dir = None
        try:
            env_tok = os.environ.get("TOKENIZER_CKPT", "").strip()
            if env_tok and os.path.isdir(env_tok):
                tokenizer_ckpt_dir = env_tok
            else:
                resume_env = os.environ.get("RESUME_FROM", "").strip()
                if resume_env and os.path.isdir(resume_env):
                    tokenizer_ckpt_dir = resume_env
                else:
                    # output_dir에서 마지막 체크포인트 탐색
                    from transformers.trainer_utils import get_last_checkpoint
                    if hasattr(training_args, "output_dir") and training_args.output_dir and os.path.isdir(training_args.output_dir):
                        last_ckpt_dir = get_last_checkpoint(training_args.output_dir)
                        if last_ckpt_dir and os.path.isdir(last_ckpt_dir):
                            tokenizer_ckpt_dir = last_ckpt_dir
        except Exception as _te:
            logger.warning(f"[Tokenizer] Failed to resolve checkpoint dir for tokenizer: {_te}")

        if tokenizer_ckpt_dir and _dir_has_tokenizer_files(tokenizer_ckpt_dir):
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt_dir, use_fast=True)
                tokenizer.model_max_length = 512
                
                logger.info(f"[Tokenizer] Loaded tokenizer from checkpoint: {tokenizer_ckpt_dir}")
                tokenizer_loaded_from_ckpt = True
            except Exception as _le:
                logger.warning(f"[Tokenizer] Failed to load tokenizer from checkpoint '{tokenizer_ckpt_dir}': {_le}")

        # 폴백: decoder 모델 이름에서 로드
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(text_model_name, use_fast=True)
        if "gpt" in (text_model_name or ""):
            tokenizer.add_special_tokens({
                    "bos_token": "<s>", "eos_token": " ", "unk_token": "<unk>", 
                "pad_token": "<pad>", "mask_token": "<mask>"
            })

        ocr_processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
        tokenizer.model_max_length = 512
        ocr_processor.tokenizer.model_max_length = 512
        # 스캔 없이 체크포인트에 있는 추가 토큰만 적용하고 싶을 때: APPLY_ADDED_TOKENS_FROM_CKPT=1
        # tokenizer는 base에서 로드하되, ckpt의 added_tokens.json/tokenizer.json로 덮어씌움
        try:
            if os.environ.get("APPLY_ADDED_TOKENS_FROM_CKPT", "0") == "1":
                tok_dir = tokenizer_ckpt_dir if (tokenizer_ckpt_dir and os.path.isdir(tokenizer_ckpt_dir)) else None
                if tok_dir and _dir_has_tokenizer_files(tok_dir):
                    _tok_override = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
                    # 토크나이저 전체를 교체하여 added_tokens를 반영
                    tokenizer = _tok_override
                    ocr_processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
                    tokenizer.model_max_length = 512
                    ocr_processor.tokenizer.model_max_length = 512
                    logger.info(f"[Tokenizer] Applied added_tokens from checkpoint without scanning: {tok_dir}")
        except Exception as _e:
            logger.warning(f"[Tokenizer] Failed to apply added_tokens from checkpoint: {_e}")

    # --- 2. 데이터셋 로드 (LMDB 또는 CSV) ---
    is_sub_char = text_model_name == "snunlp/KR-BERT-char16424"
    lmdb_env = os.environ.get('LMDB_PATHS')
    val_lmdb_env = os.environ.get('VAL_LMDB_PATHS')
    
    if lmdb_env:
        lmdb_paths = [p for p in lmdb_env.split(':') if p]
        train_dataset = ConcatOCRLmdbDataset(lmdb_paths, is_sub_char=is_sub_char)
        # LMDB 학습에도 증강 적용
        try:
            augmentator = Augmentator(
                aug_with_compose_prob=0.5, rotation_prob=0.5, rotation_square_side=min(image_processor.size.values())
            )
            if hasattr(train_dataset, 'set_transform'):
                train_dataset.set_transform(augmentator.augmentation)
        except Exception:
            pass
        if val_lmdb_env:
            val_paths = [p for p in val_lmdb_env.split(':') if p]
            valid_dataset = ConcatOCRLmdbDataset(val_paths, is_sub_char=is_sub_char)
        else:
            valid_dataset = None
    else:
        augmentator = Augmentator(
            aug_with_compose_prob=0.8, rotation_prob=0.5, rotation_square_side=min(image_processor.size.values())
        )
        train_dataset = get_dataset(dataset_args.train_csv_path, is_sub_char=is_sub_char)
        train_dataset.set_transform(augmentator.augmentation)
        valid_dataset = get_dataset(dataset_args.valid_csv_path, is_sub_char=is_sub_char)

    # --- 3. 모델 로드 및 설정 ---
    use_qlora = os.environ.get("USE_QLORA", "0") == "1"
    bnb_config = None
    if use_qlora:
        try:
            # Trainer 혼합정밀 설정과 4bit compute dtype을 정합화
            preferred_compute_dtype = (
                torch.float16 if getattr(training_args, 'fp16', False)
                else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
            )
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=preferred_compute_dtype,
            )
            logger.info(f"[QLoRA] Enabled 4bit nf4 (double quant), compute_dtype={bnb_config.bnb_4bit_compute_dtype}")
        except Exception as e:
            logger.warning(f"[QLoRA] bitsandbytes 설정 실패: {e}. QLoRA를 비활성화합니다.")
            use_qlora = False

    # 모델 로드는 토크나이저 업데이트 이후에 수행합니다.
    model = None
    # 토큰 추가/스캔 로직 제거
    
    # --- 4. 체크포인트 Config로 모델 로드 (어댑터 자동 로드시 vocab 불일치 방지) ---
    logger.info("Loading model with checkpoint configuration...")
    model_path = single_model_name if use_single_model else text_model_name
    try:
        from transformers.integrations.peft import PeftAdapterMixin as _PAM
        if hasattr(_PAM, "load_adapter"):
            _orig_load_adapter = _PAM.load_adapter
            def _noop_load_adapter(self, *args, **kwargs):
                logger.info("[PEFT] Skipping auto adapter load during base model initialization.")
                return None
            _PAM.load_adapter = _noop_load_adapter
        else:
            _orig_load_adapter = None
    except Exception as _e:
        _orig_load_adapter = None
        logger.warning(f"[PEFT] Could not monkeypatch load_adapter: {_e}")

    if use_single_model:
        try:
            ckpt_config = AutoConfig.from_pretrained(model_path)
        except Exception as _e:
            logger.warning(f"[Config] Failed to load config from single model path '{model_path}': {_e}")
            if not vision_model_name or not text_model_name:
                raise OSError(
                    "단일 체크포인트 구성이 없어서 로드할 수 없습니다. 분리 경로(encoder/decoder)로 실행하거나 유효한 단일 모델을 지정하세요."
                ) from _e
            use_single_model = False
        if getattr(ckpt_config, "quantization_config", None) is not None:
            logger.warning("[QLoRA] Removing existing quantization_config from checkpoint config to avoid expecting pre-quantized weights.")
            try:
                delattr(ckpt_config, "quantization_config")
            except Exception:
                ckpt_config.quantization_config = None
        ckpt_vocab_size = getattr(getattr(ckpt_config, "decoder", ckpt_config), "vocab_size", None)
        model = VisionEncoderDecoderModel.from_pretrained(
            single_model_name,
            config=ckpt_config,
            quantization_config=bnb_config if use_qlora else None,
            device_map="auto" if use_qlora else None,
            ignore_mismatched_sizes=True,
        )
    else:
        vision_config = AutoConfig.from_pretrained(vision_model_name)
        text_config = AutoConfig.from_pretrained(text_model_name)
        if getattr(text_config, "quantization_config", None) is not None:
            logger.warning("[QLoRA] Removing existing quantization_config from decoder checkpoint config to avoid expecting pre-quantized weights.")
            try:
                delattr(text_config, "quantization_config")
            except Exception:
                text_config.quantization_config = None
        config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
            encoder_config=vision_config, decoder_config=text_config
        )
        config.decoder.vocab_size = len(tokenizer)
        config.decoder_start_token_id = (
            tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
        )
        config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        config.vocab_size = int(config.decoder.vocab_size)
        config.eos_token_id = (
            tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
        )
        config.max_length = 512
        config.min_length = 1
        config.do_sample = False
        config.num_beams = 1
        config.early_stopping = False
        config.no_repeat_ngram_size = 0
        config.repetition_penalty = 1.0
        config.length_penalty = 1.0

        ckpt_vocab_size = getattr(text_config, "vocab_size", None)
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_pretrained_model_name_or_path=vision_model_name,
            decoder_pretrained_model_name_or_path=text_model_name,
            config=config,
            quantization_config=bnb_config if use_qlora else None,
            device_map="auto" if use_qlora else None,
            ignore_mismatched_sizes=True,
        )

    try:
        if use_qlora and prepare_model_for_kbit_training is not None:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=getattr(training_args, 'gradient_checkpointing', True))
    except Exception as _prep_e:
        logger.warning(f"[QLoRA] prepare_model_for_kbit_training failed: {_prep_e}")

    try:
        if _orig_load_adapter is not None:
            from transformers.integrations.peft import PeftAdapterMixin as _PAM2
            _PAM2.load_adapter = _orig_load_adapter
    except Exception as _e:
        logger.warning(f"[PEFT] Could not restore load_adapter: {_e}")

    # 임베딩 리사이즈 로직 제거: 체크포인트/모델의 vocab을 그대로 사용

    # --- 7. 추가 Config 설정 및 생성 파라미터 ---
    try:
        t = ocr_processor.tokenizer
        model.config.decoder.vocab_size = len(t)
        # 안전한 decoder_start_token_id 설정: bos -> cls -> sep -> pad -> 0
        _dst = getattr(model.config, "decoder_start_token_id", None)
        if not isinstance(_dst, int) or _dst is None:
            for _cid in [getattr(t, "bos_token_id", None), getattr(t, "cls_token_id", None), getattr(t, "sep_token_id", None), getattr(t, "pad_token_id", None), 0]:
                if isinstance(_cid, int) and _cid is not None and _cid >= 0:
                    model.config.decoder_start_token_id = int(_cid)
                    break
        # pad/eos 보정
        if not isinstance(getattr(model.config, "pad_token_id", None), int):
            model.config.pad_token_id = int(getattr(t, "pad_token_id", 0) or 0)
        _eos_now = getattr(model.config, "eos_token_id", None)
        if not isinstance(_eos_now, int):
            _cand_eos = getattr(t, "sep_token_id", None)
            if not isinstance(_cand_eos, int):
                _cand_eos = getattr(t, "eos_token_id", None)
            if isinstance(_cand_eos, int):
                model.config.eos_token_id = int(_cand_eos)
        model.config.vocab_size = model.config.decoder.vocab_size
        model.config.bos_token_id = None
        model.config.forced_bos_token_id = None

        # 생성 관련 파라미터 기본값
        model.config.max_length = 512
        model.config.min_length = 1
        model.config.do_sample = False
        model.config.num_beams = 1
        model.config.early_stopping = False
        
        model.config.no_repeat_ngram_size = 0
        model.config.repetition_penalty = 1.0
        model.config.length_penalty = 1.0
        model.config.remove_invalid_values = True
        model.config.forced_eos_token_id = tokenizer.sep_token_id
        model.config.num_return_sequences = 1
        model.config.use_cache = False
        # generation_config가 존재하면 동일하게 반영
        try:
            if hasattr(model, "generation_config") and getattr(model, "generation_config") is not None:
                model.generation_config.max_length = int(model.config.max_length)
                if getattr(model.generation_config, "eos_token_id", None) is None:
                    model.generation_config.eos_token_id = int(model.config.eos_token_id)
                if getattr(model.generation_config, "pad_token_id", None) is None:
                    model.generation_config.pad_token_id = int(model.config.pad_token_id)
                if getattr(model.generation_config, "decoder_start_token_id", None) is None:
                    model.generation_config.decoder_start_token_id = int(model.config.decoder_start_token_id)
        except Exception:
            pass
    except Exception:
        pass

    # --- 4. DLORA / QDLORA (PEFT 어댑터) 주입 또는 체크포인트에서 로드 ---
    _print_dlora_debug_info(model, vision_model_name, text_model_name)
    discover = _discover_attention_modules(model)
    print("==== ATTENTION MODULE DISCOVERY ====\n", discover, "\n==== END DISCOVERY ====")

    loaded_peft_from_ckpt = False
    try:
        resume_dir_ad = os.path.expanduser(os.environ.get("RESUME_FROM", "").strip())
        if resume_dir_ad and os.path.isdir(resume_dir_ad):
            def _dir_has_adapter_files(d: str) -> bool:
                try:
                    names = set(os.listdir(d))
                    if "adapter_config.json" in names and ("adapter_model.safetensors" in names or "adapter_model.bin" in names):
                        return True
                except Exception:
                    return False
                return False

            # 1) 루트에서 직접 시도
            candidate_dirs = []
            if _dir_has_adapter_files(resume_dir_ad):
                candidate_dirs.append(resume_dir_ad)
            # 2) 재귀적으로 하위 디렉터리에서 탐색 (여러 어댑터 저장 케이스)
            try:
                for _root, _dirs, _files in os.walk(resume_dir_ad):
                    if _dir_has_adapter_files(_root) and _root not in candidate_dirs:
                        candidate_dirs.append(_root)
            except Exception:
                pass

            if candidate_dirs:
                try:
                    from peft import PeftModel
                    # 첫 디렉터리로 래핑
                    first_dir = candidate_dirs[0]
                    model = PeftModel.from_pretrained(model, first_dir, is_trainable=True)
                    loaded_peft_from_ckpt = True
                    logger.info(f"[Resume] Loaded PEFT adapter from {first_dir}")
                    # 나머지 어댑터는 추가 로드
                    for extra_dir in candidate_dirs[1:]:
                        try:
                            adapter_name = os.path.basename(extra_dir.rstrip(os.sep)) or f"adapter_{len(model.peft_config) + 1}" if hasattr(model, 'peft_config') else None
                            if hasattr(model, 'load_adapter'):
                                model.load_adapter(extra_dir, adapter_name=adapter_name, is_trainable=True)
                                logger.info(f"[Resume] Loaded extra adapter from {extra_dir} as '{adapter_name}'")
                        except Exception as _lae:
                            logger.warning(f"[Resume] Failed to load extra adapter from {extra_dir}: {_lae}")
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                except Exception as _pe:
                    logger.warning(f"[Resume] Failed to load PEFT adapters from {resume_dir_ad}: {_pe}")
    except Exception:
        pass

    if LoraConfig is None:
        logger.warning("[DLORA] `peft`가 설치되지 않아 어댑터 주입을 건너뜁니다.")
    elif loaded_peft_from_ckpt:
        # 재개에서 어댑터를 불러왔으므로 신규 주입을 생략
        try:
            _activate_all_adapters(model)
        except Exception:
            pass
        _enforce_adapter_grads(model)
        try:
            model.print_trainable_parameters()
        except Exception:
            pass
    elif use_qlora:
        # 모듈 이름 스캔으로 정확한 타겟 명단 생성 (정규식 없이 suffix 기반)
        dec_suffixes = (
            ".attention.self.query", ".attention.self.key", ".attention.self.value",
            ".crossattention.self.query", ".crossattention.self.key", ".crossattention.self.value",
            ".attention.output.dense", ".crossattention.output.dense",
        )
        enc_suffixes = (
            ".attention.attention.query", ".attention.attention.key", ".attention.attention.value",
        )
        decoder_target_names = [
            n for n, _ in model.named_modules()
            if n.startswith("decoder.") and any(n.endswith(sfx) for sfx in dec_suffixes)
        ]
        encoder_target_names = [
            n for n, _ in model.named_modules()
            if (".encoder." in n) and any(n.endswith(sfx) for sfx in enc_suffixes)
        ]
        if not decoder_target_names:
            logger.warning("[QDLORA] No decoder targets found via suffix scan; falling back to query/key/value/dense substrings.")
            decoder_target_names = ["query", "key", "value", "dense"]
        if not encoder_target_names:
            logger.warning("[QDLORA] No encoder targets found via suffix scan; falling back to query/key/value substrings.")
            encoder_target_names = ["query", "key", "value"]

        decoder_lora_config = LoraConfig(
            r=int(os.environ.get('DLORA_RANK', '16')),
            lora_alpha=int(os.environ.get('DLORA_ALPHA', '32')),
            lora_dropout=0.1, 
            target_modules=decoder_target_names, 
            use_dora=False
        )
        logger.info(f"[QDLORA] Injecting decoder LoRA with targets: {decoder_target_names} (total={len(decoder_target_names)})")
        model = get_peft_model(model, decoder_lora_config, adapter_name="decoder_lora")

        encoder_dora_config = LoraConfig(
            r=int(os.environ.get('DLORA_RANK', '16')),
            lora_alpha=int(os.environ.get('DLORA_ALPHA', '32')),
            lora_dropout=0.1, 
            target_modules=encoder_target_names, 
            use_dora=True
        )
        logger.info(f"[QDLORA] Injecting encoder DoRA with targets: {encoder_target_names} (total={len(encoder_target_names)})")
        model.add_adapter("encoder_dora", encoder_dora_config)

        # 두 어댑터를 명시적으로 동시에 활성화(이름 폴백 포함), 어댑터 파라미터에 grad를 강제
        try:
            _activate_all_adapters(model)
        except Exception:
            pass
        _enforce_adapter_grads(model)
        model.print_trainable_parameters()
        _log_qlora_diagnostics(model)
    else: # 일반 DLORA 모드
        # 모듈 이름 스캔으로 정확한 타겟 명단 생성 (정규식 없이 suffix 기반)
        dec_suffixes = (
            ".attention.self.query", ".attention.self.key", ".attention.self.value",
            ".crossattention.self.query", ".crossattention.self.key", ".crossattention.self.value",
            ".attention.output.dense", ".crossattention.output.dense",
        )
        enc_suffixes = (
            ".attention.attention.query", ".attention.attention.key", ".attention.attention.value",
        )
        decoder_target_names = [
            n for n, _ in model.named_modules()
            if n.startswith("decoder.") and any(n.endswith(sfx) for sfx in dec_suffixes)
        ]
        encoder_target_names = [
            n for n, _ in model.named_modules()
            if (".encoder." in n) and any(n.endswith(sfx) for sfx in enc_suffixes)
        ]
        if not decoder_target_names:
            logger.warning("[DLORA] No decoder targets found via suffix scan; falling back to query/key/value/dense substrings.")
            decoder_target_names = ["query", "key", "value", "dense"]
        rank = int(os.environ.get('DLORA_RANK', '16'))
        alpha = int(os.environ.get('DLORA_ALPHA', '32'))

        decoder_lora_cfg = LoraConfig(
            target_modules=decoder_target_names, r=rank, lora_alpha=alpha, lora_dropout=0.1, use_dora=False
        )
        logger.info(f"[DLORA] Injecting decoder LoRA with targets: {decoder_target_names} (total={len(decoder_target_names)})")
        model = get_peft_model(model, decoder_lora_cfg, adapter_name="decoder_lora")
        if not encoder_target_names:
            logger.warning("[DLORA] No encoder targets found via suffix scan; falling back to query/key/value substrings.")
            encoder_target_names = ["query", "key", "value"]
        encoder_dora_cfg = LoraConfig(
            target_modules=encoder_target_names, r=rank, lora_alpha=alpha, lora_dropout=0.1, use_dora=True
        )
        logger.info(f"[DLORA] Injecting encoder DoRA with targets: {encoder_target_names} (total={len(encoder_target_names)})")
        model.add_adapter("encoder_dora", encoder_dora_cfg)
        
        # 두 어댑터를 명시적으로 동시에 활성화(이름 폴백 포함), 어댑터 파라미터에 grad를 강제
        try:
            _activate_all_adapters(model)
        except Exception:
            pass
        _enforce_adapter_grads(model)
        model.print_trainable_parameters()
        _log_qlora_diagnostics(model)

    # 어댑터 파라미터만 학습되도록 보장
    _enforce_adapter_grads(model)

    # 데이터 콜레이터 로드
    if not use_single_model and text_model_name and ("gpt" in text_model_name):
        data_collator = DataCollatorForGptOCR(processor=ocr_processor)
    else:
        data_collator = DataCollatorForOCR(processor=ocr_processor)

    # ================== 수동 학습 루프 시작 ==================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not use_qlora:
        model.to(device)
    

    model.config.use_cache = False
    model.encoder.config.use_cache = False
    model.decoder.config.use_cache = False
    model.train()

    # Non-QLoRA 경로에서도 gradient checkpointing 지원 (활성화 시 활성화 메모리 절감)
    try:
        if getattr(training_args, 'gradient_checkpointing', True):
            print("GRAD_CHECK")
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                try:
                    if hasattr(model, 'encoder') and hasattr(model.encoder, 'gradient_checkpointing_enable'):
                        model.encoder.config.use_cache = False
                        model.encoder.gradient_checkpointing_enable()
                except Exception:
                    pass
                try:
                    if hasattr(model, 'decoder') and hasattr(model.decoder, 'gradient_checkpointing_enable'):
                        model.decoder.config.use_cache = False
                        model.decoder.gradient_checkpointing_enable()
                except Exception:
                    pass
    except Exception:
        pass

    per_device_train_bs = training_args.per_device_train_batch_size
    os.environ["DISABLE_LABEL_TOKENIZE_IN_WORKER"] = "1"
    dl_kwargs = {}
    if training_args.dataloader_num_workers and training_args.dataloader_num_workers > 0:
        dl_kwargs["multiprocessing_context"] = 'spawn'
        dl_kwargs["persistent_workers"] = True
    loader = DataLoader(
        train_dataset,
        batch_size=per_device_train_bs,
        shuffle=True,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=training_args.dataloader_pin_memory,
        prefetch_factor = 4,
        collate_fn=data_collator,
        **dl_kwargs,
    )

    # Trainer 정합: epoch 전체 스텝 기반 기본값 설정 (eval/save/logging)
    try:
        NUM_GPU = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
    except Exception:
        NUM_GPU = 1
    total_batch = int(getattr(training_args, "per_device_train_batch_size", 1)) * int(getattr(training_args, "gradient_accumulation_steps", 1)) * max(1, int(NUM_GPU))
    try:
        one_epoch_len = max(1, len(train_dataset) // max(1, total_batch))
    except Exception:
        one_epoch_len = 1
    total_steps = max(1, int(getattr(training_args, "num_train_epochs", 1)) * one_epoch_len)
    if getattr(training_args, "eval_steps", None) is None:
        training_args.eval_steps = max(1, total_steps // 10)
    # SAVE_STEPS 환경변수가 있으면 최우선 적용, 없으면 사용자가 지정한 값(>0)을 그대로 유지
    try:
        _env_save_steps = os.environ.get("SAVE_STEPS", None)
        if _env_save_steps is not None:
            training_args.save_steps = max(1, int(_env_save_steps))
        elif getattr(training_args, "save_steps", None) is None:
            training_args.save_steps = max(1, total_steps // 10)
    except Exception:
        if getattr(training_args, "save_steps", None) is None:
            training_args.save_steps = max(1, total_steps // 10)
    if getattr(training_args, "logging_steps", None) is None:
        training_args.logging_steps = max(1, one_epoch_len // 10)
    
    # 전체 스텝(progress bar) 기준 계산
    updates_per_epoch = max(1, math.ceil(len(loader) / max(1, training_args.gradient_accumulation_steps)))
    total_update_steps = int(training_args.num_train_epochs) * updates_per_epoch
    
    # Validation loader
    eval_subset_size = int(os.environ.get("EVAL_SAMPLES", "100"))
    _eval_steps_default = int(getattr(training_args, "eval_steps", max(1, total_steps // 10)))
    eval_every = int(os.environ.get("EVAL_EVERY_STEPS", str(_eval_steps_default)))
    val_indices = []
    if valid_dataset is not None:
        n = min(int(eval_subset_size), len(valid_dataset))
        if n > 0:
            rng = random.Random(training_args.seed)
            try:
                # Concat 형태의 검증셋을 구성 요소별 균등 분배로 샘플링
                concat_obj = None
                if hasattr(valid_dataset, 'concat') and isinstance(getattr(valid_dataset, 'concat', None), torch.utils.data.ConcatDataset):
                    concat_obj = valid_dataset.concat
                elif isinstance(valid_dataset, torch.utils.data.ConcatDataset):
                    concat_obj = valid_dataset
                if concat_obj is not None:
                    parts = list(concat_obj.datasets)
                    k = len(parts)
                    if k <= 0:
                        val_indices = rng.sample(range(len(valid_dataset)), n)
                    else:
                        lengths = [len(p) for p in parts]
                        base = n // k
                        rem = n % k
                        counts = [base + (1 if i < rem else 0) for i in range(k)]
                        offsets = []
                        _c = 0
                        for L in lengths:
                            offsets.append(_c)
                            _c += L
                        for cnt, L, off in zip(counts, lengths, offsets):
                            if L <= 0 or cnt <= 0:
                                continue
                            use_cnt = cnt if cnt < L else L
                            if use_cnt < L:
                                local_idxs = rng.sample(range(L), use_cnt)
                            else:
                                local_idxs = list(range(L))
                            val_indices.extend([off + j for j in local_idxs])
                else:
                    val_indices = rng.sample(range(len(valid_dataset)), n)
            except Exception:
                val_indices = rng.sample(range(len(valid_dataset)), n)
    val_loader = None
    if val_indices:
        val_subset = Subset(valid_dataset, val_indices)
        vdl_kwargs = {}
        if training_args.dataloader_num_workers and training_args.dataloader_num_workers > 0:
            vdl_kwargs["multiprocessing_context"] = 'spawn'
            # 검증에서는 persistent_workers를 끄어 메모리 사용을 줄임
            vdl_kwargs["persistent_workers"] = False
        val_loader = DataLoader(
            val_subset,
            batch_size=training_args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
            collate_fn=data_collator,
            **vdl_kwargs,
        )

    # Optimizer & Scheduler
    # Optimizer: 선택적으로 bitsandbytes 8bit/Paged 8bit 사용 (옵티마이저 메모리 절감)
    _use_bnb_optim = os.environ.get("USE_BNB_OPTIM", "0").strip().lower() in ("1", "true", "yes")
    if _use_bnb_optim:
        try:
            import bitsandbytes as bnb  # type: ignore
            _use_paged = os.environ.get("USE_BNB_PAGED", "1").strip().lower() in ("1", "true", "yes")
            if _use_paged and hasattr(bnb.optim, 'PagedAdamW8bit'):
                optim_cls = bnb.optim.PagedAdamW8bit
            else:
                optim_cls = getattr(bnb.optim, 'AdamW8bit', None) or getattr(bnb.optim, 'Adam8bit', None)
            if optim_cls is None:
                raise ImportError("bitsandbytes 8bit AdamW not available")
            optimizer = optim_cls(
                [p for p in model.parameters() if p.requires_grad],
                lr=training_args.learning_rate,
            )
            logger.info(f"[Optim] Using bitsandbytes {optim_cls.__name__}")
        except Exception as _oe:
            logger.warning(f"[Optim] bitsandbytes optimizer not available ({_oe}); falling back to torch AdamW")
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=training_args.learning_rate,
            )
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=training_args.learning_rate,
        )
    total_update_steps = max(1, math.ceil(len(loader) / max(1, training_args.gradient_accumulation_steps)) * int(training_args.num_train_epochs))
    # Warmup 기본값 적용 (미설정 시 total의 3%)
    # Scheduler disabled: use constant LR
    scheduler = None
    try:
        logger.info("[LR] Scheduler disabled. Using constant learning rate from optimizer.")
    except Exception:
        pass
    
    # AMP
    amp_dtype = torch.bfloat16 if getattr(training_args, "bf16", False) else (torch.float16 if getattr(training_args, "fp16", False) else None)
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=bool(getattr(training_args, "fp16", False)))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=bool(getattr(training_args, "fp16", False)))
    
    if amp_dtype is torch.bfloat16:
        logger.info("[ManualTrain] AMP: bf16")
    elif amp_dtype is torch.float16:
        logger.info("[ManualTrain] AMP: fp16")
    else:
        logger.info("[ManualTrain] AMP: disabled")

    # 어댑터 동결 로직 제거 (요청에 따라 비활성화)

    # Resume
    resume_dir = os.path.expanduser(os.environ.get("RESUME_FROM", "").strip())
    global_step = 0
    if resume_dir:
        state_path = os.path.join(resume_dir, 'trainer_state.pt')
        try:
            # PyTorch 2.6+ 기본 weights_only=True로 변경됨. 신뢰된 체크포인트에 한해 안전 폴백 허용
            try:
                state = torch.load(state_path, map_location='cpu')
            except Exception:
                state = torch.load(state_path, map_location='cpu', weights_only=False)
                
            if 'optimizer' in state: 
                optimizer.load_state_dict(state['optimizer'])
            # scheduler disabled
            if 'scaler' in state and state['scaler'] is not None: scaler.load_state_dict(state['scaler'])
            if 'rng_state_cpu' in state and state['rng_state_cpu'] is not None: torch.set_rng_state(state['rng_state_cpu'])
            if 'rng_state_cuda' in state and state['rng_state_cuda'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['rng_state_cuda'])
            if 'python_random_state' in state and state['python_random_state'] is not None: random.setstate(state['python_random_state'])
            if 'numpy_random_state' in state and state['numpy_random_state'] is not None: np.random.set_state(state['numpy_random_state'])
            global_step = int(state.get('global_step', 0))
            logger.info(f"[Resume] Restored optimizer/scheduler/scaler/RNG from {resume_dir} (global_step={global_step})")


        except Exception as _re:
            logger.warning(f"[Resume] Failed to restore from {state_path}: {_re}")

    num_epochs = int(training_args.num_train_epochs)
    is_gpt_mode = (not use_single_model) and text_model_name and ("gpt" in text_model_name)

    # --- Resume alignment with Trainer ---
    start_epoch = 0
    resume_skip_batches_remaining = 0
    try:
        # updates_per_epoch/total_update_steps는 위에서 계산됨
        start_epoch = int(global_step // max(1, updates_per_epoch))
        steps_trained_in_current_epoch = max(0, int(global_step - (start_epoch * updates_per_epoch)))
        resume_skip_batches_remaining = int(steps_trained_in_current_epoch) * max(1, training_args.gradient_accumulation_steps)
        # 옵션: 재개 시 현재 에폭에서 이미 학습한 배치들을 아예 잘라낸 DataLoader로 교체하여 빠르게 시작 (ENV: RESUME_FAST=1)
        try:
            if resume_skip_batches_remaining > 0 and os.environ.get("RESUME_FAST", "1") == "1":
                per_device_bs = int(getattr(training_args, "per_device_train_batch_size", 1) or 1)
                start_sample_index = int(resume_skip_batches_remaining) * per_device_bs
                total_len = len(train_dataset)
                if start_sample_index < total_len:
                    indices = list(range(start_sample_index, total_len))
                    loader = DataLoader(
                        Subset(train_dataset, indices),
                        batch_size=per_device_bs,
                        shuffle=True,
                        num_workers=training_args.dataloader_num_workers,
                        pin_memory=training_args.dataloader_pin_memory,
                        prefetch_factor = 4,
                        collate_fn=data_collator,
                        **dl_kwargs,
                    )
                    # 이미 학습했던 배치들을 로더에서 제거했으므로 추가 skip 불필요
                    resume_skip_batches_remaining = 0
        except Exception:
            pass
    except Exception:
        start_epoch = 0
        resume_skip_batches_remaining = 0

    # Training Loop
    for epoch in range(start_epoch, num_epochs):
        pbar = tqdm(total=total_update_steps, initial=global_step, desc=f"Steps", ncols=100, dynamic_ncols=False, ascii=True, leave=True, file=sys.stdout, disable=False)
        for step, batch in enumerate(loader):
            # Fast skip previously trained batches in the resumed epoch
            if resume_skip_batches_remaining > 0:
                resume_skip_batches_remaining -= 1
                continue
            if "labels_texts" in batch:
                texts = batch.pop("labels_texts")
                
                import unicodedata as _ud
                texts = [_ud.normalize("NFKD", (t or "")) for t in texts]
                #print(texts[0])
                 
                labels_ids = ocr_processor.tokenizer(texts, padding=True, return_tensors='pt').input_ids
                # ensure long dtype for CE ignore_index behavior
                labels_ids = labels_ids.to(torch.long)
                #print(labels_ids[0])
                pad_id = ocr_processor.tokenizer.pad_token_id
                labels_ids[labels_ids == pad_id] = -100
                # HF DataCollatorForSeq2Seq 정합: pad_to_multiple, DEC_MAX_LEN, DEC_PAD_TO_MAX 적용
                try:
                    import torch.nn.functional as _F
                    _multiple = int(os.environ.get("PAD_TO_MULTIPLE", "0"))
                except Exception:
                    _multiple = 0
                try:
                    _dec_max = int(os.environ.get("DEC_MAX_LEN", "0"))
                except Exception:
                    _dec_max = 0
                _pad_to_max = os.environ.get("DEC_PAD_TO_MAX", "0") == "1"
                if labels_ids.dim() == 2:
                    _T = labels_ids.size(1)
                    _target_T = _T
                    if _pad_to_max and _dec_max and _dec_max > 0 and _target_T < _dec_max:
                        _target_T = _dec_max
                    if _multiple and _multiple > 1:
                        _target_T = ((_target_T + _multiple - 1) // _multiple) * _multiple
                    _pad_len = max(0, _target_T - _T)
                    if _pad_len > 0:
                        labels_ids = _F.pad(labels_ids, (0, _pad_len), value=-100)
                batch["labels"] = labels_ids

            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            
            with torch.amp.autocast("cuda", enabled=(amp_dtype is not None), dtype=amp_dtype):
                dec_attn_mask = (labels != -100).long()
                model_kwargs = {"pixel_values": pixel_values, "labels": labels, "decoder_attention_mask": dec_attn_mask}
                # === Decoder length/padding debugging (enable with DEC_DEBUG_LEN=1) ===
                try:
                    if os.environ.get("DEC_DEBUG_LEN", "0") == "1":
                        _every = int(os.environ.get("DEC_DEBUG_EVERY", "40"))
                        # global_step는 GA 커밋 후 증가하므로, 미니스텝 기준으로 샘플링
                        _should_log = (_every <= 1) or ((step + 1) % max(1, _every) == 0)
                        if _should_log and labels.dim() == 2:
                            _batch_lengths = (labels != -100).long().sum(dim=1)
                            _minL = int(_batch_lengths.min().item()) if _batch_lengths.numel() > 0 else 0
                            _maxL = int(_batch_lengths.max().item()) if _batch_lengths.numel() > 0 else 0
                            logger.info(f"[DEC_LEN] T={labels.size(1)} | batch_len(min={_minL}, max={_maxL}) | mask_T={dec_attn_mask.size(1)}")
                except Exception:
                    pass
                # === end debugging ===
                outputs = model(**model_kwargs)
                
                ls_factor = getattr(training_args, "label_smoothing_factor", 0.0) or 0.0
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                    vocab_size = logits.size(-1)
                    loss_raw = F.cross_entropy(
                        logits.view(-1, vocab_size),
                        labels.view(-1),
                        ignore_index=-100,
                        reduction="mean",
                        label_smoothing=float(ls_factor) if ls_factor > 0 else 0.0,
                    )
                else:
                    loss_raw = outputs.loss
                
                                # ==================== 검증 코드 추가 위치 ====================
                # 딱 한 번만 실행하여 확인
                if global_step == 0 and step == 0:
                    try:
                        print("\n" + "="*80)
                        print("CROSS ENTROPY (-100 IGNORE) VERIFICATION")
                        print("="*80)
                        
                        # 모델 출력 (logits)과 정답 (labels)을 가져옴
                        logits_all = outputs.logits  # Shape: (B, T, V)
                        labels_all = labels          # Shape: (B, T)
                        
                        # 1. PyTorch의 F.cross_entropy가 계산한 원래 loss 값
                        #    (모델이 반환한 loss는 이미 이 계산을 수행한 결과)
                        pytorch_loss = loss_raw.item()
                        print(f"[1] PyTorch's F.cross_entropy loss (with -100): {pytorch_loss:.6f}")

                        # 2. -100 위치를 직접 제거하고 수동으로 loss 계산
                        
                        #   a. 유효한 위치만 필터링
                        #      logits은 (B, T, V) -> (B*T, V)로, labels는 (B, T) -> (B*T)로 펼침
                        logits_flat = logits_all.view(-1, logits_all.size(-1))
                        labels_flat = labels_all.view(-1)
                        
                        #      -100이 아닌 위치의 인덱스만 가져옴 (mask)
                        valid_indices = (labels_flat != -100).nonzero(as_tuple=True)[0]
                        
                        #      해당 인덱스의 logits과 labels만 추출
                        logits_filtered = logits_flat[valid_indices]
                        labels_filtered = labels_flat[valid_indices]
                        
                        #   b. 필터링된 데이터로 cross_entropy loss를 다시 계산
                        #      (ignore_index를 사용하지 않음)
                        manual_loss = F.cross_entropy(logits_filtered, labels_filtered, reduction="mean").item()
                        
                        print(f"[2] Manual cross_entropy loss (after removing -100): {manual_loss:.6f}")
                        
                        # 3. 결과 비교
                        if np.isclose(pytorch_loss, manual_loss, atol=1e-5):
                            print("\n✅ VERIFICATION SUCCESS: The two loss values are nearly identical.")
                            print("This confirms that F.cross_entropy correctly ignores the -100 labels.")
                        else:
                            print("\n⚠️ VERIFICATION FAILED: The loss values do not match.")
                        
                        print("="*80 + "\n")

                    except Exception as e:
                        print(f"Verification failed with an error: {e}")
                # ==================== 검증 코드 끝 ======================
                
                ga_steps = max(1, training_args.gradient_accumulation_steps)
                remaining = len(loader) - step
                current_ga = ga_steps if remaining >= ga_steps else remaining
                loss = loss_raw / max(1, current_ga)

            scaler.scale(loss).backward()
            # 그래프/중간 텐서 참조 해제로 VRAM 누적 방지
            
            if (step + 1) % ga_steps == 0 or (step + 1) == len(loader):
                if training_args.max_grad_norm is not None and training_args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
                scaler.step(optimizer)
                try:
                    if os.environ.get("LOG_ADAPTER_GRAD", "0") == "1":
                        _log_adapter_grad_stats(model, tag="ADAPTER_GRAD@step")
                except Exception:
                    pass
                scaler.update()
                model.zero_grad(set_to_none=True)
                global_step += 1
                pbar.update(1)
                if getattr(training_args, "logging_steps", None) and (global_step % max(1, int(training_args.logging_steps)) == 0):
                    try:
                        if os.environ.get("LOG_ADAPTER_PARAM", "0") == "1":
                            _log_adapter_param_stats(model, tag="ADAPTER_PARAM@train")
                    except Exception:
                        pass
                    try:
                        lr = optimizer.param_groups[0]['lr']
                    except Exception:
                        lr = training_args.learning_rate
                    _post = f"loss {loss.item()*ga_steps:.4f} | lr {lr:.2e} | step {global_step}/{total_update_steps}"
                    if getattr(training_args, "max_grad_norm", None) is not None and getattr(training_args, "max_grad_norm", 0) > 0:
                        try:
                            _post += f" | gnorm {float(total_norm):.2f}"
                        except Exception:
                            pass
                    pbar.set_postfix_str(_post)

                # Training debug: 매 N스텝마다 현재 배치 예측을 generate 기반으로 확인
                try:
                    _dbg_every = int(os.environ.get("TRAIN_DEBUG_EVERY", "5"))
                except Exception:
                    _dbg_every = 5
                if os.environ.get("TRAIN_DEBUG_SHOW", "1") == "1" and _dbg_every > 0 and (global_step % _dbg_every == 0) and (_debug_shown_global < _DEBUG_PRINT_BUDGET):
                    try:
                        with torch.inference_mode():
                            _vp = pixel_values[:2]
                            _vl = labels[:2]
                            # generation config 준비 (평가와 동일 로직)
                            try:
                                base_gen_cfg = (model.generation_config if hasattr(model, 'generation_config') else GenerationConfig.from_model_config(model.config))
                                gen_cfg_train = base_gen_cfg.clone() if hasattr(base_gen_cfg, 'clone') else GenerationConfig(**base_gen_cfg.to_dict())
                            except Exception:
                                gen_cfg_train = GenerationConfig.from_model_config(model.config)
                            try:
                                gen_cfg_train.num_beams = int(os.environ.get("TRAIN_DEBUG_NUM_BEAMS", str(int(getattr(gen_cfg_train, "num_beams", 1) or 1))))
                                # max_length 32 강제 반영
                                gen_cfg_train.max_length = int(getattr(model.config, "max_length", 512) or 512)
                                # train 디버그는 필요 시 max_new로 제한 가능
                                _dbg_max_new = os.environ.get("TRAIN_DEBUG_MAX_NEW", None)
                                if _dbg_max_new is not None:
                                    gen_cfg_train.max_new_tokens = int(_dbg_max_new)
                            except Exception:
                                pass
                            gen_kwargs_train = {"generation_config": gen_cfg_train, "return_dict_in_generate": False, "output_scores": False, "do_sample": False}
                            with torch.amp.autocast("cuda", enabled=(amp_dtype is not None), dtype=amp_dtype):
                                _gen = model.generate(pixel_values=_vp, **gen_kwargs_train)
                            import numpy as _np
                            _pred_txt = ocr_processor.tokenizer.batch_decode(_np.asarray(_gen.detach().cpu()), skip_special_tokens=True)
                            try:
                                import unicodedata as _ud
                                _pred_txt = [_ud.normalize("NFKD", (s or "")) for s in _pred_txt]
                            except Exception:
                                pass
                            _lab = _vl.detach().cpu().clone()
                            _lab[_lab == -100] = ocr_processor.tokenizer.pad_token_id
                            _gt_txt = ocr_processor.tokenizer.batch_decode(_np.asarray(_lab), skip_special_tokens=True)
                            try:
                                import unicodedata as _ud
                                _gt_txt = [_ud.normalize("NFKD", (s or "")) for s in _gt_txt]
                            except Exception:
                                pass
                            for _p, _g in zip(_pred_txt[:5], _gt_txt[:5]):
                                if _debug_shown_global >= _DEBUG_PRINT_BUDGET:
                                    break
                                logger.info(f"[TrainPred@g{global_step}] pred='{_p}' | gt='{_g}'")
                                _debug_shown_global += 1
                            del _gen, _vp, _vl
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                    except Exception:
                        pass

                # Training diagnostics: teacher forcing 토큰 단위 비교 (첫 불일치 위치 포함)
                if os.environ.get("TRAIN_DEBUG_DIAG", "0") == "1" and (_dbg_every > 0) and (global_step % _dbg_every == 0):
                    try:
                        with torch.inference_mode():
                            _vp2 = pixel_values[:1]
                            _vl2 = labels[:1]
                            _vm2 = (_vl2 != -100).long()
                            with torch.amp.autocast("cuda", enabled=(amp_dtype is not None), dtype=amp_dtype):
                                _out2 = model(pixel_values=_vp2, labels=_vl2, decoder_attention_mask=_vm2)
                            _logits = _out2.logits  # [1, T, V]
                            import torch as _t
                            _pred_ids = _logits.argmax(dim=-1)[0]  # [T]
                            _lab_ids = _vl2[0]
                            _mask = _lab_ids != -100
                            _pred_sel = _pred_ids[_mask]
                            _lab_sel = _lab_ids[_mask]
                            _tok_acc = float((_pred_sel == _lab_sel).float().mean().item()) if _pred_sel.numel() > 0 else 0.0
                            # 첫 불일치 인덱스 찾기
                            _mismatch_idx = -1
                            if _pred_sel.numel() > 0:
                                diff = (_pred_sel != _lab_sel).nonzero()
                                if diff.numel() > 0:
                                    _mismatch_idx = int(diff[0].item())
                            # 디코딩(표시용)
                            import numpy as _np
                            _lab_np = _lab_sel.detach().cpu().numpy()
                            _pred_np = _pred_sel.detach().cpu().numpy()
                            _lab_txt = ocr_processor.tokenizer.decode(_lab_np.tolist(), skip_special_tokens=True)
                            _pred_txt2 = ocr_processor.tokenizer.decode(_pred_np.tolist(), skip_special_tokens=True)
                            try:
                                import unicodedata as _ud
                                _lab_txt = _ud.normalize("NFKD", _lab_txt)
                                _pred_txt2 = _ud.normalize("NFKD", _pred_txt2)
                            except Exception:
                                pass
                            logger.info(f"[TrainDiag@g{global_step}] token_acc={_tok_acc:.4f} mismatch_idx={_mismatch_idx}")
                            if _mismatch_idx >= 0:
                                try:
                                    _pi = int(_pred_sel[_mismatch_idx].item())
                                    _li = int(_lab_sel[_mismatch_idx].item())
                                    logger.info(f"[TrainDiag@g{global_step}] pred_id={_pi} vs lab_id={_li}")
                                except Exception:
                                    pass
                            logger.info(f"[TrainDiag@g{global_step}] pred_seq='{_pred_txt2}' | lab_seq='{_lab_txt}'")
                            # 길이 비교 (NFC/NFKD)
                            try:
                                import unicodedata as _ud
                                _pred_nfc = _ud.normalize("NFC", _pred_txt2 or "")
                                _pred_nfkd = _ud.normalize("NFKD", _pred_txt2 or "")
                                _lab_nfc = _ud.normalize("NFC", _lab_txt or "")
                                _lab_nfkd = _ud.normalize("NFKD", _lab_txt or "")
                                logger.info(
                                    f"[TrainDiagLen@g{global_step}] pred_len(nfc={len(_pred_nfc)}, nfkd={len(_pred_nfkd)}) | lab_len(nfc={len(_lab_nfc)}, nfkd={len(_lab_nfkd)})"
                                )
                            except Exception:
                                pass
                            del _out2, _logits, _pred_ids, _vp2, _vl2, _vm2
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                    except Exception:
                        pass

                # Save checkpoint
                if getattr(training_args, "save_steps", 0) and (global_step % max(1, training_args.save_steps) == 0):
                    ckpt_dir = os.path.join(training_args.output_dir, f"checkpoint-{global_step}")
                    try:
                        os.makedirs(ckpt_dir, exist_ok=True)
                        _save_with_opts(model, ckpt_dir, is_base=False, logger_prefix="Ckpt")
                        ocr_processor.save_pretrained(ckpt_dir)
                        torch.save({
                            'optimizer': optimizer.state_dict(),
                            'scaler': scaler.state_dict(),
                            'rng_state_cpu': torch.get_rng_state(),
                            'rng_state_cuda': torch.cuda.get_rng_state_all(),
                            'python_random_state': random.getstate(),
                            'numpy_random_state': np.random.get_state(),
                            'global_step': global_step,
                            'epoch': epoch,
                        }, os.path.join(ckpt_dir, 'trainer_state.pt'))
                        logger.info(f"[Checkpoint] saved {ckpt_dir}")
                    except Exception as _se:
                        logger.warning(f"[Checkpoint] save failed: {_se}")
                
                # Periodic evaluation
                if (val_loader is not None) and (global_step % max(1, eval_every) == 0):
                    model.eval()
                    eval_loss = 0.0
                    eval_count = 0
                    _pred_seqs = []
                    _label_seqs = []
                    with torch.inference_mode():
                        for vb in val_loader:
                            if "labels_texts" in vb:
                                vtexts = vb.pop("labels_texts")
                                # 라벨 NFKD 정규화 (평가 라벨도 동일 규칙 적용)
                                try:
                                    import unicodedata as _ud
                                    vtexts = [_ud.normalize("NFKD", (t or "")) for t in vtexts]
                                except Exception:
                                    pass
                                if is_gpt_mode and tokenizer.bos_token is not None and tokenizer.eos_token is not None:
                                    vtexts = [tokenizer.bos_token + (t or "") + tokenizer.eos_token for t in vtexts]
                                vlabels_ids = ocr_processor.tokenizer(vtexts, padding=True, return_tensors='pt').input_ids
                                vlabels_ids = vlabels_ids.to(torch.long)
                                vpad_id = ocr_processor.tokenizer.pad_token_id
                                vlabels_ids[vlabels_ids == vpad_id] = -100
                                # HF DataCollatorForSeq2Seq 정합: pad_to_multiple, DEC_MAX_LEN, DEC_PAD_TO_MAX 적용
                                try:
                                    import torch.nn.functional as _F
                                    _multiple = int(os.environ.get("PAD_TO_MULTIPLE", "0"))
                                except Exception:
                                    _multiple = 0
                                try:
                                    _dec_max = int(os.environ.get("DEC_MAX_LEN", "0"))
                                except Exception:
                                    _dec_max = 0
                                _pad_to_max = os.environ.get("DEC_PAD_TO_MAX", "0") == "1"
                                if vlabels_ids.dim() == 2:
                                    _T = vlabels_ids.size(1)
                                    _target_T = _T
                                    if _pad_to_max and _dec_max and _dec_max > 0 and _target_T < _dec_max:
                                        _target_T = _dec_max
                                    if _multiple and _multiple > 1:
                                        _target_T = ((_target_T + _multiple - 1) // _multiple) * _multiple
                                    _pad_len = max(0, _target_T - _T)
                                    if _pad_len > 0:
                                        vlabels_ids = _F.pad(vlabels_ids, (0, _pad_len), value=-100)
                                vb["labels"] = vlabels_ids
                            if not vb:
                                continue
                            vp = vb["pixel_values"].to(device)
                            vl = vb["labels"].to(device)
                            vmask = (vl != -100).long()
                            # 우선 전체 배치에 대해 loss 계산
                            with torch.amp.autocast("cuda", enabled=(amp_dtype is not None), dtype=amp_dtype):
                                vout = model(pixel_values=vp, labels=vl, decoder_attention_mask=vmask)
                                if hasattr(vout, "logits"):
                                    vvocab = vout.logits.size(-1)
                                    vloss = F.cross_entropy(
                                        vout.logits.view(-1, vvocab),
                                        vl.view(-1),
                                        ignore_index=-100,
                                        reduction="mean",
                                    )
                                    eval_loss += float(vloss.item()) * vp.size(0)
                                else:
                                    eval_loss += float(vout.loss.item()) * vp.size(0)
                                eval_count += vp.size(0)
                            del vout
                            # 메모리 절약을 위해 generate는 마이크로배치로 수행
                            try:
                                try:
                                    val_gen_chunk = int(os.environ.get("VAL_GEN_CHUNK", "0"))
                                except Exception:
                                    val_gen_chunk = 0
                                if val_gen_chunk <= 0:
                                    try:
                                        # 보수적으로 8 또는 eval bs의 절반 중 작은 값
                                        val_gen_chunk = max(1, min(8, int(getattr(training_args, "per_device_eval_batch_size", 8)) // 2))
                                    except Exception:
                                        val_gen_chunk = 8
                                # 생성 파라미터: config를 그대로 사용하되, 필요한 최소 옵션만 지정
                                # 필요 시 환경변수로 override 가능 (VAL_NUM_BEAMS)
                                num_beams = int(os.environ.get("VAL_NUM_BEAMS", str(int(getattr(model.generation_config if hasattr(model, 'generation_config') else model.config, "num_beams", 1) or 1))))
                                # model.generation_config를 복제하여 안전하게 사용
                                try:
                                    base_gen_cfg = (model.generation_config if hasattr(model, 'generation_config') else GenerationConfig.from_model_config(model.config))
                                    gen_cfg = base_gen_cfg.clone() if hasattr(base_gen_cfg, 'clone') else GenerationConfig(**base_gen_cfg.to_dict())
                                except Exception:
                                    gen_cfg = GenerationConfig.from_model_config(model.config)
                                # 환경변수 override 적용
                                gen_cfg.num_beams = int(num_beams)
                                # max_length 32 강제 반영
                                try:
                                    gen_cfg.max_length = int(getattr(model.config, "max_length", 512) or 512)
                                except Exception:
                                    pass
                                # 반환 최소화
                                gen_kwargs = {"generation_config": gen_cfg, "return_dict_in_generate": False, "output_scores": False, "do_sample": False}
                                prev_uc = getattr(model.config, "use_cache", None)
                                try:
                                    model.config.use_cache = False
                                except Exception:
                                    pass
                                # 디버그: 일부 예측을 사람이 읽게 출력 (최대 5개)
                                want_show = os.environ.get("VAL_SHOW_PREDS", "0") == "1"
                                show_n = 5
                                shown = 0
                                for s in range(0, vp.size(0), val_gen_chunk):
                                    vchunk = vp[s:s+val_gen_chunk]
                                    # === Generate 디버그: 설정/텐서 정보 출력 (ENV: VAL_DEBUG_GEN=1) ===
                                    if os.environ.get("VAL_DEBUG_GEN", "0") == "1":
                                        try:
                                            try:
                                                _cfg_dict = gen_cfg.to_dict()
                                            except Exception:
                                                _cfg_dict = {k: getattr(gen_cfg, k) for k in dir(gen_cfg) if not k.startswith('_') and not callable(getattr(gen_cfg, k, None))}
                                            logger.info(f"[GenDebug] generation_config: {_cfg_dict}")
                                        except Exception:
                                            pass
                                        try:
                                            def _tstats(t):
                                                try:
                                                    td = t.detach()
                                                    fm = td.float()
                                                    _min = float(fm.min().item())
                                                    _max = float(fm.max().item())
                                                    _mean = float(fm.mean().item())
                                                    _nan = bool(torch.isnan(fm).any().item())
                                                    return {"shape": list(td.shape), "dtype": str(td.dtype), "device": str(td.device), "min": _min, "max": _max, "mean": _mean, "has_nan": _nan}
                                                except Exception:
                                                    return {"shape": list(t.shape), "dtype": str(t.dtype), "device": str(t.device)}
                                            logger.info(f"[GenDebug] vchunk: {_tstats(vchunk)}")
                                        except Exception:
                                            pass
                                    with torch.inference_mode():
                                        with torch.amp.autocast("cuda", enabled=(amp_dtype is not None), dtype=amp_dtype):
                                            gen_out = model.generate(pixel_values=vchunk, **gen_kwargs)
                                    if os.environ.get("VAL_DEBUG_GEN", "0") == "1":
                                        try:
                                            print(gen_out)
                                        except Exception:
                                            pass
                                    for i in range(gen_out.size(0)):
                                        _pred_seqs.append(gen_out[i].detach().cpu().tolist())
                                        _label_seqs.append(vl[s+i].detach().cpu().tolist())
                                    if want_show and shown < show_n and (_debug_shown_global < _DEBUG_PRINT_BUDGET):
                                        try:
                                            import numpy as _np
                                            _dec = ocr_processor.tokenizer.batch_decode(_np.asarray(gen_out.detach().cpu()), skip_special_tokens=True)
                                            try:
                                                import unicodedata as _ud
                                                _dec = [_ud.normalize("NFKD", (s or "")) for s in _dec]
                                            except Exception:
                                                pass
                                            _lab = vl[s:s+len(_dec)].detach().cpu().clone()
                                            _lab[_lab == -100] = ocr_processor.tokenizer.pad_token_id
                                            _gt = ocr_processor.tokenizer.batch_decode(_np.asarray(_lab), skip_special_tokens=True)
                                            try:
                                                import unicodedata as _ud
                                                _gt = [_ud.normalize("NFKD", (s or "")) for s in _gt]
                                            except Exception:
                                                pass
                                            for _p, _g in zip(_dec, _gt):
                                                if _debug_shown_global >= _DEBUG_PRINT_BUDGET:
                                                    break
                                                logger.info(f"[EvalPred] pred='{_p}' | gt='{_g}'")
                                                try:
                                                    _pnfc = _ud.normalize("NFC", _p or "")
                                                    _pnfkd = _ud.normalize("NFKD", _p or "")
                                                    _gnfc = _ud.normalize("NFC", _g or "")
                                                    _gnfkd = _ud.normalize("NFKD", _g or "")
                                                    logger.info(
                                                        f"[EvalLen] pred_len(nfc={len(_pnfc)}, nfkd={len(_pnfkd)}) | gt_len(nfc={len(_gnfc)}, nfkd={len(_gnfkd)})"
                                                    )
                                                except Exception:
                                                    pass
                                                _debug_shown_global += 1
                                                shown += 1
                                                if shown >= show_n:
                                                    break
                                        except Exception:
                                            pass
                                    del gen_out, vchunk
                                    try:
                                        torch.cuda.empty_cache()
                                    except Exception:
                                        pass
                                try:
                                    model.config.use_cache = False
                                except Exception:
                                    pass
                            except Exception:
                                # 폴백: logits argmax (generate 실패 시)
                                with torch.amp.autocast("cuda", enabled=(amp_dtype is not None), dtype=amp_dtype):
                                    vout2 = model(pixel_values=vp, labels=vl, decoder_attention_mask=vmask)
                                    preds_ids = vout2.logits.argmax(dim=-1)
                                for i in range(preds_ids.size(0)):
                                    _pred_seqs.append(preds_ids[i].detach().cpu().tolist())
                                    _label_seqs.append(vl[i].detach().cpu().tolist())
                                del vout2, preds_ids
                                try:
                                    torch.cuda.empty_cache()
                                except Exception:
                                    pass
                            del vp, vl, vmask
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                    eval_avg = eval_loss / max(1, eval_count)
                    try:
                        from transformers.trainer_utils import EvalPrediction as _EvalPrediction
                        metrics = compute_metrics(_EvalPrediction(predictions=_pred_seqs, label_ids=_label_seqs), processor=ocr_processor)
                        acc_val = float(metrics.get("accuracy", 0.0))
                        cer_val = float(metrics.get("cer", 0.0))
                        # 선택적 NFC normalize 비교(정규화 차이로 인한 acc 저하 확인)
                        if os.environ.get("METRIC_NFC_COMPARE", "0") == "1":
                            try:
                                import unicodedata as _ud
                                tok = ocr_processor.tokenizer
                                _pad = tok.pad_token_id
                                # label 시퀀스 정규화
                                _lbl_txt = []
                                for _row in _label_seqs:
                                    arr = [x for x in _row if x != -100]
                                    _lbl_txt.append(tok.decode(arr, skip_special_tokens=True))
                                _lbl_txt_n = [_ud.normalize("NFC", s or "") for s in _lbl_txt]
                                # pred 시퀀스 정규화
                                import numpy as _np
                                _pred_txt = tok.batch_decode(_np.asarray(_pred_seqs), skip_special_tokens=True)
                                _pred_txt_n = [_ud.normalize("NFC", s or "") for s in _pred_txt]
                                _acc_n = sum(1 for a, b in zip(_pred_txt_n, _lbl_txt_n) if a == b) / max(1, len(_lbl_txt_n))
                                logger.info(f"[Eval-NFC] acc_nfc {float(_acc_n):.4f}")
                            except Exception:
                                pass
                        logger.info(f"[Eval] step {global_step}: val_loss {eval_avg:.4f} ({eval_count} samples) | acc {acc_val:.4f} | cer {cer_val:.4f}")
                    except Exception as _me:
                        logger.info(f"[Eval] step {global_step}: val_loss {eval_avg:.4f} ({eval_count} samples) | metrics failed: {_me}")
                    model.train()

        pbar.close()
    
    logger.info("[ManualTrain] Finished training")

    # --- 최종 모델 저장 ---
    try:
        if getattr(training_args, "local_rank", -1) in (-1, 0):
            logger.info(f"Saving final model and processor to {training_args.output_dir}")
            _save_with_opts(model, training_args.output_dir, is_base=False, logger_prefix="Save")
            try:
                base = None
                if hasattr(model, "get_base_model") and callable(getattr(model, "get_base_model")):
                    base = model.get_base_model()
                if base is None: base = model
                if hasattr(base, "save_pretrained"):
                    _save_with_opts(base, training_args.output_dir, is_base=True, logger_prefix="Save")
            except Exception as e:
                logger.warning(f"[Save] Final save (base model) failed: {e}")
            try:
                ocr_processor.save_pretrained(training_args.output_dir)
            except Exception as e:
                logger.warning(f"[Save] Processor save failed: {e}")
    except Exception as e:
        logger.warning(f"[Save] Final save failed: {e}")


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DatasetsArguments, MyTrainingArguments))
    model_args, dataset_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
        stream=sys.stdout,
    )

    logger.info("Training/evaluation parameters %s", training_args)

    main(model_args=model_args, dataset_args=dataset_args, training_args=training_args)