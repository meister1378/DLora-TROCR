import random
from os import PathLike
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedTokenizerBase, TrOCRProcessor
from transformers.trainer_utils import EvalPrediction

from literal import RawDataColumns
from utils.dataset_utils import to_subchar


def seed_everything(random_seed: int) -> None:
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
# ===== CER metric (evaluate.load("cer") 우선, 폴백 구현) =====
try:
    import evaluate as _evaluate  # type: ignore
    _cer_metric = _evaluate.load("cer")
except Exception:
    _cer_metric = None

def _compute_cer_fallback(preds: List[str], refs: List[str]) -> float:
    # 간단한 문자 레벤슈타인 기반 CER 폴백(정확성보단 가용성 우선)
    try:
        import Levenshtein as _lev  # type: ignore
        total_dist = 0
        total_chars = 0
        for p, r in zip(preds, refs):
            total_dist += _lev.distance(p or "", r or "")
            total_chars += max(1, len(r or ""))
        return float(total_dist) / float(total_chars) if total_chars else 0.0
    except Exception:
        # 매우 단순한 폴백: 동일 여부만 반영(실제 CER 대체용)
        total_dist = 0
        total_chars = 0
        for p, r in zip(preds, refs):
            total_dist += 0 if (p or "") == (r or "") else max(1, len(r or ""))
            total_chars += max(1, len(r or ""))
        return float(total_dist) / float(total_chars) if total_chars else 0.0



def __get_total_label(
    train_csv_path: Union[None, PathLike] = None, valid_csv_path: Union[None, PathLike] = None, is_sub_char=True
) -> List[str]:
    if not train_csv_path is None:
        train_df = pd.read_csv(train_csv_path)
    if not valid_csv_path is None:
        valid_df = pd.read_csv(valid_csv_path)

    total_df = pd.concat([train_df, valid_df]).reset_index(drop=True)

    total_labels = total_df[RawDataColumns.label]
    total_labels = list(set(total_labels))
    total_labels.sort()
    if is_sub_char:
        total_labels = list(map(lambda x: to_subchar(x), total_labels))
    return total_labels


def add_label_tokens(
    tokenizer: PreTrainedTokenizerBase,
    train_csv_path: Union[None, PathLike] = None,
    valid_csv_path: Union[None, PathLike] = None,
    is_sub_char: bool = False,
) -> None:
    """
    label을 토크나이징 해서 unk 토큰에 해당하는 단어들을 vocab에 추가해주는 함수
    """

    total_labels = __get_total_label(train_csv_path, valid_csv_path, is_sub_char)
    tokenized_labels = tokenizer(total_labels).input_ids
    unks = []
    for idx, tokenized_label in enumerate(tokenized_labels):
        if tokenizer.unk_token_id in tokenized_label:
            unks.append(total_labels[idx])
    new_tokens = list(set(unks))

    tokenizer.add_tokens(new_tokens)
    return


def has_unk_token(
    tokenizer: PreTrainedTokenizerBase,
    train_csv_path: Union[None, PathLike] = None,
    valid_csv_path: Union[None, PathLike] = None,
    is_sub_char: bool = False,
) -> bool:
    """
    label을 토크나이징 수행시 unk 토큰이 있을 경우 True, 없으면 False
    """
    total_labels = __get_total_label(train_csv_path, valid_csv_path, is_sub_char)
    has_unk = False
    tokenized_labels = tokenizer(total_labels).input_ids
    for tokenized_label in tokenized_labels:
        if tokenizer.unk_token_id in tokenized_label:
            has_unk = True
            break

    return has_unk


# ===== UNK 토큰 추가 유틸 (CSV 없이도 사용 가능) =====
def add_unk_tokens_from_texts(
    tokenizer: PreTrainedTokenizerBase,
    texts: List[str],
    *,
    char_level: bool = True,
    add_limit: int = 4096,
) -> int:
    """
    주어진 텍스트들에서 토크나이저에 없는 토큰을 추가한다.
    - char_level=True: 문자 단위로 vocab에 없는 문자들을 추가
    - char_level=False: 텍스트 전체를 하나의 토큰으로 추가(해당 텍스트가 UNK를 생성하는 경우)
    반환값: 추가된 토큰 개수
    """
    if not texts:
        return 0
    try:
        vocab = set(tokenizer.get_vocab().keys())
    except Exception:
        vocab = set()

    candidates = set()
    if char_level:
        for t in texts:
            if not isinstance(t, str):
                continue
            for ch in t:
                if ch and (ch not in vocab):
                    candidates.add(ch)
    else:
        unk_id = tokenizer.unk_token_id
        if unk_id is None:
            return 0
        ids = tokenizer(texts).input_ids
        for i, token_ids in enumerate(ids):
            if unk_id in token_ids:
                txt = texts[i]
                if isinstance(txt, str) and (txt not in vocab):
                    candidates.add(txt)

    if not candidates:
        return 0

    # 상한 제한
    if len(candidates) > add_limit:
        candidates = set(list(candidates)[:add_limit])

    try:
        num_added = tokenizer.add_tokens(list(candidates))
        return int(num_added)
    except Exception:
        return 0


def compute_metrics(pred: EvalPrediction, processor: TrOCRProcessor) -> Dict[str,float]:
    """메모리 안전하고 비정형 길이(batch 내 가변 길이)도 처리 가능한 metrics.
    - labels/preds가 리스트-중첩/불규칙 길이여도 per-sample로 디코딩
    - logits 형태(preds.ndim>=3)면 argmax 후 디코딩
    """
    tok = processor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    # 1) 라벨 per-sample 디코딩 (-100 토큰은 완전히 제거; pad로 대체하지 않음)
    labels_raw = pred.label_ids
    references: List[str] = []
    try:
        import numpy as _np
        if hasattr(labels_raw, "dtype") and labels_raw.ndim == 2:
            labels_arr = labels_raw.copy()
            labels_arr[labels_arr == -100] = pad_id
            references = tok.batch_decode(labels_arr, skip_special_tokens=True)
        else:
            # 불규칙 길이 처리: 각 시퀀스별 변환
            for seq in list(labels_raw):
                try:
                    arr = _np.asarray(seq)
                    arr = arr.copy()
                    seq_list = [int(x) for x in arr.tolist() if int(x) != -100]
                    references.append(tok.decode(seq_list, skip_special_tokens=True))
                except Exception:
                    try:
                        seq_list = [int(x) for x in seq if int(x) != -100]
                    except Exception:
                        seq_list = []
                    references.append(tok.decode(seq_list, skip_special_tokens=True))
    except Exception:
        # 최후 폴백
        try:
            references = tok.batch_decode(labels_raw, skip_special_tokens=True)
        except Exception:
            references = []

    # 2) 예측 per-sample 디코딩 (logits/ids 모두 허용)
    preds_raw = pred.predictions
    # Unwrap tuple
    if isinstance(preds_raw, tuple) and len(preds_raw) > 0:
        preds_raw = preds_raw[0]

    predictions: List[str] = []
    try:
        import numpy as _np
        if hasattr(preds_raw, "dtype"):
            preds_arr = preds_raw
            if preds_arr.ndim >= 3:
                preds_arr = preds_arr.argmax(axis=-1)
            elif preds_arr.ndim == 2 and _np.issubdtype(preds_arr.dtype, _np.floating):
                preds_arr = preds_arr.argmax(axis=-1)
            predictions = tok.batch_decode(preds_arr, skip_special_tokens=True)
        else:
            # 리스트/불규칙 길이 처리
            for row in list(preds_raw):
                try:
                    arr = _np.asarray(row)
                    if _np.issubdtype(arr.dtype, _np.floating):
                        arr = arr.argmax(axis=-1)
                    predictions.append(tok.decode(arr.tolist(), skip_special_tokens=True))
                except Exception:
                    try:
                        predictions.append(tok.decode([int(x) for x in row], skip_special_tokens=True))
                    except Exception:
                        predictions.append("")
    except Exception:
        # 최후 폴백
        try:
            predictions = tok.batch_decode(preds_raw, skip_special_tokens=True)
        except Exception:
            predictions = [""] * len(references)

    # NFKD 정규화로 예측/정답 기준 일치화
    try:
        import unicodedata as _ud
        references = [_ud.normalize("NFKD", (s or "")) for s in references]
        predictions = [_ud.normalize("NFKD", (s or "")) for s in predictions]
    except Exception:
        pass

    # Exact match accuracy
    total = len(references) if references is not None else 0
    acc = 0
    for i in range(total):
        if references[i] == predictions[i]:
            acc += 1
    acc = (acc / total) if total else 0.0

    # CER (Character Error Rate)
    if _cer_metric is not None:
        try:
            cer = float(_cer_metric.compute(predictions=predictions, references=references))
        except Exception:
            cer = _compute_cer_fallback(predictions, references)
    else:
        cer = _compute_cer_fallback(predictions, references)

    return {"accuracy": acc, "cer": cer}
