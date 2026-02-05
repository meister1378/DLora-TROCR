import os
import re
from unicodedata import normalize

import pandas as pd
from datasets import Dataset, Image
import glob
import lmdb
import pickle
import numpy as np
import cv2
from PIL import Image as PILImage
import torch
from torch.utils.data import Dataset as TorchDataset
from typing import Optional, Callable

from literal import DatasetColumns, RawDataColumns


def to_subchar(string: str) -> str:
    """
    유니코드 NFKD로 정규화
    """
    return normalize("NFKD", string)


def clean_text(text: str) -> str:
    """
    텍스트의 자모 및 공백을 삭제
    ex) "바ㄴ나 나" -> "바나나"
    """
    text = re.sub(r"[^가-힣]", "", text)
    return text


def get_dataset(csv_path: os.PathLike, is_sub_char=True) -> Dataset:
    """
    csv의 경로를 입력받아 Dataset을 리턴
    is_sub_char: "snunlp/KR-BERT-char16424"와 같은 sub_char tokenizer일 경우 True, 일반적인 토크나이저일 경우에는 False
    feature: pixel_values(PIL image), labels(str)

    """
    df = pd.read_csv(csv_path)

    data_dict = {DatasetColumns.pixel_values: df[RawDataColumns.img_path].tolist()}

    if RawDataColumns.label in df.columns:
        if is_sub_char:
            df[RawDataColumns.label] = df[RawDataColumns.label].apply(to_subchar)
        data_dict[DatasetColumns.labels] = df[RawDataColumns.label].tolist()

    dataset = Dataset.from_dict(data_dict)
    # decode=True로 이미지를 즉시 로드하고, 모든 이미지를 RGB로 변환하여 채널 오류 방지
    dataset = dataset.cast_column(DatasetColumns.pixel_values, Image(decode=True))
    dataset = dataset.map(lambda example: {'pixel_values': example['pixel_values'].convert('RGB')})
    return dataset


def load_lmdbs_as_recognition_dataset(lmdb_glob_pattern: str,
                                      is_sub_char: bool = True,
                                      min_crop_wh: int = 8,
                                      max_words: int = 200,
                                      max_label_len: int = 64,
                                      only_train: bool = True,
                                      max_total_samples: int | None = None) -> Dataset:
    """
    여러 recognition LMDB(image-/label- 키)를 병합해 HF Dataset으로 변환.
    - image-000000123: JPEG bytes
    - label-000000123: UTF-8 텍스트
    """
    lmdb_paths = sorted(glob.glob(lmdb_glob_pattern))
    if only_train:
        lmdb_paths = [p for p in lmdb_paths if 'train' in os.path.basename(p).lower()]
    if not lmdb_paths:
        raise FileNotFoundError(f"No LMDB found: {lmdb_glob_pattern}")

    img_paths = []  # 메모리 절약 위해 바이트를 즉시 디코드 -> PIL 보관 대신 파일경로 유사 식별자 사용
    texts = []
    pil_images = []

    for db_path in lmdb_paths:
        env = lmdb.open(db_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            n = int(txn.get(b"num-samples").decode())
            for idx in range(n):
                ikey = f"image-{idx:09d}".encode()
                lkey = f"label-{idx:09d}".encode()
                ib = txn.get(ikey)
                lb = txn.get(lkey)
                if ib is None or lb is None:
                    continue
                # 디코드
                npb = np.frombuffer(ib, dtype=np.uint8)
                img = cv2.imdecode(npb, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                h, w = img.shape[:2]
                if h < min_crop_wh or w < min_crop_wh:
                    continue
                txt = lb.decode('utf-8', errors='ignore').strip()
                if not txt or txt == '###':
                    continue
                if len(txt) > max_label_len:
                    txt = txt[:max_label_len]
                # BGR->RGB, PIL 변환
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil = PILImage.fromarray(img_rgb)
                pil_images.append(pil)
                texts.append(txt)
        env.close()

    data_dict = {DatasetColumns.pixel_values: pil_images, DatasetColumns.labels: texts}
    ds = Dataset.from_dict(data_dict)
    ds = ds.cast_column(DatasetColumns.pixel_values, Image())
    if max_total_samples is not None:
        ds = ds.select(range(min(int(max_total_samples), len(ds))))
    return ds


class LMDBRecogIndexDataset(TorchDataset):
    """LMDB 인식용(크롭+라벨) 스트리밍 데이터셋.
    - 전량 메모리 적재 없이 인덱스만 구축하고 __getitem__에서 디코딩
    """
    def __init__(self,
                 lmdb_glob_pattern: str,
                 min_crop_wh: int = 8,
                 max_label_len: int = 64,
                 only_train: bool = True,
                 transform: Optional[Callable] = None,
                 max_total_samples: Optional[int] = None):
        self.min_crop_wh = int(min_crop_wh)
        self.max_label_len = int(max_label_len)
        self.transform = transform
        paths = sorted(glob.glob(lmdb_glob_pattern))
        if only_train:
            paths = [p for p in paths if 'train' in os.path.basename(p).lower()]
        if not paths:
            raise FileNotFoundError(f"No LMDB found: {lmdb_glob_pattern}")
        # (lmdb_path, index) 리스트와 각 DB의 키 접두사 스킴 저장
        self.indices: list[tuple[str, int]] = []
        self.key_scheme: dict[str, tuple[str, str]] = {}  # { path: (image_prefix, label_prefix) }

        image_prefix_candidates = ["image", "img"]
        label_prefix_candidates = ["label", "word", "text"]

        for p in paths:
            env = lmdb.open(p, readonly=True, lock=False, readahead=False, meminit=False)
            try:
                with env.begin(write=False) as txn:
                    num_raw = txn.get(b"num-samples")
                    img_prefix = None
                    lbl_prefix = None
                    idx_list: list[int] = []

                    if num_raw is not None:
                        # num-samples 기반 인덱스 생성
                        try:
                            n = int(num_raw.decode())
                        except Exception:
                            n = int(num_raw)

                        # 접두사 자동 감지 (0 또는 1 시작 인덱스 고려)
                        # 이미지 접두사
                        start_idx = 0
                        for cand in image_prefix_candidates:
                            if txn.get(f"{cand}-000000000".encode()) is not None:
                                img_prefix = cand
                                start_idx = 0
                                break
                            if txn.get(f"{cand}-000000001".encode()) is not None:
                                img_prefix = cand
                                start_idx = 1
                                break
                        if img_prefix is None:
                            img_prefix = image_prefix_candidates[0]
                            start_idx = 0

                        # 라벨 접두사 (시작 인덱스 일치 여부는 아래에서 교집합으로 보정될 수 있음)
                        lbl_start_idx = start_idx
                        for cand in label_prefix_candidates:
                            if txn.get(f"{cand}-000000000".encode()) is not None:
                                lbl_prefix = cand
                                lbl_start_idx = 0
                                break
                            if txn.get(f"{cand}-000000001".encode()) is not None:
                                lbl_prefix = cand
                                lbl_start_idx = 1
                                break
                        if lbl_prefix is None:
                            lbl_prefix = label_prefix_candidates[0]
                            lbl_start_idx = start_idx

                        # 인덱스 목록 구성: 시작 인덱스 차이가 있을 수 있어 검증 후 확정
                        cand_indices = list(range(start_idx, start_idx + n))
                        # 라벨 쪽과 불일치 시 교집합만 사용 (빠른 샘플 검증)
                        # 0,1, n-1, n 몇 개만 확인하여 유효한 것만 남김
                        check_samples = [cand_indices[0]]
                        if len(cand_indices) > 1:
                            check_samples.append(cand_indices[1])
                        check_samples.append(cand_indices[-1])
                        valid_set = set()
                        for ci in check_samples:
                            ib = txn.get(f"{img_prefix}-{ci:09d}".encode())
                            lb = txn.get(f"{lbl_prefix}-{ci:09d}".encode())
                            if ib is not None and lb is not None:
                                valid_set.add(ci)
                        if valid_set:
                            # 일부라도 유효하면 전체 범위를 사용하고, __getitem__에서 누락은 건너뜀
                            idx_list = cand_indices
                        else:
                            # 시작 인덱스 0/1 모두 실패한 경우, 보수적으로 0..n-1 사용
                            idx_list = list(range(n))
                    else:
                        # num-samples가 없으면 커서로 키 스캔하여 인덱스 구축
                        image_indices = {}
                        label_indices = {}
                        with txn.cursor() as cur:
                            for k, _ in cur:
                                try:
                                    ks = k.decode()
                                except Exception:
                                    continue
                                # image/img-XXXXXXXXX
                                m_img = re.match(r"^(image|img)-(\d+)$", ks)
                                if m_img:
                                    pref = m_img.group(1)
                                    i = int(m_img.group(2))
                                    image_indices.setdefault(pref, set()).add(i)
                                    continue
                                # label/word/text-XXXXXXXXX
                                m_lbl = re.match(r"^(label|word|text)-(\d+)$", ks)
                                if m_lbl:
                                    pref = m_lbl.group(1)
                                    i = int(m_lbl.group(2))
                                    label_indices.setdefault(pref, set()).add(i)
                                    continue

                        # 접두사 선택: 가장 많은 매치를 가진 것을 우선
                        if image_indices:
                            img_prefix = max(image_indices.keys(), key=lambda k: len(image_indices[k]))
                        if label_indices:
                            lbl_prefix = max(label_indices.keys(), key=lambda k: len(label_indices[k]))

                        if not image_indices or not label_indices:
                            raise FileNotFoundError(
                                f"LMDB '{p}' does not contain recognizable 'image-*' and 'label-*' style keys"
                            )

                        common = image_indices[img_prefix] & label_indices[lbl_prefix]
                        idx_list = sorted(common)

                    # 키 스킴과 인덱스 저장
                    self.key_scheme[p] = (img_prefix, lbl_prefix)
                    self.indices.extend([(p, i) for i in idx_list])
            finally:
                env.close()
        if max_total_samples is not None:
            self.indices = self.indices[:int(max_total_samples)]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        lmdb_path, i = self.indices[idx]
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        try:
            with env.begin(write=False) as txn:
                img_prefix, lbl_prefix = self.key_scheme.get(lmdb_path, ("image", "label"))
                ib = txn.get(f"{img_prefix}-{i:09d}".encode())
                lb = txn.get(f"{lbl_prefix}-{i:09d}".encode())
                if ib is None or lb is None:
                    raise IndexError("missing keys")
                txt = lb.decode('utf-8', errors='ignore').strip()
                if not txt or txt == '###':
                    raise IndexError("empty label")
                if len(txt) > self.max_label_len:
                    txt = txt[:self.max_label_len]
                npb = np.frombuffer(ib, dtype=np.uint8)
                img = cv2.imdecode(npb, cv2.IMREAD_COLOR)
                if img is None:
                    raise IndexError("decode fail")
                h,w = img.shape[:2]
                if h < self.min_crop_wh or w < self.min_crop_wh:
                    # 너무 작은 크롭은 최소 크기로 업샘플하여 사용 (학습 중단 방지)
                    scale = max(self.min_crop_wh / max(1, h), self.min_crop_wh / max(1, w))
                    new_w = max(self.min_crop_wh, int(round(w * scale)))
                    new_h = max(self.min_crop_wh, int(round(h * scale)))
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    h, w = img.shape[:2]
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil = PILImage.fromarray(img_rgb)
                if self.transform is not None:
                    pil = self.transform(pil)
                return { 'pixel_values': pil, 'labels': txt }
        finally:
            env.close()
