#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LMDBSample 데이터셋

fast_sample_data.Sample 전처리 파이프라인을 그대로 따르면서,
이미지/어노테이션 소스만 LMDB로 바꾼 전용 데이터셋 클래스.

주의: 기존 FAST_LMDB 코드를 전혀 참조하지 않고, fast_sample_data.Sample 과
dataset.utils 의 함수들만을 기준으로 재구현.
"""

import os
from io import BytesIO
import pickle
import random
from typing import Tuple, List, Dict, Any, Optional, Union

import cv2
import lmdb
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from torch.utils import data

from dataset.utils import (
    random_scale,
    random_horizontal_flip,
    random_rotate,
    random_crop_padding_v2 as random_crop_padding,
    scale_aligned_short,
    shrink,
    get_vocabulary,
)


class LMDBSample(data.Dataset):
    def __init__(
        self,
        lmdb_path: str,
        split: str = 'train',
        is_transform: bool = False,
        img_size: Optional[Union[Tuple[int, int], int]] = None,
        short_size: int = 736,
        pooling_size: int = 9,
        with_rec: bool = False,
        read_type: str = 'cv2',
        repeat_times: int = 1,
        report_speed: bool = False,
    ) -> None:
        self.lmdb_path = lmdb_path
        self.split = split
        self.is_transform = is_transform
        self.img_size = img_size if (img_size is None or isinstance(img_size, tuple)) else (img_size, img_size)
        self.pooling_size = pooling_size
        self.short_size = short_size
        self.with_rec = with_rec
        self.read_type = read_type
        self.repeat_times = max(1, int(repeat_times))

        # Sample 과 동일한 pooling 유닛
        self.pad = nn.ZeroPad2d(padding=(pooling_size - 1) // 2)
        self.pooling = nn.MaxPool2d(kernel_size=pooling_size, stride=1)
        self.overlap_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        if not os.path.exists(self.lmdb_path):
            raise FileNotFoundError(f"LMDB 경로가 존재하지 않습니다: {self.lmdb_path}")

        # 읽기 모드로 LMDB 열기
        self.env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        # 샘플 개수 조회
        with self.env.begin(write=False) as txn:
            length_bytes = txn.get('num-samples'.encode())
            if length_bytes is None:
                raise KeyError("'num-samples' 키가 LMDB에 없습니다.")
            self.length = int(length_bytes.decode())

        # 반복 배수 적용 (len(dataset) 확대)
        self.total_length = self.length * self.repeat_times

        # 어휘 사전 (Sample 과 동일)
        self.voc, self.char2id, self.id2char = get_vocabulary('LOWERCASE')
        self.max_word_num = 200
        self.max_word_len = 32
        self._bad_indices = set()

    def __len__(self) -> int:
        return self.total_length

    def __del__(self) -> None:
        try:
            if hasattr(self, 'env') and self.env is not None:
                self.env.close()
        except Exception:
            pass

    # ---------- Sample 과 동일 동작 유틸 ----------
    def min_pooling(self, input_tensor: np.ndarray) -> np.ndarray:
        input_tensor_t = torch.tensor(input_tensor, dtype=torch.float)
        temp = input_tensor_t.sum(dim=0).to(torch.uint8)
        overlap = (temp > 1).to(torch.float32).unsqueeze(0).unsqueeze(0)
        overlap = self.overlap_pool(overlap).squeeze(0).squeeze(0)

        B = input_tensor_t.size(0)
        h_sum = input_tensor_t.sum(dim=2) > 0

        h_sum_ = h_sum.long() * torch.arange(h_sum.shape[1], 0, -1)
        h_min = torch.argmax(h_sum_, 1, keepdim=True)
        h_sum_ = h_sum.long() * torch.arange(1, h_sum.shape[1] + 1)
        h_max = torch.argmax(h_sum_, 1, keepdim=True)

        w_sum = input_tensor_t.sum(dim=1) > 0
        w_sum_ = w_sum.long() * torch.arange(w_sum.shape[1], 0, -1)
        w_min = torch.argmax(w_sum_, 1, keepdim=True)
        w_sum_ = w_sum.long() * torch.arange(1, w_sum.shape[1] + 1)
        w_max = torch.argmax(w_sum_, 1, keepdim=True)

        for i in range(B):
            region = input_tensor_t[i:i + 1, h_min[i]:h_max[i] + 1, w_min[i]:w_max[i] + 1]
            region = self.pad(region)
            region = -self.pooling(-region)
            input_tensor_t[i:i + 1, h_min[i]:h_max[i] + 1, w_min[i]:w_max[i] + 1] = region

        x = input_tensor_t.sum(dim=0).to(torch.uint8)
        x[overlap > 0] = 0
        return x.numpy()

    # ---------- LMDB 접근 ----------
    def _decode_image_bytes_to_rgb(self, img_bytes: bytes) -> Optional[np.ndarray]:
        if img_bytes is None or len(img_bytes) == 0:
            return None
        # 1) OpenCV 우선 시도
        try:
            img_np = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
            if img is not None:
                # BGRA/GRAY 처리 후 RGB로 통일
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img
        except Exception:
            pass
        # 2) PIL 대체 시도
        try:
            with BytesIO(img_bytes) as bio:
                pil = Image.open(bio)
                pil = pil.convert('RGB')
                return np.array(pil)
        except Exception:
            return None

    def _get_from_lmdb(self, index: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        real_index = index % self.length
        with self.env.begin(write=False) as txn:
            img_key = f'image-{real_index:09d}'.encode()
            gt_key = f'gt-{real_index:09d}'.encode()

            img_bytes = txn.get(img_key)
            if img_bytes is None:
                raise KeyError(f"이미지 키 누락: {img_key}")
            img = self._decode_image_bytes_to_rgb(img_bytes)
            if img is None:
                raise ValueError(f"이미지 디코딩 실패: index={real_index}")

            gt_bytes = txn.get(gt_key)
            if gt_bytes is None:
                raise KeyError(f"GT 키 누락: {gt_key}")
            gt_info = pickle.loads(gt_bytes)

        return img, gt_info

    # ---------- Sample 과 동일 파이프라인 ----------
    def _prepare_train(self, index: int) -> Dict[str, Any]:
        img, gt_info = self._get_from_lmdb(index)

        # bbox 로드 (정규화 여부 자동 판별)
        bboxes = np.array(gt_info.get('bboxes', []), dtype=np.float32)
        words: List[str] = gt_info.get('words', [])

        if bboxes.shape[0] > self.max_word_num:
            bboxes = bboxes[:self.max_word_num]
            words = words[:self.max_word_num]

        # 원본 크기 저장
        h0, w0 = img.shape[:2]

        # 이미지 스케일링 (Sample과 동일)
        if self.is_transform:
            img = random_scale(img, self.short_size, scales=[0.5, 2.0], aspects=[0.9, 1.1])

        h1, w1 = img.shape[:2]

        # 정규화 여부 판단: 0<=min<=1 and max<=1.0+eps 이면 정규화로 간주
        is_normalized = False
        if bboxes.size > 0:
            bmin = float(bboxes.min())
            bmax = float(bboxes.max())
            is_normalized = (bmin >= -1e-6 and bmax <= 1.000001)

        # 픽셀 좌표 생성
        if bboxes.size > 0:
            if is_normalized:
                # 정규화 → 현재 이미지 크기로 변환
                bboxes_pix = np.reshape(
                    bboxes * ([w1, h1] * 4),
                    (bboxes.shape[0], -1, 2)
                ).astype('int32')
            else:
                # 이미 픽셀 좌표 → 스케일 변화 반영
                scale_w = w1 / max(1, w0)
                scale_h = h1 / max(1, h0)
                bboxes_pix = np.reshape(
                    bboxes * ([scale_w, scale_h] * 4),
                    (bboxes.shape[0], -1, 2)
                ).astype('int32')

            # 마스크 생성
            gt_instance = np.zeros((h1, w1), dtype='uint8')
            training_mask = np.ones((h1, w1), dtype='uint8')
            for i in range(bboxes_pix.shape[0]):
                if i < len(words) and words[i] == '###':
                    cv2.drawContours(training_mask, [bboxes_pix[i]], -1, 0, -1)
                else:
                    cv2.drawContours(gt_instance, [bboxes_pix[i]], -1, i + 1, -1)
        else:
            bboxes_pix = np.zeros((0, 4, 2), dtype='int32')
            gt_instance = np.zeros((h1, w1), dtype='uint8')
            training_mask = np.ones((h1, w1), dtype='uint8')

        # 커널 생성 (min_pooling + shrink 병합)
        gt_kernels_list: List[np.ndarray] = []
        for i in range(len(bboxes_pix)):
            gt_kernel = np.zeros(img.shape[0:2], dtype='uint8')
            if i < len(words) and words[i] != '###':
                cv2.drawContours(gt_kernel, [bboxes_pix[i]], -1, 1, -1)
                gt_kernels_list.append(gt_kernel)
            else:
                if len(gt_kernels_list) == 0:
                    gt_kernels_list.append(gt_kernel)

        if len(gt_kernels_list) > 0:
            gt_kernels = np.array(gt_kernels_list)
            gt_kernel = self.min_pooling(gt_kernels)
        else:
            gt_kernel = np.zeros(img.shape[0:2], dtype='uint8')

        # shrink kernel
        shrink_kernel_scale = 0.1
        gt_kernel_shrinked = np.zeros(img.shape[0:2], dtype='uint8')
        kernel_bboxes = shrink(bboxes_pix, shrink_kernel_scale)
        for i in range(len(bboxes_pix)):
            if i < len(words) and words[i] != '###':
                cv2.drawContours(gt_kernel_shrinked, [kernel_bboxes[i]], -1, 1, -1)
        gt_kernel = np.maximum(gt_kernel, gt_kernel_shrinked)

        # 기하학 변환 (flip/rotate/crop+pad)
        if self.is_transform:
            imgs = [img, gt_instance, training_mask, gt_kernel]
            if not self.with_rec:
                imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs, random_angle=30)
            imgs = random_crop_padding(imgs, self.img_size)
            img, gt_instance, training_mask, gt_kernel = imgs[0], imgs[1], imgs[2], imgs[3]

        # gt_text 생성
        gt_text = gt_instance.copy()
        gt_text[gt_text > 0] = 1

        # 최종 이미지 전처리 (PIL, Blur, ColorJitter, ToTensor, Normalize)
        pil_img = Image.fromarray(img).convert('RGB')
        if self.is_transform:
            if random.random() < 0.5:
                pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=random.random()))
            pil_img = transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            )(pil_img)

        img_t = transforms.ToTensor()(pil_img)
        img_t = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_t)

        data_dict: Dict[str, Any] = dict(
            imgs=img_t,
            gt_texts=torch.from_numpy(gt_text).long(),
            gt_kernels=torch.from_numpy(gt_kernel).long(),
            training_masks=torch.from_numpy(training_mask).long(),
            gt_instances=torch.from_numpy(gt_instance).long(),
        )

        return data_dict

    def _prepare_test(self, index: int) -> Dict[str, Any]:
        img, gt_info = self._get_from_lmdb(index)

        img_meta = dict(org_img_size=np.array(img.shape[:2]))
        img = scale_aligned_short(img, self.short_size)
        img_meta.update(dict(img_size=np.array(img.shape[:2]), filename=gt_info.get('filename', f'image_{index:06d}')))

        pil_img = Image.fromarray(img).convert('RGB')
        img_t = transforms.ToTensor()(pil_img)
        img_t = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_t)

        return dict(imgs=img_t, img_metas=img_meta)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # 디코딩 실패 등 예외 발생 시 몇 차례 재시도하여 학습이 중단되지 않도록 처리
        max_retry = 6
        for attempt in range(max_retry):
            try:
                if self.split == 'train':
                    return self._prepare_train(index)
                elif self.split == 'test':
                    return self._prepare_test(index)
                else:
                    raise ValueError("split must be 'train' or 'test'")
            except ValueError:
                self._bad_indices.add(index % self.length)
                index = random.randint(0, self.total_length - 1)
                continue
        # 마지막에도 실패하면 예외를 올려 근본 원인 확인
        raise ValueError(f"샘플 로드 실패(재시도 초과). bad_indices 예시: {list(self._bad_indices)[:10]}")


