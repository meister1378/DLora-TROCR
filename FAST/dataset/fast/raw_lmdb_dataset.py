#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Raw LMDB Dataset
전처리 없이 원본 이미지를 로드하는 데이터셋 클래스
"""

import os
import sys
import pickle
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data

try:
    import lmdb
except ImportError:
    print("❌ LMDB 패키지가 설치되지 않았습니다. 'pip install lmdb' 로 설치해주세요.")
    sys.exit(1)

from dataset.utils import shrink
from dataset.utils import get_vocabulary
from dataset.utils import random_crop_padding_v2 as random_crop_padding
from dataset.utils import random_scale, random_horizontal_flip, random_rotate
from dataset.utils import scale_aligned_short


class RawLMDBDataset(data.Dataset):
    """
    원본 이미지를 전처리 없이 로드하는 LMDB 데이터셋 클래스
    
    특징:
    - 원본 이미지 그대로 로드
    - 전처리는 모델 forward 시에만 적용
    - bbox 좌표도 원본 그대로 유지
    """
    
    def __init__(self, lmdb_path, split='train', max_word_num=200, read_type='cv2'):
        """
        Args:
            lmdb_path (str): LMDB 데이터베이스 경로
            split (str): 'train' 또는 'test'
            max_word_num (int): 최대 단어 수
            read_type (str): 이미지 읽기 방식 ('cv2' 또는 'pil')
        """
        self.lmdb_path = lmdb_path
        self.split = split
        self.max_word_num = max_word_num
        self.read_type = read_type
        
        # LMDB 환경 열기
        print(f"🗂️ 원본 LMDB 데이터베이스 열기: {lmdb_path}")
        if not os.path.exists(lmdb_path):
            raise FileNotFoundError(f"LMDB 데이터베이스를 찾을 수 없습니다: {lmdb_path}")
        
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        
        # 데이터 크기 확인
        with self.env.begin(write=False) as txn:
            num_samples_key = 'num-samples'.encode()
            self.length = int(txn.get(num_samples_key).decode())
        
        print(f"📊 원본 LMDB 데이터셋 정보:")
        print(f"   - 원본 샘플 수: {self.length}")
        
        # 어휘 사전 로드
        self.voc, self.char2id, self.id2char = get_vocabulary('LOWERCASE')
        self.max_word_len = 32
    
    def __len__(self):
        return self.length
    
    def __del__(self):
        if hasattr(self, 'env'):
            self.env.close()
    
    def get_raw_image_and_gt(self, index):
        """원본 이미지와 GT 데이터를 전처리 없이 가져오기"""
        with self.env.begin(write=False) as txn:
            # 이미지 로드
            img_key = f'image-{index:09d}'.encode()
            img_data = txn.get(img_key)
            if img_data is None:
                raise KeyError(f"이미지 키를 찾을 수 없습니다: {img_key}")
            
            # 바이트 데이터를 이미지로 변환 (원본 그대로)
            img_np = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"이미지 디코딩 실패: index {index}")
            
            # BGR -> RGB 변환
            img = img[:, :, [2, 1, 0]]
            
            # GT 데이터 로드
            gt_key = f'gt-{index:09d}'.encode()
            gt_data = txn.get(gt_key)
            if gt_data is None:
                raise KeyError(f"GT 키를 찾을 수 없습니다: {gt_key}")
            
            # pickle로 직렬화된 GT 데이터 복원
            gt_info = pickle.loads(gt_data)
            
        return img, gt_info
    
    def __getitem__(self, index):
        """원본 데이터 로드 (전처리 없음)"""
        img, gt_info = self.get_raw_image_and_gt(index)
        
        # GT 정보 추출 (원본 그대로)
        bboxes = np.array(gt_info['bboxes'])  # 정규화된 좌표 (원본)
        words = gt_info['words']
        
        if bboxes.shape[0] > self.max_word_num:
            bboxes = bboxes[:self.max_word_num]
            words = words[:self.max_word_num]
        
        # 원본 이미지와 GT 정보만 반환 (전처리 없음)
        data = dict(
            raw_img=img,  # 원본 이미지 (numpy array)
            raw_bboxes=bboxes,  # 원본 bbox 좌표
            raw_words=words,  # 원본 텍스트
            raw_gt_info=gt_info  # 전체 GT 정보
        )
        
        return data


class RawConcatLMDBDataset(data.Dataset):
    """
    여러 원본 LMDB 데이터셋을 결합하는 클래스
    """
    
    def __init__(self, lmdb_paths, split='train', **kwargs):
        """
        Args:
            lmdb_paths (list): LMDB 파일 경로 리스트
            split (str): 'train' 또는 'test'
            **kwargs: RawLMDBDataset 생성자에 전달할 추가 인자들
        """
        self.datasets = []
        self.dataset_lengths = []
        
        for lmdb_path in lmdb_paths:
            print(f"📂 원본 LMDB 로드 중: {lmdb_path}")
            dataset = RawLMDBDataset(
                lmdb_path=lmdb_path,
                split=split,
                **kwargs
            )
            self.datasets.append(dataset)
            self.dataset_lengths.append(len(dataset))
        
        # 누적 길이 계산
        self.cumulative_lengths = np.cumsum([0] + self.dataset_lengths)
        
        print(f"🎯 총 원본 데이터셋 수: {len(self.datasets)}")
        print(f"📊 총 원본 샘플 수: {sum(self.dataset_lengths)}")
        
        # 각 데이터셋 크기 출력
        for i, (dataset, length) in enumerate(zip(self.datasets, self.dataset_lengths)):
            print(f"   데이터셋 {i+1}: {length} 샘플")
    
    def __len__(self):
        return sum(self.dataset_lengths)
    
    def __getitem__(self, index):
        """전체 인덱스를 데이터셋별 인덱스로 변환"""
        # 어떤 데이터셋에 속하는지 찾기
        dataset_idx = np.searchsorted(self.cumulative_lengths, index, side='right') - 1
        local_index = index - self.cumulative_lengths[dataset_idx]
        
        return self.datasets[dataset_idx][local_index] 