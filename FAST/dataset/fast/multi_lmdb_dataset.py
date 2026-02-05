#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi LMDB Dataset
여러 LMDB 데이터셋을 결합하여 사용하는 클래스
"""

import os
import random
import torch
from torch.utils.data import ConcatDataset, Dataset
from .fast_lmdb import FAST_LMDB


class MultiLMDBDataset(Dataset):
    """
    여러 LMDB 데이터셋을 결합하여 사용하는 클래스
    각 데이터셋에 가중치를 적용할 수 있습니다.
    """
    
    def __init__(self, lmdb_configs, split='train', weights=None, **kwargs):
        """
        Args:
            lmdb_configs (list): LMDB 설정 리스트
                [
                    {'path': './data/dataset1.lmdb', 'weight': 1.0},
                    {'path': './data/dataset2.lmdb', 'weight': 0.5},
                    ...
                ]
            split (str): 'train' 또는 'test'
            weights (list): 각 데이터셋의 가중치 (deprecated, lmdb_configs에서 weight 사용)
            **kwargs: FAST_LMDB 생성자에 전달할 추가 인자들
        """
        self.lmdb_configs = lmdb_configs
        self.split = split
        self.kwargs = kwargs
        
        # 개별 데이터셋들 생성
        self.datasets = []
        self.dataset_weights = []
        
        for config in lmdb_configs:
            lmdb_path = config['path']
            weight = config.get('weight', 1.0)
            
            print(f"📂 LMDB 로드 중: {lmdb_path} (가중치: {weight})")
            
            dataset = FAST_LMDB(
                lmdb_path=lmdb_path,
                split=split,
                **kwargs
            )
            
            self.datasets.append(dataset)
            self.dataset_weights.append(weight)
        
        # 가중치에 따른 샘플 인덱스 생성
        self._create_weighted_indices()
        
        print(f"🎯 총 데이터셋 수: {len(self.datasets)}")
        print(f"📊 총 샘플 수: {len(self.weighted_indices)}")
    
    def _create_weighted_indices(self):
        """가중치에 따른 샘플 인덱스 생성"""
        self.weighted_indices = []
        
        for dataset_idx, (dataset, weight) in enumerate(zip(self.datasets, self.dataset_weights)):
            # 각 데이터셋의 실제 샘플 수
            dataset_size = len(dataset)
            
            # 가중치에 따른 샘플 수 계산
            weighted_size = int(dataset_size * weight)
            
            # 인덱스 생성 (반복 샘플링 허용)
            if weight >= 1.0:
                # 가중치가 1 이상이면 반복 샘플링
                indices = list(range(dataset_size)) * int(weight)
                remaining = weighted_size - len(indices)
                if remaining > 0:
                    indices.extend(random.choices(range(dataset_size), k=remaining))
            else:
                # 가중치가 1 미만이면 부분 샘플링
                indices = random.sample(range(dataset_size), weighted_size)
            
            # (데이터셋 인덱스, 샘플 인덱스) 튜플로 저장
            for sample_idx in indices:
                self.weighted_indices.append((dataset_idx, sample_idx))
        
        # 인덱스 섞기
        random.shuffle(self.weighted_indices)
        
        print(f"📈 가중치 적용 결과:")
        for i, (dataset, weight) in enumerate(zip(self.datasets, self.dataset_weights)):
            actual_count = sum(1 for idx in self.weighted_indices if idx[0] == i)
            print(f"   데이터셋 {i+1}: {len(dataset)} -> {actual_count} 샘플 (가중치: {weight})")
    
    def __len__(self):
        return len(self.weighted_indices)
    
    def __getitem__(self, index):
        """샘플 로드"""
        dataset_idx, sample_idx = self.weighted_indices[index]
        return self.datasets[dataset_idx][sample_idx]
    
    def resample_indices(self):
        """에포크마다 인덱스 재샘플링"""
        self._create_weighted_indices()


class ConcatLMDBDataset(ConcatDataset):
    """
    PyTorch ConcatDataset을 사용한 간단한 LMDB 결합
    """
    
    def __init__(self, lmdb_paths, split='train', **kwargs):
        """
        Args:
            lmdb_paths (list): LMDB 파일 경로 리스트
            split (str): 'train' 또는 'test'
            **kwargs: FAST_LMDB 생성자에 전달할 추가 인자들
        """
        datasets = []
        
        for lmdb_path in lmdb_paths:
            print(f"📂 LMDB 로드 중: {lmdb_path}")
            dataset = FAST_LMDB(
                lmdb_path=lmdb_path,
                split=split,
                **kwargs
            )
            datasets.append(dataset)
        
        super().__init__(datasets)
        
        print(f"🎯 총 데이터셋 수: {len(datasets)}")
        print(f"📊 총 샘플 수: {len(self)}")
        
        # 각 데이터셋 크기 출력
        for i, dataset in enumerate(datasets):
            print(f"   데이터셋 {i+1}: {len(dataset)} 샘플")


# 편의 함수들
def create_multi_lmdb_dataset(config_type='weighted', **kwargs):
    """
    설정에 따라 적절한 Multi LMDB 데이터셋 생성
    
    Args:
        config_type (str): 'weighted' 또는 'concat'
        **kwargs: 데이터셋 생성자에 전달할 인자들
    """
    if config_type == 'weighted':
        return MultiLMDBDataset(**kwargs)
    elif config_type == 'concat':
        return ConcatLMDBDataset(**kwargs)
    else:
        raise ValueError(f"지원하지 않는 config_type: {config_type}")


# 사용 예시
if __name__ == '__main__':
    # 가중치 적용 예시
    lmdb_configs = [
        {'path': './data/text_in_wild.lmdb', 'weight': 1.0},
        {'path': './data/ocr_public_train.lmdb', 'weight': 0.8},
        {'path': './data/handwriting_ts5_paper_form.lmdb', 'weight': 0.5},
    ]
    
    dataset = MultiLMDBDataset(
        lmdb_configs=lmdb_configs,
        split='train',
        is_transform=True,
        short_size=640
    )
    
    print(f"결합된 데이터셋 크기: {len(dataset)}")
    
    # 첫 번째 샘플 테스트
    sample = dataset[0]
    print(f"샘플 키: {list(sample.keys())}") 