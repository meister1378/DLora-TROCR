#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi LMDB 설정 파일
여러 한국어 OCR LMDB 데이터셋을 동시에 사용하는 설정
"""

# 모델 설정 (checkpoint_7ep.pth 호환)
model = dict(
    type='FAST',
    backbone=dict(
        type='fast_backbone',
        config='/home/mango/ocr_test/FAST/config/fast/nas-configs/fast_base.config'
    ),
    neck=dict(
        type='fast_neck',
        config='/home/mango/ocr_test/FAST/config/fast/nas-configs/fast_base.config'
    ),
    detection_head=dict(
        type='fast_head',
        config='/home/mango/ocr_test/FAST/config/fast/nas-configs/fast_base.config',
        pooling_size=15,
        dropout_ratio=0.1,
        loss_text=dict(
            type='DiceLoss',
            loss_weight=0.5
        ),
        loss_kernel=dict(
            type='DiceLoss',
            loss_weight=1.0
        ),
        loss_emb=dict(
            type='EmbLoss_v1',
            feature_dim=4,
            loss_weight=0.25
        )
    )
)

# 방법 1: 가중치 적용 Multi LMDB 설정
data = dict(
    batch_size=6,
    train=dict(
        type='MultiLMDBDataset',
        lmdb_configs=[
            {'path': './data/text_in_wild.lmdb', 'weight': 1.0},           # 100% 사용
            {'path': './data/ocr_public_train.lmdb', 'weight': 0.8},       # 80% 사용
            {'path': './data/finance_logistics_train.lmdb', 'weight': 0.6}, # 60% 사용
            {'path': './data/handwriting_ts5_paper_form.lmdb', 'weight': 0.4}, # 40% 사용
            {'path': './data/public_admin_train1.lmdb', 'weight': 0.5},    # 50% 사용
        ],
        split='train',
        is_transform=True,
        img_size=640,
        short_size=640,
        with_rec=False,
        read_type='cv2'
    ),
    test=dict(
        type='ConcatLMDBDataset',
        lmdb_paths=[
            './data/ocr_public_val.lmdb',
            './data/finance_logistics_val.lmdb',
            './data/public_admin_val.lmdb'
        ],
        split='test',
        short_size=640,
        with_rec=False,
        read_type='cv2'
    )
)

# 방법 2: 단순 결합 설정 (모든 데이터 동일 비율)
data_simple_concat = dict(
    batch_size=6,
    train=dict(
        type='ConcatLMDBDataset',
        lmdb_paths=[
            './data/text_in_wild.lmdb',
            './data/ocr_public_train.lmdb',
            './data/finance_logistics_train.lmdb',
            './data/handwriting_ts5_paper_form.lmdb',
            './data/public_admin_train1.lmdb',
        ],
        split='train',
        is_transform=True,
        img_size=640,
        short_size=640,
        with_rec=False,
        read_type='cv2'
    ),
    test=dict(
        type='ConcatLMDBDataset',
        lmdb_paths=[
            './data/ocr_public_val.lmdb',
            './data/finance_logistics_val.lmdb',
            './data/public_admin_val.lmdb'
        ],
        split='test',
        short_size=640
    )
)

# 방법 3: 특정 데이터셋만 선택적으로 사용
data_selective = dict(
    batch_size=8,
    train=dict(
        type='MultiLMDBDataset',
        lmdb_configs=[
            # 텍스트 인식에 좋은 데이터셋들만 선택
            {'path': './data/text_in_wild.lmdb', 'weight': 1.5},        # 150% 사용 (중요)
            {'path': './data/ocr_public_train.lmdb', 'weight': 1.0},    # 100% 사용
            {'path': './data/handwriting_ts5_paper_form.lmdb', 'weight': 0.3}, # 30% 사용 (보조)
        ],
        split='train',
        is_transform=True,
        img_size=640,
        short_size=640
    ),
    test=dict(
        type='FAST_LMDB',  # 단일 LMDB 사용
        lmdb_path='./data/ocr_public_val.lmdb',
        split='test',
        short_size=640
    )
)

# 최적화 설정
optimizer = dict(
    type='Adam',
    lr=8e-4,  # 여러 데이터셋 사용 시 조금 낮은 학습률
    weight_decay=5e-4
)

# 학습률 스케줄러
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-7,
    by_epoch=False
)

# 훈련 설정
total_epochs = 800  # 여러 데이터셋 사용 시 더 많은 에포크
checkpoint_config = dict(interval=50)

# 로그 설정
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)

# 평가 설정
evaluation = dict(
    interval=50,
    metric='hmean'
)

# 기타 설정
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/multi_lmdb_korean_ocr'
load_from = None
resume_from = None
workflow = [('train', 1)]

# GPU 설정
gpu_ids = range(1)

# 커스텀 훅 설정 (에포크마다 데이터셋 재샘플링)
custom_hooks = [
    dict(
        type='ResampleHook',
        priority='NORMAL',
        interval=1  # 매 에포크마다 재샘플링
    )
]

# 데이터셋 조합 전략별 설명
dataset_strategies = {
    'balanced': {
        'description': '모든 데이터셋을 균등하게 사용',
        'use_case': '데이터 다양성 최대화',
        'config': 'data_simple_concat'
    },
    'weighted': {
        'description': '데이터셋별 가중치 적용',
        'use_case': '특정 데이터셋 강조, 품질 조절',
        'config': 'data'
    },
    'selective': {
        'description': '핵심 데이터셋만 선택적 사용',
        'use_case': '특정 도메인 집중, 빠른 수렴',
        'config': 'data_selective'
    }
}

# 실행 시 전략 선택 가이드
print("=" * 60)
print("📊 Multi LMDB 설정 전략:")
for strategy, info in dataset_strategies.items():
    print(f"🔹 {strategy}: {info['description']}")
    print(f"   사용 사례: {info['use_case']}")
    print(f"   설정: {info['config']}")
    print()
print("=" * 60) 