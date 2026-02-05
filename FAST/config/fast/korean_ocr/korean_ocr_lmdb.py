#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
한국어 OCR 데이터셋 LMDB 설정 파일
여러 한국어 OCR 데이터셋을 LMDB 형태로 사용하기 위한 설정
"""

# 모델 설정
model = dict(
    type='FAST',
    backbone=dict(
        type='resnet50',
        pretrained=True
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4
    ),
    bbox_head=dict(
        type='FASTHead',
        in_channels=256,
        hidden_dim=256,
        num_classes=2,
        loss_text=dict(
            type='DiceLoss',
            loss_weight=0.5
        ),
        loss_kernel=dict(
            type='DiceLoss', 
            loss_weight=1.0
        ),
        loss_emb=dict(
            type='EmbLoss',
            feature_dim=4,
            loss_weight=0.25
        )
    )
)

# 데이터셋 설정
data = dict(
    batch_size=8,
    train=dict(
        type='FAST_LMDB',
        lmdb_path='./data/korean_ocr_combined_train.lmdb',  # 여러 데이터셋 결합
        split='train',
        is_transform=True,
        img_size=640,
        short_size=640,
        repeat_times=1,
        with_rec=False,
        read_type='cv2'
    ),
    test=dict(
        type='FAST_LMDB',
        lmdb_path='./data/korean_ocr_combined_val.lmdb',
        split='test',
        short_size=640,
        with_rec=False,
        read_type='cv2'
    )
)

# 개별 데이터셋 설정 예시들
data_configs = {
    # Text in the wild 데이터셋
    'text_in_wild': dict(
        train=dict(
            type='FAST_LMDB',
            lmdb_path='./data/text_in_wild.lmdb',
            split='train',
            is_transform=True,
            img_size=640,
            short_size=640,
            repeat_times=1
        )
    ),
    
    # 공공 데이터셋
    'ocr_public': dict(
        train=dict(
            type='FAST_LMDB',
            lmdb_path='./data/ocr_public_train.lmdb',
            split='train',
            is_transform=True,
            img_size=640,
            short_size=640,
            repeat_times=1
        ),
        val=dict(
            type='FAST_LMDB',
            lmdb_path='./data/ocr_public_val.lmdb',
            split='test',
            short_size=640
        )
    ),
    
    # 금융 및 물류 데이터셋
    'finance_logistics': dict(
        train=dict(
            type='FAST_LMDB',
            lmdb_path='./data/finance_logistics_train.lmdb',
            split='train',
            is_transform=True,
            img_size=640,
            short_size=640,
            repeat_times=1
        ),
        val=dict(
            type='FAST_LMDB',
            lmdb_path='./data/finance_logistics_val.lmdb',
            split='test',
            short_size=640
        )
    ),
    
    # 손글씨 데이터셋
    'handwriting': dict(
        train=dict(
            type='FAST_LMDB',
            lmdb_path='./data/handwriting_ts5_paper_form.lmdb',
            split='train',
            is_transform=True,
            img_size=640,
            short_size=640,
            repeat_times=1
        )
    ),
    
    # 공공행정문서 데이터셋
    'public_admin': dict(
        train1=dict(
            type='FAST_LMDB',
            lmdb_path='./data/public_admin_train1.lmdb',
            split='train',
            is_transform=True,
            img_size=640,
            short_size=640,
            repeat_times=1
        ),
        train2=dict(
            type='FAST_LMDB',
            lmdb_path='./data/public_admin_train2.lmdb',
            split='train',
            is_transform=True,
            img_size=640,
            short_size=640,
            repeat_times=1
        ),
        train3=dict(
            type='FAST_LMDB',
            lmdb_path='./data/public_admin_train3.lmdb',
            split='train',
            is_transform=True,
            img_size=640,
            short_size=640,
            repeat_times=1
        ),
        val=dict(
            type='FAST_LMDB',
            lmdb_path='./data/public_admin_val.lmdb',
            split='test',
            short_size=640
        )
    )
}

# 최적화 설정
optimizer = dict(
    type='Adam',
    lr=1e-3,
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
total_epochs = 600
checkpoint_config = dict(interval=50)
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
work_dir = './work_dirs/korean_ocr_lmdb'
load_from = None
resume_from = None
workflow = [('train', 1)]

# GPU 설정
gpu_ids = range(1)  # 사용할 GPU 개수에 따라 조정 