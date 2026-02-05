# Text in the wild LMDB 데이터셋을 사용하는 FAST 모델 설정

model = dict(
    type='FAST',
    backbone=dict(
        type='fast_backbone',
        config='config/fast/nas-configs/fast_base.config'
    ),
    neck=dict(
        type='fast_neck',
        config='config/fast/nas-configs/fast_base.config'
    ),
    detection_head=dict(
        type='fast_head',
        config='config/fast/nas-configs/fast_base.config',
        pooling_size=9,
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

repeat_times = 1  # 테스트를 위해 1로 설정
data = dict(
    batch_size=2,  # 테스트를 위해 작은 배치 크기
    train=dict(
        type='FAST_LMDB',  # LMDB 데이터셋 사용
        lmdb_path='/mnt/nas/ocr_dataset/test_text_in_wild.lmdb',  # 테스트용 LMDB
        split='train',
        is_transform=True,
        img_size=640,
        short_size=640,
        pooling_size=9,
        read_type='cv2',
        repeat_times=repeat_times
    ),
    test=dict(
        type='FAST_LMDB',  # LMDB 데이터셋 사용
        lmdb_path='/mnt/nas/ocr_dataset/test_text_in_wild.lmdb',  # 테스트용 LMDB
        split='test',
        short_size=640,
        read_type='cv2'
    )
)

train_cfg = dict(
    lr=1e-3,
    schedule='polylr',
    epoch=1,  # 테스트를 위해 1 에포크만
    optimizer='Adam',
    pretrain='checkpoint_7ep.pth',
    save_interval=1,
)

test_cfg = dict(
    min_score=0.8,
    min_area=200,
    bbox_type='rect',
    result_path='./work_dirs/text_in_wild_test/results'
)

# 기타 설정
cudnn_benchmark = True
gpu_ids = range(1)
work_dir = './work_dirs/text_in_wild_test'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)] 