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

repeat_times = 10
data = dict(
    batch_size=4,  # 작은 배치 크기로 시작
    train=dict(
        type='Sample',  # 또는 사용자 정의 데이터셋 타입
        split='train',
        is_transform=True,
        img_size=736,  # 원본 설정으로 변경
        short_size=736,  # 원본 설정으로 변경
        pooling_size=9,
        read_type='cv2',
        repeat_times=repeat_times
    ),
    test=dict(
        type='Sample',
        split='test',
        short_size=736,  # 원본 설정으로 변경
        read_type='cv2'
    )
)

train_cfg = dict(
    lr=1e-4,  # 파인튜닝이므로 낮은 학습률
    schedule='polylr',
    epoch=50,  # 파인튜닝은 적은 에포크
    optimizer='Adam',
    pretrain='/home/mango/ocr_test/FAST/checkpoint_7ep.pth',
    save_interval=5,  # 자주 저장
    weight_decay=1e-4,
    momentum=0.9
)

test_cfg = dict(
    min_score=0.88,  # 원본 설정으로 변경
    min_area=250,    # 원본 설정으로 변경
    bbox_type='rect',  # 원본 설정으로 변경
    result_path='outputs/submit_finetune/'
) 