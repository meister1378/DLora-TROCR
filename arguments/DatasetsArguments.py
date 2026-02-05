from dataclasses import dataclass, field


@dataclass
class DatasetsArguments:
    train_csv_path: str = field(default=None)
    valid_csv_path: str = field(default=None)
    test_csv_path: str = field(default=None)
    result_csv_path: str = field(default=None)
    submission_csv_path: str = field(default="data/sample_submission.csv")
    # LMDB 학습을 위한 선택 인자들
    lmdb_glob_pattern: str = field(default=None)  # 예: /mnt/nas/ocr_dataset/*_annotations*.lmdb
    lmdb_min_crop_wh: int = field(default=8)
    lmdb_max_words: int = field(default=200)
    lmdb_max_label_len: int = field(default=64)
    lmdb_only_train: bool = field(default=True)  # 파일명에 train 포함만 사용
    lmdb_max_total_samples: int = field(default=None)  # 전체 학습 샘플 상한(없으면 전체)

    # WebDataset 학습을 위한 선택 인자들
    wds_shards: str = field(default=None)  # 예: /run/user/0/gvfs/.../annotations_train-*.tar
    wds_max_total_samples: int = field(default=None)

    # LMDB 학습을 위한 신규 인자들 (FAST 의존성 없이 사용)
    lmdb_paths: str = field(default=None)  # 여러 개면 ':'로 구분. 예: /data/a.lmdb:/data/b.lmdb
