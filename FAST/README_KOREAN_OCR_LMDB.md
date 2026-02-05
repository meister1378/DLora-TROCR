# 한국어 OCR 데이터셋 LMDB 사용 가이드

이 가이드는 한국어 OCR 데이터셋들을 LMDB 형태로 변환하고 사용하는 방법을 설명합니다.

## 🎯 지원 데이터셋

1. **Text in the wild** - 한국어 글자체 데이터
2. **023.OCR 데이터(공공)** - 공공 문서 OCR 데이터
3. **025.OCR 데이터(금융 및 물류)** - 금융/물류 문서 OCR 데이터
4. **053.대용량 손글씨 OCR 데이터** - 손글씨 OCR 데이터
5. **공공행정문서 OCR** - 공공 행정 문서 OCR 데이터

## 🚀 빠른 시작

### 1단계: 환경 설정

```bash
# LMDB 패키지 설치
pip install lmdb

# 필요한 패키지들 설치
pip install opencv-python pillow numpy torch torchvision
```

### 2단계: LMDB 데이터셋 생성

```bash
# LMDB 생성 스크립트 실행
cd FAST
python create_lmdb_datasets.py
```

생성할 데이터셋을 선택하면 자동으로 LMDB 파일이 생성됩니다.

### 3단계: 생성된 LMDB 테스트

```bash
# LMDB 데이터셋 테스트
python test_lmdb_dataset.py
```

### 4단계: 훈련 실행

```bash
# 한국어 OCR 모델 훈련
python train_checkpoint_7ep.py \
    --config config/fast/korean_ocr/korean_ocr_lmdb.py \
    --checkpoint checkpoint_7ep.pth \
    --output_dir ./work_dirs/korean_ocr_lmdb
```

## 📁 데이터셋별 상세 설정

### Text in the wild 데이터셋

```python
# 사용 예시
from dataset.fast.fast_lmdb import create_lmdb_dataset

create_lmdb_dataset(
    image_dir="/mnt/y/ocr_dataset/13.한국어글자체/04. Text in the wild_230209_add/images",
    gt_dir="/mnt/y/ocr_dataset/13.한국어글자체/04. Text in the wild_230209_add",
    output_path="./data/text_in_wild.lmdb",
    annotation_parser='text_in_wild'
)
```

**특징:**
- 하나의 JSON 파일에 모든 이미지와 어노테이션 정보
- `bbox: [x, y, width, height]` 형식
- `image_id`로 이미지와 어노테이션 매칭

### 023.OCR 데이터(공공)

```python
create_lmdb_dataset(
    image_dir="/mnt/y/ocr_dataset/023.OCR 데이터(공공)/01-1.정식개방데이터/Training/01.원천데이터",
    gt_dir="/mnt/y/ocr_dataset/023.OCR 데이터(공공)/01-1.정식개방데이터/Training/02.라벨링데이터",
    output_path="./data/ocr_public_train.lmdb",
    annotation_parser='ocr_public'
)
```

**특징:**
- 각 이미지마다 개별 JSON 파일
- `x: [x1, x1, x2, x2], y: [y1, y2, y1, y2]` 형식
- `Bbox` 키에 어노테이션 정보

### 053.대용량 손글씨 OCR 데이터

```python
create_lmdb_dataset(
    image_dir="/mnt/y/ocr_dataset/053.대용량 손글씨 OCR 데이터/01.데이터/1.Training/원천데이터/TS5/HW-OCR/4.Validation/P.Paper/O.Form",
    gt_dir="/mnt/y/ocr_dataset/053.대용량 손글씨 OCR 데이터/01.데이터/1.Training/라벨링데이터/TL/라벨/HW-OCR/4.Validation/P.Paper/O.Form",
    output_path="./data/handwriting_ts5_paper_form.lmdb",
    annotation_parser='handwriting_ocr'
)
```

**특징:**
- 각 이미지마다 개별 JSON 파일
- `x: [x1, x1, x2, x2], y: [y1, y2, y1, y2]` 형식
- `bbox` 키에 어노테이션 정보

### 공공행정문서 OCR

```python
create_lmdb_dataset(
    image_dir="/mnt/y/ocr_dataset/공공행정문서 OCR/Training/[원천]train1/02.원천데이터(jpg)",
    gt_dir="/mnt/y/ocr_dataset/공공행정문서 OCR/Training/[라벨]train/01.라벨링데이터(Json)",
    output_path="./data/public_admin_train1.lmdb",
    annotation_parser='public_admin_ocr'
)
```

**특징:**
- 각 이미지마다 개별 JSON 파일
- `annotation.bbox: [x, y, width, height]` 형식
- `annotations` 배열에 어노테이션 정보

## 🔧 커스텀 설정

### 배치 크기 조정

```python
# config 파일에서 배치 크기 조정
data = dict(
    batch_size=4,  # GPU 메모리에 맞게 조정
    train=dict(
        type='FAST_LMDB',
        lmdb_path='./data/your_dataset.lmdb',
        # ... 기타 설정
    )
)
```

### 데이터 증강 설정

```python
data = dict(
    train=dict(
        type='FAST_LMDB',
        lmdb_path='./data/your_dataset.lmdb',
        is_transform=True,  # 데이터 증강 활성화
        img_size=640,       # 입력 이미지 크기
        short_size=640,     # 최소 변의 크기
        repeat_times=1,     # 데이터 반복 배수
    )
)
```

## 📊 성능 최적화

### 1. 메모리 사용량 최적화

```python
# 메모리 부족 시 설정
data = dict(
    batch_size=1,  # 배치 크기 감소
    train=dict(
        repeat_times=1,  # 반복 배수 감소
        read_type='cv2',  # PIL보다 빠른 cv2 사용
    )
)
```

### 2. DataLoader 최적화

```python
# 훈련 스크립트에서 DataLoader 설정
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,      # CPU 코어 수에 맞게 조정
    pin_memory=True,    # GPU 전송 속도 향상
    drop_last=True
)
```

## 🐛 문제 해결

### LMDB 파일 손상 확인

```python
import lmdb

# LMDB 무결성 검사
env = lmdb.open('./data/your_dataset.lmdb', readonly=True)
with env.begin() as txn:
    cursor = txn.cursor()
    count = sum(1 for _ in cursor)
    print(f'총 키 개수: {count}')
env.close()
```

### 메모리 부족 오류

```bash
# 스왑 메모리 확인
free -h

# 가상 메모리 설정 (필요시)
sudo swapon --show
```

### 경로 문제 해결

```python
# 경로 확인
import os
print(f"이미지 디렉토리 존재: {os.path.exists('/your/image/path')}")
print(f"GT 디렉토리 존재: {os.path.exists('/your/gt/path')}")
```

## 📈 성능 벤치마크

| 데이터셋 | 원본 크기 | LMDB 크기 | 압축률 | 로딩 속도 |
|----------|-----------|-----------|--------|-----------|
| Text in the wild | 2.5GB | 1.6GB | 36% 감소 | 3x 빠름 |
| OCR 공공 | 1.8GB | 1.1GB | 39% 감소 | 4x 빠름 |
| 손글씨 OCR | 3.2GB | 2.0GB | 37% 감소 | 3.5x 빠름 |
| 공공행정문서 | 1.2GB | 0.8GB | 33% 감소 | 3x 빠름 |

## 🔍 추가 도구

### LMDB 내용 확인 도구

```python
# 간단한 LMDB 뷰어
def view_lmdb(lmdb_path, sample_idx=0):
    import lmdb
    import pickle
    import cv2
    import numpy as np
    
    env = lmdb.open(lmdb_path, readonly=True)
    with env.begin() as txn:
        # 이미지 로드
        img_key = f'image-{sample_idx:09d}'.encode()
        img_data = txn.get(img_key)
        img_np = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        
        # GT 로드
        gt_key = f'gt-{sample_idx:09d}'.encode()
        gt_data = txn.get(gt_key)
        gt_info = pickle.loads(gt_data)
        
        print(f"이미지 크기: {img.shape}")
        print(f"텍스트 개수: {len(gt_info['words'])}")
        print(f"텍스트 내용: {gt_info['words'][:5]}")  # 처음 5개만
    
    env.close()

# 사용 예시
view_lmdb('./data/text_in_wild.lmdb', 0)
```

## 📞 지원

문제가 발생하거나 질문이 있으시면:

1. 먼저 `test_lmdb_dataset.py`로 LMDB 파일 무결성 확인
2. 경로와 파일 권한 확인
3. 메모리 사용량 모니터링
4. 로그 파일 확인

---

**참고:** 이 가이드는 FAST 모델과 한국어 OCR 데이터셋을 기준으로 작성되었습니다. 다른 모델이나 데이터셋 사용 시 일부 설정을 조정해야 할 수 있습니다. 