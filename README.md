# OCR 파이프라인 실행 가이드 (데이터 생성 → FAST 학습 → TrOCR(DLoRA) 학습/추론)

이 문서는 **“원본 라벨(JSON) → merged_json → lookup → (FAST용) layout LMDB / (TrOCR용) annotations LMDB·JSONL → 학습/추론”**을 한 번에 따라칠 수 있게 정리한 실행 가이드입니다.

### 환경(참고)
- **OS**: Ubuntu 20.04
- **Python**: 3.9.21
- **CUDA**: 11.8
- **cuDNN**: 8.6.0

---

## 0) 공통 (권장 실행 위치)

FAST 관련 스크립트는 내부에서 `FAST/...` 같은 **상대경로를 많이 사용**합니다.  
따라서 아래처럼 **FAST 폴더에서 실행**하는 것을 권장합니다.

```bash
cd /home/mango/ocr_test
conda activate myenv
cd FAST
```

이후 가이드의 `python ...`은 특별히 적지 않으면 **`/home/mango/ocr_test/FAST`에서 실행**한다고 가정합니다.

---

## 1) (선행) gvfs FTP 마운트 확인

원본 데이터는 기본적으로 gvfs FTP 마운트 경로를 읽습니다.

- 기본 경로: `/run/user/0/gvfs/ftp:host=172.30.1.226/Y:\\ocr_dataset`

```bash
ls -lah "/run/user/0/gvfs/ftp:host=172.30.1.226/Y:\\ocr_dataset" | head
```

---

## 2) `merged_json` 만들기 (원본 라벨 JSON → *_merged.json)

`FAST/json_merged/*.json`을 입력으로 쓰는 파이프라인이므로 먼저 준비합니다.

### 2-1) 병합 스크립트 실행

- **병합 스크립트**: `merge_json_datasets.py`
- **출력**
  - 기본 출력: `/mnt/nas/ocr_dataset/json_data/*_merged.json`
  - 백업 출력: `/home/mango/ocr_test/FAST/json_merged/*_merged.json` (있으면 “완료됨”으로 인식)

```bash
python merge_json_datasets.py
```

### 2-2) Text in the wild 예외

`textinthewild_data_info.json`은 병합 스크립트가 만들지 않습니다. 원본 제공 파일을 `json_merged`로 복사해 둡니다.

```bash
cp "/run/user/0/gvfs/ftp:host=172.30.1.226/Y:\\ocr_dataset/13.한국어글자체/04. Text in the wild_230209_add/textinthewild_data_info.json" \
   "/home/mango/ocr_test/FAST/json_merged/textinthewild_data_info.json"
```

### 2-3) `json_merged` 준비 확인

```bash
ls -lah /home/mango/ocr_test/FAST/json_merged
```

필수 파일(예):
- `textinthewild_data_info.json`
- `public_admin_train_merged.json`, `public_admin_valid_merged.json`, `public_admin_train_partly_merged.json`
- `ocr_public_train_merged.json`, `ocr_public_valid_merged.json`
- `finance_logistics_train_merged.json`, `finance_logistics_valid_merged.json`
- `handwriting_train_merged.json`, `handwriting_valid_merged.json`

---

## 3) lookup 만들기 (파일명 → 원본 이미지 절대경로)

`create_all_datasets_annotations_jsonl.py` / `create_all_datasets_layout.py`는 원본 이미지를 찾기 위해 lookup을 사용합니다.

### 3-1) lookup 생성(optimized_lookup_*.py)

- 생성기: `ftp_tree_viewer.py`
- 출력: `FAST/optimized_lookup_*.py`, `FAST/unified_ocr_lookup_optimizer.py`

```bash
python ftp_tree_viewer.py
```

### 3-2) lookup 최적화(pkl.gz 변환)

- 변환기: `convert_lookup_to_pickle.py`
- 출력: `FAST/lookup_<dataset>.pkl.gz` (권장 포맷)

```bash
python convert_lookup_to_pickle.py
```

### 3-3) 생성 확인

아래 폴더에 `lookup_*.pkl.gz`가 있어야 합니다(현재 프로젝트 구조상 lookup은 `FAST/FAST/` 아래에 존재).

```bash
ls -lah FAST | head
```

---

## 4) TrOCR 학습용 annotations 생성 (크롭 + JSONL 또는 LMDB)

`create_all_datasets_annotations_jsonl.py`는 환경변수 `FAST_OUTPUT_FORMAT`으로 출력 형식을 고릅니다.
- `FAST_OUTPUT_FORMAT=jsonl` → `*.jsonl` 생성 (권장)
- `FAST_OUTPUT_FORMAT=lmdb` → `*_annotations_*.lmdb` 생성

기본 출력 경로는 `/mnt/nas/ocr_dataset` 입니다.

```bash
export FAST_OUTPUT_FORMAT=jsonl
python create_all_datasets_annotations_jsonl.py
```

생성 결과(예):
- `/mnt/nas/ocr_dataset/*_annotations_train.jsonl`
- `/mnt/nas/ocr_dataset/*_annotations_valid.jsonl`
- `/mnt/nas/ocr_dataset/crops/<dataset>/<split>/...` (인식용 크롭 이미지)

---

## 5) FAST 학습용 layout LMDB 생성 (검출 학습 데이터)

`create_all_datasets_layout.py`는 레이아웃/테이블 관련 옵션을 환경변수로 받습니다.

```bash
# (선택) 디버그 모드: 샘플 제한(기본 500) + 로그 증가
export FAST_DEBUG=0

# (선택) 레이아웃/테이블 장치 및 임계값
export FAST_LAYOUT_DEVICE=gpu   # gpu|cpu
export FAST_TABLE_DEVICE=gpu    # gpu|cpu
export FAST_LAYOUT_MODEL=PP-DocLayoutV2
export FAST_TABLE_LAYOUT_THR=0.3

python create_all_datasets_layout.py
```

생성 결과(예):
- `/mnt/nas/ocr_dataset/text_in_wild_train_layout.lmdb`
- `/mnt/nas/ocr_dataset/public_admin_train_layout.lmdb`
- `/mnt/nas/ocr_dataset/handwriting_valid_layout.lmdb`

---

## 6) FAST 학습 (LMDB 기반)

주의: `train_fast_from_lmdb.py`는 **`--deepspeed`, `--steps_per_epoch` 인자를 받지 않습니다.**

```bash
python train_fast_from_lmdb.py \
  --config config/fast/ic15/fast_sample_finetune.py \
  --lmdb_base /mnt/nas/ocr_dataset \
  --output_dir /home/mango/ocr_test/outputs/fast_lmdb_train \
  --checkpoint /home/mango/ocr_test/FAST/checkpoints/1024_finetune/checkpoint_7ep.pth \
  --batch_size 8 \
  --ga_steps 2 \
  --num_workers 8 \
  --repeat_times 1 \
  --channels_last \
  --prefetch_factor 8 \
  --val_prefetch_factor 2 \
  --pin_memory_train \
  --pin_memory_val \
  --amp \
  --compile \
  --compile_mode max-autotune \
  --save_every_steps 100 \
  --val_every_steps 100 \
  --val_max_samples 100 \
  --val_num_workers 8
```

학습 재개가 필요하면 `--resume /path/to/checkpoint_epoch_*.pth` 를 사용합니다.

---

## 7) FAST 단일 이미지 추론(검출)

```bash
python infer_single_image.py \
  --checkpoint /home/mango/ocr_test/outputs/fast_lmdb_train/checkpoint_epoch_1.pth \
  --image /path/to/image.jpg \
  --output /home/mango/ocr_test/outputs/fast_det.png \
  --config config/fast/korean_ocr/multi_lmdb_config.py \
  --device cuda
```

---

## 8) TrOCR(DLoRA) 학습 (LMDB 기반)

`/home/mango/ocr_test/train_dlora.py`는 `LMDB_PATHS` / `VAL_LMDB_PATHS` 환경변수로 학습/검증 LMDB를 받습니다.

```bash
cd /home/mango/ocr_test

export PYTHONUNBUFFERED=1 \
  TRAIN_DEBUG_SHOW=1 TRAIN_DEBUG_DIAG=0 TRAIN_DEBUG_EVERY=500 PROFILE_TRAIN=1 \
  AMP_DISABLE=0 USE_QLORA=0 \
  VAL_DEBUG_GEN=0 VAL_SHOW_PREDS=1 \
  SAVE_STEPS="10000" WARMUP_RATIO=0.0001 \
  EVAL_EVERY_STEPS="500" EVAL_SAMPLES="1000"

export VAL_LMDB_PATHS="/mnt/nas/ocr_dataset/public_admin_annotations_valid.lmdb:/mnt/nas/ocr_dataset/handwriting_annotations_valid.lmdb"
export LMDB_PATHS="/mnt/nas/ocr_dataset/public_admin_annotations_train.lmdb:/mnt/nas/ocr_dataset/text_in_wild_annotations_train.lmdb:/mnt/nas/ocr_dataset/handwriting_annotations_train.lmdb"

python -u /home/mango/ocr_test/train_dlora.py \
  --output_dir /home/mango/ocr_test/output/dlora_ko_trocr_multi \
  --model_name_or_path ddobokki/ko-trocr \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --gradient_accumulation_steps 4 \
  --dataloader_num_workers 4 \
  --learning_rate 1e-4 \
  --report_to "none" \
  --save_steps 10000
```

---

## 9) TrOCR 디렉터리 일괄 추론 → CSV

```bash
python /home/mango/ocr_test/infer_trocr_dir.py \
  --images_dir /home/mango/ocr_test/debug_samples \
  --output_csv /home/mango/ocr_test/debug_infer.csv \
  --ckpt /home/mango/ocr_test/output/dlora_ko_trocr_multi/checkpoint-60000 \
  --base ddobokki/ko-trocr \
  --device cuda
```

