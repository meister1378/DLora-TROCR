# checkpoint_7ep.pth 파인튜닝 가이드

이 가이드는 기존 `checkpoint_7ep.pth` 파일을 사용하여 FAST 모델을 파인튜닝하는 방법을 설명합니다.

## 📁 파일 구조

```
FAST/
├── checkpoint_7ep.pth                          # 사전 학습된 모델
├── train_checkpoint_7ep.py                     # 훈련 스크립트
├── config/fast/ic15/fast_checkpoint_7ep_finetune.py  # 설정 파일
├── run_checkpoint_7ep_training.sh              # 실행 스크립트
└── CHECKPOINT_7EP_TRAINING_README.md           # 이 파일
```

## 🚀 빠른 시작

### 1. 실행 권한 부여
```bash
chmod +x run_checkpoint_7ep_training.sh
```

### 2. 훈련 실행
```bash
./run_checkpoint_7ep_training.sh
```

### 3. 또는 직접 실행
```bash
python train_checkpoint_7ep.py \
    --config config/fast/ic15/fast_checkpoint_7ep_finetune.py \
    --checkpoint checkpoint_7ep.pth \
    --output_dir ./finetune_checkpoints \
    --epochs 50
```

## ⚙️ 설정 사용자화

### 훈련 파라미터 수정 (`fast_checkpoint_7ep_finetune.py`)

```python
train_cfg = dict(
    lr=1e-4,           # 학습률 (파인튜닝이므로 낮게)
    schedule='polylr',  # 스케줄러 타입
    epoch=50,          # 총 에포크 수
    optimizer='Adam',  # 옵티마이저
    save_interval=5,   # 체크포인트 저장 간격
)

data = dict(
    batch_size=4,      # 배치 크기
    train=dict(
        img_size=640,  # 입력 이미지 크기
        # ... 데이터셋 설정
    )
)
```

### 손실 함수 가중치 조정

```python
detection_head=dict(
    loss_text=dict(type='DiceLoss', loss_weight=0.5),    # 텍스트 검출
    loss_kernel=dict(type='DiceLoss', loss_weight=1.0),  # 커널 검출
    loss_emb=dict(type='EmbLoss_v1', loss_weight=0.25)   # 임베딩
)
```

## 📊 훈련 모니터링

### 훈련 로그 예시
```
🚀 FAST 훈련 초기화 완료
   - 설정 파일: config/fast/ic15/fast_checkpoint_7ep_finetune.py
   - 체크포인트: checkpoint_7ep.pth
   - 출력 디렉토리: ./finetune_checkpoints

📦 체크포인트 로드: checkpoint_7ep.pth
   - EMA 상태 딕셔너리 사용
   - 시작 에포크: 4
✅ 체크포인트 로드 완료

📚 에포크 5 훈련 시작...
   배치 [10/100] Loss: 0.8245 (Avg: 0.8156) Text: 0.3241 Kernel: 0.4123 Emb: 0.0881
   배치 [20/100] Loss: 0.7834 (Avg: 0.8001) Text: 0.3156 Kernel: 0.3987 Emb: 0.0691

📊 에포크 5 완료 (120.5초)
   - 평균 총 손실: 0.7823
   - 평균 텍스트 손실: 0.3089
   - 평균 커널 손실: 0.3945
   - 평균 임베딩 손실: 0.0789
   - 학습률: 0.000095

💾 체크포인트 저장: ./finetune_checkpoints/checkpoint_epoch_5.pth
```

## 📂 결과 파일

훈련 완료 후 다음 파일들이 생성됩니다:

```
finetune_checkpoints/
├── checkpoint_epoch_5.pth      # 5에포크 체크포인트
├── checkpoint_epoch_10.pth     # 10에포크 체크포인트
├── ...
├── checkpoint_latest.pth       # 가장 최근 체크포인트
└── checkpoint_best.pth         # 최고 성능 체크포인트
```

## 🔧 고급 설정

### 1. 멀티 GPU 훈련

```bash
# GPU 여러 개 사용 (예: 0,1,2,3번 GPU)
export CUDA_VISIBLE_DEVICES=0,1,2,3
python train_checkpoint_7ep.py --config config.py --checkpoint checkpoint_7ep.pth
```

### 2. 학습률 스케줄러 변경

```python
# 설정 파일에서
train_cfg = dict(
    schedule='step',     # 또는 'cosine', 'poly' 등
    step_size=20,       # StepLR의 경우
    gamma=0.1           # 학습률 감소 비율
)
```

### 3. 데이터 증강 설정

```python
data = dict(
    train=dict(
        is_transform=True,      # 데이터 증강 활성화
        img_size=640,          # 훈련 이미지 크기
        short_size=640,        # 최소 이미지 크기
        # 추가 증강 옵션들...
    )
)
```

## 🐛 문제 해결

### 1. CUDA 메모리 부족
```python
# 배치 크기 줄이기
data = dict(batch_size=2)  # 4 → 2로 변경
```

### 2. 체크포인트 로드 오류
```bash
# 체크포인트 파일 경로 확인
ls -la checkpoint_7ep.pth

# 파일 권한 확인
chmod 644 checkpoint_7ep.pth
```

### 3. 설정 파일 오류
```bash
# 설정 파일 문법 검사
python -c "from mmcv import Config; cfg = Config.fromfile('config/fast/ic15/fast_checkpoint_7ep_finetune.py')"
```

## 📈 성능 평가

### 훈련된 모델 테스트
```bash
python test.py \
    config/fast/ic15/fast_checkpoint_7ep_finetune.py \
    finetune_checkpoints/checkpoint_best.pth \
    --eval
```

### 추론 실행
```bash
python inference_single.py \
    --config config/fast/ic15/fast_checkpoint_7ep_finetune.py \
    --checkpoint finetune_checkpoints/checkpoint_best.pth \
    --image your_image.jpg
```

## 💡 팁

1. **파인튜닝 시작**: 낮은 학습률(1e-4)로 시작
2. **조기 종료**: 검증 손실이 증가하면 훈련 중단
3. **체크포인트**: 정기적으로 저장하여 훈련 중단에 대비
4. **모니터링**: 손실 그래프를 통해 수렴 상태 확인
5. **데이터**: 고품질 라벨링된 데이터 사용 권장

## 📞 문의

훈련 과정에서 문제가 발생하면:
1. 로그 메시지 확인
2. GPU 메모리 상태 점검
3. 설정 파일 검토
4. 체크포인트 파일 무결성 확인 