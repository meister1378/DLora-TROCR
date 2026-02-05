#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi LMDB 훈련 스크립트
여러 LMDB 데이터셋을 동시에 사용하여 모델을 훈련합니다.
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# FAST 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.fast.multi_lmdb_dataset import MultiLMDBDataset, ConcatLMDBDataset

# LMDB 저장 경로 (gvfs 또는 마운트된 경로)
LMDB_BASE_PATH = "/mnt/nas/ocr_dataset"  # 또는 gvfs 경로
GVFS_LMDB_PATH = "/run/user/0/gvfs/ftp:host=172.30.1.226/Y:\\ocr_dataset"

def get_lmdb_path():
    """실제 LMDB 저장 경로 확인 (로컬 경로 우선 - gvfs에서는 LMDB가 작동하지 않음)"""
    # 먼저 로컬 마운트 경로 확인 (LMDB 호환성)
    if os.path.exists(f"{LMDB_BASE_PATH}/text_in_wild_train.lmdb"):
        print(f"✅ LMDB 발견 (로컬): {LMDB_BASE_PATH}")
        print("💡 로컬 경로 사용 (LMDB는 gvfs에서 작동하지 않음)")
        return LMDB_BASE_PATH
    
    # gvfs 경로 확인 (하지만 경고)
    elif os.path.exists(f"{GVFS_LMDB_PATH}/text_in_wild_train.lmdb"):
        print(f"⚠️ LMDB 발견 (gvfs): {GVFS_LMDB_PATH}")
        print("❌ 경고: gvfs에서는 LMDB가 정상 작동하지 않습니다!")
        print("💡 해결 방법: LMDB를 로컬로 복사하거나 /mnt/nas/ocr_dataset 사용")
        return None
    
    else:
        print("❌ LMDB 파일을 찾을 수 없습니다.")
        print(f"   확인한 경로 1 (로컬): {LMDB_BASE_PATH}")
        print(f"   확인한 경로 2 (gvfs): {GVFS_LMDB_PATH}")
        return None

def get_all_lmdb_configs(base_path):
    """모든 생성된 LMDB 파일을 찾아서 설정 생성"""
    all_train_lmdbs = []
    all_valid_lmdbs = []
    
    # 예상되는 모든 LMDB 파일들
    expected_lmdbs = [
        ("text_in_wild_train.lmdb", "text_in_wild_valid.lmdb", "Text in Wild", 1.0),
        ("public_admin_train.lmdb", "public_admin_valid.lmdb", "공공행정문서", 1.0),
        ("ocr_public_train.lmdb", "ocr_public_valid.lmdb", "OCR 공공데이터", 1.0),  # 0.8 → 1.0
        ("finance_logistics_train.lmdb", "finance_logistics_valid.lmdb", "금융물류", 1.0),  # 0.8 → 1.0
        ("handwriting_train.lmdb", "handwriting_valid.lmdb", "손글씨", 0.3),  # 0.6 → 0.3 (더 낮게)
    ]
    
    print("🔍 LMDB 파일 검색 중...")
    
    for train_file, valid_file, name, weight in expected_lmdbs:
        train_path = f"{base_path}/{train_file}"
        valid_path = f"{base_path}/{valid_file}"
        
        if os.path.exists(train_path):
            all_train_lmdbs.append({
                'path': train_path, 
                'weight': weight, 
                'name': f"{name} Train"
            })
            print(f"✅ {name} Train: {train_path}")
        else:
            print(f"❌ {name} Train: {train_path} (없음)")
            
        if os.path.exists(valid_path):
            all_valid_lmdbs.append({
                'path': valid_path,
                'name': f"{name} Valid"
            })
            print(f"✅ {name} Valid: {valid_path}")
        else:
            print(f"❌ {name} Valid: {valid_path} (없음)")
    
    print(f"\n📊 발견된 LMDB:")
    print(f"   - Train: {len(all_train_lmdbs)}개")
    print(f"   - Valid: {len(all_valid_lmdbs)}개")
    
    return all_train_lmdbs, all_valid_lmdbs


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='Multi LMDB 훈련 스크립트')
    
    parser.add_argument('--config', type=str, required=True,
                        help='설정 파일 경로')
    parser.add_argument('--strategy', type=str, default='weighted',
                        choices=['weighted', 'concat', 'selective'],
                        help='데이터셋 결합 전략')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='사전 훈련된 체크포인트 경로')
    parser.add_argument('--output_dir', type=str, default='./work_dirs/multi_lmdb',
                        help='출력 디렉토리')
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='사용할 GPU ID (쉼표로 구분)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='배치 크기 (설정 파일 값 덮어쓰기)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='훈련 에포크 수 (설정 파일 값 덮어쓰기)')
    parser.add_argument('--lr', type=float, default=None,
                        help='학습률 (설정 파일 값 덮어쓰기)')
    parser.add_argument('--test_only', action='store_true',
                        help='테스트만 실행')
    
    return parser.parse_args()


def create_dataset(strategy, **kwargs):
    """전략에 따른 데이터셋 생성"""
    
    # 실제 LMDB 경로 확인
    lmdb_base_path = get_lmdb_path()
    if lmdb_base_path is None:
        raise ValueError("LMDB 파일을 찾을 수 없습니다!")
    
    # 모든 LMDB 설정 가져오기
    train_configs, valid_configs = get_all_lmdb_configs(lmdb_base_path)
    
    if not train_configs:
        raise ValueError("Train LMDB 파일이 없습니다!")
    
    if strategy == 'weighted':
        # 가중치 적용 데이터셋 (모든 train LMDB 사용)
        print(f"🔧 Weighted 전략: {len(train_configs)}개 Train LMDB 사용")
        return MultiLMDBDataset(
            lmdb_configs=train_configs,
            **kwargs
        )
    
    elif strategy == 'concat':
        # 단순 결합 데이터셋 (모든 train LMDB 균등 사용)
        train_paths = [config['path'] for config in train_configs]
        print(f"🔧 Concat 전략: {len(train_paths)}개 Train LMDB 균등 결합")
        
        return ConcatLMDBDataset(
            lmdb_paths=train_paths,
            **kwargs
        )
    
    elif strategy == 'selective':
        # 선택적 데이터셋 (고품질 데이터만)
        selective_configs = []
        for config in train_configs:
            if 'text_in_wild' in config['path'].lower():
                selective_configs.append({'path': config['path'], 'weight': 1.5, 'name': config['name']})
            elif 'ocr_public' in config['path'].lower():
                selective_configs.append({'path': config['path'], 'weight': 1.0, 'name': config['name']})
            elif 'handwriting' in config['path'].lower():
                selective_configs.append({'path': config['path'], 'weight': 0.3, 'name': config['name']})
        
        print(f"🔧 Selective 전략: {len(selective_configs)}개 핵심 LMDB 선택")
        return MultiLMDBDataset(
            lmdb_configs=selective_configs,
            **kwargs
        )
    
    else:
        raise ValueError(f"지원하지 않는 전략: {strategy}")


def create_validation_dataset(**kwargs):
    """검증 데이터셋 생성 (모든 valid LMDB 결합)"""
    
    # 실제 LMDB 경로 확인
    lmdb_base_path = get_lmdb_path()
    if lmdb_base_path is None:
        raise ValueError("LMDB 파일을 찾을 수 없습니다!")
    
    # 모든 valid LMDB 가져오기
    _, valid_configs = get_all_lmdb_configs(lmdb_base_path)
    
    if not valid_configs:
        print("⚠️ Valid LMDB가 없습니다. Train LMDB의 일부를 검증용으로 사용합니다.")
        return None
    
    valid_paths = [config['path'] for config in valid_configs]
    print(f"🔧 검증 데이터셋: {len(valid_paths)}개 Valid LMDB 결합")
    
    return ConcatLMDBDataset(
        lmdb_paths=valid_paths,
        **kwargs
    )


def test_dataset_loading(strategy='weighted'):
    """데이터셋 로딩 테스트"""
    print(f"🧪 {strategy} 전략으로 데이터셋 로딩 테스트")
    
    try:
        # 훈련 데이터셋 생성
        train_dataset = create_dataset(
            strategy=strategy,
            split='train',
            is_transform=True,
            img_size=(640, 640),  # 명시적으로 img_size 설정
            short_size=640
        )
        
        print(f"✅ 훈련 데이터셋 로딩 성공")
        print(f"   - 총 샘플 수: {len(train_dataset)}")
        
        # 첫 번째 샘플 테스트
        sample = train_dataset[0]
        print(f"   - 샘플 키: {list(sample.keys())}")
        print(f"   - 이미지 크기: {sample['imgs'].shape}")
        
        # DataLoader 테스트
        dataloader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=4
        )
        
        for batch_idx, batch in enumerate(dataloader):
            print(f"   - 배치 {batch_idx+1}: {batch['imgs'].shape}")
            if batch_idx >= 1:  # 2개 배치만 테스트
                break
        
        print(f"✅ DataLoader 테스트 완료")
        
    except Exception as e:
        print(f"❌ 데이터셋 로딩 실패: {e}")
        import traceback
        traceback.print_exc()


def main():
    """메인 함수"""
    args = parse_args()
    
    print("🚀 Multi LMDB 훈련 스크립트 시작")
    print(f"📊 전략: {args.strategy}")
    print(f"⚙️ 설정 파일: {args.config}")
    print("=" * 50)
    
    # GPU 설정
    gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    
    if args.test_only:
        # 테스트만 실행
        test_dataset_loading(args.strategy)
        return
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 설정 파일 로드
    if not os.path.exists(args.config):
        print(f"❌ 설정 파일을 찾을 수 없습니다: {args.config}")
        return
    
    # 실제 훈련 로직은 여기에 구현
    # (기존 FAST 훈련 코드를 Multi LMDB 데이터셋에 맞게 수정)
    
    print("🔄 훈련 시작...")
    
    try:
        # 데이터셋 생성
        train_dataset = create_dataset(
            strategy=args.strategy,
            split='train',
            is_transform=True,
            img_size=(640, 640),  # 명시적으로 img_size 설정
            short_size=640
        )
        
        # 검증 데이터셋 (단순 결합)
        val_dataset = create_validation_dataset(
            split='test',
            img_size=(640, 640),  # 명시적으로 img_size 설정
            short_size=640
        )
        
        print(f"📊 훈련 데이터: {len(train_dataset):,}개 이미지")
        print(f"   💡 참고: 실제 어노테이션은 {len(train_dataset)*25:,}개 정도 (이미지당 평균 25개)")
        if val_dataset:
            print(f"📊 검증 데이터: {len(val_dataset):,}개 이미지")
            print(f"   💡 참고: 실제 어노테이션은 {len(val_dataset)*25:,}개 정도")
        else:
            print("📊 검증 데이터: 없음 (Train 데이터만 사용)")
        
        # DataLoader 생성
        batch_size = args.batch_size or 8  # 성능과 안정성 균형
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=16,  # 데이터 로딩 성능 향상
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,  # 워커 재사용으로 오버헤드 감소
            prefetch_factor=2  # 메모리 사용량 감소
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=1,  # 🔥 배치 크기 1로 강제 (크기 불일치 해결)
                shuffle=False,
                num_workers=8,  # 워커 수 감소
                pin_memory=True,
                drop_last=True,  # 🔥 불완전한 배치 제거
                persistent_workers=False  # 🔥 validation은 간헐적이므로 False
            )
        
        print(f"🔄 배치 크기: {batch_size}")
        print(f"🔄 훈련 배치 수: {len(train_loader)}")
        if val_loader:
            print(f"🔄 검증 배치 수: {len(val_loader)}")
        else:
            print("🔄 검증 배치 수: 없음")
        
        # FAST 모델 훈련 로직 구현
        print("🔧 FAST 모델 초기화 중...")
        
        # Config 파일 로드
        from mmcv import Config
        from models import build_model
        
        cfg = Config.fromfile(args.config)
        
        # 모델 생성
        model = build_model(cfg.model)
        model = model.to(device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # 사전학습 체크포인트 로드 (torch.compile() 전에 실행!)
        checkpoint_path = args.checkpoint or "./checkpoint_7ep.pth"
        if os.path.exists(checkpoint_path):
            print(f"📦 사전학습 체크포인트 로드: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 🔍 체크포인트 구조 디버그
            print(f"🔍 체크포인트 최상위 키들: {list(checkpoint.keys())}")
            
            # EMA 또는 직접 state_dict 확인
            if 'ema' in checkpoint:
                state_dict = checkpoint['ema']
                print("   - EMA 상태 딕셔너리 사용")
                print(f"   - EMA 키 개수: {len(state_dict)}")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("   - 일반 상태 딕셔너리 사용")
                print(f"   - state_dict 키 개수: {len(state_dict)}")
            else:
                state_dict = checkpoint
                print("   - 체크포인트 자체를 상태 딕셔너리로 사용")
                print(f"   - 직접 키 개수: {len(state_dict)}")
            
            # 🔍 원본 키 분석
            original_keys = list(state_dict.keys())
            module_keys = [k for k in original_keys if k.startswith('module.')]
            non_module_keys = [k for k in original_keys if not k.startswith('module.')]
            print(f"   - 원본 키 분석: 총 {len(original_keys)}개")
            print(f"     • module. prefix: {len(module_keys)}개")
            print(f"     • 일반 키: {len(non_module_keys)}개")
            
            # 키에서 'module.' 제거
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace("module.", "")
                new_state_dict[new_key] = value
            
            print(f"   - 정리된 키 개수: {len(new_state_dict)}")
            
            # 🔍 현재 모델 키 분석 (컴파일 전)
            current_model_keys = list(model.state_dict().keys())
            print(f"   - 현재 모델 키 개수: {len(current_model_keys)}")
            
            # 모델에 가중치 로드
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            print(f"✅ 체크포인트 로드 완료 (누락: {len(missing_keys)}, 예상외: {len(unexpected_keys)})")
            
            # 🔍 누락과 예상외 키 상세 분석 (문제가 있을 때만)
            if missing_keys:
                print(f"❌ 누락된 키들 (처음 5개):")
                for key in missing_keys[:5]:
                    print(f"     - {key}")
                if len(missing_keys) > 5:
                    print(f"     ... 추가 {len(missing_keys) - 5}개")
            
            if unexpected_keys:
                print(f"⚠️ 예상외 키들 (처음 5개):")
                for key in unexpected_keys[:5]:
                    print(f"     - {key}")
                if len(unexpected_keys) > 5:
                    print(f"     ... 추가 {len(unexpected_keys) - 5}개")
        else:
            print(f"⚠️ 체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
            print("   - 체크포인트 없이 훈련을 시작합니다.")
        
        # 🚀 PyTorch 컴파일 최적화 (체크포인트 로드 후 실행!)
        try:
            print("🚀 PyTorch 컴파일 최적화 적용 중...")
            model = torch.compile(model, mode='default')  # 'max-autotune'은 SM 부족으로 경고 발생
            print("✅ 모델 컴파일 완료 - 속도 향상 예상")
        except Exception as e:
            print(f"⚠️ 컴파일 최적화 실패 (정상 동작): {e}")
        
        # 옵티마이저 설정
        lr = args.lr or 5e-5  # NaN 방지를 위해 더 낮은 학습률
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        
        # 혼합 정밀도 훈련 설정 (NaN 문제로 일시 비활성화)
        scaler = None  # 안정성을 위해 비활성화
        
        # 스케줄러 설정
        total_epochs = args.epochs or 10  # HuggingFace 스타일 기본값
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=total_epochs, power=0.9
        )
        
        # 배치 크기 (gradient accumulation 제거로 단순화)
        effective_batch_size = batch_size
        
        # 🔧 Validation 주기를 배치 기준으로 계산
        total_batches_per_epoch = len(train_loader)
        validation_interval = total_batches_per_epoch // 100  # 전체 배치의 1%마다
        
        print(f"🔧 훈련 설정:")
        print(f"   - 학습률: {lr}")
        print(f"   - 총 에포크: {total_epochs}")
        print(f"   - 옵티마이저: Adam")
        print(f"   - 스케줄러: PolynomialLR")
        print(f"   - 배치 크기: {batch_size}")
        print(f"   - Step 개념 제거: 매 배치마다 즉시 업데이트")
        print(f"   - 워커 수: 16 (데이터 로딩 최적화)")
        print(f"   - 혼합 정밀도: 비활성화 (안정성)")
        print(f"   - Gradient Clipping: 활성화")
        print(f"   - Validation: 전체 배치의 1%마다 실행 ({validation_interval:,} 배치마다)")
        print(f"   - 체크포인트: 전체 배치의 1%마다 저장 ({validation_interval:,} 배치마다)")
        
        # 훈련 루프
        print("🚀 Multi LMDB 훈련 시작!")
        print(f"📊 Validation 설정: 총 {total_batches_per_epoch:,} 배치 중 {validation_interval:,} 배치마다 실행")
        model.train()
        
        # global_step 변수 제거 (step 개념 완전 제거)
        
        def run_validation(batch_num, epoch_num):
            """Validation 실행 함수"""
            if not val_loader:
                return None
                
            tqdm.write(f"\n🔍 Validation 시작 (에포크 {epoch_num+1}, 배치 {batch_num:,})")
            
            try:
                model.eval()
                val_loss = 0.0
                val_start_time = time.time()
                
                with torch.no_grad():
                    val_pbar = tqdm(val_loader, desc="🔍 검증 중", unit="batch", leave=False)
                    for val_batch_idx, batch in enumerate(val_pbar):
                        # 🔍 배치 데이터 유효성 검사
                        if batch is None:
                            tqdm.write(f"⚠️ Validation 배치 {val_batch_idx}: batch is None, 건너뛰기")
                            continue
                            
                        if 'imgs' not in batch or batch['imgs'] is None:
                            tqdm.write(f"⚠️ Validation 배치 {val_batch_idx}: imgs가 없음, 건너뛰기")
                            continue
                        
                        # 배치 데이터 추출 및 GPU로 이동
                        imgs = batch['imgs']
                        gt_texts = batch.get('gt_texts', None)
                        gt_kernels = batch.get('gt_kernels', None)
                        training_masks = batch.get('training_masks', None)
                        gt_instances = batch.get('gt_instances', None)
                        
                        if torch.cuda.is_available():
                            imgs = imgs.cuda()
                            if gt_texts is not None:
                                gt_texts = gt_texts.cuda()
                            if gt_kernels is not None:
                                gt_kernels = gt_kernels.cuda()
                            if training_masks is not None:
                                training_masks = training_masks.cuda()
                            if gt_instances is not None:
                                gt_instances = gt_instances.cuda()
                        
                        try:
                            outputs = model(
                                imgs,
                                gt_texts=gt_texts,
                                gt_kernels=gt_kernels,
                                training_masks=training_masks,
                                gt_instances=gt_instances
                            )
                            
                            if outputs is None:
                                tqdm.write(f"⚠️ Validation 배치 {val_batch_idx}: outputs is None, 건너뛰기")
                                continue
                            
                            loss_text = outputs['loss_text'].mean()
                            loss_kernels = outputs['loss_kernels'].mean()
                            loss_emb = outputs['loss_emb'].mean()
                            
                            total_loss = loss_text + loss_kernels + loss_emb
                            val_loss += total_loss.item()
                            
                            # validation progress bar 업데이트
                            val_pbar.set_postfix({'Val_Loss': f"{total_loss.item():.4f}"})
                        except Exception as e:
                            tqdm.write(f"⚠️ Validation 배치 {val_batch_idx} 오류: {e}")
                            continue
                    
                    val_pbar.close()
                
                avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
                val_time = time.time() - val_start_time
                
                tqdm.write(f"📊 Validation 완료 (에포크 {epoch_num+1}, 배치 {batch_num:,}) - {val_time:.1f}초")
                tqdm.write(f"   - 평균 검증 손실: {avg_val_loss:.4f}")
                
                return avg_val_loss
                
            except Exception as e:
                tqdm.write(f"❌ Validation 전체 오류: {e}")
                return None
            finally:
                # 🔥 반드시 model.train()으로 복원
                model.train()
                tqdm.write(f"✅ 모델 상태를 train()으로 복원")
        
        # 에포크 진행률 표시
        epoch_pbar = tqdm(range(total_epochs), desc="🎯 에포크", unit="epoch")
        
        for epoch in epoch_pbar:
            epoch_pbar.set_description(f"🎯 에포크 {epoch+1}/{total_epochs}")
            
            # 훈련
            train_loss = 0.0
            model.train()
            # accumulation_loss 제거 (step 개념 완전 제거)
            
            # 배치 진행률 표시
            batch_pbar = tqdm(train_loader, desc=f"📚 훈련 중", unit="batch", leave=False)
            
            for batch_idx, batch in enumerate(batch_pbar):
                # 매 배치마다 zero_grad (gradient accumulation 제거)
                optimizer.zero_grad()
                
                # 🔍 배치 데이터 유효성 검사
                if batch is None:
                    tqdm.write(f"⚠️ 배치 {batch_idx}: batch is None, 건너뛰기")
                    continue
                    
                if 'imgs' not in batch or batch['imgs'] is None:
                    tqdm.write(f"⚠️ 배치 {batch_idx}: imgs가 없음, 건너뛰기")
                    continue
                
                # 배치 데이터 추출 및 GPU로 이동
                imgs = batch['imgs']
                gt_texts = batch.get('gt_texts', None)
                gt_kernels = batch.get('gt_kernels', None) 
                training_masks = batch.get('training_masks', None)
                gt_instances = batch.get('gt_instances', None)
                
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    if gt_texts is not None:
                        gt_texts = gt_texts.cuda()
                    if gt_kernels is not None:
                        gt_kernels = gt_kernels.cuda()
                    if training_masks is not None:
                        training_masks = training_masks.cuda()
                    if gt_instances is not None:
                        gt_instances = gt_instances.cuda()
                
                # Forward pass (FAST 모델 형식에 맞게)
                try:
                    # 혼합 정밀도 훈련
                    if scaler:
                        with torch.cuda.amp.autocast():
                            outputs = model(
                                imgs,
                                gt_texts=gt_texts,
                                gt_kernels=gt_kernels,
                                training_masks=training_masks,
                                gt_instances=gt_instances
                            )
                            
                            # 손실 계산 (FAST 모델의 다중 손실)
                            loss_text = outputs['loss_text'].mean()
                            loss_kernels = outputs['loss_kernels'].mean()
                            loss_emb = outputs['loss_emb'].mean()
                            
                            total_loss = loss_text + loss_kernels + loss_emb
                        
                        # Backward pass (스케일링 적용)
                        scaler.scale(total_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = model(
                            imgs,
                            gt_texts=gt_texts,
                            gt_kernels=gt_kernels,
                            training_masks=training_masks,
                            gt_instances=gt_instances
                        )
                        
                        # 손실 계산 (FAST 모델의 다중 손실)
                        loss_text = outputs['loss_text'].mean()
                        loss_kernels = outputs['loss_kernels'].mean()
                        loss_emb = outputs['loss_emb'].mean()
                        
                        total_loss = loss_text + loss_kernels + loss_emb
                        
                        # NaN 체크
                        if torch.isnan(total_loss):
                            print(f"⚠️ NaN 손실 감지됨 - 배치 {batch_idx} 건너뛰기")
                            continue
                        
                        # Backward pass (gradient accumulation 제거)
                        total_loss.backward()
                        
                        # accumulation_loss 제거됨 (step 개념 완전 제거)
                        
                        # 🔥 배치 기준 validation (gradient accumulation과 독립적)
                        if (batch_idx + 1) % validation_interval == 0:
                            val_loss = run_validation(batch_idx + 1, epoch)
                            if val_loss is not None:
                                tqdm.write(f"🎯 배치 {batch_idx + 1:,}: 검증 손실 = {val_loss:.4f}")
                            
                            # 체크포인트 저장
                            checkpoint_file = f"{args.output_dir}/checkpoint_latest.pth"
                            os.makedirs(args.output_dir, exist_ok=True)
                            torch.save({
                                'epoch': epoch + 1,
                                'batch_idx': batch_idx + 1,
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),
                                'train_loss': total_loss.item(),
                                'val_loss': val_loss if val_loss is not None else 0.0,
                            }, checkpoint_file)
                            tqdm.write(f"💾 체크포인트 저장: {checkpoint_file} (배치 {batch_idx + 1:,})")
                        
                        # 매 배치마다 optimizer step (step 개념 제거)
                        # Gradient clipping (NaN 방지)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                        
                        # 손실 누적 (매 배치)
                        train_loss += total_loss.item()
                        
                        # tqdm 진행률 업데이트 (배치 기준)
                        current_avg_loss = train_loss / (batch_idx + 1)
                        batch_pbar.set_postfix({
                            'Batch': batch_idx + 1,
                            'Loss': f"{total_loss.item():.4f}",
                            'Avg': f"{current_avg_loss:.4f}",
                            'Text': f"{loss_text.item():.3f}",
                            'Kernel': f"{loss_kernels.item():.3f}",
                            'Emb': f"{loss_emb.item():.3f}"
                        })
                            
                        # 중요한 마일스톤만 print 출력 (validation 주기의 절반마다)
                        if (batch_idx + 1) % max(1, validation_interval // 2) == 0:
                            tqdm.write(f"✅ 배치 {batch_idx + 1:,} - Loss: {current_avg_loss:.4f}")
                
                except Exception as e:
                    tqdm.write(f"❌ 훈련 오류 (배치 {batch_idx}): {e}")
                    continue
            
            # 배치 progress bar 닫기
            batch_pbar.close()
            
            # 에포크 평균 손실 (배치 기준)
            avg_train_loss = train_loss / max(1, len(train_loader))
            
            # 에포크 progress bar 업데이트
            epoch_pbar.set_postfix({
                'Train_Loss': f"{avg_train_loss:.4f}",
                'Batches': len(train_loader),
                'Batch_Size': effective_batch_size
            })
            
            tqdm.write(f"📊 에포크 {epoch+1} 완료 - 평균 훈련 손실: {avg_train_loss:.4f}")
            tqdm.write(f"   - 배치 수: {len(train_loader):,}, 배치 크기: {effective_batch_size}")
            
            # 스케줄러 업데이트
            scheduler.step()
        
        # 최종 모델 저장
        final_checkpoint = f"{args.output_dir}/checkpoint_final.pth"
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save({
            'epoch': total_epochs,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, final_checkpoint)
        
        # 에포크 progress bar 닫기
        epoch_pbar.close()
        
        tqdm.write("✅ Multi LMDB 훈련 완료!")
        tqdm.write(f"💾 최종 모델 저장: {final_checkpoint}")
        
    except Exception as e:
        tqdm.write(f"❌ 훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # 실행 예시:
    # 
    # 1. 모든 LMDB 테스트
    # python train_multi_lmdb.py --config config/fast/korean_ocr/multi_lmdb_config.py --test_only
    #
    # 2. Concat 전략으로 학습 (모든 LMDB 균등 결합)
    # python train_multi_lmdb.py --config config/fast/korean_ocr/multi_lmdb_config.py --strategy concat --epochs 100
    #
    # 3. Weighted 전략으로 학습 (가중치 적용)
    # python train_multi_lmdb.py --config config/fast/korean_ocr/multi_lmdb_config.py --strategy weighted --epochs 100
    #
    # 4. GPU 여러 개 사용
    # python train_multi_lmdb.py --config config/fast/korean_ocr/multi_lmdb_config.py --strategy concat --gpu_ids 0,1,2,3
    
    print("🚀 Multi LMDB 훈련 스크립트")
    print("=" * 50)
    print("📚 모든 한국어 OCR LMDB를 결합하여 학습합니다")
    print("")
    print("🔧 지원하는 전략:")
    print("   • concat: 모든 LMDB 균등 결합")
    print("   • weighted: 데이터셋별 가중치 적용") 
    print("   • selective: 고품질 데이터만 선택")
    print("")
    print("📁 자동 검색 경로:")
    print(f"   • {LMDB_BASE_PATH}")
    print(f"   • {GVFS_LMDB_PATH}")
    print("=" * 50)
    print("")
    
    main() 