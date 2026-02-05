#!/usr/bin/env python3
"""
Multi LMDB 데이터셋을 사용한 FAST 모델 훈련 (DeepSpeed 적용)
"""

import argparse
import os
import sys
import time
import torch
import deepspeed
from torch.utils.data import DataLoader
from mmcv import Config
from tqdm import tqdm

# FAST 모듈 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import build_model
from dataset.fast.multi_lmdb_dataset import MultiLMDBDataset, ConcatLMDBDataset


def create_dataset(strategy, **kwargs):
    """훈련 데이터셋 생성 함수"""
    # LMDB 경로들 
    lmdb_paths = [
        "/mnt/nas/ocr_dataset/text_in_wild_train.lmdb",
        "/mnt/nas/ocr_dataset/public_admin_train.lmdb", 
        "/mnt/nas/ocr_dataset/ocr_public_train.lmdb",
        "/mnt/nas/ocr_dataset/finance_logistics_train.lmdb",
        "/mnt/nas/ocr_dataset/handwriting_train.lmdb"
    ]
    
    # 존재하는 LMDB만 필터링
    existing_paths = [path for path in lmdb_paths if os.path.exists(path)]
    print(f"🔧 훈련 데이터셋: {len(existing_paths)}개 LMDB 발견")
    
    if strategy == 'concat':
        return ConcatLMDBDataset(existing_paths, **kwargs)
    else:
        # 기본적으로 concat 사용
        return ConcatLMDBDataset(existing_paths, **kwargs)


def create_validation_dataset(**kwargs):
    """검증 데이터셋 생성 함수"""
    lmdb_paths = [
        "/mnt/nas/ocr_dataset/text_in_wild_valid.lmdb",
        "/mnt/nas/ocr_dataset/public_admin_valid.lmdb",
        "/mnt/nas/ocr_dataset/ocr_public_valid.lmdb", 
        "/mnt/nas/ocr_dataset/finance_logistics_valid.lmdb",
        "/mnt/nas/ocr_dataset/handwriting_valid.lmdb"
    ]
    
    # 존재하는 LMDB만 필터링
    existing_paths = [path for path in lmdb_paths if os.path.exists(path)]
    print(f"🔧 검증 데이터셋: {len(existing_paths)}개 Valid LMDB 결합")
    
    if existing_paths:
        return ConcatLMDBDataset(existing_paths, **kwargs)
    else:
        return None


def get_args():
    parser = argparse.ArgumentParser(description='DeepSpeed Multi LMDB FAST 훈련')
    
    # 우리 custom arguments 먼저 추가
    parser.add_argument('--config', type=str, required=True,
                        help='설정 파일 경로')
    parser.add_argument('--strategy', type=str, default='concat',
                        choices=['balanced', 'weighted', 'selective', 'concat'],
                        help='데이터셋 결합 전략')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='사전학습 체크포인트 경로')
    parser.add_argument('--output_dir', type=str, default='./work_dirs/multi_lmdb_deepspeed',
                        help='출력 디렉토리')
    
    # TrOCR와 동일한 배치 사이즈 설정 (auto를 위한 명시적 지정)
    parser.add_argument('--per_device_train_batch_size', type=int, default=8,
                        help='훈련시 디바이스 당 배치 사이즈')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=8,
                        help='평가시 디바이스 당 배치 사이즈')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help='그래디언트 누적 스텝')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='학습률')
    
    # DeepSpeed가 필요한 --local_rank 수동 추가
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for DeepSpeed')
    
    # 이제 DeepSpeed arguments 추가 (--local_rank 제외)
    parser = deepspeed.add_config_arguments(parser)
    
    return parser.parse_args()


def main():
    args = get_args()
    
    # 설정 로드
    cfg = Config.fromfile(args.config)
    
    print("🚀 DeepSpeed Multi LMDB FAST 훈련 시작!")
    print(f"📁 설정 파일: {args.config}")
    print(f"📊 전략: {args.strategy}")
    print(f"⚡ DeepSpeed 설정: {args.deepspeed_config}")
    
    # 훈련 데이터셋 생성
    print(f"\n🔧 훈련 데이터셋: {args.strategy} 전략으로 모든 LMDB 결합")
    train_dataset = create_dataset(
        'weighted' if args.strategy == 'weighted' else 'concat',
        split='train',
        is_transform=True,
        img_size=(640, 640),
        short_size=640
    )
    
    # 검증 데이터셋 생성 (선택적)
    print(f"🔧 검증 데이터셋: 5개 Valid LMDB 결합")
    val_dataset = create_validation_dataset(
        split='test',
        is_transform=False,
        img_size=(640, 640),
        short_size=640
    )
    
    print(f"📊 훈련 데이터: {len(train_dataset):,}개 이미지")
    print(f"   💡 참고: 실제 어노테이션은 {len(train_dataset)*25:,}개 정도 (이미지당 평균 25개)")
    if val_dataset:
        print(f"📊 검증 데이터: {len(val_dataset):,}개 이미지")
        print(f"   💡 참고: 실제 어노테이션은 {len(val_dataset)*25:,}개 정도")
    else:
        print("📊 검증 데이터: 없음 (Train 데이터만 사용)")
    
    # 모델 생성
    print("🔧 FAST 모델 초기화 중...")
    model = build_model(cfg.model)
    
    # 체크포인트 로드
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"📦 사전학습 체크포인트 로드: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        
        # EMA 또는 직접 state_dict 확인
        if 'ema' in checkpoint:
            state_dict = checkpoint['ema']
            print("   - EMA 상태 딕셔너리 사용")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("   - 일반 상태 딕셔너리 사용")
        else:
            state_dict = checkpoint
            print("   - 체크포인트 자체를 상태 딕셔너리로 사용")
        
        # 키에서 'module.' 제거
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value
        
        # 모델에 가중치 로드
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        print(f"✅ 체크포인트 로드 완료 (누락: {len(missing_keys)}, 예상외: {len(unexpected_keys)})")
    else:
        print(f"⚠️ 체크포인트 파일을 찾을 수 없습니다: {args.checkpoint}")
        print("   - 체크포인트 없이 훈련을 시작합니다.")
    
    # DeepSpeed 초기화 (마이크로 배치 4로 105,374 배치)
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        args=args,
        model=model
    )
    
    # 마이크로 배치 크기로 DataLoader 생성 (배치 수 105,374)
    micro_batch_size = args.per_device_train_batch_size  # 4 (GPU 메모리 고정)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,  # 4 (배치 수 105,374)
        shuffle=True,
        num_workers=8,  # 4 → 8로 증가 (CPU 코어 활용)
        pin_memory=True,  # False → True (GPU 전송 속도 향상)
        drop_last=True,
        persistent_workers=True,  # False → True (워커 재사용)
        prefetch_factor=4  # 1 → 4로 증가 (미리 로딩)
    )
    
    # 검증 데이터로더 (마이크로 배치 크기 사용)
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=micro_batch_size,  # 4 (GPU 메모리 고정)
            shuffle=False,
            num_workers=8,  # 4 → 8로 증가 (CPU 활용)  
            pin_memory=True,  # False → True (GPU 전송 속도 향상)
            persistent_workers=True  # False → True (워커 재사용)
        )
    
    print(f"🔄 배치 크기: {model_engine.train_micro_batch_size_per_gpu()}")
    print(f"🔄 훈련 배치 수: {len(train_loader)}")
    if val_loader:
        print(f"🔄 검증 배치 수: {len(val_loader)}")
    else:
        print("🔄 검증 배치 수: 없음")
    
    # 효과적 배치 크기 계산 (다른 프로젝트 방식)
    effective_batch_size = model_engine.train_micro_batch_size_per_gpu() * model_engine.gradient_accumulation_steps()
    
    print(f"🔧 DeepSpeed 설정:")
    print(f"   - 마이크로 배치 크기: {model_engine.train_micro_batch_size_per_gpu()}")
    print(f"   - 그래디언트 누적: {model_engine.gradient_accumulation_steps()}")
    print(f"   - 효과적 배치 크기: {effective_batch_size}")
    print(f"   - GPU 메모리 사용: {model_engine.train_micro_batch_size_per_gpu()} 배치 크기만")
    print(f"   - 혼합 정밀도: {model_engine.fp16_enabled()}")
    
    # 훈련 루프
    print("🚀 DeepSpeed Multi LMDB 훈련 시작!")
    
    total_epochs = 10  # HuggingFace 스타일
    # 에포크 진행률 표시
    epoch_pbar = tqdm(range(total_epochs), desc="🎯 에포크", unit="epoch")
    
    for epoch in epoch_pbar:
        epoch_pbar.set_description(f"🎯 에포크 {epoch+1}/{total_epochs}")
        
        # 훈련
        model_engine.train()
        train_loss = 0.0
        start_time = time.time()
        
        # 배치 진행률 표시
        batch_pbar = tqdm(train_loader, desc=f"📚 훈련 중", unit="batch", leave=False)
        
        for batch_idx, batch in enumerate(batch_pbar):
            # 배치 데이터 추출 및 GPU로 이동
            imgs = batch['imgs'].cuda()
            gt_texts = batch['gt_texts'].cuda() if 'gt_texts' in batch and batch['gt_texts'] is not None else None
            gt_kernels = batch['gt_kernels'].cuda() if 'gt_kernels' in batch and batch['gt_kernels'] is not None else None
            training_masks = batch['training_masks'].cuda() if 'training_masks' in batch and batch['training_masks'] is not None else None
            gt_instances = batch['gt_instances'].cuda() if 'gt_instances' in batch and batch['gt_instances'] is not None else None
            
            # Forward pass
            try:
                outputs = model_engine(
                    imgs,
                    gt_texts=gt_texts,
                    gt_kernels=gt_kernels,
                    training_masks=training_masks,
                    gt_instances=gt_instances
                )
                
                # 손실 계산
                loss_text = outputs['loss_text'].mean()
                loss_kernels = outputs['loss_kernels'].mean()
                loss_emb = outputs['loss_emb'].mean()
                
                total_loss = loss_text + loss_kernels + loss_emb
                
                # DeepSpeed backward (자동 gradient accumulation)
                model_engine.backward(total_loss)
                model_engine.step()  # DeepSpeed가 자동으로 gradient accumulation 처리
                train_loss += total_loss.item()
                
                # tqdm 진행률 업데이트
                avg_loss = train_loss / (batch_idx + 1)
                elapsed = time.time() - start_time
                batches_per_sec = (batch_idx + 1) / elapsed if elapsed > 0 else 0
                
                batch_pbar.set_postfix({
                    'Loss': f"{total_loss.item():.4f}",
                    'Avg': f"{avg_loss:.4f}",
                    'Text': f"{loss_text.item():.3f}",
                    'Kernel': f"{loss_kernels.item():.3f}",
                    'Emb': f"{loss_emb.item():.3f}",
                    'Speed': f"{batches_per_sec:.1f}b/s",
                    'GPU_Mem': f"배치{model_engine.train_micro_batch_size_per_gpu()}고정"
                })
                
                # 중요한 마일스톤만 print 출력
                if (batch_idx + 1) % 1000 == 0:
                    tqdm.write(f"✅ 배치 {batch_idx+1:,} - Loss: {total_loss.item():.4f} (GPU 메모리: 배치 {model_engine.train_micro_batch_size_per_gpu()} 고정)")
            
            except Exception as e:
                tqdm.write(f"❌ 훈련 오류 (배치 {batch_idx}): {e}")
                continue
        
        # 배치 progress bar 닫기
        batch_pbar.close()
        
        # 에포크 통계
        epoch_time = time.time() - start_time
        avg_train_loss = train_loss / len(train_loader)
        
        # 에포크 progress bar 업데이트
        epoch_pbar.set_postfix({
            'Train_Loss': f"{avg_train_loss:.4f}",
            'Time': f"{epoch_time:.1f}s"
        })
        
        tqdm.write(f"📊 에포크 {epoch+1} 완료 ({epoch_time:.1f}초)")
        tqdm.write(f"   - 평균 훈련 손실: {avg_train_loss:.4f}")
        
        # 검증 (10 에포크마다)
        if val_loader and (epoch + 1) % 2 == 0:  # 더 자주 검증
            model_engine.eval()
            val_loss = 0.0
            val_start = time.time()
            
            tqdm.write(f"\n🔍 Validation 시작 (에포크 {epoch+1})")
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc="🔍 검증 중", unit="batch", leave=False)
                for batch in val_pbar:
                    imgs = batch['imgs']
                    gt_texts = batch['gt_texts'] if 'gt_texts' in batch else None
                    gt_kernels = batch['gt_kernels'] if 'gt_kernels' in batch else None
                    training_masks = batch['training_masks'] if 'training_masks' in batch else None
                    gt_instances = batch['gt_instances'] if 'gt_instances' in batch else None
                    
                    # 검증 데이터를 GPU로 이동 (효율적인 방식)
                    imgs = imgs.cuda()
                    gt_texts = gt_texts.cuda() if gt_texts is not None else None
                    gt_kernels = gt_kernels.cuda() if gt_kernels is not None else None
                    training_masks = training_masks.cuda() if training_masks is not None else None
                    gt_instances = gt_instances.cuda() if gt_instances is not None else None
                    
                    try:
                        outputs = model_engine(
                            imgs,
                            gt_texts=gt_texts,
                            gt_kernels=gt_kernels,
                            training_masks=training_masks,
                            gt_instances=gt_instances
                        )
                        
                        loss_text = outputs['loss_text'].mean()
                        loss_kernels = outputs['loss_kernels'].mean()
                        loss_emb = outputs['loss_emb'].mean()
                        
                        total_loss = loss_text + loss_kernels + loss_emb
                        val_loss += total_loss.item()
                        
                        # validation progress bar 업데이트
                        val_pbar.set_postfix({'Val_Loss': f"{total_loss.item():.4f}"})
                    except:
                        continue
                
                val_pbar.close()
            
            avg_val_loss = val_loss / len(val_loader)
            val_time = time.time() - val_start
            tqdm.write(f"📊 검증 완료 ({val_time:.1f}초)")
            tqdm.write(f"   - 평균 검증 손실: {avg_val_loss:.4f}")
        
        # 체크포인트 저장 (5 에포크마다)
        if (epoch + 1) % 5 == 0:
            checkpoint_dir = f"{args.output_dir}/checkpoint_latest"
            os.makedirs(checkpoint_dir, exist_ok=True)
            model_engine.save_checkpoint(checkpoint_dir)
            tqdm.write(f"💾 체크포인트 저장: {checkpoint_dir} (에포크 {epoch+1})")
    
    # 에포크 progress bar 닫기
    epoch_pbar.close()
    
    tqdm.write("✅ 훈련 완료!")
    
    # 최종 모델 저장
    final_checkpoint = f"{args.output_dir}/final_model"
    os.makedirs(final_checkpoint, exist_ok=True)
    model_engine.save_checkpoint(final_checkpoint)
    tqdm.write(f"💾 최종 모델 저장: {final_checkpoint}")
    print(f"🎉 훈련 완료! 최종 모델 저장: {final_checkpoint}")


if __name__ == '__main__':
    main() 