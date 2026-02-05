#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FAST 모델 파인튜닝 훈련 스크립트
checkpoint_7ep.pth를 기반으로 사용자 데이터에 파인튜닝
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# FAST 관련 import
sys.path.append('/home/mango/ocr_test/FAST')
from mmcv import Config
from models import build_model
from models.utils import rep_model_convert, fuse_module
from dataset import build_data_loader


class FastTrainer:
    def __init__(self, config_path, checkpoint_path, output_dir='./checkpoints'):
        """
        FAST 훈련 클래스 초기화
        
        Args:
            config_path (str): 설정 파일 경로
            checkpoint_path (str): 사전 학습된 체크포인트 경로
            output_dir (str): 체크포인트 저장 디렉토리
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 설정 로드
        self.cfg = Config.fromfile(config_path)
        
        # 모델 초기화
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.start_epoch = 0
        
        print(f"🚀 FAST 훈련 초기화 완료")
        print(f"   - 설정 파일: {config_path}")
        print(f"   - 체크포인트: {checkpoint_path}")
        print(f"   - 출력 디렉토리: {output_dir}")
        print(f"   - 디바이스: {self.device}")
    
    def build_model(self):
        """모델 구성 및 체크포인트 로드"""
        print("🔧 모델 구성 중...")
        
        # 모델 생성
        self.model = build_model(self.cfg.model)
        self.model = self.model.to(self.device)
        
        # 체크포인트 로드
        if os.path.isfile(self.checkpoint_path):
            print(f"📦 체크포인트 로드: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # state_dict 추출
            if 'ema' in checkpoint:
                state_dict = checkpoint['ema']
                print("   - EMA 상태 딕셔너리 사용")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("   - 일반 상태 딕셔너리 사용")
            else:
                state_dict = checkpoint
                print("   - 직접 상태 딕셔너리 사용")
            
            # 에포크 정보 추출
            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch']
                print(f"   - 시작 에포크: {self.start_epoch}")
            
            # 키에서 'module.' 제거
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace("module.", "")
                new_state_dict[new_key] = value
            
            # 모델에 가중치 로드
            missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
            
            if missing_keys:
                print(f"   ⚠️ 누락된 키: {len(missing_keys)}개")
            if unexpected_keys:
                print(f"   ⚠️ 예상치 못한 키: {len(unexpected_keys)}개")
            
            print("✅ 체크포인트 로드 완료")
        else:
            raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {self.checkpoint_path}")
        
        # 훈련 모드로 설정
        self.model.train()
        
    def build_optimizer(self):
        """옵티마이저 및 스케줄러 구성"""
        print("🔧 옵티마이저 구성 중...")
        
        # 학습률 및 옵티마이저 설정
        lr = getattr(self.cfg.train_cfg, 'lr', 1e-3)
        optimizer_type = getattr(self.cfg.train_cfg, 'optimizer', 'Adam')
        
        if optimizer_type == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_type == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=lr, 
                momentum=0.9, 
                weight_decay=1e-4
            )
        else:
            raise ValueError(f"지원하지 않는 옵티마이저: {optimizer_type}")
        
        # 스케줄러 설정
        schedule_type = getattr(self.cfg.train_cfg, 'schedule', 'polylr')
        if schedule_type == 'polylr':
            # Polynomial learning rate decay
            total_epochs = getattr(self.cfg.train_cfg, 'epoch', 100)
            self.scheduler = optim.lr_scheduler.PolynomialLR(
                self.optimizer, 
                total_iters=total_epochs,
                power=0.9
            )
        else:
            # 기본 스케줄러
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=30, 
                gamma=0.1
            )
        
        print(f"   - 옵티마이저: {optimizer_type}")
        print(f"   - 초기 학습률: {lr}")
        print(f"   - 스케줄러: {schedule_type}")
        print("✅ 옵티마이저 구성 완료")
    
    def build_dataloader(self):
        """데이터로더 구성"""
        print("🔧 데이터로더 구성 중...")
        
        # 훈련 데이터로더
        train_dataset = build_data_loader(self.cfg.data.train)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=getattr(self.cfg.data, 'batch_size', 4),
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        print(f"   - 배치 크기: {getattr(self.cfg.data, 'batch_size', 4)}")
        print(f"   - 훈련 데이터 수: {len(train_dataset)}")
        print(f"   - 배치 수: {len(self.train_loader)}")
        print("✅ 데이터로더 구성 완료")
    
    def train_epoch(self, epoch):
        """한 에포크 훈련"""
        self.model.train()
        total_loss = 0.0
        total_loss_text = 0.0
        total_loss_kernel = 0.0
        total_loss_emb = 0.0
        
        print(f"\n📚 에포크 {epoch+1} 훈련 시작...")
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 데이터를 GPU로 이동
            imgs = batch['imgs'].to(self.device)
            gt_texts = batch['gt_texts'].to(self.device)
            gt_kernels = batch['gt_kernels'].to(self.device)
            training_masks = batch['training_masks'].to(self.device)
            gt_instances = batch['gt_instances'].to(self.device)
            
            # 옵티마이저 초기화
            self.optimizer.zero_grad()
            
            # 순방향 패스
            outputs = self.model(
                imgs=imgs,
                gt_texts=gt_texts,
                gt_kernels=gt_kernels,
                training_masks=training_masks,
                gt_instances=gt_instances
            )
            
            # 손실 계산
            loss_text = outputs['loss_text'].mean()
            loss_kernels = outputs['loss_kernels'].mean()
            loss_emb = outputs['loss_emb'].mean()
            
            total_loss_batch = loss_text + loss_kernels + loss_emb
            
            # 역방향 패스
            total_loss_batch.backward()
            self.optimizer.step()
            
            # 손실 누적
            total_loss += total_loss_batch.item()
            total_loss_text += loss_text.item()
            total_loss_kernel += loss_kernels.item()
            total_loss_emb += loss_emb.item()
            
            # 진행 상황 출력
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"   배치 [{batch_idx+1}/{len(self.train_loader)}] "
                      f"Loss: {total_loss_batch.item():.4f} "
                      f"(Avg: {avg_loss:.4f}) "
                      f"Text: {loss_text.item():.4f} "
                      f"Kernel: {loss_kernels.item():.4f} "
                      f"Emb: {loss_emb.item():.4f}")
        
        # 에포크 통계
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(self.train_loader)
        avg_loss_text = total_loss_text / len(self.train_loader)
        avg_loss_kernel = total_loss_kernel / len(self.train_loader)
        avg_loss_emb = total_loss_emb / len(self.train_loader)
        
        print(f"📊 에포크 {epoch+1} 완료 ({epoch_time:.1f}초)")
        print(f"   - 평균 총 손실: {avg_loss:.4f}")
        print(f"   - 평균 텍스트 손실: {avg_loss_text:.4f}")
        print(f"   - 평균 커널 손실: {avg_loss_kernel:.4f}")
        print(f"   - 평균 임베딩 손실: {avg_loss_emb:.4f}")
        print(f"   - 학습률: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        return avg_loss
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.cfg
        }
        
        # 일반 체크포인트 저장
        checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 최신 체크포인트 저장
        latest_path = os.path.join(self.output_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, latest_path)
        
        # 최고 성능 모델 저장
        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f"💾 최고 성능 모델 저장: {best_path}")
        
        print(f"💾 체크포인트 저장: {checkpoint_path}")
    
    def train(self, num_epochs=None):
        """전체 훈련 프로세스"""
        if num_epochs is None:
            num_epochs = getattr(self.cfg.train_cfg, 'epoch', 100)
        
        print(f"🚀 FAST 모델 훈련 시작")
        print(f"   - 총 에포크: {num_epochs}")
        print(f"   - 시작 에포크: {self.start_epoch}")
        print("=" * 60)
        
        best_loss = float('inf')
        save_interval = getattr(self.cfg.train_cfg, 'save_interval', 10)
        
        for epoch in range(self.start_epoch, num_epochs):
            # 훈련
            avg_loss = self.train_epoch(epoch)
            
            # 스케줄러 업데이트
            self.scheduler.step()
            
            # 체크포인트 저장
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
            
            # 주기적 저장 또는 최고 성능일 때 저장
            if (epoch + 1) % save_interval == 0 or is_best or (epoch + 1) == num_epochs:
                self.save_checkpoint(epoch, avg_loss, is_best)
        
        print("🎉 훈련 완료!")
        print(f"   - 최고 성능 손실: {best_loss:.4f}")
        print(f"   - 체크포인트 저장 위치: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='FAST 모델 파인튜닝 훈련')
    parser.add_argument('--config', type=str, required=True, help='설정 파일 경로')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_7ep.pth', help='사전 학습된 체크포인트 경로')
    parser.add_argument('--output_dir', type=str, default='./finetune_checkpoints', help='체크포인트 저장 디렉토리')
    parser.add_argument('--epochs', type=int, default=None, help='훈련 에포크 수')
    parser.add_argument('--gpu', type=str, default='0', help='사용할 GPU ID')
    
    args = parser.parse_args()
    
    # GPU 설정
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # 훈련 실행
    trainer = FastTrainer(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir
    )
    
    # 구성 요소 초기화
    trainer.build_model()
    trainer.build_optimizer()
    trainer.build_dataloader()
    
    # 훈련 시작
    trainer.train(num_epochs=args.epochs)


if __name__ == '__main__':
    main() 