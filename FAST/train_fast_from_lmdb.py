#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LMDB 기반 FAST 학습 스크립트

요구사항: fast_sample_finetune.py 전처리/하이퍼파라미터를 그대로 사용하고,
데이터 소스만 LMDB로 대체한다. public_admin_train_partly 은 학습에서 제외한다.
"""

import os
import random
import logging
import sys
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from mmcv import Config
from typing import Optional
import warnings
from tqdm import tqdm
import cv2
import math
import torch.nn.functional as F
import time
ds = None

sys.path.append('/home/mango/ocr_test/FAST')

from models import build_model
from dataset.fast.lmdb_sample_data import LMDBSample


def discover_lmdbs(base_dir: str):
    """훈련/검증 LMDB 자동 발견 (partly 제외)"""
    expected = [
        ('text_in_wild_train.lmdb', 'text_in_wild_valid.lmdb'),
        ('public_admin_train.lmdb', 'public_admin_valid.lmdb'),
        ('ocr_public_train.lmdb', 'ocr_public_valid.lmdb'),
        ('finance_logistics_train.lmdb', 'finance_logistics_valid.lmdb'),
        ('handwriting_train.lmdb', 'handwriting_valid.lmdb'),
    ]
    train_paths, valid_paths = [], []
    for tr, va in expected:
        tr_p = os.path.join(base_dir, tr)
        va_p = os.path.join(base_dir, va)
        if os.path.exists(tr_p) and 'partly' not in tr_p:
            train_paths.append(tr_p)
        if os.path.exists(va_p) and 'partly' not in va_p:
            valid_paths.append(va_p)
    return train_paths, valid_paths


def build_dataloaders(
    cfg: Config,
    lmdb_base: str,
    num_workers: int = 16,
    override_repeat_times: Optional[int] = None,
    prefetch_factor_train: Optional[int] = 4,
    pin_memory_train: bool = True,
    prefetch_factor_val: Optional[int] = 2,
    pin_memory_val: bool = True,
    use_gpu_aug: bool = False,
    val_max_samples: Optional[int] = None,
    val_num_workers_override: Optional[int] = None,
    seed_for_train_shuffle: Optional[int] = None,
):
    batch_size = cfg.data.batch_size
    img_size = cfg.data.train.img_size
    short_size = cfg.data.train.short_size
    pooling_size = cfg.data.train.pooling_size
    read_type = cfg.data.train.read_type
    repeat_times = cfg.data.train.repeat_times
    if override_repeat_times is not None:
        repeat_times = int(override_repeat_times)

    train_lmdbs, valid_lmdbs = discover_lmdbs(lmdb_base)
    if len(train_lmdbs) == 0:
        raise RuntimeError(f"학습용 LMDB가 없습니다: {lmdb_base}")

    train_datasets = [
        LMDBSample(
            lmdb_path=p,
            split='train',
            is_transform=not use_gpu_aug,
            img_size=img_size,
            short_size=short_size,
            pooling_size=pooling_size,
            read_type=read_type,
            repeat_times=repeat_times,
        )
        for p in train_lmdbs
    ]
    train_concat = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]

    if len(valid_lmdbs) > 0:
        # 검증에서도 손실 계산을 위해 gt_* 생성이 필요하므로 split='train' 으로 설정하고
        # is_transform=False 로 랜덤 증강만 비활성화한다.
        val_datasets = [
            LMDBSample(
                lmdb_path=p,
                split='train',
                is_transform=False,
                img_size=img_size,
                short_size=short_size,
                pooling_size=pooling_size,
                read_type=read_type,
                repeat_times=1,
            )
            for p in valid_lmdbs
        ]
        val_concat = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
        # 검증 샘플 수 제한 (랜덤 서브셋)
        if val_max_samples is not None:
            try:
                total = len(val_concat)
                k = int(val_max_samples)
                if k > 0 and k < total:
                    indices = random.sample(range(total), k)
                    from torch.utils.data import Subset
                    val_concat = Subset(val_concat, indices)
            except Exception:
                pass
    else:
        val_concat = None

    # 워커 초기화 훅(현재는 특별 동작 없음)
    def _worker_init_fn(_):
        return None

    train_loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=bool(pin_memory_train),
        drop_last=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=_worker_init_fn,
    )
    # 재현 가능한 셔플을 위해 generator 설정(옵션)
    if seed_for_train_shuffle is not None:
        try:
            g = torch.Generator()
            g.manual_seed(int(seed_for_train_shuffle))
            train_loader_kwargs['generator'] = g
        except Exception:
            pass
    if num_workers > 0 and prefetch_factor_train is not None:
        train_loader_kwargs['prefetch_factor'] = int(prefetch_factor_train)
    train_loader = DataLoader(train_concat, **train_loader_kwargs)
    val_loader = None
    if val_concat is not None:
        # 검증 워커 수: 명시 오버라이드 우선, 없으면 학습 워커의 절반(최소 1)
        if val_num_workers_override is not None:
            val_num_workers = max(0, int(val_num_workers_override))
        else:
            # 검증 워커 수는 과한 재생성을 막기 위해 보수적으로 제한
            val_num_workers = min(4, max(1, num_workers // 2))
        val_loader_kwargs = dict(
            batch_size=1,
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=bool(pin_memory_val),
            drop_last=False,
            persistent_workers=(val_num_workers > 0),
            worker_init_fn=_worker_init_fn,
        )
        if val_num_workers > 0 and prefetch_factor_val is not None:
            val_loader_kwargs['prefetch_factor'] = int(prefetch_factor_val)
        val_loader = DataLoader(val_concat, **val_loader_kwargs)
    return train_loader, val_loader


def _rand_uniform(low: float, high: float) -> float:
    return float(torch.empty(1).uniform_(low, high).item())


def _build_affine_matrix(angle_deg: float, do_flip: bool) -> torch.Tensor:
    """Create 2x3 affine matrix combining optional horizontal flip and rotation about image center.
    Angle in degrees, positive = counter-clockwise.
    """
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    # rotation
    rot = torch.tensor([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]], dtype=torch.float32)
    if do_flip:
        flip = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
        rot = rot @ flip
    return rot


def _warp_tensor(x: torch.Tensor, M: torch.Tensor, mode: str, out_h: int, out_w: int) -> torch.Tensor:
    """x: (C,H,W), M: (2,3) affine in normalized coords, returns (C,H,W) with size preserved before crop.
    We'll produce same H,W as input first, then crop/pad later.
    """
    c, h, w = x.shape
    M_batch = M.unsqueeze(0)
    grid = F.affine_grid(M_batch, size=(1, c, h, w), align_corners=False)
    y = F.grid_sample(x.unsqueeze(0), grid, mode=mode, padding_mode='zeros', align_corners=False)
    return y.squeeze(0)


def _random_crop_pad(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """x: (C,H,W). If H/W smaller, pad; then random crop to target size."""
    c, h, w = x.shape
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h > 0 or pad_w > 0:
        # pad evenly (bottom/right if odd)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        x = F.pad(x.unsqueeze(0), (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0.0).squeeze(0)
        h = x.shape[1]
        w = x.shape[2]
    # random crop
    if h > target_h:
        top = int(_rand_uniform(0, h - target_h + 1))
    else:
        top = 0
    if w > target_w:
        left = int(_rand_uniform(0, w - target_w + 1))
    else:
        left = 0
    return x[:, top:top + target_h, left:left + target_w]


def build_optimizer_and_scheduler(cfg: Config, model: torch.nn.Module):
    lr = cfg.train_cfg.lr
    opt_type = cfg.train_cfg.optimizer
    weight_decay = getattr(cfg.train_cfg, 'weight_decay', 0.0)
    if opt_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == 'SGD':
        momentum = getattr(cfg.train_cfg, 'momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"지원하지 않는 옵티마이저: {opt_type}")

    # 간단한 poly lr 스케줄러
    total_epochs = cfg.train_cfg.epoch
    scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_epochs, power=0.9)
    return optimizer, scheduler


def train_loop(
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    device,
    cfg: Config,
    out_dir: str,
    ds_engine=None,
    quiet: bool = False,
    use_channels_last: bool = False,
    save_every_steps: Optional[int] = None,
    val_every_steps: Optional[int] = None,
    ga_steps: int = 1,
    start_epoch: int = 0,
    initial_global_step: int = 0,
    initial_update_steps: int = 0,
    skip_micro_batches: int = 0,
):
    epochs = cfg.train_cfg.epoch
    # 요청: 매 epoch 저장
    save_interval = 1

    global_step = int(initial_global_step)
    best_val_loss = float('inf')
    for epoch in range(int(start_epoch), epochs):
        model.train()
        # 업데이트 스텝 기반 진행바
        total_updates = int(math.ceil(len(train_loader) / max(1, ga_steps)))
        pbar = tqdm(total=total_updates, desc=f"Epoch {epoch+1}/{epochs}", disable=quiet)
        # 프로파일 누적
        data_time_sum = 0.0
        h2d_time_sum = 0.0
        fwd_time_sum = 0.0
        bwd_time_sum = 0.0
        step_time_sum = 0.0
        prof_count = 0
        iter_start = time.time()
        update_steps_done = 0
        # 에폭 재개 시 초기 업데이트 스텝/프로그레스 복구
        if epoch == int(start_epoch) and int(initial_update_steps) > 0:
            try:
                pbar.update(min(int(initial_update_steps), total_updates))
                update_steps_done = int(initial_update_steps)
            except Exception:
                pass
        # 학습 루프 (재개 시 마이크로 배치 스킵 적용)
        if epoch == int(start_epoch) and int(skip_micro_batches) > 0:
            try:
                _it = iter(train_loader)
                to_skip = int(skip_micro_batches)
                for _ in range(to_skip):
                    next(_it, None)
                data_iter = enumerate(_it, start=to_skip)
                iter_start = time.time()
            except Exception:
                data_iter = enumerate(train_loader)
        else:
            data_iter = enumerate(train_loader)

        for batch_idx, batch in data_iter:
            # 데이터 로딩 시간(이전 반복 끝~현재 배치 수신까지)
            data_time = time.time() - iter_start
            t0 = time.time()
            imgs = batch['imgs'].to(device, non_blocking=True)
            if use_channels_last:
                imgs = imgs.contiguous(memory_format=torch.channels_last)
            gt_texts = batch['gt_texts'].to(device, non_blocking=True)
            gt_kernels = batch['gt_kernels'].to(device, non_blocking=True)
            training_masks = batch['training_masks'].to(device, non_blocking=True)
            gt_instances = batch['gt_instances'].to(device, non_blocking=True)
            h2d_time = time.time() - t0

            # DS FP16과 native AMP를 동시에 쓰지 않도록 입력 변환은 AMP일 때 생략
            try:
                if ds_engine is not None and getattr(ds_engine, 'fp16_enabled', False) and not getattr(train_loop, '_use_autocast', False):
                    imgs = imgs.half()
            except Exception:
                pass

            # 그래디언트 초기화: DS 미사용 시에만 누적 윈도우 시작에서 초기화
            if ds_engine is None:
                if ((batch_idx) % max(1, ga_steps)) == 0:
                    optimizer.zero_grad(set_to_none=True)
            # GPU 타이밍 이벤트
            use_cuda_timing = torch.cuda.is_available()
            if use_cuda_timing:
                e_fwd_start = torch.cuda.Event(enable_timing=True)
                e_fwd_end = torch.cuda.Event(enable_timing=True)
                e_bwd_end = torch.cuda.Event(enable_timing=True)
                e_step_end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                e_fwd_start.record()

            # autocast 사용 여부는 외부에서 컨트롤 (native AMP)
            outputs = None  # type: ignore
            if hasattr(torch, 'amp'):
                autocast_ctx = torch.amp.autocast('cuda', enabled=getattr(train_loop, '_use_autocast', False))
            else:
                class _Dummy:
                    def __enter__(self):
                        return None
                    def __exit__(self, exc_type, exc, tb):
                        return False
                autocast_ctx = _Dummy()

            with autocast_ctx:
                if ds_engine is not None:
                    outputs = ds_engine(
                        imgs,
                        gt_texts=gt_texts,
                        gt_kernels=gt_kernels,
                        training_masks=training_masks,
                        gt_instances=gt_instances,
                    )
                else:
                    outputs = model(
                        imgs,
                        gt_texts=gt_texts,
                        gt_kernels=gt_kernels,
                        training_masks=training_masks,
                        gt_instances=gt_instances,
                    )
            if use_cuda_timing:
                e_fwd_end.record()
            # 손실 구성요소 분해
            l_text = outputs['loss_text'].mean()
            l_kernel = outputs['loss_kernels'].mean()
            l_emb = outputs['loss_emb'].mean()
            loss = l_text + l_kernel + l_emb

            # NaN/Inf 손실은 즉시 스킵(업데이트 금지)
            try:
                if not torch.isfinite(loss):
                    if not quiet:
                        print("[warn] non-finite loss detected; skip step")
                    # 그래디언트 정리(DS/비-DS 모두 안전하게)
                    try:
                        if ds_engine is not None:
                            if hasattr(ds_engine, 'optimizer') and ds_engine.optimizer is not None:
                                ds_engine.optimizer.zero_grad(set_to_none=True)
                            if hasattr(ds_engine, 'zero_grad'):
                                ds_engine.zero_grad()
                        else:
                            optimizer.zero_grad(set_to_none=True)
                    except Exception:
                        try:
                            optimizer.zero_grad(set_to_none=True)
                        except Exception:
                            pass
                    # 다음 배치로
                    iter_start = time.time()
                    continue
            except Exception:
                pass
            scaler = getattr(train_loop, '_scaler', None)
            # 업데이트 경계 판단은 스텝 호출 전(pre) 시점에서 수행해야 정확
            did_update_pre = False
            if ds_engine is not None:
                try:
                    did_update_pre = bool(ds_engine.is_gradient_accumulation_boundary())
                except Exception:
                    did_update_pre = (((batch_idx + 1) % max(1, ga_steps)) == 0)
            else:
                did_update_pre = (((batch_idx + 1) % max(1, ga_steps)) == 0)
            if scaler is not None and getattr(train_loop, '_use_autocast', False):
                # native AMP 경로
                if ds_engine is None and max(1, ga_steps) > 1:
                    # 수동 누적: 손실을 평균화
                    scaled_loss = loss / float(max(1, ga_steps))
                    scaler.scale(scaled_loss).backward()
                else:
                    scaler.scale(loss).backward()
                if use_cuda_timing:
                    e_bwd_end.record()
                if ds_engine is None:
                    if did_update_pre:
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    # DS 경로에서는 engine.step을 매 마이크로배치 호출, 내부에서 경계 시에만 실제 업데이트
                    ds_engine.step()
                if use_cuda_timing:
                    e_step_end.record()
            else:
                if ds_engine is not None:
                    ds_engine.backward(loss)
                    if use_cuda_timing:
                        e_bwd_end.record()
                    ds_engine.step()
                    if use_cuda_timing:
                        e_step_end.record()
                else:
                    if max(1, ga_steps) > 1:
                        (loss / float(max(1, ga_steps))).backward()
                    else:
                        loss.backward()
                    if use_cuda_timing:
                        e_bwd_end.record()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if did_update_pre:
                        optimizer.step()
                    if use_cuda_timing:
                        e_step_end.record()
            # 진행바 출력 억제 옵션
            if not quiet:
                # 주기적으로 세부 타이밍 표시
                if use_cuda_timing:
                    torch.cuda.synchronize()
                    fwd_ms = e_fwd_start.elapsed_time(e_fwd_end)
                    bwd_ms = e_fwd_end.elapsed_time(e_bwd_end)
                    step_ms = e_bwd_end.elapsed_time(e_step_end)
                    fwd_time_sum += fwd_ms / 1000.0
                    bwd_time_sum += bwd_ms / 1000.0
                    step_time_sum += step_ms / 1000.0
                data_time_sum += data_time
                h2d_time_sum += h2d_time
                prof_count += 1
                if prof_count % 50 == 0:
                    avg = lambda s: (s / max(1, prof_count))
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'data_s': f"{avg(data_time_sum):.3f}",
                        'h2d_s': f"{avg(h2d_time_sum):.3f}",
                        'fwd_s': f"{avg(fwd_time_sum):.3f}",
                        'bwd_s': f"{avg(bwd_time_sum):.3f}",
                        'step_s': f"{avg(step_time_sum):.3f}",
                    })
                else:
                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            # 글로벌 스텝 증가 및 표시
            global_step += 1
            # 업데이트 스텝 증가 및 pbar 업데이트 (pre-judged)
            if did_update_pre:
                update_steps_done += 1
                try:
                    pbar.update(1)
                except Exception:
                    pass
            # 진행바 설명 업데이트
            try:
                pbar.set_description(f"Epoch {epoch+1}/{epochs} | micro {global_step} | step {update_steps_done}")
            except Exception:
                pass

            # 업데이트 스텝 기준 주기적 검증
            if val_loader is not None and val_every_steps is not None and val_every_steps > 0 and did_update_pre and (update_steps_done % val_every_steps == 0):
                model.train()  # FAST 손실 경로 유지
                val_loss = 0.0
                num_seen = 0
                with torch.inference_mode():
                    # 검증 진입 전 VRAM 정리
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    for batch in val_loader:
                        imgs = batch['imgs'].to(device, non_blocking=True)
                        gt_texts = batch['gt_texts'].to(device, non_blocking=True)
                        gt_kernels = batch['gt_kernels'].to(device, non_blocking=True)
                        training_masks = batch['training_masks'].to(device, non_blocking=True)
                        gt_instances = batch['gt_instances'].to(device, non_blocking=True)
                        try:
                            if ds_engine is not None and getattr(ds_engine, 'fp16_enabled', False):
                                imgs = imgs.half()
                        except Exception:
                            pass
                        # AMP 검증: 학습에서 AMP 사용 시 검증에도 동일 적용하여 메모리 절감
                        if hasattr(torch, 'amp'):
                            v_autocast = torch.amp.autocast('cuda', enabled=getattr(train_loop, '_use_autocast', False))
                        else:
                            class _Dummy:
                                def __enter__(self):
                                    return None
                                def __exit__(self, exc_type, exc, tb):
                                    return False
                            v_autocast = _Dummy()
                        with v_autocast:
                            if ds_engine is not None:
                                outputs = ds_engine(
                                    imgs,
                                    gt_texts=gt_texts,
                                    gt_kernels=gt_kernels,
                                    training_masks=training_masks,
                                    gt_instances=gt_instances,
                                )
                            else:
                                outputs = model(
                                    imgs,
                                    gt_texts=gt_texts,
                                    gt_kernels=gt_kernels,
                                    training_masks=training_masks,
                                    gt_instances=gt_instances,
                                )
                        loss_v = outputs['loss_text'].mean() + outputs['loss_kernels'].mean() + outputs['loss_emb'].mean()
                        val_loss += float(loss_v.item())
                        num_seen += 1
                        if num_seen >= max(1, getattr(cfg.data.train, 'val_max_samples', 1000)):
                            break
                # 검증 종료 후 VRAM 정리
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                if num_seen > 0:
                    val_loss /= num_seen
                    if not quiet:
                        print(f"[step {update_steps_done}] Val loss: {val_loss:.4f}")
                    # 베스트 체크포인트 저장
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        try:
                            os.makedirs(out_dir, exist_ok=True)
                            best_path = os.path.join(out_dir, 'best_checkpoint.pth')
                            torch.save({
                                'epoch': epoch + 1,
                                'global_step': global_step,
                                'val_loss': best_val_loss,
                                'state_dict': (ds_engine.module.state_dict() if ds_engine is not None else model.state_dict()),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),
                            }, best_path)
                            if not quiet:
                                print(f"Saved best: {best_path} ({best_val_loss:.4f})")
                        except Exception:
                            pass

            # 업데이트 스텝 기준 주기적 체크포인트 저장
            if save_every_steps is not None and save_every_steps > 0 and did_update_pre and (update_steps_done % save_every_steps == 0):
                try:
                    os.makedirs(out_dir, exist_ok=True)
                    step_path = os.path.join(out_dir, f'checkpoint_step_{update_steps_done}.pth')
                    torch.save({
                        'epoch': epoch + 1,
                        'global_step': global_step,
                        'update_steps': update_steps_done,
                        'state_dict': (ds_engine.module.state_dict() if ds_engine is not None else model.state_dict()),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                    }, step_path)
                    if not quiet:
                        print(f"Saved: {step_path}")
                except Exception:
                    pass
            # 다음 배치 로딩 구간 측정 기준 갱신
            iter_start = time.time()

        # 검증 (옵션) - 스텝 기반 검증이 활성화되어 있으면 에폭 말 검증은 생략해 중복/리셋처럼 보이는 현상 완화
        if val_loader is not None and not (val_every_steps is not None and val_every_steps > 0):
            model.train()  # FAST의 손실 경로를 위해 train 모드 유지
            # 에폭 단위 검증 전 캐시 정리
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            val_loss = 0.0
            with torch.inference_mode():
                for batch in tqdm(val_loader, desc="Valid", leave=False, disable=quiet):
                    imgs = batch['imgs'].to(device, non_blocking=True)
                    gt_texts = batch['gt_texts'].to(device, non_blocking=True)
                    gt_kernels = batch['gt_kernels'].to(device, non_blocking=True)
                    training_masks = batch['training_masks'].to(device, non_blocking=True)
                    gt_instances = batch['gt_instances'].to(device, non_blocking=True)
                    # 검증 고정 해상도 리샘플(동적 shape 감소) - 이미지와 마스크 모두 동일 크기로 맞춤
                    try:
                        target = getattr(cfg.data.train, 'img_size', 736)
                        if not isinstance(target, (list, tuple)):
                            target = (int(target), int(target))
                        th, tw = int(target[0]), int(target[1])
                        need_resize = imgs.dim() == 4 and (imgs.shape[-2] != th or imgs.shape[-1] != tw)
                        if need_resize:
                            imgs = torch.nn.functional.interpolate(imgs, size=(th, tw), mode='bilinear', align_corners=False)
                            def _resize_mask(t: torch.Tensor) -> torch.Tensor:
                                if t.dim() == 3:
                                    t = torch.nn.functional.interpolate(t.unsqueeze(1).float(), size=(th, tw), mode='nearest').squeeze(1)
                                    return t.to(torch.long)
                                return t
                            gt_texts = _resize_mask(gt_texts)
                            gt_kernels = _resize_mask(gt_kernels)
                            training_masks = _resize_mask(training_masks)
                            gt_instances = _resize_mask(gt_instances)
                    except Exception:
                        pass
                    if use_channels_last:
                        imgs = imgs.contiguous(memory_format=torch.channels_last)
                    try:
                        if ds_engine is not None and getattr(ds_engine, 'fp16_enabled', False):
                            imgs = imgs.half()
                    except Exception:
                        pass
                    # AMP 검증: 학습에서 AMP 사용 시 동일 적용
                    if hasattr(torch, 'amp'):
                        v_autocast = torch.amp.autocast('cuda', enabled=getattr(train_loop, '_use_autocast', False))
                    else:
                        class _Dummy:
                            def __enter__(self):
                                return None
                            def __exit__(self, exc_type, exc, tb):
                                return False
                        v_autocast = _Dummy()
                    with v_autocast:
                        if ds_engine is not None:
                            outputs = ds_engine(
                                imgs,
                                gt_texts=gt_texts,
                                gt_kernels=gt_kernels,
                                training_masks=training_masks,
                                gt_instances=gt_instances,
                            )
                        else:
                            outputs = model(
                                imgs,
                                gt_texts=gt_texts,
                                gt_kernels=gt_kernels,
                                training_masks=training_masks,
                                gt_instances=gt_instances,
                            )
                    loss = outputs['loss_text'].mean() + outputs['loss_kernels'].mean() + outputs['loss_emb'].mean()
                    val_loss += loss.item()
            val_loss /= max(1, len(val_loader))
            if not quiet:
                print(f"Val loss: {val_loss:.4f}")

        scheduler.step()

        # 체크포인트 저장
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
            ckpt = {
                'epoch': epoch + 1,
                'global_step': global_step,
                'state_dict': (ds_engine.module.state_dict() if ds_engine is not None else model.state_dict()),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(ckpt, path)
            if not quiet:
                print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='FAST 설정 파일(.py)')
    parser.add_argument('--lmdb_base', type=str, default='/mnt/nas/ocr_dataset', help='LMDB 루트 디렉토리')
    parser.add_argument('--output_dir', type=str, default='./outputs/fast_lmdb_train')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None, help='학습 재개할 체크포인트(.pth). 없으면 --checkpoint로 가중치만 로드')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--repeat_times', type=int, default=None, help='설정의 repeat_times를 일시적으로 덮어쓰기')
    # steps_per_epoch 제거: 업데이트 스텝 기반으로 표시/제어
    parser.add_argument('--save_every_steps', type=int, default=1000, help='N 스텝마다 즉시 체크포인트 저장')
    parser.add_argument('--val_every_steps', type=int, default=1000, help='N 스텝마다 간이 검증 수행')
    parser.add_argument('--ga_steps', type=int, default=1, help='gradient accumulation steps (유효 배치 확장)')
    parser.add_argument('--batch_size', type=int, default=None, help='학습 배치 크기(설정보다 우선)')
    parser.add_argument('--quiet', action='store_true', default=False, help='콘솔 로그 최소화')
    parser.add_argument('--channels_last', action='store_true', default=False, help='channels_last 메모리 포맷 사용')
    parser.add_argument('--compile', action='store_true', default=False, help='torch.compile 사용(Pytorch>=2.0)')
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead', choices=['default','reduce-overhead','max-autotune'], help='torch.compile mode')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='train DataLoader prefetch_factor')
    parser.add_argument('--val_prefetch_factor', type=int, default=2, help='val DataLoader prefetch_factor')
    parser.add_argument('--pin_memory_train', action='store_true', default=True, help='train DataLoader pin_memory')
    parser.add_argument('--no_pin_memory_train', action='store_true', default=False, help='train DataLoader pin_memory 비활성')
    parser.add_argument('--pin_memory_val', action='store_true', default=True, help='val DataLoader pin_memory')
    # intra/interop thread 강제 설정 제거(자동 최적화 권장)
    parser.add_argument('--amp', action='store_true', default=False, help='native AMP(torch.cuda.amp) 사용')
    parser.add_argument('--val_max_samples', type=int, default=None, help='검증 랜덤 샘플 수 제한')
    parser.add_argument('--val_fix_size', action='store_true', default=True, help='검증 시 입력/마스크를 고정 img_size로 리샘플링')
    parser.add_argument('--val_num_workers', type=int, default=None, help='검증 DataLoader 워커 수(미지정 시 train의 절반)')
    parser.add_argument('--seed', type=int, default=None, help='데이터/셔플 재현을 위한 시드. 재개 시 동일 시드 권장')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    # 전역 시드 고정(옵션)
    if args.seed is not None:
        try:
            torch.manual_seed(int(args.seed))
            random.seed(int(args.seed))
            os.environ['PYTHONHASHSEED'] = str(int(args.seed))
        except Exception:
            pass
    # 배치 크기 오버라이드
    if args.batch_size is not None:
        try:
            cfg.data.batch_size = int(args.batch_size)
        except Exception:
            pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg.model).to(device)
    if args.channels_last:
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception:
            pass
    # torch.compile (PyTorch>=2.0)
    try:
        if args.compile and hasattr(torch, 'compile'):
            # 그래프 브레이크 완화: 스칼라 출력 캡처 허용
            try:
                import torch._dynamo as dynamo
                dynamo.config.capture_scalar_outputs = True
                dynamo.config.suppress_errors = True
                dynamo.config.verbose = False
                os.environ.setdefault('TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS', '1')
            except Exception:
                pass
            # 동적 shape로 인한 과도한 CUDAGraph 캡처를 회피
            try:
                import torch._inductor.config as inductor_config
                if hasattr(inductor_config, 'triton'):
                    setattr(inductor_config.triton, 'cudagraphs', False)
                    setattr(inductor_config.triton, 'cudagraph_skip_dynamic_graphs', True)
                    # 경고 비활성화(Optional)
                    setattr(inductor_config.triton, 'cudagraph_dynamic_shape_warn_limit', None)
            except Exception:
                pass
            # 경고/로깅 소음 억제
            for name in [
                'torch._inductor',
                'torch._dynamo',
                'torch.fx.experimental.symbolic_shapes',
                'torch.utils._sympy',
            ]:
                try:
                    logging.getLogger(name).setLevel(logging.ERROR)
                except Exception:
                    pass
            try:
                warnings.filterwarnings('ignore', message='.*skipping cudagraphs.*')
            except Exception:
                pass
            mode_to_use = args.compile_mode if args.compile_mode else 'reduce-overhead'
            model = torch.compile(model, mode=mode_to_use)
    except Exception:
        pass

    # cudnn 최적화
    try:
        cudnn.benchmark = True
    except Exception:
        pass
    # TF32 활성화 (Ampere 이상에서 컨볼루션/매트멀 가속)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        cudnn.allow_tf32 = True
    except Exception:
        pass
    try:
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('medium')
    except Exception:
        pass

    # OpenCV 스레드 제한 해제 (시스템 기본에 따름)

    # CPU intra-op 쓰레드 제한(과다 스레드 경합 방지)
    # CPU 스레드 수는 PyTorch 기본 자동 설정 사용

    # repeat_times 확인 출력
    if not args.quiet:
        try:
            current_rt = getattr(cfg.data, 'repeat_times', None) or getattr(cfg.data.train, 'repeat_times', None)
            if args.repeat_times is not None:
                print(f"repeat_times(override): {args.repeat_times} (config: {current_rt})")
            else:
                print(f"repeat_times: {current_rt}")
        except Exception:
            pass

    # 사전학습 가중치 (옵션)
    pretrain = getattr(cfg.train_cfg, 'pretrain', None)
    if args.checkpoint:
        pretrain = args.checkpoint
    if pretrain and os.path.exists(pretrain):
        state = torch.load(pretrain, map_location=device)
        if 'state_dict' in state:
            state = state['state_dict']
        state = {k.replace('module.', ''): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing and not args.quiet:
            print(f"Missing keys: {len(missing)}")
        if unexpected and not args.quiet:
            print(f"Unexpected keys: {len(unexpected)}")

    # 학습 재개 상태
    resume_ckpt = None
    start_epoch = 0
    initial_global_step = 0
    initial_update_steps = 0
    skip_micro_batches = 0
    if args.resume and os.path.exists(args.resume):
        try:
            resume_state = torch.load(args.resume, map_location=device)
            # 모델 상태
            if 'state_dict' in resume_state:
                model.load_state_dict(resume_state['state_dict'], strict=False)
            # 에폭/스텝
            saved_epoch = int(resume_state.get('epoch', 0))
            # step 체크포인트는 epoch+1로 저장되어 있으므로, 같은 에폭 내에서 재개하려면 -1 보정
            if 'update_steps' in resume_state:
                start_epoch = max(0, saved_epoch - 1)
                initial_update_steps = int(resume_state.get('update_steps', 0))
                if int(args.ga_steps) > 0:
                    skip_micro_batches = int(initial_update_steps) * int(args.ga_steps)
            else:
                start_epoch = saved_epoch
            initial_global_step = int(resume_state.get('global_step', 0))
            resume_ckpt = resume_state
            if not args.quiet:
                print(f"Resuming: epoch={start_epoch}, global_step={initial_global_step}, update_steps={initial_update_steps}, skip_micro={skip_micro_batches}")
        except Exception as e:
            if not args.quiet:
                print(f"[warn] resume 로드 실패: {e}")

    # pin_memory 옵션 정리
    pin_memory_train = bool(args.pin_memory_train and not args.no_pin_memory_train)
    pin_memory_val = bool(args.pin_memory_val)

    train_loader, val_loader = build_dataloaders(
        cfg,
        args.lmdb_base,
        num_workers=max(0, args.num_workers),
        override_repeat_times=args.repeat_times,
        prefetch_factor_train=args.prefetch_factor,
        pin_memory_train=pin_memory_train,
        prefetch_factor_val=args.val_prefetch_factor,
        pin_memory_val=pin_memory_val,
        val_max_samples=args.val_max_samples,
        val_num_workers_override=args.val_num_workers,
        seed_for_train_shuffle=args.seed,
    )
    optimizer, scheduler = build_optimizer_and_scheduler(cfg, model)
    # 재개 시 옵티마이저/스케줄러 로드
    if resume_ckpt is not None:
        try:
            if 'optimizer' in resume_ckpt:
                optimizer.load_state_dict(resume_ckpt['optimizer'])
            if 'scheduler' in resume_ckpt:
                scheduler.load_state_dict(resume_ckpt['scheduler'])
        except Exception as e:
            if not args.quiet:
                print(f"[warn] optimizer/scheduler state 로드 실패: {e}")
    # DeepSpeed 완전 비활성화
    ds_engine = None
    # steps_per_epoch 출력 제거
    # AMP 설정 공유용 속성 부여
    if args.amp and hasattr(torch, 'amp'):
        try:
            train_loop._use_autocast = True  # type: ignore[attr-defined]
            train_loop._scaler = torch.amp.GradScaler('cuda')  # type: ignore[attr-defined]
        except Exception:
            pass

    train_loop(
        model if ds_engine is None else ds_engine.module,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        device,
        cfg,
        args.output_dir,
        ds_engine=ds_engine,
        quiet=args.quiet,
        use_channels_last=bool(args.channels_last),
        save_every_steps=int(max(0, args.save_every_steps or 0)) if args.save_every_steps else None,
        val_every_steps=int(max(0, args.val_every_steps or 0)) if args.val_every_steps else None,
        ga_steps=int(max(1, args.ga_steps)),
        start_epoch=int(start_epoch),
        initial_global_step=int(initial_global_step),
        initial_update_steps=int(initial_update_steps),
    )


if __name__ == '__main__':
    main()


