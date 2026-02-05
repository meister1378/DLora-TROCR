#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
모든 한국어 OCR 데이터셋을 train/valid로 분리하여 LMDB 생성하는 통합 스크립트
전체 데이터셋 변환 (제한 없음)
데이터셋별 전용 함수로 분리하여 유지보수성 향상
최적화된 lookup 함수 활용으로 성능 대폭 개선
"""

import os
import sys
import json
import pickle
import time
import numpy as np
import cv2
# import torch
from tqdm import tqdm
import lmdb
import random
import gc
import subprocess
from pathlib import Path
import orjson
import ijson  # 스트리밍 JSON 파싱
import orjson
# import bigjson  # 제거됨 - orjson 사용
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from tqdm.contrib.concurrent import process_map, thread_map
import uuid
import psutil
import time
from typing import Optional

# FAST 모듈 import
sys.path.append('.')
sys.path.append('FAST')  # 🚀 최적화된 lookup 함수들을 위한 경로
from dataset.fast.fast_lmdb import FAST_LMDB

# 🚀 최적화된 lookup 딕셔너리들 (pickle 방식)
optimized_lookups = {}

# bbox 디버그 출력 플래그 (전역 변수)
bbox_debug_flags = {
    'text_in_wild': False,
    'public_admin': False,
    'ocr_public': False,
    'finance_logistics': False,
    'handwriting': False
}

def _order_points_clockwise(points: np.ndarray) -> np.ndarray:
    """사각형 4점을 TL, TR, BR, BL 시계방향으로 정렬한다."""
    if points.shape != (4, 2):
        points = points.reshape(-1, 2)[:4]
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1).reshape(-1)
    tl = points[np.argmin(s)]
    br = points[np.argmax(s)]
    tr = points[np.argmin(diff)]
    bl = points[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def normalize_ic15_clockwise_flat8(bbox_flat8):
    """[x1,y1,x2,y2,x3,y3,x4,y4]을 IC15 표준 순서(TL,TR,BR,BL)로 정규화한다."""
    try:
        if not isinstance(bbox_flat8, (list, tuple)) or len(bbox_flat8) != 8:
            return bbox_flat8
        pts = np.array(bbox_flat8, dtype=np.float32).reshape(-1, 2)
        ordered = _order_points_clockwise(pts)
        return ordered.reshape(-1).astype(float).tolist()
    except Exception:
        return bbox_flat8

def load_optimized_lookup(dataset_name):
    """최적화된 lookup 딕셔너리를 pickle에서 로드 (5-10배 빠름)"""
    try:
        if dataset_name in optimized_lookups:
            return optimized_lookups[dataset_name]
        
        # 1. 압축된 pickle 파일 시도 (우선순위)
        pkl_gz_file = f"FAST/lookup_{dataset_name}.pkl.gz"
        if os.path.exists(pkl_gz_file):
            print(f"  🚀 압축된 pickle 딕셔너리 로드: {pkl_gz_file}")
            import gzip
            with gzip.open(pkl_gz_file, 'rb') as f:
                lookup_dict = pickle.load(f)
            optimized_lookups[dataset_name] = lookup_dict
            return lookup_dict
        
        # 2. 일반 pickle 파일 시도
        pkl_file = f"FAST/lookup_{dataset_name}.pkl"
        if os.path.exists(pkl_file):
            print(f"  🚀 pickle 딕셔너리 로드: {pkl_file}")
            with open(pkl_file, 'rb') as f:
                lookup_dict = pickle.load(f)
            optimized_lookups[dataset_name] = lookup_dict
            return lookup_dict
        
        # 3. 기존 Python 모듈 방식 (fallback)
        module_name = f"optimized_lookup_{dataset_name}"
        if os.path.exists(f"FAST/{module_name}.py"):
            print(f"  🐌 fallback Python 함수 로드: {module_name}")
            module = __import__(module_name)
            lookup_func = getattr(module, f"lookup_{dataset_name}")
            optimized_lookups[dataset_name] = lookup_func
            return lookup_func
        
        print(f"  ⚠️ 최적화된 lookup 파일 없음: {dataset_name} (fallback 사용)")
        return None
            
    except Exception as e:
        print(f"  ⚠️ 최적화된 lookup 로드 실패: {e} (fallback 사용)")
        return None

def scan_directory_recursive(directory, target_filename, extensions=('.jpg', '.png', '.jpeg')):
    """os.scandir을 사용한 재귀적 파일 검색 (os.walk보다 빠름)"""
    if not os.path.exists(directory):
        return None
    
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file() and entry.name == target_filename:
                    return entry.path
                elif entry.is_dir() and not entry.name.startswith('.'):
                    # 재귀적으로 하위 디렉토리 검색
                    result = scan_directory_recursive(entry.path, target_filename, extensions)
                    if result:
                        return result
    except (OSError, PermissionError):
        pass
    
    return None

def optimized_find_image_path(filename, base_path, dataset_name, fallback_cache=None):
    """최적화된 이미지 경로 찾기 (pickle 딕셔너리 우선, fallback 지원)"""
    # 1. 최적화된 lookup 딕셔너리/함수 시도
    lookup_obj = load_optimized_lookup(dataset_name)
    if lookup_obj:
        try:
            # pickle 딕셔너리인 경우 (새로운 방식)
            if isinstance(lookup_obj, dict):
                # 직접 딕셔너리 접근 (O(1), 초고속)
                if filename in lookup_obj:
                    result = lookup_obj[filename]
                    if result and os.path.exists(result):
                        return result
                
                # 확장자 추가해서 시도
                for ext in ['.png', '.jpg', '.jpeg']:
                    filename_with_ext = f"{filename}{ext}"
                    if filename_with_ext in lookup_obj:
                        result = lookup_obj[filename_with_ext]
                        if result and os.path.exists(result):
                            return result
                    
                    # 확장자 제거해서 시도
                    filename_no_ext = filename.replace(ext, '')
                    if filename_no_ext in lookup_obj:
                        result = lookup_obj[filename_no_ext]
                        if result and os.path.exists(result):
                            return result
            
            # 기존 함수인 경우 (fallback)
            elif callable(lookup_obj):
                result = lookup_obj(filename, base_path)
                if result and os.path.exists(result):
                    return result
                    
        except Exception as e:
            print(f"  ⚠️ 최적화된 lookup 실패: {e}")
    
    # 2. Fallback 캐시 사용
    if fallback_cache and filename in fallback_cache:
        return fallback_cache[filename]
    
    # 3. 마지막 fallback: os.scandir 재귀 검색 (os.walk보다 빠름)
    print(f"  🚀 Fallback os.scandir 사용: {filename}")
    return scan_directory_recursive(base_path, filename)
    
    return None

# FTP 마운트된 데이터셋 기본 경로
FTP_BASE_PATH = "/run/user/0/gvfs/ftp:host=172.30.1.226/Y:\\ocr_dataset"
# 로컬 LMDB 생성 경로
LOCAL_OUTPUT_PATH = "/mnt/nas/ocr_dataset"
# 합쳐진 JSON 파일 경로
MERGED_JSON_PATH = "/home/mango/ocr_test/FAST/json_merged"

# 인식(Recognition) 크롭을 LMDB에 동반 저장하기 위한 유틸

def _iter_recog_crops_bytes(img_bytes: bytes, gt_info: dict):
    """인식용 크롭을 메모리에서 생성하여 (jpg_bytes, label) 로 yield"""
    try:
        img_np = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if img is None:
            return
        h, w = img.shape[:2]

        bboxes = gt_info.get('bboxes', []) or []
        words = gt_info.get('words', []) or []
        count = min(len(bboxes), len(words))
        for j in range(count):
            coords = bboxes[j]
            text = words[j]
            if text is None:
                continue
            label = str(text).strip()
            if label == "" or label == "###":
                continue
            try:
                xs = coords[0::2]
                ys = coords[1::2]
                x1 = max(0, min(int(min(xs)), w - 1))
                y1 = max(0, min(int(min(ys)), h - 1))
                x2 = max(0, min(int(max(xs)), w))
                y2 = max(0, min(int(max(ys)), h))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                ok, enc = cv2.imencode('.jpg', crop)
                if not ok:
                    continue
                yield enc.tobytes(), label
            except Exception:
                continue
    except Exception:
        return

def scan_images_with_scandir(image_dir, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    """scandir을 사용한 빠른 이미지 파일 검색"""
    image_files = {}
    
    try:
        with os.scandir(image_dir) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith(extensions):
                    image_files[entry.name] = entry.path
    except Exception as e:
        print(f"⚠️ scandir 실패: {e}")
    
    return image_files

def scan_images_recursive_with_scandir(base_dir, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    """os.scandir을 사용한 재귀적 이미지 파일 검색 (os.walk 대체)"""
    image_files = {}
    
    def _scan_recursive(directory):
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.lower().endswith(extensions):
                        image_files[entry.name] = entry.path
                    elif entry.is_dir() and not entry.name.startswith('.'):
                        _scan_recursive(entry.path)
        except (OSError, PermissionError) as e:
            print(f"⚠️ 디렉토리 스캔 실패 {directory}: {e}")
    
    if os.path.exists(base_dir):
        _scan_recursive(base_dir)
    
    return image_files

def build_image_cache_parallel(base_path, dataset_type):
    """병렬로 이미지 경로 캐시 구축"""
    print(f"🔄 병렬 이미지 경로 캐시 구축 중... ({dataset_type})")
    cache = {}
    
    def scan_directory(directory):
        """디렉토리 스캔 함수"""
        local_cache = {}
        if os.path.exists(directory):
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        if entry.is_file() and entry.name.lower().endswith(('.jpg', '.png', '.jpeg')):
                            local_cache[entry.name] = entry.path
            except Exception as e:
                print(f"⚠️ 디렉토리 스캔 실패: {directory} - {e}")
        return local_cache
    
    # 스캔할 디렉토리 목록
    scan_dirs = []
    
    if dataset_type == "ocr_public":
        for split in ['Training', 'Validation']:
            scan_dirs.append(f"{base_path}/{split}/01.원천데이터")
    
    elif dataset_type == "finance_logistics":
        for split in ['Training', 'Validation']:
            scan_dirs.append(f"{base_path}/{split}/01.원천데이터")
    
    elif dataset_type == "handwriting":
        for split in ['1.Training', '2.Validation']:
            scan_dirs.append(f"{base_path}/{split}/원천데이터")
    
    elif dataset_type == "public_admin":
        for train_num in [1, 2, 3]:
            scan_dirs.append(f"{base_path}/Training/[원천]train{train_num}/02.원천데이터(jpg)")
        scan_dirs.append(f"{base_path}/Validation/[원천]validation/02.원천데이터(Jpg)")
    
    # 병렬 스캔 실행
    max_workers = min(mp.cpu_count(), 16)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_dir = {executor.submit(scan_directory, dir_path): dir_path for dir_path in scan_dirs}
        
        for future in tqdm(as_completed(future_to_dir), total=len(scan_dirs), desc="디렉토리 스캔"):
            local_cache = future.result()
            cache.update(local_cache)
    
    print(f"  ✅ 캐시 완료: {len(cache)}개 파일")
    return cache

def cleanup_memory():
    """강력한 메모리 정리"""
    # 1. 가비지 컬렉션
    collected = gc.collect()
    
    # 2. CUDA 메모리 정리 (가능한 경우)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    
    # 3. 시스템 메모리 상태 확인
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    print(f"  🧹 메모리 정리: GC {collected}개 해제, 현재 사용량: {memory_mb:.1f}MB")

def is_ftp_mounted():
    """gvfs FTP가 연결되어 있는지 확인"""
    gvfs_path = "/run/user/0/gvfs/ftp:host=172.30.1.226/Y:\ocr_dataset"
    return os.path.exists(gvfs_path)

def load_json_with_orjson(json_path):
    """JSON 파일을 orjson으로 로드하는 함수 (고속)"""
    print(f"📄 JSON 파일 로드 중: {json_path}")
    
    # 파일 크기 확인
    file_size = os.path.getsize(json_path)
    file_size_gb = file_size / (1024**3)
    print(f"  📊 파일 크기: {file_size_gb:.2f} GB")
    
    try:
        # orjson으로 로드 (고속)
        print("  🚀 orjson으로 로드 중...")
        with open(json_path, 'rb') as f:
            data = orjson.loads(f.read())
        print("  ✅ orjson 로드 성공")
        return data, None  # (data, file_handle)
        
    except MemoryError:
        print("  ⚠️ 메모리 부족 - 메모리 정리 후 재시도...")
        cleanup_memory()
        
        # 메모리 정리 후 재시도
        with open(json_path, 'rb') as f:
            data = orjson.loads(f.read())
        print("  ✅ orjson 로드 성공 (재시도)")
        return data, None
        
    except Exception as e:
        print(f"  ❌ JSON 로드 실패: {e}")
        raise

def safe_close_file(file_handle):
    """파일 핸들을 안전하게 닫는 함수"""
    if file_handle:
        try:
            file_handle.close()
        except:
            pass

# ============================================================================
# Text in the Wild 데이터셋 전용 함수
# ============================================================================

def create_text_in_wild_train_valid(max_samples=500):
    """Text in the wild train/valid LMDB 생성"""
    print("=" * 60)
    print("🧪 Text in the wild train/valid LMDB 생성")
    print("=" * 60)
    
    base_path = f"{FTP_BASE_PATH}/13.한국어글자체/04. Text in the wild_230209_add"
    json_path = f"{MERGED_JSON_PATH}/textinthewild_data_info.json"
    train_output_path = f"{LOCAL_OUTPUT_PATH}/text_in_wild_annotations_train.lmdb"
    valid_output_path = f"{LOCAL_OUTPUT_PATH}/text_in_wild_annotations_valid.lmdb"
    
    if os.path.exists(json_path):
        create_lmdb_text_in_wild_split(base_path, json_path, train_output_path, valid_output_path, 
                                     train_ratio=0.9, max_samples=max_samples, random_seed=42)
        cleanup_memory()
    else:
        print(f"❌ JSON 파일을 찾을 수 없습니다: {json_path}")

def create_lmdb_text_in_wild_split(base_path, json_path, train_output_path, valid_output_path, train_ratio=0.9, max_samples=None, random_seed=42):
    """Text in the wild LMDB 생성 (합쳐진 JSON에서 train/valid 분할)"""
    print(f"🧪 Text in the wild LMDB 생성 중... (train/valid {train_ratio}:{1-train_ratio} 분할)")
    
    random.seed(random_seed)
    os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(valid_output_path), exist_ok=True)
    
    # Text in the Wild는 작은 파일이므로 orjson으로 빠르게 처리
    print(f"📄 JSON 파일 로드 중: {json_path}")
    
    # orjson을 사용한 전체 JSON 로드 (빠른 처리)
    with open(json_path, 'rb') as f:
        data = orjson.loads(f.read())
    
    # images와 annotations 처리 (빠른 Python 리스트 사용)
    images_info = {img['id']: img for img in data.get('images', [])}
    image_annotations = {}
    for ann in data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)
    
    # JSON 데이터 즉시 해제 (메모리 절약)
    del data
    gc.collect()
    print(f"  🗑️ JSON 원본 데이터 메모리 해제 완료")
    
    # 이미지 ID 리스트를 섞어서 train/valid 분할
    img_ids = list(images_info.keys())
    
    if max_samples and len(img_ids) > max_samples:
        img_ids = img_ids[:max_samples]
        print(f"📊 {max_samples}개 샘플로 제한")
    elif max_samples is None:
        print(f"📊 전체 데이터 처리: {len(img_ids)}개 이미지")
    
    random.shuffle(img_ids)
    train_size = int(len(img_ids) * train_ratio)
    train_img_ids = img_ids[:train_size]
    valid_img_ids = img_ids[train_size:]
    
    print(f"📊 총 {len(img_ids)}개 이미지 -> Train: {len(train_img_ids)}개, Valid: {len(valid_img_ids)}개")
    
    # Training LMDB 생성
    create_lmdb_text_in_wild_from_ids(base_path, images_info, image_annotations, train_img_ids, train_output_path, "Training")
    
    # 즉시 메모리 해제
    del train_img_ids
    gc.collect()
    print(f"🗑️ Training 데이터 메모리 해제 완료")
    
    # Validation LMDB 생성
    create_lmdb_text_in_wild_from_ids(base_path, images_info, image_annotations, valid_img_ids, valid_output_path, "Validation")
    
    # 모든 데이터 메모리 해제
    del valid_img_ids
    del images_info
    del image_annotations
    gc.collect()
    print(f"🗑️ 모든 데이터 메모리 해제 완료")

# ============================================================================
# 공통 병렬 처리 함수들
# ============================================================================

def process_single_text_wild_image(args):
    """Text in Wild 단일 이미지 처리 함수 (병렬 처리용)"""
    img_id, img_info, annotations, base_path, lookup_dict = args
    
    try:
        # 파일명에 확장자 추가 (.jpg)
        img_file_name = img_info['file_name']
        if not img_file_name.endswith('.jpg'):
            img_file_name = f"{img_file_name}.jpg"
        
        # 🚀 최적화된 경로 찾기 (딕셔너리 직접 접근)
        img_path = None
        if lookup_dict and isinstance(lookup_dict, dict):
            if img_file_name in lookup_dict:
                img_path = lookup_dict[img_file_name]
            else:
                # 확장자 변형 시도
                for ext in ['.png', '.jpeg']:
                    alt_name = img_file_name.replace('.jpg', ext)
                    if alt_name in lookup_dict:
                        img_path = lookup_dict[alt_name]
                        break
        
        # fallback: 타입별 경로 로직
        if not img_path:
            img_type = img_info.get('type', 'book')
            if img_type == "book":
                image_dir = f"{base_path}/01_textinthewild_book_images_new/01_textinthewild_book_images_new/book"
            elif img_type == "sign":
                image_dir = f"{base_path}/01_textinthewild_signboard_images_new/01_textinthewild_signboard_images_new/Signboard"
            elif img_type == "traffic sign":
                image_dir = f"{base_path}/01_textinthewild_traffic_sign_images_new/01_textinthewild_traffic_sign_images_new/Traffic_Sign"
            elif img_type == "product":
                image_dir = f"{base_path}/01_textinthewild_goods_images_new/01_textinthewild_goods_images_new/Goods"
            else:
                image_dir = f"{base_path}/01_textinthewild_book_images_new/01_textinthewild_book_images_new/book"
            
            img_path = os.path.join(image_dir, img_file_name)
        
        if not img_path or not os.path.exists(img_path):
            return None
        
        # 이미지 로드
        with open(img_path, 'rb') as f:
            img_data = f.read()
        
        # 어노테이션 처리
        bboxes = []
        words = []
        
        for ann in annotations:
            # bbox: [x, y, width, height] -> [x1, y1, x2, y1, x2, y2, x1, y2]
            x, y, w, h = ann['bbox']
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            # 원본 좌표를 그대로 사용 (클리핑 없음)
            pixel_coords = [x1, y1, x2, y1, x2, y2, x1, y2]
            
            # bbox 형태 한 번만 출력
            if not bbox_debug_flags['text_in_wild']:
                print(f"📋 Text in Wild bbox 형태: 원본 [x={x}, y={y}, w={w}, h={h}] -> 통일 [x1={x1}, y1={y1}, x2={x2}, y1={y1}, x2={x2}, y2={y2}, x1={x1}, y2={y2}]")
                bbox_debug_flags['text_in_wild'] = True
            
            bboxes.append(pixel_coords)
            words.append(ann['text'])
        
        gt_info = {
            'bboxes': bboxes,
            'words': words,
            'filename': img_info['file_name']
        }
        
        return (img_id, img_data, gt_info)
        
    except Exception as e:
        return None

def process_single_public_admin_image(args):
    """공공행정문서 단일 이미지 처리 함수 (병렬 처리용)"""
    img_info, annotations, base_path, lookup_dict, dataset_lookup_name, image_path_cache = args
    
    try:
        # 파일명 추출
        img_file_name = img_info.get('image.file.name', '')
        image_category = img_info.get('image.category', '')
        image_make_code = img_info.get('image.make.code', '')
        image_make_year = img_info.get('image.make.year', '')
        
        if not img_file_name:
            return None
        
        # 이미지 경로 찾기
        img_path = optimized_find_image_path(img_file_name, base_path, dataset_lookup_name, image_path_cache)
        if not img_path or not os.path.exists(img_path):
            return None
        
        # 이미지 로드
        with open(img_path, 'rb') as f:
            img_data = f.read()
        
        # 어노테이션 처리
        bboxes = []
        words = []
        img_id = img_info.get('id')
        
        for ann in annotations:
            # annotation.bbox: [x, y, width, height] -> [x1, y1, x2, y1, x2, y2, x1, y2]
            x, y, w, h = ann['annotation.bbox']
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            # 원본 픽셀 좌표를 그대로 사용 (8개 좌표 형태로 통일)
            pixel_coords = [x1, y1, x2, y1, x2, y2, x1, y2]
            
            # bbox 형태 한 번만 출력
            if not bbox_debug_flags['public_admin']:
                print(f"📋 Public Admin bbox 형태: 원본 [x={x}, y={y}, w={w}, h={h}] -> 통일 [x1={x1}, y1={y1}, x2={x2}, y1={y1}, x2={x2}, y2={y2}, x1={x1}, y2={y2}]")
                bbox_debug_flags['public_admin'] = True
            
            bboxes.append(pixel_coords)
            words.append(ann['annotation.text'])
        
        gt_info = {
            'bboxes': bboxes,
            'words': words,
            'filename': img_file_name
        }
        
        return (img_id, img_data, gt_info)
        
    except Exception as e:
        return None

def process_single_ocr_public_image(args):
    """OCR 공공 단일 이미지 처리 함수 (병렬 처리용)"""
    img_info, annotations, base_path, dataset_lookup_name, image_path_cache = args
    
    try:
        img_file_name = img_info.get('file_name', '')
        
        # 확장자 확인
        if not img_file_name.endswith(('.jpg', '.png', '.jpeg')):
            img_file_name = f"{img_file_name}.jpg"
        
        # 이미지 경로 찾기
        img_path = optimized_find_image_path(img_file_name, base_path, dataset_lookup_name, image_path_cache)
        if not img_path or not os.path.exists(img_path):
            return None
        
        # 이미지 로드
        with open(img_path, 'rb') as f:
            img_data = f.read()
        
        # 어노테이션 처리
        bboxes = []
        words = []
        img_id = img_info.get('id')
        
        for ann in annotations:
            bbox_coords = ann['bbox']
            try:
                # 원본 bbox가 [x1, x2, x3, x4, y1, y2, y3, y4] 형태인지 확인
                if len(bbox_coords) == 8:
                    # x, y 좌표 분리
                    x_coords = bbox_coords[:4]  # [x1, x2, x3, x4]
                    y_coords = bbox_coords[4:]  # [y1, y2, y3, y4]
                    
                    # 올바른 순서로 재구성: [x1, y1, x2, y2, x3, y3, x4, y4]
                    pixel_coords = []
                    for i in range(4):
                        pixel_coords.extend([x_coords[i], y_coords[i]])
                    # IC15 표준 시계방향으로 정규화
                    pixel_coords = normalize_ic15_clockwise_flat8(pixel_coords)
                    
                    # bbox 형태 한 번만 출력
                    if not bbox_debug_flags['ocr_public']:
                        print(f"📋 OCR Public bbox 형태: 원본 [x1={x_coords[0]}, x2={x_coords[1]}, x3={x_coords[2]}, x4={x_coords[3]}, y1={y_coords[0]}, y2={y_coords[1]}, y3={y_coords[2]}, y4={y_coords[3]}] -> 수정 [x1={pixel_coords[0]}, y1={pixel_coords[1]}, x2={pixel_coords[2]}, y2={pixel_coords[3]}, x3={pixel_coords[4]}, y3={pixel_coords[5]}, x4={pixel_coords[6]}, y4={pixel_coords[7]}]")
                        bbox_debug_flags['ocr_public'] = True
                    
                    bboxes.append(pixel_coords)
                    words.append(ann['text'])
                else:
                    # 기존 로직 (8개가 아닌 경우)
                    x_coords = [bbox_coords[0], bbox_coords[2], bbox_coords[4], bbox_coords[6]]
                    y_coords = [bbox_coords[1], bbox_coords[3], bbox_coords[5], bbox_coords[7]]
                    
                    # 원본 픽셀 좌표를 그대로 사용
                    pixel_coords = [
                        x_coords[0], y_coords[0],
                        x_coords[1], y_coords[1],
                        x_coords[2], y_coords[2],
                        x_coords[3], y_coords[3]
                    ]
                    # IC15 표준 시계방향으로 정규화
                    pixel_coords = normalize_ic15_clockwise_flat8(pixel_coords)
                    
                    bboxes.append(pixel_coords)
                    words.append(ann['text'])
            except (IndexError, TypeError):
                try:
                    # 4개 좌표인지 확인 (x, y, w, h)
                    x, y, w, h = bbox_coords[0], bbox_coords[1], bbox_coords[2], bbox_coords[3]
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    
                    # 원본 좌표를 그대로 사용 (클리핑 없음)
                    pixel_coords = [x1, y1, x2, y1, x2, y2, x1, y2]
                    
                    # bbox 형태 한 번만 출력
                    if not bbox_debug_flags['ocr_public']:
                        print(f"📋 OCR Public bbox 형태: 원본 [x={x}, y={y}, w={w}, h={h}] -> 통일 [x1={x1}, y1={y1}, x2={x2}, y1={y1}, x2={x2}, y2={y2}, x1={x1}, y2={y2}]")
                        bbox_debug_flags['ocr_public'] = True
                    
                    bboxes.append(pixel_coords)
                    words.append(ann['text'])
                except (IndexError, TypeError):
                    continue
        
        gt_info = {
            'bboxes': bboxes,
            'words': words,
            'filename': img_file_name
        }
        
        return (img_id, img_data, gt_info)
        
    except Exception as e:
        return None

def process_single_finance_logistics_image(args):
    """금융물류 단일 이미지 처리 함수 (병렬 처리용)"""
    sub_dataset, img_info_data, annotations_for_dataset = args
    
    if not annotations_for_dataset:
        return None
        
    try:
        # 이미지 로드
        with open(img_info_data['file_path'], 'rb') as f:
            img_data = f.read()
        
        # 어노테이션 처리 (기존 로직 그대로)
        bboxes = []
        words = []
        img_w = img_info_data['width']
        img_h = img_info_data['height']
        
        for ann in annotations_for_dataset:
            bbox_coords = ann.get('bbox', [])
            
            try:
                # 🚀 bigjson Array를 안전하게 Python list로 변환
                if hasattr(bbox_coords, '__getitem__') and not isinstance(bbox_coords, list):
                    # bigjson Array인 경우 Python list로 변환
                    bbox_list = []
                    try:
                        for i in range(8):  # 최대 8개까지 시도
                            bbox_list.append(bbox_coords[i])
                    except (IndexError, TypeError):
                        # 8개보다 적으면 4개 시도
                        try:
                            bbox_list = []
                            for i in range(4):
                                bbox_list.append(bbox_coords[i])
                        except (IndexError, TypeError):
                            continue
                    bbox_coords = bbox_list
                
                # 8개 좌표인지 확인 (4개 점의 x,y)
                if len(bbox_coords) >= 8:
                    # merged JSON에서 [x1,x2,x3,x4,y1,y2,y3,y4] 형태를 [x1,y1,x2,y2,x3,y3,x4,y4]로 변환
                    x_coords = bbox_coords[:4]  # [x1, x2, x3, x4]
                    y_coords = bbox_coords[4:]  # [y1, y2, y3, y4]
                    
                    # 올바른 순서로 재구성: [x1, y1, x2, y2, x3, y3, x4, y4]
                    pixel_coords = []
                    for i in range(4):
                        pixel_coords.extend([x_coords[i], y_coords[i]])
                    # IC15 표준 시계방향으로 정규화
                    pixel_coords = normalize_ic15_clockwise_flat8(pixel_coords)
                    
                    # bbox 형태 한 번만 출력
                    if not bbox_debug_flags['finance_logistics']:
                        print(f"📋 Finance Logistics bbox 형태: 원본 [x1={x_coords[0]}, x2={x_coords[1]}, x3={x_coords[2]}, x4={x_coords[3]}, y1={y_coords[0]}, y2={y_coords[1]}, y3={y_coords[2]}, y4={y_coords[3]}] -> 수정 [x1={pixel_coords[0]}, y1={pixel_coords[1]}, x2={pixel_coords[2]}, y2={pixel_coords[3]}, x3={pixel_coords[4]}, y3={pixel_coords[5]}, x4={pixel_coords[6]}, y4={pixel_coords[7]}]")
                        bbox_debug_flags['finance_logistics'] = True
                    
                    bboxes.append(pixel_coords)
                    words.append(ann.get('text', ''))
                elif len(bbox_coords) >= 4:
                    # 4개 좌표인지 확인 (x, y, w, h)
                    x, y, w, h = bbox_coords[0], bbox_coords[1], bbox_coords[2], bbox_coords[3]
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    
                    # 원본 좌표를 그대로 사용 (클리핑 없음)
                    pixel_coords = [x1, y1, x2, y1, x2, y2, x1, y2]
                    # IC15 표준 시계방향으로 정규화
                    pixel_coords = normalize_ic15_clockwise_flat8(pixel_coords)
                    
                    # bbox 형태 한 번만 출력
                    if not bbox_debug_flags['finance_logistics']:
                        print(f"📋 Finance Logistics bbox 형태: 원본 [x={x}, y={y}, w={w}, h={h}] -> 통일 [x1={x1}, y1={y1}, x2={x2}, y1={y1}, x2={x2}, y2={y2}, x1={x1}, y2={y2}]")
                        bbox_debug_flags['finance_logistics'] = True
                    
                    bboxes.append(pixel_coords)
                    words.append(ann.get('text', ''))
            except (IndexError, TypeError, ValueError):
                # bbox가 잘못된 형식이면 건너뛰기
                continue
        
        gt_info = {
            'bboxes': bboxes,
            'words': words,
            'filename': img_info_data['filename']
        }
        
        return (sub_dataset, img_data, gt_info)
        
    except Exception as e:
        return None

def process_single_handwriting_image(args):
    """손글씨 단일 이미지 처리 함수 (병렬 처리용)
    args 형태:
      - (img_file_name, img_info_data) (이전 호환)
      - (img_file_name, img_info_data, annotations_for_image) (신규)
    """
    if len(args) == 3:
        img_file_name, img_info_data, annotations_for_image = args
    else:
        img_file_name, img_info_data = args
        annotations_for_image = []

    try:
        img_path = img_info_data['file_path']
        if not os.path.exists(img_path):
            return None

        # 이미지 로드
        with open(img_path, 'rb') as f:
            img_data = f.read()

        bboxes = []
        words = []

        # 1) 우선 merged JSON의 annotations 사용 (있을 경우)
        if annotations_for_image:
            for ann in annotations_for_image:
                bbox_coords = ann.get('bbox', [])
                try:
                    if isinstance(bbox_coords, list) and len(bbox_coords) >= 8:
                        # [x1,x2,x3,x4,y1,y2,y3,y4] -> interleave -> normalize
                        x_coords = bbox_coords[:4]
                        y_coords = bbox_coords[4:]
                        pixel_coords = []
                        for i in range(4):
                            pixel_coords.extend([x_coords[i], y_coords[i]])
                        pixel_coords = normalize_ic15_clockwise_flat8(pixel_coords)
                        if not bbox_debug_flags['handwriting']:
                            print(f"📋 Handwriting bbox(merged) 형태: x={x_coords}, y={y_coords} -> {pixel_coords}")
                            bbox_debug_flags['handwriting'] = True
                        bboxes.append(pixel_coords)
                        words.append(ann.get('text', ''))
                    elif isinstance(bbox_coords, list) and len(bbox_coords) >= 4:
                        # [x,y,w,h]
                        x, y, w, h = bbox_coords[0], bbox_coords[1], bbox_coords[2], bbox_coords[3]
                        x1, y1, x2, y2 = x, y, x + w, y + h
                        pixel_coords = [x1, y1, x2, y1, x2, y2, x1, y2]
                        pixel_coords = normalize_ic15_clockwise_flat8(pixel_coords)
                        bboxes.append(pixel_coords)
                        words.append(ann.get('text', ''))
                except Exception:
                    continue

        # 2) fallback: original_json_path에서 직접 읽기
        if not bboxes:
            original_json_path = img_info_data.get("original_json_path", "")
            # 경로가 절대 경로가 아니면 base_path와 합치는 처리는 상위에서 보장
            if original_json_path and os.path.exists(original_json_path):
                try:
                    json_data, json_file_handle = load_json_with_orjson(original_json_path)
                    try:
                        if 'bbox' in json_data:
                            for bbox_info in json_data['bbox']:
                                x_coords = bbox_info.get('x')
                                y_coords = bbox_info.get('y')
                                if x_coords is None or y_coords is None:
                                    continue
                                pixel_coords = []
                                for i in range(4):
                                    pixel_coords.extend([x_coords[i], y_coords[i]])
                                pixel_coords = normalize_ic15_clockwise_flat8(pixel_coords)
                                if not bbox_debug_flags['handwriting']:
                                    print(f"📋 Handwriting bbox(orig) 형태: x={x_coords}, y={y_coords} -> {pixel_coords}")
                                    bbox_debug_flags['handwriting'] = True
                                bboxes.append(pixel_coords)
                                words.append(bbox_info.get('data', ''))
                    finally:
                        safe_close_file(json_file_handle)
                except Exception:
                    pass

        gt_info = {
            'bboxes': bboxes,
            'words': words,
            'filename': img_info_data['filename']
        }
        return (img_file_name, img_data, gt_info)
    except Exception:
        return None

def create_parallel_lmdb_from_args(process_args, output_path, split_name, process_func, max_workers=None):
    """공통 병렬 LMDB 생성 함수 (메모리 절약형)"""
    print(f"🚀 {split_name} 병렬 LMDB 생성 중... ({len(process_args)}개 샘플)")
    
    # CPU 코어 수에 따른 최적 워커 수
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 16)  # 워커 수를 16개로 증가
    print(f"  🔧 병렬 워커 수: {max_workers}개")
    
    # LMDB 환경 생성 (메모리 최적화 설정)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    env = lmdb.open(output_path, 
                    map_size=1099511627776,  # 1TB
                    writemap=True,  # 메모리 매핑 최적화
                    meminit=False,  # 메모리 초기화 비활성화
                    map_async=True)  # 비동기 맵핑
    
    print(f"  🔄 병렬 처리 + 즉시 저장 시작...")
    
    idx = 0
    recog_saved_total = 0
    start_time = time.time()
    
    # 청크 단위로 스트리밍 처리
    chunk_size = 10000  # 10000개씩 청크로 나누어 처리
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # process_args를 청크 단위로 순회
        for chunk_start in tqdm(range(0, len(process_args), chunk_size), desc=f"{split_name} 청크 처리"):
            chunk_end = min(chunk_start + chunk_size, len(process_args))
            chunk_args = process_args[chunk_start:chunk_end]
            
            # 현재 청크의 future만 생성
            futures = {executor.submit(process_func, arg) for arg in chunk_args}
            
            # 더 작은 트랜잭션 단위로 분할 (메모리 누적 방지)
            txn_batch_size = 500  # 500개씩 트랜잭션 분할 (더 작게)
            batch_count = 0
            txn = None
            
            # 현재 청크의 작업만 처리
            for future in as_completed(futures):
                result = future.result()
                
                if result is not None:
                    img_id, img_data, gt_info = result
                    
                    # 새 트랜잭션 시작 (배치 단위)
                    if batch_count % txn_batch_size == 0:
                        if txn is not None:
                            txn.commit()  # 이전 트랜잭션 커밋
                        txn = env.begin(write=True)  # 새 트랜잭션 시작
                    
                    # Detection 원본 미저장, word-level만 저장
                    try:
                        for crop_jpg, label in _iter_recog_crops_bytes(img_data, gt_info):
                            img_key = f'image-{idx:09d}'.encode()
                            lab_key = f'label-{idx:09d}'.encode()
                            txn.put(img_key, crop_jpg)
                            txn.put(lab_key, label.encode('utf-8', errors='ignore'))
                            idx += 1
                            batch_count += 1
                        
                    except Exception:
                        pass
                    
                    # 즉시 메모리 해제
                    del result
                    del img_data
                    del gt_info
            
            # 마지막 트랜잭션 커밋
            if txn is not None:
                txn.commit()
            del chunk_args, futures
            
            # 강제 가비지 컬렉션
            collected = gc.collect()
            print(f"  🗑️ 청크 {chunk_start//chunk_size + 1} 완료: {idx}개 처리, GC {collected}개 해제")
        
        # 마지막에 샘플 수 저장 (word-level 샘플 수)
        txn = env.begin(write=True)
        txn.put('num-samples'.encode(), str(idx).encode())
        txn.commit()
        try:
            env.sync()
        except Exception:
            pass
    
    env.close()
    
    # 최종 메모리 해제
    del process_args
    gc.collect()
    
    total_time = time.time() - start_time
    speed = idx / total_time if total_time > 0 else 0
    print(f"✅ {split_name} 병렬 LMDB 생성 완료: {idx}개 크롭 샘플")
    print(f"   ⏱️ 총 소요 시간: {total_time:.2f}초")
    print(f"   🚀 처리 속도: {speed:.1f} samples/sec")
    print(f"🗑️ {split_name} 모든 메모리 해제 완료")


# ============================================================================
# JSONL 저장 유틸리티 (ERNIE SFT 포맷)
# ============================================================================

def _build_ocr_jsonl_record(image_path: str, words: list[str]) -> dict:
    """ERNIE SFT VL 포맷의 한 레코드를 생성한다.

    - image_info: 로컬 이미지 경로를 그대로 사용
    - text_info: [mask("OCR:"), no_mask(정답 텍스트)]
    """
    # 단순 결합(공백 구분). 필요시 줄바꿈 규칙으로 바꿔도 됨
    target_text = " ".join([w for w in (words or []) if isinstance(w, str) and w.strip()])
    return {
        "image_info": [
            {"matched_text_index": 0, "image_url": image_path},
        ],
        "text_info": [
            {"text": "OCR:", "tag": "mask"},
            {"text": target_text, "tag": "no_mask"},
        ],
    }


def _save_crops_and_make_records(image_path: str, bboxes: list[list[float]], words: list[str], crop_dir: str, prefix: str, mirror_root: Optional[str] = None) -> list[dict]:
    """원본 이미지에서 bboxes로 크롭을 저장하고 JSONL 레코드 리스트를 반환한다.
    
    mirror_root가 주어지면, image_path의 부모 디렉토리에서 mirror_root에 대한 상대 경로를 계산하여
    crop_dir 아래에 동일한 폴더 트리로 분산 저장한다. (기존 경로처럼 여러 폴더)
    mirror_root가 없으면 image_path의 상위 디렉토리 마지막 2단계를 사용해 분산 저장한다.
    """
    try:
        # 분산 저장 대상 디렉토리 결정
        dst_dir = crop_dir
        try:
            parent_dir = os.path.dirname(image_path)
            if mirror_root and os.path.exists(mirror_root):
                # base_path 기준으로 상대 경로 트리를 유지
                rel_dir = os.path.relpath(parent_dir, mirror_root)
                # 너무 상위(../)로 올라가는 경우 방지
                if not rel_dir.startswith(".."):
                    dst_dir = os.path.join(crop_dir, rel_dir)
                else:
                    # mirror_root와 무관하면 마지막 2단계만 유지
                    parts = Path(parent_dir).parts
                    tail_parts = parts[-2:] if len(parts) >= 2 else parts
                    dst_dir = os.path.join(crop_dir, *tail_parts)
            else:
                # 기본: 마지막 2단계 폴더 구성
                parts = Path(parent_dir).parts
                tail_parts = parts[-2:] if len(parts) >= 2 else parts
                dst_dir = os.path.join(crop_dir, *tail_parts)
        except Exception:
            dst_dir = crop_dir

        os.makedirs(dst_dir, exist_ok=True)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            return []
        h, w = img.shape[:2]

        records: list[dict] = []
        count = min(len(bboxes or []), len(words or []))
        stem = Path(image_path).stem
        for i in range(count):
            coords = bboxes[i]
            label = str(words[i] if words[i] is not None else "").strip()
            if not label or label == "###":
                continue
            try:
                xs = [coords[0], coords[2], coords[4], coords[6]]
                ys = [coords[1], coords[3], coords[5], coords[7]]
                x1 = max(0, min(int(min(xs)), w - 1))
                y1 = max(0, min(int(min(ys)), h - 1))
                x2 = max(0, min(int(max(xs)), w))
                y2 = max(0, min(int(max(ys)), h))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                # 고유 파일명 생성
                uniq = uuid.uuid4().hex[:8]
                out_path = os.path.join(dst_dir, f"{prefix}_{stem}_{i:06d}_{uniq}.jpg")
                ok = cv2.imwrite(out_path, crop)
                if not ok:
                    continue
                records.append(_build_ocr_jsonl_record(out_path, [label]))
            except Exception:
                continue
        return records
    except Exception:
        return []


def create_parallel_jsonl_from_args(process_args, output_path, split_name, to_json_func, max_workers=None, max_total_samples: Optional[int] = None):
    """공통 병렬 JSONL 생성 함수.

    process_args: 작업 인자 리스트
    to_json_func: 각 인자에서 (image_path, words) 또는 dict(JSON 직렬화 가능) 반환
    """
    print(f"🚀 {split_name} 병렬 JSONL 생성 중... ({len(process_args)}개 샘플)")

    # 워커/청크 환경 설정
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 16)
    chunk_size_env = os.getenv("FAST_JSONL_CHUNK_SIZE")
    try:
        chunk_size = int(chunk_size_env) if chunk_size_env else 10000
    except ValueError:
        chunk_size = 10000
    process_chunksize_env = os.getenv("FAST_JSONL_PROCESS_CHUNKSIZE")
    try:
        process_chunksize = int(process_chunksize_env) if process_chunksize_env else 32
    except ValueError:
        process_chunksize = 32
    print(f"  🔧 병렬 워커 수: {max_workers}개, 청크 크기: {chunk_size}, 프로세스 청크: {process_chunksize}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 반환 객체를 일관된 레코드 리스트로 정규화
    def _normalize_to_records(item) -> list[dict]:
        records: list[dict] = []
        if item is None:
            return records
        # dict 단건
        if isinstance(item, dict):
            return [item]
        # 리스트/튜플 묶음
        if isinstance(item, (list, tuple)) and item:
            # 요소가 dict/tuple/list
            first = item[0]
            if isinstance(first, dict):
                # 이미 dict 리스트
                for elem in item:
                    if isinstance(elem, dict):
                        records.append(elem)
                return records
            # (path, words)들의 리스트
            for elem in item:
                try:
                    pth, wrds = elem
                    records.append(_build_ocr_jsonl_record(pth, wrds))
                except Exception:
                    continue
            return records
        # 단일 (path, words)
        try:
            pth, wrds = item
            return [_build_ocr_jsonl_record(pth, wrds)]
        except Exception:
            return []

    total_written = 0
    start_time = time.time()

    with open(output_path, "w", encoding="utf-8") as fout:
        # 청크 스트리밍: 각 청크에서 결과를 즉시 소비하며 기록 (대용량 메모리 누적 방지)
        for chunk_start in tqdm(range(0, len(process_args), chunk_size), desc=f"{split_name} 청크 처리"):
            chunk_end = min(chunk_start + chunk_size, len(process_args))
            chunk_args = process_args[chunk_start:chunk_end]

            used_thread_fallback = False
            try:
                # 프로세스 풀을 직접 사용하여 imap_unordered로 스트리밍 소비
                ctx = mp.get_context("spawn")
                with ctx.Pool(processes=max_workers) as pool:
                    iterator = pool.imap_unordered(to_json_func, chunk_args, chunksize=process_chunksize)
                    for item in tqdm(iterator, total=len(chunk_args), desc=f"{split_name} 변환"):
                        if max_total_samples is not None and total_written >= max_total_samples:
                            break
                        for record in _normalize_to_records(item):
                            if max_total_samples is not None and total_written >= max_total_samples:
                                break
                            line = orjson.dumps(record).decode("utf-8")
                            fout.write(line + "\n")
                            total_written += 1
                    pool.close()
                    pool.join()
            except Exception:
                # 폴백: 스레드 풀로 스트리밍 소비
                used_thread_fallback = True
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(to_json_func, arg) for arg in chunk_args]
                    for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{split_name} 변환(thread)"):
                        if max_total_samples is not None and total_written >= max_total_samples:
                            break
                        item = fut.result()
                        for record in _normalize_to_records(item):
                            if max_total_samples is not None and total_written >= max_total_samples:
                                break
                            line = orjson.dumps(record).decode("utf-8")
                            fout.write(line + "\n")
                            total_written += 1

            # 파일 및 메모리 정리
            fout.flush()
            try:
                os.fsync(fout.fileno())
            except Exception:
                pass
            # 강제 가비지 컬렉션 및 glibc trim (리눅스)
            collected = gc.collect()
            try:
                if os.name == "posix" and os.getenv("FAST_JSONL_TRIM", "1") == "1":
                    import ctypes, ctypes.util  # 지연 import
                    libc = ctypes.CDLL(ctypes.util.find_library("c"))
                    libc.malloc_trim(0)
            except Exception:
                pass
            # RSS 출력(디버그)
            try:
                process = psutil.Process()
                rss_mb = process.memory_info().rss / 1024 / 1024
                print(f"  🗑️ 청크 {(chunk_start // chunk_size) + 1} 완료: 누적 {total_written}개, GC {collected}개 해제, RSS {rss_mb:.1f}MB, {'thread_fallback' if used_thread_fallback else 'proc'}")
            except Exception:
                print(f"  🗑️ 청크 {(chunk_start // chunk_size) + 1} 완료: 누적 {total_written}개, GC {collected}개 해제")

            if max_total_samples is not None and total_written >= max_total_samples:
                break

    total_time = time.time() - start_time
    speed = total_written / total_time if total_time > 0 else 0
    print(f"✅ {split_name} JSONL 생성 완료: {total_written}개 샘플")
    print(f"   ⏱️ 총 소요 시간: {total_time:.2f}초")
    print(f"   🚀 처리 속도: {speed:.1f} samples/sec")


# ----------------------------------------------------------------------------
# per-dataset JSONL line 생성기 (이미지 경로 + 단어 리스트)
# ----------------------------------------------------------------------------

def text_in_wild_to_jsonl(args):
    """(img_id, img_info, annotations, base_path, lookup_dict, crop_dir) → [records]"""
    img_id, img_info, annotations, base_path, lookup_dict, crop_dir = args
    try:
        img_file_name = img_info.get("file_name")
        if not img_file_name.endswith(".jpg"):
            img_file_name = f"{img_file_name}.jpg"

        # 경로 찾기 (lookup → fallback 경로 규칙)
        img_path = None
        if lookup_dict and isinstance(lookup_dict, dict):
            img_path = lookup_dict.get(img_file_name)
            if not img_path:
                for ext in [".png", ".jpeg"]:
                    alt = img_file_name.replace(".jpg", ext)
                    if alt in lookup_dict:
                        img_path = lookup_dict[alt]
                        break
        if not img_path:
            img_type = img_info.get("type", "book")
            if img_type == "book":
                image_dir = f"{base_path}/01_textinthewild_book_images_new/01_textinthewild_book_images_new/book"
            elif img_type == "sign":
                image_dir = f"{base_path}/01_textinthewild_signboard_images_new/01_textinthewild_signboard_images_new/Signboard"
            elif img_type == "traffic sign":
                image_dir = f"{base_path}/01_textinthewild_traffic_sign_images_new/01_textinthewild_traffic_sign_images_new/Traffic_Sign"
            elif img_type == "product":
                image_dir = f"{base_path}/01_textinthewild_goods_images_new/01_textinthewild_goods_images_new/Goods"
            else:
                image_dir = f"{base_path}/01_textinthewild_book_images_new/01_textinthewild_book_images_new/book"
            img_path = os.path.join(image_dir, img_file_name)
        if not img_path or not os.path.exists(img_path):
            return None

        # bboxes (x,y,w,h) → 8좌표, words
        bboxes = []
        words = []
        for ann in annotations:
            x, y, w_box, h_box = ann['bbox']
            x1, y1, x2, y2 = x, y, x + w_box, y + h_box
            pixel_coords = [x1, y1, x2, y1, x2, y2, x1, y2]
            bboxes.append(pixel_coords)
            words.append(ann.get('text', ''))

        return _save_crops_and_make_records(img_path, bboxes, words, crop_dir, prefix="tiw", mirror_root=base_path)
    except Exception:
        return None


def public_admin_to_jsonl(args):
    """(img_info, annotations, base_path, lookup_dict, dataset_lookup_name, image_path_cache, crop_dir)
       → [records]
    """
    img_info, annotations, base_path, lookup_dict, dataset_lookup_name, image_path_cache, crop_dir = args
    try:
        img_file_name = img_info.get("image.file.name", "")
        if not img_file_name:
            return None
        img_path = optimized_find_image_path(img_file_name, base_path, dataset_lookup_name, image_path_cache)
        if not img_path or not os.path.exists(img_path):
            return None
        bboxes = []
        words = []
        for ann in annotations:
            x, y, w_box, h_box = ann['annotation.bbox']
            x1, y1, x2, y2 = x, y, x + w_box, y + h_box
            pixel_coords = [x1, y1, x2, y1, x2, y2, x1, y2]
            bboxes.append(pixel_coords)
            words.append(ann.get('annotation.text', ''))
        return _save_crops_and_make_records(img_path, bboxes, words, crop_dir, prefix="pa", mirror_root=base_path)
    except Exception:
        return None


def ocr_public_to_jsonl(args):
    """(img_info, annotations, base_path, dataset_lookup_name, image_path_cache, crop_dir) → [records]"""
    img_info, annotations, base_path, dataset_lookup_name, image_path_cache, crop_dir = args
    try:
        img_file_name = img_info.get("file_name", "")
        if not img_file_name.endswith((".jpg", ".png", ".jpeg")):
            img_file_name = f"{img_file_name}.jpg"
        img_path = optimized_find_image_path(img_file_name, base_path, dataset_lookup_name, image_path_cache)
        if not img_path or not os.path.exists(img_path):
            return None
        bboxes = []
        words = []
        for ann in annotations:
            bbox_coords = ann.get('bbox', [])
            try:
                if len(bbox_coords) == 8:
                    x_coords = bbox_coords[:4]
                    y_coords = bbox_coords[4:]
                    pixel_coords = []
                    for i in range(4):
                        pixel_coords.extend([x_coords[i], y_coords[i]])
                    pixel_coords = normalize_ic15_clockwise_flat8(pixel_coords)
                else:
                    x_coords = [bbox_coords[0], bbox_coords[2], bbox_coords[4], bbox_coords[6]]
                    y_coords = [bbox_coords[1], bbox_coords[3], bbox_coords[5], bbox_coords[7]]
                    pixel_coords = [
                        x_coords[0], y_coords[0],
                        x_coords[1], y_coords[1],
                        x_coords[2], y_coords[2],
                        x_coords[3], y_coords[3],
                    ]
                    pixel_coords = normalize_ic15_clockwise_flat8(pixel_coords)
                bboxes.append(pixel_coords)
                words.append(ann.get('text', ''))
            except (IndexError, TypeError):
                try:
                    x, y, w_box, h_box = bbox_coords[0], bbox_coords[1], bbox_coords[2], bbox_coords[3]
                    x1, y1, x2, y2 = x, y, x + w_box, y + h_box
                    pixel_coords = [x1, y1, x2, y1, x2, y2, x1, y2]
                    pixel_coords = normalize_ic15_clockwise_flat8(pixel_coords)
                    bboxes.append(pixel_coords)
                    words.append(ann.get('text', ''))
                except Exception:
                    continue
        return _save_crops_and_make_records(img_path, bboxes, words, crop_dir, prefix="ocrp", mirror_root=base_path)
    except Exception:
        return None


def finance_logistics_to_jsonl(args):
    """(sub_dataset, img_info_data, annotations_for_dataset, crop_dir) → [records]"""
    sub_dataset, img_info_data, annotations_for_dataset, crop_dir = args
    try:
        img_path = img_info_data.get("file_path")
        if not img_path or not os.path.exists(img_path):
            return None
        bboxes = []
        words = []
        for ann in (annotations_for_dataset or []):
            bbox_coords = ann.get('bbox', [])
            try:
                if isinstance(bbox_coords, list) and len(bbox_coords) >= 8:
                    x_coords = bbox_coords[:4]
                    y_coords = bbox_coords[4:]
                    pixel_coords = []
                    for i in range(4):
                        pixel_coords.extend([x_coords[i], y_coords[i]])
                    pixel_coords = normalize_ic15_clockwise_flat8(pixel_coords)
                elif isinstance(bbox_coords, list) and len(bbox_coords) >= 4:
                    x, y, w_box, h_box = bbox_coords[0], bbox_coords[1], bbox_coords[2], bbox_coords[3]
                    x1, y1, x2, y2 = x, y, x + w_box, y + h_box
                    pixel_coords = [x1, y1, x2, y1, x2, y2, x1, y2]
                    pixel_coords = normalize_ic15_clockwise_flat8(pixel_coords)
                else:
                    continue
                bboxes.append(pixel_coords)
                words.append(ann.get('text', ''))
            except Exception:
                continue
        return _save_crops_and_make_records(img_path, bboxes, words, crop_dir, prefix="fin")
    except Exception:
        return None


def handwriting_to_jsonl(args):
    """(img_file_name, img_info_data, annotations_for_image?, crop_dir) → [records]"""
    if len(args) == 3:
        img_file_name, img_info_data, annotations_for_image = args
        crop_dir = None
    else:
        img_file_name, img_info_data, annotations_for_image, crop_dir = args
    try:
        img_path = img_info_data.get("file_path")
        if not img_path or not os.path.exists(img_path):
            return None
        bboxes = []
        words = []
        if annotations_for_image:
            for ann in annotations_for_image:
                bbox_coords = ann.get('bbox', [])
                try:
                    if isinstance(bbox_coords, list) and len(bbox_coords) >= 8:
                        x_coords = bbox_coords[:4]
                        y_coords = bbox_coords[4:]
                        pixel_coords = []
                        for i in range(4):
                            pixel_coords.extend([x_coords[i], y_coords[i]])
                        pixel_coords = normalize_ic15_clockwise_flat8(pixel_coords)
                    elif isinstance(bbox_coords, list) and len(bbox_coords) >= 4:
                        x, y, w_box, h_box = bbox_coords[0], bbox_coords[1], bbox_coords[2], bbox_coords[3]
                        x1, y1, x2, y2 = x, y, x + w_box, y + h_box
                        pixel_coords = [x1, y1, x2, y1, x2, y2, x1, y2]
                        pixel_coords = normalize_ic15_clockwise_flat8(pixel_coords)
                    else:
                        continue
                    bboxes.append(pixel_coords)
                    words.append(ann.get('text', ''))
                except Exception:
                    continue
        return _save_crops_and_make_records(img_path, bboxes, words, crop_dir, prefix="hw")
    except Exception:
        return None


# ----------------------------------------------------------------------------
# per-dataset JSONL 오케스트레이션 함수
# ----------------------------------------------------------------------------

def create_jsonl_text_in_wild_split(base_path, json_path, train_output_path, valid_output_path, train_ratio=0.9, max_samples=None, random_seed=42):
    print(f"🧪 Text in the wild JSONL 생성 중... (train/valid {train_ratio}:{1-train_ratio} 분할)")

    random.seed(random_seed)
    os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(valid_output_path), exist_ok=True)

    with open(json_path, 'rb') as f:
        data = orjson.loads(f.read())

    images_info = {img['id']: img for img in data.get('images', [])}
    image_annotations = {}
    for ann in data.get('annotations', []):
        img_id = ann.get('image_id')
        image_annotations.setdefault(img_id, []).append(ann)

    del data
    gc.collect()

    img_ids = list(images_info.keys())
    if max_samples and len(img_ids) > max_samples:
        img_ids = img_ids[:max_samples]
    random.shuffle(img_ids)
    train_size = int(len(img_ids) * train_ratio)
    train_img_ids = img_ids[:train_size]
    valid_img_ids = img_ids[train_size:]

    dataset_lookup_name = "text_in_wild"
    lookup_dict = load_optimized_lookup(dataset_lookup_name)
    crop_dir_train = os.path.join(LOCAL_OUTPUT_PATH, "crops", "text_in_wild", "train")
    crop_dir_valid = os.path.join(LOCAL_OUTPUT_PATH, "crops", "text_in_wild", "valid")

    # train
    train_args = []
    for img_id in train_img_ids:
        img_info = images_info.get(img_id)
        anns = image_annotations.get(img_id, [])
        train_args.append((img_id, img_info, anns, base_path, lookup_dict, crop_dir_train))
    create_parallel_jsonl_from_args(train_args, train_output_path, "TextInWild-Train", text_in_wild_to_jsonl, max_total_samples=max_samples)

    # valid
    valid_args = []
    for img_id in valid_img_ids:
        img_info = images_info.get(img_id)
        anns = image_annotations.get(img_id, [])
        valid_args.append((img_id, img_info, anns, base_path, lookup_dict, crop_dir_valid))
    create_parallel_jsonl_from_args(valid_args, valid_output_path, "TextInWild-Valid", text_in_wild_to_jsonl, max_total_samples=max_samples)


def create_jsonl_public_admin_from_json(base_path, json_path, output_path, dataset_name, max_samples=None):
    print(f"🧪 {dataset_name} JSONL 생성 중...")
    data, file_handle = load_json_with_orjson(json_path)
    try:
        images = data.get('images', [])
        total_images = len(images) if isinstance(images, list) else 0
        if max_samples and total_images > max_samples:
            indices = list(range(total_images))
            random.seed(42)
            random.shuffle(indices)
            indices = indices[:max_samples]
        else:
            indices = list(range(total_images))

        image_annotations = {}
        annotations = data.get('annotations', [])
        i = 0
        while True:
            try:
                ann = annotations[i]
                img_id = ann.get('image_id', ann.get('id'))
                image_annotations.setdefault(img_id, []).append(ann)
                i += 1
            except IndexError:
                break

        del data
        del annotations
        gc.collect()

        if 'train_partly' in dataset_name.lower() or ('train' in dataset_name.lower() and 'partly' in dataset_name.lower()):
            dataset_lookup_name = "public_admin_train_partly"
        elif 'train' in dataset_name.lower() and 'partly' not in dataset_name.lower():
            dataset_lookup_name = "public_admin_train"
        else:
            dataset_lookup_name = "public_admin_valid"

        lookup_func = load_optimized_lookup(dataset_lookup_name)
        image_path_cache = {}
        if not lookup_func:
            for train_num in [1, 2, 3]:
                image_dir = f"{base_path}/Training/[원천]train{train_num}/02.원천데이터(jpg)"
                if os.path.exists(image_dir):
                    scanned_files = scan_images_recursive_with_scandir(image_dir, extensions=(".jpg",))
                    image_path_cache.update(scanned_files)
            image_dir = f"{base_path}/Validation/[원천]validation/02.원천데이터(Jpg)"
            if os.path.exists(image_dir):
                scanned_files = scan_images_recursive_with_scandir(image_dir, extensions=(".jpg",))
                image_path_cache.update(scanned_files)

        process_args = []
        crop_dir = os.path.join(LOCAL_OUTPUT_PATH, "crops", "public_admin", "train" if 'train' in dataset_name else ("train_partly" if 'partly' in dataset_name.lower() else "valid"))
        for i in indices:
            img_info = images[i]
            img_id = img_info.get('id', i)
            anns = image_annotations.get(img_id, [])
            process_args.append((img_info, anns, base_path, lookup_func, dataset_lookup_name, image_path_cache, crop_dir))

        del images
        del image_annotations
        gc.collect()

        create_parallel_jsonl_from_args(process_args, output_path, dataset_name, public_admin_to_jsonl, max_total_samples=max_samples)
    finally:
        safe_close_file(file_handle)


def create_jsonl_ocr_public_from_json(base_path, json_path, output_path, dataset_name, max_samples=None):
    print(f"🧪 {dataset_name} JSONL 생성 중...")
    data, file_handle = load_json_with_orjson(json_path)
    try:
        images = data.get('images', [])
        if hasattr(images, '__getitem__') and not isinstance(images, list):
            images_list = []
            chunk_size = 10000
            i = 0
            while True:
                try:
                    chunk = []
                    for j in range(chunk_size):
                        try:
                            chunk.append(images[i + j])
                        except IndexError:
                            break
                    images_list.extend(chunk)
                    i += len(chunk)
                    if len(chunk) < chunk_size:
                        break
                    if i % 20000 == 0:
                        gc.collect()
                except IndexError:
                    break
            images = images_list

        if max_samples and len(images) > max_samples:
            random.seed(42)
            random.shuffle(images)
            images = images[:max_samples]

        image_annotations = {}
        annotations = data.get('annotations', [])
        if hasattr(annotations, '__getitem__') and not isinstance(annotations, list):
            chunk_size = 10000
            annotations_list = []
            i = 0
            while True:
                try:
                    chunk = []
                    for j in range(chunk_size):
                        try:
                            chunk.append(annotations[i + j])
                        except IndexError:
                            break
                    annotations_list.extend(chunk)
                    i += len(chunk)
                    if len(chunk) < chunk_size:
                        break
                    if i % 50000 == 0:
                        gc.collect()
                except IndexError:
                    break
            annotations = annotations_list

        for ann in tqdm(annotations, desc="어노테이션 그룹화"):
            img_id = ann.get('image_id', ann.get('id'))
            image_annotations.setdefault(img_id, []).append(ann)

        del data
        del annotations

        dataset_lookup_name = "ocr_public_train" if 'train' in dataset_name.lower() else "ocr_public_valid"
        lookup_func = load_optimized_lookup(dataset_lookup_name)
        image_path_cache = {}
        if not lookup_func:
            if 'train' in dataset_name.lower():
                image_dir = f"{base_path}/Training/01.원천데이터"
            else:
                image_dir = f"{base_path}/Validation/01.원천데이터"
            if os.path.exists(image_dir):
                scanned_files = scan_images_recursive_with_scandir(image_dir, extensions=(".jpg", ".png", ".jpeg"))
                image_path_cache.update(scanned_files)

        process_args = []
        crop_dir = os.path.join(LOCAL_OUTPUT_PATH, "crops", "ocr_public", "train" if 'train' in dataset_name.lower() else "valid")
        for img_info in images:
            img_id = img_info.get('id')
            anns = image_annotations.get(img_id, [])
            process_args.append((img_info, anns, base_path, dataset_lookup_name, image_path_cache, crop_dir))

        del images
        del image_annotations

        create_parallel_jsonl_from_args(process_args, output_path, dataset_name, ocr_public_to_jsonl, max_total_samples=max_samples)
    finally:
        safe_close_file(file_handle)


def create_jsonl_finance_logistics_from_json(base_path, json_path, output_path, dataset_name, max_samples=None):
    print(f"🧪 {dataset_name} JSONL 생성 중...")
    data, file_handle = load_json_with_orjson(json_path)
    try:
        images = data.get('images', [])
        annotations = data.get('annotations', [])

        dataset_lookup_name = "finance_logistics_train" if 'train' in dataset_name.lower() else "finance_logistics_valid"
        lookup_func = load_optimized_lookup(dataset_lookup_name)
        fallback_cache = {}
        if not lookup_func:
            if 'train' in dataset_name.lower():
                scan_dirs = [f"{base_path}/Training/01.원천데이터"]
            else:
                scan_dirs = [f"{base_path}/Validation/01.원천데이터"]
            for scan_dir in scan_dirs:
                if os.path.exists(scan_dir):
                    scanned_files = scan_images_recursive_with_scandir(scan_dir, extensions=(".png",))
                    fallback_cache.update(scanned_files)

        image_info = {}
        target_count = max_samples if max_samples else None
        i = 0
        matched_count = 0
        while True:
            try:
                img = images[i]
                sub_dataset = img.get('sub_dataset', '')
                filename = f"{sub_dataset}.png"
                img_path = optimized_find_image_path(filename, base_path, dataset_lookup_name, fallback_cache)
                if img_path:
                    image_info[sub_dataset] = {
                        'file_path': img_path,
                        'width': img.get('width', 1000),
                        'height': img.get('height', 1000),
                        'filename': filename,
                    }
                    matched_count += 1
                i += 1
                if target_count and matched_count >= target_count:
                    break
            except IndexError:
                break

        all_annotations = {}
        for ann in annotations:
            sub_dataset = ann.get('sub_dataset', '')
            if sub_dataset in image_info:
                all_annotations.setdefault(sub_dataset, []).append({
                    'bbox': ann.get('bbox', []),
                    'text': ann.get('text', ''),
                    'sub_dataset': sub_dataset,
                })

        process_args = []
        crop_dir = os.path.join(LOCAL_OUTPUT_PATH, "crops", "finance_logistics", "train" if 'train' in dataset_name.lower() else "valid")
        for sub_dataset, info in image_info.items():
            anns = all_annotations.get(sub_dataset, [])
            process_args.append((sub_dataset, info, anns, crop_dir))

        del images
        del annotations
        del image_info
        del all_annotations

        create_parallel_jsonl_from_args(process_args, output_path, dataset_name, finance_logistics_to_jsonl, max_total_samples=max_samples)
    finally:
        safe_close_file(file_handle)


def create_jsonl_handwriting_from_json(base_path, json_path, output_path, dataset_name, max_samples=None):
    print(f"🧪 {dataset_name} JSONL 생성 중...")
    with open(json_path, 'rb') as f:
        data = orjson.loads(f.read())
    try:
        images = data.get('images', [])
        if max_samples and len(images) > max_samples:
            random.seed(42)
            random.shuffle(images)
            images = images[:max_samples]

        dataset_lookup_name = "handwriting_train" if 'train' in dataset_name.lower() else "handwriting_valid"
        lookup_func = load_optimized_lookup(dataset_lookup_name)
        fallback_cache = {}
        if not lookup_func:
            if 'train' in dataset_name.lower():
                scan_dirs = [f"{base_path}/1.Training/원천데이터"]
            else:
                scan_dirs = [f"{base_path}/2.Validation/원천데이터"]
            for scan_dir in scan_dirs:
                if os.path.exists(scan_dir):
                    scanned_files = scan_images_recursive_with_scandir(scan_dir, extensions=(".png",))
                    fallback_cache.update(scanned_files)

        filename_to_info = {}
        for img in images:
            img_file_name = img.get('file_name', '')
            if img_file_name and not img_file_name.endswith('.png'):
                filename = f"{img_file_name}.png"
            else:
                filename = img_file_name
            img_path = optimized_find_image_path(filename, base_path, dataset_lookup_name, fallback_cache)
            if img_path:
                filename_to_info[img_file_name] = {
                    'file_path': img_path,
                    'width': img.get('width', 1000),
                    'height': img.get('height', 1000),
                    'filename': filename,
                    'original_json_path': img.get('original_json_path', ''),
                }

        image_id_to_filename = {img.get('id'): img.get('file_name', '') for img in images}
        annotations = data.get('annotations', [])
        image_annotations = {}
        for ann in annotations:
            img_id = ann.get('image_id')
            key_fname = image_id_to_filename.get(img_id)
            if not key_fname:
                continue
            image_annotations.setdefault(key_fname, []).append(ann)

        process_args = []
        crop_dir = os.path.join(LOCAL_OUTPUT_PATH, "crops", "handwriting", "train" if 'train' in dataset_name.lower() else "valid")
        for img_file_name, info in filename_to_info.items():
            anns = image_annotations.get(img_file_name, [])
            process_args.append((img_file_name, info, anns, crop_dir))

        del images
        del annotations
        del filename_to_info
        del image_annotations

        create_parallel_jsonl_from_args(process_args, output_path, dataset_name, handwriting_to_jsonl, max_total_samples=max_samples)
    except Exception as e:
        print(f"❌ 손글씨 JSONL 생성 실패: {e}")
        raise

def create_lmdb_text_in_wild_from_ids(base_path, images_info, image_annotations, img_ids, output_path, split_name):
    """Text in the wild 이미지 ID 리스트로부터 LMDB 생성 (thread_map 병렬처리 버전)"""
    print(f"🚀 {split_name} 병렬 LMDB 생성 중... ({len(img_ids)}개 샘플)")
    
    # CPU 코어 수에 따른 최적 워커 수
    max_workers = min(mp.cpu_count(), 16)  # 워커 수를 16개로 증가
    print(f"  🔧 병렬 워커 수: {max_workers}개")
    
    # 🚀 lookup 딕셔너리 사전 로드
    dataset_lookup_name = "text_in_wild"
    lookup_dict = load_optimized_lookup(dataset_lookup_name)
    
    # 병렬 처리용 데이터 준비
    process_args = []
    for img_id in img_ids:
        if img_id not in images_info:
            continue
        img_info = images_info[img_id]
        annotations = image_annotations.get(img_id, [])
        process_args.append((img_id, img_info, annotations, base_path, lookup_dict))
    
    print(f"  📊 처리할 데이터: {len(process_args)}개")
    
    # JSON 데이터 메모리 해제 (가장 큰 메모리 사용 부분)
    del images_info
    del image_annotations
    del img_ids
    gc.collect()
    print(f"  🗑️ JSON 데이터 메모리 해제 완료")
    
    # 🚀 병렬 처리 + 즉시 LMDB 저장 (메모리 절약)
    start_time = time.time()
    
    # LMDB 환경 생성 (메모리 최적화 설정)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    env = lmdb.open(output_path, 
                    map_size=1099511627776,  # 1TB
                    writemap=True,  # 메모리 매핑 최적화
                    meminit=False,  # 메모리 초기화 비활성화
                    map_async=True)  # 비동기 맵핑
    
    print(f"  🔄 병렬 처리 + 즉시 저장 시작...")
    
    idx = 0
    
    # 청크 단위로 스트리밍 처리
    chunk_size = 10000  # 10000개씩 청크로 나누어 처리
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # process_args를 청크 단위로 순회
        for chunk_start in tqdm(range(0, len(process_args), chunk_size), desc=f"{split_name} 청크 처리"):
            chunk_end = min(chunk_start + chunk_size, len(process_args))
            chunk_args = process_args[chunk_start:chunk_end]
            
            # 현재 청크의 future만 생성
            futures = {executor.submit(process_single_text_wild_image, arg) for arg in chunk_args}
            
            # 더 작은 트랜잭션 단위로 분할 (메모리 누적 방지)
            txn_batch_size = 500  # 500개씩 트랜잭션 분할 (더 작게)
            batch_count = 0
            txn = None
            
            # 현재 청크의 작업만 처리
            for future in as_completed(futures):
                result = future.result()
                
                if result is not None:
                    img_id, img_data, gt_info = result
                    
                    # 새 트랜잭션 시작 (배치 단위)
                    if batch_count % txn_batch_size == 0:
                        if txn is not None:
                            txn.commit()  # 이전 트랜잭션 커밋
                        txn = env.begin(write=True)  # 새 트랜잭션 시작
                    
                    # 인식용: 단어 단위 크롭을 LMDB에 저장 (원본/GT 저장 안 함)
                    for w_idx, (crop_jpg, label) in enumerate(_iter_recog_crops_bytes(img_data, gt_info)):
                        label_bytes = label.encode('utf-8', errors='ignore')
                        if not label_bytes:
                            continue
                        img_key = f'image-{idx:09d}'.encode()
                        lab_key = f'label-{idx:09d}'.encode()
                        txn.put(img_key, crop_jpg)
                        txn.put(lab_key, label_bytes)
                        idx += 1
                        batch_count += 1

                    # 즉시 메모리 해제
                    del result
                    del img_data
                    del gt_info
            
            # 마지막 트랜잭션 커밋
            if txn is not None:
                txn.commit()
            del chunk_args, futures
            
            # 강제 가비지 컬렉션
            collected = gc.collect()
            print(f"  🗑️ 청크 {chunk_start//chunk_size + 1} 완료: {idx}개 처리, GC {collected}개 해제")
        
        # 마지막 커밋 (새 트랜잭션으로)
        txn = env.begin(write=True)
        txn.put('num-samples'.encode(), str(idx).encode())
        txn.commit()
    
    env.close()
    
    # 최종 메모리 해제
    del process_args
    del lookup_dict
    gc.collect()
    
    total_time = time.time() - start_time
    speed = idx / total_time if total_time > 0 else 0
    print(f"✅ {split_name} 병렬 LMDB 생성 완료: {idx}개 샘플")
    print(f"   ⏱️ 총 소요 시간: {total_time:.2f}초")
    print(f"   🚀 처리 속도: {speed:.1f} samples/sec")
    print(f"🗑️ {split_name} 모든 메모리 해제 완료")

# ============================================================================
# 공공행정문서 데이터셋 전용 함수
# ============================================================================

def create_public_admin_train_valid(max_samples=500):
    """공공행정문서 OCR train/valid LMDB 생성"""
    print("=" * 60)
    print("🧪 공공행정문서 OCR train/valid LMDB 생성")
    print("=" * 60)
    
    base_path = f"{FTP_BASE_PATH}/공공행정문서 OCR"
    train_json_path = f"{MERGED_JSON_PATH}/public_admin_train_merged.json"
    valid_json_path = f"{MERGED_JSON_PATH}/public_admin_valid_merged.json"
    train_output_path = f"{LOCAL_OUTPUT_PATH}/public_admin_annotations_train.lmdb"
    valid_output_path = f"{LOCAL_OUTPUT_PATH}/public_admin_annotations_valid.lmdb"
    
    # Training LMDB 생성
    if os.path.exists(train_json_path):
        print(f"📊 Training JSON 파일 발견: {train_json_path}")
        create_lmdb_public_admin_from_json(base_path, train_json_path, train_output_path, "공공행정문서 Train", max_samples)
        cleanup_memory()
    else:
        print(f"❌ Training JSON 파일을 찾을 수 없습니다: {train_json_path}")
    
    # Validation LMDB 생성
    if os.path.exists(valid_json_path):
        print(f"📊 Validation JSON 파일 발견: {valid_json_path}")
        create_lmdb_public_admin_from_json(base_path, valid_json_path, valid_output_path, "공공행정문서 Valid", max_samples)
        cleanup_memory()
    else:
        print(f"❌ Validation JSON 파일을 찾을 수 없습니다: {valid_json_path}")

def create_public_admin_train_partly(max_samples=500):
    """공공행정문서 OCR train_partly LMDB 생성 (학습 데이터셋)"""
    print("=" * 60)
    print("🧪 공공행정문서 OCR train_partly LMDB 생성")
    print("=" * 60)
    
    base_path = f"{FTP_BASE_PATH}/공공행정문서 OCR"
    train_json_path = f"{MERGED_JSON_PATH}/public_admin_train_partly_merged.json"
    train_output_path = f"{LOCAL_OUTPUT_PATH}/public_admin_annotations_train_partly.lmdb"
    
    # Training LMDB 생성
    if os.path.exists(train_json_path):
        print(f"📊 Training JSON 파일 발견: {train_json_path}")
        create_lmdb_public_admin_from_json(base_path, train_json_path, train_output_path, "공공행정문서 Train Partly", max_samples)
        cleanup_memory()
    else:
        print(f"❌ Training JSON 파일을 찾을 수 없습니다: {train_json_path}")

def create_lmdb_public_admin_from_json(base_path, json_path, output_path, dataset_name, max_samples=None):
    """공공행정문서 JSON 파일로부터 LMDB 생성"""
    print(f"🧪 {dataset_name} LMDB 생성 중...")
    
    # JSON 파일 로드 (orjson 방식)
    data, file_handle = load_json_with_orjson(json_path)
    
    try:
        # images와 annotations 처리 (orjson으로 로드된 Python 리스트)
        images = data.get('images', [])
        print(f"📊 JSON 파일 로드 완료: orjson Python 리스트 접근")
        
        # 샘플 수 제한을 위해 인덱스 기반 처리
        total_images = 0
        for _ in images:
            total_images += 1
        
        if max_samples and total_images > max_samples:
            print(f"📊 {max_samples}개 샘플로 제한 (총 {total_images}개 중)")
            # 인덱스 리스트 생성 후 섞기
            indices = list(range(total_images))
            random.seed(42)
            random.shuffle(indices)
            indices = indices[:max_samples]
        else:
            indices = list(range(total_images))
        
        # 이미지별로 어노테이션 그룹화
        image_annotations = {}
        annotations = data.get('annotations', [])
        
        # len() 호출 없이 안전한 반복 처리
        i = 0
        while True:
            try:
                ann = annotations[i]
                
                if i % 10000 == 0:  # 1만개마다 진행상황
                    print(f"    📊 어노테이션 처리: {i+1}개")
                
                img_id = ann.get('image_id', ann.get('id'))
                if img_id not in image_annotations:
                    image_annotations[img_id] = []
                image_annotations[img_id].append(ann)
                
                i += 1
            except IndexError:
                break
        
        print(f"  ✅ 어노테이션 그룹화 완료: {len(image_annotations)}개 이미지")
        
        # 🚀 즉시 원본 JSON 데이터 해제 (메모리 절약)
        del data
        del annotations
        gc.collect()
        print(f"  🗑️ 원본 JSON 데이터 메모리 해제 완료")
        
        # 🚀 최적화된 lookup 함수 활용
        print("  🔄 최적화된 이미지 경로 준비 중...")
        # dataset_name에 따라 정확한 lookup 이름 결정
        if 'train_partly' in dataset_name.lower() or ('train' in dataset_name.lower() and 'partly' in dataset_name.lower()):
            dataset_lookup_name = "public_admin_train_partly"
        elif 'train' in dataset_name.lower() and 'partly' not in dataset_name.lower():
            dataset_lookup_name = "public_admin_train"
        else:
            dataset_lookup_name = "public_admin_valid"
        lookup_func = load_optimized_lookup(dataset_lookup_name)
        
        # Fallback용 캐시 (최적화된 lookup이 없는 경우에만)
        image_path_cache = {}
        if not lookup_func:
            print("  🔄 Fallback 이미지 파일 경로 캐시 생성 중...")
            # Training 폴더들 스캔 (os.scandir 사용)
            for train_num in [1, 2, 3]:
                image_dir = f"{base_path}/Training/[원천]train{train_num}/02.원천데이터(jpg)"
                if os.path.exists(image_dir):
                    scanned_files = scan_images_recursive_with_scandir(image_dir, extensions=('.jpg',))
                    image_path_cache.update(scanned_files)
            
            # Validation 폴더 스캔 (os.scandir 사용)
            image_dir = f"{base_path}/Validation/[원천]validation/02.원천데이터(Jpg)"
            if os.path.exists(image_dir):
                scanned_files = scan_images_recursive_with_scandir(image_dir, extensions=('.jpg',))
                image_path_cache.update(scanned_files)
        
        print(f"  ✅ 이미지 경로 준비 완료: {'최적화된 lookup 사용' if lookup_func else f'{len(image_path_cache)}개 fallback 캐시'}")
        
        # 🚀 병렬 처리용 데이터 준비
        process_args = []
        for i, img_idx in enumerate(indices):
            img_info = images[img_idx]  # orjson Python 리스트에서 직접 접근
            img_id = img_info.get('id', i)
            annotations = image_annotations.get(img_id, [])
            process_args.append((img_info, annotations, base_path, lookup_func, dataset_lookup_name, image_path_cache))
        
        print(f"  📊 병렬 처리용 데이터 준비 완료: {len(process_args)}개")
        
        # 🚀 즉시 원본 딕셔너리 삭제 (메모리 해제)
        del images
        del image_annotations
        del indices
        gc.collect()
        print(f"  🗑️ 원본 딕셔너리 메모리 해제 완료")
        
        # 🚀 병렬 LMDB 생성
        create_parallel_lmdb_from_args(process_args, output_path, dataset_name, process_single_public_admin_image)
        
    finally:
        # 파일 핸들 정리
        safe_close_file(file_handle)

# ============================================================================
# OCR 공공 데이터셋 전용 함수
# ============================================================================

def create_ocr_public_train_valid(max_samples=500):
    """023.OCR 데이터(공공) train/valid LMDB 생성"""
    print("=" * 60)
    print("🧪 023.OCR 데이터(공공) train/valid LMDB 생성")
    print("=" * 60)
    
    base_path = f"{FTP_BASE_PATH}/023.OCR 데이터(공공)/01-1.정식개방데이터"
    train_json_path = f"{MERGED_JSON_PATH}/ocr_public_train_merged.json"
    valid_json_path = f"{MERGED_JSON_PATH}/ocr_public_valid_merged.json"
    train_output_path = f"{LOCAL_OUTPUT_PATH}/ocr_public_annotations_train.lmdb"
    valid_output_path = f"{LOCAL_OUTPUT_PATH}/ocr_public_annotations_valid.lmdb"
    
    # Training LMDB 생성
    if os.path.exists(train_json_path):
        print(f"📊 Training JSON 파일 발견: {train_json_path}")
        create_lmdb_ocr_public_from_json(base_path, train_json_path, train_output_path, "OCR 공공 Train", max_samples)
        cleanup_memory()
    else:
        print(f"❌ Training JSON 파일을 찾을 수 없습니다: {train_json_path}")
    
    # Validation LMDB 생성
    if os.path.exists(valid_json_path):
        print(f"📊 Validation JSON 파일 발견: {valid_json_path}")
        create_lmdb_ocr_public_from_json(base_path, valid_json_path, valid_output_path, "OCR 공공 Valid", max_samples)
        cleanup_memory()
    else:
        print(f"❌ Validation JSON 파일을 찾을 수 없습니다: {valid_json_path}")

def create_lmdb_ocr_public_from_json(base_path, json_path, output_path, dataset_name, max_samples=None, use_groups=False):
    """OCR 공공 JSON 파일로부터 LMDB 생성"""
    print(f"🧪 {dataset_name} LMDB 생성 중...")
    
    if use_groups:
        # 그룹별 처리
        def process_group(group_data, original_path):
            # 그룹별 처리 로직
            print(f"  📝 그룹 데이터 처리: {len(group_data['images'])}개 이미지")
            return len(group_data['images'])
        
        total_processed = process_json_by_groups(json_path, process_group, max_samples)
        print(f"✅ 그룹별 처리 완료: 총 {total_processed}개 처리됨")
        return
    
    # 기존 방식 (전체 JSON 로드)
    # JSON 파일 로드 (orjson 방식)
    data, file_handle = load_json_with_orjson(json_path)
    
    try:
        # images와 annotations 처리 (빠른 처리를 위해 Python 리스트로 변환)
        images = data.get('images', [])
        print(f"📊 JSON 파일 로드 완료")
        
        # bigjson Array를 Python 리스트로 변환 (메모리 절약을 위해 청크 단위로)
        if hasattr(images, '__getitem__') and not isinstance(images, list):
            print("  🔄 bigjson Array를 Python 리스트로 변환 중... (메모리 절약)")
            images_list = []
            chunk_size = 10000  # 청크로 변환
            i = 0
            while True:
                try:
                    chunk = []
                    for j in range(chunk_size):
                        try:
                            chunk.append(images[i + j])
                        except IndexError:
                            break
                    images_list.extend(chunk)
                    i += len(chunk)
                    if len(chunk) < chunk_size:
                        break
                    print(f"    📊 변환 진행: {i}개")
                    # 청크마다 메모리 정리
                    if i % 20000 == 0:
                        gc.collect()
                except IndexError:
                    break
            images = images_list
            print(f"  ✅ 변환 완료: {len(images)}개 이미지")
        
        if max_samples and len(images) > max_samples:
            print(f"📊 {max_samples}개 샘플로 제한 (총 {len(images)}개 중)")
            random.seed(42)
            random.shuffle(images)
            images = images[:max_samples]
        
        # 이미지별로 어노테이션 그룹화 (병렬 처리)
        image_annotations = {}
        annotations = data.get('annotations', [])
        
        # bigjson Array를 Python 리스트로 변환 (메모리 절약 방식)
        if hasattr(annotations, '__getitem__') and not isinstance(annotations, list):
            print("  🔄 어노테이션 bigjson Array를 Python 리스트로 변환 중... (메모리 절약)")
            # 작은 청크 단위로 변환하여 메모리 절약
            chunk_size = 10000  # 청크
            annotations_list = []
            i = 0
            while True:
                try:
                    chunk = []
                    for j in range(chunk_size):
                        try:
                            chunk.append(annotations[i + j])
                        except IndexError:
                            break
                    annotations_list.extend(chunk)
                    i += len(chunk)
                    if len(chunk) < chunk_size:
                        break
                    print(f"    📊 어노테이션 변환 진행: {i}개")
                    # 청크마다 메모리 정리
                    if i % 50000 == 0:
                        gc.collect()
                except IndexError:
                    break
            annotations = annotations_list
            print(f"  ✅ 어노테이션 변환 완료: {len(annotations)}개")
        
        # 어노테이션 그룹화 (bigjson은 스레드 안전하지 않으므로 순차 처리)
        print("  🔄 어노테이션 그룹화 중...")
        for ann in tqdm(annotations, desc="어노테이션 그룹화"):
            img_id = ann.get('image_id', ann.get('id'))
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)
        
        print(f"  ✅ 어노테이션 그룹화 완료: {len(image_annotations)}개 이미지")
        
        # 🚀 즉시 원본 JSON 데이터 해제 (메모리 절약)
        del data
        del annotations
        print(f"  🗑️ 원본 JSON 데이터 메모리 해제 완료")
        
        # 🚀 최적화된 lookup 함수 활용
        print("  🔄 최적화된 이미지 경로 준비 중...")
        dataset_lookup_name = "ocr_public_train" if 'train' in dataset_name.lower() else "ocr_public_valid"
        lookup_func = load_optimized_lookup(dataset_lookup_name)
        
        # Fallback용 캐시 (최적화된 lookup이 없는 경우에만)
        image_path_cache = {}
        if not lookup_func:
            print("  🔄 Fallback 이미지 경로 캐시 구축 중...")
            # Training/Validation 구분
            if 'train' in dataset_name.lower():
                image_dir = f"{base_path}/Training/01.원천데이터"
            else:
                image_dir = f"{base_path}/Validation/01.원천데이터"
            
            # 실제 디렉토리에서 이미지 파일 스캔 (os.scandir 사용)
            if os.path.exists(image_dir):
                scanned_files = scan_images_recursive_with_scandir(image_dir, extensions=('.jpg', '.png', '.jpeg'))
                image_path_cache.update(scanned_files)
        
        print(f"  ✅ 이미지 경로 준비 완료: {'최적화된 lookup 사용' if lookup_func else f'{len(image_path_cache)}개 fallback 캐시'}")
        
        # 🚀 병렬 처리용 데이터 준비
        process_args = []
        for img_info in images:
            img_id = img_info.get('id')
            annotations = image_annotations.get(img_id, [])
            process_args.append((img_info, annotations, base_path, dataset_lookup_name, image_path_cache))
        
        print(f"  📊 병렬 처리용 데이터 준비 완료: {len(process_args)}개")
        
        # 🚀 즉시 원본 딕셔너리 삭제 (메모리 해제)
        del images
        del image_annotations
        print(f"  🗑️ 원본 딕셔너리 메모리 해제 완료")
        
        # 🚀 병렬 LMDB 생성
        create_parallel_lmdb_from_args(process_args, output_path, dataset_name, process_single_ocr_public_image)
        
    finally:
        # 파일 핸들 정리
        safe_close_file(file_handle)

# ============================================================================
# 금융물류 데이터셋 전용 함수
# ============================================================================

def create_finance_logistics_train_valid(max_samples=None):
    """025.OCR 데이터(금융 및 물류) train/valid LMDB 생성 (전체 데이터)"""
    print("=" * 60)
    print("🧪 025.OCR 데이터(금융 및 물류) train/valid LMDB 생성")
    print("=" * 60)
    
    base_path = f"{FTP_BASE_PATH}/025.OCR 데이터(금융 및 물류)/01-1.정식개방데이터"
    train_json_path = f"{MERGED_JSON_PATH}/finance_logistics_train_merged.json"
    valid_json_path = f"{MERGED_JSON_PATH}/finance_logistics_valid_merged.json"
    train_output_path = f"{LOCAL_OUTPUT_PATH}/finance_logistics_annotations_train.lmdb"
    valid_output_path = f"{LOCAL_OUTPUT_PATH}/finance_logistics_annotations_valid.lmdb"
    
    # Training LMDB 생성
    if os.path.exists(train_json_path):
        print(f"📊 Training JSON 파일 발견: {train_json_path}")
        create_lmdb_finance_logistics_from_json(base_path, train_json_path, train_output_path, "금융물류 Train", max_samples)
        cleanup_memory()
    else:
        print(f"❌ Training JSON 파일을 찾을 수 없습니다: {train_json_path}")
    
    # Validation LMDB 생성
    if os.path.exists(valid_json_path):
        print(f"📊 Validation JSON 파일 발견: {valid_json_path}")
        create_lmdb_finance_logistics_from_json(base_path, valid_json_path, valid_output_path, "금융물류 Valid", max_samples)
        cleanup_memory()
    else:
        print(f"❌ Validation JSON 파일을 찾을 수 없습니다: {valid_json_path}")

def create_lmdb_finance_logistics_from_json(base_path, json_path, output_path, dataset_name, max_samples=None):
    """금융물류 JSON 파일로부터 LMDB 생성 (초고속 버전: orjson 직접 사용)"""
    print(f"🧪 {dataset_name} LMDB 생성 중...")
    
    # JSON 파일 로드 (orjson 방식)
    data, file_handle = load_json_with_orjson(json_path)
    
    try:
        # 🚀 최적화 1: bigjson Array 직접 사용 (변환 없음)
        images = data.get('images', [])
        annotations = data.get('annotations', [])
        print(f"📊 JSON 파일 로드 완료 - bigjson Array 직접 사용")
        
        # 🚀 최적화 2: 최적화된 lookup 함수 활용
        print("  🔄 최적화된 이미지 경로 준비 중...")
        dataset_lookup_name = "finance_logistics_train" if 'train' in dataset_name.lower() else "finance_logistics_valid"
        lookup_func = load_optimized_lookup(dataset_lookup_name)
        
        # Fallback용 스캔 (최적화된 lookup이 없는 경우에만)
        fallback_cache = {}
        if not lookup_func:
            print("  🔄 Fallback 이미지 파일 스캔 중...")
            # Training/Validation 구분
            if 'train' in dataset_name.lower():
                scan_dirs = [f"{base_path}/Training/01.원천데이터"]
            else:
                scan_dirs = [f"{base_path}/Validation/01.원천데이터"]
            
            for scan_dir in scan_dirs:
                if os.path.exists(scan_dir):
                    scanned_files = scan_images_recursive_with_scandir(scan_dir, extensions=('.png',))
                    fallback_cache.update(scanned_files)
        
        print(f"  ✅ 이미지 경로 준비 완료: {'최적화된 lookup 사용' if lookup_func else f'{len(fallback_cache)}개 fallback 캐시'}")
        
        # 🚀 최적화 3: bigjson 이미지 정보 추출 (500개만 선택, 빠르게)
        print("  🔄 이미지 정보 매핑 중...")
        image_info = {}  # sub_dataset → image_info
        
        # 🚀 전체 이미지 처리 (max_samples가 있으면 제한)
        target_count = max_samples if max_samples else None  # 전체 데이터 처리
        if target_count:
            print(f"  📊 목표 이미지 수: {target_count}개 (제한)")
        else:
            print(f"  📊 전체 이미지 처리 (제한 없음)")
        
        i = 0
        matched_count = 0
        while True:
            try:
                img = images[i]
                sub_dataset = img.get('sub_dataset', '')
                filename = f"{sub_dataset}.png"
                
                # 🚀 최적화된 경로 찾기
                img_path = optimized_find_image_path(filename, base_path, dataset_lookup_name, fallback_cache)
                if img_path:
                    image_info[sub_dataset] = {
                        'file_path': img_path,
                        'width': img.get('width', 1000),
                        'height': img.get('height', 1000),
                        'filename': filename
                    }
                    matched_count += 1
                
                i += 1
                if i % 10000 == 0:
                    if target_count:
                        print(f"    📊 매핑 진행: {i}개 처리, {matched_count}개 매칭 (목표: {target_count}개)")
                    else:
                        print(f"    📊 매핑 진행: {i}개 처리, {matched_count}개 매칭 (전체 처리)")
                
                # 목표 달성시 조기 종료 🎯 (target_count가 설정된 경우만)
                if target_count and matched_count >= target_count:
                    print(f"    🎯 목표 달성: {matched_count}개 이미지 선택 완료!")
                    break
                    
            except IndexError:
                break
        
        print(f"  ✅ 이미지 정보 매핑 완료: {len(image_info)}개")
        
        # 🚀 최적화 4: 단순 순차 어노테이션 처리 (sub_dataset 기반)
        print("  🔄 순차 어노테이션 처리...")
        
        all_annotations = {}
        total_found = 0
        
        print(f"  🚀 순차 처리 시작 (Iterator 방식)")
        
        # 🚀 bigjson Array를 Iterator로 안전하게 처리
        i = 0
        for ann in annotations:
            try:
                # 🚀 None 체크로 끝 감지
                if ann is None:
                    print(f"    🏁 어노테이션 끝 감지 (None) - 총 {total_found:,}개 처리 완료")
                    break
                
                # ann이 빈 값이거나 올바르지 않은 경우 체크
                if not ann or not hasattr(ann, 'get'):
                    i += 1
                    continue
                
                sub_dataset = ann.get('sub_dataset', '')
                
                if sub_dataset in image_info:
                    if sub_dataset not in all_annotations:
                        all_annotations[sub_dataset] = []
                    
                    # 🚀 bigjson Array bbox를 안전하게 Python list로 변환
                    bbox_data = ann.get('bbox', [])
                    safe_bbox = []
                    
                    if bbox_data:
                        try:
                            # bigjson Array인 경우 안전하게 변환
                            if hasattr(bbox_data, '__getitem__') and not isinstance(bbox_data, list):
                                # 최대 8개까지 시도
                                for j in range(8):
                                    try:
                                        safe_bbox.append(bbox_data[j])
                                    except (IndexError, TypeError):
                                        break
                            else:
                                safe_bbox = bbox_data
                        except Exception:
                            safe_bbox = []
                    
                    all_annotations[sub_dataset].append({
                        'bbox': safe_bbox,
                        'text': ann.get('text', ''),
                        'sub_dataset': sub_dataset
                    })
                    total_found += 1
                
                i += 1
                if i % 100000 == 0:
                    print(f"    📊 처리 진행: {i:,}개, 발견: {total_found:,}개")
                    
            except Exception as e:
                if i % 100000 == 0:
                    print(f"    ⚠️ 오류 발생: {e}")
                i += 1
                continue
        
        print(f"  ✅ 어노테이션 처리 완료: {len(all_annotations)}개 이미지, {total_found:,}개 어노테이션")
        
        # 🚀 즉시 원본 JSON 데이터 해제 (메모리 절약)
        del data
        del annotations
        print(f"  🗑️ 원본 JSON 데이터 메모리 해제 완료")
        
        # 🚀 최적화 5: 병행처리 LMDB 생성 (간단한 ThreadPoolExecutor)
        print("  🔄 병행처리 LMDB 생성 중...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 병렬 처리용 데이터 준비 (all_annotations를 포함)
        process_args = []
        for sub_dataset, img_info_data in image_info.items():
            annotations_for_dataset = all_annotations.get(sub_dataset, [])
            process_args.append((sub_dataset, img_info_data, annotations_for_dataset))
        
        print(f"  🚀 병행처리 시작: {len(process_args)}개 이미지, 16개 워커")
        
        # 🚀 즉시 원본 딕셔너리 삭제 (메모리 해제)
        del image_info
        del all_annotations
        print(f"  🗑️ 원본 딕셔너리 메모리 해제 완료")
        
        # 청크 단위로 스트리밍 처리하여 메모리 절약
        create_parallel_lmdb_from_args(process_args, output_path, dataset_name, process_single_finance_logistics_image, max_workers=16)
        
    finally:
        # 파일 핸들 정리
        safe_close_file(file_handle)

# ============================================================================
# 손글씨 데이터셋 전용 함수
# ============================================================================

def create_handwriting_train_valid(max_samples=500):
    """053.대용량 손글씨 OCR train/valid LMDB 생성"""
    print("=" * 60)
    print("🧪 053.대용량 손글씨 OCR train/valid LMDB 생성")
    print("=" * 60)
    
    base_path = f"{FTP_BASE_PATH}/053.대용량 손글씨 OCR 데이터/01.데이터"
    train_json_path = f"{MERGED_JSON_PATH}/handwriting_train_merged.json"
    valid_json_path = f"{MERGED_JSON_PATH}/handwriting_valid_merged.json"
    train_output_path = f"{LOCAL_OUTPUT_PATH}/handwriting_annotations_train.lmdb"
    valid_output_path = f"{LOCAL_OUTPUT_PATH}/handwriting_annotations_valid.lmdb"
    
    # Training LMDB 생성
    if os.path.exists(train_json_path):
        print(f"📊 Training JSON 파일 발견: {train_json_path}")
        create_lmdb_handwriting_from_json(base_path, train_json_path, train_output_path, "손글씨 Train", None)
        test_fast_model_input(train_output_path)
        cleanup_memory()
    else:
        print(f"❌ Training JSON 파일을 찾을 수 없습니다: {train_json_path}")
    
    # Validation LMDB 생성
    if os.path.exists(valid_json_path):
        print(f"📊 Validation JSON 파일 발견: {valid_json_path}")
        create_lmdb_handwriting_from_json(base_path, valid_json_path, valid_output_path, "손글씨 Valid", None)
        test_fast_model_input(valid_output_path)
        cleanup_memory()
    else:
        print(f"❌ Validation JSON 파일을 찾을 수 없습니다: {valid_json_path}")

def create_lmdb_handwriting_from_json(base_path, json_path, output_path, dataset_name, max_samples=None):
    """손글씨 JSON 파일로부터 LMDB 생성 (orjson 최적화 버전)"""
    print(f"🧪 {dataset_name} LMDB 생성 중...")
    print(f"📋 bbox 형태: [x1, y1, x2, y1, x2, y2, x3, y3] -> [x1, y1, x2, y1, x2, y2, x3, y3] (8개 좌표)")
    
    # 📄 손글씨는 orjson으로 빠르게 로드
    print(f"📄 JSON 파일 로드 중: {json_path}")
    with open(json_path, 'rb') as f:
        data = orjson.loads(f.read())
    print("✅ orjson 로드 성공")
    
    try:
        # 🚀 최적화 1: orjson으로 로드된 Python 리스트 직접 사용
        images = data.get('images', [])
        print(f"📊 JSON 파일 로드 완료: {len(images)}개 이미지")
        
        # 🚀 최적화 2: scandir로 실제 이미지 파일 스캔 (한 번만)
        print("  🔄 scandir로 실제 이미지 파일 스캔 중...")
        filename_to_path = {}
        
        # 🚀 최적화된 lookup 함수 활용
        dataset_lookup_name = "handwriting_train" if 'train' in dataset_name.lower() else "handwriting_valid"
        lookup_func = load_optimized_lookup(dataset_lookup_name)
        
        # Fallback용 스캔 (최적화된 lookup이 없는 경우에만)
        fallback_cache = {}
        if not lookup_func:
            print("  🔄 Fallback 이미지 파일 스캔 중...")
            # Training/Validation 구분해서 스캔 (os.scandir 사용)
            if 'train' in dataset_name.lower():
                scan_dirs = [f"{base_path}/1.Training/원천데이터"]
            else:
                scan_dirs = [f"{base_path}/2.Validation/원천데이터"]
            
            for scan_dir in scan_dirs:
                if os.path.exists(scan_dir):
                    scanned_files = scan_images_recursive_with_scandir(scan_dir, extensions=('.png',))
                    fallback_cache.update(scanned_files)
        
        print(f"  ✅ 이미지 경로 준비 완료: {'최적화된 lookup 사용' if lookup_func else f'{len(fallback_cache)}개 fallback 캐시'}")
        
        # 🚀 최적화 3: 이미지 정보 추출
        print("  🔄 이미지 정보 매핑 중...")
        image_info = {}  # file_name → image_info
        
        target_count = max_samples if max_samples else None
        if target_count:
            print(f"  📊 목표 이미지 수: {target_count}개 (제한)")
        else:
            print(f"  📊 전체 이미지 처리 (제한 없음)")
        
        # orjson 로드된 리스트이므로 len() 사용 가능
        if target_count and len(images) > target_count:
            print(f"📊 {target_count}개 샘플로 제한 (총 {len(images)}개 중)")
            random.seed(42)
            random.shuffle(images)
            images = images[:target_count]
        
        matched_count = 0
        for img in images:
            img_file_name = img.get('file_name', '')
            
            # 확장자 추가
            if img_file_name and not img_file_name.endswith('.png'):
                filename = f"{img_file_name}.png"
            else:
                filename = img_file_name
            
            # 🚀 최적화된 경로 찾기
            img_path = optimized_find_image_path(filename, base_path, dataset_lookup_name, fallback_cache)
            if img_path:
                image_info[img_file_name] = {
                    'file_path': img_path,
                    'width': img.get('width', 1000),
                    'height': img.get('height', 1000),
                    'filename': filename,
                    'original_json_path': img.get('original_json_path', '')
                }
                matched_count += 1
        
        print(f"  ✅ 이미지 정보 매핑 완료: {len(image_info)}개")

        # 🚀 어노테이션을 image_id 기준으로 그룹화
        annotations = data.get('annotations', [])
        print("  🔄 어노테이션 그룹화 중...")
        image_id_to_filename = {}
        for img in images:
            fid = img.get('id')
            fname = img.get('file_name', '')
            image_id_to_filename[fid] = fname

        image_annotations = {}
        for ann in annotations:
            img_id = ann.get('image_id')
            key_fname = image_id_to_filename.get(img_id)
            if not key_fname:
                continue
            if key_fname not in image_annotations:
                image_annotations[key_fname] = []
            image_annotations[key_fname].append(ann)
        print(f"  ✅ 어노테이션 그룹화 완료: {len(image_annotations)}개 이미지")
        
        # 🚀 즉시 원본 JSON 데이터 해제 (메모리 절약)
        del data
        print(f"  🗑️ 원본 JSON 데이터 메모리 해제 완료")
        
        # 🚀 병렬 처리용 데이터 준비 (이미지별 어노테이션 전달)
        process_args = []
        for img_file_name, info in image_info.items():
            anns = image_annotations.get(img_file_name, [])
            process_args.append((img_file_name, info, anns))
        print(f"  📊 병렬 처리용 데이터 준비 완료: {len(process_args)}개")
        
        # 🚀 즉시 원본 딕셔너리 삭제 (메모리 해제)
        del images
        del image_info
        del fallback_cache
        print(f"  🗑️ 원본 딕셔너리 메모리 해제 완료")
        
        # 🚀 병렬 LMDB 생성
        create_parallel_lmdb_from_args(process_args, output_path, dataset_name, process_single_handwriting_image)
        
    except Exception as e:
        print(f"❌ 손글씨 LMDB 생성 실패: {e}")
        raise

# ============================================================================
# 공통 유틸리티 함수
# ============================================================================

def group_images_by_original_json(data):
    """이미지들을 original_json_path별로 그룹화"""
    groups = {}
    
    for img in data.get('images', []):
        original_path = img.get('original_json_path', '')
        if original_path not in groups:
            groups[original_path] = []
        groups[original_path].append(img)
    
    return groups

def process_json_by_groups(json_path, process_func, max_samples=None):
    """JSON 파일을 원본 파일별 그룹으로 나누어 처리"""
    print(f"📄 JSON 파일을 그룹별로 처리 중: {json_path}")
    
    # JSON 파일 로드 (fallback 방식)
    data, file_handle = load_json_with_orjson(json_path)
    
    try:
        # 원본 JSON 파일별로 그룹화
        groups = group_images_by_original_json(data)
        print(f"📊 총 {len(groups)}개의 원본 JSON 파일 그룹 발견")
        
        # 각 그룹별로 처리
        total_processed = 0
        for original_path, images in groups.items():
            if max_samples and total_processed >= max_samples:
                break
                
            print(f"🔍 그룹 처리 중: {os.path.basename(original_path)} ({len(images)}개 이미지)")
            
            # 그룹별 데이터 구성
            group_data = {
                'images': images,
                'annotations': [ann for ann in data.get('annotations', []) 
                              if any(img.get('original_json_path') == original_path 
                                    for img in images if img.get('id') == ann.get('image_id'))],
                'info': data.get('info', {}),
                'categories': data.get('categories', [])
            }
            
            # 처리 함수 호출
            processed_count = process_func(group_data, original_path)
            total_processed += processed_count
            
            print(f"✅ 그룹 처리 완료: {processed_count}개 처리됨 (총 {total_processed}개)")
        
        return total_processed
        
    finally:
        # 파일 핸들 정리
        safe_close_file(file_handle)

def test_fast_model_input(lmdb_path):
    """생성된 LMDB가 FAST 모델의 입력 형식에 맞는지 테스트"""
    print(f"\n🔍 FAST 모델 입력 형식 테스트: {lmdb_path}")
    
    try:
        dataset = FAST_LMDB(
            lmdb_path=lmdb_path,
            split='train',
            is_transform=False,
            img_size=(640, 640),
            short_size=640
        )
        
        print(f"📊 데이터셋 정보:")
        print(f"   - 총 샘플 수: {len(dataset)}")
        
        # 몇 개 샘플 테스트
        for i in range(min(5, len(dataset))):
            print(f"\n🧪 샘플 {i+1} 테스트:")
            
            img, gt_info = dataset.get_image_and_gt(i)
            print(f"   - 원본 이미지 형태: {img.shape}")
            print(f"   - 바운딩 박스 수: {len(gt_info['bboxes'])}")
            print(f"   - 텍스트 수: {len(gt_info['words'])}")
            print(f"   - 파일명: {gt_info['filename']}")
            
            if gt_info['bboxes']:
                print(f"   - 첫 번째 텍스트: {gt_info['words'][0]}")
                if len(gt_info['words']) > 1:
                    print(f"   - 두 번째 텍스트: {gt_info['words'][1]}")
                if len(gt_info['words']) > 2:
                    print(f"   - 세 번째 텍스트: {gt_info['words'][2]}")
        
        print(f"✅ FAST 모델 입력 형식 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"❌ FAST 모델 입력 형식 테스트 실패: {e}")
        return False

def main():
    """메인 함수"""
    # 출력 형식 선택: env 또는 기본값(jsonl)
    output_format = os.getenv("FAST_OUTPUT_FORMAT", "jsonl").lower()
    max_samples_env = os.getenv("FAST_MAX_SAMPLES")
    try:
        max_samples_limit = int(max_samples_env) if max_samples_env is not None else None
    except ValueError:
        max_samples_limit = None
    use_jsonl = (output_format == "jsonl")

    print(
        "🚀 모든 한국어 OCR 데이터셋 train/valid "
        + ("JSONL" if use_jsonl else "LMDB")
        + " 생성 (전체 데이터, 제한 없음)"
    )
    print("=" * 60)
    
    # gvfs FTP 경로 확인
    if not is_ftp_mounted():
        print("❌ gvfs FTP 경로 확인 실패")
        print("💡 파일 관리자에서 FTP 서버에 접속하여 gvfs 마운트를 활성화해주세요")
        return
    
    if not os.path.exists(FTP_BASE_PATH):
        print("❌ gvfs FTP 경로 확인 실패")
        return
    
    print("✅ gvfs FTP 경로 확인 완료")
    
    # 🚀 최적화된 lookup 파일 상태 확인 (pickle 우선)
    print("\n🔍 최적화된 lookup 파일 상태 확인:")
    datasets = [
        "handwriting_train", "handwriting_valid", 
        "finance_logistics_train", "finance_logistics_valid",
        "ocr_public_train", "ocr_public_valid",
        "public_admin_train", "public_admin_train_partly", "public_admin_valid"
    ]
    
    available_count = 0
    pickle_count = 0
    py_count = 0
    
    for dataset in datasets:
        pkl_gz_file = f"FAST/lookup_{dataset}.pkl.gz"
        pkl_file = f"FAST/lookup_{dataset}.pkl"
        py_file = f"FAST/optimized_lookup_{dataset}.py"
        
        if os.path.exists(pkl_gz_file):
            print(f"  🚀 {dataset} (압축된 pickle - 최고속)")
            available_count += 1
            pickle_count += 1
        elif os.path.exists(pkl_file):
            print(f"  ⚡ {dataset} (pickle - 고속)")
            available_count += 1
            pickle_count += 1
        elif os.path.exists(py_file):
            print(f"  🐌 {dataset} (Python 모듈 - 저속)")
            available_count += 1
            py_count += 1
        else:
            print(f"  ⚠️ {dataset} (fallback 사용)")
    
    print(f"\n📊 최적화된 lookup: {available_count}/{len(datasets)}개 사용 가능")
    print(f"   🚀 Pickle: {pickle_count}개 (고속)")
    print(f"   🐌 Python: {py_count}개 (저속)")
    
    if available_count == 0:
        print("💡 ftp_tree_viewer.py를 실행해서 최적화된 lookup 함수들을 생성하면 속도가 대폭 개선됩니다!")
        print("💡 그 다음 convert_lookup_to_pickle.py를 실행해서 pickle로 변환하면 더욱 빨라집니다!")
    elif pickle_count == 0 and py_count > 0:
        print("💡 convert_lookup_to_pickle.py를 실행해서 Python 모듈을 pickle로 변환하면 5-10배 빨라집니다!")
    elif pickle_count < len(datasets):
        print("💡 일부 lookup만 pickle로 최적화됨. 누락된 것들은 convert_lookup_to_pickle.py로 변환하세요!")
    else:
        print("🚀 모든 lookup이 pickle로 최적화됨! 최고 성능으로 실행됩니다!")
    
    # 출력 디렉토리 생성
    os.makedirs(LOCAL_OUTPUT_PATH, exist_ok=True)
    
    # 이미 완료된 산출물 확인
    completed_lmdbs = []
    completed_jsonls = []
    lmdb_paths = [
        f"{LOCAL_OUTPUT_PATH}/text_in_wild_annotations_train.lmdb",
        f"{LOCAL_OUTPUT_PATH}/text_in_wild_annotations_valid.lmdb",
        f"{LOCAL_OUTPUT_PATH}/public_admin_annotations_train.lmdb",
        f"{LOCAL_OUTPUT_PATH}/public_admin_annotations_train_partly.lmdb",
        f"{LOCAL_OUTPUT_PATH}/public_admin_annotations_valid.lmdb",
        f"{LOCAL_OUTPUT_PATH}/ocr_public_annotations_train.lmdb",
        f"{LOCAL_OUTPUT_PATH}/ocr_public_annotations_valid.lmdb",
        f"{LOCAL_OUTPUT_PATH}/finance_logistics_annotations_train.lmdb",
        f"{LOCAL_OUTPUT_PATH}/finance_logistics_annotations_valid.lmdb",
        f"{LOCAL_OUTPUT_PATH}/handwriting_annotations_train.lmdb",
        f"{LOCAL_OUTPUT_PATH}/handwriting_annotations_valid.lmdb"
    ]
    jsonl_paths = [
        f"{LOCAL_OUTPUT_PATH}/text_in_wild_annotations_train.jsonl",
        f"{LOCAL_OUTPUT_PATH}/text_in_wild_annotations_valid.jsonl",
        f"{LOCAL_OUTPUT_PATH}/public_admin_annotations_train.jsonl",
        f"{LOCAL_OUTPUT_PATH}/public_admin_annotations_train_partly.jsonl",
        f"{LOCAL_OUTPUT_PATH}/public_admin_annotations_valid.jsonl",
        f"{LOCAL_OUTPUT_PATH}/ocr_public_annotations_train.jsonl",
        f"{LOCAL_OUTPUT_PATH}/ocr_public_annotations_valid.jsonl",
        f"{LOCAL_OUTPUT_PATH}/finance_logistics_annotations_train.jsonl",
        f"{LOCAL_OUTPUT_PATH}/finance_logistics_annotations_valid.jsonl",
        f"{LOCAL_OUTPUT_PATH}/handwriting_annotations_train.jsonl",
        f"{LOCAL_OUTPUT_PATH}/handwriting_annotations_valid.jsonl",
    ]
    
    for lmdb_path in lmdb_paths:
        if os.path.exists(lmdb_path):
            completed_lmdbs.append(lmdb_path)
            print(f"✅ 이미 완료됨: {os.path.basename(lmdb_path)}")
    for jsonl_path in jsonl_paths:
        if os.path.exists(jsonl_path):
            completed_jsonls.append(jsonl_path)
            print(f"✅ 이미 완료됨: {os.path.basename(jsonl_path)}")
    
    if use_jsonl:
        # JSONL 경로 지정
        text_wild_train_jsonl = f"{LOCAL_OUTPUT_PATH}/text_in_wild_annotations_train.jsonl"
        text_wild_valid_jsonl = f"{LOCAL_OUTPUT_PATH}/text_in_wild_annotations_valid.jsonl"
        public_admin_train_jsonl = f"{LOCAL_OUTPUT_PATH}/public_admin_annotations_train.jsonl"
        public_admin_train_partly_jsonl = f"{LOCAL_OUTPUT_PATH}/public_admin_annotations_train_partly.jsonl"
        public_admin_valid_jsonl = f"{LOCAL_OUTPUT_PATH}/public_admin_annotations_valid.jsonl"
        ocr_public_train_jsonl = f"{LOCAL_OUTPUT_PATH}/ocr_public_annotations_train.jsonl"
        ocr_public_valid_jsonl = f"{LOCAL_OUTPUT_PATH}/ocr_public_annotations_valid.jsonl"
        finance_train_jsonl = f"{LOCAL_OUTPUT_PATH}/finance_logistics_annotations_train.jsonl"
        finance_valid_jsonl = f"{LOCAL_OUTPUT_PATH}/finance_logistics_annotations_valid.jsonl"
        handwriting_train_jsonl = f"{LOCAL_OUTPUT_PATH}/handwriting_annotations_train.jsonl"
        handwriting_valid_jsonl = f"{LOCAL_OUTPUT_PATH}/handwriting_annotations_valid.jsonl"

        # Text in the wild
        if text_wild_train_jsonl not in completed_jsonls:
            base_path = f"{FTP_BASE_PATH}/13.한국어글자체/04. Text in the wild_230209_add"
            json_path = f"{MERGED_JSON_PATH}/textinthewild_data_info.json"
            if os.path.exists(json_path):
                create_jsonl_text_in_wild_split(
                    base_path,
                    json_path,
                    text_wild_train_jsonl,
                    text_wild_valid_jsonl,
                    train_ratio=0.9,
                    max_samples=max_samples_limit,
                    random_seed=42,
                )
            else:
                print(f"❌ JSON 파일을 찾을 수 없습니다: {json_path}")
        else:
            print("⏭️ Text in the wild train/valid JSONL 이미 완료됨")

        # Public Admin
        base_path = f"{FTP_BASE_PATH}/공공행정문서 OCR"
        train_json_path = f"{MERGED_JSON_PATH}/public_admin_train_merged.json"
        valid_json_path = f"{MERGED_JSON_PATH}/public_admin_valid_merged.json"
        train_partly_json_path = f"{MERGED_JSON_PATH}/public_admin_train_partly_merged.json"

        if os.path.exists(train_json_path) and public_admin_train_jsonl not in completed_jsonls:
            create_jsonl_public_admin_from_json(base_path, train_json_path, public_admin_train_jsonl, "공공행정문서 Train", max_samples_limit)
        if os.path.exists(train_partly_json_path) and public_admin_train_partly_jsonl not in completed_jsonls:
            create_jsonl_public_admin_from_json(base_path, train_partly_json_path, public_admin_train_partly_jsonl, "공공행정문서 Train Partly", max_samples_limit)
        if os.path.exists(valid_json_path) and public_admin_valid_jsonl not in completed_jsonls:
            create_jsonl_public_admin_from_json(base_path, valid_json_path, public_admin_valid_jsonl, "공공행정문서 Valid", max_samples_limit)

        # OCR Public
        base_path = f"{FTP_BASE_PATH}/023.OCR 데이터(공공)/01-1.정식개방데이터"
        train_json_path = f"{MERGED_JSON_PATH}/ocr_public_train_merged.json"
        valid_json_path = f"{MERGED_JSON_PATH}/ocr_public_valid_merged.json"
        if os.path.exists(train_json_path) and ocr_public_train_jsonl not in completed_jsonls:
            create_jsonl_ocr_public_from_json(base_path, train_json_path, ocr_public_train_jsonl, "OCR 공공 Train", max_samples_limit)
        if os.path.exists(valid_json_path) and ocr_public_valid_jsonl not in completed_jsonls:
            create_jsonl_ocr_public_from_json(base_path, valid_json_path, ocr_public_valid_jsonl, "OCR 공공 Valid", max_samples_limit)

        # Finance & Logistics
        base_path = f"{FTP_BASE_PATH}/025.OCR 데이터(금융 및 물류)/01-1.정식개방데이터"
        train_json_path = f"{MERGED_JSON_PATH}/finance_logistics_train_merged.json"
        valid_json_path = f"{MERGED_JSON_PATH}/finance_logistics_valid_merged.json"
        if os.path.exists(train_json_path) and finance_train_jsonl not in completed_jsonls:
            create_jsonl_finance_logistics_from_json(base_path, train_json_path, finance_train_jsonl, "금융물류 Train", max_samples_limit)
        if os.path.exists(valid_json_path) and finance_valid_jsonl not in completed_jsonls:
            create_jsonl_finance_logistics_from_json(base_path, valid_json_path, finance_valid_jsonl, "금융물류 Valid", max_samples_limit)

        # Handwriting
        base_path = f"{FTP_BASE_PATH}/053.대용량 손글씨 OCR 데이터/01.데이터"
        train_json_path = f"{MERGED_JSON_PATH}/handwriting_train_merged.json"
        valid_json_path = f"{MERGED_JSON_PATH}/handwriting_valid_merged.json"
        if os.path.exists(train_json_path) and handwriting_train_jsonl not in completed_jsonls:
            create_jsonl_handwriting_from_json(base_path, train_json_path, handwriting_train_jsonl, "손글씨 Train", max_samples_limit)
        if os.path.exists(valid_json_path) and handwriting_valid_jsonl not in completed_jsonls:
            create_jsonl_handwriting_from_json(base_path, valid_json_path, handwriting_valid_jsonl, "손글씨 Valid", max_samples_limit)
    else:
        # 각 데이터셋별로 train/valid LMDB 생성 (완료된 것 제외) - 전체 데이터 처리
        if f"{LOCAL_OUTPUT_PATH}/text_in_wild_annotations_train.lmdb" not in completed_lmdbs:
            create_text_in_wild_train_valid(max_samples=max_samples_limit)
        else:
            print("⏭️ Text in the wild train/valid LMDB 이미 완료됨")
    
        if f"{LOCAL_OUTPUT_PATH}/public_admin_annotations_train.lmdb" not in completed_lmdbs:
            create_public_admin_train_valid(max_samples=max_samples_limit)
        else:
            print("⏭️ 공공행정문서 OCR train/valid LMDB 이미 완료됨")
    
        if f"{LOCAL_OUTPUT_PATH}/public_admin_annotations_train_partly.lmdb" not in completed_lmdbs:
            create_public_admin_train_partly(max_samples=max_samples_limit)
        else:
            print("⏭️ 공공행정문서 OCR train_partly LMDB 이미 완료됨")
        
        if f"{LOCAL_OUTPUT_PATH}/ocr_public_annotations_train.lmdb" not in completed_lmdbs:
            create_ocr_public_train_valid(max_samples=max_samples_limit)
        else:
            print("⏭️ 023.OCR 데이터(공공) train/valid LMDB 이미 완료됨")
    
        if f"{LOCAL_OUTPUT_PATH}/finance_logistics_annotations_train.lmdb" not in completed_lmdbs:
            create_finance_logistics_train_valid(max_samples=max_samples_limit)
        else:
            print("⏭️ 025.OCR 데이터(금융 및 물류) train/valid LMDB 이미 완료됨")
    
        if f"{LOCAL_OUTPUT_PATH}/handwriting_annotations_train.lmdb" not in completed_lmdbs:
            create_handwriting_train_valid(max_samples=max_samples_limit)
        else:
            print("⏭️ 053.대용량 손글씨 OCR train/valid LMDB 이미 완료됨")
    
    print("\n" + "=" * 60)
    if use_jsonl:
        print("✅ 모든 데이터셋 train/valid JSONL 생성 완료! (전체 데이터 변환)")
        print("\n📁 생성된 JSONL 파일들:")
        for jsonl_path in jsonl_paths:
            if os.path.exists(jsonl_path):
                print(f"   - {jsonl_path}")
    else:
        print("✅ 모든 데이터셋 train/valid LMDB 생성 완료! (전체 데이터 변환)")
        print("\n📁 생성된 LMDB 파일들:")
        for lmdb_path in lmdb_paths:
            if os.path.exists(lmdb_path):
                print(f"   - {lmdb_path}")

if __name__ == '__main__':
    main() 