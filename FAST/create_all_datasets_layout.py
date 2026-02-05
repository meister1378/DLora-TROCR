#가장정상2512101700
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
import queue
import random
import sqlite3
import gc
import subprocess
from pathlib import Path
import orjson
import ijson  # 스트리밍 JSON 파싱
# import bigjson  # 제거됨 - orjson 사용
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from tqdm.contrib.concurrent import process_map, thread_map
import psutil
import time
from io import BytesIO
try:
    from turbojpeg import TurboJPEG  # libjpeg-turbo 가속 인코더
except Exception:
    TurboJPEG = None

# PIL for EXIF orientation handling (optional)
try:
    from PIL import Image, ImageOps
except Exception:
    Image = None
    ImageOps = None

# FAST 모듈 import
sys.path.append('.')
sys.path.append('FAST')  # 🚀 최적화된 lookup 함수들을 위한 경로
from dataset.fast.fast_lmdb import FAST_LMDB
from paddleocr import LayoutDetection  # LayoutDetection 통합
try:
    from paddleocr import TableCellsDetection  # 테이블 셀 탐지
except Exception:
    TableCellsDetection = None
try:
    import paddle  # GPU 사용 가능 여부 진단용 (선택)
except Exception:
    paddle = None
try:
    from pyinpaint import Inpaint  # Inpaint 마스킹
except Exception:
    Inpaint = None
try:
    from paddleocr import PaddleOCR  # 회전 탐지(Angle CLS)
except Exception:
    PaddleOCR = None

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

# ===== LayoutDetection 전역 리소스 =====
LAYOUT_MODEL = None
LAYOUT_MODEL_LOCK = threading.Lock()
LAYOUT_LABELS_TO_USE = {'text', 'paragraph_title', 'figure_title', 'doc_title', 'vision_footnote', 'number', 'abstract', 'aside_text', 'reference_content','vertical_text', 'table'}
LAYOUT_THRESHOLD = 0.5
TABLE_THRESHOLD = 0.3
# 테이블 라벨 검출은 더 낮은 임계값 사용 (환경변수로 조정 가능)
TABLE_LAYOUT_THRESHOLD = float(os.environ.get('FAST_TABLE_LAYOUT_THR', '0.3'))
LAYOUT_MODEL_NAME = os.environ.get('FAST_LAYOUT_MODEL', 'PP-DocLayoutV2')
LAYOUT_DEVICE = os.environ.get('FAST_LAYOUT_DEVICE', 'gpu').strip().lower()  # 'gpu' | 'cpu'
TABLE_DEVICE = os.environ.get('FAST_TABLE_DEVICE', 'gpu').strip().lower()    # 'gpu' | 'cpu'
GPU_ID = os.environ.get('FAST_GPU_ID', None)  # e.g., '0'

# ===== 메모리 안전 가드(코드 고정값) =====
# - 셀 검출용 테이블 크롭의 최장변 상한
CELL_CROP_MAX_SIDE = 1280
# - 전역 테이블 배처 큐의 최대 대기 크롭 수
TABLE_AGG_MAX_PENDING = 64

# ===== 전역 예측 캐시 (GPU 선계산 결과 저장) =====
PREDICTION_CACHE = {}  # key: img_path -> {'layout': [...], 'tables': [...]]}
PRED_CACHE_LOCK = threading.Lock()
# 캐시 상한 (메모리 폭주 방지) - 고정값으로 운영
PRED_CACHE_MAX = 256

def _cache_update(img_path, layout=None, tables=None, table_cells=None):
    """전역 예측 캐시에 안전하게 쓰고, 상한을 초과하면 오래된 항목부터 제거."""
    try:
        with PRED_CACHE_LOCK:
            entry = PREDICTION_CACHE.get(img_path) or {}
            if layout is not None:
                entry['layout'] = layout
            if tables is not None:
                entry['tables'] = tables
            if table_cells is not None:
                entry['table_cells'] = table_cells
            # dict는 삽입 순서 보존(Python 3.7+)
            if img_path in PREDICTION_CACHE:
                # 재삽입하여 최신으로 갱신
                del PREDICTION_CACHE[img_path]
            PREDICTION_CACHE[img_path] = entry
            # 상한 초과 시 오래된 항목부터 제거
            while len(PREDICTION_CACHE) > PRED_CACHE_MAX:
                try:
                    old_key = next(iter(PREDICTION_CACHE))
                    del PREDICTION_CACHE[old_key]
                except Exception:
                    break
    except Exception:
        pass

# ===== OpenCV 쓰레드 수 제한(과도한 오버서브스크립션 방지) =====
try:
    # OpenCV 내부 스레드를 보수적으로 제한(메모리 피크 완화)
    cv2.setNumThreads(2)
except Exception:
    pass

# ===== JPEG 인코딩 가속/품질 설정 =====
JPEG_QUALITY = int(os.environ.get("FAST_JPEG_QUALITY", "80"))
JPEG_OPTIMIZE = int(os.environ.get("FAST_JPEG_OPTIMIZE", "0"))
JPEG_PROGRESSIVE = int(os.environ.get("FAST_JPEG_PROGRESSIVE", "0"))
_jpeg = None
if TurboJPEG is not None:
    try:
        _jpeg = TurboJPEG()
    except Exception:
        _jpeg = None

def fast_encode_jpg(img):
    """
    빠른 JPEG 인코딩:
    - turbojpeg 사용 가능 시 turbojpeg로 인코딩
    - 그 외에는 OpenCV imencode + 낮은 오버헤드 옵션 사용
    반환: (ok: bool, buf: bytes-like)
    """
    if _jpeg is not None:
        try:
            buf = _jpeg.encode(
                img,
                quality=JPEG_QUALITY,
                progressive=bool(JPEG_PROGRESSIVE)
            )
            return True, buf
        except Exception:
            pass
    # OpenCV 경로
    try:
        flags = [
            int(cv2.IMWRITE_JPEG_QUALITY), int(max(1, min(100, JPEG_QUALITY))),
            int(cv2.IMWRITE_JPEG_PROGRESSIVE), int(bool(JPEG_PROGRESSIVE)),
            int(cv2.IMWRITE_JPEG_OPTIMIZE), int(bool(JPEG_OPTIMIZE)),
        ]
        ok, buf = cv2.imencode('.jpg', img, flags)
        return ok, bytes(buf) if ok else (False, None)
    except Exception:
        return False, None

# ===== 전역 GPU 프리페치 워커(지속 실행) =====
GPU_PREFETCH_QUEUE = None
GPU_PREFETCH_THREAD = None
GPU_PREFETCH_STOP = threading.Event()
GPU_PREFETCH_BATCH = int(os.environ.get("FAST_LAYOUT_BATCH", "64"))
GPU_PREFETCH_QUEUE_MAX = int(os.environ.get("FAST_PREFETCH_QUEUE", "4096"))
PREFETCH_TABLES = int(os.environ.get("FAST_PREFETCH_TABLES", "0"))  # 1이면 테이블 셀까지 백그라운드 예측

def _gpu_prefetch_worker():
    """전역 큐에서 경로를 뽑아 배치 예측 → PREDICTION_CACHE 저장을 지속 수행."""
    global GPU_PREFETCH_QUEUE
    try:
        model = get_layout_model()
    except Exception:
        model = None
    pending = []
    seen = set()
    while not GPU_PREFETCH_STOP.is_set():
        try:
            # 큐에서 빠르게 최대한 모아서 배치 구성
            try:
                p = GPU_PREFETCH_QUEUE.get(timeout=0.05)
                if p and p not in seen and os.path.exists(p):
                    seen.add(p)
                    pending.append(p)
            except Exception:
                pass
            # 배치가 차거나, stop 상태에서 잔여 처리
            if (len(pending) >= GPU_PREFETCH_BATCH) or (GPU_PREFETCH_STOP.is_set() and pending):
                batch = pending[:GPU_PREFETCH_BATCH]
                pending = pending[GPU_PREFETCH_BATCH:]
                # 예측 호출
                out_list = []
                if model and batch:
                    try:
                        with LAYOUT_MODEL_LOCK:
                            out_list = model.predict(batch, batch_size=len(batch), layout_nms=True, threshold=LAYOUT_THRESHOLD)
                    except Exception:
                        out_list = []
                # 캐시에 반영
                for pth, res in zip(batch, out_list or [None]*len(batch)):
                    boxes = []
                    try:
                        for b in getattr(res, 'boxes', []):
                            label = b.get('label')
                            coord = b.get('coordinate')
                            if label in LAYOUT_LABELS_TO_USE and isinstance(coord, (list, tuple)) and len(coord) == 4:
                                boxes.append({
                                    'label': label,
                                    'coordinate': [float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3])],
                                    'score': float(b.get('score', 1.0))
                                })
                    except Exception:
                        boxes = []
                    tables = [b for b in boxes if isinstance(b.get('label'), str) and b.get('label').lower() == 'table']
                    _cache_update(pth, layout=boxes, tables=tables)
        except Exception:
            # 워커는 절대 죽지 않도록 모든 예외 삼킴
            pass
    # 루프 종료 후 잔여 처리
    if pending:
        try:
            with LAYOUT_MODEL_LOCK:
                out_list = model.predict(pending, batch_size=len(pending), layout_nms=True, threshold=LAYOUT_THRESHOLD)
        except Exception:
            out_list = []
        for pth, res in zip(pending, out_list or [None]*len(pending)):
            boxes = []
            try:
                for b in getattr(res, 'boxes', []):
                    label = b.get('label')
                    coord = b.get('coordinate')
                    if label in LAYOUT_LABELS_TO_USE and isinstance(coord, (list, tuple)) and len(coord) == 4:
                        boxes.append({
                            'label': label,
                            'coordinate': [float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3])],
                            'score': float(b.get('score', 1.0))
                        })
            except Exception:
                boxes = []
            tables = [b for b in boxes if isinstance(b.get('label'), str) and b.get('label').lower() == 'table']
            _cache_update(pth, layout=boxes, tables=tables)

def _layout_predict_batch_numpy(img_paths, threshold):
    """여러 이미지 '경로 리스트'를 한 번에 predict하고 캐시에 저장."""
    if not img_paths:
        return
    model = get_layout_model()
    _log_verbose(f"[layout/batch] start paths={len(img_paths)} thr={threshold}")
    # 경로 리스트를 그대로 전달
    # 유효 경로만
    keeps = [p for p in img_paths if p and os.path.exists(p)]
    if not keeps:
        _log_verbose(f"[layout/batch] no valid paths")
        return
    # 보수적 배치 크기 고정
    bs = min(len(keeps), 8)
    t0 = time.time()
    try:
        with LAYOUT_MODEL_LOCK:
            out_list = model.predict(keeps, batch_size=bs, layout_nms=True, threshold=threshold)
    except Exception as e:
        _log_verbose(f"[layout/batch] predict error: {e}")
        out_list = [None] * len(keeps)
    t1 = time.time()
    _log_verbose(f"[layout/batch] predict done: n={len(keeps)} bs={bs} ms={(t1-t0)*1000:.1f}")
    # 결과 구조 디버그 (상위 3개)
    try:
        for i, r in enumerate(out_list[:3] or []):
            _debug_inspect_layout_result(r, tag=f"batch[{i}]")
    except Exception:
        pass
    # 캐시 반영
    total_boxes = 0
    total_tables = 0
    for p, res in zip(keeps, out_list or []):
        boxes = []
        try:
            raw_list = _extract_layout_boxes(res)
            for b in (raw_list or []):
                label = b.get('label')
                coord = b.get('coordinate')
                if isinstance(label, str) and label in LAYOUT_LABELS_TO_USE and isinstance(coord, (list, tuple)) and len(coord) == 4:
                    boxes.append({
                        'label': label,
                        'coordinate': [float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3])],
                        'score': float(b.get('score', 1.0))
                    })
        except Exception:
            boxes = []
        tables = [b for b in boxes if isinstance(b.get('label'), str) and b.get('label').lower() == 'table']
        total_boxes += len(boxes)
        total_tables += len(tables)
        _cache_update(p, layout=boxes, tables=tables)
    _log_verbose(f"[layout/batch] cache saved: images={len(keeps)} boxes={total_boxes} tables={total_tables}")

def _start_gpu_prefetch_worker(batch_size=None):
    """GPU 프리페치 워커를 1회만 기동."""
    global GPU_PREFETCH_QUEUE, GPU_PREFETCH_THREAD, GPU_PREFETCH_BATCH
    if GPU_PREFETCH_THREAD is not None and GPU_PREFETCH_THREAD.is_alive():
        return
    if batch_size and isinstance(batch_size, int) and batch_size > 0:
        GPU_PREFETCH_BATCH = batch_size
    GPU_PREFETCH_STOP.clear()
    GPU_PREFETCH_QUEUE = queue.Queue(maxsize=GPU_PREFETCH_QUEUE_MAX)
    GPU_PREFETCH_THREAD = threading.Thread(target=_gpu_prefetch_worker, name="GPU-Prefetch-Worker", daemon=True)
    GPU_PREFETCH_THREAD.start()

def _stop_gpu_prefetch_worker():
    """GPU 프리페치 워커 종료."""
    global GPU_PREFETCH_THREAD, GPU_PREFETCH_QUEUE
    try:
        GPU_PREFETCH_STOP.set()
        if GPU_PREFETCH_THREAD is not None:
            GPU_PREFETCH_THREAD.join(timeout=2.0)
    except Exception:
        pass
    finally:
        GPU_PREFETCH_THREAD = None
        GPU_PREFETCH_QUEUE = None

def _gpu_prefetch_enqueue(paths):
    """경로들을 전역 큐에 비차단으로 담는다(중복 허용, 워커에서 제거)."""
    global GPU_PREFETCH_QUEUE
    if not paths:
        return
    if GPU_PREFETCH_QUEUE is None:
        return
    for p in paths:
        try:
            if p and os.path.exists(p):
                GPU_PREFETCH_QUEUE.put_nowait(p)
        except Exception:
            # 큐가 가득 차면 드랍(다음 청크에서 다시 시도될 것)
            pass

# ===== 회전(Angle) 감지 리소스/캐시 =====
ROT_MODEL = None
ROT_MODEL_LOCK = threading.Lock()
ROTATION_CACHE = {}  # key: img_path -> angle (0/90/180/270 or float)
ROT_CACHE_LOCK = threading.Lock()

def get_rotation_model():
	"""회전 감지 비활성화: 더 이상 PaddleOCR 초기화하지 않음."""
	return None

def _parse_angle_from_ocr_result(one_output):
	"""PaddleOCR 결과에서 doc_preprocessor_res.angle 추출. 실패 시 0 반환."""
	try:
		dp = one_output.get('doc_preprocessor_res', {})
		angle = dp.get('angle', 0)
		if isinstance(angle, (int, float)):
			return int(angle) % 360
	except Exception:
		pass
	return 0

def _detect_rotation_batch(img_paths, batch_size=8):
	"""비활성화: 회전 각도 선계산 사용 안 함."""
	return

def _prefetch_rotations_for_args(args_list, path_extractor, batch_size=8):
	"""args 리스트에서 경로를 추출해 회전 각도를 GPU 배치로 선계산."""
	if not args_list or path_extractor is None:
		return
	paths = []
	for arg in args_list:
		try:
			p = path_extractor(arg)
		except Exception:
			p = None
		if not p or not os.path.exists(p):
			continue
		with ROT_CACHE_LOCK:
			if p in ROTATION_CACHE:
				continue
		paths.append(p)
	if not paths:
		return
	# 입력 순서 유지한 중복 제거
	paths = list(dict.fromkeys(paths))
	_detect_rotation_batch(paths, batch_size=batch_size)

def _apply_rotation_if_needed(crop_img, original_img_path):
	"""회전 보정 비활성화(더 이상 PaddleOCR 기반 회전 감지 사용 안 함)."""
	if crop_img is None or crop_img.size == 0:
		return crop_img
	return crop_img

# ===== Debug/Test 모드 (샘플 제한 및 이미지/라벨 저장) =====
# 환경변수 FAST_DEBUG 또는 DEBUG_MODE 둘 중 하나라도 참이면 디버그 모드
DEBUG_MODE = str(os.environ.get('FAST_DEBUG') or os.environ.get('DEBUG_MODE') or '0').lower() in ('1', 'true', 'yes', 'y')
DEBUG_SAMPLE_LIMIT = 500
STRICT_ID_ORDER = os.environ.get('FAST_STRICT_ID_ORDER', '1') == '1'  # 위치 무시, id 순서만 사용
# 진행 로그 토글: FAST_VERBOSE가 1/true/yes/y면 상세 로그 출력
VERBOSE_LOG = str(os.environ.get('FAST_VERBOSE', '0')).lower() in ('1', 'true', 'yes', 'y')

def _log_verbose(msg):
    try:
        if VERBOSE_LOG:
            print(msg)
    except Exception:
        pass

def _inpaint_preserve_regions(crop_img, preserve_polys, feather_ratio=0.06, dilate_ratio=0.12):
    """
    preserve_polys: list[np.ndarray(N,2)] crop 좌표계 폴리곤들(보존 영역)
    보존 영역 외를 인페인트하고, 보존 영역은 feather 블렌딩으로 원본 유지.
    """
    # 인페인트 전체 스킵 옵션
    try:
        if int(os.environ.get("FAST_INPAINT", "1")) == 0:
            return crop_img
    except Exception:
        pass
    if crop_img is None or crop_img.size == 0:
        return crop_img
    Hc, Wc = crop_img.shape[:2]
    if Hc == 0 or Wc == 0:
        return crop_img
    mask = np.ones((Hc, Wc), dtype=np.uint8) * 255
    preserve = np.zeros((Hc, Wc), dtype=np.uint8)
    if preserve_polys:
        try:
            cv2.fillPoly(mask, preserve_polys, 0)
            cv2.fillPoly(preserve, preserve_polys, 1)
        except Exception:
            pass
        # 폰트 가장자리 보존을 위해 소폭 확대 후 마스크 반전
        try:
            # 보존 폴리곤 평균 높이로 팽창 커널 결정
            hs = []
            for poly in preserve_polys:
                if poly.size == 0:
                    continue
                y1 = max(0, np.min(poly[:, 1])); y2 = min(Hc, np.max(poly[:, 1]))
                hs.append(max(1, int(y2 - y1)))
            median_h = float(np.median(hs)) if hs else 8.0
            dilate_px = int(max(2, round(dilate_ratio * median_h)))
            ksz = max(1, dilate_px * 2 + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
            preserve_d = cv2.dilate(preserve, kernel, iterations=1)
            mask = np.where(preserve_d > 0, 0, 255).astype(np.uint8)
        except Exception:
            pass
    crop_orig = crop_img.copy()
    # 인페인트
    bg_restored = None
    if 'Inpaint' in globals() and Inpaint is not None:
        try:
            inp = Inpaint(crop_img, mask)
            out = inp()
            if out is not None and out.shape == crop_img.shape:
                bg_restored = out
        except Exception:
            bg_restored = None
    if bg_restored is None:
        try:
            mask_cv = (mask > 0).astype(np.uint8) * 255
            bg_restored = cv2.inpaint(crop_img, mask_cv, 3, cv2.INPAINT_TELEA)
        except Exception:
            bg_restored = None
    if bg_restored is None:
        return crop_img
    # 보존 영역은 원본, 나머지는 인페인트 결과를 feather 블렌딩
    m = (mask == 0).astype(np.float32)
    try:
        # 가우시안 블러 스킵 옵션
        if int(os.environ.get("FAST_GBLUR", "1")) != 0:
            feather_px = max(1, int(round(feather_ratio * (median_h if 'median_h' in locals() else 12.0))))
            kf = max(1, feather_px * 2 + 1)
            m = cv2.GaussianBlur(m, (kf, kf), 0)
    except Exception:
        pass
    if len(crop_img.shape) == 3:
        m3 = np.repeat(m[:, :, None], 3, axis=2)
    else:
        m3 = m
    blended = (crop_orig.astype(np.float32) * m3 + bg_restored.astype(np.float32) * (1.0 - m3)).clip(0, 255).astype(crop_orig.dtype)
    return blended

def _decode_image_bytes(img_bytes):
    """바이트를 OpenCV 이미지로 디코딩."""
    try:
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def _assign_words_to_layout(word_aabbs, layout_aabbs, min_overlap_ratio=0.5):
    """각 단어를 가장 크게 겹치는 레이아웃에 할당. 반환: word_index -> layout_index (미할당은 -1)."""
    assigned = [-1] * len(word_aabbs)
    for wi, wa in enumerate(word_aabbs):
        wa_area = _area(wa)
        if wa_area <= 0:
            continue
        best_idx = -1
        best_ratio = 0.0
        for li, la in enumerate(layout_aabbs):
            inter = _intersection_area(wa, la)
            ratio = inter / wa_area if wa_area > 0 else 0.0
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = li
        if best_idx >= 0 and best_ratio >= min_overlap_ratio:
            assigned[wi] = best_idx
    return assigned

 

def get_layout_model():
    """LayoutDetection 모델 싱글톤 초기화/반환."""
    global LAYOUT_MODEL
    if LAYOUT_MODEL is None:
        with LAYOUT_MODEL_LOCK:
            if LAYOUT_MODEL is None:
                # 디바이스 강제 설정 (폴백 없음)
                if LAYOUT_DEVICE == 'gpu':
                    if paddle is None or not paddle.is_compiled_with_cuda():
                        raise RuntimeError("[layout] PaddlePaddle CUDA 빌드가 필요합니다 (현재 GPU 요청).")
                    if GPU_ID is not None:
                        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
                        try:
                            paddle.device.set_device(f"gpu:{int(GPU_ID)}")
                        except Exception:
                            paddle.device.set_device("gpu")
                    else:
                        paddle.device.set_device("gpu")
                else:
                    if paddle is not None:
                        paddle.device.set_device("cpu")
                print(f"[layout] init LayoutDetection(model={LAYOUT_MODEL_NAME}, device={LAYOUT_DEVICE}, gpu_id={GPU_ID})")
                # 인자 호환성: 디바이스는 paddle 전역 설정으로 강제
                LAYOUT_MODEL = LayoutDetection(model_name=LAYOUT_MODEL_NAME)
                try:
                    if paddle is not None:
                        print(f"[layout] current device: {paddle.device.get_device()}")
                except Exception:
                    pass
    return LAYOUT_MODEL

TABLE_MODEL = None
TABLE_MODEL_LOCK = threading.Lock()

PUBLIC_ADMIN_SQLITE_PATH = None
PUBLIC_ADMIN_SQLITE_CONN = None

def _get_public_admin_sqlite_conn(sqlite_path):
    """공공행정문서 어노테이션 임시 SQLite 커넥션(싱글톤)"""
    global PUBLIC_ADMIN_SQLITE_CONN, PUBLIC_ADMIN_SQLITE_PATH
    if PUBLIC_ADMIN_SQLITE_CONN is not None and PUBLIC_ADMIN_SQLITE_PATH == sqlite_path:
        return PUBLIC_ADMIN_SQLITE_CONN
    # 새로 열기
    try:
        conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=OFF;")
        PUBLIC_ADMIN_SQLITE_CONN = conn
        PUBLIC_ADMIN_SQLITE_PATH = sqlite_path
        return conn
    except Exception:
        return None

def get_table_model():
    """TableCellsDetection 모델 싱글톤 초기화/반환."""
    global TABLE_MODEL
    if TableCellsDetection is None:
        print("[table] TableCellsDetection 모듈을 불러오지 못했습니다. 셀 검출이 비활성화됩니다.")
        return None
    if TABLE_MODEL is None:
        with TABLE_MODEL_LOCK:
            if TABLE_MODEL is None:
                # 디바이스 강제 설정 (폴백 없음)
                if TABLE_DEVICE == 'gpu':
                    if paddle is None or not paddle.is_compiled_with_cuda():
                        raise RuntimeError("[table] PaddlePaddle CUDA 빌드가 필요합니다 (현재 GPU 요청).")
                    if GPU_ID is not None:
                        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
                        try:
                            paddle.device.set_device(f"gpu:{int(GPU_ID)}")
                        except Exception:
                            paddle.device.set_device("gpu")
                    else:
                        paddle.device.set_device("gpu")
                else:
                    if paddle is not None:
                        paddle.device.set_device("cpu")
                print(f"[table] init TableCellsDetection(model=RT-DETR-L_wired_table_cell_det, device={TABLE_DEVICE}, gpu_id={GPU_ID})")
                # 인자 호환성: 디바이스는 paddle 전역 설정으로 강제
                TABLE_MODEL = TableCellsDetection(model_name="RT-DETR-L_wired_table_cell_det")
                try:
                    if paddle is not None:
                        print(f"[table] current device: {paddle.device.get_device()}")
                except Exception:
                    pass
    return TABLE_MODEL

def _to_flat8_from_xyxy(x1, y1, x2, y2):
    """사각형 [x1,y1,x2,y2]를 8좌표(flat8)로 변환."""
    return [float(x1), float(y1), float(x2), float(y1), float(x2), float(y2), float(x1), float(y2)]

def _aabb_from_flat8(bflat8):
    """flat8 bbox에서 AABB(minx,miny,maxx,maxy)로 변환."""
    xs = bflat8[0::2]
    ys = bflat8[1::2]
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))

def _rotate_flat8_180(bflat8, img_w, img_h):
    """이미지 중심 기준 180도 회전된 flat8 좌표 반환 (축 정렬 박스 가정)."""
    x1, y1, x2, y2 = _aabb_from_flat8(bflat8)
    rx1 = float(max(0.0, min(img_w, img_w - x2)))
    ry1 = float(max(0.0, min(img_h, img_h - y2)))
    rx2 = float(max(0.0, min(img_w, img_w - x1)))
    ry2 = float(max(0.0, min(img_h, img_h - y1)))
    return [rx1, ry1, rx2, ry1, rx2, ry2, rx1, ry2]

def _rotate_flat8_90_cw(bflat8, img_w, img_h):
    """이미지 기준 90도 시계 회전 flat8 변환. 입력은 원본(W,H), 출력 좌표계는 (H,W)."""
    # 점들 생성
    pts = [
        (bflat8[0], bflat8[1]),
        (bflat8[2], bflat8[3]),
        (bflat8[4], bflat8[5]),
        (bflat8[6], bflat8[7]),
    ]
    # (x', y') = (H - y, x)
    pts_r = [(float(img_h - y), float(x)) for (x, y) in pts]
    xs = [p[0] for p in pts_r]
    ys = [p[1] for p in pts_r]
    rx1, ry1, rx2, ry2 = min(xs), min(ys), max(xs), max(ys)
    # 출력 좌표계는 (H, W)
    rx1 = max(0.0, min(img_h, rx1)); rx2 = max(0.0, min(img_h, rx2))
    ry1 = max(0.0, min(img_w, ry1)); ry2 = max(0.0, min(img_w, ry2))
    return [rx1, ry1, rx2, ry1, rx2, ry2, rx1, ry2]

def _rotate_flat8_270_cw(bflat8, img_w, img_h):
    """이미지 기준 270도 시계(=90도 반시계) 회전 flat8 변환. 입력은 원본(W,H), 출력 좌표계는 (H,W)."""
    # (x', y') = (y, W - x)
    pts = [
        (bflat8[0], bflat8[1]),
        (bflat8[2], bflat8[3]),
        (bflat8[4], bflat8[5]),
        (bflat8[6], bflat8[7]),
    ]
    pts_r = [(float(y), float(img_w - x)) for (x, y) in pts]
    xs = [p[0] for p in pts_r]
    ys = [p[1] for p in pts_r]
    rx1, ry1, rx2, ry2 = min(xs), min(ys), max(xs), max(ys)
    rx1 = max(0.0, min(img_h, rx1)); rx2 = max(0.0, min(img_h, rx2))
    ry1 = max(0.0, min(img_w, ry1)); ry2 = max(0.0, min(img_w, ry2))
    return [rx1, ry1, rx2, ry1, rx2, ry2, rx1, ry2]

def _intersection_area(a, b):
    """두 AABB의 교차 면적."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    w = max(0.0, ix2 - ix1)
    h = max(0.0, iy2 - iy1)
    return w * h

def _area(aabb):
    """AABB 면적."""
    x1, y1, x2, y2 = aabb
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def _extract_layout_boxes(res):
    """
    LayoutDetection 결과 객체에서 박스 리스트를 표준화해 추출.
    지원 키: boxes | result | preds | predictions | bbox/box/coordinate/points
    반환: [{'label': str, 'coordinate': [x1,y1,x2,y2], 'score': float}, ...]
    """
    candidates = []
    try:
        # dict 형태
        if isinstance(res, dict):
            for k in ('boxes', 'result', 'preds', 'predictions'):
                v = res.get(k)
                if isinstance(v, (list, tuple)):
                    candidates = v
                    break
        else:
            # 객체 속성
            for k in ('boxes', 'result', 'preds', 'predictions'):
                v = getattr(res, k, None)
                if isinstance(v, (list, tuple)):
                    candidates = v
                    break
    except Exception:
        candidates = []
    norm = []
    for b in candidates or []:
        try:
            if isinstance(b, dict):
                label = b.get('label')
                coord = b.get('coordinate')
                if coord is None:
                    coord = b.get('bbox') or b.get('box')
                if coord is None:
                    pts = b.get('points') or b.get('poly') or b.get('polygon')
                    if isinstance(pts, (list, tuple)) and len(pts) >= 4:
                        xs = [p[0] for p in pts[:4]]
                        ys = [p[1] for p in pts[:4]]
                        coord = [min(xs), min(ys), max(xs), max(ys)]
                score = b.get('score', b.get('confidence', 1.0))
            else:
                label = getattr(b, 'label', None)
                coord = getattr(b, 'coordinate', None)
                if coord is None:
                    coord = getattr(b, 'bbox', None) or getattr(b, 'box', None)
                score = getattr(b, 'score', getattr(b, 'confidence', 1.0))
            if isinstance(coord, (list, tuple)) and len(coord) >= 4:
                x1, y1, x2, y2 = float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3])
                norm.append({'label': label, 'coordinate': [x1, y1, x2, y2], 'score': float(score if isinstance(score, (int, float)) else 1.0)})
        except Exception:
            continue
    return norm
def _extract_cell_boxes(res):
    """
    TableCellsDetection 결과 객체에서 'cell' 박스만 표준화해 추출.
    지원 키: boxes | result | preds | predictions | bbox/box/coordinate/points
    반환: [{'label': 'cell', 'coordinate': [x1,y1,x2,y2], 'score': float}, ...]
    """
    candidates = []
    try:
        if isinstance(res, dict):
            for k in ('boxes', 'result', 'preds', 'predictions'):
                v = res.get(k)
                if isinstance(v, (list, tuple)):
                    candidates = v
                    break
        else:
            for k in ('boxes', 'result', 'preds', 'predictions'):
                v = getattr(res, k, None)
                if isinstance(v, (list, tuple)):
                    candidates = v
                    break
    except Exception:
        candidates = []
    norm = []
    labels_count = {}
    for b in candidates or []:
        try:
            if isinstance(b, dict):
                label = b.get('label')
                coord = b.get('coordinate') or b.get('bbox') or b.get('box')
                if coord is None:
                    pts = b.get('points') or b.get('poly') or b.get('polygon')
                    if isinstance(pts, (list, tuple)) and len(pts) >= 4:
                        xs = [p[0] for p in pts[:4]]
                        ys = [p[1] for p in pts[:4]]
                        coord = [min(xs), min(ys), max(xs), max(ys)]
                score = b.get('score', b.get('confidence', 1.0))
            else:
                label = getattr(b, 'label', None)
                coord = getattr(b, 'coordinate', None) or getattr(b, 'bbox', None) or getattr(b, 'box', None)
                score = getattr(b, 'score', getattr(b, 'confidence', 1.0))
            try:
                labels_count[label] = labels_count.get(label, 0) + 1
            except Exception:
                pass
            if isinstance(coord, (list, tuple)) and len(coord) >= 4:
                x1, y1, x2, y2 = float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3])
                # 라벨이 문자열이고 'cell' 포함시만 사용. 라벨이 None/숫자인 경우는 제외(디버그로 확인)
                if isinstance(label, str) and 'cell' in label.lower():
                    norm.append({'label': 'cell', 'coordinate': [x1, y1, x2, y2], 'score': float(score if isinstance(score, (int, float)) else 1.0)})
        except Exception:
            continue
    # 라벨 분포 디버깅
    try:
        print(f"[debug] cell_extract: labels={labels_count} kept={len(norm)}")
    except Exception:
        pass
    return norm
def _sort_word_indices_by_reading_order(word_aabbs):
    """좌->우, 상->하 간단 정렬 키로 인덱스 반환."""
    # y 우선(상단), 다음 x
    indices = list(range(len(word_aabbs)))
    indices.sort(key=lambda i: (word_aabbs[i][1], word_aabbs[i][0]))
    return indices

def _debug_inspect_layout_result(res, tag=""):
    """
    LayoutDetection 단일 결과 객체 구조 간단 점검용 디버그.
    - 타입
    - dict 키 / 대표 키들의 길이
    - 객체 속성 존재 여부
    """
    try:
        info = {'tag': tag, 'type': type(res).__name__}
        if isinstance(res, dict):
            keys = list(res.keys())
            info['dict_keys'] = keys[:16]
            for k in ('boxes', 'result', 'preds', 'predictions'):
                v = res.get(k, None)
                info[f'len_{k}'] = (len(v) if isinstance(v, (list, tuple)) else (-1 if v is None else 1))
        else:
            for k in ('boxes', 'result', 'preds', 'predictions'):
                v = getattr(res, k, None)
                info[f'len_{k}'] = (len(v) if isinstance(v, (list, tuple)) else (-1 if v is None else 1))
        _log_verbose(f"[layout/debug] {info}")
    except Exception:
        pass

def run_layout_detection(img_path):
    """이미지 경로로 LayoutDetection 수행하고, 필요한 라벨의 [x1,y1,x2,y2] 리스트 반환."""
    try:
        # 캐시 우선 사용
        try:
            with PRED_CACHE_LOCK:
                cached = PREDICTION_CACHE.get(img_path)
                if cached and 'layout' in cached:
                    return cached.get('layout') or []
        except Exception:
            pass
        model = get_layout_model()
        _log_verbose(f"[layout] call predict: path={img_path} thr={LAYOUT_THRESHOLD}")
        t0 = time.time()
        with LAYOUT_MODEL_LOCK:
            # 경로 문자열(또는 경로 리스트)을 그대로 전달
            output = model.predict(img_path, batch_size=8, layout_nms=True, threshold=LAYOUT_THRESHOLD)
        t1 = time.time()
        _log_verbose(f"[layout] predict(ms)={(t1-t0)*1000:.1f} path={os.path.basename(img_path)}")
        if not output:
            _log_verbose(f"[layout] result: 0 boxes")
            return []
        res = output[0]
        # 결과 파싱
        boxes = []
        try:
            raw_boxes = _extract_layout_boxes(res)
            labels_count = {}
            for b in raw_boxes:
                label = b.get('label')
                coord = b.get('coordinate')
                try:
                    labels_count[label] = labels_count.get(label, 0) + 1
                except Exception:
                    pass
                if label in LAYOUT_LABELS_TO_USE and isinstance(coord, (list, tuple)) and len(coord) == 4:
                    boxes.append({'label': label, 'coordinate': [float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3])], 'score': float(b.get('score', 1.0))})
            _log_verbose(f"[layout] result: boxes_total={len(raw_boxes)} boxes_used={len(boxes)} labels={labels_count}")
            if boxes:
                _cache_update(img_path, layout=boxes)
                return boxes
        except Exception:
            pass
    except Exception:
        _log_verbose(f"[layout] exception in predict for {img_path}")
        return []

def run_layout_tables(img_path):
    """이미지에서 layout 결과 중 table 라벨만 반환."""
    try:
        print(f"[debug] run_layout_tables: path={img_path}")
        # 캐시 우선 사용
        try:
            with PRED_CACHE_LOCK:
                cached = PREDICTION_CACHE.get(img_path)
                if cached and 'tables' in cached:
                    # 캐시가 존재하면 비어 있어도 재추론하지 않고 그대로 반환
                    tables_cached = cached.get('tables') or []
                    print(f"[debug] run_layout_tables: cached tables={len(tables_cached)}")
                    return tables_cached
        except Exception:
            pass
        model = get_layout_model()
        _log_verbose(f"[layout] call predict(for table): path={img_path} thr={TABLE_LAYOUT_THRESHOLD}")
        with LAYOUT_MODEL_LOCK:
            output = model.predict(img_path, batch_size=8, layout_nms=True, threshold=TABLE_LAYOUT_THRESHOLD)
        if not output or not output[0]:
            _log_verbose(f"[layout] table result: 0 boxes")
            print(f"[debug] run_layout_tables: model returned 0")
            return []
        res = output[0]
        tables = []
        try:
            raw_boxes = _extract_layout_boxes(res)
            print(f"[debug] run_layout_tables: raw_boxes={len(raw_boxes)}")
            labels_count = {}
            for b in raw_boxes:
                label = b.get('label')
                try:
                    labels_count[label] = labels_count.get(label, 0) + 1
                except Exception:
                    pass
                coord = b.get('coordinate')
                if isinstance(label, str) and label.lower() == 'table' and isinstance(coord, (list, tuple)) and len(coord) == 4:
                    tables.append({'label': label, 'coordinate': [float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3])], 'score': float(b.get('score', 1.0))})
            _log_verbose(f"[layout] table result: total={len(raw_boxes)} tables={len(tables)} labels={labels_count}")
            print(f"[debug] run_layout_tables: tables={len(tables)} sample={tables[:3] if tables else []}")
            _cache_update(img_path, tables=tables)
            return tables
        except Exception:
            pass
        # JSON 폴백 완전 제거
    except Exception:
        _log_verbose(f"[layout] exception in predict for {img_path}")
        return []

def _flat8_to_crop_poly(flat8, crop_x1, crop_y1, crop_x2, crop_y2):
    """단어 flat8을 크롭 좌표계 폴리곤(np.int32)로 변환. 크롭 밖은 잘림."""
    try:
        xs = [float(flat8[0]), float(flat8[2]), float(flat8[4]), float(flat8[6])]
        ys = [float(flat8[1]), float(flat8[3]), float(flat8[5]), float(flat8[7])]
        poly = []
        for i in range(4):
            xi = int(round(xs[i] - crop_x1))
            yi = int(round(ys[i] - crop_y1))
            poly.append([xi, yi])
        poly = np.array(poly, dtype=np.int32)
        Hc = max(0, int(crop_y2 - crop_y1))
        Wc = max(0, int(crop_x2 - crop_x1))
        if Hc <= 0 or Wc <= 0:
            return None
        poly[:, 0] = np.clip(poly[:, 0], 0, Wc)
        poly[:, 1] = np.clip(poly[:, 1], 0, Hc)
        return poly
    except Exception:
        return None

def merge_words_by_layout(bboxes_flat8, words, layout_boxes, word_ids=None, prefer_id_order=False, word_orients=None):
    """
    단어 단위 라벨(bboxes_flat8/words)을 LayoutDetection 박스별 문장으로 병합.
    layout_boxes: [{'label': str, 'coordinate': [x1,y1,x2,y2], 'score': float}, ...]
    반환: (merged_bboxes_flat8, merged_texts)
    """
    if not bboxes_flat8 or not words or not layout_boxes:
        return [], []
    word_aabbs = [_aabb_from_flat8(b) for b in bboxes_flat8]
    assigned = [-1] * len(words)  # word -> layout idx

    layout_aabbs = []
    for lb in layout_boxes:
        x1, y1, x2, y2 = lb['coordinate']
        layout_aabbs.append((float(x1), float(y1), float(x2), float(y2)))

    # 각 단어를 가장 많이 겹치는 레이아웃 박스에 할당 (단어 면적 대비 0.5 이상 겹치면 할당)
    for wi, wa in enumerate(word_aabbs):
        best_idx = -1
        best_ratio = 0.0
        wa_area = _area(wa)
        if wa_area <= 0.0:
            continue
        for li, la in enumerate(layout_aabbs):
            inter = _intersection_area(wa, la)
            ratio = inter / wa_area if wa_area > 0 else 0.0
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = li
        if best_idx >= 0 and best_ratio >= 0.5:
            assigned[wi] = best_idx

    # 레이아웃 박스별로 단어 수집 후 정렬/병합
    merged_bboxes = []
    merged_texts = []
    def _id_key(i):
        if word_ids is None:
            return i
        try:
            return int(str(word_ids[i]))
        except Exception:
            return str(word_ids[i])
    def _compose_multiline_sentence(idxs):
        """단어 인덱스 리스트를 개행 포함 문장으로 결합."""
        if not idxs:
            return ""
        # word_ud/word_du가 포함되었거나, 레이아웃이 세로 지배적(높이/너비 비 ≥ 2)인 경우: 컬럼 클러스터링
        force_column = False
        has_orient = False
        if word_orients is not None:
            try:
                has_orient = any(((i < len(word_orients)) and (word_orients[i] in ('ud', 'du'))) for i in idxs)
            except Exception:
                has_orient = False
        if not has_orient:
            # 세로 지배도 휴리스틱
            try:
                xs = []; ys = []
                for j in idxs:
                    x1j, y1j, x2j, y2j = word_aabbs[j]
                    xs += [x1j, x2j]; ys += [y1j, y2j]
                span_x = max(xs) - min(xs) if xs else 0.0
                span_y = max(ys) - min(ys) if ys else 0.0
                force_column = (span_y > 0 and span_x > 0 and (span_y / max(1.0, span_x) >= 2.0))
            except Exception:
                force_column = False
        if has_orient or force_column:
            # x중심/폭 기반 컬럼 클러스터링
            widths = []
            xcenters = {}
            ycenters = {}
            for j in idxs:
                x1j, y1j, x2j, y2j = word_aabbs[j]
                widths.append(max(1.0, abs(x2j - x1j)))
                xcenters[j] = (x1j + x2j) * 0.5
                ycenters[j] = (y1j + y2j) * 0.5
            median_w = float(np.median(widths)) if widths else 8.0
            x_thresh = max(10.0, 0.8 * median_w)
            # x 정렬 후 컬럼 묶기
            order_x = sorted(idxs, key=lambda i: xcenters[i])
            columns = []
            cur = []
            basex = None
            for j in order_x:
                cx = xcenters[j]
                if basex is None:
                    basex = cx; cur = [j]
                else:
                    if abs(cx - basex) > x_thresh:
                        if cur:
                            columns.append(cur)
                        basex = cx; cur = [j]
                    else:
                        cur.append(j)
            if cur:
                columns.append(cur)
            # 각 컬럼 내부 방향성 결정 및 y 정렬
            col_strings = []
            for col in columns:
                if word_orients is not None:
                    cnt_ud_c = sum(1 for j in col if (j < len(word_orients) and word_orients[j] == 'ud'))
                    cnt_du_c = sum(1 for j in col if (j < len(word_orients) and word_orients[j] == 'du'))
                else:
                    cnt_ud_c = cnt_du_c = 0
                reverse_c = (cnt_ud_c > cnt_du_c)  # True면 아래→위
                col_sorted = sorted(col, key=lambda j: ycenters[j], reverse=reverse_c)
                col_str = " ".join(str(words[t]) if words[t] is not None else "" for t in col_sorted).strip()
                if col_str:
                    col_strings.append(col_str)
            return "\n".join(col_strings).strip()
        # 엄격 ID 순서 모드: ID 순서를 따르되, y축 변동이 임계치를 넘으면 개행
        if prefer_id_order and STRICT_ID_ORDER:
            # 1) 우선 y축 기준으로 라인 클러스터링 → 줄 단위 안정화
            stats = []
            for j in idxs:
                x1i = word_aabbs[j][0]
                y1i = word_aabbs[j][1]
                x2i = word_aabbs[j][2]
                y2i = word_aabbs[j][3]
                cx = (x1i + x2i) / 2.0
                cy = (y1i + y2i) / 2.0
                h = max(1.0, y2i - y1i)
                w = max(1.0, x2i - x1i)
                stats.append((j, cx, cy, h, w))
            if not stats:
                return ""
            median_h = float(np.median([s[3] for s in stats])) if stats else 8.0
            median_w = float(np.median([s[4] for s in stats])) if stats else 12.0
            y_thresh = max(8.0, 0.6 * median_h)
            x_thresh = max(8.0, 0.6 * median_w)
            # y 중심으로 정렬 후 라인 클러스터
            stats.sort(key=lambda t: t[2])
            lines_idx = []
            cur = []
            cur_y = stats[0][2]
            for j, cx, cy, h, w in stats:
                if cur and abs(cy - cur_y) > y_thresh:
                    lines_idx.append(cur)
                    cur = [j]
                    cur_y = cy
                else:
                    cur.append(j)
                    # y 기준선은 과도하게 이동하지 않게 완만히 업데이트
                    cur_y = 0.7 * cur_y + 0.3 * cy
            if cur:
                lines_idx.append(cur)
            # 2) 각 라인 내에서는 좌→우 우선, ID는 보조 키로 안정화
            out_lines = []
            for line in lines_idx:
                # 왼쪽으로 큰 랩어라운드가 발생하면(AND 조건) 서브 라인 분할
                sub_lines = []
                sub = []
                # ID 순으로 먼저 훑어보며 랩어라운드 감지
                for j in sorted(line, key=lambda t: _id_key(t)):
                    cx = (word_aabbs[j][0] + word_aabbs[j][2]) / 2.0
                    cy = (word_aabbs[j][1] + word_aabbs[j][3]) / 2.0
                    if sub:
                        prev = sub[-1]
                        prev_cx = (word_aabbs[prev][0] + word_aabbs[prev][2]) / 2.0
                        basey = np.mean([(word_aabbs[k][1] + word_aabbs[k][3]) / 2.0 for k in sub])
                        large_y = abs(cy - basey) > y_thresh
                        large_x_wrap = (cx < prev_cx - x_thresh) and (abs(cy - basey) > 0.25 * median_h)
                        if large_y and large_x_wrap:
                            sub_lines.append(sub)
                            sub = [j]
                            continue
                    sub.append(j)
                if sub:
                    sub_lines.append(sub)
                # 최종: 각 서브라인은 x 우선, id 보조로 정렬 후 합치기
                for sline in sub_lines:
                    sline.sort(key=lambda t: (((word_aabbs[t][0] + word_aabbs[t][2]) / 2.0), _id_key(t)))
                    out_lines.append(" ".join(str(words[t]) if words[t] is not None else "" for t in sline).strip())
            return "\n".join([ln for ln in out_lines if ln])
        # 단어 중심/높이/가로 중심 계산
        stats = []
        for i in idxs:
            x1, y1, x2, y2 = word_aabbs[i][0], word_aabbs[i][1], word_aabbs[i][2], word_aabbs[i][3]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            h = max(1.0, y2 - y1)
            stats.append((i, cx, cy, h))
        if not stats:
            return ""
        median_h = float(np.median([s[3] for s in stats]))
        y_thresh = max(8.0, 0.6 * median_h)
        # 우선 y중심으로 정렬 후 라인 클러스터링
        stats.sort(key=lambda t: t[2])  # by cy
        lines_idx = []
        cur_line = []
        if stats:
            cur_y = stats[0][2]
        else:
            cur_y = 0.0
        for i, cx, cy, h in stats:
            if cur_line and abs(cy - cur_y) > y_thresh:
                lines_idx.append(cur_line)
                cur_line = [i]
                cur_y = cy
            else:
                cur_line.append(i)
                cur_y = 0.7 * cur_y + 0.3 * cy
        if cur_line:
            lines_idx.append(cur_line)
        # 라인 순서 안정화: 라인의 평균 y로 정렬
        def line_key(line):
            ys = []
            for j in line:
                ys.append((word_aabbs[j][1] + word_aabbs[j][3]) / 2.0)
            return np.mean(ys) if ys else 0.0
        lines_idx.sort(key=line_key)
        # 각 라인 내 정렬: 좌→우 우선, 필요시 id로 보조 안정화
        out_lines = []
        for line in lines_idx:
            # 좌->우 정렬
            line.sort(key=lambda j: ((word_aabbs[j][0] + word_aabbs[j][2]) / 2.0))
            if prefer_id_order and word_ids is not None:
                # 같은 x 근처일 때 id 순서를 보조로 사용 (안정화)
                line.sort(key=lambda j: (_id_key(j)))
                # 최종적으로 x 우선, id 보조의 안정화 정렬
                line.sort(key=lambda j: (((word_aabbs[j][0] + word_aabbs[j][2]) / 2.0), _id_key(j)))
            out_lines.append(" ".join(str(words[j]) if words[j] is not None else "" for j in line).strip())
        return "\n".join([ln for ln in out_lines if ln])
    for li, la in enumerate(layout_aabbs):
        word_indices = [i for i, a in enumerate(assigned) if a == li]
        if not word_indices:
            continue
        # 정렬: id 우선 옵션, 없으면 좌->우 상->하
        text = _compose_multiline_sentence(word_indices)
        if not text:
            continue
        x1, y1, x2, y2 = la
        flat8 = _to_flat8_from_xyxy(x1, y1, x2, y2)
        flat8 = normalize_ic15_clockwise_flat8(flat8)
        merged_bboxes.append(flat8)
        merged_texts.append(text)

    # 레이아웃에 할당되지 않은 단어는 그대로 보존
    for i, a in enumerate(assigned):
        if a == -1:
            merged_bboxes.append(bboxes_flat8[i])
            merged_texts.append(words[i])

    return merged_bboxes, merged_texts

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
    train_output_path = f"{LOCAL_OUTPUT_PATH}/text_in_wild_train_layout.lmdb"
    valid_output_path = f"{LOCAL_OUTPUT_PATH}/text_in_wild_valid_layout.lmdb"
    
    if os.path.exists(json_path):
        create_lmdb_text_in_wild_split(base_path, json_path, train_output_path, valid_output_path, 
                                     train_ratio=0.9, max_samples=max_samples, random_seed=42)
        
        test_fast_model_input(train_output_path)
        test_fast_model_input(valid_output_path)
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
        # 라벨 키 대소문자/문자열/불리언 혼재 대비
        def _as_bool(v):
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return v != 0
            if isinstance(v, str):
                return v.strip().lower() in ('1','true','t','y','yes')
            return False
        meta = {}
        try:
            # 평탄화 없이 키만 소문자화하여 조회
            for k, v in (img_info or {}).items():
                if isinstance(k, str):
                    meta[k.lower()] = v
        except Exception:
            pass
        usd_flag = _as_bool(meta.get('usd', False))  # 180도
        ud_flag  = _as_bool(meta.get('ud', False))   # 270도(반시계)
        du_flag  = _as_bool(meta.get('du', False))   # 90도(시계)
        try:
            img_w = int(img_info.get('width') or 0)
            img_h = int(img_info.get('height') or 0)
        except Exception:
            img_w = 0
            img_h = 0
        word_ids = []
        char_bboxes = []
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
            _log_verbose(f"[ocr_public] early-exit: image path not found {img_file_name}")
            return None
        
        # 이미지 로드 후 회전 플래그에 따라 즉시 회전 적용 (레이아웃 이전 단계)
        with open(img_path, 'rb') as f:
            img_data = f.read()
        layout_img_path = img_path
        tmp_rot_path = None
        if usd_flag or ud_flag or du_flag:
            try:
                img_cv = _decode_image_bytes(img_data)
                if img_cv is not None and img_cv.size > 0:
                    # 원본 이미지 크기(회전 전) 확보: JSON width/height가 0이거나 누락된 경우 보정
                    try:
                        oh, ow = img_cv.shape[:2]
                        if (img_w or 0) <= 0 or (img_h or 0) <= 0:
                            img_w, img_h = int(ow), int(oh)
                    except Exception:
                        pass
                    if usd_flag:
                        img_cv = cv2.rotate(img_cv, cv2.ROTATE_180)
                    elif ud_flag:
                        # ud: 90도 반시계(= 270도 시계와 동일)
                        img_cv = cv2.rotate(img_cv, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif du_flag:
                        # du: 90도 시계
                        img_cv = cv2.rotate(img_cv, cv2.ROTATE_90_CLOCKWISE)
                    ok, buf = fast_encode_jpg(img_cv)
                    if ok:
                        img_data = bytes(buf)
                        # 레이아웃은 경로 기반이므로 임시 파일로 전달
                        import tempfile, os as _os
                        fd, tmp_rot_path = tempfile.mkstemp(prefix="tiw_rot_", suffix=".jpg")
                        _os.write(fd, img_data)
                        _os.close(fd)
                        layout_img_path = tmp_rot_path
                        try:
                            ang = 180 if usd_flag else (270 if ud_flag else 90)  # 표시용: usd=180, ud=270(CCW), du=90(CW)
                        except Exception:
                            ang = -1
                        # 회전 적용은 기본 로그 토글과 무관하게 1줄은 남겨 확인 용이하게 함
                        print(f"[tiw][rotate] applied angle={ang} file={os.path.basename(img_path)} tmp={os.path.basename(layout_img_path)}")
            except Exception:
                pass
        
        # 어노테이션 처리
        bboxes = []
        words = []
        word_ids = []
        # 단어 단위 방향성 플래그(ud/du) 기록
        word_orients = []
        
        for ann in annotations:
            attrs = ann.get('attributes', {})
            # character 박스는 마스킹용으로만 수집
            try:
                if isinstance(attrs, dict) and str(attrs.get('class', '')).lower() == 'character':
                    cx, cy, cw, ch = ann['bbox']
                    cx1, cy1, cx2, cy2 = cx, cy, cx + cw, cy + ch
                    cflat = [cx1, cy1, cx2, cy1, cx2, cy2, cx1, cy2]
                    if img_w > 0 and img_h > 0:
                        if usd_flag:
                            cflat = _rotate_flat8_180(cflat, img_w, img_h)
                        elif ud_flag:
                            # ud: 90도 반시계(=270도 시계)
                            cflat = _rotate_flat8_270_cw(cflat, img_w, img_h)
                        elif du_flag:
                            # du: 90도 시계
                            cflat = _rotate_flat8_90_cw(cflat, img_w, img_h)
                    char_bboxes.append(cflat)
            except Exception:
                pass
            # attributes.class가 'word'인 경우만 라벨 처리
            attrs = ann.get('attributes', {})
            cls_value = None
            if isinstance(attrs, dict):
                cls_value = str(attrs.get('class', '')).lower()
            if cls_value != 'word':
                continue
            # bbox: [x, y, width, height] -> [x1, y1, x2, y1, x2, y2, x1, y2]
            x, y, w, h = ann['bbox']
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            # 원본 좌표를 그대로 사용 (클리핑 없음)
            pixel_coords = [x1, y1, x2, y1, x2, y2, x1, y2]
            if img_w > 0 and img_h > 0:
                if usd_flag:
                    pixel_coords = _rotate_flat8_180(pixel_coords, img_w, img_h)
                elif ud_flag:
                    # ud: 270도(반시계)
                    pixel_coords = _rotate_flat8_270_cw(pixel_coords, img_w, img_h)
                elif du_flag:
                    # du: 90도(시계)
                    pixel_coords = _rotate_flat8_90_cw(pixel_coords, img_w, img_h)
            
            # bbox 형태 한 번만 출력
            if not bbox_debug_flags['text_in_wild']:
                print(f"📋 Text in Wild bbox 형태: 원본 [x={x}, y={y}, w={w}, h={h}] -> 통일 [x1={x1}, y1={y1}, x2={x2}, y1={y1}, x2={x2}, y2={y2}, x1={x1}, y2={y2}]")
                bbox_debug_flags['text_in_wild'] = True
            
            bboxes.append(pixel_coords)
            words.append(ann['text'])
            # id 수집
            word_ids.append(ann.get('id'))
            # annotation 단위 방향성(word_ud/word_du) 플래그 수집
            try:
                attrs_lc = {}
                if isinstance(attrs, dict):
                    for k, v in attrs.items():
                        if isinstance(k, str):
                            attrs_lc[k.lower()] = v
                w_ud = False
                w_du = False
                if attrs_lc:
                    # 문자열/불리언/숫자 혼용 대응
                    def _as_bool2(v):
                        if isinstance(v, bool): return v
                        if isinstance(v, (int, float)): return v != 0
                        if isinstance(v, str): return v.strip().lower() in ('1','true','t','y','yes')
                        return False
                    w_ud = _as_bool2(attrs_lc.get('word_ud', False))
                    w_du = _as_bool2(attrs_lc.get('word_du', False))
                if w_ud:
                    word_orients.append('ud')
                elif w_du:
                    word_orients.append('du')
                else:
                    word_orients.append(None)
            except Exception:
                word_orients.append(None)
        
        # 📦 LayoutDetection 기반 문장 병합
        orig_bboxes = list(bboxes)
        orig_words = list(words)
        orig_ids = list(word_ids)
        try:
            layout_boxes = run_layout_detection(layout_img_path)
            multi_samples = []
            if layout_boxes:
                # 병합된 전체 문장(원본 이미지 기준)은 더이상 LMDB에 저장하지 않음.
                # 대신 레이아웃별 인페인트 크롭을 개별 샘플로 저장.
                # 레이아웃-단어 매핑 준비
                img_cv_full = _decode_image_bytes(img_data)
                if img_cv_full is None:
                    layout_boxes = []
                else:
                    # img_cv_full은 이미 usd/ud/du 플래그 반영됨(위에서 회전 적용)
                    H, W = img_cv_full.shape[:2]
                    word_aabbs = [_aabb_from_flat8(b) for b in orig_bboxes]
                    layout_aabbs = []
                    for lb in layout_boxes:
                        x1, y1, x2, y2 = lb['coordinate']
                        layout_aabbs.append((max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))))
                    assigned = _assign_words_to_layout(word_aabbs, layout_aabbs, min_overlap_ratio=0.15)
                    # character 매핑
                    char_aabbs = []
                    if char_bboxes:
                        try:
                            char_aabbs = [_aabb_from_flat8(b) for b in char_bboxes]
                        except Exception:
                            char_aabbs = []
                    char_assigned = []
                    if char_aabbs:
                        char_assigned = _assign_words_to_layout(char_aabbs, layout_aabbs, min_overlap_ratio=0.3)
                    # 레이아웃별로 샘플 생성
                    for li, la in enumerate(layout_aabbs):
                        # 중심점 포함 기준으로 더 엄격하게 필터링(겹침 혼선 방지)
                        lx1, ly1, lx2, ly2 = la
                        idxs = []
                        for i, a in enumerate(assigned):
                            if a != li:
                                continue
                            wx1, wy1, wx2, wy2 = word_aabbs[i]
                            cx = (wx1 + wx2) * 0.5
                            cy = (wy1 + wy2) * 0.5
                            # 레이아웃 중심 포함 + 단어 박스의 대부분이 레이아웃에 포함되어야 함(>=90%)
                            if (lx1 <= cx <= lx2) and (ly1 <= cy <= ly2):
                                wa = (float(wx1), float(wy1), float(wx2), float(wy2))
                                la_aabb = (float(lx1), float(ly1), float(lx2), float(ly2))
                                w_area = _area(wa)
                                inter = _intersection_area(wa, la_aabb)
                                coverage = (inter / w_area) if w_area > 0 else 0.0
                                if coverage >= 0.6:
                                    idxs.append(i)
                        if not idxs:
                            continue
                        # ud/du 단어가 포함된 레이아웃이면: 개행 없이 방향성 반영 단일 라인 생성
                        has_ud = False
                        has_du = False
                        for i2 in idxs:
                            try:
                                if word_orients[i2] == 'ud': has_ud = True
                                elif word_orients[i2] == 'du': has_du = True
                            except Exception:
                                pass
                        if has_ud or has_du:
                            # 열(세로 컬럼) 클러스터링 → 각 컬럼 내부 y 정렬(ud: 아래→위, du: 위→아래) → 컬럼 간 ", "로 결합
                            # 1) x중심/폭 기반 컬럼 클러스터링 임계치 계산
                            widths = []
                            xcenters = {}
                            ycenters = {}
                            for i2 in idxs:
                                wx1, wy1, wx2, wy2 = word_aabbs[i2]
                                widths.append(max(1.0, abs(wx2 - wx1)))
                                xcenters[i2] = (wx1 + wx2) * 0.5
                                ycenters[i2] = (wy1 + wy2) * 0.5
                            median_w = float(np.median(widths)) if widths else 8.0
                            x_thresh = max(8.0, 0.8 * median_w)
                            # 2) x중심으로 정렬 후, 가까운 것끼리 컬럼으로 묶기
                            order_x = sorted(idxs, key=lambda i: xcenters[i])
                            columns = []
                            cur = []
                            basex = None
                            for i2 in order_x:
                                cx = xcenters[i2]
                                if basex is None:
                                    basex = cx; cur = [i2]
                                else:
                                    if abs(cx - basex) > x_thresh:
                                        if cur:
                                            columns.append(cur)
                                        basex = cx; cur = [i2]
                                    else:
                                        cur.append(i2)
                            if cur:
                                columns.append(cur)
                            # 3) 각 컬럼 내부 방향성 결정 및 y정렬
                            col_strings = []
                            for col in columns:
                                cnt_ud_c = sum(1 for i2 in col if (i2 < len(word_orients) and word_orients[i2] == 'ud'))
                                cnt_du_c = sum(1 for i2 in col if (i2 < len(word_orients) and word_orients[i2] == 'du'))
                                reverse_c = (cnt_ud_c > cnt_du_c)  # True면 아래→위
                                col_sorted = sorted(col, key=lambda i: ycenters[i], reverse=reverse_c)
                                col_str = " ".join(str(orig_words[j]) if orig_words[j] is not None else "" for j in col_sorted).strip()
                                if col_str:
                                    col_strings.append(col_str)
                            sentence = "\n".join(col_strings).strip()
                        else:
                            # 기존 로직: 문장 결합(개행 포함): y-클러스터 → 라인 내 x1 정렬
                            # 1) 단어 높이 기반 임계치
                            heights = []
                            for i2 in idxs:
                                y1i = word_aabbs[i2][1]; y2i = word_aabbs[i2][3]
                                heights.append(max(1.0, y2i - y1i))
                            median_h = float(np.median(heights)) if heights else 8.0
                            y_thresh = max(8.0, 0.6 * median_h)
                            # 2) y중심으로 정렬 후 라인 클러스터
                            order_y = sorted(idxs, key=lambda i: ((word_aabbs[i][1] + word_aabbs[i][3]) * 0.5))
                            lines_idx = []
                            cur = []
                            basey = None
                            for i2 in order_y:
                                cy = (word_aabbs[i2][1] + word_aabbs[i2][3]) * 0.5
                                if basey is None:
                                    basey = cy; cur = [i2]
                                else:
                                    if abs(cy - basey) > y_thresh:
                                        lines_idx.append(cur)
                                        basey = cy; cur = [i2]
                                    else:
                                        cur.append(i2)
                            if cur:
                                lines_idx.append(cur)
                            # 3) 각 라인 내 x1(좌측) 정렬, id 동점 보조
                            def _id_fallback(i):
                                try:
                                    return int(str(orig_ids[i]))
                                except Exception:
                                    return 10**9  # id가 없으면 뒤로
                            line_strings = []
                            for arr in lines_idx:
                                arr_sorted = sorted(
                                    arr,
                                    key=lambda i: (min(word_aabbs[i][0], word_aabbs[i][2]), _id_fallback(i))
                                )
                                line_str = " ".join(str(orig_words[j]) if orig_words[j] is not None else "" for j in arr_sorted).strip()
                                if line_str:
                                    line_strings.append(line_str)
                            sentence = "\n".join(line_strings)
                        if not sentence:
                            continue
                        # 크롭 영역: 원래 레이아웃 AABB 대신, 선택된 단어들의 합집합 AABB로 확장하여
                        # 단어가 부분 절단되지 않도록 보장
                        if not idxs:
                            continue
                        ux1 = min(word_aabbs[i][0] for i in idxs)
                        uy1 = min(word_aabbs[i][1] for i in idxs)
                        ux2 = max(word_aabbs[i][2] for i in idxs)
                        uy2 = max(word_aabbs[i][3] for i in idxs)
                        x1 = max(0, int(np.floor(ux1)))
                        y1 = max(0, int(np.floor(uy1)))
                        x2 = min(W, int(np.ceil(ux2)))
                        y2 = min(H, int(np.ceil(uy2)))
                        if x2<=x1 or y2<=y1:
                            continue
                        sentence_list = merge_words_by_layout(
                            [orig_bboxes[k] for k in idxs],
                            [orig_words[k] for k in idxs],
                            [{'label': 'layout', 'coordinate': [x1, y1, x2, y2], 'score': 1.0}],
                            word_ids=[orig_ids[k] for k in idxs],
                            prefer_id_order=True,
                            word_orients=[word_orients[k] for k in idxs] if word_orients else None
                        )[1]
                        sentence = sentence_list[0] if sentence_list else ""
                        # 이미지 단위 ud/du 라벨만 개행 제거, 단어 단위 방향이 있으면 그대로 유지
                        has_word_orient_local = any((word_orients[k] in ('ud','du')) for k in idxs) if word_orients else False
                        if (ud_flag or du_flag) and sentence and not has_word_orient_local:
                            try:
                                sentence = sentence.replace("\n", " ")
                            except Exception:
                                pass
                        if not sentence:
                            continue                        
                        crop = img_cv_full[y1:y2, x1:x2].copy()
                        # character 인페인트 + 문자 보존 합성
                        if char_aabbs and Inpaint is not None:
                            try:
                                # 레이아웃 할당 대신, 단어 합집합 AABB(ux1..uy2) 내부의 문자 선택
                                ch_idxs = []
                                for ci, b in enumerate(char_bboxes):
                                    try:
                                        cx = (b[0] + b[2] + b[4] + b[6]) * 0.25
                                        cy = (b[1] + b[3] + b[5] + b[7]) * 0.25
                                        if (x1 <= cx <= x2) and (y1 <= cy <= y2):
                                            ch_idxs.append(ci)
                                    except Exception:
                                        continue
                                if ch_idxs:
                                    _log_verbose(f"[text_in_wild] char-mask used: {len(ch_idxs)} chars for layout {li}")
                                    Hc, Wc = (y2-y1), (x2-x1)
                                    mask = np.ones((Hc, Wc), dtype=np.uint8) * 255
                                    preserve = np.zeros((Hc, Wc), dtype=np.uint8)
                                    polys=[]; char_heights=[]
                                    for ci in ch_idxs:
                                        b = char_bboxes[ci]
                                        poly = np.array([
                                            [max(0, min(Wc, int(round(b[0]-x1)))), max(0, min(Hc, int(round(b[1]-y1))))],
                                            [max(0, min(Wc, int(round(b[2]-x1)))), max(0, min(Hc, int(round(b[3]-y1))))],
                                            [max(0, min(Wc, int(round(b[4]-x1)))), max(0, min(Hc, int(round(b[5]-y1))))],
                                            [max(0, min(Wc, int(round(b[6]-x1)))), max(0, min(Hc, int(round(b[7]-y1))))],
                                        ], dtype=np.int32)
                                        polys.append(poly)
                                        by1 = max(0, min(poly[:,1])); by2 = min(Hc, max(poly[:,1]))
                                        char_heights.append(max(1, by2-by1))
                                    try:
                                        # 단어 ROI를 만들고 char 마스크를 항상 그 내부로 클리핑
                                        word_roi = np.zeros((Hc, Wc), dtype=np.uint8)
                                        for i2 in idxs:
                                            b = orig_bboxes[i2]
                                            wpoly = np.array([
                                                [max(0, min(Wc, int(round(b[0]-x1)))), max(0, min(Hc, int(round(b[1]-y1))))],
                                                [max(0, min(Wc, int(round(b[2]-x1)))), max(0, min(Hc, int(round(b[3]-y1))))],
                                                [max(0, min(Wc, int(round(b[4]-x1)))), max(0, min(Hc, int(round(b[5]-y1))))],
                                                [max(0, min(Wc, int(round(b[6]-x1)))), max(0, min(Hc, int(round(b[7]-y1))))],
                                            ], dtype=np.int32)
                                            cv2.fillPoly(word_roi, [wpoly], 1)
                                        cv2.fillPoly(mask, polys, 0)
                                        cv2.fillPoly(preserve, polys, 1)
                                        # preserve를 단어 ROI로 제한
                                        preserve = (preserve & word_roi).astype(np.uint8)
                                    except Exception:
                                        pass
                                    try:
                                        median_h2 = float(np.median(char_heights)) if char_heights else 8.0
                                        # 팽창 강도 추가 완화(아티팩트 감소)
                                        dilate_px = int(max(1, round(0.04*median_h2)))
                                        ksz = max(1, dilate_px*2+1)
                                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
                                        preserve_dil = cv2.dilate(preserve, kernel, iterations=1)
                                        # 가드 밴드(문자 가장자리 주변은 인페인트 금지) 추가
                                        guard_px = int(max(1, round(0.02*median_h2)))
                                        gsz = max(1, guard_px*2+1)
                                        gkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gsz, gsz))
                                        guard = cv2.dilate(preserve, gkernel, iterations=1)
                                        mask = np.where(preserve_dil>0, 0, 255).astype(np.uint8)
                                        mask[guard>0] = 0
                                    except Exception:
                                        pass
                                    crop_orig = crop.copy()
                                    bg_restored = None
                                    try:
                                        inp = Inpaint(crop, mask)
                                        out1 = inp()
                                        if out1 is not None and out1.shape == crop.shape:
                                            bg_restored = out1
                                    except Exception:
                                        bg_restored = None
                                    if bg_restored is None:
                                        try:
                                            mask_cv = (mask>0).astype(np.uint8)*255
                                            # Navier-Stokes로 변경하여 경계 색 번짐 완화
                                            bg_restored = cv2.inpaint(crop, mask_cv, 3, cv2.INPAINT_NS)
                                        except Exception:
                                            bg_restored = None
                                    if bg_restored is not None:
                                        m = (mask==0).astype(np.float32)
                                        try:
                                            feather = max(1, int(round(0.04*(median_h2 if 'median_h2' in locals() else 8.0))))
                                            kf = max(1, feather*2+1)
                                            m = cv2.GaussianBlur(m, (kf, kf), 0)
                                        except Exception:
                                            pass
                                        if len(crop.shape)==3:
                                            m3 = np.repeat(m[:, :, None], 3, axis=2)
                                        else:
                                            m3 = m
                                        crop = (crop_orig.astype(np.float32)*m3 + bg_restored.astype(np.float32)*(1.0-m3)).clip(0,255).astype(crop_orig.dtype)
                            except Exception:
                                pass
                        # 인페인트/일반 크롭을 JPEG로 인코딩 (회전 보정 후)
                        crop = _apply_rotation_if_needed(crop, img_path)
                        ok, buf = fast_encode_jpg(crop)
                        if not ok:
                            continue
                        img_bytes_li = bytes(buf)
                        # GT: 크롭 전체 박스 1개 + 문장 1개 (회전 후 실제 크기 기준)
                        h, w = crop.shape[:2]
                        flat8 = [0.0, 0.0, float(w), 0.0, float(w), float(h), 0.0, float(h)]
                        gt_li = {
                            'bboxes': [flat8],
                            'words': [sentence],
                            'filename': f"{img_info['file_name']}_layout_{li:02d}.jpg"
                        }
                        multi_samples.append((f"{img_id}_{li}", img_bytes_li, gt_li))
            # multi_samples가 있으면 반환
            if layout_boxes and multi_samples:
                return multi_samples
            # 레이아웃이 없거나 실패 시, 기존 단어 단위 반환
            if layout_boxes:
                merged_bboxes, merged_words = merge_words_by_layout(bboxes, words, layout_boxes, word_ids=word_ids, prefer_id_order=True, word_orients=word_orients)
                if merged_bboxes and merged_words:
                    # 이미지 단위 ud/du 라벨만 개행 제거 (단어 단위 방향이 있으면 유지)
                    if (ud_flag or du_flag):
                        try:
                            merged_words = [w.replace("\n", " ") if isinstance(w, str) else w for w in merged_words]
                        except Exception:
                            pass
                    bboxes, words = merged_bboxes, merged_words
        except Exception as e:
            _log_verbose(f"[text_in_wild] exception before detection: {img_file_name} err={type(e).__name__}")
            pass

        gt_info = {
            'bboxes': bboxes,
            'words': words,
            'filename': img_info['file_name']
        }
        # 레이아웃 단위 디버그 저장 로직 제거됨
        return None
        
    except Exception as e:
        return None
    finally:
        try:
            if 'tmp_rot_path' in locals() and tmp_rot_path and os.path.exists(tmp_rot_path):
                os.remove(tmp_rot_path)
        except Exception:
            pass

def process_single_public_admin_image(args):
    """공공행정문서 단일 이미지 처리 함수 (병렬 처리용)"""
    img_info, annotations, base_path, lookup_dict, dataset_lookup_name, image_path_cache = args
    
    try:
        word_ids = []
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
        word_ids = []
        word_ids = []
        
        # 어노테이션 로딩 (메모리 절약: SQLite에서 지연 로드 가능)
        if isinstance(annotations, dict) and annotations.get('sqlite') and annotations.get('image_id') is not None:
            sqlite_path = annotations['sqlite']
            target_img_id = annotations['image_id']
            anns_loaded = []
            try:
                conn = _get_public_admin_sqlite_conn(sqlite_path)
                if conn is not None:
                    cur = conn.cursor()
                    for row in cur.execute("SELECT ann FROM a WHERE image_id=?", (int(target_img_id),)):
                        try:
                            anns_loaded.append(orjson.loads(row[0]))
                        except Exception:
                            continue
                    cur.close()
            except Exception:
                anns_loaded = []
        else:
            anns_loaded = annotations or []

        for ann in anns_loaded:
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
            ann_id = ann.get('id') or ann.get('annotation.id')
            if ann_id is None:
                ann_id = len(word_ids)
            word_ids.append(ann_id)
    
        # 📦 LayoutDetection 기반 문장 병합
        orig_bboxes = list(bboxes)
        orig_words = list(words)
        try:
            layout_boxes = run_layout_detection(img_path)
            # 레이아웃 텍스트가 없으면 테이블 셀 탐지 시도
            if not layout_boxes:
                tables = run_layout_tables(img_path)
                if tables:
                    # 테이블 셀 탐지
                    table_model = get_table_model()
                    if table_model is None:
                        return None
                    img_cv_full = _decode_image_bytes(img_data)
                    if img_cv_full is None:
                        return None
                    H, W = img_cv_full.shape[:2]
                    word_aabbs = [_aabb_from_flat8(b) for b in orig_bboxes]
                    results = []
                    # 캐시된 테이블 셀 우선 사용
                    cached_cells = None
                    try:
                        with PRED_CACHE_LOCK:
                            cached = PREDICTION_CACHE.get(img_path) or {}
                            cached_cells = cached.get('table_cells')
                    except Exception:
                        cached_cells = None
                    if cached_cells:
                        # cached_cells: [(tx1,ty1,tx2,ty2,[(cx1,cy1,cx2,cy2), ...]), ...]
                        for (tx1, ty1, tx2, ty2, cells) in cached_cells or []:
                            for (cx1, cy1, cx2, cy2) in cells or []:
                                cell_aabb = (float(cx1), float(cy1), float(cx2), float(cy2))
                                idxs = []
                                for wi, wa in enumerate(word_aabbs):
                                    inter = _intersection_area(wa, cell_aabb)
                                    wa_area = _area(wa)
                                    if wa_area > 0 and (inter / wa_area) >= 0.2:
                                        idxs.append(wi)
                                if not idxs:
                                    continue
                                def _id_key(i):
                                    try:
                                        return int(str(word_ids[i]))
                                    except Exception:
                                        return str(word_ids[i])
                                idxs.sort(key=lambda j: ((word_aabbs[j][0] + word_aabbs[j][2]) / 2.0, _id_key(j)))
                                sentence = merge_words_by_layout(
                                    [orig_bboxes[k] for k in idxs],
                                    [orig_words[k] for k in idxs],
                                    [{'label': 'cell', 'coordinate': [cx1, cy1, cx2, cy2], 'score': 1.0}],
                                    word_ids=[word_ids[k] for k in idxs],
                                    prefer_id_order=True
                                )[1][0] if idxs else ""
                                if not sentence:
                                    continue
                                crop = img_cv_full[cy1:cy2, cx1:cx2]
                                try:
                                    preserve_polys = []
                                    for k in idxs:
                                        poly = _flat8_to_crop_poly(orig_bboxes[k], cx1, cy1, cx2, cy2)
                                        if poly is not None and poly.size > 0:
                                            preserve_polys.append(poly)
                                    if preserve_polys:
                                        crop = _inpaint_preserve_regions(crop, preserve_polys)
                                except Exception:
                                    pass
                                crop = _apply_rotation_if_needed(crop, img_path)
                                ok, buf = fast_encode_jpg(crop)
                                if not ok:
                                    continue
                                img_bytes_li = bytes(buf)
                                h, w = crop.shape[:2]
                                flat8 = [0.0, 0.0, float(w), 0.0, float(w), float(h), 0.0, float(h)]
                                gt_li = {'bboxes': [flat8], 'words': [sentence], 'filename': f"{img_file_name}_cell_{cx1}_{cy1}.jpg"}
                                results.append((f"{img_id}_cell_{cx1}_{cy1}", img_bytes_li, gt_li))
                    else:
                        # 배치로 표 내부 셀 박스 예측
                        batch_cells = _predict_table_cells_batch(img_cv_full, tables, img_id)
                        for tx1, ty1, tx2, ty2, cells in (batch_cells or []):
                            for (cx1, cy1, cx2, cy2) in cells:
                                cell_aabb = (float(cx1), float(cy1), float(cx2), float(cy2))
                                idxs = []
                                for wi, wa in enumerate(word_aabbs):
                                    inter = _intersection_area(wa, cell_aabb)
                                    wa_area = _area(wa)
                                    if wa_area > 0 and (inter / wa_area) >= 0.2:
                                        idxs.append(wi)
                                if not idxs:
                                    continue
                                def _id_key(i):
                                    try:
                                        return int(str(word_ids[i]))
                                    except Exception:
                                        return str(word_ids[i])
                                idxs.sort(key=lambda j: ((word_aabbs[j][0] + word_aabbs[j][2]) / 2.0, _id_key(j)))
                                sentence = merge_words_by_layout(
                                    [orig_bboxes[k] for k in idxs],
                                    [orig_words[k] for k in idxs],
                                    [{'label': 'cell', 'coordinate': [cx1, cy1, cx2, cy2], 'score': 1.0}],
                                    word_ids=[word_ids[k] for k in idxs],
                                    prefer_id_order=True
                                )[1][0] if idxs else ""
                                if not sentence:
                                    continue
                                crop = img_cv_full[cy1:cy2, cx1:cx2]
                                try:
                                    preserve_polys = []
                                    for k in idxs:
                                        poly = _flat8_to_crop_poly(orig_bboxes[k], cx1, cy1, cx2, cy2)
                                        if poly is not None and poly.size > 0:
                                            preserve_polys.append(poly)
                                    if preserve_polys:
                                        crop = _inpaint_preserve_regions(crop, preserve_polys)
                                except Exception:
                                    pass
                                crop = _apply_rotation_if_needed(crop, img_path)
                                ok, buf = fast_encode_jpg(crop)
                                if not ok:
                                    continue
                                img_bytes_li = bytes(buf)
                                h, w = crop.shape[:2]
                                flat8 = [0.0, 0.0, float(w), 0.0, float(w), float(h), 0.0, float(h)]
                                gt_li = {'bboxes': [flat8], 'words': [sentence], 'filename': f"{img_file_name}_cell_{cx1}_{cy1}.jpg"}
                                results.append((f"{img_id}_cell_{cx1}_{cy1}", img_bytes_li, gt_li))
                            if not sentence:
                                continue
                            # 셀 크롭 저장
                            crop = img_cv_full[cy1:cy2, cx1:cx2]
                            # inpaint: 라벨 단어만 보존, 나머지 배경 복원
                            try:
                                preserve_polys = []
                                for k in idxs:
                                    poly = _flat8_to_crop_poly(orig_bboxes[k], cx1, cy1, cx2, cy2)
                                    if poly is not None and poly.size > 0:
                                        preserve_polys.append(poly)
                                if preserve_polys:
                                    crop = _inpaint_preserve_regions(crop, preserve_polys)
                            except Exception:
                                pass
                            # 인페인트/일반 크롭을 JPEG로 인코딩 (회전 보정 후)
                            crop = _apply_rotation_if_needed(crop, img_path)
                            ok, buf = fast_encode_jpg(crop)
                            if not ok:
                                continue
                            img_bytes_li = bytes(buf)
                            h, w = crop.shape[:2]
                            flat8 = [0.0, 0.0, float(w), 0.0, float(w), float(h), 0.0, float(h)]
                            gt_li = {'bboxes': [flat8], 'words': [sentence], 'filename': f"{img_file_name}_cell_{cx1}_{cy1}.jpg"}
                            results.append((f"{img_id}_cell_{cx1}_{cy1}", img_bytes_li, gt_li))
                    if results:
                        return results
                # table도 없으면 제외
                return None
            multi_samples = []
            if layout_boxes:
                img_cv_full = _decode_image_bytes(img_data)
                if img_cv_full is not None:
                    H, W = img_cv_full.shape[:2]
                    word_aabbs = [_aabb_from_flat8(b) for b in orig_bboxes]
                    layout_aabbs = []
                    for lb in layout_boxes:
                        x1, y1, x2, y2 = lb['coordinate']
                        layout_aabbs.append((max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))))
                    assigned = _assign_words_to_layout(word_aabbs, layout_aabbs, min_overlap_ratio=0.3)
                    for li, la in enumerate(layout_aabbs):
                        idxs = [i for i, a in enumerate(assigned) if a == li]
                        if not idxs:
                            continue
                        x1, y1, x2, y2 = la
                        # ID 우선(STRICT_ID_ORDER) 병합기로 문장 구성
                        sentence_list = merge_words_by_layout(
                            [orig_bboxes[k] for k in idxs],
                            [orig_words[k] for k in idxs],
                            [{'label': 'layout', 'coordinate': [x1, y1, x2, y2], 'score': 1.0}],
                            word_ids=[word_ids[k] for k in idxs],
                            prefer_id_order=True
                        )[1]
                        sentence = sentence_list[0] if sentence_list else ""
                        if not sentence:
                            continue
                        x1, y1, x2, y2 = la
                        if x2<=x1 or y2<=y1:
                            continue
                        crop = img_cv_full[y1:y2, x1:x2].copy()
                        # inpaint: 라벨 단어만 보존
                        try:
                            preserve_polys = []
                            for k in idxs:
                                poly = _flat8_to_crop_poly(orig_bboxes[k], x1, y1, x2, y2)
                                if poly is not None and poly.size > 0:
                                    preserve_polys.append(poly)
                            if preserve_polys:
                                crop = _inpaint_preserve_regions(crop, preserve_polys)
                        except Exception:
                            pass
                        # 인페인트/일반 크롭을 JPEG로 인코딩 (회전 보정 후)
                        crop = _apply_rotation_if_needed(crop, img_path)
                        ok, buf = fast_encode_jpg(crop)
                        if not ok:
                            continue
                        img_bytes_li = bytes(buf)
                        h, w = crop.shape[:2]
                        flat8 = [0.0,0.0,float(w),0.0,float(w),float(h),0.0,float(h)]
                        gt_li = {'bboxes':[flat8], 'words':[sentence], 'filename': f"{img_file_name}_layout_{li:02d}.jpg"}
                        multi_samples.append((f"{img_id}_{li}", img_bytes_li, gt_li))
            # 테이블 셀 샘플도 항상 추가 시도 (public_admin에서도 레이아웃 유무와 무관)
            try:
                tables = run_layout_tables(img_path)
                print(f"[debug] public_admin(always): tables_found={len(tables) if tables else 0}")
                cell_samples = []
                if tables:
                    table_model = get_table_model()
                    img_cv_full = _decode_image_bytes(img_data)
                    if table_model is not None and img_cv_full is not None:
                        H, W = img_cv_full.shape[:2]
                        word_aabbs = [_aabb_from_flat8(b) for b in orig_bboxes]
                        print(f"[debug] public_admin(always): call batch tables={len(tables)}")
                        batch_cells = _predict_table_cells_batch(img_cv_full, tables, img_id)
                        print(f"[debug] public_admin(always): batch_cells groups={len(batch_cells)}")
                        for (tx1, ty1, tx2, ty2, cells) in (batch_cells or []):
                            for (cx1, cy1, cx2, cy2) in (cells or []):
                                cell_aabb = (float(cx1), float(cy1), float(cx2), float(cy2))
                                idxs = []
                                for wi, wa in enumerate(word_aabbs):
                                    inter = _intersection_area(wa, cell_aabb)
                                    wa_area = _area(wa)
                                    if wa_area > 0 and (inter / wa_area) >= 0.5:
                                        idxs.append(wi)
                                if not idxs:
                                    continue
                                def _id_key(i):
                                    try:
                                        return int(str(word_ids[i]))
                                    except Exception:
                                        return str(word_ids[i])
                                idxs.sort(key=lambda j: ((word_aabbs[j][0] + word_aabbs[j][2]) / 2.0, _id_key(j)))
                                sentence = merge_words_by_layout(
                                    [orig_bboxes[k] for k in idxs],
                                    [orig_words[k] for k in idxs],
                                    [{'label': 'cell', 'coordinate': [cx1, cy1, cx2, cy2], 'score': 1.0}],
                                    word_ids=[word_ids[k] for k in idxs],
                                    prefer_id_order=True
                                )[1][0] if idxs else ""
                                if not sentence:
                                    continue
                                crop = img_cv_full[cy1:cy2, cx1:cx2]
                                # 인페인트/일반 크롭을 JPEG로 인코딩 (회전 보정 후)
                                crop = _apply_rotation_if_needed(crop, img_path)
                                ok, buf = fast_encode_jpg(crop)
                                if not ok:
                                    continue
                                img_bytes_li = bytes(buf)
                                h, w = crop.shape[:2]
                                flat8 = [0.0, 0.0, float(w), 0.0, float(w), float(h), 0.0, float(h)]
                                gt_li = {'bboxes': [flat8], 'words': [sentence], 'filename': f"{img_file_name}_cell_{cx1}_{cy1}.jpg"}
                                cell_samples.append((f"{img_id}_cell_{cx1}_{cy1}", img_bytes_li, gt_li))
                # 배치로 표 내부 셀 박스 예측
                batch_cells = _predict_table_cells_batch(img_cv_full, tables, img_id)
                for tx1, ty1, tx2, ty2, cells in (batch_cells or []):
                    for (cx1, cy1, cx2, cy2) in cells:
                        cell_aabb = (float(cx1), float(cy1), float(cx2), float(cy2))
                        idxs = []
                        for wi, wa in enumerate(word_aabbs):
                            inter = _intersection_area(wa, cell_aabb)
                            wa_area = _area(wa)
                            if wa_area > 0 and (inter / wa_area) >= 0.2:
                                idxs.append(wi)
                        if not idxs:
                            continue
                        def _id_key(i):
                            try:
                                return int(str(word_ids[i]))
                            except Exception:
                                return str(word_ids[i])
                        idxs.sort(key=lambda j: ((word_aabbs[j][0] + word_aabbs[j][2]) / 2.0, _id_key(j)))
                        sentence = merge_words_by_layout(
                            [orig_bboxes[k] for k in idxs],
                            [orig_words[k] for k in idxs],
                            [{'label': 'cell', 'coordinate': [cx1, cy1, cx2, cy2], 'score': 1.0}],
                            word_ids=[word_ids[k] for k in idxs],
                            prefer_id_order=True
                        )[1][0] if idxs else ""
                        if not sentence:
                            continue
                        crop = img_cv_full[cy1:cy2, cx1:cx2]
                        try:
                            preserve_polys = []
                            for k in idxs:
                                poly = _flat8_to_crop_poly(orig_bboxes[k], cx1, cy1, cx2, cy2)
                                if poly is not None and poly.size > 0:
                                    preserve_polys.append(poly)
                            if preserve_polys:
                                crop = _inpaint_preserve_regions(crop, preserve_polys)
                        except Exception:
                            pass
                        crop = _apply_rotation_if_needed(crop, img_path)
                        ok, buf = fast_encode_jpg(crop)
                        if not ok:
                            continue
                        img_bytes_li = bytes(buf)
                        h, w = crop.shape[:2]
                        flat8 = [0.0, 0.0, float(w), 0.0, float(w), float(h), 0.0, float(h)]
                        gt_li = {'bboxes': [flat8], 'words': [sentence], 'filename': f"{img_file_name}_cell_{cx1}_{cy1}.jpg"}
                        cell_samples.append((f"{img_id}_cell_{cx1}_{cy1}", img_bytes_li, gt_li))
            except Exception:
                pass
            # 테이블 셀 결과가 있으면 우선 반환 (레이아웃 결과는 테이블과 중복될 수 있어 제외)
            if 'cell_samples' in locals() and cell_samples:
                return cell_samples
            if multi_samples:
                return multi_samples
            if layout_boxes:
                merged_bboxes, merged_words = merge_words_by_layout(bboxes, words, layout_boxes, word_ids=word_ids, prefer_id_order=True)
                if merged_bboxes and merged_words:
                    bboxes, words = merged_bboxes, merged_words
        except Exception:
            pass
        
        gt_info = {
            'bboxes': bboxes,
            'words': words,
            'filename': img_file_name
        }
        # 레이아웃 단위 디버그 저장 로직 제거됨
        
        return None
        
    except Exception as e:
        return None

def process_single_ocr_public_image(args):
    """OCR 공공 단일 이미지 처리 함수 (병렬 처리용)"""
    img_info, annotations, base_path, dataset_lookup_name, image_path_cache = args
    
    try:
        # 필수 상태 초기화
        word_ids = []
        layout_boxes = None
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
                    ann_id = ann.get('id')
                    if ann_id is None:
                        ann_id = len(word_ids)
                    word_ids.append(ann_id)
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
                    ann_id = ann.get('id')
                    if ann_id is None:
                        ann_id = len(word_ids)
                    word_ids.append(ann_id)
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
                    ann_id = ann.get('id')
                    if ann_id is None:
                        ann_id = len(word_ids)
                    word_ids.append(ann_id)
                except (IndexError, TypeError):
                    continue
        
        # 📦 LayoutDetection 기반 문장 병합
        orig_bboxes = list(bboxes)
        orig_words = list(words)
        try:
            layout_boxes = run_layout_detection(img_path)
            layout_cnt = len(layout_boxes) if layout_boxes else 0
            results = []
            # 1) 레이아웃 텍스트 샘플 생성
            if layout_boxes:
                img_cv_full = _decode_image_bytes(img_data)
                if img_cv_full is not None:
                    H, W = img_cv_full.shape[:2]
                    word_aabbs = [_aabb_from_flat8(b) for b in orig_bboxes]
                    layout_aabbs = []
                    for lb in layout_boxes:
                        x1, y1, x2, y2 = lb['coordinate']
                        layout_aabbs.append((max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))))
                    assigned = _assign_words_to_layout(word_aabbs, layout_aabbs, min_overlap_ratio=0.3)
                    for li, la in enumerate(layout_aabbs):
                        idxs = [i for i, a in enumerate(assigned) if a == li]
                        if not idxs:
                            continue
                        x1, y1, x2, y2 = la
                        sentence_list = merge_words_by_layout(
                            [orig_bboxes[k] for k in idxs],
                            [orig_words[k] for k in idxs],
                            [{'label': 'layout', 'coordinate': [x1, y1, x2, y2], 'score': 1.0}],
                            word_ids=[word_ids[k] for k in idxs],
                            prefer_id_order=True
                        )[1]
                        sentence = sentence_list[0] if sentence_list else ""
                        if not sentence:
                            continue
                        x1, y1, x2, y2 = la
                        if x2<=x1 or y2<=y1:
                            continue
                        crop = img_cv_full[y1:y2, x1:x2].copy()
                        # inpaint: 라벨 단어만 보존
                        try:
                            preserve_polys = []
                            for k in idxs:
                                poly = _flat8_to_crop_poly(orig_bboxes[k], x1, y1, x2, y2)
                                if poly is not None and poly.size > 0:
                                    preserve_polys.append(poly)
                            if preserve_polys:
                                crop = _inpaint_preserve_regions(crop, preserve_polys)
                        except Exception:
                            pass
                        # 인페인트/일반 크롭을 JPEG로 인코딩 (회전 보정 후)
                        crop = _apply_rotation_if_needed(crop, img_path)
                        ok, buf = fast_encode_jpg(crop)
                        if not ok:
                            continue
                        img_bytes_li = bytes(buf)
                        h, w = crop.shape[:2]
                        flat8 = [0.0,0.0,float(w),0.0,float(w),float(h),0.0,float(h)]
                        gt_li = {'bboxes':[flat8], 'words':[sentence], 'filename': f"{img_file_name}_layout_{li:02d}.jpg"}
                        results.append((f"{img_id}_{li}", img_bytes_li, gt_li))
            # 2) 테이블 셀 샘플 추가
            tables = run_layout_tables(img_path)
            table_cnt = len(tables) if tables else 0
            cell_results = []
            if tables:
                table_model = get_table_model()
                img_cv_full = _decode_image_bytes(img_data)
                if table_model is not None and img_cv_full is not None:
                    H, W = img_cv_full.shape[:2]
                    word_aabbs = [_aabb_from_flat8(b) for b in orig_bboxes]
                    # 캐시된 table_cells 우선 사용
                    cached_cells = None
                    try:
                        with PRED_CACHE_LOCK:
                            cached = PREDICTION_CACHE.get(img_path) or {}
                            cached_cells = cached.get('table_cells')
                    except Exception:
                        cached_cells = None
                    if cached_cells:
                        for (tx1, ty1, tx2, ty2, cells) in cached_cells or []:
                            for (cx1, cy1, cx2, cy2) in cells or []:
                                cell_aabb = (float(cx1), float(cy1), float(cx2), float(cy2))
                                idxs = []
                                for wi, wa in enumerate(word_aabbs):
                                    inter = _intersection_area(wa, cell_aabb)
                                    wa_area = _area(wa)
                                    # 셀 내부 포함 비율 완화: 50% -> 20%
                                    if wa_area > 0 and (inter / wa_area) >= 0.2:
                                        idxs.append(wi)
                                if not idxs:
                                    continue
                                def _id_key(i):
                                    try:
                                        return int(str(word_ids[i]))
                                    except Exception:
                                        return str(word_ids[i])
                                idxs.sort(key=lambda j: ((word_aabbs[j][0] + word_aabbs[j][2]) / 2.0, _id_key(j)))
                                sentence = merge_words_by_layout(
                                    [orig_bboxes[k] for k in idxs],
                                    [orig_words[k] for k in idxs],
                                    [{'label': 'cell', 'coordinate': [cx1, cy1, cx2, cy2], 'score': 1.0}],
                                    word_ids=[word_ids[k] for k in idxs],
                                    prefer_id_order=True
                                )[1][0] if idxs else ""
                                if not sentence:
                                    continue
                                crop = img_cv_full[cy1:cy2, cx1:cx2]
                                try:
                                    preserve_polys = []
                                    for k in idxs:
                                        poly = _flat8_to_crop_poly(orig_bboxes[k], cx1, cy1, cx2, cy2)
                                        if poly is not None and poly.size > 0:
                                            preserve_polys.append(poly)
                                    if preserve_polys:
                                        crop = _inpaint_preserve_regions(crop, preserve_polys)
                                except Exception:
                                    pass
                                crop = _apply_rotation_if_needed(crop, img_path)
                                ok, buf = fast_encode_jpg(crop)
                                if not ok:
                                    continue
                                img_bytes_li = bytes(buf)
                                h, w = crop.shape[:2]
                                flat8 = [0.0, 0.0, float(w), 0.0, float(w), float(h), 0.0, float(h)]
                                gt_li = {'bboxes': [flat8], 'words': [sentence], 'filename': f"{img_file_name}_cell_{cx1}_{cy1}.jpg"}
                                cell_results.append((f"{img_id}_cell_{cx1}_{cy1}", img_bytes_li, gt_li))
                    else:
                        print(f"[cells] use BATCH tables={len(tables)} id={img_id}")
                        batch_cells = _predict_table_cells_batch(img_cv_full, tables, img_id)
                        for tx1, ty1, tx2, ty2, cells in (batch_cells or []):
                            for (cx1, cy1, cx2, cy2) in cells:
                                cell_aabb = (float(cx1), float(cy1), float(cx2), float(cy2))
                                idxs = []
                                for wi, wa in enumerate(word_aabbs):
                                    inter = _intersection_area(wa, cell_aabb)
                                    wa_area = _area(wa)
                                    # 셀 내부 포함 비율 완화: 50% -> 20%
                                    if wa_area > 0 and (inter / wa_area) >= 0.2:
                                        idxs.append(wi)
                                if not idxs:
                                    continue
                                def _id_key(i):
                                    try:
                                        return int(str(word_ids[i]))
                                    except Exception:
                                        return str(word_ids[i])
                                idxs.sort(key=lambda j: ((word_aabbs[j][0] + word_aabbs[j][2]) / 2.0, _id_key(j)))
                                sentence = merge_words_by_layout(
                                    [orig_bboxes[k] for k in idxs],
                                    [orig_words[k] for k in idxs],
                                    [{'label': 'cell', 'coordinate': [cx1, cy1, cx2, cy2], 'score': 1.0}],
                                    word_ids=[word_ids[k] for k in idxs],
                                    prefer_id_order=True
                                )[1][0] if idxs else ""
                                if not sentence:
                                    continue
                                crop = img_cv_full[cy1:cy2, cx1:cx2]
                                # inpaint: 라벨 단어만 보존
                                try:
                                    preserve_polys = []
                                    for k in idxs:
                                        poly = _flat8_to_crop_poly(orig_bboxes[k], cx1, cy1, cx2, cy2)
                                        if poly is not None and poly.size > 0:
                                            preserve_polys.append(poly)
                                    if preserve_polys:
                                        crop = _inpaint_preserve_regions(crop, preserve_polys)
                                except Exception:
                                    pass
                                # 인페인트/일반 크롭을 JPEG로 인코딩 (회전 보정 후)
                                crop = _apply_rotation_if_needed(crop, img_path)
                                ok, buf = fast_encode_jpg(crop)
                                if not ok:
                                    continue
                                img_bytes_li = bytes(buf)
                                h, w = crop.shape[:2]
                                flat8 = [0.0, 0.0, float(w), 0.0, float(w), float(h), 0.0, float(h)]
                                gt_li = {'bboxes': [flat8], 'words': [sentence], 'filename': f"{img_file_name}_cell_{cx1}_{cy1}.jpg"}
                                cell_results.append((f"{img_id}_cell_{cx1}_{cy1}", img_bytes_li, gt_li))
            # 셀이 있으면 셀만 반환 (레이아웃 중복 제거)
            if cell_results:
                _log_verbose(f"[ocr_public] keep cells: {img_file_name} layouts={layout_cnt} tables={table_cnt} cells={len(cell_results)}")
                return cell_results
            if results:
                _log_verbose(f"[ocr_public] keep layouts: {img_file_name} layouts={layout_cnt} kept={len(results)}")
                return results
            if layout_boxes:
                merged_bboxes, merged_words = merge_words_by_layout(bboxes, words, layout_boxes, word_ids=word_ids, prefer_id_order=True)
                if merged_bboxes and merged_words:
                    bboxes, words = merged_bboxes, merged_words
            _log_verbose(f"[ocr_public] exclude: {img_file_name} layouts={layout_cnt} tables={table_cnt} cells=0 (no detections)")
            return None
        except Exception:
            return None
        
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
        # 안전한 이미지 ID (sub_dataset 또는 파일명 기반)
        try:
            safe_img_id = str(img_info_data.get('id') or sub_dataset or os.path.splitext(os.path.basename(img_info_data['file_path']))[0])
        except Exception:
            safe_img_id = os.path.splitext(os.path.basename(img_info_data['file_path']))[0]
        
        # 어노테이션 처리 (기존 로직 그대로)
        bboxes = []
        words = []
        img_w = img_info_data['width']
        img_h = img_info_data['height']
        word_ids = []
        
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
                    ann_id = ann.get('id')
                    if ann_id is None:
                        ann_id = len(word_ids)
                    word_ids.append(ann_id)
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
                    ann_id = ann.get('id')
                    if ann_id is None:
                        ann_id = len(word_ids)
                    word_ids.append(ann_id)
            except (IndexError, TypeError, ValueError):
                # bbox가 잘못된 형식이면 건너뛰기
                continue
        
        # 📦 LayoutDetection 기반 문장 병합
        orig_bboxes = list(bboxes)
        orig_words = list(words)
        try:
            layout_boxes = run_layout_detection(img_info_data['file_path'])
            layout_cnt = len(layout_boxes) if layout_boxes else 0
            results = []
            # 1) 테이블 셀 샘플
            tables = run_layout_tables(img_info_data['file_path'])
            table_cnt = len(tables) if tables else 0
            cell_results = []
            table_model = get_table_model()
            img_cv_full = _decode_image_bytes(img_data)
            if table_model is not None and img_cv_full is not None and tables:
                    # 배치 예측 경로 (테이블 크롭들을 한 번에 예측) - 캐시 우선
                    word_aabbs = [_aabb_from_flat8(b) for b in orig_bboxes]
                    # 캐시된 table_cells 먼저 확인
                    batch_cells = None
                    try:
                        with PRED_CACHE_LOCK:
                            cached = PREDICTION_CACHE.get(img_info_data['file_path']) or {}
                            batch_cells = cached.get('table_cells')
                    except Exception:
                        batch_cells = None
                    if not batch_cells:
                        batch_cells = _predict_table_cells_batch(img_cv_full, tables, safe_img_id)
                    for tx1, ty1, tx2, ty2, cells in (batch_cells or []):
                        for (cx1, cy1, cx2, cy2) in cells:
                            cell_aabb = (float(cx1), float(cy1), float(cx2), float(cy2))
                            idxs = []
                            for wi, wa in enumerate(word_aabbs):
                                inter = _intersection_area(wa, cell_aabb)
                                wa_area = _area(wa)
                                # 셀 내부 포함 비율 완화: 50% -> 20%
                                if wa_area > 0 and (inter / wa_area) >= 0.2:
                                    idxs.append(wi)
                            if not idxs:
                                continue
                            def _id_key(i):
                                try:
                                    return int(str(word_ids[i]))
                                except Exception:
                                    return str(word_ids[i])
                            idxs.sort(key=lambda j: ((word_aabbs[j][0] + word_aabbs[j][2]) / 2.0, _id_key(j)))
                            sentence = merge_words_by_layout(
                                [orig_bboxes[k] for k in idxs],
                                [orig_words[k] for k in idxs],
                                [{'label': 'cell', 'coordinate': [cx1, cy1, cx2, cy2], 'score': 1.0}],
                                word_ids=[word_ids[k] for k in idxs],
                                prefer_id_order=True
                            )[1][0] if idxs else ""
                            if not sentence:
                                continue
                            crop = img_cv_full[cy1:cy2, cx1:cx2]
                            # inpaint: 라벨 단어만 보존
                            try:
                                preserve_polys = []
                                for k in idxs:
                                    poly = _flat8_to_crop_poly(orig_bboxes[k], cx1, cy1, cx2, cy2)
                                    if poly is not None and poly.size > 0:
                                        preserve_polys.append(poly)
                                if preserve_polys:
                                    crop = _inpaint_preserve_regions(crop, preserve_polys)
                            except Exception:
                                pass
                            # 인페인트/일반 크롭을 JPEG로 인코딩 (회전 보정 후)
                            crop = _apply_rotation_if_needed(crop, img_info_data['file_path'])
                            ok, buf = fast_encode_jpg(crop)
                            if not ok:
                                continue
                            img_bytes_li = bytes(buf)
                            h, w = crop.shape[:2]
                            flat8 = [0.0, 0.0, float(w), 0.0, float(w), float(h), 0.0, float(h)]
                            gt_li = {'bboxes': [flat8], 'words': [sentence], 'filename': f"{os.path.basename(img_info_data['file_path'])}_cell_{cx1}_{cy1}.jpg"}
                            cell_results.append((f"{safe_img_id}_cell_{cx1}_{cy1}", img_bytes_li, gt_li))
            elif table_model is not None and img_cv_full is not None and (not tables) and int(os.environ.get("FAST_PAGE_LEVEL_CELL","0")) == 1:
                # 테이블이 없으면 페이지 전체에서 셀 탐지
                H, W = img_cv_full.shape[:2]
                cells = []
                try:
                    with TABLE_MODEL_LOCK:
                        _log_verbose(f"[cells] page-level predict: {os.path.basename(img_info_data['file_path'])} thr={TABLE_THRESHOLD}")
                        cell_out = table_model.predict(img_info_data['file_path'], threshold=TABLE_THRESHOLD, batch_size=1)
                    first = cell_out[0] if isinstance(cell_out, (list, tuple)) and cell_out else cell_out
                    boxes_list = []
                    if isinstance(first, dict):
                        for k in ('boxes', 'result', 'preds', 'predictions'):
                            if k in first and isinstance(first[k], (list, tuple)):
                                boxes_list = first[k]
                                break
                    else:
                        for k in ('boxes', 'result', 'preds', 'predictions'):
                            v = getattr(first, k, None)
                            if isinstance(v, (list, tuple)):
                                boxes_list = v
                                break
                    for b in boxes_list or []:
                        try:
                            _lbl = b.get('label') if isinstance(b, dict) else getattr(b, 'label', None)
                            if isinstance(_lbl, str) and ('cell' in _lbl):
                                coord = b.get('coordinate') if isinstance(b, dict) else getattr(b, 'coordinate', None)
                                if coord is None and isinstance(b, dict):
                                    coord = b.get('bbox') or b.get('box')
                                if isinstance(coord, (list, tuple)) and len(coord) >= 4:
                                    cx1, cy1, cx2, cy2 = coord[:4]
                                    cx1 = int(max(0, min(W, cx1))); cy1 = int(max(0, min(H, cy1)))
                                    cx2 = int(max(0, min(W, cx2))); cy2 = int(max(0, min(H, cy2)))
                                    if cx2 > cx1 and cy2 > cy1:
                                        cells.append((cx1, cy1, cx2, cy2))
                        except Exception:
                            continue
                except Exception:
                    cells = []
                word_aabbs = [_aabb_from_flat8(b) for b in orig_bboxes]
                for (cx1, cy1, cx2, cy2) in cells:
                    cell_aabb = (float(cx1), float(cy1), float(cx2), float(cy2))
                    idxs = []
                    for wi, wa in enumerate(word_aabbs):
                        inter = _intersection_area(wa, cell_aabb)
                        wa_area = _area(wa)
                        if wa_area > 0 and (inter / wa_area) >= 0.2:
                            idxs.append(wi)
                    if not idxs:
                        continue
                    def _id_key(i):
                        try:
                            return int(str(word_ids[i]))
                        except Exception:
                            return str(word_ids[i])
                    idxs.sort(key=lambda j: ((word_aabbs[j][0] + word_aabbs[j][2]) / 2.0, _id_key(j)))
                    sentence = merge_words_by_layout(
                        [orig_bboxes[k] for k in idxs],
                        [orig_words[k] for k in idxs],
                        [{'label': 'cell', 'coordinate': [cx1, cy1, cx2, cy2], 'score': 1.0}],
                        word_ids=[word_ids[k] for k in idxs],
                        prefer_id_order=True
                    )[1][0] if idxs else ""
                    if not sentence:
                        continue
                    crop = img_cv_full[cy1:cy2, cx1:cx2]
                    try:
                        preserve_polys = []
                        for k in idxs:
                            poly = _flat8_to_crop_poly(orig_bboxes[k], cx1, cy1, cx2, cy2)
                            if poly is not None and poly.size > 0:
                                preserve_polys.append(poly)
                        if preserve_polys:
                            crop = _inpaint_preserve_regions(crop, preserve_polys)
                    except Exception:
                        pass
                    crop = _apply_rotation_if_needed(crop, img_info_data['file_path'])
                    ok, buf = fast_encode_jpg(crop)
                    if not ok:
                        continue
                    img_bytes_li = bytes(buf)
                    h, w = crop.shape[:2]
                    flat8 = [0.0, 0.0, float(w), 0.0, float(w), float(h), 0.0, float(h)]
                    gt_li = {'bboxes': [flat8], 'words': [sentence], 'filename': f"{os.path.basename(img_info_data['file_path'])}_cell_{cx1}_{cy1}.jpg"}
                    cell_results.append((f"{safe_img_id}_cell_{cx1}_{cy1}", img_bytes_li, gt_li))
            elif img_cv_full is not None and tables:
                    # Fallback: TableCellsDetection 사용 불가 시, 테이블 내부 각 단어를 "셀"로 취급하여 크롭 생성
                    H, W = img_cv_full.shape[:2]
                    word_aabbs = [_aabb_from_flat8(b) for b in orig_bboxes]
                    processed_words = set()
                    # 배치 기반 셀 검출로 전환 (표 크롭들을 한 번에 처리)
                    batch_cells = _predict_table_cells_batch(img_cv_full, tables, safe_img_id)
                    for tx1, ty1, tx2, ty2, cells in (batch_cells or []):
                        for (cx1, cy1, cx2, cy2) in (cells or []):
                            cell_aabb = (float(cx1), float(cy1), float(cx2), float(cy2))
                            idxs = []
                            for wi, wa in enumerate(word_aabbs):
                                inter = _intersection_area(wa, cell_aabb)
                                wa_area = _area(wa)
                                # 셀 내부 포함 비율 완화: 50% -> 20%
                                if wa_area > 0 and (inter / wa_area) >= 0.2:
                                    idxs.append(wi)
                            if not idxs:
                                continue
                            def _id_key(i):
                                try:
                                    return int(str(word_ids[i]))
                                except Exception:
                                    return str(word_ids[i])
                            idxs.sort(key=lambda j: ((word_aabbs[j][0] + word_aabbs[j][2]) / 2.0, _id_key(j)))
                            sentence = merge_words_by_layout(
                                [orig_bboxes[k] for k in idxs],
                                [orig_words[k] for k in idxs],
                                [{'label': 'cell', 'coordinate': [cx1, cy1, cx2, cy2], 'score': 1.0}],
                                word_ids=[word_ids[k] for k in idxs],
                                prefer_id_order=True
                            )[1][0] if idxs else ""
                            if not sentence:
                                continue
                            crop = img_cv_full[cy1:cy2, cx1:cx2]
                            # inpaint: 라벨 단어만 보존
                            try:
                                preserve_polys = []
                                for k in idxs:
                                    poly = _flat8_to_crop_poly(orig_bboxes[k], cx1, cy1, cx2, cy2)
                                    if poly is not None and poly.size > 0:
                                        preserve_polys.append(poly)
                                if preserve_polys:
                                    crop = _inpaint_preserve_regions(crop, preserve_polys)
                            except Exception:
                                pass
                            crop = _apply_rotation_if_needed(crop, img_path)
                            ok, buf = fast_encode_jpg(crop)
                            if not ok:
                                continue
                            img_bytes_li = bytes(buf)
                            h, w = crop.shape[:2]
                            flat8 = [0.0, 0.0, float(w), 0.0, float(w), float(h), 0.0, float(h)]
                            gt_li = {'bboxes': [flat8], 'words': [sentence], 'filename': f"{img_file_name}_cell_{cx1}_{cy1}.jpg"}
                            cell_results.append((f"{img_id}_cell_{cx1}_{cy1}", img_bytes_li, gt_li))
                    # 레거시 단건 처리 비활성화
                    for _ in []:
                        tx1, ty1, tx2, ty2 = map(int, tb['coordinate'])
                        tx1 = max(0, min(W, tx1)); tx2 = max(0, min(W, tx2))
                        ty1 = max(0, min(H, ty1)); ty2 = max(0, min(H, ty2))
                        if tx2 <= tx1 or ty2 <= ty1:
                            continue
                        table_aabb = (float(tx1), float(ty1), float(tx2), float(ty2))
                        for wi, wa in enumerate(word_aabbs):
                            if wi in processed_words:
                                continue
                            inter = _intersection_area(wa, table_aabb)
                            wa_area = _area(wa)
                            # 테이블 내부에 20% 이상 포함된 단어만 사용
                            if wa_area <= 0 or (inter / wa_area) < 0.2:
                                continue
                            x1, y1, x2, y2 = int(wa[0]), int(wa[1]), int(wa[2]), int(wa[3])
                            # 안전 패딩
                            pad = 2
                            cx1 = max(0, x1 - pad); cy1 = max(0, y1 - pad)
                            cx2 = min(W, x2 + pad); cy2 = min(H, y2 + pad)
                            if cx2 <= cx1 or cy2 <= cy1:
                                continue
                            crop = img_cv_full[cy1:cy2, cx1:cx2]
                            # 인페인트: 해당 단어만 보존 (선택적)
                            try:
                                poly = _flat8_to_crop_poly(orig_bboxes[wi], cx1, cy1, cx2, cy2)
                                if poly is not None and poly.size > 0:
                                    crop = _inpaint_preserve_regions(crop, [poly])
                            except Exception:
                                pass
                            crop = _apply_rotation_if_needed(crop, img_info_data['file_path'])
                            ok, buf = fast_encode_jpg(crop)
                            if not ok:
                                continue
                            img_bytes_li = bytes(buf)
                            h, w = crop.shape[:2]
                            flat8 = [0.0, 0.0, float(w), 0.0, float(w), float(h), 0.0, float(h)]
                            sentence = str(orig_words[wi]) if wi < len(orig_words) else ""
                            if not sentence:
                                continue
                            gt_li = {'bboxes': [flat8], 'words': [sentence], 'filename': f"{os.path.basename(img_info_data['file_path'])}_wordcell_{cx1}_{cy1}.jpg"}
                            cell_results.append((f"{safe_img_id}_wordcell_{cx1}_{cy1}", img_bytes_li, gt_li))
                            processed_words.add(wi)
            # 셀 결과만 사용 (레이아웃/폴백 제거)
            if cell_results:
                _log_verbose(f"[finance_logistics] keep cells: {os.path.basename(img_info_data['file_path'])} layouts={layout_cnt} tables={table_cnt} cells={len(cell_results)}")
                return cell_results
            _log_verbose(f"[finance_logistics] exclude(no cells): {os.path.basename(img_info_data['file_path'])} layouts={layout_cnt} tables={table_cnt}")
            return None
        except Exception:
            pass
        
        gt_info = {
            'bboxes': bboxes,
            'words': words,
            'filename': img_info_data['filename']
        }
        # 레이아웃 단위 디버그 저장 로직 제거됨
        
        return None
        
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
        word_ids = []

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
                        ann_id = ann.get('id')
                        if ann_id is None:
                            ann_id = len(word_ids)
                        word_ids.append(ann_id)
                    elif isinstance(bbox_coords, list) and len(bbox_coords) >= 4:
                        # [x,y,w,h]
                        x, y, w, h = bbox_coords[0], bbox_coords[1], bbox_coords[2], bbox_coords[3]
                        x1, y1, x2, y2 = x, y, x + w, y + h
                        pixel_coords = [x1, y1, x2, y1, x2, y2, x1, y2]
                        pixel_coords = normalize_ic15_clockwise_flat8(pixel_coords)
                        bboxes.append(pixel_coords)
                        words.append(ann.get('text', ''))
                        ann_id = ann.get('id')
                        if ann_id is None:
                            ann_id = len(word_ids)
                        word_ids.append(ann_id)
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
                                word_ids.append(len(word_ids))
                    finally:
                        safe_close_file(json_file_handle)
                except Exception:
                    pass

        # 📦 LayoutDetection 기반 문장 병합
        orig_bboxes = list(bboxes)
        orig_words = list(words)
        try:
            layout_boxes = run_layout_detection(img_path)
            layout_cnt = len(layout_boxes) if layout_boxes else 0
            results = []
            # 1) 테이블 셀 샘플
            tables = run_layout_tables(img_path)
            table_cnt = len(tables) if tables else 0
            cell_results = []
            if tables:
                table_model = get_table_model()
                img_cv_full = _decode_image_bytes(img_data)
                if table_model is not None and img_cv_full is not None:
                    H, W = img_cv_full.shape[:2]
                    word_aabbs = [_aabb_from_flat8(b) for b in orig_bboxes]
                    # 배치 기반 셀 검출로 전환 (표 크롭들을 한 번에 처리)
                    batch_cells = _predict_table_cells_batch(img_cv_full, tables, img_file_name)
                    for tx1, ty1, tx2, ty2, cells in (batch_cells or []):
                        for (cx1, cy1, cx2, cy2) in (cells or []):
                            cell_aabb = (float(cx1), float(cy1), float(cx2), float(cy2))
                            idxs = []
                            for wi, wa in enumerate(word_aabbs):
                                inter = _intersection_area(wa, cell_aabb)
                                wa_area = _area(wa)
                                if wa_area > 0 and (inter / wa_area) >= 0.5:
                                    idxs.append(wi)
                            if not idxs:
                                continue
                            def _id_key(i):
                                try:
                                    return int(str(word_ids[i]))
                                except Exception:
                                    return str(word_ids[i])
                            idxs.sort(key=lambda j: ((word_aabbs[j][0] + word_aabbs[j][2]) / 2.0, _id_key(j)))
                            sentence = merge_words_by_layout(
                                [orig_bboxes[k] for k in idxs],
                                [orig_words[k] for k in idxs],
                                [{'label': 'cell', 'coordinate': [cx1, cy1, cx2, cy2], 'score': 1.0}],
                                word_ids=[word_ids[k] for k in idxs],
                                prefer_id_order=True
                            )[1][0] if idxs else ""
                            if not sentence:
                                continue
                            crop = img_cv_full[cy1:cy2, cx1:cx2]
                            # inpaint: 라벨 단어만 보존 (선택)
                            try:
                                preserve_polys = []
                                for k in idxs:
                                    poly = _flat8_to_crop_poly(orig_bboxes[k], cx1, cy1, cx2, cy2)
                                    if poly is not None and poly.size > 0:
                                        preserve_polys.append(poly)
                                if preserve_polys:
                                    crop = _inpaint_preserve_regions(crop, preserve_polys)
                            except Exception:
                                pass
                            # 인코딩
                            crop = _apply_rotation_if_needed(crop, img_path)
                            ok, buf = fast_encode_jpg(crop)
                            if not ok:
                                continue
                            img_bytes_li = bytes(buf)
                            h, w = crop.shape [:2]
                            flat8 = [0.0, 0.0, float(w), 0.0, float(w), float(h), 0.0, float(h)]
                            gt_li = {'bboxes': [flat8], 'words': [sentence], 'filename': f"{img_file_name}_cell_{cx1}_{cy1}.jpg"}
                            cell_results.append((f"{img_file_name}_cell_{cx1}_{cy1}", img_bytes_li, gt_li))
                    # 레거시 단건 처리 비활성화
                    for _ in []:
                        tx1, ty1, tx2, ty2 = map(int, tb['coordinate'])
                        tx1 = max(0, min(W, tx1)); tx2 = max(0, min(W, tx2))
                        ty1 = max(0, min(H, ty1)); ty2 = max(0, min(H, ty2))
                        if tx2 <= tx1 or ty2 <= ty1:
                            continue
                        crop_path = f"/tmp/ti_table_{os.getpid()}_{img_file_name}_{tx1}_{ty1}.jpg"
                        try:
                            cv2.imwrite(crop_path, img_cv_full[ty1:ty2, tx1:tx2])
                        except Exception:
                            continue
                        with TABLE_MODEL_LOCK:
                            cell_out = table_model.predict(crop_path, threshold=TABLE_THRESHOLD, batch_size=16)
                        try:
                            os.remove(crop_path)
                        except Exception:
                            pass
                        if not cell_out:
                            continue
                        cells = []
                        try:
                            for b in getattr(cell_out[0], 'boxes', []):
                                if b.get('label') == 'cell':
                                    cx1, cy1, cx2, cy2 = b.get('coordinate', [0,0,0,0])
                                    cells.append((int(tx1 + cx1), int(ty1 + cy1), int(tx1 + cx2), int(ty1 + cy2)))
                        except Exception:
                            pass
                        for (cx1, cy1, cx2, cy2) in cells:
                            cell_aabb = (float(cx1), float(cy1), float(cx2), float(cy2))
                            idxs = []
                            for wi, wa in enumerate(word_aabbs):
                                inter = _intersection_area(wa, cell_aabb)
                                wa_area = _area(wa)
                                if wa_area > 0 and (inter / wa_area) >= 0.5:
                                    idxs.append(wi)
                            if not idxs:
                                continue
                            def _id_key(i):
                                try:
                                    return int(str(word_ids[i]))
                                except Exception:
                                    return str(word_ids[i])
                            idxs.sort(key=lambda j: ((word_aabbs[j][0] + word_aabbs[j][2]) / 2.0, _id_key(j)))
                            sentence = merge_words_by_layout(
                                [orig_bboxes[k] for k in idxs],
                                [orig_words[k] for k in idxs],
                                [{'label': 'cell', 'coordinate': [cx1, cy1, cx2, cy2], 'score': 1.0}],
                                word_ids=[word_ids[k] for k in idxs],
                                prefer_id_order=True
                            )[1][0] if idxs else ""
                            if not sentence:
                                continue
                            crop = img_cv_full[cy1:cy2, cx1:cx2]
                            # inpaint: 라벨 단어만 보존
                            try:
                                preserve_polys = []
                                for k in idxs:
                                    poly = _flat8_to_crop_poly(orig_bboxes[k], cx1, cy1, cx2, cy2)
                                    if poly is not None and poly.size > 0:
                                        preserve_polys.append(poly)
                                if preserve_polys:
                                    crop = _inpaint_preserve_regions(crop, preserve_polys)
                            except Exception:
                                pass
                            # 인페인트/일반 크롭을 JPEG로 인코딩 (회전 보정 후)
                            crop = _apply_rotation_if_needed(crop, img_path)
                            ok, buf = fast_encode_jpg(crop)
                            if not ok:
                                continue
                            img_bytes_li = bytes(buf)
                            h, w = crop.shape[:2]
                            flat8 = [0.0, 0.0, float(w), 0.0, float(w), float(h), 0.0, float(h)]
                            gt_li = {'bboxes': [flat8], 'words': [sentence], 'filename': f"{img_file_name}_cell_{cx1}_{cy1}.jpg"}
                            cell_results.append((f"{img_file_name}_cell_{cx1}_{cy1}", img_bytes_li, gt_li))
            # 2) 레이아웃 크롭 샘플
            if layout_boxes:
                img_cv_full = _decode_image_bytes(img_data)
                if img_cv_full is not None:
                    H, W = img_cv_full.shape[:2]
                    word_aabbs = [_aabb_from_flat8(b) for b in orig_bboxes]
                    layout_aabbs = []
                    for lb in layout_boxes:
                        x1, y1, x2, y2 = lb['coordinate']
                        layout_aabbs.append((max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))))
                    assigned = _assign_words_to_layout(word_aabbs, layout_aabbs, min_overlap_ratio=0.3)
                    for li, la in enumerate(layout_aabbs):
                        idxs = [i for i, a in enumerate(assigned) if a == li]
                        if not idxs:
                            continue
                        x1, y1, x2, y2 = la
                        if x2 <= x1 or y2 <= y1:
                            continue
                        # 레이아웃 박스 내 단어들을 문장으로 병합
                        def _id_key(i):
                            try:
                                return int(str(word_ids[i]))
                            except Exception:
                                return str(word_ids[i])
                        # 좌->우 + id 보조 정렬
                        idxs.sort(key=lambda j: ((word_aabbs[j][0] + word_aabbs[j][2]) / 2.0, _id_key(j)))
                        sentence = merge_words_by_layout(
                            [orig_bboxes[k] for k in idxs],
                            [orig_words[k] for k in idxs],
                            [{'label': 'layout', 'coordinate': [x1, y1, x2, y2], 'score': 1.0}],
                            word_ids=[word_ids[k] for k in idxs],
                            prefer_id_order=True
                        )[1][0] if idxs else ""
                        if not sentence:
                            continue
                        crop = img_cv_full[y1:y2, x1:x2].copy()
                        # inpaint: 라벨 단어만 보존
                        try:
                            preserve_polys = []
                            for k in idxs:
                                poly = _flat8_to_crop_poly(orig_bboxes[k], x1, y1, x2, y2)
                                if poly is not None and poly.size > 0:
                                    preserve_polys.append(poly)
                            if preserve_polys:
                                crop = _inpaint_preserve_regions(crop, preserve_polys)
                        except Exception:
                            pass
                        # 인페인트/일반 크롭을 JPEG로 인코딩 (회전 보정 후)
                        crop = _apply_rotation_if_needed(crop, img_path)
                        ok, buf = fast_encode_jpg(crop)
                        if not ok:
                            continue
                        img_bytes_li = bytes(buf)
                        h, w = crop.shape[:2]
                        flat8 = [0.0, 0.0, float(w), 0.0, float(w), float(h), 0.0, float(h)]
                        gt_li = {'bboxes': [flat8], 'words': [sentence], 'filename': f"{img_file_name}_layout_{li:02d}.jpg"}
                        results.append((f"{img_file_name}_{li}", img_bytes_li, gt_li))
            # 셀이 있으면 셀만 반환
            if cell_results:
                _log_verbose(f"[handwriting] keep cells: {img_file_name} layouts={layout_cnt} tables={table_cnt} cells={len(cell_results)}")
                return cell_results
            if results:
                _log_verbose(f"[handwriting] keep layouts: {img_file_name} layouts={layout_cnt} kept={len(results)}")
                return results
            # 3) 마지막 폴백: 레이아웃/셀 모두 없으면 단어 단위 크롭 생성
            if orig_bboxes and orig_words:
                img_cv_full = _decode_image_bytes(img_data)
                if img_cv_full is not None:
                    H, W = img_cv_full.shape[:2]
                    word_samples = []
                    for wi, b in enumerate(orig_bboxes):
                        x1, y1, x2, y2 = map(int, _aabb_from_flat8(b))
                        # 안전 패딩
                        pad = 2
                        cx1 = max(0, x1 - pad); cy1 = max(0, y1 - pad)
                        cx2 = min(W, x2 + pad); cy2 = min(H, y2 + pad)
                        if cx2 <= cx1 or cy2 <= cy1:
                            continue
                        crop = img_cv_full[cy1:cy2, cx1:cx2]
                        # 선택적 인페인트: 해당 단어만 보존
                        try:
                            poly = _flat8_to_crop_poly(b, cx1, cy1, cx2, cy2)
                            if poly is not None and poly.size > 0:
                                crop = _inpaint_preserve_regions(crop, [poly])
                        except Exception:
                            pass
                        crop = _apply_rotation_if_needed(crop, img_path)
                        ok, buf = fast_encode_jpg(crop)
                        if not ok:
                            continue
                        img_bytes_li = bytes(buf)
                        h, w = crop.shape[:2]
                        flat8 = [0.0, 0.0, float(w), 0.0, float(w), float(h), 0.0, float(h)]
                        sentence = str(orig_words[wi]) if wi < len(orig_words) else ""
                        if not sentence:
                            continue
                        gt_li = {'bboxes': [flat8], 'words': [sentence], 'filename': f"{img_file_name}_word_{cx1}_{cy1}.jpg"}
                        word_samples.append((f"{img_file_name}_word_{cx1}_{cy1}", img_bytes_li, gt_li))
                    if word_samples:
                        _log_verbose(f"[handwriting] keep words(fallback): {img_file_name} word_samples={len(word_samples)}")
                        return word_samples
            # 제외
            _log_verbose(f"[handwriting] exclude: {img_file_name} layouts={layout_cnt} tables={table_cnt} cells=0 words=0")
            return None
        except Exception:
            pass

        gt_info = {
            'bboxes': bboxes,
            'words': words,
            'filename': img_info_data['filename']
        }
        # 레이아웃 단위 디버그 저장 로직 제거됨
        return None
    except Exception:
        return None

def create_parallel_lmdb_from_args(process_args, output_path, split_name, process_func, max_workers=None, path_extractor=None, gpu_prefetch_batch_size=None):
    """공통 병렬 LMDB 생성 함수 (메모리 절약형)
    path_extractor: 각 arg에서 이미지 경로를 뽑는 함수(배치 GPU 선계산용)
    gpu_prefetch_batch_size: 선계산 배치 크기(설정 시 선계산 활성화)
    """
    print(f"🚀 {split_name} 병렬 LMDB 생성 중... ({len(process_args)}개 샘플)")
    
    # CPU 코어 수에 따른 최적 워커 수
    if max_workers is None:
        # CPU 코어 수 기준 상한 16
        _cpus = mp.cpu_count() or 16
        max_workers = min(_cpus, 16)
    print(f"  🔧 병렬 워커 수: {max_workers}개")
    
    # LMDB 환경 생성 (메모리 최적화 설정)
    # lmdb.open(subdir=True 기본)에서는 output_path(디렉토리)가 실제로 존재해야 한다.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    env = lmdb.open(output_path, 
                    map_size=1099511627776,  # 1TB
                    writemap=True,  # 메모리 매핑 최적화
                    meminit=False,  # 메모리 초기화 비활성화
                    map_async=True)  # 비동기 맵핑
    
    # 스키마/메타 명시 기록
    try:
        txn_meta = env.begin(write=True)
        txn_meta.put('scheme'.encode(), 'det'.encode())
        txn_meta.put('format_version'.encode(), '1'.encode())
        txn_meta.put('image_ext'.encode(), 'jpg'.encode())
        txn_meta.put('serializer'.encode(), 'pickle'.encode())
        txn_meta.put('bboxes_type'.encode(), 'ic15_flat8'.encode())
        txn_meta.commit()
    except Exception:
        pass
    
    print(f"  🔄 병렬 처리 + 즉시 저장 시작...")
    
    idx = 0
    start_time = time.time()
    
    # 청크 단위로 스트리밍 처리
    chunk_size = 1000  # 10000개씩 청크로 나누어 처리
    
    # 전역 GPU 프리페치 워커 시작(비활성화 고정: 메모리 급증 방지)
    _use_prefetch = False
    if _use_prefetch:
        try:
            _start_gpu_prefetch_worker(gpu_prefetch_batch_size)
        except Exception:
            _use_prefetch = False

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # process_args를 청크 단위로 순회
        for chunk_start in tqdm(range(0, len(process_args), chunk_size), desc=f"{split_name} 청크 처리"):
            chunk_end = min(chunk_start + chunk_size, len(process_args))
            chunk_args = process_args[chunk_start:chunk_end]

            # 동기 레이아웃 배치 선예측(배열 입력)으로 캐시 채우기
            if path_extractor is not None:
                try:
                    paths = []
                    for arg in chunk_args:
                        try:
                            pth = path_extractor(arg)
                        except Exception:
                            pth = None
                        if pth and os.path.exists(pth):
                            paths.append(pth)
                    if paths:
                        paths = list(dict.fromkeys(paths))
                        _layout_predict_batch_numpy(paths, threshold=LAYOUT_THRESHOLD)
                except Exception:
                    pass

            # GPU 선계산 큐 투입(옵션, 중복 제거)
            if _use_prefetch:
                try:
                    paths = []
                    for arg in chunk_args:
                        try:
                            pth = path_extractor(arg)
                        except Exception:
                            pth = None
                        if pth and os.path.exists(pth):
                            paths.append(pth)
                    if paths:
                        # 입력 순서를 유지하면서 중복 제거
                        paths = list(dict.fromkeys(paths))
                        _gpu_prefetch_enqueue(paths)
                except Exception:
                    pass
            
            # 현재 청크의 future만 생성
            futures = {executor.submit(process_func, arg) for arg in chunk_args}
            
            # 더 작은 트랜잭션 단위로 분할 (메모리 누적 방지)
            txn_batch_size = int(os.environ.get('FAST_TXN_BATCH', '1000'))  # 환경변수로 조절
            batch_count = 0
            txn = None
            
            # 현재 청크의 작업만 처리
            for future in as_completed(futures):
                result = future.result()
                
                if result is not None:
                    # 결과가 단일 샘플(tuple) 또는 다중 샘플(list[tuple]) 모두 처리
                    results_iter = result if isinstance(result, list) else [result]
                    for item in results_iter:
                        if item is None:
                            continue
                        try:
                            img_id, img_data, gt_info = item
                        except Exception:
                            continue
                        
                        # 새 트랜잭션 시작 (배치 단위)
                        if batch_count % txn_batch_size == 0:
                            if txn is not None:
                                txn.commit()  # 이전 트랜잭션 커밋
                            txn = env.begin(write=True)  # 새 트랜잭션 시작
                        
                        # LMDB에 즉시 저장
                        img_key = f'image-{idx:09d}'.encode()
                        gt_key = f'gt-{idx:09d}'.encode()
                        
                        txn.put(img_key, img_data)
                        txn.put(gt_key, pickle.dumps(gt_info))
                        idx += 1
                        batch_count += 1
                        
                        # 즉시 메모리 해제
                        del img_data
                        del gt_info
            
            # 마지막 트랜잭션 커밋
            if txn is not None:
                txn.commit()
            del chunk_args, futures
            
            # 강제 가비지 컬렉션
            collected = gc.collect()
            print(f"  🗑️ 청크 {chunk_start//chunk_size + 1} 완료: {idx}개 처리, GC {collected}개 해제")
        
        # 마지막에 샘플 수 저장
        txn = env.begin(write=True)
        txn.put('num-samples'.encode(), str(idx).encode())
        txn.commit()
    
    env.close()
    # GPU 프리페치 워커 정리
    if _use_prefetch:
        try:
            _stop_gpu_prefetch_worker()
        except Exception:
            pass
    
    # 최종 메모리 해제
    del process_args
    gc.collect()
    
    total_time = time.time() - start_time
    speed = idx / total_time if total_time > 0 else 0
    print(f"✅ {split_name} 병렬 LMDB 생성 완료: {idx}개 샘플")
    print(f"   ⏱️ 총 소요 시간: {total_time:.2f}초")
    print(f"   🚀 처리 속도: {speed:.1f} samples/sec")
    print(f"🗑️ {split_name} 모든 메모리 해제 완료")

def create_lmdb_text_in_wild_from_ids(base_path, images_info, image_annotations, img_ids, output_path, split_name):
    """Text in the wild 이미지 ID 리스트로부터 LMDB 생성 (thread_map 병렬처리 버전)"""
    print(f"🚀 {split_name} 병렬 LMDB 생성 중... ({len(img_ids)}개 샘플)")
    
    # CPU 코어 수에 따른 최적 워커 수
    max_workers = min(mp.cpu_count(), 16)
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
    # lmdb.open(subdir=True 기본)에서는 output_path(디렉토리)가 실제로 존재해야 한다.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    env = lmdb.open(output_path, 
                    map_size=1099511627776,  # 1TB
                    writemap=True,  # 메모리 매핑 최적화
                    meminit=False,  # 메모리 초기화 비활성화
                    map_async=True)  # 비동기 맵핑
    
    # 스키마/메타 명시 기록
    try:
        txn_meta = env.begin(write=True)
        txn_meta.put('scheme'.encode(), 'det'.encode())
        txn_meta.put('format_version'.encode(), '1'.encode())
        txn_meta.put('image_ext'.encode(), 'jpg'.encode())
        txn_meta.put('serializer'.encode(), 'pickle'.encode())
        txn_meta.put('bboxes_type'.encode(), 'ic15_flat8'.encode())
        txn_meta.commit()
    except Exception:
        pass
    
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
                    # Text in Wild의 경우 result가 다중 샘플(list)일 수 있음
                    results_iter = result if isinstance(result, list) else [result]
                    for item in results_iter:
                        if item is None:
                            continue
                        # item 형태: (img_id, img_bytes, gt_info)
                        try:
                            img_id_item, img_data_item, gt_info_item = item
                        except Exception:
                            continue
                        # 새 트랜잭션 시작 (배치 단위)
                        if batch_count % txn_batch_size == 0:
                            if txn is not None:
                                txn.commit()  # 이전 트랜잭션 커밋
                            txn = env.begin(write=True)  # 새 트랜잭션 시작
                        
                        # LMDB에 즉시 저장
                        img_key = f'image-{idx:09d}'.encode()
                        gt_key = f'gt-{idx:09d}'.encode()
                        
                        txn.put(img_key, img_data_item)
                        txn.put(gt_key, pickle.dumps(gt_info_item))
                        idx += 1
                        batch_count += 1
                        
                        # 즉시 메모리 해제
                        del img_data_item
                        del gt_info_item
            
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
    train_output_path = f"{LOCAL_OUTPUT_PATH}/public_admin_train_layout.lmdb"
    valid_output_path = f"{LOCAL_OUTPUT_PATH}/public_admin_valid_layout.lmdb"
    
    # Training LMDB 생성
    if os.path.exists(train_json_path):
        print(f"📊 Training JSON 파일 발견: {train_json_path}")
        create_lmdb_public_admin_from_json(base_path, train_json_path, train_output_path, "공공행정문서 Train", max_samples)
        test_fast_model_input(train_output_path)
        cleanup_memory()
    else:
        print(f"❌ Training JSON 파일을 찾을 수 없습니다: {train_json_path}")
    
    # Validation LMDB 생성
    if os.path.exists(valid_json_path):
        print(f"📊 Validation JSON 파일 발견: {valid_json_path}")
        create_lmdb_public_admin_from_json(base_path, valid_json_path, valid_output_path, "공공행정문서 Valid", max_samples)
        test_fast_model_input(valid_output_path)
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
    train_output_path = f"{LOCAL_OUTPUT_PATH}/public_admin_train_partly_layout.lmdb"
    
    # Training LMDB 생성
    if os.path.exists(train_json_path):
        print(f"📊 Training JSON 파일 발견: {train_json_path}")
        create_lmdb_public_admin_from_json(base_path, train_json_path, train_output_path, "공공행정문서 Train Partly", max_samples)
        test_fast_model_input(train_output_path)
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
        anns = data.get('annotations', [])
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
        
        # 메모리 절약: 이미지별 어노테이션을 메모리 dict로 만들지 않고 SQLite 임시 DB에 저장
        tmp_sqlite = f"/tmp/public_admin_anns_{os.getpid()}.db"
        try:
            if os.path.exists(tmp_sqlite):
                os.remove(tmp_sqlite)
        except Exception:
            pass
        print(f"  💾 임시 SQLite 생성: {tmp_sqlite}")
        conn = sqlite3.connect(tmp_sqlite)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=OFF;")
        conn.execute("CREATE TABLE a(image_id INTEGER, ann TEXT)")
        # 대용량 인서트
        batch = []
        count = 0
        for ann in anns:
            try:
                img_id = ann.get('image_id', ann.get('id'))
                batch.append((int(img_id), orjson.dumps(ann).decode('utf-8')))
                if len(batch) >= 10000:
                    conn.executemany("INSERT INTO a(image_id, ann) VALUES (?,?)", batch)
                    conn.commit()
                    count += len(batch)
                    print(f"    🧱 SQLite 적재: {count}개")
                    batch = []
            except Exception:
                continue
        if batch:
            conn.executemany("INSERT INTO a(image_id, ann) VALUES (?,?)", batch)
            conn.commit()
            count += len(batch)
        print(f"  ✅ SQLite 적재 완료: 총 {count}개")
        # 즉시 원본 JSON 메모리 해제
        del data
        del anns
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
        
        # 🚀 병렬 처리용 데이터 준비 (주석: annotations는 SQLite 경로/이미지ID만 전달)
        process_args = []
        for i, img_idx in enumerate(indices):
            img_info = images[img_idx]  # orjson Python 리스트에서 직접 접근
            img_id = img_info.get('id', i)
            ann_ref = {'sqlite': tmp_sqlite, 'image_id': img_id}
            process_args.append((img_info, ann_ref, base_path, lookup_func, dataset_lookup_name, image_path_cache))
        print(f"  📊 병렬 처리용 데이터 준비 완료: {len(process_args)}개")
        # 🚀 즉시 원본 리스트 해제
        del images
        del indices
        gc.collect()
        print(f"  🗑️ 원본 리스트 메모리 해제 완료")
        
        # 🚀 병렬 LMDB 생성
        gpu_bs = int(os.environ.get('FAST_LAYOUT_BATCH', '8'))
        try:
            # 전역 경로 설정 (워커에서 사용)
            global PUBLIC_ADMIN_SQLITE_PATH
            PUBLIC_ADMIN_SQLITE_PATH = tmp_sqlite
            create_parallel_lmdb_from_args(
                process_args, output_path, dataset_name, process_single_public_admin_image,
                path_extractor=_extract_path_public_admin, gpu_prefetch_batch_size=gpu_bs
            )
        finally:
            try:
                conn.close()
            except Exception:
                pass
            # 병렬 처리 종료 후 임시 DB 삭제 시도 (다음 실행 시 새로 생성)
            try:
                if os.path.exists(tmp_sqlite):
                    os.remove(tmp_sqlite)
            except Exception:
                pass
        
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
    train_output_path = f"{LOCAL_OUTPUT_PATH}/ocr_public_train_layout.lmdb"
    valid_output_path = f"{LOCAL_OUTPUT_PATH}/ocr_public_valid_layout.lmdb"
    
    # Training LMDB 생성
    if os.path.exists(train_json_path):
        print(f"📊 Training JSON 파일 발견: {train_json_path}")
        create_lmdb_ocr_public_from_json(base_path, train_json_path, train_output_path, "OCR 공공 Train", max_samples)
        test_fast_model_input(train_output_path)
        cleanup_memory()
    else:
        print(f"❌ Training JSON 파일을 찾을 수 없습니다: {train_json_path}")
    
    # Validation LMDB 생성
    if os.path.exists(valid_json_path):
        print(f"📊 Validation JSON 파일 발견: {valid_json_path}")
        create_lmdb_ocr_public_from_json(base_path, valid_json_path, valid_output_path, "OCR 공공 Valid", max_samples)
        test_fast_model_input(valid_output_path)
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
    data, file_handle = load_json_with_orjson(json_path)
    try:
        images = data.get('images', [])
        print(f"📊 JSON 파일 로드 완료")
        if max_samples and len(images) > max_samples:
            print(f"📊 {max_samples}개 샘플로 제한 (총 {len(images)}개 중)")
            random.seed(42)
            random.shuffle(images)
            images = images[:max_samples]
        # 이미지별 어노테이션 그룹화
        image_annotations = {}
        annotations = data.get('annotations', [])
        print("  🔄 어노테이션 그룹화 중...")
        for ann in tqdm(annotations, desc="어노테이션 그룹화"):
            img_id = ann.get('image_id', ann.get('id'))
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)
        print(f"  ✅ 어노테이션 그룹화 완료: {len(image_annotations)}개 이미지")
        del data
        del annotations
        print(f"  🗑️ 원본 JSON 데이터 메모리 해제 완료")
    finally:
        safe_close_file(file_handle)
        file_handle = None
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
    gpu_bs = int(os.environ.get('FAST_LAYOUT_BATCH', '8'))
    create_parallel_lmdb_from_args(
        process_args, output_path, dataset_name, process_single_ocr_public_image,
        path_extractor=_extract_path_ocr_public, gpu_prefetch_batch_size=gpu_bs
    )
    # (임시 인덱스 사용 안 함)

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
    train_output_path = f"{LOCAL_OUTPUT_PATH}/finance_logistics_train_layout.lmdb"
    valid_output_path = f"{LOCAL_OUTPUT_PATH}/finance_logistics_valid_layout.lmdb"
    
    # Training LMDB 생성
    if os.path.exists(train_json_path):
        print(f"📊 Training JSON 파일 발견: {train_json_path}")
        create_lmdb_finance_logistics_from_json(base_path, train_json_path, train_output_path, "금융물류 Train", max_samples)
        test_fast_model_input(train_output_path)
        cleanup_memory()
    else:
        print(f"❌ Training JSON 파일을 찾을 수 없습니다: {train_json_path}")
    
    # Validation LMDB 생성
    if os.path.exists(valid_json_path):
        print(f"📊 Validation JSON 파일 발견: {valid_json_path}")
        create_lmdb_finance_logistics_from_json(base_path, valid_json_path, valid_output_path, "금융물류 Valid", max_samples)
        test_fast_model_input(valid_output_path)
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
        gpu_bs = int(os.environ.get('FAST_LAYOUT_BATCH', '8'))
        create_parallel_lmdb_from_args(
            process_args, output_path, dataset_name, process_single_finance_logistics_image,
            max_workers=16, path_extractor=_extract_path_finance_logistics, gpu_prefetch_batch_size=gpu_bs
        )
        
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
    train_output_path = f"{LOCAL_OUTPUT_PATH}/handwriting_train_layout.lmdb"
    valid_output_path = f"{LOCAL_OUTPUT_PATH}/handwriting_valid_layout.lmdb"
    
    # Training LMDB 생성
    if os.path.exists(train_json_path):
        print(f"📊 Training JSON 파일 발견: {train_json_path}")
        create_lmdb_handwriting_from_json(base_path, train_json_path, train_output_path, "손글씨 Train", max_samples)
        test_fast_model_input(train_output_path)
        cleanup_memory()
    else:
        print(f"❌ Training JSON 파일을 찾을 수 없습니다: {train_json_path}")
    
    # Validation LMDB 생성
    if os.path.exists(valid_json_path):
        print(f"📊 Validation JSON 파일 발견: {valid_json_path}")
        create_lmdb_handwriting_from_json(base_path, valid_json_path, valid_output_path, "손글씨 Valid", max_samples)
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
        gpu_bs = int(os.environ.get('FAST_LAYOUT_BATCH', '8'))
        create_parallel_lmdb_from_args(
            process_args, output_path, dataset_name, process_single_handwriting_image,
            path_extractor=_extract_path_handwriting, gpu_prefetch_batch_size=gpu_bs
        )
        
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

def _predict_table_cells_batch(img_cv_full, tables, id_tag):
    """여러 테이블 영역에 대해 셀 박스를 배치 예측.
    반환: [(tx1,ty1,tx2,ty2, [(cx1,cy1,cx2,cy2), ...]), ...]
    """
    if img_cv_full is None or img_cv_full.size == 0 or not tables:
        return []
    table_model = get_table_model()
    if table_model is None:
        return []
    H, W = img_cv_full.shape[:2]
    # 유효 테이블만 정리
    table_regions = []
    for tb in tables:
        try:
            tx1, ty1, tx2, ty2 = map(int, tb['coordinate'])
            tx1 = max(0, min(W, tx1)); tx2 = max(0, min(W, tx2))
            ty1 = max(0, min(H, ty1)); ty2 = max(0, min(H, ty2))
            if tx2 <= tx1 or ty2 <= ty1:
                continue
            table_regions.append((tx1, ty1, tx2, ty2))
        except Exception:
            continue
    if not table_regions:
        return []
    # 전역 배처(여러 이미지 간 집계 배치) 사용 여부 - 코드 고정(메모리 가드 적용)
    use_agg = True
    if use_agg:
        # 요청들을 전부 제출 후, 결과 일괄 대기
        batcher = _get_global_table_batcher()
        reqs = []
        for (tx1, ty1, tx2, ty2) in table_regions:
            try:
                crop = img_cv_full[ty1:ty2, tx1:tx2]
                if crop is None or crop.size == 0:
                    continue
                # 메모리/VRAM 피크 완화: 최장변 상한으로 다운스케일
                ch, cw = crop.shape[:2]
                m = max(ch, cw)
                if m > CELL_CROP_MAX_SIDE:
                    scale = CELL_CROP_MAX_SIDE / float(m)
                    new_w = max(1, int(cw * scale))
                    new_h = max(1, int(ch * scale))
                    crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
                fut = batcher.submit(crop, (tx1, ty1, tx2, ty2))
                reqs.append(((tx1, ty1, tx2, ty2), fut))
            except Exception:
                continue
        results = []
        for (tx1, ty1, tx2, ty2), fut in reqs:
            # 결과 대기 시간 고정(메모리 방어 목적)
            cells = fut.result(timeout=60.0)
            if not isinstance(cells, list):
                cells = []
            results.append((tx1, ty1, tx2, ty2, cells))
        print(f"[cell/agg] collected={len(reqs)} returned={len(results)}")
        return results
    # 메모리 배열 기반 배치 예측 (임시 파일 사용 안 함, 단일 이미지 내 배치)
    print(f"[cell/batch] start total_crops={len(table_regions)} in_memory=1")
    if not table_regions:
        return []
    table_bs = 8  # 보수적 고정
    results = []
    total_cells = 0
    t_batch0 = time.time()
    for s in range(0, len(table_regions), table_bs):
        chunk = table_regions[s:s+table_bs]
        batch = []
        for (tx1, ty1, tx2, ty2) in chunk:
            try:
                crop = img_cv_full[ty1:ty2, tx1:tx2]
                if crop is None or crop.size == 0:
                    continue
                ch, cw = crop.shape[:2]
                m = max(ch, cw)
                if m > CELL_CROP_MAX_SIDE:
                    scale = CELL_CROP_MAX_SIDE / float(m)
                    new_w = max(1, int(cw * scale))
                    new_h = max(1, int(ch * scale))
                    crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
                batch.append((crop, (tx1, ty1, tx2, ty2)))
            except Exception:
                continue
        if not batch:
            continue
        batch_arrays = [arr for arr, _ in batch]
        bs = min(len(batch_arrays), table_bs)
        t0 = time.time()
        with TABLE_MODEL_LOCK:
            cell_out_list = table_model.predict(batch_arrays, threshold=TABLE_THRESHOLD, batch_size=bs)
        t1 = time.time()
        out_cells = 0
        for (_, region), cell_out in zip(batch, cell_out_list or []):
            tx1, ty1, tx2, ty2 = region
            cells = []
            try:
                first = cell_out[0] if isinstance(cell_out, (list, tuple)) and cell_out else cell_out
                try:
                    dbg_type = type(first).__name__
                    if isinstance(first, dict):
                        dbg_keys = list(first.keys())[:16]
                    else:
                        dbg_keys = [k for k in ('boxes','result','preds','predictions') if getattr(first,k,None) is not None]
                    print(f"[debug] cells/raw: type={dbg_type} keys={dbg_keys}")
                except Exception:
                    pass
                parsed = _extract_cell_boxes(first)
                for b in parsed or []:
                    cx1, cy1, cx2, cy2 = b['coordinate']
                    cells.append((int(tx1 + cx1), int(ty1 + cy1), int(tx1 + cx2), int(ty1 + cy2)))
            except Exception:
                cells = []
            results.append((tx1, ty1, tx2, ty2, cells))
            out_cells += len(cells)
        print(f"[cell/chunk] {s//table_bs+1}/{(len(table_regions)+table_bs-1)//table_bs} n={len(batch_arrays)} bs={bs} ms={(t1-t0)*1000:.1f} cells={out_cells}")
        total_cells += out_cells
        # 메모리 해제 힌트
        del batch_arrays
    t_batch1 = time.time()
    print(f"[cell/batch] done total_crops={len(table_regions)} total_cells={total_cells} total_ms={(t_batch1-t_batch0)*1000:.1f}")
    return results

def _gpu_prefetch_layout_for_paths(img_paths, batch_size=8):
    """주어진 경로들에 대해 LayoutDetection을 배치로 수행하고 캐시에 저장."""
    if not img_paths:
        return
    try:
        # 전역 테이블 배처가 필요하면 초기화만 선행(모델 lazy init)
        try:
            if str(os.environ.get('FAST_TABLE_AGG', '1')).lower() in ('1','true','yes','y'):
                _get_global_table_batcher()
        except Exception:
            pass
        model = get_layout_model()
        paths = [p for p in img_paths if p and os.path.exists(p)]
        if not paths:
            return
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i+batch_size]
            try:
                t0 = time.time()
                with LAYOUT_MODEL_LOCK:
                    out_list = model.predict(batch, batch_size=len(batch), layout_nms=True, threshold=LAYOUT_THRESHOLD)
                t1 = time.time()
                _log_verbose(f"[layout/prefetch] batch={len(batch)} ms={(t1-t0)*1000:.1f}")
            except Exception:
                out_list = []
                for p in batch:
                    try:
                        with LAYOUT_MODEL_LOCK:
                            single = model.predict(p, batch_size=1, layout_nms=True, threshold=LAYOUT_THRESHOLD)
                        out_list.append(single[0] if single else None)
                    except Exception:
                        out_list.append(None)
            for p, res in zip(batch, out_list):
                boxes = []
                try:
                    for b in getattr(res, 'boxes', []):
                        label = b.get('label')
                        coord = b.get('coordinate')
                        if label in LAYOUT_LABELS_TO_USE and isinstance(coord, (list, tuple)) and len(coord) == 4:
                            boxes.append({'label': label, 'coordinate': [float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3])], 'score': float(b.get('score', 1.0))})
                except Exception:
                    boxes = []
                # 대소문자 무시로 테이블 라벨 추출
                tables = [b for b in boxes if isinstance(b.get('label'), str) and b.get('label').lower() == 'table']
                _cache_update(p, layout=boxes, tables=tables)
                # 선택: 테이블 셀까지 사전 예측하여 캐시에 저장
                if PREFETCH_TABLES and tables:
                    try:
                        with open(p, 'rb') as _f:
                            _arr = np.frombuffer(_f.read(), dtype=np.uint8)
                        img_cv_full = cv2.imdecode(_arr, cv2.IMREAD_COLOR)
                    except Exception:
                        img_cv_full = None
                    if img_cv_full is not None:
                        try:
                            cells = _predict_table_cells_batch(img_cv_full, tables, f"pf")
                        except Exception:
                            cells = None
                        if cells:
                            _cache_update(p, table_cells=cells)  # [(tx1,ty1,tx2,ty2,[(cx1,cy1,cx2,cy2)..]), ...]
    except Exception:
        pass


def _prefetch_predictions_for_args(args_list, path_extractor, batch_size=8):
    """args 리스트에서 경로를 추출해 GPU 배치 선계산."""
    try:
        paths = []
        for arg in args_list:
            try:
                p = path_extractor(arg)
            except Exception:
                p = None
            if p and os.path.exists(p):
                paths.append(p)
        if not paths:
            return
        # 중복 제거 (입력 순서 유지)
        paths = list(dict.fromkeys(paths))
        _gpu_prefetch_layout_for_paths(paths, batch_size=batch_size)
    except Exception:
        pass


def _extract_path_text_in_wild(arg):
    """(img_id, img_info, annotations, base_path, lookup_dict) -> img_path"""
    try:
        _, img_info, _, base_path, lookup_dict = arg
        img_file_name = img_info.get('file_name', '')
        if img_file_name and not img_file_name.endswith('.jpg'):
            img_file_name = f"{img_file_name}.jpg"
        if lookup_dict and isinstance(lookup_dict, dict):
            if img_file_name in lookup_dict:
                return lookup_dict[img_file_name]
            for ext in ['.png', '.jpeg']:
                alt = img_file_name.replace('.jpg', ext)
                if alt in lookup_dict:
                    return lookup_dict[alt]
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
        return os.path.join(image_dir, img_file_name) if img_file_name else None
    except Exception:
        return None


def _extract_path_public_admin(arg):
    """(img_info, annotations, base_path, lookup_dict, dataset_lookup_name, image_path_cache) -> img_path"""
    try:
        img_info, _, base_path, _, dataset_lookup_name, image_path_cache = arg
        img_file_name = img_info.get('image.file.name', '')
        if not img_file_name:
            return None
        return optimized_find_image_path(img_file_name, base_path, dataset_lookup_name, image_path_cache)
    except Exception:
        return None


def _extract_path_ocr_public(arg):
    """(img_info, annotations, base_path, dataset_lookup_name, image_path_cache) -> img_path"""
    try:
        img_info, _, base_path, dataset_lookup_name, image_path_cache = arg
        img_file_name = img_info.get('file_name', '')
        if img_file_name and not img_file_name.endswith(('.jpg', '.png', '.jpeg')):
            img_file_name = f"{img_file_name}.jpg"
        return optimized_find_image_path(img_file_name, base_path, dataset_lookup_name, image_path_cache)
    except Exception:
        return None


def _extract_path_finance_logistics(arg):
    """(sub_dataset, img_info_data, annotations_for_dataset) -> file_path"""
    try:
        _, info, _ = arg
        return info.get('file_path')
    except Exception:
        return None


def _extract_path_handwriting(arg):
    """(img_file_name, img_info_data, anns) -> file_path"""
    try:
        _, info, _ = arg if len(arg) == 3 else (arg[0], arg[1], [])
        return info.get('file_path')
    except Exception:
        return None

# ============================================================================
# 전역 테이블 배처 (여러 이미지의 테이블 크롭을 모아 대배치 추론)
# ============================================================================
class _TableCellsFuture:
    def __init__(self):
        import threading
        self._event = threading.Event()
        self._result = None
    def set_result(self, value):
        self._result = value
        try:
            self._event.set()
        except Exception:
            pass
    def result(self, timeout=None):
        try:
            ok = self._event.wait(timeout)
            if not ok:
                return None
        except Exception:
            return None
        return self._result

class _TableBatcher:
    def __init__(self, table_bs=24, timeout_ms=50):
        import threading
        self._q = []
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._table_bs = int(table_bs)
        self._timeout_ms = int(timeout_ms)
        self._max_pending = int(TABLE_AGG_MAX_PENDING)
        self._thr = threading.Thread(target=self._worker, daemon=True)
        self._thr.start()
    def submit(self, crop_array, region_tuple):
        fut = _TableCellsFuture()
        with self._cv:
            # 대기열이 가득 차면 공간 날 때까지 대기(백프레셔)
            while len(self._q) >= self._max_pending:
                self._cv.wait(timeout=0.05)
            self._q.append((crop_array, region_tuple, fut))
            self._cv.notify()
        return fut
    def _worker(self):
        import time
        model = get_table_model()
        if model is None:
            return
        while True:
            try:
                with self._cv:
                    t_start = None
                    batch = []
                    while True:
                        if self._q:
                            if t_start is None:
                                t_start = time.time()
                            batch.append(self._q.pop(0))
                            if len(batch) >= self._table_bs:
                                break
                        else:
                            # 대기열이 비었으면 대기
                            self._cv.wait(timeout=self._timeout_ms / 1000.0)
                        if t_start is not None and (time.time() - t_start) * 1000.0 >= self._timeout_ms:
                            break
                    if not batch:
                        continue
                    # 큐 소비 알림(프로듀서 깨우기)
                    self._cv.notify_all()
                # 락 밖에서 추론
                arrays = [arr for (arr, _, _) in batch]
                regions = [r for (_, r, _) in batch]
                futs = [f for (_, _, f) in batch]
                bs = len(arrays)
                t0 = time.time()
                with TABLE_MODEL_LOCK:
                    out_list = model.predict(arrays, threshold=TABLE_THRESHOLD, batch_size=bs)
                t1 = time.time()
                total_cells = 0
                results_cells = []
                for region, out in zip(regions, out_list or []):
                    tx1, ty1, tx2, ty2 = region
                    cells = []
                    try:
                        first = out[0] if isinstance(out, (list, tuple)) and out else out
                        parsed = _extract_cell_boxes(first)
                        for b in parsed or []:
                            cx1, cy1, cx2, cy2 = b['coordinate']
                            cells.append((int(tx1 + cx1), int(ty1 + cy1), int(tx1 + cx2), int(ty1 + cy2)))
                    except Exception:
                        cells = []
                    results_cells.append(cells)
                    total_cells += len(cells)
                # futures 해제
                for fut, cells in zip(futs, results_cells):
                    fut.set_result(cells)
                print(f"[cell/agg] batch n={bs} ms={(t1-t0)*1000:.1f} cells={total_cells}")
                # 메모리 해제 힌트
                del arrays
            except Exception:
                # 에러 시 잠깐 쉬고 루프 지속
                time.sleep(0.01)

_TABLE_BATCHER = None
_TABLE_BATCHER_LOCK = threading.Lock()

def _get_global_table_batcher():
    global _TABLE_BATCHER
    if _TABLE_BATCHER is not None:
        return _TABLE_BATCHER
    with _TABLE_BATCHER_LOCK:
        if _TABLE_BATCHER is not None:
            return _TABLE_BATCHER
        # 코드 고정값으로 초기화(환경변수 미사용)
        bs = 8
        timeout_ms = 50
        _TABLE_BATCHER = _TableBatcher(table_bs=bs, timeout_ms=timeout_ms)
        print(f"[cell/agg] initialized table_bs={bs} timeout_ms={timeout_ms}")
        return _TABLE_BATCHER


def main():
    """메인 함수"""
    print("🚀 모든 한국어 OCR 데이터셋 train/valid LMDB 생성 (전체 데이터, 제한 없음)")
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
    
    # LMDB 유효성(샘플 존재) 검사 함수
    def _lmdb_has_samples(lmdb_path: str) -> bool:
        try:
            if not os.path.exists(lmdb_path):
                return False
            env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
            with env.begin() as txn:
                num = txn.get('num-samples'.encode())
                if num is not None:
                    try:
                        return int(num) > 0
                    except Exception:
                        return False
                # num-samples 키가 없으면 이미지 키 존재 여부로 대체 확인
                cur = txn.cursor()
                try:
                    # 우선 image- 접두사 탐색
                    if cur.set_range(b'image-'):
                        return True
                    # 아무 키나 하나라도 있으면 완료로 간주
                    if cur.first():
                        return True
                    return False
                finally:
                    cur.close()
        except Exception:
            return False
    
    # 이미 완료된 LMDB 확인
    completed_lmdbs = []
    lmdb_paths = [
        f"{LOCAL_OUTPUT_PATH}/text_in_wild_train_layout.lmdb",
        f"{LOCAL_OUTPUT_PATH}/text_in_wild_valid_layout.lmdb",
        f"{LOCAL_OUTPUT_PATH}/public_admin_train_layout.lmdb",
        f"{LOCAL_OUTPUT_PATH}/public_admin_train_partly_layout.lmdb",
        f"{LOCAL_OUTPUT_PATH}/public_admin_valid_layout.lmdb",
        f"{LOCAL_OUTPUT_PATH}/ocr_public_train_layout.lmdb",
        f"{LOCAL_OUTPUT_PATH}/ocr_public_valid_layout.lmdb",
        f"{LOCAL_OUTPUT_PATH}/finance_logistics_train_layout.lmdb",
        f"{LOCAL_OUTPUT_PATH}/finance_logistics_valid_layout.lmdb",
        f"{LOCAL_OUTPUT_PATH}/handwriting_train_layout.lmdb",
        f"{LOCAL_OUTPUT_PATH}/handwriting_valid_layout.lmdb"
    ]
    
    for lmdb_path in lmdb_paths:
        if os.path.exists(lmdb_path):
            completed_lmdbs.append(lmdb_path)
            print(f"✅ 이미 완료됨: {os.path.basename(lmdb_path)}")
    
    # 각 데이터셋별로 train/valid LMDB 생성 (완료된 것 제외) - 전체 데이터 처리
    _tiw_train = f"{LOCAL_OUTPUT_PATH}/text_in_wild_train_layout.lmdb"
    _tiw_valid = f"{LOCAL_OUTPUT_PATH}/text_in_wild_valid_layout.lmdb"
    if (not _lmdb_has_samples(_tiw_train)) or (not _lmdb_has_samples(_tiw_valid)):
        create_text_in_wild_train_valid(max_samples=(DEBUG_SAMPLE_LIMIT if DEBUG_MODE else None))
    else:
        print("⏭️ Text in the wild train/valid LMDB 이미 완료됨")
    
    _pa_train = f"{LOCAL_OUTPUT_PATH}/public_admin_train_layout.lmdb"
    _pa_valid = f"{LOCAL_OUTPUT_PATH}/public_admin_valid_layout.lmdb"
    if (not _lmdb_has_samples(_pa_train)) or (not _lmdb_has_samples(_pa_valid)):
        create_public_admin_train_valid(max_samples=(DEBUG_SAMPLE_LIMIT if DEBUG_MODE else None))
    else:
        print("⏭️ 공공행정문서 OCR train/valid LMDB 이미 완료됨")
    
    _pa_part = f"{LOCAL_OUTPUT_PATH}/public_admin_train_partly_layout.lmdb"
    _pa_part_alt = f"{LOCAL_OUTPUT_PATH}/public_admin_train_partly.lmdb"
    if not (_lmdb_has_samples(_pa_part) or _lmdb_has_samples(_pa_part_alt)):
        create_public_admin_train_partly(max_samples=(DEBUG_SAMPLE_LIMIT if DEBUG_MODE else None))
    else:
        print("⏭️ 공공행정문서 OCR train_partly LMDB 이미 완료됨")
    
    _ocr_train = f"{LOCAL_OUTPUT_PATH}/ocr_public_train_layout.lmdb"
    _ocr_valid = f"{LOCAL_OUTPUT_PATH}/ocr_public_valid_layout.lmdb"
    if (not _lmdb_has_samples(_ocr_train)) or (not _lmdb_has_samples(_ocr_valid)):
        create_ocr_public_train_valid(max_samples=(DEBUG_SAMPLE_LIMIT if DEBUG_MODE else None))
    else:
        print("⏭️ 023.OCR 데이터(공공) train/valid LMDB 이미 완료됨")
    
    _fl_train = f"{LOCAL_OUTPUT_PATH}/finance_logistics_train_layout.lmdb"
    _fl_valid = f"{LOCAL_OUTPUT_PATH}/finance_logistics_valid_layout.lmdb"
    if (not _lmdb_has_samples(_fl_train)) or (not _lmdb_has_samples(_fl_valid)):
        create_finance_logistics_train_valid(max_samples=(DEBUG_SAMPLE_LIMIT if DEBUG_MODE else None))
    else:
        print("⏭️ 025.OCR 데이터(금융 및 물류) train/valid LMDB 이미 완료됨")
    
    _hw_train = f"{LOCAL_OUTPUT_PATH}/handwriting_train_layout.lmdb"
    _hw_valid = f"{LOCAL_OUTPUT_PATH}/handwriting_valid_layout.lmdb"
    if (not _lmdb_has_samples(_hw_train)) or (not _lmdb_has_samples(_hw_valid)):
        create_handwriting_train_valid(max_samples=(DEBUG_SAMPLE_LIMIT if DEBUG_MODE else None))
    else:
        print("⏭️ 053.대용량 손글씨 OCR train/valid LMDB 이미 완료됨")
    
    print("\n" + "=" * 60)
    print("✅ 모든 데이터셋 train/valid LMDB 생성 완료! (전체 데이터 변환)")
    print("\n📁 생성된 LMDB 파일들:")
    for lmdb_path in lmdb_paths:
        print(f"   - {lmdb_path}")

if __name__ == '__main__':
    main() 