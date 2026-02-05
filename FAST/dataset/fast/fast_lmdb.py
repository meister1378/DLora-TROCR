#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FAST LMDB Dataset
LMDB 형태로 저장된 데이터를 로드하는 FAST 데이터셋 클래스
"""

import os
import sys
import random
import pickle
import numpy as np
import cv2
import mmcv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from torch.utils import data
from tqdm import tqdm

try:
    import lmdb
except ImportError:
    print("❌ LMDB 패키지가 설치되지 않았습니다. 'pip install lmdb' 로 설치해주세요.")
    sys.exit(1)

from dataset.utils import shrink
from dataset.utils import get_vocabulary
from dataset.utils import random_crop_padding_v2 as random_crop_padding
from dataset.utils import random_scale, random_horizontal_flip, random_rotate
from dataset.utils import scale_aligned_short


class FAST_LMDB(data.Dataset):
    """
    LMDB 형태로 저장된 데이터를 위한 FAST 데이터셋 클래스
    
    LMDB 구조:
    - 이미지: key='image-{:09d}'.format(idx), value=image_bytes
    - GT: key='gt-{:09d}'.format(idx), value=pickle.dumps(annotations)
    - 길이: key='num-samples', value=str(total_samples)
    """
    
    def __init__(self, lmdb_path, split='train', is_transform=False, img_size=None, 
                 short_size=736, pooling_size=9, with_rec=False, read_type='cv2',
                 repeat_times=1, report_speed=False):
        """
        Args:
            lmdb_path (str): LMDB 데이터베이스 경로
            split (str): 'train' 또는 'test'
            is_transform (bool): 데이터 증강 활성화 여부
            img_size (tuple/int): 입력 이미지 크기
            short_size (int): 최소 변의 크기
            pooling_size (int): 풀링 크기
            with_rec (bool): 인식 태스크 포함 여부
            read_type (str): 이미지 읽기 방식 ('cv2' 또는 'pil')
            repeat_times (int): 데이터 반복 배수
            report_speed (bool): 속도 측정 모드
        """
        self.lmdb_path = lmdb_path
        self.split = split
        self.is_transform = is_transform
        self.img_size = img_size if (img_size is None or isinstance(img_size, tuple)) else (img_size, img_size)
        self.pooling_size = pooling_size
        self.short_size = short_size
        self.with_rec = with_rec
        self.read_type = read_type
        self.repeat_times = repeat_times

        # 풀링 레이어 초기화
        self.pad = nn.ZeroPad2d(padding=(pooling_size - 1) // 2)
        self.pooling = nn.MaxPool2d(kernel_size=pooling_size, stride=1)
        self.overlap_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # LMDB 환경 열기
        print(f"🗂️ LMDB 데이터베이스 열기: {lmdb_path}")
        if not os.path.exists(lmdb_path):
            raise FileNotFoundError(f"LMDB 데이터베이스를 찾을 수 없습니다: {lmdb_path}")
        
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        
        # 데이터 크기 확인
        with self.env.begin(write=False) as txn:
            num_samples_key = 'num-samples'.encode()
            self.length = int(txn.get(num_samples_key).decode())
        
        # 반복 배수 적용
        self.total_length = self.length * repeat_times
        
        print(f"📊 LMDB 데이터셋 정보:")
        print(f"   - 원본 샘플 수: {self.length}")
        print(f"   - 반복 배수: {repeat_times}")
        print(f"   - 총 길이: {self.total_length}")
        
        # 어휘 사전 로드
        self.voc, self.char2id, self.id2char = get_vocabulary('LOWERCASE')
        self.max_word_num = 200
        self.max_word_len = 32

    def __len__(self):
        return self.total_length

    def __del__(self):
        """소멸자에서 LMDB 환경 닫기"""
        if hasattr(self, 'env'):
            self.env.close()

    def get_image_and_gt(self, index):
        """LMDB에서 이미지와 GT 데이터를 로드"""
        # 반복 인덱스 처리
        real_index = index % self.length
        
        with self.env.begin(write=False) as txn:
            # 이미지 로드
            img_key = f'image-{real_index:09d}'.encode()
            img_data = txn.get(img_key)
            if img_data is None:
                raise KeyError(f"이미지 키를 찾을 수 없습니다: {img_key}")
            
            # 바이트 데이터를 이미지로 변환
            img_np = np.frombuffer(img_data, dtype=np.uint8)
            if self.read_type == 'cv2':
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"⚠️ 이미지 디코딩 실패: index {real_index}, 다른 이미지로 대체")
                    # 다른 유효한 인덱스로 재시도
                    return self.get_image_and_gt((index + 1) % len(self))
                img = img[:, :, [2, 1, 0]]  # BGR -> RGB
            else:  # PIL
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"⚠️ 이미지 디코딩 실패: index {real_index}, 다른 이미지로 대체")
                    # 다른 유효한 인덱스로 재시도
                    return self.get_image_and_gt((index + 1) % len(self))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # GT 데이터 로드
            gt_key = f'gt-{real_index:09d}'.encode()
            gt_data = txn.get(gt_key)
            if gt_data is None:
                raise KeyError(f"GT 키를 찾을 수 없습니다: {gt_key}")
            
            # pickle로 직렬화된 GT 데이터 복원
            gt_info = pickle.loads(gt_data)
            
        return img, gt_info

    def min_pooling(self, input_tensor):
        """오버랩 영역을 처리하는 Min Pooling"""
        input_tensor = torch.tensor(input_tensor, dtype=torch.float)
        temp = input_tensor.sum(dim=0).to(torch.uint8)
        overlap = (temp > 1).to(torch.float32).unsqueeze(0).unsqueeze(0)
        overlap = self.overlap_pool(overlap).squeeze(0).squeeze(0)

        B = input_tensor.size(0)
        h_sum = input_tensor.sum(dim=2) > 0
        
        h_sum_ = h_sum.long() * torch.arange(h_sum.shape[1], 0, -1)
        h_min = torch.argmax(h_sum_, 1, keepdim=True)
        h_sum_ = h_sum.long() * torch.arange(1, h_sum.shape[1] + 1)
        h_max = torch.argmax(h_sum_, 1, keepdim=True)

        w_sum = input_tensor.sum(dim=1) > 0
        w_sum_ = w_sum.long() * torch.arange(w_sum.shape[1], 0, -1)
        w_min = torch.argmax(w_sum_, 1, keepdim=True)
        w_sum_ = w_sum.long() * torch.arange(1, w_sum.shape[1] + 1)
        w_max = torch.argmax(w_sum_, 1, keepdim=True)

        for i in range(B):
            region = input_tensor[i:i + 1, h_min[i]:h_max[i] + 1, w_min[i]:w_max[i] + 1]
            region = self.pad(region)
            region = -self.pooling(-region)
            input_tensor[i:i + 1, h_min[i]:h_max[i] + 1, w_min[i]:w_max[i] + 1] = region

        x = input_tensor.sum(dim=0).to(torch.uint8)
        x[overlap > 0] = 0  # overlapping regions
        return x.numpy()

    def prepare_train_data(self, index):
        """훈련 데이터 준비"""
        img, gt_info = self.get_image_and_gt(index)
        
        # GT 정보 추출
        bboxes = np.array(gt_info['bboxes'])  # 정규화된 좌표
        words = gt_info['words']
        
        if bboxes.shape[0] > self.max_word_num:
            bboxes = bboxes[:self.max_word_num]
            words = words[:self.max_word_num]

        # 데이터 증강
        if self.is_transform:
            img = random_scale(img, self.short_size, scales=[0.5, 2.0], aspects=[0.9, 1.1])

        # GT 마스크 생성
        gt_instance = np.zeros(img.shape[0:2], dtype='uint8')
        training_mask = np.ones(img.shape[0:2], dtype='uint8')
        
        if bboxes.shape[0] > 0:
            # 정규화된 좌표를 실제 픽셀 좌표로 변환
            bboxes = np.reshape(bboxes * ([img.shape[1], img.shape[0]] * 4),
                                (bboxes.shape[0], -1, 2)).astype('int32')
            
            for i in range(bboxes.shape[0]):
                if words[i] == '###':
                    cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)
                else:
                    cv2.drawContours(gt_instance, [bboxes[i]], -1, i + 1, -1)

        # 커널 생성
        gt_kernels = []
        for i in range(len(bboxes)):
            gt_kernel = np.zeros(img.shape[0:2], dtype='uint8')
            if words[i] != '###':
                cv2.drawContours(gt_kernel, [bboxes[i]], -1, 1, -1)
                gt_kernels.append(gt_kernel)
            else:
                if len(gt_kernels) == 0:
                    gt_kernels.append(gt_kernel)
        
        if len(gt_kernels) > 0:
            gt_kernels = np.array(gt_kernels)
            gt_kernel = self.min_pooling(gt_kernels)
        else:
            gt_kernel = np.zeros(img.shape[0:2], dtype='uint8')

        # 커널 수축
        shrink_kernel_scale = 0.1
        gt_kernel_shrinked = np.zeros(img.shape[0:2], dtype='uint8')
        kernel_bboxes = shrink(bboxes, shrink_kernel_scale)
        for i in range(bboxes.shape[0]):
            if words[i] != '###':
                cv2.drawContours(gt_kernel_shrinked, [kernel_bboxes[i]], -1, 1, -1)
        gt_kernel = np.maximum(gt_kernel, gt_kernel_shrinked)

        # 기하학적 변환
        if self.is_transform:
            imgs = [img, gt_instance, training_mask, gt_kernel]

            if not self.with_rec:
                imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs, random_angle=30)
            imgs = random_crop_padding(imgs, self.img_size)
            img, gt_instance, training_mask, gt_kernel = imgs[0], imgs[1], imgs[2], imgs[3]

        # 텍스트 마스크
        gt_text = gt_instance.copy()
        gt_text[gt_text > 0] = 1

        # 이미지 전처리
        img = Image.fromarray(img)
        img = img.convert('RGB')
        if self.is_transform:
            if random.random() < 0.5:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
            img = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)(img)

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        gt_text = torch.from_numpy(gt_text).long()
        gt_kernel = torch.from_numpy(gt_kernel).long()
        training_mask = torch.from_numpy(training_mask).long()
        gt_instance = torch.from_numpy(gt_instance).long()

        data = dict(
            imgs=img,
            gt_texts=gt_text,
            gt_kernels=gt_kernel,
            training_masks=training_mask,
            gt_instances=gt_instance,
        )

        return data

    def prepare_test_data(self, index):
        """테스트 데이터 준비"""
        img, gt_info = self.get_image_and_gt(index)
        filename = gt_info.get('filename', f'image_{index:06d}')

        img_meta = dict(
            org_img_size=np.array(img.shape[:2])
        )

        img = scale_aligned_short(img, self.short_size)
        img_meta.update(dict(
            img_size=np.array(img.shape[:2]),
            filename=filename
        ))

        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        data = dict(
            imgs=img,
            img_metas=img_meta
        )

        return data

    def __getitem__(self, index):
        """데이터 로드"""
        if self.split == 'train':
            return self.prepare_train_data(index)
        elif self.split == 'test':
            return self.prepare_test_data(index)
        else:
            raise ValueError(f"지원하지 않는 split: {self.split}")


def create_lmdb_dataset(image_dir, gt_dir, output_path, annotation_parser='ic15'):
    """
    일반 이미지 데이터셋을 LMDB 형태로 변환
    
    Args:
        image_dir (str): 이미지 디렉토리 경로
        gt_dir (str): GT 파일 디렉토리 경로  
        output_path (str): 출력 LMDB 경로
        annotation_parser (str): GT 파싱 방식 ('ic15', 'ic17mlt', 'text_in_wild', 'ocr_public', 'handwriting_ocr', 'public_admin_ocr')
    """
    import json
    
    print(f"🔄 LMDB 데이터셋 생성 중...")
    print(f"   - 이미지 디렉토리: {image_dir}")
    print(f"   - GT 디렉토리: {gt_dir}")
    print(f"   - 출력 경로: {output_path}")
    print(f"   - 파서 타입: {annotation_parser}")
    
    # LMDB 환경 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    env = lmdb.open(output_path, map_size=1099511627776)  # 1TB
    
    if annotation_parser == 'text_in_wild':
        # Text in the wild 데이터셋 처리 (하나의 JSON 파일에 모든 정보)
        json_files = [f for f in os.listdir(gt_dir) if f.endswith('.json')]
        if not json_files:
            raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {gt_dir}")
        
        json_path = os.path.join(gt_dir, json_files[0])
        print(f"📄 JSON 파일 로드 중: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        images_info = {img['id']: img for img in data['images']}
        
        # 이미지별로 어노테이션 그룹화
        image_annotations = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)
        
        print(f"📊 총 {len(images_info)}개 이미지 발견")
        
        with env.begin(write=True) as txn:
            idx = 0
            for img_id, img_info in tqdm(images_info.items(), desc="Text in the wild 처리 중", total=len(images_info)):
                
                # 이미지 로드
                img_path = os.path.join(image_dir, img_info['file_name'])
                if not os.path.exists(img_path):
                    print(f"⚠️ 이미지 파일을 찾을 수 없습니다: {img_path}")
                    continue
                
                with open(img_path, 'rb') as f:
                    img_data = f.read()
                
                # 어노테이션 처리
                bboxes = []
                words = []
                if img_id in image_annotations:
                    for ann in image_annotations[img_id]:
                        # bbox: [x, y, width, height] -> [x1, y1, x2, y2, x3, y3, x4, y4]
                        x, y, w, h = ann['bbox']
                        x1, y1, x2, y2 = x, y, x + w, y + h
                        
                        # 정규화
                        img_w, img_h = img_info['width'], img_info['height']
                        normalized_coords = [x1/img_w, y1/img_h, x2/img_w, y1/img_h, 
                                           x2/img_w, y2/img_h, x1/img_w, y2/img_h]
                        
                        bboxes.append(normalized_coords)
                        words.append(ann['text'])
                
                gt_info = {
                    'bboxes': bboxes,
                    'words': words,
                    'filename': img_info['file_name']
                }
                
                # LMDB에 저장
                img_key = f'image-{idx:09d}'.encode()
                gt_key = f'gt-{idx:09d}'.encode()
                
                txn.put(img_key, img_data)
                txn.put(gt_key, pickle.dumps(gt_info))
                idx += 1
            
            # 총 샘플 수 저장
            txn.put('num-samples'.encode(), str(idx).encode())
    
    elif annotation_parser in ['ocr_public', 'handwriting_ocr']:
        # 023.OCR 데이터(공공), 053.대용량 손글씨 OCR 데이터 처리
        # 각 이미지마다 개별 JSON 파일
        img_names = []
        for ext in ['.jpg', '.png', '.JPG', '.PNG', '.jpeg', '.JPEG']:
            img_names.extend([f for f in os.listdir(image_dir) if f.endswith(ext)])
        
        print(f"📊 총 {len(img_names)}개 이미지 발견")
        
        with env.begin(write=True) as txn:
            idx = 0
            for img_name in tqdm(img_names, desc=f"{annotation_parser} 처리 중", total=len(img_names)):
                
                # 이미지 로드
                img_path = os.path.join(image_dir, img_name)
                with open(img_path, 'rb') as f:
                    img_data = f.read()
                
                # JSON 파일 로드
                json_name = img_name.split('.')[0] + '.json'
                json_path = os.path.join(gt_dir, json_name)
                
                bboxes = []
                words = []
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 이미지 크기 정보
                    img_w = data['Images']['width']
                    img_h = data['Images']['height']
                    
                    # 바운딩 박스 처리
                    bbox_key = 'Bbox' if annotation_parser == 'ocr_public' else 'bbox'
                    for bbox_info in data[bbox_key]:
                        # x: [x1, x1, x2, x2], y: [y1, y2, y1, y2] -> [x1, y1, x2, y1, x2, y2, x1, y2]
                        x_coords = bbox_info['x']
                        y_coords = bbox_info['y']
                        
                        # 정규화
                        normalized_coords = [
                            x_coords[0]/img_w, y_coords[0]/img_h,  # x1, y1
                            x_coords[2]/img_w, y_coords[0]/img_h,  # x2, y1
                            x_coords[2]/img_w, y_coords[1]/img_h,  # x2, y2
                            x_coords[0]/img_w, y_coords[1]/img_h   # x1, y2
                        ]
                        
                        bboxes.append(normalized_coords)
                        words.append(bbox_info['data'])
                else:
                    print(f"⚠️ JSON 파일을 찾을 수 없습니다: {json_path}")
                
                gt_info = {
                    'bboxes': bboxes,
                    'words': words,
                    'filename': img_name
                }
                
                # LMDB에 저장
                img_key = f'image-{idx:09d}'.encode()
                gt_key = f'gt-{idx:09d}'.encode()
                
                txn.put(img_key, img_data)
                txn.put(gt_key, pickle.dumps(gt_info))
                idx += 1
            
            # 총 샘플 수 저장
            txn.put('num-samples'.encode(), str(idx).encode())
    
    elif annotation_parser == 'public_admin_ocr':
        # 공공행정문서 OCR 데이터 처리
        img_names = []
        for ext in ['.jpg', '.png', '.JPG', '.PNG', '.jpeg', '.JPEG']:
            img_names.extend([f for f in os.listdir(image_dir) if f.endswith(ext)])
        
        print(f"📊 총 {len(img_names)}개 이미지 발견")
        
        with env.begin(write=True) as txn:
            idx = 0
            for img_name in tqdm(img_names, desc="공공행정문서 OCR 처리 중", total=len(img_names)):
                
                # 이미지 로드
                img_path = os.path.join(image_dir, img_name)
                with open(img_path, 'rb') as f:
                    img_data = f.read()
                
                # JSON 파일 로드
                json_name = img_name.split('.')[0] + '.json'
                json_path = os.path.join(gt_dir, json_name)
                
                bboxes = []
                words = []
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 이미지 크기 정보
                    img_w = data['images'][0]['image.width']
                    img_h = data['images'][0]['image.height']
                    
                    # 바운딩 박스 처리
                    for ann in data['annotations']:
                        # annotation.bbox: [x, y, width, height] -> [x1, y1, x2, y2, x3, y3, x4, y4]
                        x, y, w, h = ann['annotation.bbox']
                        x1, y1, x2, y2 = x, y, x + w, y + h
                        
                        # 정규화
                        normalized_coords = [x1/img_w, y1/img_h, x2/img_w, y1/img_h, 
                                           x2/img_w, y2/img_h, x1/img_w, y2/img_h]
                        
                        bboxes.append(normalized_coords)
                        words.append(ann['annotation.text'])
                else:
                    print(f"⚠️ JSON 파일을 찾을 수 없습니다: {json_path}")
                
                gt_info = {
                    'bboxes': bboxes,
                    'words': words,
                    'filename': img_name
                }
                
                # LMDB에 저장
                img_key = f'image-{idx:09d}'.encode()
                gt_key = f'gt-{idx:09d}'.encode()
                
                txn.put(img_key, img_data)
                txn.put(gt_key, pickle.dumps(gt_info))
                idx += 1
            
            # 총 샘플 수 저장
            txn.put('num-samples'.encode(), str(idx).encode())
    
    else:
        # 기존 IC15, IC17MLT 등의 txt 파일 처리
        img_names = []
        for ext in ['.jpg', '.png', '.JPG', '.PNG', '.jpeg', '.JPEG']:
            img_names.extend([f for f in os.listdir(image_dir) if f.endswith(ext)])
        
        print(f"📊 총 {len(img_names)}개 이미지 발견")
        
        with env.begin(write=True) as txn:
            for idx, img_name in enumerate(tqdm(img_names, desc=f"{annotation_parser} 처리 중", total=len(img_names))):
                
                # 이미지 로드 및 인코딩
                img_path = os.path.join(image_dir, img_name)
                with open(img_path, 'rb') as f:
                    img_data = f.read()
                
                # GT 파일 파싱
                gt_name = img_name.split('.')[0] + '.txt'
                if annotation_parser == 'ic17mlt':
                    gt_name = 'gt_' + gt_name
                gt_path = os.path.join(gt_dir, gt_name)
                
                # GT 데이터 파싱 (기본적인 IC15 형식)
                if os.path.exists(gt_path):
                    with open(gt_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    bboxes = []
                    words = []
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split(',')
                        if len(parts) >= 8:
                            # IC15 형식: x1,y1,x2,y2,x3,y3,x4,y4,text
                            coords = [int(x) for x in parts[:8]]
                            text = ','.join(parts[8:]) if len(parts) > 8 else '???'
                            
                            # 정규화 (LMDB 저장을 위해 이미지 크기 필요)
                            img_cv = cv2.imread(img_path)
                            h, w = img_cv.shape[:2]
                            normalized_coords = [c / w if i % 2 == 0 else c / h for i, c in enumerate(coords)]
                            
                            bboxes.append(normalized_coords)
                            words.append(text)
                    
                    gt_info = {
                        'bboxes': bboxes,
                        'words': words,
                        'filename': img_name
                    }
                else:
                    print(f"⚠️ GT 파일을 찾을 수 없습니다: {gt_path}")
                    gt_info = {'bboxes': [], 'words': [], 'filename': img_name}
                
                # LMDB에 저장
                img_key = f'image-{idx:09d}'.encode()
                gt_key = f'gt-{idx:09d}'.encode()
                
                txn.put(img_key, img_data)
                txn.put(gt_key, pickle.dumps(gt_info))
            
            # 총 샘플 수 저장
            txn.put('num-samples'.encode(), str(len(img_names)).encode())
    
    env.close()
    print(f"✅ LMDB 데이터셋 생성 완료: {output_path}")


if __name__ == '__main__':
    # 사용 예시
    print("🧪 FAST LMDB 데이터셋 테스트")
    
    # 예시: 기존 데이터를 LMDB로 변환
    # create_lmdb_dataset(
    #     image_dir='/path/to/images',
    #     gt_dir='/path/to/gt',
    #     output_path='/path/to/output.lmdb'
    # )
    
    # 예시: LMDB 데이터셋 로드
    # dataset = FAST_LMDB(
    #     lmdb_path='/path/to/dataset.lmdb',
    #     split='train',
    #     is_transform=True
    # )
    # print(f"데이터셋 크기: {len(dataset)}") 