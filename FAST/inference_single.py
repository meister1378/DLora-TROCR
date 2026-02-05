#!/usr/bin/env python3
"""
FAST 모델을 사용한 단일 이미지 텍스트 검출 추론 스크립트
"""

import torch
import argparse
import os
import sys
import cv2
import numpy as np
from PIL import Image
import mmcv
from mmcv import Config

# FAST 모델 관련 imports
sys.path.append('.')
from dataset import build_data_loader
from models import build_model
from models.utils import fuse_module, rep_model_convert
from utils import ResultFormat, AverageMeter
from dataset.utils import get_img, scale_aligned_short
import torchvision.transforms as transforms
import json

def preprocess_image(image_path, short_size=640):
    """
    단일 이미지를 FAST 모델 입력 형태로 전처리
    """
    # 이미지 로드
    img = get_img(image_path, read_type='cv2')
    original_img = img.copy()
    
    # 크기 조정
    img = scale_aligned_short(img, short_size=short_size)
    
    # PIL로 변환 후 정규화
    img_pil = Image.fromarray(img)
    img_pil = img_pil.convert('RGB')
    
    # 텐서 변환 및 정규화
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img_pil).unsqueeze(0)  # 배치 차원 추가
    
    # 메타데이터 준비
    img_meta = {
        'filename': [os.path.basename(image_path)],
        'org_img_size': [original_img.shape[:2]],  # [H, W]
        'img_size': [img.shape[:2]]  # [H, W]
    }
    
    return img_tensor, img_meta, original_img

def load_model_and_checkpoint(config_path, checkpoint_path):
    """
    FAST 모델과 체크포인트를 로드
    """
    # 설정 로드
    cfg = Config.fromfile(config_path)
    
    # 모델 생성
    model = build_model(cfg.model)
    
    # GPU 사용 가능하면 GPU로
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 체크포인트 로드
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # state_dict 추출 (ema 사용 시 ema, 아니면 state_dict)
        if 'ema' in checkpoint:
            state_dict = checkpoint['ema']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 키에서 'module.' 제거
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value
        
        model.load_state_dict(new_state_dict)
        print("✅ Checkpoint loaded successfully!")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # 모델 최적화
    model = rep_model_convert(model)
    model = fuse_module(model)
    model.eval()
    
    return model, cfg, device

def inference_single_image(model, img_tensor, img_meta, cfg, device):
    """
    단일 이미지에 대해 추론 수행
    """
    # 데이터를 GPU로
    img_tensor = img_tensor.to(device)
    
    # 추론
    with torch.no_grad():
        data = {
            'imgs': img_tensor,
            'img_metas': img_meta,
            'cfg': cfg
        }
        outputs = model(**data)
    
    return outputs

def visualize_results(img, results, output_path, min_score=0.5):
    """
    검출 결과를 시각화
    """
    img_vis = img.copy()
    
    for result in results['results']:
        bboxes = result['bboxes']
        scores = result['scores']
        
        for bbox, score in zip(bboxes, scores):
            if score > min_score:
                # bbox는 [x1, y1, x2, y2, x3, y3, x4, y4] 형태
                bbox = np.array(bbox).reshape(-1, 2).astype(np.int32)
                
                # 폴리곤 그리기
                cv2.polylines(img_vis, [bbox], True, (0, 255, 0), 2)
                
                # 스코어 표시
                cv2.putText(img_vis, f'{score:.3f}', 
                           (int(bbox[0][0]), int(bbox[0][1]-5)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 결과 저장
    cv2.imwrite(output_path, img_vis)
    print(f"✅ 시각화 결과 저장: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='FAST 단일 이미지 추론')
    parser.add_argument('--config', default='config/fast/ic15/fast_sample_finetune_test.py',
                       help='설정 파일 경로')
    parser.add_argument('--checkpoint', default='checkpoint_7ep.pth',
                       help='체크포인트 파일 경로')
    parser.add_argument('--image', default='5350034-2011-0001-0019.jpg',
                       help='입력 이미지 경로')
    parser.add_argument('--output', default='output_detection.jpg',
                       help='출력 이미지 경로')
    parser.add_argument('--min_score', type=float, default=0.5,
                       help='최소 검출 스코어')
    parser.add_argument('--short_size', type=int, default=640,
                       help='이미지 리사이즈 크기')
    
    args = parser.parse_args()
    
    print("=== FAST 텍스트 검출 추론 ===")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Image: {args.image}")
    print(f"Output: {args.output}")
    
    # 모델 로드
    print("\n1. 모델 로딩...")
    model, cfg, device = load_model_and_checkpoint(args.config, args.checkpoint)
    
    # 이미지 전처리
    print("\n2. 이미지 전처리...")
    img_tensor, img_meta, original_img = preprocess_image(args.image, args.short_size)
    print(f"Original image shape: {original_img.shape}")
    print(f"Processed image shape: {img_tensor.shape}")
    
    # 추론
    print("\n3. 추론 수행...")
    results = inference_single_image(model, img_tensor, img_meta, cfg, device)
    
    # 결과 출력
    print("\n4. 결과 처리...")
    num_detections = len(results['results'][0]['bboxes'])
    print(f"검출된 텍스트 영역 수: {num_detections}")
    
    # JSON으로 결과 저장
    results_json = {
        'image': args.image,
        'detections': []
    }
    
    for bbox, score in zip(results['results'][0]['bboxes'], results['results'][0]['scores']):
        if score > args.min_score:
            results_json['detections'].append({
                'bbox': bbox.tolist() if hasattr(bbox, 'tolist') else bbox,
                'score': float(score)
            })
    
    with open('detection_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 검출 결과 JSON 저장: detection_results.json")
    
    # 시각화
    print("\n5. 결과 시각화...")
    visualize_results(original_img, results, args.output, args.min_score)
    
    print("\n🎉 추론 완료!")
    print(f"총 {len(results_json['detections'])}개의 텍스트 영역을 검출했습니다.")

if __name__ == '__main__':
    main() 