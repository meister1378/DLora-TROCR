#!/usr/bin/env python3
"""
LMDB 데이터 검증 스크립트
이미지와 어노테이션이 제대로 되어 있는지 확인
"""

import os
import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import lmdb
import pickle

# FAST 관련 imports
sys.path.append('.')
from dataset.fast.fast_lmdb import FAST_LMDB

def visualize_annotations(img, sample, output_path):
    """어노테이션 시각화"""
    
    # 이미지를 RGB로 변환
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img
    
    # matplotlib로 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 원본 이미지
    ax1.imshow(img_rgb)
    ax1.set_title('Original Image', fontsize=16)
    ax1.axis('off')
    
    # 어노테이션이 포함된 이미지
    ax2.imshow(img_rgb)
    ax2.set_title('Image with Annotations', fontsize=16)
    ax2.axis('off')
    
    # 어노테이션 그리기
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    color_idx = 0
    
    # GT 텍스트 영역 그리기
    if 'gt_texts' in sample and len(sample['gt_texts']) > 0:
        gt_texts = sample['gt_texts']
        print(f"     📝 시각화할 GT 텍스트: {len(gt_texts)}개")
        
        # 실제 bbox 정보가 있는지 확인
        if 'gt_instances' in sample and len(sample['gt_instances']) > 0:
            gt_instances = sample['gt_instances']
            print(f"     🏷️ GT 인스턴스 정보: {len(gt_instances)}개")
            
            for i, (text, instance) in enumerate(zip(gt_texts, gt_instances)):
                if isinstance(instance, torch.Tensor):
                    # Tensor를 numpy로 변환
                    bbox = instance.cpu().numpy()
                    if len(bbox) >= 4:  # 최소 4개 좌표
                        # bbox 좌표 추출 (x1, y1, x2, y2 형태로 가정)
                        x1, y1, x2, y2 = bbox[:4]
                        
                        # 사각형 그리기
                        rect = patches.Rectangle(
                            (x1, y1), x2-x1, y2-y1,
                            linewidth=2, edgecolor=colors[color_idx % len(colors)],
                            facecolor='none'
                        )
                        ax2.add_patch(rect)
                        
                        # 텍스트 표시
                        ax2.text(x1, y1-5, f'{i+1}: {text[:10]}...', 
                                fontsize=8, color=colors[color_idx % len(colors)],
                                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                        
                        color_idx += 1
                        print(f"       📍 텍스트 {i+1}: {text[:20]}... (좌표: {x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
                else:
                    # bbox 정보가 없으면 시뮬레이션
                    x, y = 50 + (i % 5) * 100, 50 + (i // 5) * 50
                    
                    rect = patches.Rectangle(
                        (x, y), 80, 30,
                        linewidth=2, edgecolor=colors[color_idx % len(colors)],
                        facecolor='none'
                    )
                    ax2.add_patch(rect)
                    
                    ax2.text(x, y-5, f'{i+1}: {text[:10]}...', 
                            fontsize=8, color=colors[color_idx % len(colors)],
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                    
                    color_idx += 1
        else:
            # bbox 정보가 없으면 시뮬레이션
            for i, text in enumerate(gt_texts):
                x, y = 50 + (i % 5) * 100, 50 + (i // 5) * 50
                
                rect = patches.Rectangle(
                    (x, y), 80, 30,
                    linewidth=2, edgecolor=colors[color_idx % len(colors)],
                    facecolor='none'
                )
                ax2.add_patch(rect)
                
                ax2.text(x, y-5, f'{i+1}: {text[:10]}...', 
                        fontsize=8, color=colors[color_idx % len(colors)],
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                
                color_idx += 1
    
    # GT 커널 영역 그리기 (있는 경우)
    if 'gt_kernels' in sample and len(sample['gt_kernels']) > 0:
        gt_kernels = sample['gt_kernels']
        print(f"     🎯 시각화할 GT 커널: {len(gt_kernels)}개")
        
        for i, kernel in enumerate(gt_kernels):
            # 커널 영역을 다른 색상으로 표시
            x, y = 50 + (i % 5) * 100, 200 + (i // 5) * 50
            
            rect = patches.Rectangle(
                (x, y), 80, 30,
                linewidth=2, edgecolor='yellow',
                facecolor='none', linestyle='--'
            )
            ax2.add_patch(rect)
            
            ax2.text(x, y-5, f'K{i+1}', 
                    fontsize=8, color='yellow',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.8))
    
    plt.tight_layout()
    
    # 결과 저장
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"     💾 어노테이션 시각화 저장: {output_path}")
    
    plt.show()

def simple_visualize_annotations(img, sample, output_path):
    """실제 어노테이션 시각화"""
    
    # 이미지를 RGB로 변환
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img
    
    # matplotlib로 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 원본 이미지
    ax1.imshow(img_rgb)
    ax1.set_title('Original Image', fontsize=16)
    ax1.axis('off')
    
    # 어노테이션이 포함된 이미지
    ax2.imshow(img_rgb)
    ax2.set_title('Image with Real Annotations', fontsize=16)
    ax2.axis('off')
    
    # 실제 어노테이션 그리기
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    color_idx = 0
    
    # GT 텍스트와 실제 bbox 매칭
    if 'gt_texts' in sample and 'gt_instances' in sample:
        gt_texts = sample['gt_texts']
        gt_instances = sample['gt_instances']
        
        print(f"     📊 실제 어노테이션 정보:")
        print(f"       - 텍스트 수: {len(gt_texts)}")
        print(f"       - 인스턴스 수: {len(gt_instances)}")
        
        # 텍스트와 bbox 매칭
        for i, (text, instance) in enumerate(zip(gt_texts, gt_instances)):
            if isinstance(instance, torch.Tensor):
                # Tensor를 numpy로 변환
                bbox = instance.cpu().numpy()
                
                if len(bbox) >= 4:  # 최소 4개 좌표
                    # bbox 좌표 추출 (실제 좌표)
                    x1, y1, x2, y2 = bbox[:4]
                    
                    # 실제 위치에 사각형 그리기
                    rect = patches.Rectangle(
                        (x1, y1), x2-x1, y2-y1,
                        linewidth=2, edgecolor=colors[color_idx % len(colors)],
                        facecolor='none'
                    )
                    ax2.add_patch(rect)
                    
                    # 텍스트 표시
                    ax2.text(x1, y1-5, f'{i+1}: {text[:10]}...', 
                            fontsize=8, color=colors[color_idx % len(colors)],
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                    
                    color_idx += 1
                    print(f"       📍 텍스트 {i+1}: '{text[:20]}...' (좌표: {x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
                else:
                    print(f"       ⚠️ 텍스트 {i+1}: bbox 좌표 부족 ({len(bbox)}개)")
            else:
                print(f"       ⚠️ 텍스트 {i+1}: Tensor가 아님 ({type(instance)})")
    
    plt.tight_layout()
    
    # 결과 저장
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"     💾 실제 어노테이션 시각화 저장: {output_path}")
    
    plt.close()  # 메모리 절약을 위해 닫기

def get_raw_image_and_gt(lmdb_path, index):
    """원본 이미지와 GT 데이터를 전처리 없이 가져오기"""
    try:
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        
        with env.begin(write=False) as txn:
            # 이미지 로드
            img_key = f'image-{index:09d}'.encode()
            img_data = txn.get(img_key)
            if img_data is None:
                return None, None
            
            # 바이트 데이터를 이미지로 변환 (원본 그대로)
            img_np = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            if img is None:
                return None, None
            
            # BGR -> RGB 변환
            img = img[:, :, [2, 1, 0]]
            
            # GT 데이터 로드
            gt_key = f'gt-{index:09d}'.encode()
            gt_data = txn.get(gt_key)
            if gt_data is None:
                return img, None
            
            # pickle로 직렬화된 GT 데이터 복원
            gt_info = pickle.loads(gt_data)
            
        env.close()
        return img, gt_info
        
    except Exception as e:
        print(f"     ⚠️ 원본 데이터 로드 오류: {e}")
        return None, None

def verify_lmdb_data():
    """LMDB 데이터 검증"""
    
    print("🔍 LMDB 데이터 검증 시작")
    
    # LMDB 경로들
    lmdb_paths = [
        '/mnt/nas/ocr_dataset/text_in_wild_train.lmdb',
        '/mnt/nas/ocr_dataset/public_admin_train.lmdb',
        '/mnt/nas/ocr_dataset/ocr_public_train.lmdb',
        '/mnt/nas/ocr_dataset/finance_logistics_train.lmdb',
        '/mnt/nas/ocr_dataset/handwriting_train.lmdb'
    ]
    
    for i, lmdb_path in enumerate(lmdb_paths):
        if os.path.exists(lmdb_path):
            print(f"\n📂 데이터셋 {i+1}: {os.path.basename(lmdb_path)}")
            
            try:
                # 원본 설정으로 데이터셋 생성
                dataset = FAST_LMDB(
                    lmdb_path=lmdb_path,
                    split='train',
                    is_transform=True,
                    img_size=736,  # 원본 설정 사용
                    short_size=736,  # 원본 설정 사용
                    pooling_size=9,
                    read_type='cv2'
                )
                
                print(f"   📊 데이터셋 크기: {len(dataset)}")
                
                # 원본 이미지와 GT 데이터 가져오기
                for sample_idx in range(min(5, len(dataset))):
                    print(f"   📋 샘플 {sample_idx+1}:")
                    
                    # 원본 이미지와 GT 데이터 가져오기
                    raw_img, raw_gt = get_raw_image_and_gt(lmdb_path, sample_idx)
                    
                    if raw_img is not None:
                        print(f"     🖼️ 원본 이미지: {raw_img.shape}, dtype: {raw_img.dtype}")
                        print(f"     🖼️ 범위: {raw_img.min()} ~ {raw_img.max()}")
                        
                        # 원본 이미지 저장
                        output_path = f"verify_sample_{i+1}_{sample_idx+1}_raw.jpg"
                        cv2.imwrite(output_path, cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR))
                        print(f"     💾 원본 이미지 저장: {output_path}")
                        
                        # GT 정보 출력
                        if raw_gt is not None:
                            print(f"     📝 원본 GT 정보:")
                            if 'bboxes' in raw_gt:
                                bboxes = raw_gt['bboxes']
                                print(f"       - bboxes 개수: {len(bboxes)}")
                                if len(bboxes) > 0:
                                    print(f"       - 첫 번째 bbox: {bboxes[0]}")
                            
                            if 'words' in raw_gt:
                                words = raw_gt['words']
                                print(f"       - words 개수: {len(words)}")
                                if len(words) > 0:
                                    print(f"       - 첫 번째 word: '{words[0]}'")
                            
                            # 원본 bbox 그리기
                            if 'bboxes' in raw_gt and 'words' in raw_gt:
                                bboxes = raw_gt['bboxes']
                                words = raw_gt['words']
                                
                                if len(bboxes) > 0:
                                    img_with_bbox = raw_img.copy()
                                    
                                    for j, (bbox, word) in enumerate(zip(bboxes, words)):
                                        if word != '###':  # 무시할 텍스트가 아닌 경우
                                            # bbox 좌표를 실제 픽셀 좌표로 변환
                                            h, w = raw_img.shape[:2]
                                            bbox_pixels = np.array(bbox) * [w, h, w, h, w, h, w, h]
                                            bbox_pixels = bbox_pixels.reshape(-1, 2).astype(np.int32)
                                            
                                            # bbox 그리기
                                            cv2.polylines(img_with_bbox, [bbox_pixels], True, (0, 255, 0), 2)
                                            
                                            # 텍스트 표시
                                            center = np.mean(bbox_pixels, axis=0).astype(int)
                                            cv2.putText(img_with_bbox, f"{j+1}:{word[:10]}", 
                                                      (center[0], center[1]), 
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                                    
                                    # bbox가 그려진 이미지 저장
                                    bbox_output_path = f"verify_sample_{i+1}_{sample_idx+1}_raw_with_bbox.jpg"
                                    cv2.imwrite(bbox_output_path, cv2.cvtColor(img_with_bbox, cv2.COLOR_RGB2BGR))
                                    print(f"     💾 bbox 이미지 저장: {bbox_output_path}")
                                else:
                                    print(f"     ⚠️ bbox 정보가 없습니다")
                        else:
                            print(f"     ⚠️ GT 정보가 없습니다")
                    else:
                        print(f"     ❌ 원본 이미지 로드 실패")
                        
                        print()  # 빈 줄 추가
                    
                    print()
                
                # 데이터셋 통계
                print(f"   📊 데이터셋 통계:")
                print(f"     - 총 샘플 수: {len(dataset)}")
                
                # 랜덤 샘플링으로 추가 검증
                import random
                random_indices = random.sample(range(len(dataset)), min(10, len(dataset)))
                
                valid_samples = 0
                total_annotations = 0
                
                for idx in random_indices:
                    try:
                        sample = dataset[idx]
                        if 'imgs' in sample and 'gt_texts' in sample:
                            valid_samples += 1
                            total_annotations += len(sample['gt_texts'])
                    except Exception as e:
                        print(f"     ⚠️ 샘플 {idx} 오류: {e}")
                
                print(f"     - 유효한 샘플: {valid_samples}/{len(random_indices)}")
                print(f"     - 평균 어노테이션 수: {total_annotations/max(1, valid_samples):.1f}")
                
            except Exception as e:
                print(f"   ❌ 오류: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"   ❌ 파일 없음: {lmdb_path}")

def test_with_original_config():
    """원본 설정으로 detection 테스트"""
    
    print("\n🔧 원본 설정으로 detection 테스트")
    
    try:
        from mmcv import Config
        from models import build_model
        
        # 원본 설정 파일 사용
        config_path = "config/fast/ic15/fast_sample_finetune.py"
        checkpoint_path = "outputs/validation_test/checkpoint_latest.pth"
        
        if not os.path.exists(config_path):
            print(f"❌ 설정 파일 없음: {config_path}")
            return
        
        if not os.path.exists(checkpoint_path):
            print(f"❌ 체크포인트 없음: {checkpoint_path}")
            return
        
        # 설정 로드
        cfg = Config.fromfile(config_path)
        print(f"📊 원본 설정:")
        print(f"   - 이미지 크기: {cfg.data.train.img_size}")
        print(f"   - test_cfg: {cfg.test_cfg}")
        
        # 모델 로드
        model = build_model(cfg.model)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 키 정리
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value
        
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        
        # 테스트 이미지로 detection
        test_image_path = "../WF_2000_5320060_0005_0005.jpg"
        
        if os.path.exists(test_image_path):
            from dataset.utils import get_img, scale_aligned_short
            import torchvision.transforms as transforms
            from PIL import Image
            
            img = get_img(test_image_path, read_type='cv2')
            original_img = img.copy()
            
            # 원본 설정에 맞춰 크기 조정
            img = scale_aligned_short(img, short_size=736)  # 원본 설정
            
            # PIL 변환 및 정규화
            img_pil = Image.fromarray(img)
            img_pil = img_pil.convert('RGB')
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(img_pil).unsqueeze(0).to(device)
            
            # 메타데이터
            img_meta = {
                'filename': [os.path.basename(test_image_path)],
                'org_img_size': [original_img.shape[:2]],
                'img_size': [img.shape[:2]]
            }
            
            # 추론
            with torch.no_grad():
                data = {
                    'imgs': img_tensor,
                    'img_metas': img_meta,
                    'cfg': cfg
                }
                outputs = model(**data)
            
            # 결과 분석
            if 'results' in outputs and len(outputs['results']) > 0:
                results = outputs['results'][0]
                bboxes = results.get('bboxes', [])
                scores = results.get('scores', [])
                
                print(f"🎯 원본 설정 검출 결과: {len(bboxes)}개")
                
                # 원본 임계값 적용
                high_score_count = sum(1 for score in scores if score > 0.88)  # 원본 임계값
                print(f"📈 높은 신뢰도 (0.88+): {high_score_count}개")
                
                if len(scores) > 0:
                    max_score = max(scores)
                    avg_score = sum(scores) / len(scores)
                    print(f"📊 최고 스코어: {max_score:.3f}")
                    print(f"📊 평균 스코어: {avg_score:.3f}")
                
                print(f"✅ 원본 설정 detection 완료")
            else:
                print(f"❌ 원본 설정 검출 결과 없음")
        
    except Exception as e:
        print(f"❌ 원본 설정 테스트 오류: {e}")
        import traceback
        traceback.print_exc()

def main():
    """메인 함수"""
    print("🚀 LMDB 데이터 검증 시작")
    
    # LMDB 데이터 검증
    verify_lmdb_data()
    
    # 원본 설정으로 테스트
    test_with_original_config()
    
    print("\n🎉 LMDB 데이터 검증 완료!")

if __name__ == '__main__':
    main() 