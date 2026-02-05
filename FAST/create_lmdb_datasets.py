#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LMDB 데이터셋 생성 스크립트
각 데이터셋별로 LMDB를 생성합니다.
"""

import os
import sys
from dataset.fast.fast_lmdb import create_lmdb_dataset

def create_text_in_wild_lmdb():
    """Text in the wild 데이터셋 LMDB 생성"""
    print("🔄 Text in the wild 데이터셋 LMDB 생성 시작")
    
    # 실제 NAS 경로 설정
    base_path = "/mnt/nas/ocr_dataset/13.한국어글자체/04. Text in the wild_230209_add"
    
    # 이미지들이 여러 하위 디렉토리에 분산되어 있으므로 통합 처리 필요
    image_dirs = [
        f"{base_path}/01_textinthewild_book_images_new/01_textinthewild_book_images_new/book",
        f"{base_path}/01_textinthewild_goods_images_new",  # 경로 확인 필요
        f"{base_path}/01_textinthewild_signboard_images_new",  # 경로 확인 필요
        f"{base_path}/01_textinthewild_traffic_sign_images_new"  # 경로 확인 필요
    ]
    
    gt_file = f"{base_path}/textinthewild_data_info.json"
    output_path = "/mnt/nas/ocr_dataset/text_in_wild.lmdb"
    
    try:
        # Text in the wild는 하나의 JSON 파일에 모든 정보가 있으므로
        # 첫 번째 이미지 디렉토리를 대표로 사용하고, GT는 JSON 파일 디렉토리
        create_lmdb_dataset(
            image_dir=image_dirs[0],  # 일단 book 디렉토리만 사용
            gt_dir=base_path,  # JSON 파일이 있는 디렉토리
            output_path=output_path,
            annotation_parser='text_in_wild'
        )
        print("✅ Text in the wild LMDB 생성 완료")
    except Exception as e:
        print(f"❌ Text in the wild LMDB 생성 실패: {e}")

def create_ocr_public_lmdb():
    """023.OCR 데이터(공공) LMDB 생성"""
    print("🔄 023.OCR 데이터(공공) LMDB 생성 시작")
    
    base_path = "/mnt/nas/ocr_dataset/023.OCR 데이터(공공)/01-1.정식개방데이터"
    
    # Training과 Validation 데이터 모두 처리
    datasets = [
        {
            'name': 'ocr_public_train',
            'image_dir': f"{base_path}/Training/01.원천데이터",
            'gt_dir': f"{base_path}/Training/02.라벨링데이터",
            'output_path': "/mnt/nas/ocr_dataset/ocr_public_train.lmdb"
        },
        {
            'name': 'ocr_public_val',
            'image_dir': f"{base_path}/Validation/01.원천데이터",
            'gt_dir': f"{base_path}/Validation/02.라벨링데이터",
            'output_path': "/mnt/nas/ocr_dataset/ocr_public_val.lmdb"
        }
    ]
    
    for dataset in datasets:
        try:
            print(f"📂 {dataset['name']} 처리 중...")
            
            # 하위 디렉토리들을 순회하여 처리해야 할 수 있음
            if os.path.exists(dataset['image_dir']) and os.path.exists(dataset['gt_dir']):
                create_lmdb_dataset(
                    image_dir=dataset['image_dir'],
                    gt_dir=dataset['gt_dir'],
                    output_path=dataset['output_path'],
                    annotation_parser='ocr_public'
                )
                print(f"✅ {dataset['name']} LMDB 생성 완료")
            else:
                print(f"⚠️ {dataset['name']} 경로를 찾을 수 없습니다")
                print(f"   이미지: {dataset['image_dir']}")
                print(f"   GT: {dataset['gt_dir']}")
        except Exception as e:
            print(f"❌ {dataset['name']} LMDB 생성 실패: {e}")

def create_finance_logistics_lmdb():
    """025.OCR 데이터(금융 및 물류) LMDB 생성"""
    print("🔄 025.OCR 데이터(금융 및 물류) LMDB 생성 시작")
    
    base_path = "/mnt/nas/ocr_dataset/025.OCR 데이터(금융 및 물류)/01-1.정식개방데이터"
    
    datasets = [
        {
            'name': 'finance_logistics_train',
            'image_dir': f"{base_path}/Training/01.원천데이터",
            'gt_dir': f"{base_path}/Training/02.라벨링데이터",
            'output_path': "/mnt/nas/ocr_dataset/finance_logistics_train.lmdb"
        },
        {
            'name': 'finance_logistics_val',
            'image_dir': f"{base_path}/Validation/01.원천데이터",
            'gt_dir': f"{base_path}/Validation/02.라벨링데이터",
            'output_path': "/mnt/nas/ocr_dataset/finance_logistics_val.lmdb"
        }
    ]
    
    for dataset in datasets:
        try:
            print(f"📂 {dataset['name']} 처리 중...")
            
            if os.path.exists(dataset['image_dir']) and os.path.exists(dataset['gt_dir']):
                create_lmdb_dataset(
                    image_dir=dataset['image_dir'],
                    gt_dir=dataset['gt_dir'],
                    output_path=dataset['output_path'],
                    annotation_parser='ocr_public'  # 구조가 동일하므로 같은 파서 사용
                )
                print(f"✅ {dataset['name']} LMDB 생성 완료")
            else:
                print(f"⚠️ {dataset['name']} 경로를 찾을 수 없습니다")
        except Exception as e:
            print(f"❌ {dataset['name']} LMDB 생성 실패: {e}")

def create_handwriting_lmdb():
    """053.대용량 손글씨 OCR 데이터 LMDB 생성"""
    print("🔄 053.대용량 손글씨 OCR 데이터 LMDB 생성 시작")
    
    base_path = "/mnt/nas/ocr_dataset/053.대용량 손글씨 OCR 데이터/01.데이터"
    
    # 복잡한 구조이므로 일부만 먼저 테스트
    datasets = [
        {
            'name': 'handwriting_sample',
            'image_dir': f"{base_path}/1.Training/원천데이터",  # 경로 확인 필요
            'gt_dir': f"{base_path}/1.Training/라벨링데이터",   # 경로 확인 필요
            'output_path': "/mnt/nas/ocr_dataset/handwriting_sample.lmdb"
        }
    ]
    
    for dataset in datasets:
        try:
            print(f"📂 {dataset['name']} 처리 중...")
            
            if os.path.exists(dataset['image_dir']) and os.path.exists(dataset['gt_dir']):
                create_lmdb_dataset(
                    image_dir=dataset['image_dir'],
                    gt_dir=dataset['gt_dir'],
                    output_path=dataset['output_path'],
                    annotation_parser='handwriting_ocr'
                )
                print(f"✅ {dataset['name']} LMDB 생성 완료")
            else:
                print(f"⚠️ {dataset['name']} 경로를 찾을 수 없습니다")
                print(f"   이미지: {dataset['image_dir']}")
                print(f"   GT: {dataset['gt_dir']}")
        except Exception as e:
            print(f"❌ {dataset['name']} LMDB 생성 실패: {e}")

def create_public_admin_lmdb():
    """공공행정문서 OCR LMDB 생성"""
    print("🔄 공공행정문서 OCR LMDB 생성 시작")
    
    base_path = "/mnt/nas/ocr_dataset/공공행정문서 OCR"
    
    # 확인된 경로들
    datasets = [
        {
            'name': 'public_admin_train1',
            'image_dir': f"{base_path}/Training/[원천]train1/02.원천데이터(jpg)",
            'gt_dir': f"{base_path}/Training/[라벨]train/01.라벨링데이터(Json)",
            'output_path': "/mnt/nas/ocr_dataset/public_admin_train1.lmdb"
        },
        {
            'name': 'public_admin_train2',
            'image_dir': f"{base_path}/Training/[원천]train2/02.원천데이터(jpg)",
            'gt_dir': f"{base_path}/Training/[라벨]train/01.라벨링데이터(Json)",
            'output_path': "/mnt/nas/ocr_dataset/public_admin_train2.lmdb"
        },
        {
            'name': 'public_admin_train3',
            'image_dir': f"{base_path}/Training/[원천]train3/02.원천데이터(jpg)",
            'gt_dir': f"{base_path}/Training/[라벨]train/01.라벨링데이터(Json)",
            'output_path': "/mnt/nas/ocr_dataset/public_admin_train3.lmdb"
        },
        {
            'name': 'public_admin_val',
            'image_dir': f"{base_path}/Validation/[원천]validation/02.원천데이터(Jpg)",
            'gt_dir': f"{base_path}/Validation/[라벨]validation/01.라벨링데이터(Json)",
            'output_path': "/mnt/nas/ocr_dataset/public_admin_val.lmdb"
        }
    ]
    
    for dataset in datasets:
        try:
            print(f"📂 {dataset['name']} 처리 중...")
            
            if os.path.exists(dataset['image_dir']) and os.path.exists(dataset['gt_dir']):
                create_lmdb_dataset(
                    image_dir=dataset['image_dir'],
                    gt_dir=dataset['gt_dir'],
                    output_path=dataset['output_path'],
                    annotation_parser='public_admin_ocr'
                )
                print(f"✅ {dataset['name']} LMDB 생성 완료")
            else:
                print(f"⚠️ {dataset['name']} 경로를 찾을 수 없습니다")
                print(f"   이미지: {dataset['image_dir']}")
                print(f"   GT: {dataset['gt_dir']}")
        except Exception as e:
            print(f"❌ {dataset['name']} LMDB 생성 실패: {e}")

def check_paths():
    """경로 존재 여부 확인"""
    print("🔍 경로 존재 여부 확인 중...")
    
    paths_to_check = [
        "/mnt/nas/ocr_dataset/13.한국어글자체/04. Text in the wild_230209_add",
        "/mnt/nas/ocr_dataset/13.한국어글자체/04. Text in the wild_230209_add/textinthewild_data_info.json",
        "/mnt/nas/ocr_dataset/023.OCR 데이터(공공)/01-1.정식개방데이터/Training",
        "/mnt/nas/ocr_dataset/025.OCR 데이터(금융 및 물류)",
        "/mnt/nas/ocr_dataset/053.대용량 손글씨 OCR 데이터/01.데이터",
        "/mnt/nas/ocr_dataset/공공행정문서 OCR/Training/[원천]train1",
        "/mnt/nas/ocr_dataset/공공행정문서 OCR/Validation/[원천]validation"
    ]
    
    for path in paths_to_check:
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"   {status} {path}")

def main():
    """메인 함수"""
    print("🚀 LMDB 데이터셋 생성 스크립트 시작")
    print("=" * 50)
    
    # 출력 디렉토리 생성
    os.makedirs("/mnt/nas/ocr_dataset", exist_ok=True)
    
    # 경로 확인
    check_paths()
    print()
    
    # 사용자 선택
    print("생성할 데이터셋을 선택하세요:")
    print("0. 경로 확인만")
    print("1. Text in the wild")
    print("2. 023.OCR 데이터(공공)")
    print("3. 025.OCR 데이터(금융 및 물류)")
    print("4. 053.대용량 손글씨 OCR 데이터")
    print("5. 공공행정문서 OCR")
    print("6. 전체 데이터셋")
    
    choice = input("선택 (0-6): ").strip()
    
    if choice == '0':
        print("경로 확인 완료!")
    elif choice == '1':
        create_text_in_wild_lmdb()
    elif choice == '2':
        create_ocr_public_lmdb()
    elif choice == '3':
        create_finance_logistics_lmdb()
    elif choice == '4':
        create_handwriting_lmdb()
    elif choice == '5':
        create_public_admin_lmdb()
    elif choice == '6':
        print("🔄 전체 데이터셋 생성 시작...")
        create_text_in_wild_lmdb()
        create_ocr_public_lmdb()
        create_finance_logistics_lmdb()
        create_handwriting_lmdb()
        create_public_admin_lmdb()
    else:
        print("❌ 잘못된 선택입니다.")
        sys.exit(1)
    
    print("=" * 50)
    print("✅ LMDB 데이터셋 생성 완료!")

if __name__ == '__main__':
    main() 