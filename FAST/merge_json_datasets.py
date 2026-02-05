#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import subprocess
import time
import gc
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import numba
from numba import jit, prange

# FTP 설정
FTP_BASE_PATH = "/run/user/0/gvfs/ftp:host=172.30.1.226/Y:\\ocr_dataset"
LOCAL_OUTPUT_PATH = "/mnt/nas/ocr_dataset/json_data"
BACKUP_OUTPUT_PATH = "/home/mango/ocr_test/FAST/json_merged"  # 기존 경로도 백업용으로 사용

@jit(nopython=True, cache=True)
def calculate_file_size_mb(file_size_bytes):
    """파일 크기를 MB로 변환 (numba 최적화)"""
    return file_size_bytes / (1024.0 * 1024.0)

@jit(nopython=True, cache=True)
def check_step_save_condition(current_count, step_count, step_size=100000):
    """10000개 단위 저장 조건 확인 (numba 최적화)"""
    return current_count >= (step_count + 1) * step_size

@jit(nopython=True, cache=True)
def calculate_id_offsets(images_count, annotations_count):
    """ID 오프셋 계산 (numba 최적화)"""
    return images_count, annotations_count

def cleanup_memory():
    """메모리 정리"""
    gc.collect()
    try:
        subprocess.run(['sync'], check=False)
        subprocess.run(['echo', '1'], stdout=subprocess.PIPE, check=False)
        subprocess.run(['tee', '/proc/sys/vm/drop_caches'], input=b'1', check=False)
    except:
        pass

def force_cleanup_memory():
    """강제 메모리 정리"""
    for _ in range(3):
        gc.collect()
    try:
        subprocess.run(['sync'], check=False)
        subprocess.run(['echo', '3'], stdout=subprocess.PIPE, check=False)
        subprocess.run(['tee', '/proc/sys/vm/drop_caches'], input=b'3', check=False)
    except:
        pass

def remount_ftp_for_large_file():
    """대용량 파일 처리를 위해 gvfs 경로 재확인"""
    gvfs_path = "/run/user/0/gvfs/ftp:host=172.30.1.226/Y:\\ocr_dataset"
    
    # gvfs 경로가 여전히 존재하는지 확인
    if os.path.exists(gvfs_path):
        print("✅ gvfs FTP 경로 재확인 완료")
        return True
    else:
        print("❌ gvfs FTP 경로가 연결되지 않음")
        print("💡 파일 관리자에서 FTP 서버에 재접속해주세요")
        return False

def setup_ftp_mount():
    """gvfs FTP 경로 확인"""
    print("🔄 gvfs FTP 경로 확인 중...")
    
    # gvfs를 사용한 FTP 접근
    gvfs_path = "/run/user/0/gvfs/ftp:host=172.30.1.226/Y:\\ocr_dataset"
    
    # gvfs 경로가 존재하는지 확인
    if os.path.exists(gvfs_path):
        print("✅ gvfs FTP 경로 확인 완료")
        return True
    else:
        print("❌ gvfs FTP 경로를 찾을 수 없음")
        print("💡 파일 관리자에서 FTP 서버에 접속하여 gvfs 마운트를 활성화해주세요")
        return False

def load_json_simple(json_path):
    """간단한 JSON 파일 로드 (메모리 매핑 사용)"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        return None

def load_json_mmap(json_path):
    """메모리 매핑을 사용한 JSON 파일 로드 (더 빠름)"""
    try:
        import mmap
        with open(json_path, 'r', encoding='utf-8') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                data = json.loads(mm.read().decode('utf-8'))
        return data
    except Exception as e:
        # 메모리 매핑 실패시 일반 방법으로 폴백
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except:
            return None

def process_json_file(json_file, dataset_name):
    """개별 JSON 파일 처리"""
    data = load_json_mmap(str(json_file))
    if not data:
        return None
    
    original_path = str(json_file)
    result = {
        'images': [],
        'annotations': [],
        'original_path': original_path,
        'sub_dataset': json_file.stem
    }
    
    if 'public_admin' in dataset_name:
        # 공공행정문서: images, annotations 구조
        result['images'] = [
            {**img, 'dataset': dataset_name, 'sub_dataset': json_file.stem, 'original_json_path': original_path}
            for img in data.get('images', [])
        ]
        result['annotations'] = [
            {**ann, 'dataset': dataset_name, 'sub_dataset': json_file.stem, 'original_json_path': original_path}
            for ann in data.get('annotations', [])
        ]
    
    elif any(name in dataset_name for name in ['ocr_public', 'finance_logistics', 'handwriting']):
        # OCR 공공/금융/손글씨: Images, bbox 구조
        if 'Images' in data:
            img_info = data['Images']
            result['images'] = [{
                'dataset': dataset_name,
                'sub_dataset': json_file.stem,
                'original_json_path': original_path,
                'width': img_info.get('width', 0),
                'height': img_info.get('height', 0),
                'file_name': img_info.get('filename', json_file.stem)
            }]
            
            # bbox 정보 처리
            bbox_key = 'bbox' if 'bbox' in data else 'Bbox'
            if bbox_key in data:
                result['annotations'] = [
                    {
                        'dataset': dataset_name,
                        'sub_dataset': json_file.stem,
                        'original_json_path': original_path,
                        'bbox': bbox_info.get('x', []) + bbox_info.get('y', []),
                        'text': bbox_info.get('data', '')
                    }
                    for bbox_info in data[bbox_key]
                ]
    
    return result

def save_intermediate_result(dataset_data, dataset_name, step_count):
    """10000개 단위로 중간 결과 저장"""
    output_path = os.path.join(LOCAL_OUTPUT_PATH, f"{dataset_name}_100000_step_save_{step_count:03d}.json")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_data, f, ensure_ascii=False, indent=2)
        
        # numba 최적화된 파일 크기 계산
        file_size_mb = calculate_file_size_mb(os.path.getsize(output_path))
        print(f"100000_step_save_{step_count:03d}: {len(dataset_data['images']):,}개 이미지, {file_size_mb:.1f}MB")
        
    except Exception as e:
        print(f"❌ 중간 저장 실패: {e}")

def load_json_with_retry(json_path):
    """JSON 파일 로드 (재시도 로직 포함)"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return data
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                cleanup_memory()
                force_cleanup_memory()
                if not remount_ftp_for_large_file():
                    return None
                time.sleep(3)
            else:
                return load_json_via_local_download(json_path)
        except Exception as e:
            if attempt < max_retries - 1:
                cleanup_memory()
                force_cleanup_memory()
                if not remount_ftp_for_large_file():
                    return None
                time.sleep(3)
            else:
                return load_json_via_local_download(json_path)
    return None

def load_json_via_local_download(json_path):
    """로컬 다운로드로 JSON 파일 로드"""
    temp_json_path = "/tmp/temp_json_file.json"
    try:
        subprocess.run(['cp', json_path, temp_json_path], check=True)
        with open(temp_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        os.remove(temp_json_path)
        return data
    except Exception as e:
        print(f"❌ 로컬 다운로드 실패: {e}")
        if os.path.exists(temp_json_path):
            os.remove(temp_json_path)
        return None

def find_latest_step_save(dataset_name):
    """가장 최근 중간 저장 파일 찾기"""
    step_files = []
    for file in os.listdir(LOCAL_OUTPUT_PATH):
        if file.startswith(f"{dataset_name}_100000_step_save_") and file.endswith('.json'):
            step_files.append(file)
    
    if not step_files:
        return None, 0
    
    # 파일명에서 step 번호 추출하여 정렬
    step_files.sort(key=lambda x: int(x.split('_')[-1].replace('.json', '')))
    latest_file = step_files[-1]
    latest_step = int(latest_file.split('_')[-1].replace('.json', ''))
    
    return os.path.join(LOCAL_OUTPUT_PATH, latest_file), latest_step

def load_latest_step_data(dataset_name):
    """가장 최근 중간 저장 파일 로드"""
    latest_file, latest_step = find_latest_step_save(dataset_name)
    
    if latest_file and os.path.exists(latest_file):
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"📂 중간 저장 파일 로드: {latest_file} (step {latest_step})")
            print(f"📊 현재 진행 상황: {len(data['images']):,}개 이미지")
            return data, latest_step
        except Exception as e:
            print(f"❌ 중간 저장 파일 로드 실패: {e}")
            return None, 0
    
    return None, 0

def merge_json_datasets():
    """각 한국어 OCR 데이터셋별로 JSON 파일 생성 (100000개씩 중간 저장)"""
    print("🚀 한국어 OCR 데이터셋별 JSON 파일 생성 시작 (100000개씩 중간 저장)")
    print("=" * 60)
    
    # 출력 디렉토리 생성
    os.makedirs(LOCAL_OUTPUT_PATH, exist_ok=True)
    
    # create_all_datasets_500.py 기준 실제 경로로 수정
    datasets = [
        {
            "name": "public_admin_train",
            "path": f"{FTP_BASE_PATH}/공공행정문서 OCR/Training/[라벨]train/01.라벨링데이터(Json)",
            "type": "multi",
            "pattern": "**/*.json"
        },
        {
            "name": "public_admin_train_partly",
            "path": f"{FTP_BASE_PATH}/공공행정문서 OCR/Training/[라벨]train_partly_labling",
            "type": "multi",
            "pattern": "**/*.json"
        },
        {
            "name": "public_admin_valid",
            "path": f"{FTP_BASE_PATH}/공공행정문서 OCR/Validation/[라벨]validation/01.라벨링데이터(Json)",
            "type": "multi",
            "pattern": "**/*.json"
        },
        {
            "name": "ocr_public_train",
            "path": f"{FTP_BASE_PATH}/023.OCR 데이터(공공)/01-1.정식개방데이터/Training/02.라벨링데이터",
            "type": "multi",
            "pattern": "**/*.json"
        },
        {
            "name": "ocr_public_valid",
            "path": f"{FTP_BASE_PATH}/023.OCR 데이터(공공)/01-1.정식개방데이터/Validation/02.라벨링데이터",
            "type": "multi",
            "pattern": "**/*.json"
        },
        {
            "name": "finance_logistics_train",
            "path": f"{FTP_BASE_PATH}/025.OCR 데이터(금융 및 물류)/01-1.정식개방데이터/Training/02.라벨링데이터",
            "type": "multi",
            "pattern": "**/*.json"
        },
        {
            "name": "finance_logistics_valid",
            "path": f"{FTP_BASE_PATH}/025.OCR 데이터(금융 및 물류)/01-1.정식개방데이터/Validation/02.라벨링데이터",
            "type": "multi",
            "pattern": "**/*.json"
        },
        {
            "name": "handwriting_train",
            "path": f"{FTP_BASE_PATH}/053.대용량 손글씨 OCR 데이터/01.데이터/1.Training/라벨링데이터",
            "type": "multi",
            "pattern": "**/*.json"
        },
        {
            "name": "handwriting_valid",
            "path": f"{FTP_BASE_PATH}/053.대용량 손글씨 OCR 데이터/01.데이터/2.Validation/라벨링데이터",
            "type": "multi",
            "pattern": "**/*.json"
        }
    ]
    
    created_files = []
    
    # 이미 완료된 데이터셋들 확인
    completed_datasets = []
    for dataset in datasets:
        # 새로운 경로 확인
        output_file = os.path.join(LOCAL_OUTPUT_PATH, f"{dataset['name']}_merged.json")
        # 기존 경로도 확인
        old_output_file = os.path.join(BACKUP_OUTPUT_PATH, f"{dataset['name']}_merged.json")
        
        if os.path.exists(output_file):
            # numba 최적화된 파일 크기 계산
            file_size = calculate_file_size_mb(os.path.getsize(output_file))
            print(f"✅ {dataset['name']} 이미 완료됨 (파일 크기: {file_size:.2f} MB)")
            completed_datasets.append(dataset['name'])
        elif os.path.exists(old_output_file):
            # 기존 경로에 있는 파일 확인
            file_size = calculate_file_size_mb(os.path.getsize(old_output_file))
            print(f"✅ {dataset['name']} 이미 완료됨 (기존 경로, 파일 크기: {file_size:.2f} MB)")
            completed_datasets.append(dataset['name'])
        else:
            print(f"🔄 {dataset['name']} 처리 필요")
    
    print(f"\n📊 처리할 데이터셋: {len(datasets) - len(completed_datasets)}개")
    print(f"📊 건너뛸 데이터셋: {len(completed_datasets)}개")
    
    for dataset in datasets:
        # 이미 완료된 데이터셋은 건너뛰기
        if dataset['name'] in completed_datasets:
            print(f"\n⏭️ {dataset['name']} 건너뛰기 (이미 완료됨)")
            continue
            
        print(f"\n📊 {dataset['name']} 데이터셋 처리 중...")
        
        # 각 데이터셋별로 새로운 데이터 구조 생성
        dataset_data = {
            "images": [],
            "annotations": [],
            "categories": [],
            "info": {
                "description": f"Korean OCR Dataset - {dataset['name']}",
                "version": "1.0",
                "year": 2024,
                "contributor": "OCR Test Project"
            }
        }
        
        if dataset['type'] == 'single':
            # 단일 JSON 파일
            json_path = dataset['path']
            if os.path.exists(json_path):
                data = load_json_with_retry(json_path)
                if data:
                    # 이미지 ID 재매핑
                    image_id_offset = len(dataset_data['images'])
                    annotation_id_offset = len(dataset_data['annotations'])
                    
                    # 이미지 정보 추가
                    for i, img in enumerate(data.get('images', [])):
                        # id가 없으면 새로 생성
                        if 'id' not in img:
                            img['id'] = image_id_offset + i
                        else:
                            img['id'] += image_id_offset
                        img['dataset'] = dataset['name']
                        dataset_data['images'].append(img)
                    
                    # 어노테이션 정보 추가
                    for i, ann in enumerate(data.get('annotations', [])):
                        # id가 없으면 새로 생성
                        if 'id' not in ann:
                            ann['id'] = annotation_id_offset + i
                        else:
                            ann['id'] += annotation_id_offset
                        
                        # image_id가 없으면 첫 번째 이미지의 id를 사용
                        if 'image_id' not in ann:
                            ann['image_id'] = image_id_offset
                        else:
                            ann['image_id'] += image_id_offset
                            
                        ann['dataset'] = dataset['name']
                        dataset_data['annotations'].append(ann)
                    
                    # 카테고리 정보 추가 (중복 제거)
                    for cat in data.get('categories', []):
                        if cat not in dataset_data['categories']:
                            dataset_data['categories'].append(cat)
                    
                    print(f"✅ {dataset['name']}: {len(data.get('images', []))}개 이미지, {len(data.get('annotations', []))}개 어노테이션")
                else:
                    print(f"❌ {dataset['name']} JSON 로드 실패")
            else:
                print(f"❌ {dataset['name']} 경로를 찾을 수 없음: {json_path}")
        
        elif dataset['type'] == 'multi':
            # 여러 JSON 파일
            dataset_path = dataset['path']
    
            if os.path.exists(dataset_path):
                # 중간 저장 파일 확인 및 로드
                resume_data, resume_step = load_latest_step_data(dataset['name'])
                
                if resume_data:
                    # 중간 저장 파일에서 이어서 시작
                    dataset_data = resume_data
                    step_count = resume_step
                    print(f"🔄 {dataset['name']} 중간 저장 파일에서 이어서 시작 (step {step_count})")
                else:
                    # 새로 시작
                    step_count = 0
                
                # os.scandir을 사용한 재귀적 파일 검색
                json_files = []
                
                def scan_directory_recursive(directory):
                    try:
                        with os.scandir(directory) as entries:
                            for entry in entries:
                                if entry.is_file() and entry.name.endswith('.json'):
                                    json_files.append(Path(entry.path))
                                elif entry.is_dir():
                                    scan_directory_recursive(entry.path)
                    except Exception as e:
                        print(f"❌ scandir 실패: {e}")
                
                scan_directory_recursive(dataset_path)
                print(f"📁 {len(json_files)}개 JSON 파일 발견")
                
                # 스레드 수 설정 (디버그 결과 기반 - 1개가 최고 성능)
                num_workers = 20  # 디버그에서 가장 빠른 성능
                print(f"🔧 스레드 수: {num_workers}개")
                
                # 순차 처리하면서 10000개씩 중간 저장
                image_id_offset = len(dataset_data['images'])
                annotation_id_offset = len(dataset_data['annotations'])
                
                # 이미 처리된 파일 수만큼 건너뛰기
                processed_count = step_count * 100000
                if processed_count > 0:
                    print(f"⏭️ 이미 처리된 {processed_count:,}개 파일 건너뛰기")
                
                # 병렬 처리로 개선 (원래 로직 유지)
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    # 이미 처리된 파일 수만큼 건너뛰기
                    remaining_files = json_files[processed_count:]
                    
                    # 병렬로 future 생성
                    futures = {executor.submit(process_json_file, json_file, dataset['name']): json_file 
                             for json_file in remaining_files}
                    
                    # tqdm으로 진행률 표시하면서 병렬 처리
                    for future in tqdm(as_completed(futures), total=len(futures), desc=f"{dataset['name']} 처리"):
                        json_file = futures[future]
                        result = future.result()
                        
                        if result:
                            # 이미지 ID 재매핑 (원래 로직 유지)
                            for j, img in enumerate(result['images']):
                                if 'id' not in img:
                                    img['id'] = image_id_offset + j
                                else:
                                    img['id'] += image_id_offset
                                dataset_data['images'].append(img)
                            
                            # 어노테이션 ID 재매핑 (원래 로직 유지)
                            for j, ann in enumerate(result['annotations']):
                                if 'id' not in ann:
                                    ann['id'] = annotation_id_offset + j
                                else:
                                    ann['id'] += annotation_id_offset
                                
                                if 'image_id' not in ann:
                                    ann['image_id'] = image_id_offset
                                else:
                                    ann['image_id'] += image_id_offset
                                
                                dataset_data['annotations'].append(ann)
                            
                            # numba 최적화된 ID 오프셋 업데이트 (원래 로직 유지)
                            image_id_offset, annotation_id_offset = calculate_id_offsets(
                                len(dataset_data['images']), 
                                len(dataset_data['annotations'])
                            )
                            
                            # numba 최적화된 10000개 단위 저장 조건 확인 (원래 로직 유지)
                            if check_step_save_condition(len(dataset_data['images']), step_count):
                                step_count += 1 
                                save_intermediate_result(dataset_data, dataset['name'], step_count)
            else:
                print(f"❌ {dataset['name']} 경로를 찾을 수 없음: {dataset_path}")
        
        # 각 데이터셋별로 개별 파일 저장
        if dataset_data['images'] or dataset_data['annotations']:
            output_path = os.path.join(LOCAL_OUTPUT_PATH, f"{dataset['name']}_merged.json")
            print(f"\n💾 {dataset['name']} 최종 저장 중: {output_path}")
            
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(dataset_data, f, ensure_ascii=False, indent=2)
                
                # numba 최적화된 파일 크기 확인
                file_size_mb = calculate_file_size_mb(os.path.getsize(output_path))
                print(f"✅ {dataset['name']} 저장 완료!")
                print(f"📊 {len(dataset_data['images']):,}개 이미지")
                print(f"📊 {len(dataset_data['annotations']):,}개 어노테이션")
                print(f"📊 {len(dataset_data['categories'])}개 카테고리")
                print(f"📁 파일 크기: {file_size_mb:.2f} MB")
                print(f"📁 저장 위치: {output_path}")
                
                created_files.append(output_path)
                
            except Exception as e:
                print(f"❌ {dataset['name']} 파일 저장 실패: {e}")
        
        # 메모리 정리
        cleanup_memory()
    
    # 생성된 파일 목록 반환
    print(f"\n🎉 모든 데이터셋 처리 완료!")
    print(f"📁 생성된 파일들:")
    for file_path in created_files:
        print(f"   - {file_path}")
    
    return created_files

def main():
    """메인 함수"""
    print("🚀 한국어 OCR 데이터셋 JSON 병합 도구 (100000개씩 중간 저장 + Numba 최적화)")
    print("=" * 60)
    
    # gvfs FTP 경로 확인
    if not setup_ftp_mount():
        print("❌ gvfs FTP 경로 확인 실패")
        return
    
    # gvfs 경로 확인
    if not os.path.exists(FTP_BASE_PATH):
        print("❌ gvfs FTP 경로 확인 실패")
        return
    
    print("✅ gvfs FTP 경로 확인 완료")
    
    # 각 데이터셋별 JSON 파일 생성
    created_files = merge_json_datasets()
    
    if created_files:
        print("\n🎉 모든 작업이 완료되었습니다!")
        print(f"📁 생성된 JSON 파일들: {len(created_files)}개")
        print("\n💡 이제 이 파일들을 사용하여 LMDB를 생성할 수 있습니다.")
    else:
        print("\n❌ JSON 파일 생성 작업이 실패했습니다.")

if __name__ == "__main__":
    main()
