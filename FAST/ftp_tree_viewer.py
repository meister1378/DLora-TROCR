#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OCR 데이터셋 완전 매핑 규칙 분석 및 최적화 함수 생성 (수정된 버전)
"""

import os
import sys
import orjson
import bigjson
from pathlib import Path
from collections import defaultdict
import random
import re

# FTP 마운트된 데이터셋 기본 경로
FTP_BASE_PATH = "/run/user/0/gvfs/ftp:host=172.30.1.226/Y:\\ocr_dataset"
MERGED_JSON_PATH = "/home/mango/ocr_test/FAST/json_merged"

def analyze_complete_mapping_rules(json_path, base_path, dataset_name, sample_count=None):
    """완전한 매핑 규칙 분석 및 최적화 함수 생성 (수정된 버전)"""
    print(f"\n{'='*60}")
    print(f"🎯 {dataset_name} 완전 매핑 규칙 분석 (수정된 버전)")
    print(f"{'='*60}")
    
    # 경로 검증 및 수정
    corrected_base_path = fix_base_path(base_path, dataset_name)
    print(f"📂 수정된 베이스 경로: {corrected_base_path}")
    
    # 1. JSON 구조 분석
    json_patterns = analyze_json_structure_enhanced(json_path, dataset_name, sample_count)
    
    # 2. 디렉토리 구조 분석 (제한 없이)
    file_patterns = analyze_directory_structure_enhanced(corrected_base_path, dataset_name)
    
    # 3. 실제 파일명 패턴 분석 (새로 추가)
    actual_filename_patterns = analyze_actual_filenames(corrected_base_path, dataset_name)
    
    # 4. 매핑 규칙 생성 (개선된 버전)
    mapping_rules = create_mapping_rules_enhanced(json_patterns, file_patterns, actual_filename_patterns, dataset_name)
    
    # 5. 매핑 규칙 테스트
    test_mapping_accuracy_enhanced(mapping_rules, json_patterns, file_patterns, dataset_name)
    
    # 6. 최적화 함수 코드 생성
    generate_optimized_lookup_function(mapping_rules, dataset_name)
    
    return mapping_rules

def fix_base_path(base_path, dataset_name):
    """데이터셋 이름에 따라 올바른 베이스 경로 설정"""
    if "손글씨" in dataset_name:
        if "Train" in dataset_name:
            return f"{FTP_BASE_PATH}/053.대용량 손글씨 OCR 데이터/01.데이터/1.Training/원천데이터"
        else:
            return f"{FTP_BASE_PATH}/053.대용량 손글씨 OCR 데이터/01.데이터/2.Validation/원천데이터"
    elif "OCR공공" in dataset_name:
        if "Train" in dataset_name:
            return f"{FTP_BASE_PATH}/023.OCR 데이터(공공)/01-1.정식개방데이터/Training/01.원천데이터"
        else:
            return f"{FTP_BASE_PATH}/023.OCR 데이터(공공)/01-1.정식개방데이터/Validation/01.원천데이터"
    elif "금융물류" in dataset_name:
        if "Train" in dataset_name:
            return f"{FTP_BASE_PATH}/025.OCR 데이터(금융 및 물류)/01-1.정식개방데이터/Training/01.원천데이터"
        else:
            return f"{FTP_BASE_PATH}/025.OCR 데이터(금융 및 물류)/01-1.정식개방데이터/Validation/01.원천데이터"
    
    return base_path

def analyze_actual_filenames(base_path, dataset_name):
    """실제 파일명 패턴 분석 (제한 없이 전체 스캔)"""
    print(f"\n🔍 {dataset_name} 실제 파일명 패턴 분석:")
    
    if not os.path.exists(base_path):
        print(f"❌ 경로 없음: {base_path}")
        return {}
    
    filename_patterns = defaultdict(list)
    directory_counts = defaultdict(int)
    sample_filenames = []
    total_files = 0
    
    print(f"  🔄 전체 디렉토리 무제한 스캔 중...")
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                relative_dir = os.path.relpath(root, base_path)
                directory_counts[relative_dir] += 1
                
                # 파일명 패턴 분석
                if total_files < 200:  # 샘플 수집
                    sample_filenames.append((file, relative_dir))
                
                # 카테고리별 분류
                category = extract_category_from_filename(file, dataset_name)
                if category:
                    filename_patterns[category].append((file, relative_dir))
                
                total_files += 1
    
    print(f"  ✅ 실제 파일 분석 완료: {total_files:,}개 파일")
    print(f"    📊 디렉토리별 파일 수:")
    
    # 상위 10개 디렉토리 출력
    top_dirs = sorted(directory_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for dir_path, count in top_dirs:
        print(f"      📂 {dir_path}: {count:,}개")
    
    # 실제 파일명 샘플 출력
    print(f"    📝 실제 파일명 샘플 (처음 10개):")
    for i, (filename, dir_path) in enumerate(sample_filenames[:10]):
        print(f"      {i+1}. {filename} (📂 {dir_path})")
    
    # 카테고리별 실제 파일 예시
    print(f"    📋 카테고리별 실제 파일 예시:")
    for category, files in filename_patterns.items():
        if files:
            example_file, example_dir = files[0]
            print(f"      {category}: {example_file} (📂 {example_dir})")
    
    return {
        'total_files': total_files,
        'directory_counts': dict(directory_counts),
        'filename_patterns': dict(filename_patterns),
        'sample_filenames': sample_filenames[:100]  # 상위 100개만 저장
    }

def analyze_json_structure_enhanced(json_path, dataset_name, sample_count=None):
    """향상된 JSON 구조 분석"""
    print(f"\n📄 {dataset_name} JSON 구조 정밀 분석:")
    
    if not os.path.exists(json_path):
        print(f"❌ JSON 파일 없음: {json_path}")
        return {}
    
    data = None
    file_handle = None
    
    try:
        file_size_gb = os.path.getsize(json_path) / (1024**3)
        print(f"  파일 크기: {file_size_gb:.2f} GB")
        
        # 파일 크기에 따라 로더 선택
        if file_size_gb > 1.0:
            print("  🔄 bigjson으로 로드...")
            file_handle = open(json_path, 'rb')
            data = bigjson.load(file_handle)
        else:
            print("  🔄 orjson으로 로드...")
            with open(json_path, 'rb') as f:
                data = orjson.loads(f.read())
        
        images = data.get('images', [])
        annotations = data.get('annotations', [])
        
        print(f"  📊 로드된 데이터: 이미지 {len(images) if isinstance(images, list) else 'bigjson Array'}, "
              f"어노테이션 {len(annotations) if isinstance(annotations, list) else 'bigjson Array'}")
        
        # 샘플 데이터 추출 (더 넓은 범위에서)
        patterns = {
            'file_names': [],
            'sub_datasets': [],
            'categories': defaultdict(list),
            'path_patterns': defaultdict(list),
            'total_samples': 0
        }
        
        # 전체 데이터 또는 샘플링
        if isinstance(images, list):
            total_images = len(images)
            if sample_count is None:
                # 전체 처리
                sample_indices = range(total_images)
                print(f"  🔄 전체 {total_images}개 이미지 분석 중...")
            elif sample_count < total_images:
                # 전체 구간에서 균등 샘플링
                step = max(1, total_images // sample_count)
                sample_indices = [i * step for i in range(sample_count)]
                print(f"  🔄 {len(sample_indices)}개 샘플 분석 중 (넓은 범위)...")
            else:
                sample_indices = range(total_images)
                print(f"  🔄 전체 {total_images}개 이미지 분석 중...")
        else:
            # bigjson Array의 경우
            if sample_count is None:
                print(f"  🔄 bigjson Array 전체 분석 중 (순차 처리)...")
                sample_indices = None  # 특별 처리
            else:
                # bigjson Array의 경우 더 넓은 범위에서 샘플링
                sample_indices = [i * 1000 for i in range(sample_count)]
                print(f"  🔄 {len(sample_indices)}개 샘플 분석 중 (넓은 범위)...")
        
        
        if sample_indices is None:
            # bigjson Array 전체 처리 (순차)
            i = 0
            while True:
                try:
                    img = images[i]
                    if img is None:
                        break
                    
                    # 데이터셋별 파일명 필드
                    if "공공행정" in dataset_name:
                        file_name = img.get('image.file.name', '')
                    else:
                        file_name = img.get('file_name', '')
                    
                    sub_dataset = img.get('sub_dataset', '')
                    original_path = img.get('original_json_path', '')
                    
                    if file_name:
                        patterns['file_names'].append(file_name)
                        
                        # 카테고리 추출
                        category = extract_category_from_filename(file_name, dataset_name)
                        if category:
                            patterns['categories'][category].append(file_name)
                        
                        # 경로 패턴 추출
                        if original_path:
                            path_pattern = extract_path_pattern(original_path)
                            patterns['path_patterns'][path_pattern].append(file_name)
                    
                    if sub_dataset:
                        patterns['sub_datasets'].append(sub_dataset)
                    
                    patterns['total_samples'] += 1
                    
                    # 진행상황 출력 (10000개마다)
                    if i % 10000 == 0:
                        print(f"    📊 JSON 분석 진행: {patterns['total_samples']:,}개")
                    
                    i += 1
                    
                except (IndexError, TypeError) as e:
                    break
        else:
            # 인덱스 기반 처리
            for i in sample_indices:
                try:
                    img = images[i]
                    
                    # 데이터셋별 파일명 필드
                    if "공공행정" in dataset_name:
                        file_name = img.get('image.file.name', '')
                    else:
                        file_name = img.get('file_name', '')
                    
                    sub_dataset = img.get('sub_dataset', '')
                    original_path = img.get('original_json_path', '')
                    
                    if file_name:
                        patterns['file_names'].append(file_name)
                        
                        # 카테고리 추출
                        category = extract_category_from_filename(file_name, dataset_name)
                        if category:
                            patterns['categories'][category].append(file_name)
                        
                        # 경로 패턴 추출
                        if original_path:
                            path_pattern = extract_path_pattern(original_path)
                            patterns['path_patterns'][path_pattern].append(file_name)
                    
                    if sub_dataset:
                        patterns['sub_datasets'].append(sub_dataset)
                    
                    patterns['total_samples'] += 1
                    
                except (IndexError, TypeError) as e:
                    break
        
        print(f"  ✅ JSON 분석 완료: {patterns['total_samples']}개 샘플")
        print(f"    📋 발견된 카테고리: {list(patterns['categories'].keys())}")
        print(f"    📂 경로 패턴: {len(patterns['path_patterns'])}개")
        
        # JSON 파일명 샘플 출력 (디버깅용)
        print(f"    📝 JSON 파일명 샘플 (처음 10개):")
        for i, fname in enumerate(patterns['file_names'][:10]):
            print(f"      {i+1}. {fname}")
        
        return patterns
        
    except Exception as e:
        print(f"  ❌ JSON 분석 실패: {e}")
        return {}
    
    finally:
        if file_handle:
            try:
                file_handle.close()
            except:
                pass

def analyze_directory_structure_enhanced(base_path, dataset_name):
    """향상된 디렉토리 구조 분석 (제한 없음)"""
    print(f"\n📁 {dataset_name} 디렉토리 구조 정밀 분석 (무제한):")
    
    if not os.path.exists(base_path):
        print(f"❌ 경로 없음: {base_path}")
        return {}
    
    # 디렉토리별 파일 패턴 수집
    dir_patterns = {}
    category_locations = defaultdict(list)
    filename_to_path = {}
    path_templates = []
    
    print(f"  🔄 전체 디렉토리 구조 무제한 스캔...")
    file_count = 0
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, base_path)
                
                # 파일명 → 경로 매핑
                filename_to_path[file] = full_path
                
                # 확장자 없는 파일명도 매핑
                name_without_ext = os.path.splitext(file)[0]
                filename_to_path[name_without_ext] = full_path
                
                # 카테고리별 위치 매핑
                category = extract_category_from_filename(file, dataset_name)
                if category:
                    dir_path = os.path.dirname(relative_path)
                    category_locations[category].append(dir_path)
                
                # 경로 템플릿 생성
                path_template = create_path_template(relative_path, file, dataset_name)
                if path_template not in path_templates:
                    path_templates.append(path_template)
                
                file_count += 1
                
                # 진행상황 출력 (10000개마다)
                if file_count % 10000 == 0:
                    print(f"    📊 스캔 진행: {file_count:,}개")
    
    print(f"  ✅ 디렉토리 분석 완료: {file_count:,}개 파일")
    print(f"    📊 카테고리별 위치:")
    
    # 카테고리별 대표 경로 출력
    for category, locations in category_locations.items():
        unique_locations = list(set(locations))
        print(f"      {category}: {len(unique_locations)}개 위치")
        for loc in unique_locations[:3]:  # 상위 3개만
            print(f"        📂 {loc}")
    
    return {
        'filename_to_path': filename_to_path,
        'category_locations': dict(category_locations),
        'path_templates': path_templates,
        'total_files': file_count
    }

def extract_category_from_filename(filename, dataset_name):
    """파일명에서 카테고리 추출 (확장된 버전)"""
    if "손글씨" in dataset_name:
        for cat in ["4TO", "4PO", "4PR", "4TR"]:
            if f"_{cat}_" in filename:
                return cat
    elif "OCR공공" in dataset_name:
        for cat in ["AF", "CST", "CT", "DI", "EN", "EV", "WF"]:
            if f"{cat}_" in filename or filename.startswith(cat):
                return cat
    elif "금융물류" in dataset_name:
        for cat in ["BL", "PL", "NV"]:
            if f"_{cat}_" in filename:
                return cat
        if "_F_" in filename:  # 금융
            return "F"
    elif "공공행정" in dataset_name:
        # 공공행정은 카테고리가 디렉토리 구조에 있음
        return "ADMIN"
    return None

def extract_path_pattern(original_path):
    """original_json_path에서 패턴 추출"""
    if not original_path:
        return "unknown"
    
    # 경로에서 패턴 추출
    parts = original_path.split('/')
    
    # 중요한 부분들 추출
    patterns = []
    for part in parts:
        if any(keyword in part.lower() for keyword in ['training', 'validation', 'train', 'valid']):
            patterns.append('SPLIT')
        elif any(keyword in part.lower() for keyword in ['원천', 'source', '데이터']):
            patterns.append('SOURCE')
        elif any(keyword in part.lower() for keyword in ['라벨', 'label']):
            patterns.append('LABEL')
    
    return '_'.join(patterns) if patterns else 'unknown'

def create_path_template(relative_path, filename, dataset_name):
    """상대 경로를 템플릿으로 변환"""
    # 파일명을 {FILENAME}으로 대체
    template = relative_path.replace(filename, "{FILENAME}")
    
    # 카테고리 패턴 대체
    category = extract_category_from_filename(filename, dataset_name)
    if category:
        template = template.replace(category, "{CATEGORY}")
    
    # 공통 패턴들 대체
    template = re.sub(r'\d{4,}', '{NUMBER}', template)  # 4자리 이상 숫자
    template = re.sub(r'TS\d+', 'TS{N}', template)  # TS1, TS2 등
    template = re.sub(r'VS\d*', 'VS{N}', template)  # VS, VS1 등
    
    return template

def create_mapping_rules_enhanced(json_patterns, file_patterns, actual_filename_patterns, dataset_name):
    """향상된 매핑 규칙 생성"""
    print(f"\n🎯 {dataset_name} 향상된 매핑 규칙 생성:")
    
    rules = {
        'dataset_name': dataset_name,
        'direct_lookup': {},  # filename → full_path
        'category_rules': {},  # category → path_pattern
        'fallback_patterns': [],  # 우선순위별 검색 패턴
        'optimization_code': ""
    }
    
    if not json_patterns or not file_patterns:
        print("  ❌ 패턴 데이터 부족")
        return rules
    
    # 실제 파일명과 JSON 파일명 비교 분석
    print(f"  🔍 실제 파일명 vs JSON 파일명 매핑 분석:")
    filename_to_path = file_patterns.get('filename_to_path', {})
    json_filenames = json_patterns['file_names']  # 전체 테스트
    
    successful_mappings = 0
    mapping_examples = []
    
    for json_filename in json_filenames:
        found_path = None
        
        # 1. 직접 매핑 시도
        if json_filename in filename_to_path:
            found_path = filename_to_path[json_filename]
            rules['direct_lookup'][json_filename] = found_path
            successful_mappings += 1
            mapping_examples.append((json_filename, found_path, "직접"))
        else:
            # 2. 확장자 추가 시도
            for ext in ['.png', '.jpg', '.jpeg']:
                candidate = f"{json_filename}{ext}"
                if candidate in filename_to_path:
                    found_path = filename_to_path[candidate]
                    rules['direct_lookup'][json_filename] = found_path
                    successful_mappings += 1
                    mapping_examples.append((json_filename, found_path, "확장자"))
                    break
            
            # 3. 부분 매칭 시도
            if not found_path:
                json_base = json_filename.replace('IMG_OCR_53_', '').replace('IMG_OCR_', '')
                for actual_filename, actual_path in filename_to_path.items():
                    if json_base in actual_filename or actual_filename in json_base:
                        found_path = actual_path
                        rules['direct_lookup'][json_filename] = found_path
                        successful_mappings += 1
                        mapping_examples.append((json_filename, found_path, "부분매칭"))
                        break
    
    direct_success_rate = successful_mappings / len(json_filenames) * 100
    print(f"    📊 매핑 성공률: {direct_success_rate:.1f}% ({successful_mappings}/{len(json_filenames)})")
    
    # 성공 사례 출력
    if mapping_examples:
        print(f"    ✅ 성공 매핑 사례 (처음 5개):")
        for i, (json_name, file_path, method) in enumerate(mapping_examples[:5]):
            print(f"      {i+1}. [{method}] {json_name} → {os.path.basename(file_path)}")
    
    # 카테고리별 규칙 생성
    category_locations = file_patterns.get('category_locations', {})
    for category, locations in category_locations.items():
        # 가장 빈번한 위치를 대표 경로로 선택
        location_counts = defaultdict(int)
        for loc in locations:
            location_counts[loc] += 1
        
        if location_counts:
            most_common_location = max(location_counts, key=location_counts.get)
            rules['category_rules'][category] = most_common_location
            print(f"    {category} → {most_common_location}")
    
    # 폴백 패턴 생성
    path_templates = file_patterns.get('path_templates', [])
    rules['fallback_patterns'] = sorted(set(path_templates), key=lambda x: x.count('{'))
    
    print(f"  📋 생성된 규칙:")
    print(f"    - 직접 매핑: {len(rules['direct_lookup'])}개")
    print(f"    - 카테고리 규칙: {len(rules['category_rules'])}개")
    print(f"    - 폴백 패턴: {len(rules['fallback_patterns'])}개")
    
    return rules

def test_mapping_accuracy_enhanced(rules, json_patterns, file_patterns, dataset_name):
    """향상된 매핑 규칙의 정확도 테스트"""
    print(f"\n🧪 {dataset_name} 향상된 매핑 규칙 정확도 테스트:")
    
    if not json_patterns or not rules:
        print("  ❌ 테스트 데이터 부족")
        return
    
    test_files = json_patterns['file_names']  # 전체 테스트
    success_count = 0
    failure_cases = []
    success_cases = []
    
    for test_file in test_files:
        found_path = None
        
        # 1. 직접 룩업 시도
        if test_file in rules['direct_lookup']:
            found_path = rules['direct_lookup'][test_file]
        else:
            # 2. 카테고리 기반 검색
            category = extract_category_from_filename(test_file, dataset_name)
            if category and category in rules['category_rules']:
                category_dir = rules['category_rules'][category]
                # 베이스 경로에서 카테고리 디렉토리 찾기
                base_path = list(file_patterns.get('filename_to_path', {}).values())[0] if file_patterns.get('filename_to_path') else ""
                if base_path:
                    base_path = os.path.dirname(base_path)
                    for ext in ['.png', '.jpg', '.jpeg']:
                        candidate_path = os.path.join(base_path, category_dir, f"{test_file}{ext}")
                        if os.path.exists(candidate_path):
                            found_path = candidate_path
                            break
        
        if found_path and os.path.exists(found_path):
                success_count += 1
                success_cases.append((test_file, found_path))
        else:
            failure_cases.append(test_file)
    
    accuracy = success_count / len(test_files) * 100
    print(f"  📊 향상된 매핑 정확도: {accuracy:.1f}% ({success_count}/{len(test_files)})")
    
    if success_cases:
        print(f"  ✅ 성공 사례 (처음 3개):")
        for i, (test_file, found_path) in enumerate(success_cases[:3]):
            print(f"    {i+1}. {test_file} → {os.path.basename(found_path)}")
    
    if failure_cases:
        print(f"  ❌ 실패 사례 (처음 5개):")
        for case in failure_cases[:5]:
            print(f"    - {case}")

def dataset_name_to_english(dataset_name):
    """데이터셋 이름을 영어로 변환"""
    name_mapping = {
        '손글씨_Train': 'handwriting_train',
        '손글씨_Valid': 'handwriting_valid', 
        'OCR공공_Train': 'ocr_public_train',
        'OCR공공_Valid': 'ocr_public_valid',
        '금융물류_Train': 'finance_logistics_train',
        '금융물류_Valid': 'finance_logistics_valid',
        '공공행정_Train': 'public_admin_train',
        '공공행정_Train_Partly': 'public_admin_train_partly',
        '공공행정_Valid': 'public_admin_valid',
        'TextInWild': 'text_in_wild'
    }
    return name_mapping.get(dataset_name, dataset_name.lower().replace(' ', '_'))

def generate_optimized_lookup_function(rules, dataset_name):
    """최적화된 조회 함수 코드 생성 (수정된 버전)"""
    print(f"\n🚀 {dataset_name} 최적화 함수 코드 생성:")
    
    english_name = dataset_name_to_english(dataset_name)
    function_name = f"lookup_{english_name}"
    
    # 함수 코드 생성
    code = f'''
def {function_name}(filename, base_path):
    """
    {dataset_name} 파일명을 실제 경로로 변환하는 최적화된 함수
    Generated by ftp_tree_viewer.py
    """
    # 1. 직접 매핑 (가장 빠름)
    direct_mappings = {{
'''
    
    # 직접 매핑 딕셔너리 추가 (전체)
    for filename, path in rules['direct_lookup'].items():
        code += f'        "{filename}": "{path}",\n'
    
    code += f'''    }}
    
    if filename in direct_mappings:
        return direct_mappings[filename]
    
    # 2. 카테고리 기반 매핑
    category_rules = {{
'''
    
    # 카테고리 규칙 추가
    for category, location in rules['category_rules'].items():
        code += f'        "{category}": "{location}",\n'
    
    code += f'''    }}
    
    # 카테고리 추출
    category = None'''
    
    # 카테고리 추출 로직 추가
    if "손글씨" in dataset_name:
        code += '''
    for cat in ["4TO", "4PO", "4PR", "4TR"]:
        if f"_{cat}_" in filename:
            category = cat
            break'''
    elif "OCR공공" in dataset_name:
        code += '''
    for cat in ["AF", "CST", "CT", "DI", "EN", "EV", "WF"]:
        if f"{cat}_" in filename or filename.startswith(cat):
            category = cat
            break'''
    elif "금융물류" in dataset_name:
        code += '''
    for cat in ["BL", "PL", "NV"]:
        if f"_{cat}_" in filename:
            category = cat
            break
    if "_F_" in filename:
        category = "F"'''
    
    code += '''
    
    # 카테고리 기반 경로 생성
    if category and category in category_rules:
        category_dir = category_rules[category]
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate_path = os.path.join(base_path, category_dir, f"{filename}{ext}")
            if os.path.exists(candidate_path):
                return candidate_path
    
    # 3. 폴백: 전체 스캔 (느림)
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                if filename in file or file.startswith(filename):
                    return os.path.join(root, file)
    
    return None
'''
    
    rules['optimization_code'] = code
    
    print(f"  ✅ 함수 코드 생성 완료: {function_name}")
    
    # 파일로 저장 (디렉토리 생성)
    output_dir = "FAST"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/optimized_lookup_{english_name}.py"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"#!/usr/bin/env python\n# -*- coding: utf-8 -*-\n")
            f.write(f'"""{dataset_name} 최적화된 파일 조회 함수"""\n\n')
            f.write("import os\n\n")
            f.write(code)
        
        print(f"  💾 코드 저장 성공: {output_file}")
    except Exception as e:
        print(f"  ❌ 코드 저장 실패: {e}")

def main():
    """메인 분석 함수 - 모든 데이터셋 분석"""
    print("🚀 OCR 데이터셋 완전 매핑 규칙 분석 및 최적화 함수 생성 (전체 데이터셋)")
    print("=" * 60)
    
    # FTP 마운트 확인
    if not os.path.exists(FTP_BASE_PATH):
        print("❌ FTP 마운트 없음")
        return
    
    print("✅ FTP 마운트 확인 완료")
    
    # 분석할 모든 데이터셋 (train/valid 포함)
    datasets = [
        # 손글씨 OCR 데이터
        {
            'name': '손글씨_Train',
            'json_path': f"{MERGED_JSON_PATH}/handwriting_train_merged.json",
            'base_path': f"{FTP_BASE_PATH}/053.대용량 손글씨 OCR 데이터/01.데이터/1.Training/원천데이터"
        },
        {
            'name': '손글씨_Valid',
            'json_path': f"{MERGED_JSON_PATH}/handwriting_valid_merged.json",
            'base_path': f"{FTP_BASE_PATH}/053.대용량 손글씨 OCR 데이터/01.데이터/2.Validation/원천데이터"
        },
        
        # OCR 공공 데이터
        {
            'name': 'OCR공공_Train',
            'json_path': f"{MERGED_JSON_PATH}/ocr_public_train_merged.json",
            'base_path': f"{FTP_BASE_PATH}/023.OCR 데이터(공공)/01-1.정식개방데이터/Training/01.원천데이터"
        },
        {
            'name': 'OCR공공_Valid',
            'json_path': f"{MERGED_JSON_PATH}/ocr_public_valid_merged.json",
            'base_path': f"{FTP_BASE_PATH}/023.OCR 데이터(공공)/01-1.정식개방데이터/Validation/01.원천데이터"
        },
        
        # 금융물류 데이터
        {
            'name': '금융물류_Train',
            'json_path': f"{MERGED_JSON_PATH}/finance_logistics_train_merged.json",
            'base_path': f"{FTP_BASE_PATH}/025.OCR 데이터(금융 및 물류)/01-1.정식개방데이터/Training/01.원천데이터"
        },
        {
            'name': '금융물류_Valid',
            'json_path': f"{MERGED_JSON_PATH}/finance_logistics_valid_merged.json",
            'base_path': f"{FTP_BASE_PATH}/025.OCR 데이터(금융 및 물류)/01-1.정식개방데이터/Validation/01.원천데이터"
        },
        
        # 공공행정문서 OCR
        {
            'name': '공공행정_Train',
            'json_path': f"{MERGED_JSON_PATH}/public_admin_train_merged.json",
            'base_path': f"{FTP_BASE_PATH}/공공행정문서 OCR/Training"
        },
        {
            'name': '공공행정_Train_Partly',
            'json_path': f"{MERGED_JSON_PATH}/public_admin_train_partly_merged.json",
            'base_path': f"{FTP_BASE_PATH}/공공행정문서 OCR/Training"
        },
        {
            'name': '공공행정_Valid',
            'json_path': f"{MERGED_JSON_PATH}/public_admin_valid_merged.json",
            'base_path': f"{FTP_BASE_PATH}/공공행정문서 OCR/Validation"
        },
        
        # Text in the Wild (한국어글자체)
        {
            'name': 'TextInWild',
            'json_path': f"{MERGED_JSON_PATH}/textinthewild_data_info.json",
            'base_path': f"{FTP_BASE_PATH}/13.한국어글자체/04. Text in the wild_230209_add"
        }
    ]
    
    # 전체 매핑 규칙
    all_mapping_rules = {}
    successful_datasets = []
    failed_datasets = []
    
    # 각 데이터셋 분석
    for i, dataset in enumerate(datasets):
        print(f"\n{'='*60}")
        print(f"🎯 진행상황: {i+1}/{len(datasets)} - {dataset['name']}")
        print(f"{'='*60}")
        
        # 이미 생성된 lookup 파일 확인
        english_name = dataset_name_to_english(dataset['name'])
        lookup_file = f"FAST/optimized_lookup_{english_name}.py"
        
        if os.path.exists(lookup_file):
            print(f"⏭️ {dataset['name']}: 이미 생성된 lookup 파일 스킵 - {lookup_file}")
            successful_datasets.append(dataset['name'])
            continue
        
        if os.path.exists(dataset['json_path']):
            try:
                mapping_rules = analyze_complete_mapping_rules(
                    dataset['json_path'],
                    dataset['base_path'],
                    dataset['name']
                )
                all_mapping_rules[dataset['name']] = mapping_rules
                
                # 성공 여부 판단
                if mapping_rules and mapping_rules.get('direct_lookup'):
                    successful_datasets.append(dataset['name'])
                    print(f"✅ {dataset['name']}: 매핑 성공!")
                else:
                    failed_datasets.append(dataset['name'])
                    print(f"❌ {dataset['name']}: 매핑 실패")
                    
            except Exception as e:
                print(f"❌ {dataset['name']} 분석 중 오류: {e}")
                failed_datasets.append(dataset['name'])
        else:
            print(f"⚠️ {dataset['name']} JSON 파일 없음: {dataset['json_path']}")
            failed_datasets.append(dataset['name'])
    
    # 종합 결과 출력
    print(f"\n{'='*80}")
    print("🎉 전체 데이터셋 매핑 규칙 분석 완료!")
    print(f"{'='*80}")
    
    print(f"\n📊 분석 결과 요약:")
    print(f"   📈 성공: {len(successful_datasets)}개 데이터셋")
    print(f"   📉 실패: {len(failed_datasets)}개 데이터셋")
    print(f"   📋 전체: {len(datasets)}개 데이터셋")
    
    if successful_datasets:
        print(f"\n✅ 성공한 데이터셋:")
        for name in successful_datasets:
            rules = all_mapping_rules.get(name, {})
            direct_count = len(rules.get('direct_lookup', {}))
            category_count = len(rules.get('category_rules', {}))
            print(f"   🎯 {name}: 직접매핑 {direct_count}개, 카테고리규칙 {category_count}개")
    
    if failed_datasets:
        print(f"\n❌ 실패한 데이터셋:")
        for name in failed_datasets:
            print(f"   💥 {name}")
    
    # 전체 최적화 함수 통합 코드 생성
    generate_unified_optimization_code(all_mapping_rules)
    
    print(f"\n📋 다음 단계:")
    print(f"   1. ✅ 전체 데이터셋 매핑 규칙 분석 완료")
    print(f"   2. 🔄 create_all_datasets_500_clean.py에 최적화 함수 통합")
    print(f"   3. 🚀 bigjson + 병렬처리 최적화 적용")
    print(f"   4. 🧪 실제 LMDB 생성 테스트")

def generate_unified_optimization_code(all_mapping_rules):
    """모든 데이터셋의 최적화 함수를 통합한 코드 생성"""
    print(f"\n🚀 통합 최적화 함수 코드 생성:")
    
    unified_code = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
모든 OCR 데이터셋의 통합 최적화 조회 함수
Generated by ftp_tree_viewer.py
"""

import os

class OCRDatasetOptimizedLookup:
    """모든 OCR 데이터셋에 대한 최적화된 파일 조회 클래스"""
    
    def __init__(self):
        """초기화 - 모든 데이터셋의 직접 매핑 테이블 로드"""
        self.dataset_mappings = {
'''
    
    # 각 데이터셋별 직접 매핑 추가
    for dataset_name, rules in all_mapping_rules.items():
        if rules and rules.get('direct_lookup'):
            unified_code += f'            "{dataset_name}": {{\n'
            
            # 직접 매핑 딕셔너리 추가 (전체)
            direct_mappings = rules.get('direct_lookup', {})
            for filename, path in direct_mappings.items():
                unified_code += f'                "{filename}": "{path}",\n'
            
            unified_code += f'            }},\n'
    
    unified_code += '''        }
        
        self.category_rules = {
'''
    
    # 각 데이터셋별 카테고리 규칙 추가
    for dataset_name, rules in all_mapping_rules.items():
        if rules and rules.get('category_rules'):
            unified_code += f'            "{dataset_name}": {{\n'
            
            category_rules = rules.get('category_rules', {})
            for category, location in category_rules.items():
                unified_code += f'                "{category}": "{location}",\n'
            
            unified_code += f'            }},\n'
    
    unified_code += '''        }
    
    def lookup_file(self, filename, dataset_name, base_path):
        """통합 파일 조회 함수"""
        # 1. 직접 매핑 시도
        dataset_mappings = self.dataset_mappings.get(dataset_name, {})
        if filename in dataset_mappings:
            return dataset_mappings[filename]
        
        # 2. 확장자 추가 시도
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = f"{filename}{ext}"
            if candidate in dataset_mappings:
                return dataset_mappings[candidate]
        
        # 3. 카테고리 기반 조회
        category = self.extract_category(filename, dataset_name)
        if category:
            dataset_category_rules = self.category_rules.get(dataset_name, {})
            if category in dataset_category_rules:
                category_dir = dataset_category_rules[category]
                for ext in ['.png', '.jpg', '.jpeg']:
                    candidate_path = os.path.join(base_path, category_dir, f"{filename}{ext}")
                    if os.path.exists(candidate_path):
                        return candidate_path
        
        # 4. 폴백: 부분 매칭
        filename_base = filename.replace('IMG_OCR_53_', '').replace('IMG_OCR_', '')
        for mapped_file, mapped_path in dataset_mappings.items():
            if filename_base in mapped_file or mapped_file in filename_base:
                return mapped_path
        
        # 5. 최후의 수단: 전체 스캔
        return self.fallback_scan(filename, base_path)
    
    def extract_category(self, filename, dataset_name):
        """파일명에서 카테고리 추출"""
        if "손글씨" in dataset_name:
            for cat in ["4TO", "4PO", "4PR", "4TR"]:
                if f"_{cat}_" in filename:
                    return cat
        elif "OCR공공" in dataset_name:
            for cat in ["AF", "CST", "CT", "DI", "EN", "EV", "WF"]:
                if f"{cat}_" in filename or filename.startswith(cat):
                    return cat
        elif "금융물류" in dataset_name:
            for cat in ["BL", "PL", "NV"]:
                if f"_{cat}_" in filename:
                    return cat
            if "_F_" in filename:
                return "F"
        elif "공공행정" in dataset_name:
            return "ADMIN"
        return None
    
    def fallback_scan(self, filename, base_path):
        """폴백: 전체 디렉토리 스캔"""
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    if filename in file or file.startswith(filename):
                        return os.path.join(root, file)
        return None

# 전역 인스턴스
ocr_lookup = OCRDatasetOptimizedLookup()

def optimized_file_lookup(filename, dataset_name, base_path):
    """모든 데이터셋에 대한 통합 최적화 조회 함수"""
    return ocr_lookup.lookup_file(filename, dataset_name, base_path)
'''
    
    # 파일 저장
    output_dir = "FAST"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/unified_ocr_lookup_optimizer.py"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(unified_code)
        
        print(f"  ✅ 통합 최적화 클래스 생성 완료")
        print(f"  💾 파일 저장: {output_file}")
        print(f"  📊 포함된 데이터셋: {len(all_mapping_rules)}개")
        
        # 통계 출력
        total_direct_mappings = sum(len(rules.get('direct_lookup', {})) for rules in all_mapping_rules.values())
        total_category_rules = sum(len(rules.get('category_rules', {})) for rules in all_mapping_rules.values())
        
        print(f"  📈 전체 직접 매핑: {total_direct_mappings:,}개")
        print(f"  📈 전체 카테고리 규칙: {total_category_rules}개")
        
    except Exception as e:
        print(f"  ❌ 통합 코드 저장 실패: {e}")

if __name__ == '__main__':
    main() 