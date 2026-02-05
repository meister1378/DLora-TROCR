#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pickle
import gzip
import time
from pathlib import Path
from tqdm import tqdm

def create_ocr_public_lookup():
    """ocr_public_train_merged.json에서 lookup 딕셔너리 생성"""
    
    json_file = "json_merged/ocr_public_train_merged.json"
    
    if not os.path.exists(json_file):
        print(f"❌ {json_file} 파일을 찾을 수 없습니다!")
        return
    
    print(f"🚀 {json_file}에서 lookup 딕셔너리 생성 중...")
    
    # JSON 파일 로드
    start_time = time.time()
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    load_time = time.time() - start_time
    print(f"📁 JSON 로드 완료: {load_time:.2f}초")
    
    # 구조 확인
    if 'images' not in data or 'annotations' not in data:
        print("❌ 올바른 JSON 구조가 아닙니다!")
        return
    
    print(f"📊 이미지 개수: {len(data['images']):,}")
    print(f"📊 어노테이션 개수: {len(data['annotations']):,}")
    
    # lookup 딕셔너리 생성
    lookup_dict = {}
    
    print("🔍 이미지 정보를 lookup 딕셔너리로 변환 중...")
    
    # 이미지 정보 처리
    for img in tqdm(data['images'], desc="이미지 처리"):
        if 'file_name' in img:
            filename = img['file_name']
            
            # 파일명에서 확장자 제거
            base_name = os.path.splitext(filename)[0]
            
            # 여러 형태로 저장 (확장자 유무)
            lookup_dict[base_name] = {
                'file_name': filename,
                'width': img.get('width', 0),
                'height': img.get('height', 0),
                'id': img.get('id', 0),
                'dataset': img.get('dataset', 'ocr_public_train'),
                'sub_dataset': img.get('sub_dataset', ''),
                'original_json_path': img.get('original_json_path', ''),
                'type': 'image'
            }
            
            # 확장자가 있는 경우도 저장
            if '.' in filename:
                lookup_dict[filename] = lookup_dict[base_name]
    
    # 어노테이션 정보도 추가 (이미지 ID로 매핑)
    print("🔍 어노테이션 정보 처리 중...")
    
    annotation_lookup = {}
    for ann in tqdm(data['annotations'], desc="어노테이션 처리"):
        image_id = ann.get('image_id', 0)
        if image_id not in annotation_lookup:
            annotation_lookup[image_id] = []
        annotation_lookup[image_id].append(ann)
    
    # 이미지 lookup에 어노테이션 정보 추가
    for img in data['images']:
        img_id = img.get('id', 0)
        if img_id in annotation_lookup:
            base_name = os.path.splitext(img.get('file_name', ''))[0]
            if base_name in lookup_dict:
                lookup_dict[base_name]['annotations'] = annotation_lookup[img_id]
    
    # 통계 출력
    print(f"\n📊 Lookup 딕셔너리 생성 완료:")
    print(f"   🔑 키 개수: {len(lookup_dict):,}")
    print(f"   🖼️ 이미지 정보: {len([v for v in lookup_dict.values() if v.get('type') == 'image']):,}")
    print(f"   📝 어노테이션 포함: {len([v for v in lookup_dict.values() if 'annotations' in v]):,}")
    
    # 파일 저장
    print("\n💾 파일 저장 중...")
    
    # 1. 일반 Pickle 파일
    pickle_file = "lookup_ocr_public_train.pkl"
    start_time = time.time()
    with open(pickle_file, 'wb') as f:
        pickle.dump(lookup_dict, f)
    pickle_time = time.time() - start_time
    
    pickle_size = os.path.getsize(pickle_file) / (1024 * 1024)
    print(f"   📁 {pickle_file}: {pickle_size:.1f}MB ({pickle_time:.2f}초)")
    
    # 2. 압축된 Pickle 파일
    pickle_gz_file = "lookup_ocr_public_train.pkl.gz"
    start_time = time.time()
    with gzip.open(pickle_gz_file, 'wb') as f:
        pickle.dump(lookup_dict, f)
    gz_time = time.time() - start_time
    
    gz_size = os.path.getsize(pickle_gz_file) / (1024 * 1024)
    print(f"   📁 {pickle_gz_file}: {gz_size:.1f}MB ({gz_time:.2f}초)")
    
    # 3. Python 모듈 파일 (선택적)
    py_file = "optimized_lookup_ocr_public_train.py"
    start_time = time.time()
    
    with open(py_file, 'w', encoding='utf-8') as f:
        f.write("#!/usr/bin/env python3\n")
        f.write("# -*- coding: utf-8 -*-\n\n")
        f.write("def lookup_ocr_public_train(filename, base_path):\n")
        f.write("    \"\"\"OCR 공개 데이터셋 lookup 함수\"\"\"\n")
        f.write("    \n")
        f.write("    # lookup 딕셔너리\n")
        f.write("    lookup_dict = {\n")
        
        # 딕셔너리 내용을 Python 코드로 변환
        count = 0
        for key, value in lookup_dict.items():
            if count < 1000:  # 처음 1000개만 (파일 크기 제한)
                f.write(f"        '{key}': {repr(value)},\n")
                count += 1
            else:
                f.write(f"        # ... {len(lookup_dict) - 1000}개 더 있음\n")
                break
        
        f.write("    }\n")
        f.write("    \n")
        f.write("    # 파일명에서 확장자 제거\n")
        f.write("    base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename\n")
        f.write("    \n")
        f.write("    # lookup 시도\n")
        f.write("    if base_name in lookup_dict:\n")
        f.write("        return lookup_dict[base_name]\n")
        f.write("    elif filename in lookup_dict:\n")
        f.write("        return lookup_dict[filename]\n")
        f.write("    else:\n")
        f.write("        return None\n")
    
    py_time = time.time() - start_time
    py_size = os.path.getsize(py_file) / (1024 * 1024)
    print(f"   📁 {py_file}: {py_size:.1f}MB ({py_time:.2f}초)")
    
    # 성능 테스트
    print(f"\n🧪 성능 테스트:")
    
    # 테스트 파일들
    test_files = list(lookup_dict.keys())[:5]
    
    # Pickle 로드 테스트
    start_time = time.time()
    with open(pickle_file, 'rb') as f:
        test_lookup = pickle.load(f)
    pickle_load_time = time.time() - start_time
    
    # 압축 Pickle 로드 테스트
    start_time = time.time()
    with gzip.open(pickle_gz_file, 'rb') as f:
        test_lookup_gz = pickle.load(f)
    gz_load_time = time.time() - start_time
    
    print(f"   📁 Pickle 로드: {pickle_load_time:.4f}초")
    print(f"   📁 압축 Pickle 로드: {gz_load_time:.4f}초")
    
    # lookup 테스트
    start_time = time.time()
    for test_file in test_files:
        if test_file in test_lookup:
            result = test_lookup[test_file]
    lookup_time = time.time() - start_time
    
    print(f"   🔍 Lookup 속도: {lookup_time:.6f}초 ({len(test_files)}회)")
    
    print(f"\n✅ Lookup 딕셔너리 생성 완료!")
    print(f"💡 사용법:")
    print(f"   📁 {pickle_file} - 일반 Pickle 파일")
    print(f"   📁 {pickle_gz_file} - 압축된 Pickle 파일 (권장)")
    print(f"   📁 {py_file} - Python 모듈 파일")

def test_lookup_function():
    """생성된 lookup 함수 테스트"""
    
    # 압축된 Pickle 파일 로드
    pickle_gz_file = "lookup_ocr_public_train.pkl.gz"
    
    if not os.path.exists(pickle_gz_file):
        print(f"❌ {pickle_gz_file} 파일이 없습니다!")
        return
    
    print(f"🧪 {pickle_gz_file} 테스트 중...")
    
    # 로드
    start_time = time.time()
    with gzip.open(pickle_gz_file, 'rb') as f:
        lookup_dict = pickle.load(f)
    load_time = time.time() - start_time
    
    print(f"📁 로드 시간: {load_time:.4f}초")
    print(f"📊 딕셔너리 크기: {len(lookup_dict):,}개 키")
    
    # 테스트
    test_files = list(lookup_dict.keys())[:10]
    
    print(f"\n🔍 Lookup 테스트:")
    for test_file in test_files:
        result = lookup_dict.get(test_file)
        if result:
            print(f"   ✅ {test_file}: {result.get('file_name', 'N/A')}")
        else:
            print(f"   ❌ {test_file}: 찾을 수 없음")

if __name__ == "__main__":
    print("🚀 OCR 공개 데이터셋 Lookup 생성 도구")
    print("=" * 50)
    
    # lookup 생성
    create_ocr_public_lookup()
    
    print("\n" + "=" * 50)
    
    # 테스트
    test_lookup_function() 