#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from collections import defaultdict

def check_file_matching():
    """JSON 파일의 파일명과 실제 FTP 서버의 파일을 비교"""
    
    # 경로 설정
    json_path = "/home/mango/ocr_test/FAST/json_merged/finance_logistics_train_merged.json"
    ftp_base = "/run/user/0/gvfs/ftp:host=172.30.1.226/Y:\\ocr_dataset/025.OCR 데이터(금융 및 물류)/01-1.정식개방데이터"
    
    print("🔍 JSON 파일과 실제 파일 매칭 확인")
    print("=" * 60)
    
    # 1. JSON 파일에서 파일명 패턴 분석 (처음 부분만)
    print("📊 JSON 파일 분석 중...")
    
    # JSON 파일의 처음 부분만 읽어서 구조 확인
    with open(json_path, 'r', encoding='utf-8') as f:
        # 처음 1000줄만 읽어서 "file_name" 패턴 찾기
        content = ""
        for i, line in enumerate(f):
            if i > 1000:  # 처음 1000줄만
                break
            content += line
            if '"images"' in line and '[' in line:
                break
    
    # "file_name" 패턴 찾기
    import re
    file_name_pattern = r'"file_name":\s*"([^"]+)"'
    file_names = re.findall(file_name_pattern, content)
    
    print(f"📈 JSON 파일에서 발견된 파일명 패턴:")
    print(f"   - 발견된 파일명 수: {len(file_names)}")
    
    # 카테고리별 분류
    categories = defaultdict(list)
    for file_name in file_names[:100]:  # 처음 100개만 분석
        parts = file_name.split('_')
        if len(parts) >= 4:
            category = parts[3]  # BL, PL, NV, CO, ET 등
            categories[category].append(file_name)
    
    print(f"   - 발견된 카테고리: {list(categories.keys())}")
    for cat, files in categories.items():
        print(f"   - {cat}: {len(files)}개")
        if len(files) <= 5:
            print(f"     예시: {files}")
        else:
            print(f"     예시: {files[:3]} ... {files[-2:]}")
    
    print("\n" + "=" * 60)
    
    # 2. 실제 FTP 서버에서 파일 확인
    print("📁 FTP 서버 파일 확인 중...")
    
    # Training 폴더에서 실제 파일 확인
    training_path = f"{ftp_base}/Training/01.원천데이터"
    actual_files = defaultdict(list)
    
    if os.path.exists(training_path):
        for root, dirs, files in os.walk(training_path):
            for file in files:
                if file.endswith('.png'):
                    # 파일명에서 카테고리 추출
                    parts = file.replace('.png', '').split('_')
                    if len(parts) >= 4:
                        category = parts[3]
                        actual_files[category].append(file)
    
    print(f"📈 FTP 서버 파일 분석 결과:")
    print(f"   - Training 폴더: {training_path}")
    print(f"   - 발견된 카테고리: {list(actual_files.keys())}")
    for cat, files in actual_files.items():
        print(f"   - {cat}: {len(files)}개")
        if len(files) <= 5:
            print(f"     예시: {files}")
        else:
            print(f"     예시: {files[:3]} ... {files[-2:]}")
    
    print("\n" + "=" * 60)
    
    # 3. 매칭 확인
    print("🔍 JSON vs 실제 파일 매칭 확인")
    
    json_categories = set(categories.keys())
    actual_categories = set(actual_files.keys())
    
    print(f"JSON에만 있는 카테고리: {json_categories - actual_categories}")
    print(f"실제에만 있는 카테고리: {actual_categories - json_categories}")
    print(f"공통 카테고리: {json_categories & actual_categories}")
    
    # 4. 공통 카테고리에서 실제 매칭 확인
    common_categories = json_categories & actual_categories
    if common_categories:
        print(f"\n📋 공통 카테고리 매칭 확인:")
        for cat in common_categories:
            json_files = set(categories[cat])
            actual_files_set = set(actual_files[cat])
            
            # 확장자 추가해서 비교
            json_files_with_ext = {f"{f}.png" for f in json_files}
            
            matched = json_files_with_ext & actual_files_set
            print(f"   {cat}: {len(matched)}/{len(json_files)} 매칭")
            
            if len(matched) > 0:
                print(f"     매칭 예시: {list(matched)[:3]}")
            if len(json_files) > len(matched):
                missing = json_files_with_ext - actual_files_set
                print(f"     누락 예시: {list(missing)[:3]}")
    
    # 5. 폴더 구조 확인
    print(f"\n📂 폴더 구조 확인:")
    if os.path.exists(training_path):
        for item in os.listdir(training_path):
            item_path = os.path.join(training_path, item)
            if os.path.isdir(item_path):
                file_count = len([f for f in os.listdir(item_path) if f.endswith('.png')])
                print(f"   {item}: {file_count}개 파일")

if __name__ == "__main__":
    check_file_matching() 