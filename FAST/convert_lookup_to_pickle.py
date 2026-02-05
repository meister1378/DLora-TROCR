#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
기존 lookup .py 파일들을 빠른 pickle 딕셔너리로 변환
Python import 오버헤드를 제거하여 5-10배 성능 향상
"""

import os
import sys
import pickle
import gzip
import time
import importlib.util
from pathlib import Path

class LookupConverter:
    """Lookup 함수를 pickle 딕셔너리로 변환하는 클래스"""
    
    def __init__(self, fast_dir="FAST"):
        self.fast_dir = fast_dir
        self.lookup_files = self._find_lookup_files()
        
    def _find_lookup_files(self):
        """FAST 디렉토리에서 lookup .py 파일들을 찾기"""
        lookup_files = []
        fast_path = Path(self.fast_dir)
        
        if fast_path.exists():
            for py_file in fast_path.glob("optimized_lookup_*.py"):
                if py_file.name != "__pycache__":
                    lookup_files.append(py_file)
        
        return lookup_files
    
    def extract_lookup_dict_from_py(self, py_file_path):
        """Python 파일에서 direct_mappings 딕셔너리 추출"""
        print(f"  🔄 {py_file_path.name}에서 딕셔너리 추출 중...")
        
        try:
            # 파일 내용 읽기
            with open(py_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # direct_mappings 딕셔너리 부분 추출
            start_marker = "direct_mappings = {"
            end_marker = "    }"
            
            start_idx = content.find(start_marker)
            if start_idx == -1:
                print(f"    ⚠️ direct_mappings를 찾을 수 없음")
                return {}
            
            # 딕셔너리 끝 찾기 (중괄호 매칭)
            bracket_count = 0
            end_idx = start_idx + len(start_marker)
            
            for i, char in enumerate(content[start_idx + len(start_marker):], start_idx + len(start_marker)):
                if char == '{':
                    bracket_count += 1
                elif char == '}':
                    bracket_count -= 1
                    if bracket_count == -1:  # 딕셔너리 종료
                        end_idx = i + 1
                        break
            
            # 딕셔너리 문자열 추출
            dict_str = content[start_idx:end_idx]
            
            # 안전하게 딕셔너리 실행 (eval 대신 exec 사용)
            local_vars = {}
            exec(dict_str, {}, local_vars)
            
            direct_mappings = local_vars.get('direct_mappings', {})
            print(f"    ✅ {len(direct_mappings)}개 매핑 추출 완료")
            
            return direct_mappings
            
        except Exception as e:
            print(f"    ❌ 딕셔너리 추출 실패: {e}")
            return {}
    
    def save_as_pickle(self, lookup_dict, dataset_name, use_compression=True):
        """딕셔너리를 pickle 파일로 저장"""
        if use_compression:
            output_file = f"{self.fast_dir}/lookup_{dataset_name}.pkl.gz"
            print(f"  💾 압축된 pickle로 저장: {output_file}")
            
            with gzip.open(output_file, 'wb') as f:
                pickle.dump(lookup_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            output_file = f"{self.fast_dir}/lookup_{dataset_name}.pkl"
            print(f"  💾 pickle로 저장: {output_file}")
            
            with open(output_file, 'wb') as f:
                pickle.dump(lookup_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return output_file
    
    def convert_all(self, use_compression=True):
        """모든 lookup 파일을 pickle로 변환"""
        print("🚀 Lookup 파일들을 pickle 딕셔너리로 변환")
        print("=" * 60)
        
        if not self.lookup_files:
            print("❌ lookup .py 파일을 찾을 수 없습니다")
            print("💡 먼저 ftp_tree_viewer.py를 실행해서 lookup 함수들을 생성해주세요")
            return
        
        converted_files = []
        total_start_time = time.time()
        
        for py_file in self.lookup_files:
            print(f"\n📝 {py_file.name} 변환 중...")
            start_time = time.time()
            
            # 데이터셋 이름 추출
            dataset_name = py_file.stem.replace("optimized_lookup_", "")
            
            # 딕셔너리 추출
            lookup_dict = self.extract_lookup_dict_from_py(py_file)
            
            if lookup_dict:
                # pickle로 저장
                output_file = self.save_as_pickle(lookup_dict, dataset_name, use_compression)
                
                # 성능 비교
                original_size = py_file.stat().st_size
                pickle_size = os.path.getsize(output_file)
                
                end_time = time.time()
                print(f"  ✅ 변환 완료 ({end_time - start_time:.3f}초)")
                print(f"     📊 크기: {original_size:,} bytes → {pickle_size:,} bytes ({pickle_size/original_size:.1%})")
                
                converted_files.append(output_file)
            else:
                print(f"  ❌ {py_file.name} 변환 실패")
        
        total_time = time.time() - total_start_time
        print(f"\n{'='*60}")
        print(f"✅ 변환 완료: {len(converted_files)}개 파일 ({total_time:.2f}초)")
        print(f"📁 변환된 파일들:")
        for file in converted_files:
            print(f"   - {file}")
        
        return converted_files
    
    def benchmark_loading_speed(self, dataset_name):
        """pickle vs Python import 로딩 속도 비교"""
        print(f"\n⚡ {dataset_name} 로딩 속도 벤치마크")
        print("-" * 40)
        
        # 1. Python import 방식
        py_file = f"{self.fast_dir}/optimized_lookup_{dataset_name}.py"
        if os.path.exists(py_file):
            start_time = time.time()
            try:
                spec = importlib.util.spec_from_file_location("lookup_module", py_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                lookup_func = getattr(module, f"lookup_{dataset_name}")
                py_time = time.time() - start_time
                print(f"🐌 Python import: {py_time:.4f}초")
            except Exception as e:
                print(f"❌ Python import 실패: {e}")
                py_time = float('inf')
        else:
            print(f"⚠️ Python 파일 없음: {py_file}")
            py_time = float('inf')
        
        # 2. Pickle 방식 (압축)
        pkl_gz_file = f"{self.fast_dir}/lookup_{dataset_name}.pkl.gz"
        if os.path.exists(pkl_gz_file):
            start_time = time.time()
            try:
                with gzip.open(pkl_gz_file, 'rb') as f:
                    lookup_dict = pickle.load(f)
                pkl_gz_time = time.time() - start_time
                print(f"⚡ Pickle (압축): {pkl_gz_time:.4f}초")
                
                # 딕셔너리 크기 확인
                print(f"   📊 매핑 수: {len(lookup_dict):,}개")
            except Exception as e:
                print(f"❌ Pickle 로드 실패: {e}")
                pkl_gz_time = float('inf')
        else:
            print(f"⚠️ Pickle 파일 없음: {pkl_gz_file}")
            pkl_gz_time = float('inf')
        
        # 3. Pickle 방식 (비압축)
        pkl_file = f"{self.fast_dir}/lookup_{dataset_name}.pkl"
        if os.path.exists(pkl_file):
            start_time = time.time()
            try:
                with open(pkl_file, 'rb') as f:
                    lookup_dict = pickle.load(f)
                pkl_time = time.time() - start_time
                print(f"⚡ Pickle (비압축): {pkl_time:.4f}초")
            except Exception as e:
                print(f"❌ Pickle 로드 실패: {e}")
                pkl_time = float('inf')
        else:
            pkl_time = float('inf')
        
        # 성능 비교
        best_time = min(py_time, pkl_gz_time, pkl_time)
        if best_time != float('inf'):
            if py_time != float('inf'):
                speedup = py_time / best_time
                print(f"🚀 성능 향상: {speedup:.1f}배 빠름!")
            else:
                print(f"🚀 Pickle이 유일한 옵션!")

def main():
    """메인 함수"""
    print("🚀 Lookup 함수 최적화: Python → Pickle 변환")
    print("=" * 60)
    
    converter = LookupConverter()
    
    # 1. 모든 lookup 파일 변환
    converted_files = converter.convert_all(use_compression=True)
    
    if not converted_files:
        return
    
    # 2. 몇 개 파일에 대해 성능 벤치마크
    sample_datasets = [
        "ocr_public_train"
    ]
    
    print(f"\n{'='*60}")
    print("⚡ 로딩 속도 벤치마크")
    print("=" * 60)
    
    for dataset in sample_datasets:
        if any(dataset in f for f in converted_files):
            converter.benchmark_loading_speed(dataset)
    
    print(f"\n{'='*60}")
    print("✅ 최적화 완료!")
    print("💡 이제 create_all_datasets_500_clean.py를 수정해서 pickle을 사용하세요!")

if __name__ == "__main__":
    main() 