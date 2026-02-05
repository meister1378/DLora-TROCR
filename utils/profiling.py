"""
성능 병목 분석을 위한 프로파일링 유틸리티
"""
import os
import time
import psutil
import torch
import threading
from typing import Dict, Any, Optional
from contextlib import contextmanager
from collections import defaultdict
import json


class PerformanceProfiler:
    """성능 병목 분석을 위한 프로파일러"""
    
    def __init__(self):
        self.enabled = os.environ.get("PROFILE_TRAIN", "0") == "1"
        self.stats = defaultdict(list)
        self.current_step = 0
        self.lock = threading.Lock()
        
    def is_enabled(self) -> bool:
        return self.enabled
        
    @contextmanager
    def profile_section(self, section_name: str):
        """특정 섹션의 성능을 측정하는 컨텍스트 매니저"""
        if not self.enabled:
            yield
            return
            
        start_time = time.perf_counter()
        start_mem = None
        start_gpu_mem = None
        
        try:
            start_mem = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            if torch.cuda.is_available():
                start_gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        except Exception:
            pass
            
        try:
            yield
        finally:
            end_time = time.perf_counter()
            elapsed_ms = (end_time - start_time) * 1000
            
            stats = {
                "step": self.current_step,
                "section": section_name,
                "elapsed_ms": round(elapsed_ms, 2),
                "timestamp": time.time()
            }
            
            # 메모리 사용량 변화
            if start_mem is not None:
                try:
                    end_mem = psutil.Process().memory_info().rss / 1024 / 1024
                    stats["mem_delta_mb"] = round(end_mem - start_mem, 2)
                    stats["mem_current_mb"] = round(end_mem, 2)
                except Exception:
                    pass
                    
            # GPU 메모리 사용량 변화
            if start_gpu_mem is not None:
                try:
                    end_gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
                    stats["gpu_mem_delta_mb"] = round(end_gpu_mem - start_gpu_mem, 2)
                    stats["gpu_mem_current_mb"] = round(end_gpu_mem, 2)
                    stats["gpu_mem_cached_mb"] = round(torch.cuda.memory_reserved() / 1024 / 1024, 2)
                except Exception:
                    pass
                    
            with self.lock:
                self.stats[section_name].append(stats)
                
    def log_batch_stats(self, batch_data: Dict[str, Any], section: str = "batch"):
        """배치 관련 통계 로깅"""
        if not self.enabled:
            return
            
        stats = {
            "step": self.current_step,
            "section": section,
            "timestamp": time.time()
        }
        
        # 배치에서 프로파일링 정보 추출
        for key, value in batch_data.items():
            if key.startswith("collate_") or key.endswith("_ms"):
                stats[key] = value
                
        with self.lock:
            self.stats[section].append(stats)
            
    def increment_step(self):
        """스텝 카운터 증가"""
        self.current_step += 1
        
    def get_summary(self, last_n_steps: Optional[int] = None) -> Dict[str, Any]:
        """성능 통계 요약 반환"""
        if not self.enabled:
            return {}
            
        with self.lock:
            summary = {}
            
            for section_name, section_stats in self.stats.items():
                if not section_stats:
                    continue
                    
                # 최근 N 스텝만 분석
                if last_n_steps:
                    section_stats = [s for s in section_stats if s["step"] >= self.current_step - last_n_steps]
                    
                if not section_stats:
                    continue
                    
                # 시간 통계
                elapsed_times = [s.get("elapsed_ms", 0) for s in section_stats if "elapsed_ms" in s]
                if elapsed_times:
                    summary[f"{section_name}_avg_ms"] = round(sum(elapsed_times) / len(elapsed_times), 2)
                    summary[f"{section_name}_max_ms"] = round(max(elapsed_times), 2)
                    summary[f"{section_name}_min_ms"] = round(min(elapsed_times), 2)
                    summary[f"{section_name}_total_ms"] = round(sum(elapsed_times), 2)
                    
                # 메모리 통계
                mem_deltas = [s.get("mem_delta_mb", 0) for s in section_stats if "mem_delta_mb" in s]
                if mem_deltas:
                    summary[f"{section_name}_avg_mem_delta_mb"] = round(sum(mem_deltas) / len(mem_deltas), 2)
                    summary[f"{section_name}_max_mem_delta_mb"] = round(max(mem_deltas), 2)
                    
                # GPU 메모리 통계
                gpu_deltas = [s.get("gpu_mem_delta_mb", 0) for s in section_stats if "gpu_mem_delta_mb" in s]
                if gpu_deltas:
                    summary[f"{section_name}_avg_gpu_delta_mb"] = round(sum(gpu_deltas) / len(gpu_deltas), 2)
                    summary[f"{section_name}_max_gpu_delta_mb"] = round(max(gpu_deltas), 2)
                    
            return summary
            
    def print_summary(self, last_n_steps: Optional[int] = 50):
        """성능 통계 요약 출력"""
        if not self.enabled:
            return
            
        summary = self.get_summary(last_n_steps)
        if not summary:
            return
            
        print(f"\n=== Performance Summary (last {last_n_steps or 'all'} steps) ===")
        
        # 섹션별로 그룹화하여 출력
        sections = set()
        for key in summary.keys():
            section = key.split('_')[0]
            sections.add(section)
            
        for section in sorted(sections):
            section_data = {k: v for k, v in summary.items() if k.startswith(section)}
            if section_data:
                print(f"\n[{section.upper()}]")
                for key, value in sorted(section_data.items()):
                    clean_key = key.replace(f"{section}_", "")
                    print(f"  {clean_key}: {value}")
                    
    def save_detailed_stats(self, filepath: str):
        """상세 통계를 JSON 파일로 저장"""
        if not self.enabled:
            return
            
        with self.lock:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(dict(self.stats), f, indent=2, ensure_ascii=False)
                
    def clear_stats(self):
        """통계 초기화"""
        with self.lock:
            self.stats.clear()
            self.current_step = 0


# 전역 프로파일러 인스턴스
global_profiler = PerformanceProfiler()


