#!/usr/bin/env python3
"""
인식(Recognition)용 LMDB를 사전 셔플하여 새로운 LMDB로 저장하는 유틸리티.

지원 스킴:
- recog_simple: image-000000000, label-000000000
- recog_wimg  : wimg-000000000-xxxx, wlab-000000000-xxxx

사용 예시:
  python -u tools/shuffle_lmdb_recog.py \
    --lmdb_paths /path/a.lmdb:/path/b.lmdb \
    --output_root /dest/shuffled_lmdb \
    --seed 42 --commit_interval 2000

주의:
- 매우 큰 LMDB(수백만~수천만 샘플)의 경우 전체 키 스캔에 시간이 걸립니다.
- 출력 LMDB 용량은 입력과 유사하므로 map_size를 입력 data.mdb 크기 기준으로 여유 있게 설정합니다.
"""

import os
import re
import sys
import argparse
import lmdb
import random
from typing import List, Tuple, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def detect_scheme(txn: lmdb.Transaction) -> Tuple[str, dict]:
    """LMDB 키 스킴 감지 및 간단 통계 반환."""
    prefix_counts = {b'image-': 0, b'wimg-': 0, b'gt-': 0, b'label-': 0, b'wlab-': 0}
    cur = txn.cursor()
    limit_scan = 50000
    i = 0
    for k, _ in cur:
        if i >= limit_scan:
            break
        for p in list(prefix_counts.keys()):
            if k.startswith(p):
                prefix_counts[p] += 1
                break
        i += 1

    if prefix_counts[b'image-'] and prefix_counts[b'label-']:
        scheme = 'recog_simple'
    elif prefix_counts[b'wimg-'] and prefix_counts[b'wlab-']:
        scheme = 'recog_wimg'
    elif prefix_counts[b'image-'] and prefix_counts[b'gt-']:
        scheme = 'det'  # 지원하지 않음(사전 셔플 대상 아님)
    else:
        scheme = 'unknown'
    return scheme, prefix_counts


def list_pairs_recog_simple(txn: lmdb.Transaction, num_samples: int, max_samples: int = 0) -> List[bytes]:
    """image- 키를 순회하며 label-이 존재하는 id만 수집. 반환은 raw id 바이트(9~12자리 등)."""
    pairs = []
    cur = txn.cursor()
    cur.first()
    count = 0
    for k, v in cur:
        if not k.startswith(b'image-'):
            continue
        id_part = k[len(b'image-'):]
        lab_key = b'label-' + id_part
        if txn.get(lab_key) is None:
            continue
        pairs.append(id_part)
        count += 1
        if max_samples > 0 and count >= max_samples:
            break
    return pairs


def list_pairs_recog_wimg(txn: lmdb.Transaction, num_samples: int, max_samples: int = 0) -> List[bytes]:
    """wimg- 키를 순회하며 wlab-이 존재하는 id suffix를 수집. 전체 키(접두사 제외) 바이트를 저장."""
    pairs = []
    cur = txn.cursor()
    cur.first()
    count = 0
    for k, v in cur:
        if not k.startswith(b'wimg-'):
            continue
        suffix = k[len(b'wimg-'):]
        lab_key = b'wlab-' + suffix
        if txn.get(lab_key) is None:
            continue
        pairs.append(suffix)
        count += 1
        if max_samples > 0 and count >= max_samples:
            break
    return pairs


def get_num_samples(txn: lmdb.Transaction) -> Optional[int]:
    raw = txn.get(b'num-samples')
    if raw is None:
        return None
    try:
        return int(raw.decode())
    except Exception:
        return None


def guess_map_size_bytes(input_lmdb_path: str, growth_factor: float = 1.2) -> int:
    data_mdb = os.path.join(input_lmdb_path, 'data.mdb')
    lock_mdb = os.path.join(input_lmdb_path, 'lock.mdb')
    size = 0
    if os.path.isfile(data_mdb):
        size += os.path.getsize(data_mdb)
    if os.path.isfile(lock_mdb):
        size += os.path.getsize(lock_mdb)
    if size <= 0:
        # fallback 8GB
        size = 8 * 1024**3
    target = int(size * growth_factor)
    # 최소 1GB
    return max(target, 1 * 1024**3)


def write_shuffled_recog_simple(src_txn: lmdb.Transaction, dst_env: lmdb.Environment, ids: List[bytes], commit_interval: int = 1000) -> int:
    written = 0
    with dst_env.begin(write=True) as wtx:
        wtx.put(b'num-samples', str(len(ids)).encode('utf-8'))
    txn = dst_env.begin(write=True)
    for i, id_part in enumerate(tqdm(ids, desc='Writing(recog_simple)')):
        img_key = b'image-' + id_part
        lab_key = b'label-' + id_part
        img_bytes = src_txn.get(img_key)
        lab_bytes = src_txn.get(lab_key)
        if img_bytes is None or lab_bytes is None:
            continue
        out_id = f"{i+1:09d}".encode('ascii')
        txn.put(b'image-' + out_id, img_bytes)
        txn.put(b'label-' + out_id, lab_bytes)
        written += 1
        if (i + 1) % commit_interval == 0:
            txn.commit()
            txn = dst_env.begin(write=True)
    txn.commit()
    return written


def write_shuffled_recog_wimg(src_txn: lmdb.Transaction, dst_env: lmdb.Environment, suffixes: List[bytes], commit_interval: int = 1000) -> int:
    written = 0
    with dst_env.begin(write=True) as wtx:
        wtx.put(b'num-samples', str(len(suffixes)).encode('utf-8'))
    txn = dst_env.begin(write=True)
    for i, suffix in enumerate(tqdm(suffixes, desc='Writing(recog_wimg)')):
        img_key = b'wimg-' + suffix
        lab_key = b'wlab-' + suffix
        img_bytes = src_txn.get(img_key)
        lab_bytes = src_txn.get(lab_key)
        if img_bytes is None or lab_bytes is None:
            continue
        out_id = f"{i+1:09d}".encode('ascii')
        txn.put(b'image-' + out_id, img_bytes)
        txn.put(b'label-' + out_id, lab_bytes)
        written += 1
        if (i + 1) % commit_interval == 0:
            txn.commit()
            txn = dst_env.begin(write=True)
    txn.commit()
    return written


def shuffle_one_lmdb(input_path: str, output_root: str, seed: int, max_samples: int, commit_interval: int, overwrite: bool) -> None:
    if not os.path.isdir(input_path):
        print(f"[skip] not a directory: {input_path}")
        return
    base = os.path.basename(os.path.normpath(input_path))
    out_path = os.path.join(output_root, base.replace('.lmdb', '') + '_shuffled.lmdb')
    if os.path.exists(out_path):
        if overwrite:
            # 안전하게 기존 폴더 내용을 제거
            for name in os.listdir(out_path):
                try:
                    os.remove(os.path.join(out_path, name))
                except Exception:
                    pass
        else:
            print(f"[skip] exists: {out_path} (use --overwrite to replace)")
            return
    os.makedirs(out_path, exist_ok=True)

    print(f"[open] src: {input_path}")
    src_env = lmdb.open(input_path, readonly=True, lock=False, readahead=False, meminit=False, max_readers=2048)
    with src_env.begin(write=False) as stx:
        scheme, counts = detect_scheme(stx)
        n = get_num_samples(stx)
        print(f"  scheme={scheme} counts={counts} num-samples={n if n is not None else 'unknown'}")

        ids: List[bytes] = []
        # 1) 명시 스킴 처리
        if scheme == 'recog_simple':
            ids = list_pairs_recog_simple(stx, n or 0, max_samples=max_samples)
        elif scheme == 'recog_wimg':
            ids = list_pairs_recog_wimg(stx, n or 0, max_samples=max_samples)
        elif scheme == 'det':
            print(f"  [skip] unsupported scheme for shuffle: {scheme}")
            return
        else:
            # 2) Fallback probe: image-만 보이는 경우라도 실제 label- 페어가 존재하는지 소량 확인
            probed = False
            if counts.get(b'image-', 0) > 0:
                trial = list_pairs_recog_simple(stx, n or 0, max_samples=min(5000, max(1000, n or 1000)))
                if len(trial) > 0:
                    scheme = 'recog_simple'
                    # 전체 수집(또는 --max_samples 적용)
                    ids = list_pairs_recog_simple(stx, n or 0, max_samples=max_samples)
                    probed = True
            if (not probed) and counts.get(b'wimg-', 0) > 0:
                trial = list_pairs_recog_wimg(stx, n or 0, max_samples=min(5000, max(1000, n or 1000)))
                if len(trial) > 0:
                    scheme = 'recog_wimg'
                    # 전체 수집(또는 --max_samples 적용)
                    ids = list_pairs_recog_wimg(stx, n or 0, max_samples=max_samples)
                    probed = True
            if not probed:
                print(f"  [skip] unsupported scheme for shuffle: {scheme}")
                return

    print(f"  collected pairs: {len(ids)}")
    rnd = random.Random(seed)
    rnd.shuffle(ids)

    map_size = guess_map_size_bytes(input_path, growth_factor=1.2)
    print(f"[open] dst: {out_path} (map_size ~ {map_size/1024**3:.2f} GB)")
    dst_env = lmdb.open(out_path, map_size=map_size, subdir=True, lock=True)
    with src_env.begin(write=False) as stx:
        if scheme == 'recog_simple':
            written = write_shuffled_recog_simple(stx, dst_env, ids, commit_interval=commit_interval)
        else:
            written = write_shuffled_recog_wimg(stx, dst_env, ids, commit_interval=commit_interval)
    dst_env.sync()
    dst_env.close()
    src_env.close()
    print(f"  done: written={written} → {out_path}")


def _sum_map_size_bytes(input_paths: List[str], growth_factor: float = 1.2) -> int:
    total = 0
    for p in input_paths:
        total += guess_map_size_bytes(p, growth_factor=1.0)
    # 합산 후 성장 계수 적용
    return int(total * growth_factor)


def merge_round_robin(id_lists: List[List[bytes]], seed: int) -> List[Tuple[int, bytes]]:
    """각 리스트를 내부적으로 셔플한 뒤, 라운드로빈으로 균등 섞기.
    반환: (src_index, id_bytes) 리스트
    """
    rnd = random.Random(seed)
    for lst in id_lists:
        rnd.shuffle(lst)
    positions = [0] * len(id_lists)
    lengths = [len(lst) for lst in id_lists]
    total = sum(lengths)
    merged: List[Tuple[int, bytes]] = []
    idx = 0
    while len(merged) < total:
        if lengths[idx] > 0 and positions[idx] < lengths[idx]:
            merged.append((idx, id_lists[idx][positions[idx]]))
            positions[idx] += 1
        idx = (idx + 1) % len(id_lists)
        # 모두 소진되면 종료
        if all(positions[i] >= lengths[i] for i in range(len(id_lists))):
            break
    return merged


def _fetch_one(env: lmdb.Environment, scheme: str, id_part: bytes) -> Optional[Tuple[bytes, bytes]]:
    """스레드 안전: 각 호출에서 자체 read-txn을 열어 bytes를 반환."""
    try:
        with env.begin(write=False) as txn:
            if scheme == 'recog_simple':
                img_key = b'image-' + id_part
                lab_key = b'label-' + id_part
            else:  # 'recog_wimg'
                img_key = b'wimg-' + id_part
                lab_key = b'wlab-' + id_part
            img_bytes = txn.get(img_key)
            if img_bytes is None:
                return None
            lab_bytes = txn.get(lab_key)
            if lab_bytes is None:
                return None
            return (img_bytes, lab_bytes)
    except Exception:
        return None


def write_shuffled_merged(src_envs: List[lmdb.Environment], src_paths: List[str], schemes: List[str], merged: List[Tuple[int, bytes]], dst_path: str, commit_interval: int, workers: int = 0, prefetch: int = 512) -> int:
    os.makedirs(dst_path, exist_ok=True)
    # 맵 사이즈 추정: 입력 합산
    map_size = _sum_map_size_bytes(src_paths, growth_factor=1.2)
    print(f"[open] dst: {dst_path} (map_size ~ {map_size/1024**3:.2f} GB)")
    dst_env = lmdb.open(dst_path, map_size=map_size, subdir=True, lock=True)
    written = 0
    with dst_env.begin(write=True) as wtx:
        wtx.put(b'num-samples', str(len(merged)).encode('utf-8'))
    txn = dst_env.begin(write=True)
    total = len(merged)
    idx_global = 0
    try:
        pbar = tqdm(total=total, desc='Writing(merged)', leave=True)
        while idx_global < total:
            chunk = merged[idx_global: idx_global + max(1, prefetch)]
            fetched: list = [None] * len(chunk)
            if workers and workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = {}
                    for j, (src_idx, id_part) in enumerate(chunk):
                        futures[ex.submit(_fetch_one, src_envs[src_idx], schemes[src_idx], id_part)] = j
                    for fut in as_completed(futures):
                        j = futures[fut]
                        try:
                            fetched[j] = fut.result()
                        except Exception:
                            fetched[j] = None
            else:
                for j, (src_idx, id_part) in enumerate(chunk):
                    fetched[j] = _fetch_one(src_envs[src_idx], schemes[src_idx], id_part)

            for j, res in enumerate(fetched):
                i = idx_global + j
                if res is None:
                    continue
                img_bytes, lab_bytes = res
                out_id = f"{i+1:09d}".encode('ascii')
                txn.put(b'image-' + out_id, img_bytes)
                txn.put(b'label-' + out_id, lab_bytes)
                written += 1
                if (i + 1) % commit_interval == 0:
                    txn.commit()
                    txn = dst_env.begin(write=True)
            idx_global += len(chunk)
            pbar.update(len(chunk))
        txn.commit()
        pbar.close()
    finally:
        dst_env.sync()
        dst_env.close()
    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Shuffle recognition LMDB(s) and write to new LMDB(s).')
    p.add_argument('--lmdb_paths', type=str, required=True, help='콜론(:) 구분 다중 LMDB 경로')
    p.add_argument('--output_root', type=str, default='./shuffled_lmdb', help='개별 셔플 출력 루트 디렉터리')
    p.add_argument('--merge_output', type=str, default=None, help='지정 시 모든 입력을 균등 셔플하여 단일 LMDB로 저장')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_samples', type=int, default=0, help='0이면 전체 사용')
    p.add_argument('--commit_interval', type=int, default=2000)
    p.add_argument('--workers', type=int, default=0, help='병렬 프리페치 스레드 수 (0=비활성)')
    p.add_argument('--prefetch', type=int, default=512, help='프리페치 청크 크기')
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    paths = [p for p in args.lmdb_paths.split(':') if p]
    if not paths:
        print('no input lmdb paths')
        return
    if args.merge_output:
        # 병합 모드: 각 입력에서 페어 수집 → 라운드로빈 균등 셔플 → 단일 LMDB로 저장
        src_envs: List[lmdb.Environment] = []
        schemes: List[str] = []
        ids_per_src: List[List[bytes]] = []
        try:
            for p in paths:
                print(f"[open] src: {p}")
                env = lmdb.open(p, readonly=True, lock=False, readahead=False, meminit=False, max_readers=2048)
                src_envs.append(env)
                with env.begin(write=False) as stx:
                    scheme, counts = detect_scheme(stx)
                    n = get_num_samples(stx)
                    print(f"  scheme={scheme} counts={counts} num-samples={n if n is not None else 'unknown'}")
                    ids: List[bytes] = []
                    if scheme == 'recog_simple':
                        ids = list_pairs_recog_simple(stx, n or 0, max_samples=args.max_samples)
                    elif scheme == 'recog_wimg':
                        ids = list_pairs_recog_wimg(stx, n or 0, max_samples=args.max_samples)
                    else:
                        # fallback probe → 전체 수집
                        trial = list_pairs_recog_simple(stx, n or 0, max_samples=min(5000, max(1000, n or 1000)))
                        if len(trial) > 0:
                            scheme = 'recog_simple'
                            ids = list_pairs_recog_simple(stx, n or 0, max_samples=args.max_samples)
                        else:
                            trial = list_pairs_recog_wimg(stx, n or 0, max_samples=min(5000, max(1000, n or 1000)))
                            if len(trial) > 0:
                                scheme = 'recog_wimg'
                                ids = list_pairs_recog_wimg(stx, n or 0, max_samples=args.max_samples)
                    if not ids:
                        print(f"  [skip] unsupported or empty: {p}")
                        continue
                    schemes.append(scheme)
                    ids_per_src.append(ids)
            if not ids_per_src:
                print("no valid sources to merge")
                return
            merged = merge_round_robin(ids_per_src, seed=args.seed)
            # 출력 경로 처리
            out_path = args.merge_output
            if os.path.exists(out_path):
                if args.overwrite:
                    for name in os.listdir(out_path):
                        try:
                            os.remove(os.path.join(out_path, name))
                        except Exception:
                            pass
                else:
                    print(f"[skip] exists: {out_path} (use --overwrite to replace)")
                    return
            written = write_shuffled_merged(
                src_envs, paths, schemes, merged, out_path,
                commit_interval=args.commit_interval,
                workers=args.workers, prefetch=args.prefetch,
            )
            print(f"done: written={written} → {out_path}")
        finally:
            for e in src_envs:
                try:
                    e.close()
                except Exception:
                    pass
    else:
        # 개별 셔플 모드
        for p in paths:
            try:
                shuffle_one_lmdb(
                    input_path=p,
                    output_root=args.output_root,
                    seed=args.seed,
                    max_samples=args.max_samples,
                    commit_interval=args.commit_interval,
                    overwrite=args.overwrite,
                )
            except Exception as e:
                print(f"[error] {p}: {e}")


if __name__ == '__main__':
    main()


