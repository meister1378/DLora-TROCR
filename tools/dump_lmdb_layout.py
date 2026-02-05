#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LMDB 레이아웃/인식용 데이터 덤프 스크립트
 - image-/gt- 또는 image-/label- 스킴 자동 감지
 - 지정 개수만 이미지/라벨을 디스크로 저장
사용 예:
  python -u tools/dump_lmdb_layout.py --lmdb /mnt/nas/ocr_dataset/public_admin_train_layout.lmdb --out ./lmdb_dump --max 20
"""
import os
import lmdb
import argparse
import pickle
from tqdm import tqdm
import orjson


def detect_scheme(txn: lmdb.Transaction):
    """
    지원 스킴:
      - det: image-XXXXXXXXX, gt-XXXXXXXXX
      - recog: image-XXXXXXXXX, label-XXXXXXXXX
    """
    # 빠른 직접 탐색(초기 몇 개 키만 프로빙)
    for i in (0, 1, 2, 3, 4, 5, 10, 100, 1000):
        if txn.get(f'image-{i:09d}'.encode()):
            img_hit = True
            break
    else:
        img_hit = False
    for i in (0, 1, 2, 3, 4, 5, 10, 100, 1000):
        if txn.get(f'gt-{i:09d}'.encode()):
            gt_hit = True
            break
    else:
        gt_hit = False
    for i in (0, 1, 2, 3, 4, 5, 10, 100, 1000):
        if txn.get(f'label-{i:09d}'.encode()):
            lab_hit = True
            break
    else:
        lab_hit = False
    if img_hit and gt_hit:
        return 'det'
    if img_hit and lab_hit:
        return 'recog'
    # 커서 스캔(전체 순회, 조기 종료)
    cur = txn.cursor()
    has_image = False
    has_gt = False
    has_label = False
    for k, _ in cur:
        if k.startswith(b'image-'):
            has_image = True
        elif k.startswith(b'gt-'):
            has_gt = True
        elif k.startswith(b'label-'):
            has_label = True
        if (has_image and has_gt) or (has_image and has_label):
            break
    if has_image and has_gt:
        return 'det'
    if has_image and has_label:
        return 'recog'
    return 'unknown'


def get_num_samples(txn: lmdb.Transaction) -> int:
    raw = txn.get(b'num-samples')
    if raw is None:
        # num-samples가 없는 경우 추정
        n = 0
        cur = txn.cursor()
        for k, _ in cur:
            if k.startswith(b'image-'):
                n += 1
        return n
    try:
        return int(raw.decode('utf-8'))
    except Exception:
        try:
            return int(raw)
        except Exception:
            return 0


def dump_lmdb(lmdb_path: str, out_dir: str, max_samples: int = 50):
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, 'images')
    lab_dir = os.path.join(out_dir, 'labels')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)

    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False, max_readers=2048)
    with env.begin(write=False) as txn:
        scheme = detect_scheme(txn)
        n = get_num_samples(txn)
        if n == 0:
            print(f'no samples detected in {lmdb_path}')
            return
        print(f'  scheme={scheme} num-samples={n}')
        total = min(max_samples, n) if max_samples > 0 else n
        for i in tqdm(range(total), desc='dump'):
            key_suffix = f'{i:09d}'.encode('ascii')
            img_key = b'image-' + key_suffix
            lab_key = (b'gt-' if scheme == 'det' else b'label-') + key_suffix
            img_bytes = txn.get(img_key)
            lab_bytes = txn.get(lab_key)
            if img_bytes is None or lab_bytes is None:
                continue
            # 이미지 저장
            img_name = f'img-{i:09d}.jpg'
            with open(os.path.join(img_dir, img_name), 'wb') as f:
                f.write(img_bytes)
            # 라벨 저장
            try:
                gt_info = pickle.loads(lab_bytes)
            except Exception:
                # orjson 등 다른 포맷 대비
                try:
                    gt_info = orjson.loads(lab_bytes)
                except Exception:
                    gt_info = {'raw': str(lab_bytes[:64])}
            with open(os.path.join(lab_dir, img_name + '.json'), 'wb') as f:
                f.write(orjson.dumps(gt_info))
    env.close()
    print(f'dumped {lmdb_path} -> {out_dir}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lmdb', required=True, help='LMDB 경로')
    ap.add_argument('--out', required=True, help='덤프 출력 디렉토리')
    ap.add_argument('--max', type=int, default=50, help='최대 덤프 샘플 수(0=전체)')
    args = ap.parse_args()
    dump_lmdb(args.lmdb, args.out, max_samples=args.max)


if __name__ == '__main__':
    main()






