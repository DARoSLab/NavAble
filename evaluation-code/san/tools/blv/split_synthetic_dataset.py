#!/usr/bin/env python
"""Deterministically split the synthetic train set into train/val/test.

After running preprocess_synthetic.py (which writes everything to
data/synthetic/img_dir/train and ann_dir/train), use this script to carve
out val and/or test splits from that pool.

Usage:
    python tools/blv/split_synthetic_dataset.py --data-root data/synthetic
    python tools/blv/split_synthetic_dataset.py --data-root data/synthetic --split-ratio 0.9,0.1,0.0
    python tools/blv/split_synthetic_dataset.py --data-root data/synthetic --dry-run
    python tools/blv/split_synthetic_dataset.py --data-root data/synthetic --reset-targets
"""

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blv_pipeline.constants import SYNTHETIC_DATASET_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--data-root',
        default=None,
        help='Root of the synthetic dataset (default: from constants).',
    )
    parser.add_argument(
        '--split-ratio',
        default='0.9,0.1,0.0',
        help='Comma-separated train,val,test ratios (default: 0.9,0.1,0.0).',
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument(
        '--mode',
        choices=['copy', 'move'],
        default='move',
        help='Copy or move files from train to val/test (default: move).',
    )
    parser.add_argument(
        '--reset-targets',
        action='store_true',
        help='Delete existing val/test dirs before splitting (idempotent reruns).',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would happen without moving/copying files.',
    )
    return parser.parse_args()


def list_stems(directory: Path) -> list:
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob('*.png'))


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root or SYNTHETIC_DATASET_ROOT)

    ratios = [float(x) for x in args.split_ratio.split(',')]
    if len(ratios) != 3:
        raise SystemExit('--split-ratio must have exactly 3 values: train,val,test')
    ratio_sum = sum(ratios)
    if abs(ratio_sum - 1.0) > 1e-6:
        raise SystemExit(f'Ratios must sum to 1.0, got {ratio_sum}')
    split_names = ['train', 'val', 'test']

    img_train = data_root / 'img_dir' / 'train'
    ann_train = data_root / 'ann_dir' / 'train'

    if not img_train.is_dir() or not ann_train.is_dir():
        raise SystemExit(f'Expected {img_train} and {ann_train} to exist.')

    if args.reset_targets and not args.dry_run:
        for split in ('val', 'test'):
            for sub in ('img_dir', 'ann_dir'):
                target = data_root / sub / split
                if target.is_dir():
                    shutil.rmtree(target)
                    print(f'Removed {target}')

    img_stems = set(list_stems(img_train))
    ann_stems = set(list_stems(ann_train))
    shared = sorted(img_stems & ann_stems)
    print(f'Found {len(shared)} paired stems in train/')

    rng = random.Random(args.seed)
    rng.shuffle(shared)

    counts = {}
    total = len(shared)
    assigned = 0
    for i, name in enumerate(split_names):
        if i == len(split_names) - 1:
            counts[name] = total - assigned
        else:
            n = round(ratios[i] * total)
            counts[name] = n
            assigned += n

    assignments = {}
    offset = 0
    for name in split_names:
        n = counts[name]
        for stem in shared[offset:offset + n]:
            assignments[stem] = name
        offset += n

    summary = {name: 0 for name in split_names}
    moved_or_copied = 0
    skipped_same_path = 0

    for stem, target_split in assignments.items():
        summary[target_split] += 1
        if target_split == 'train':
            skipped_same_path += 1
            continue

        for sub in ('img_dir', 'ann_dir'):
            src = data_root / sub / 'train' / f'{stem}.png'
            dst_dir = data_root / sub / target_split
            dst = dst_dir / f'{stem}.png'

            if args.dry_run:
                print(f'  [{args.mode}] {src} -> {dst}')
                continue

            dst_dir.mkdir(parents=True, exist_ok=True)
            if args.mode == 'move':
                shutil.move(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))

        moved_or_copied += 1

    print(f'\nSplit summary (seed={args.seed}):')
    for name in split_names:
        print(f'  {name}: {summary[name]}')
    print(f'  moved_or_copied: {moved_or_copied}')
    print(f'  skipped_same_path (train->train): {skipped_same_path}')

    if not args.dry_run:
        summary_path = data_root / 'annotations_coco' / 'split_summary.json'
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open('w') as f:
            json.dump(
                dict(
                    seed=args.seed,
                    ratios=dict(zip(split_names, ratios)),
                    counts=summary,
                    moved_or_copied=moved_or_copied,
                    skipped_same_path=skipped_same_path,
                    mode=args.mode,
                ),
                f,
                indent=2,
            )
        print(f'Summary written to {summary_path}')


if __name__ == '__main__':
    main()
