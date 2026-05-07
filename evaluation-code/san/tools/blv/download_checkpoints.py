#!/usr/bin/env python3
"""Download all required pretrained model checkpoints to checkpoints/pretrained/.

Run this script once after cloning the repository on a new machine (e.g.
the Unity cluster) before launching any training or evaluation.

Usage:
    python tools/blv/download_checkpoints.py          # download all
    python tools/blv/download_checkpoints.py --model mask2former segformer
    python tools/blv/download_checkpoints.py --dry-run
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blv_pipeline.constants import CHECKPOINT_URLS, PRETRAINED_CKPT_DIR, PRETRAINED_CHECKPOINTS

# Filenames for each model key (destination under PRETRAINED_CKPT_DIR)
DEST_NAMES = {
    'mask2former': 'mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.pth',
    'segformer':   'segformer_mit-b5_640x640_160k_ade20k_20210801_121243-41d2845b.pth',
    'san_b16':     'san-vit-b16_20230906-fd0a7684.pth',
    'san_l14':     'san-vit-l14_20230907-a11e098f.pth',
    'san_clip_b16': 'clip_vit-base-patch16-224_3rdparty-d08f8887.pth',
    'san_clip_l14': 'clip-vit-large-patch14_3rdparty-d08f8887.pth',
}

# Which models are required for the currently implemented configs
REQUIRED_MODELS = ['mask2former', 'segformer', 'san_b16']
# san_clip_b16 is fetched automatically by mmseg during SAN inference,
# but can be pre-downloaded here to avoid runtime network calls.
OPTIONAL_MODELS = ['san_l14', 'san_clip_b16', 'san_clip_l14']

ALL_MODELS = REQUIRED_MODELS + OPTIONAL_MODELS


def _progress(count: int, block_size: int, total: int) -> None:
    downloaded = count * block_size
    if total > 0:
        pct = min(100.0, downloaded * 100.0 / total)
        bar = int(pct / 2)
        print(
            f'\r  [{"#" * bar}{" " * (50 - bar)}] {pct:5.1f}%  '
            f'({downloaded / 1e6:.1f} / {total / 1e6:.1f} MB)',
            end='',
            flush=True,
        )


def download_file(url: str, dest: Path, dry_run: bool = False) -> None:
    if dest.exists():
        print(f'  Already exists, skipping: {dest.name}')
        return
    if dry_run:
        print(f'  [dry-run] Would download: {url}')
        print(f'         -> {dest}')
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f'  Downloading {dest.name} ...')
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()  # newline after progress bar
        print(f'  Saved to {dest}')
    except Exception as exc:
        print(f'\n  ERROR downloading {url}: {exc}')
        if dest.exists():
            dest.unlink()
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model',
        nargs='+',
        choices=ALL_MODELS + ['all', 'required'],
        default=['required'],
        help=(
            'Which models to download. '
            '"required" (default) downloads the three models used in all configs. '
            '"all" includes optional SAN variants.'
        ),
    )
    parser.add_argument('--dry-run', action='store_true', help='Print URLs without downloading.')
    parser.add_argument(
        '--dest-dir',
        default=PRETRAINED_CKPT_DIR,
        help=f'Directory to save checkpoints (default: {PRETRAINED_CKPT_DIR}).',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dest_dir = Path(args.dest_dir).expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    models = args.model
    if 'all' in models:
        models = ALL_MODELS
    elif 'required' in models:
        models = REQUIRED_MODELS

    print(f'Destination: {dest_dir}')
    print(f'Models to download: {models}\n')

    failed = []
    for key in models:
        url = CHECKPOINT_URLS.get(key)
        if url is None:
            print(f'WARNING: No URL configured for "{key}", skipping.')
            continue
        dest = dest_dir / DEST_NAMES[key]
        print(f'[{key}]')
        try:
            download_file(url, dest, dry_run=args.dry_run)
        except Exception:
            failed.append(key)

    print('\n--- Summary ---')
    if not args.dry_run:
        for key in models:
            dest = dest_dir / DEST_NAMES.get(key, '')
            if key in failed:
                status = 'FAILED'
            elif dest.exists():
                status = f'OK  ({dest.stat().st_size / 1e6:.0f} MB)'
            else:
                status = 'SKIPPED'
            print(f'  {key:20s}: {status}')

    if failed:
        print(f'\nFailed downloads: {failed}')
        sys.exit(1)
    else:
        print('\nAll downloads complete.')


if __name__ == '__main__':
    main()
