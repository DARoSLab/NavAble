#!/usr/bin/env python
"""Convert Isaac Sim semantic data into mmseg and COCO-style outputs."""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blv_pipeline.constants import BLV_CLASSES, IGNORE_INDEX
from blv_pipeline.data_utils import (
    build_synthetic_color_map,
    class_pixel_counts,
    ensure_dir,
    extract_semantic_instances,
    instances_to_coco_annotations,
    remap_rgba_mask,
    sanitize_name,
    save_index_mask,
    save_rgb_image,
    write_coco_json,
)
from blv_pipeline.runtime import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--isaac-root',
        default=None,
        help='Root Isaac Sim data directory.',
    )
    parser.add_argument(
        '--out-dir',
        default=None,
        help='Output synthetic dataset directory.',
    )
    parser.add_argument(
        '--split',
        default='train',
        help='Output split name under img_dir/ and ann_dir/ (default: train).',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Optional cap on processed samples for smoke tests.',
    )
    parser.add_argument(
        '--summary-path',
        default=None,
        help='Optional path for the preprocessing summary JSON.',
    )
    parser.add_argument(
        '--sample-strategy',
        choices=['round_robin', 'sequential'],
        default='round_robin',
        help='How limited smoke-test samples are selected.',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help=(
            'Number of parallel worker processes. '
            'Defaults to os.cpu_count() which on SLURM equals your --cpus-per-task / -c value. '
            'Set to 1 to disable multiprocessing.'
        ),
    )
    return parser.parse_args()


def stem_index(files, prefix: str, suffix: str):
    mapping = {}
    for path in files:
        name = path.name
        if name.startswith(prefix) and name.endswith(suffix):
            mapping[name[len(prefix):-len(suffix)]] = path
    return mapping


def _glob_stem_index(run_dir: Path, prefix: str, suffix: str):
    """Use glob to find files matching prefix*suffix, avoiding slow iterdir()."""
    mapping = {}
    for path in run_dir.glob(f'{prefix}*{suffix}'):
        name = path.name
        mapping[name[len(prefix):-len(suffix)]] = path
    return mapping


def collect_run_entries(source_folders):
    run_entries = []
    for source_folder in source_folders:
        for run_dir in sorted(source_folder.iterdir()):
            if not run_dir.is_dir():
                continue
            # Use targeted glob patterns instead of listing all ~6000 files
            # per Run directory – avoids thousands of stat() calls on NFS.
            rgb_files = _glob_stem_index(run_dir, 'rgb_', '.png')
            seg_files = _glob_stem_index(run_dir, 'semantic_segmentation_', '.png')
            label_files = _glob_stem_index(run_dir, 'semantic_segmentation_labels_', '.json')
            # Exclude label stems from seg_files (their prefix overlaps)
            seg_files = {k: v for k, v in seg_files.items()
                         if not v.name.startswith('semantic_segmentation_labels_')}
            run_entries.append(
                dict(
                    source_folder=source_folder,
                    run_dir=run_dir,
                    rgb_files=rgb_files,
                    seg_files=seg_files,
                    label_files=label_files,
                    shared_stems=sorted(set(rgb_files) & set(seg_files) & set(label_files)),
                )
            )
    return run_entries


def iter_entries(run_entries, strategy: str):
    if strategy == 'sequential':
        for entry in run_entries:
            for stem in entry['shared_stems']:
                yield entry, stem
        return

    per_folder_entries = defaultdict(list)
    for entry in run_entries:
        per_folder_entries[entry['source_folder'].name].append(entry)

    folder_order = sorted(per_folder_entries)
    folder_streams = []
    for folder_name in folder_order:
        stems = []
        for entry in per_folder_entries[folder_name]:
            for stem in entry['shared_stems']:
                stems.append((entry, stem))
        folder_streams.append(stems)

    for items in zip_longest(*folder_streams):
        for item in items:
            if item is not None:
                yield item


# ---------------------------------------------------------------------------
# Per-sample worker (must be top-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _process_sample(task: dict) -> dict:
    """Process one (entry, stem) pair and return result dict or a skip marker.

    Returns a dict with either:
      - ``status='ok'`` and image/annotation data ready to aggregate, or
      - ``status='empty'`` if the mask has no BLV pixels, or
      - ``status='error'`` with an ``error`` string.
    """
    # Unpack task – all paths are passed as strings for pickle safety
    source_folder_name: str = task['source_folder_name']
    run_dir_name: str = task['run_dir_name']
    stem: str = task['stem']
    rgb_path = Path(task['rgb_path'])
    seg_path = Path(task['seg_path'])
    label_path = Path(task['label_path'])
    img_out = Path(task['img_out'])
    mask_out = Path(task['mask_out'])
    ignore_index: int = task['ignore_index']

    try:
        with label_path.open('r', encoding='utf-8') as handle:
            label_map = json.load(handle)

        color_map = build_synthetic_color_map(source_folder_name, label_map)
        semantic_rgba = np.asarray(Image.open(seg_path).convert('RGBA'))
        remapped_mask = remap_rgba_mask(semantic_rgba, color_map, ignore_index=ignore_index)

        if np.all(remapped_mask == ignore_index):
            return {
                'status': 'empty',
                'source_folder_name': source_folder_name,
            }

        source_slug = sanitize_name(source_folder_name)
        output_name = f'{source_slug}__{run_dir_name.lower()}__{stem}.png'
        rgb_dest = img_out / output_name
        mask_dest = mask_out / output_name

        save_rgb_image(rgb_path, rgb_dest)
        save_index_mask(mask_dest, remapped_mask)

        instances = extract_semantic_instances(
            remapped_mask,
            score_maps=None,
            ignore_index=ignore_index,
            json_serializable=True,
        )

        # Return raw instances; image_id / ann_id assigned in the main process
        return {
            'status': 'ok',
            'source_folder_name': source_folder_name,
            'output_name': output_name,
            'width': int(remapped_mask.shape[1]),
            'height': int(remapped_mask.shape[0]),
            'instances': instances,
            'pixel_counts': dict(class_pixel_counts(remapped_mask)),
        }

    except Exception as exc:  # noqa: BLE001
        return {
            'status': 'error',
            'source_folder_name': source_folder_name,
            'error': str(exc),
        }


def _build_tasks(iterator, img_out: Path, mask_out: Path) -> list:
    """Materialise the lazy iterator into a list of serialisable task dicts."""
    tasks = []
    for entry, stem in iterator:
        tasks.append({
            'source_folder_name': entry['source_folder'].name,
            'run_dir_name': entry['run_dir'].name,
            'stem': stem,
            'rgb_path': str(entry['rgb_files'][stem]),
            'seg_path': str(entry['seg_files'][stem]),
            'label_path': str(entry['label_files'][stem]),
            'img_out': str(img_out),
            'mask_out': str(mask_out),
            'ignore_index': IGNORE_INDEX,
        })
    return tasks


def main() -> None:
    args = parse_args()
    if args.isaac_root is None:
        from blv_pipeline.constants import SYNTHETIC_DATA_SOURCE
        args.isaac_root = SYNTHETIC_DATA_SOURCE
    if args.out_dir is None:
        from blv_pipeline.constants import SYNTHETIC_DATASET_ROOT
        args.out_dir = SYNTHETIC_DATASET_ROOT
    isaac_root = Path(args.isaac_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    img_out = ensure_dir(out_dir / 'img_dir' / args.split)
    mask_out = ensure_dir(out_dir / 'ann_dir' / args.split)
    coco_dir = ensure_dir(out_dir / 'annotations_coco')
    coco_path = coco_dir / f'{args.split}.json'
    summary_path = (
        Path(args.summary_path).expanduser().resolve()
        if args.summary_path
        else coco_dir / f'{args.split}_summary.json'
    )

    # Number of workers: explicit --workers, else os.cpu_count().
    # On a SLURM job with `#SBATCH -c N` (--cpus-per-task N), os.cpu_count()
    # returns N, so this automatically respects the allocated cores.
    num_workers = args.workers if args.workers is not None else os.cpu_count() or 1
    num_workers = max(1, num_workers)

    images = []
    annotations = []
    ann_id = 1
    image_id = 1
    processed = 0
    skipped_empty = 0
    error_count = 0
    incomplete_counts = Counter()
    folder_summary = defaultdict(lambda: dict(valid=0, empty=0, missing_rgb=0, missing_seg=0, missing_labels=0, errors=0))
    pixel_summary = Counter()

    source_folders = [path for path in sorted(isaac_root.iterdir()) if path.is_dir()]
    run_entries = collect_run_entries(source_folders)

    # Accumulate incomplete-pair stats (fast, no I/O per file)
    for entry in run_entries:
        source_folder = entry['source_folder']
        rgb_files = entry['rgb_files']
        seg_files = entry['seg_files']
        label_files = entry['label_files']
        incomplete_counts['missing_rgb'] += len((set(seg_files) & set(label_files)) - set(rgb_files))
        incomplete_counts['missing_seg'] += len((set(rgb_files) & set(label_files)) - set(seg_files))
        incomplete_counts['missing_labels'] += len((set(rgb_files) & set(seg_files)) - set(label_files))
        folder_summary[source_folder.name]['missing_rgb'] += len((set(seg_files) & set(label_files)) - set(rgb_files))
        folder_summary[source_folder.name]['missing_seg'] += len((set(rgb_files) & set(label_files)) - set(seg_files))
        folder_summary[source_folder.name]['missing_labels'] += len((set(rgb_files) & set(seg_files)) - set(label_files))

    iterator = iter_entries(
        run_entries,
        strategy=args.sample_strategy if args.limit is not None else 'sequential',
    )

    # Materialise tasks (apply --limit before submitting)
    all_tasks = _build_tasks(iterator, img_out, mask_out)
    if args.limit is not None:
        all_tasks = all_tasks[: args.limit]

    total = len(all_tasks)
    print(f'Submitting {total} samples to {num_workers} worker(s)...')

    if num_workers == 1:
        # Single-process path – easier to debug, no spawn overhead
        results_iter = map(_process_sample, all_tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=num_workers)
        futures = {executor.submit(_process_sample, task): i for i, task in enumerate(all_tasks)}

        def results_iter():
            for future in as_completed(futures):
                yield future.result()

        results_iter = results_iter()

    # Aggregate in submission order doesn't matter for COCO (ids assigned here)
    for result in results_iter:
        status = result['status']
        folder_name = result['source_folder_name']

        if status == 'empty':
            skipped_empty += 1
            folder_summary[folder_name]['empty'] += 1

        elif status == 'error':
            error_count += 1
            folder_summary[folder_name]['errors'] += 1
            print(f'  [WARN] error in {folder_name}: {result["error"]}', file=sys.stderr)

        elif status == 'ok':
            image_annotations, ann_id = instances_to_coco_annotations(
                result['instances'],
                image_id=image_id,
                ann_id_start=ann_id,
            )
            annotations.extend(image_annotations)
            images.append(
                dict(
                    id=image_id,
                    file_name=result['output_name'],
                    width=result['width'],
                    height=result['height'],
                )
            )
            image_id += 1
            processed += 1
            folder_summary[folder_name]['valid'] += 1
            pixel_summary.update(result['pixel_counts'])

    if num_workers > 1:
        executor.shutdown(wait=False)

    write_coco_json(coco_path, images, annotations)
    summary = dict(
        isaac_root=str(isaac_root),
        out_dir=str(out_dir),
        split=args.split,
        num_workers=num_workers,
        processed_samples=processed,
        skipped_empty_masks=skipped_empty,
        errors=error_count,
        incomplete_pairs=dict(incomplete_counts),
        per_folder=folder_summary,
        class_pixels={
            BLV_CLASSES[class_id]: pixel_summary.get(class_id, 0)
            for class_id in range(len(BLV_CLASSES))
        },
    )
    write_json(summary_path, summary)

    print(f'Processed {processed} synthetic samples into {out_dir}')
    if skipped_empty:
        print(f'Skipped {skipped_empty} samples with no mapped BLV pixels')
    if error_count:
        print(f'Errors (logged above): {error_count}')
    if incomplete_counts:
        print(f'Incomplete stems skipped due to missing pair files: {dict(incomplete_counts)}')
    print('Per-class pixel counts:')
    for class_id, class_name in enumerate(BLV_CLASSES):
        print(f'  {class_name}: {pixel_summary.get(class_id, 0)}')


if __name__ == '__main__':
    main()
