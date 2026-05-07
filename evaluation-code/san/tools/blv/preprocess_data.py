#!/usr/bin/env python
"""Normalize real-world semantic masks into the BLV training layout."""

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blv_pipeline.constants import BLV_CLASSES, IGNORE_INDEX
from blv_pipeline.data_utils import (
    apply_class_lut,
    class_pixel_counts,
    discover_real_source,
    ensure_dir,
    extract_semantic_instances,
    instances_to_coco_annotations,
    label_studio_to_dataset,
    load_class_map,
    load_index_mask,
    parse_label_studio_json,
    parse_split_ratio,
    sanitize_name,
    save_index_mask,
    save_rgb_image,
    stratified_split,
    write_coco_json,
)
from blv_pipeline.runtime import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', default=None, help='Root real-data directory (img/mask pairs).')
    parser.add_argument(
        '--out-dir',
        default=None,
        help='Output real-world dataset directory.',
    )
    parser.add_argument(
        '--label-studio-json',
        default=None,
        help='Path to a Label Studio JSON export file (alternative to --src).',
    )
    parser.add_argument(
        '--image-root',
        default=None,
        help='Directory containing source images referenced in the Label Studio export.',
    )
    parser.add_argument(
        '--split-ratio',
        default='0.8,0.1,0.1',
        help='Comma-separated train,val,test ratios for flat datasets.',
    )
    parser.add_argument(
        '--class-map',
        default=None,
        help=(
            'Optional JSON: raw semantic mask pixel values (source IDs) → BLV class IDs 0–9. '
            'Use when masks are not already encoded as BLV indices. '
            'Any source ID not listed becomes ignore (255). '
            'See README “Class remapping” section.'
        ),
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed for stratified splits.')
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Optional cap on processed samples for quick validation.',
    )
    return parser.parse_args()


def process_split(
    split_name: str,
    pairs,
    out_dir: Path,
    class_map,
    limit: int = None,
):
    img_out = ensure_dir(out_dir / 'img_dir' / split_name)
    mask_out = ensure_dir(out_dir / 'ann_dir' / split_name)
    coco_dir = ensure_dir(out_dir / 'annotations_coco')
    coco_path = coco_dir / f'{split_name}.json'

    images = []
    annotations = []
    ann_id = 1
    pixel_summary = Counter()
    processed = 0
    skipped_empty = 0

    for pair in pairs:
        if limit is not None and processed >= limit:
            break

        remapped_mask = apply_class_lut(load_index_mask(Path(pair['mask_path'])), class_map)
        if np_all_ignore(remapped_mask):
            skipped_empty += 1
            continue

        output_name = f'{sanitize_name(pair["stem"])}.png'
        save_rgb_image(Path(pair['image_path']), img_out / output_name)
        save_index_mask(mask_out / output_name, remapped_mask)

        instances = extract_semantic_instances(
            remapped_mask,
            score_maps=None,
            ignore_index=IGNORE_INDEX,
            json_serializable=True,
        )
        image_annotations, ann_id = instances_to_coco_annotations(
            instances,
            image_id=processed + 1,
            ann_id_start=ann_id,
        )
        annotations.extend(image_annotations)
        images.append(
            dict(
                id=processed + 1,
                file_name=output_name,
                width=int(remapped_mask.shape[1]),
                height=int(remapped_mask.shape[0]),
            )
        )
        pixel_summary.update(class_pixel_counts(remapped_mask))
        processed += 1

    write_coco_json(coco_path, images, annotations)
    return dict(
        processed=processed,
        skipped_empty=skipped_empty,
        class_pixels={
            BLV_CLASSES[class_id]: pixel_summary.get(class_id, 0)
            for class_id in range(len(BLV_CLASSES))
        },
    )


def np_all_ignore(mask):
    return bool((mask == IGNORE_INDEX).all())


def main() -> None:
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = str(Path(PROJECT_ROOT) / 'data' / 'real')
    out_dir = Path(args.out_dir).expanduser().resolve()

    if args.label_studio_json:
        ls_path = Path(args.label_studio_json).expanduser().resolve()
        image_root = Path(args.image_root).expanduser().resolve() if args.image_root else None
        entries = parse_label_studio_json(ls_path, image_root=image_root)
        print(f'Parsed {len(entries)} images from Label Studio export')
        pairs = label_studio_to_dataset(entries, out_dir, split_name='all')
        print(f'Converted {len(pairs)} images with valid annotations')
        discovered = {'all': pairs}
    elif args.src:
        source_root = Path(args.src).expanduser().resolve()
        discovered = discover_real_source(source_root)
    else:
        raise SystemExit('Provide either --src or --label-studio-json')

    class_map = load_class_map(args.class_map)

    if 'all' in discovered:
        split_ratios = parse_split_ratio(args.split_ratio)
        assigned = stratified_split(discovered['all'], split_ratios, seed=args.seed)
    else:
        assigned = discovered

    source_desc = args.label_studio_json or args.src
    overall_summary = dict(source=str(source_desc), out_dir=str(out_dir), splits={})
    for split_name in ('train', 'val', 'test'):
        pairs = assigned.get(split_name, [])
        split_summary = process_split(
            split_name=split_name,
            pairs=pairs,
            out_dir=out_dir,
            class_map=class_map,
            limit=args.limit,
        )
        overall_summary['splits'][split_name] = split_summary
        print(f'{split_name}: processed={split_summary["processed"]}, skipped_empty={split_summary["skipped_empty"]}')
        for class_name, count in split_summary['class_pixels'].items():
            print(f'  {class_name}: {count}')

    write_json(out_dir / 'annotations_coco' / 'real_preprocess_summary.json', overall_summary)


if __name__ == '__main__':
    main()

