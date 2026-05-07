"""Convert Final_annotations_with_val (COCO polygon) → mmseg semseg PNG.

Source layout:
    Final_annotations_with_val/{Train,Val,Test}/<class_name>/{annotations.json, images/}
Output layout:
    data/real_v2/{img_dir,ann_dir}/{train,val,test}/<file_name>.png

COCO category_id (1..11) is written directly into the mask (V2 schema). bg=0.
Polygons are painted in descending area order so smaller objects (e.g. buttons
on doors) are not overwritten by larger overlapping ones.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SRC_DEFAULT = os.environ.get("BLV_RAW_REAL_DATA", "")  # set BLV_RAW_REAL_DATA to your COCO-format real dataset path
DST_DEFAULT = os.path.join(
    os.environ.get("BLV_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))),
    "data", "real_v2",
)

SPLIT_MAP = {"Train": "train", "Val": "val", "Test": "test"}


def convert_split(src_root: Path, dst_root: Path, split_in: str, split_out: str) -> dict:
    img_out = dst_root / "img_dir" / split_out
    ann_out = dst_root / "ann_dir" / split_out
    img_out.mkdir(parents=True, exist_ok=True)
    ann_out.mkdir(parents=True, exist_ok=True)

    n_imgs = 0
    n_anns = 0
    label_hist: dict[int, int] = defaultdict(int)
    skipped_missing = 0

    for cls_folder in sorted(os.listdir(src_root / split_in)):
        cls_dir = src_root / split_in / cls_folder
        ann_path = cls_dir / "annotations.json"
        img_dir = cls_dir / "images"
        if not ann_path.is_file():
            continue
        with open(ann_path) as f:
            coco = json.load(f)

        id_to_meta = {im["id"]: im for im in coco["images"]}
        per_img: dict[int, list] = defaultdict(list)
        for a in coco["annotations"]:
            per_img[a["image_id"]].append(a)

        for img_id, meta in id_to_meta.items():
            fname = meta["file_name"]
            w, h = meta["width"], meta["height"]
            src_img = img_dir / fname
            if not src_img.is_file():
                skipped_missing += 1
                continue

            mask = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(mask)
            anns = sorted(
                per_img.get(img_id, []),
                key=lambda a: a.get("area", 0),
                reverse=True,  # large first → small overwrites
            )
            for a in anns:
                cat = int(a["category_id"])
                seg = a.get("segmentation")
                if not (isinstance(seg, list) and seg and isinstance(seg[0], list)):
                    continue
                for poly in seg:
                    if len(poly) < 6:  # need at least 3 points
                        continue
                    pts = list(zip(poly[0::2], poly[1::2]))
                    draw.polygon(pts, fill=cat)
                n_anns += 1

            # Write mask (single-channel uint8 PNG)
            mask.save(ann_out / fname, format="PNG")

            # Symlink image (avoid duplicating ~5KB-1MB files for 5.4k images)
            dst_img = img_out / fname
            if dst_img.exists() or dst_img.is_symlink():
                dst_img.unlink()
            dst_img.symlink_to(src_img.resolve())

            arr = np.asarray(mask, dtype=np.uint8)
            for v in np.unique(arr).tolist():
                label_hist[int(v)] += 1
            n_imgs += 1

    return dict(
        split=split_out,
        n_imgs=n_imgs,
        n_anns=n_anns,
        skipped_missing=skipped_missing,
        label_hist=dict(sorted(label_hist.items())),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=SRC_DEFAULT)
    p.add_argument("--dst", default=DST_DEFAULT)
    args = p.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    print(f"Source:      {src_root}")
    print(f"Destination: {dst_root}")
    dst_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for split_in, split_out in SPLIT_MAP.items():
        print(f"\n=== Converting {split_in} → {split_out} ===")
        s = convert_split(src_root, dst_root, split_in, split_out)
        print(f"  imgs={s['n_imgs']}  anns={s['n_anns']}  skipped_missing={s['skipped_missing']}")
        print(f"  label histogram (unique-label → image-count): {s['label_hist']}")
        summaries.append(s)

    # Write meta.json mirroring opensrc / syn schema
    meta = {
        "classes": [
            "background", "elevator", "elevator_button", "door_button", "crosswalk",
            "pedestrian_signal", "aps_button", "bus_stop", "bus_stop_sign",
            "handrail", "escalator", "turnstile",
        ],
        "class_name_to_id": {
            "background": 0, "elevator": 1, "elevator_button": 2, "door_button": 3,
            "crosswalk": 4, "pedestrian_signal": 5, "aps_button": 6, "bus_stop": 7,
            "bus_stop_sign": 8, "handrail": 9, "escalator": 10, "turnstile": 11,
        },
        "background_id": 0,
        "ignore_label": 255,
        "source": "Final_annotations_with_val (COCO) rasterized 2026-04-27",
        "split_counts": {s["split"]: s["n_imgs"] for s in summaries},
    }
    with open(dst_root / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nWrote {dst_root / 'meta.json'}")


if __name__ == "__main__":
    main()
