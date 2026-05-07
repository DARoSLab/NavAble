#!/usr/bin/env python3
"""
python coco_to_mmseg.py \
  --root raw_dataset \
  --out-root mmseg_dataset \
  --clean
  """
import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

CLASS_NAME_TO_ID = {
    "background": 0,
    "elevator": 1,
    "elevator_button": 2,
    "door_button": 3,
    "crosswalk": 4,
    "pedestrian_signal": 5,
    "aps_button": 6,
    "bus_stop": 7,
    "bus_stop_sign": 8,
    "handrail": 9,
    "escalator": 10,
    "turnstile": 11,
}

BACKGROUND_ID = 0

def norm(s):
    return str(s).lower().replace("-", "_").replace(" ", "_")

def map_cat(name):
    n = norm(name)
    return CLASS_NAME_TO_ID.get(n, None)

def draw_mask(h, w, anns):
    mask = np.full((h, w), BACKGROUND_ID, dtype=np.uint8)
    for ann in anns:
        cid = ann["category_id"]
        for seg in ann.get("segmentation", []):
            if len(seg) < 6:
                continue
            pts = np.array(seg).reshape(-1, 2).astype(np.int32)
            if len(pts) < 3:
                continue
            cv2.fillPoly(mask, [pts], cid)
    return mask

def process_split(split_dir: Path, out_root: Path, split: str):
    json_files = list(split_dir.rglob("annotations_adjusted.json"))

    print(f"[INFO] {split}: found {len(json_files)} jsons")

    out_img = out_root / "img_dir" / split
    out_ann = out_root / "ann_dir" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_ann.mkdir(parents=True, exist_ok=True)

    split_list = []

    for jf in json_files:
        with open(jf) as f:
            coco = json.load(f)

        cats = {c["id"]: c["name"] for c in coco["categories"]}
        anns_by_img = defaultdict(list)

        for ann in coco["annotations"]:
            cname = cats.get(ann["category_id"], "")
            cid = map_cat(cname)
            if cid is None:
                continue
            ann["category_id"] = cid
            anns_by_img[ann["image_id"]].append(ann)

        img_root = jf.parent / "images"

        for im in coco["images"]:
            img_id = im["id"]
            fname = Path(im["file_name"]).name

            src = img_root / fname
            if not src.exists():
                continue

            # Prefix with folder name to avoid filename collisions
            prefix = jf.parent.name
            stem = f"{prefix}__{Path(fname).stem}"

            dst_img = out_img / f"{stem}.png"
            dst_ann = out_ann / f"{stem}.png"

            img = cv2.imread(str(src))
            if img is None:
                continue

            cv2.imwrite(str(dst_img), img)

            h, w = img.shape[:2]
            mask = draw_mask(h, w, anns_by_img.get(img_id, []))
            cv2.imwrite(str(dst_ann), mask)

            split_list.append(stem)

    # split txt
    split_dir_out = out_root / "splits"
    split_dir_out.mkdir(exist_ok=True)

    with open(split_dir_out / f"{split}.txt", "w") as f:
        for s in split_list:
            f.write(s + "\n")

    print(f"[DONE] {split}: {len(split_list)} images")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_root)

    if args.clean and out_root.exists():
        shutil.rmtree(out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        process_split(root / split, out_root, "valid" if split=="val" else split)

    print("\n=== DONE ===")

if __name__ == "__main__":
    main()