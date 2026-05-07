#!/usr/bin/env python3
"""
python3 adjust_json.py --root raw_dataset
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


TARGET_ANN_KEYS = [
    "id",
    "image_id",
    "category_id",
    "segmentation",
    "bbox",
    "area",
    "iscrowd",
    "ignore",
]


def normalize_annotation(ann: Dict[str, Any]) -> Dict[str, Any]:
    out = {}

    out["id"] = ann.get("id")
    out["image_id"] = ann.get("image_id")
    out["category_id"] = ann.get("category_id")
    out["segmentation"] = ann.get("segmentation", [])
    out["bbox"] = ann.get("bbox", [])
    out["area"] = ann.get("area", 0.0)
    out["iscrowd"] = ann.get("iscrowd", 0)
    out["ignore"] = ann.get("ignore", 0)

    return out


def normalize_coco(data: Dict[str, Any]) -> Dict[str, Any]:
    out = {}

    out["info"] = data.get("info", {})
    out["images"] = data.get("images", [])
    out["categories"] = data.get("categories", [])

    anns = data.get("annotations", [])
    if not isinstance(anns, list):
        anns = []

    norm_anns: List[Dict[str, Any]] = []
    for ann in anns:
        if isinstance(ann, dict):
            norm_anns.append(normalize_annotation(ann))

    out["annotations"] = norm_anns
    return out


def is_coco_like(data: Any) -> bool:
    return (
        isinstance(data, dict)
        and isinstance(data.get("images"), list)
        and isinstance(data.get("annotations"), list)
        and isinstance(data.get("categories"), list)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="real-data", help="JSON root")
    ap.add_argument("--pattern", default="*.json", help="File pattern")
    ap.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite original files instead of creating *_adjusted.json",
    )
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(root.rglob(args.pattern))

    total = 0
    changed = 0
    skipped = 0

    for p in files:
        total += 1
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if not is_coco_like(data):
                skipped += 1
                print(f"[SKIP] not coco-like: {p}")
                continue

            norm = normalize_coco(data)

            was_same = (data == norm)

            if args.inplace:
                out_path = p
            else:
                out_path = p.with_name(f"{p.stem}_adjusted{p.suffix}")

            with out_path.open("w", encoding="utf-8") as f:
                json.dump(norm, f, ensure_ascii=False, indent=2)

            if not was_same:
                changed += 1
            print(f"[OK] {p} -> {out_path}")

        except Exception as e:
            skipped += 1
            print(f"[ERR] {p}: {e}")

    print("\n=== adjust_json summary ===")
    print(f"total: {total}")
    print(f"changed: {changed}")
    print(f"skipped/error: {skipped}")
    print(f"normalized annotation keys: {TARGET_ANN_KEYS}")


if __name__ == "__main__":
    main()