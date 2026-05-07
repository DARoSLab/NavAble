#!/usr/bin/env python3
"""
Flatten collected_data by object into:
out_root/<object_name>/<target_type>/

Example:
python classify_to_processed.py \
  --src-root PATH_TO_ORIGINAL_DATA \
  --out-root PATH_TO_PROCESSED_DATA \
  --log-every 1000
"""
from __future__ import annotations

import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional


TARGET_MAP: Dict[str, str] = {
    "rgb_": "rgb",
    "semantic_segmentation_": "semantic_segmentation",
    "semantic_segmentation_labels_": "semantic_segmentation_labels",
    "bounding_box_2d_tight_": "bounding_box_2d_tight",
    "bounding_box_2d_tight_labels_": "bounding_box_2d_tight_labels",
    "bounding_box_2d_tight_prim_paths_": "bounding_box_2d_tight_prim_paths",
}
SORTED_PREFIXES = sorted(TARGET_MAP.keys(), key=len, reverse=True)


def detect_target_dir(file_name: str) -> Optional[str]:
    for prefix in SORTED_PREFIXES:
        if file_name.startswith(prefix):
            return TARGET_MAP[prefix]
    return None


def resolve_dup_path(dst: Path) -> Path:
    if not dst.exists():
        return dst
    stem = dst.stem
    suf = dst.suffix
    k = 1
    while True:
        cand = dst.with_name(f"{stem}__dup{k}{suf}")
        if not cand.exists():
            return cand
        k += 1


def move_or_copy(src: Path, dst: Path, move_mode: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move_mode:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description="Classify files into object/type flat structure")
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--move", action="store_true", help="move instead of copy (default: copy)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log-every", type=int, default=500)
    args = ap.parse_args()

    src_root = Path(args.src_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    if not src_root.exists():
        print(f"[ERROR] src root not found: {src_root}")
        return
    if out_root == src_root:
        print("[ERROR] out-root must be different from src-root")
        return

    out_is_inside_src = str(out_root).startswith(str(src_root) + "/")

    total_seen = 0
    total_target = 0
    total_done = 0
    total_skipped = 0

    by_object = Counter()
    by_type = Counter()
    object_type = defaultdict(Counter)

    print(f"[INFO] src_root: {src_root}")
    print(f"[INFO] out_root: {out_root}")
    print(f"[INFO] mode: {'MOVE' if args.move else 'COPY'}")

    for p in src_root.rglob("*"):
        if not p.is_file():
            continue
        total_seen += 1

        if out_is_inside_src and str(p).startswith(str(out_root) + "/"):
            continue

        tdir = detect_target_dir(p.name)
        if tdir is None:
            total_skipped += 1
            continue

        rel = p.relative_to(src_root)
        if len(rel.parts) < 2:
            total_skipped += 1
            continue

        obj_name = rel.parts[0]

        sub_parts = rel.parts[1:-1]  
        if sub_parts:
            prefix = "__".join(sub_parts)
            dst_name = f"{prefix}__{p.name}"
        else:
            dst_name = p.name

        dst = out_root / obj_name / tdir / dst_name
        dst = resolve_dup_path(dst)

        if args.dry_run:
            print(f"[DRY] {p} -> {dst}")
        else:
            move_or_copy(p, dst, move_mode=args.move)

        total_target += 1
        total_done += 1
        by_object[obj_name] += 1
        by_type[tdir] += 1
        object_type[obj_name][tdir] += 1

        if total_done % max(1, args.log_every) == 0:
            print(f"[PROGRESS] processed={total_done} targets, seen={total_seen}")

    print("\n=== Summary ===")
    print(f"seen_files: {total_seen}")
    print(f"matched_target_files: {total_target}")
    print(f"processed_files: {total_done}")
    print(f"non_target_skipped: {total_skipped}")

    print("\n[By object]")
    for obj, n in sorted(by_object.items()):
        print(f"  {obj}: {n}")

    print("\n[By type]")
    for t, n in sorted(by_type.items()):
        print(f"  {t}: {n}")

    print("\n[By object x type]")
    for obj in sorted(object_type.keys()):
        row = object_type[obj]
        items = ", ".join(f"{k}={row[k]}" for k in sorted(row.keys()))
        print(f"  {obj}: {items}")


if __name__ == "__main__":
    main()
