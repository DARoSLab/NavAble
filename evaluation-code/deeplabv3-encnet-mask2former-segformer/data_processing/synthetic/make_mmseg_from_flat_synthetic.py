#!/usr/bin/env python3
"""
make_mmseg_from_flat_synthetic.py

Build an mmseg-compatible dataset from a synthetic data tree.

Auto-detected layouts:

  Flat (single-class):
    <root>/
        rgb/
        semantic_segmentation/
        semantic_segmentation_labels/

  Nested (multi-class, one subdir per scene/class):
    <root>/
        elevator/{rgb, semantic_segmentation, semantic_segmentation_labels}
        bus_stop/{...}
        ...

Usage:
    python make_mmseg_from_flat_synthetic.py \\
        --dataset-root <path> \\
        --out-root    <path> \\
        --clean
"""

import argparse
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np


# ─── class taxonomy ──────────────────────────────────────────────────────────
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
IGNORE_LABEL = 255


@dataclass
class Sample:
    stem: str
    rgb_path: Path
    seg_path: Path
    label_path: Path
    classes: Set[int]


# ─── name normalization & class mapping ──────────────────────────────────────
def norm(s: str) -> str:
    return re.sub(r"[\s\-]+", "_", s.strip().lower())


def map_class_name(raw_name: str) -> Optional[str]:
    c = norm(raw_name)
    if c in CLASS_NAME_TO_ID:
        return c
    mapping = {
        "elevator_button,elevatorrequestbuttons": "elevator_button",
        "elevator_button,elevatorbutton": "elevator_button",
        "elevatorbutton": "elevator_button",
        "elevatorrequestbuttons": "elevator_button",
        "push_to_open_button": "door_button",
        "accessibility_button": "door_button",
        "cross_walk": "crosswalk",
        "pedestriancrossing,roadmarkings": "crosswalk",
        "pedestriancrossing": "crosswalk",
        "zebra_crossing": "crosswalk",
        "traffic_signal": "pedestrian_signal",
        "pedestrian_traffic_light": "pedestrian_signal",
        "pedestriansignal": "pedestrian_signal",
        "push_button": "aps_button",
        "pedestrian_button": "aps_button",
        "ped_pushbutton": "aps_button",
        "button": "aps_button",
        "busstop": "bus_stop",
        "bus_shelter": "bus_stop",
        "busstopsignstand": "bus_stop_sign",
        "bus_stop_sign_stand": "bus_stop_sign",
        "railing": "handrail",
        "unlabelled": "background",
        "unlabeled": "background",
    }
    return mapping.get(c)


# ─── RGBA color parsing ──────────────────────────────────────────────────────
def parse_rgba_key(color_key: str) -> Optional[Tuple[int, int, int, int]]:
    s = color_key.strip().strip("(").strip(")")
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        return None
    try:
        return tuple(int(x) for x in parts)
    except ValueError:
        return None


def make_rgba_mask(seg_img: np.ndarray, rgba: Tuple[int, int, int, int]) -> np.ndarray:
    """label json color key is RGBA. cv2.imread reads as BGRA / BGR."""
    r, g, b, a = rgba
    if seg_img is None or seg_img.ndim != 3:
        if seg_img is None:
            return np.zeros((0, 0), dtype=bool)
        return np.zeros(seg_img.shape[:2], dtype=bool)
    if seg_img.shape[2] == 4:
        return (
            (seg_img[:, :, 0] == b)
            & (seg_img[:, :, 1] == g)
            & (seg_img[:, :, 2] == r)
            & (seg_img[:, :, 3] == a)
        )
    return (
        (seg_img[:, :, 0] == b)
        & (seg_img[:, :, 1] == g)
        & (seg_img[:, :, 2] == r)
    )


# ─── pairing logic ───────────────────────────────────────────────────────────
def strip_known_suffix(stem: str) -> str:
    patterns = [
        r"^(.*)__semantic_segmentation_labels_(\d+)$",
        r"^(.*)__semantic_segmentation_(\d+)$",
        r"^(.*)__rgb_(\d+)$",
        r"^(.*)__frame_(\d+)$",
    ]
    for p in patterns:
        m = re.match(p, stem)
        if m:
            return f"{m.group(1)}__{m.group(2)}"
    return stem


def find_pair(root: Path, label_path: Path) -> Tuple[Optional[Path], Optional[Path], str]:
    rgb_dir = root / "rgb"
    seg_dir = root / "semantic_segmentation"
    label_stem = label_path.stem

    candidates = [
        (rgb_dir / f"{label_stem}.png",
         seg_dir / f"{label_stem}.png",
         label_stem),
    ]

    m = re.match(r"^(.*)__semantic_segmentation_labels_(\d+)$", label_stem)
    if m:
        prefix, idx = m.group(1), m.group(2)
        candidates.extend([
            (rgb_dir / f"{prefix}__rgb_{idx}.png",
             seg_dir / f"{prefix}__semantic_segmentation_{idx}.png",
             f"{prefix}__{idx}"),
            (rgb_dir / f"{prefix}__frame_{idx}.png",
             seg_dir / f"{prefix}__frame_{idx}.png",
             f"{prefix}__{idx}"),
        ])

    m = re.match(r"^(.*)__frame_(\d+)$", label_stem)
    if m:
        prefix, idx = m.group(1), m.group(2)
        candidates.extend([
            (rgb_dir / f"{prefix}__frame_{idx}.png",
             seg_dir / f"{prefix}__frame_{idx}.png",
             f"{prefix}__{idx}"),
            (rgb_dir / f"{prefix}__rgb_{idx}.png",
             seg_dir / f"{prefix}__semantic_segmentation_{idx}.png",
             f"{prefix}__{idx}"),
        ])

    base = strip_known_suffix(label_stem)
    candidates.append(
        (rgb_dir / f"{base}.png",
         seg_dir / f"{base}.png",
         base),
    )

    for rgb_path, seg_path, out_stem in candidates:
        if rgb_path.exists() and seg_path.exists():
            safe_stem = re.sub(r"[^A-Za-z0-9_.\-]+", "_", out_stem)
            return rgb_path, seg_path, safe_stem

    return None, None, label_stem


# ─── label JSON parsing ──────────────────────────────────────────────────────
def read_present_classes(label_path: Path) -> Set[int]:
    with label_path.open("r", encoding="utf-8") as f:
        label_map = json.load(f)
    present: Set[int] = set()
    if not isinstance(label_map, dict):
        return present
    for _, meta in label_map.items():
        if not isinstance(meta, dict):
            continue
        mapped = map_class_name(str(meta.get("class", "")))
        if mapped is None:
            continue
        cid = CLASS_NAME_TO_ID.get(mapped)
        if cid is None or cid == BACKGROUND_ID:
            continue
        present.add(cid)
    return present


# ─── sample collection (nested-aware) ────────────────────────────────────────
def collect_samples(root: Path) -> List[Sample]:
    direct_lbl = root / "semantic_segmentation_labels"
    if direct_lbl.exists():
        sub_roots = [root]
    else:
        sub_roots = sorted([
            d for d in root.iterdir()
            if d.is_dir() and (d / "semantic_segmentation_labels").exists()
        ])
        if not sub_roots:
            raise FileNotFoundError(
                f"No semantic_segmentation_labels found under {root} "
                f"(neither directly nor in any subdirectory)"
            )

    samples: List[Sample] = []
    multi = not (len(sub_roots) == 1 and sub_roots[0] == root)

    for sub_root in sub_roots:
        lbl_dir = sub_root / "semantic_segmentation_labels"
        prefix = f"{sub_root.name}__" if multi else ""
        n_added = 0
        n_skip_pair = 0
        n_skip_empty = 0

        for label_path in sorted(lbl_dir.glob("*.json")):
            rgb_path, seg_path, stem = find_pair(sub_root, label_path)
            if rgb_path is None or seg_path is None:
                n_skip_pair += 1
                continue
            classes = read_present_classes(label_path)
            if len(classes) == 0:
                n_skip_empty += 1
                continue
            samples.append(Sample(
                stem=f"{prefix}{stem}",
                rgb_path=rgb_path,
                seg_path=seg_path,
                label_path=label_path,
                classes=classes,
            ))
            n_added += 1

        tag = sub_root.name if multi else "(root)"
        print(f"[INFO] {tag:24s}  added={n_added:6d}  no-pair={n_skip_pair:5d}  empty={n_skip_empty:5d}")

    return samples


# ─── stats ───────────────────────────────────────────────────────────────────
def class_counts(samples: List[Sample]) -> Dict[int, int]:
    counts = {cid: 0 for cid in CLASS_NAME_TO_ID.values() if cid != BACKGROUND_ID}
    for s in samples:
        for cid in s.classes:
            if cid != BACKGROUND_ID:
                counts[cid] += 1
    return counts


def annotation_counts(samples: List[Sample], require_pixel_hit: bool = True) -> Dict[int, int]:
    counts = {cid: 0 for cid in CLASS_NAME_TO_ID.values() if cid != BACKGROUND_ID}
    for s in samples:
        try:
            with s.label_path.open("r", encoding="utf-8") as f:
                label_map = json.load(f)
        except Exception as e:
            print(f"[WARN] failed to read label: {s.label_path} ({e})")
            continue
        if not isinstance(label_map, dict):
            continue
        seg = None
        if require_pixel_hit:
            seg = cv2.imread(str(s.seg_path), cv2.IMREAD_UNCHANGED)
            if seg is None:
                continue
        for color_key, meta in label_map.items():
            if not isinstance(meta, dict):
                continue
            mapped = map_class_name(str(meta.get("class", "")))
            if mapped is None:
                continue
            cid = CLASS_NAME_TO_ID.get(mapped)
            if cid is None or cid == BACKGROUND_ID:
                continue
            if require_pixel_hit:
                rgba = parse_rgba_key(color_key)
                if rgba is None:
                    continue
                hit = make_rgba_mask(seg, rgba)
                if not hit.any():
                    continue
            counts[cid] += 1
    return counts


# ─── splitting ───────────────────────────────────────────────────────────────
def train_only_split(
    samples: List[Sample],
    seed: int,
    val_size: int = 1,
    test_size: int = 1,
) -> Dict[str, List[Sample]]:
    """Put nearly everything into train; pick a tiny fixed number for val/test."""
    rng = random.Random(seed)
    n = len(samples)
    val_size = max(0, min(val_size, n))
    test_size = max(0, min(test_size, n - val_size))

    shuffled = samples[:]
    rng.shuffle(shuffled)

    val = shuffled[:val_size]
    test = shuffled[val_size:val_size + test_size]
    train = shuffled[val_size + test_size:]
    return {"train": train, "val": val, "test": test}


def multilabel_stratified_split(
    samples: List[Sample],
    seed: int,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> Dict[str, List[Sample]]:
    rng = random.Random(seed)
    split_names = ["train", "val", "test"]
    split_ratio = dict(zip(split_names, ratios))

    total_class_count = class_counts(samples)
    target = {
        split: {cid: total_class_count[cid] * split_ratio[split] for cid in total_class_count}
        for split in split_names
    }
    current = {split: {cid: 0 for cid in total_class_count} for split in split_names}

    total_n = len(samples)
    n_train = int(total_n * ratios[0])
    n_val = int(total_n * ratios[1])
    n_test = total_n - n_train - n_val
    target_n = {"train": n_train, "val": n_val, "test": n_test}

    out = {split: [] for split in split_names}
    rarity = {cid: 1.0 / max(1, total_class_count[cid]) for cid in total_class_count}

    shuffled = samples[:]
    rng.shuffle(shuffled)
    shuffled.sort(
        key=lambda s: sum(rarity.get(cid, 0.0) for cid in s.classes),
        reverse=True,
    )

    for s in shuffled:
        best_split = None
        best_score = None
        for split in split_names:
            size_penalty = max(0, len(out[split]) - target_n[split]) * 10.0
            class_gain = 0.0
            for cid in s.classes:
                deficit = target[split].get(cid, 0.0) - current[split].get(cid, 0)
                if deficit > 0:
                    class_gain += deficit / max(1.0, target[split].get(cid, 1.0))
            score = class_gain - size_penalty
            if best_score is None or score > best_score:
                best_score = score
                best_split = split
        out[best_split].append(s)
        for cid in s.classes:
            if cid in current[best_split]:
                current[best_split][cid] += 1

    return out


# ─── mask generation ─────────────────────────────────────────────────────────
def make_mmseg_mask(seg_path: Path, label_path: Path) -> Optional[np.ndarray]:
    seg = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
    if seg is None:
        print(f"[WARN] read fail seg: {seg_path}")
        return None
    with label_path.open("r", encoding="utf-8") as f:
        label_map = json.load(f)
    if not isinstance(label_map, dict):
        print(f"[WARN] invalid label json: {label_path}")
        return None

    mask = np.full(seg.shape[:2], BACKGROUND_ID, dtype=np.uint8)
    for color_key, meta in label_map.items():
        if not isinstance(meta, dict):
            continue
        mapped = map_class_name(str(meta.get("class", "")))
        if mapped is None:
            continue
        cid = CLASS_NAME_TO_ID.get(mapped)
        if cid is None:
            continue
        rgba = parse_rgba_key(color_key)
        if rgba is None:
            continue
        hit = make_rgba_mask(seg, rgba)
        if hit.any():
            mask[hit] = cid
    return mask


# ─── output IO ───────────────────────────────────────────────────────────────
def ensure_dirs(out_root: Path):
    for split in ["train", "val", "test"]:
        (out_root / "img_dir" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "ann_dir" / split).mkdir(parents=True, exist_ok=True)
    (out_root / "splits").mkdir(parents=True, exist_ok=True)


def print_split_stats(split_map: Dict[str, List[Sample]]):
    id_to_name = {v: k for k, v in CLASS_NAME_TO_ID.items()}
    print("\n=== Split stats by image-level class presence ===")
    for split, items in split_map.items():
        counts = class_counts(items)
        total = sum(counts.values())
        print(f"\n[{split}] images={len(items)} class_presence_total={total}")
        for cid in sorted(counts):
            print(f"  {cid:2d} {id_to_name[cid]:22s} {counts[cid]:6d}")
    print("\n=== Split stats by annotation/instance count ===")
    for split, items in split_map.items():
        counts = annotation_counts(items, require_pixel_hit=True)
        total = sum(counts.values())
        print(f"\n[{split}] images={len(items)} annotations={total}")
        for cid in sorted(counts):
            print(f"  {cid:2d} {id_to_name[cid]:22s} {counts[cid]:6d}")


def write_meta(out_root: Path):
    classes = [None] * len(CLASS_NAME_TO_ID)
    for name, cid in CLASS_NAME_TO_ID.items():
        classes[cid] = name
    meta = {
        "classes": classes,
        "class_name_to_id": CLASS_NAME_TO_ID,
        "background_id": BACKGROUND_ID,
        "ignore_label": IGNORE_LABEL,
    }
    with (out_root / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, help="data root (flat or nested)")
    ap.add_argument("--out-root", required=True, help="output mmseg dataset root")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--train-only", action="store_true",
                    help="Put almost everything in train; val/test get tiny fixed counts.")
    ap.add_argument("--val-size", type=int, default=1,
                    help="Number of samples in val when --train-only (default 1).")
    ap.add_argument("--test-size", type=int, default=1,
                    help="Number of samples in test when --train-only (default 1).")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    ensure_dirs(out_root)

    samples = collect_samples(dataset_root)
    print(f"\n[INFO] collected samples: {len(samples)}")
    if len(samples) == 0:
        raise RuntimeError("No samples collected. Check naming/structure.")

    if args.train_only:
        split_map = train_only_split(
            samples,
            seed=args.seed,
            val_size=args.val_size,
            test_size=args.test_size,
        )
        print(f"[INFO] split mode: TRAIN-ONLY (val={args.val_size}, test={args.test_size})")
    else:
        split_map = multilabel_stratified_split(samples, seed=args.seed)
        print("[INFO] split mode: stratified 0.8/0.1/0.1")

    print(
        f"[INFO] train={len(split_map['train'])}, "
        f"val={len(split_map['val'])}, "
        f"test={len(split_map['test'])}"
    )
    print_split_stats(split_map)

    used_stems = set()
    for split in ["train", "val", "test"]:
        names: List[str] = []
        total = len(split_map[split])
        for i, s in enumerate(split_map[split], 1):
            stem = s.stem
            if stem in used_stems:
                k = 1
                cand = f"{stem}_{k}"
                while cand in used_stems:
                    k += 1
                    cand = f"{stem}_{k}"
                stem = cand
            used_stems.add(stem)

            out_img = out_root / "img_dir" / split / f"{stem}.png"
            out_ann = out_root / "ann_dir" / split / f"{stem}.png"

            rgb = cv2.imread(str(s.rgb_path), cv2.IMREAD_COLOR)
            if rgb is None:
                print(f"[WARN] read fail rgb: {s.rgb_path}")
                continue
            mask = make_mmseg_mask(s.seg_path, s.label_path)
            if mask is None:
                continue

            cv2.imwrite(str(out_img), rgb)
            cv2.imwrite(str(out_ann), mask)
            names.append(stem)

            if i % max(1, args.log_every) == 0 or i == total:
                print(f"[{split}] [{i}/{total}] saved={len(names)}")

        with (out_root / "splits" / f"{split}.txt").open("w", encoding="utf-8") as f:
            for n in names:
                f.write(n + "\n")

    write_meta(out_root)
    print("\n=== Done ===")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()
