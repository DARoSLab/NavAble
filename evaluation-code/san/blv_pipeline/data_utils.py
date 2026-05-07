"""Dataset preprocessing and annotation helpers for the BLV pipeline."""

import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .constants import (
    BLV_CATEGORIES,
    BLV_NAME_TO_ID,
    IGNORE_INDEX,
    LABEL_STUDIO_TO_BLV,
    SYNTHETIC_LABEL_RULES,
)
from .runtime import write_json

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from pycocotools import mask as mask_utils
except ImportError:  # pragma: no cover
    mask_utils = None

IMAGE_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png'}


def ensure_cv2() -> None:
    if cv2 is None:
        raise ImportError('opencv-python is required for connected components.')


def sanitize_name(value: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]+', '_', value).strip('_').lower()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_split_ratio(text: str) -> Dict[str, float]:
    values = [float(item) for item in text.split(',')]
    if len(values) != 3:
        raise ValueError('split ratio must contain train,val,test fractions')
    total = sum(values)
    if total <= 0:
        raise ValueError('split ratio sum must be positive')
    normalized = [item / total for item in values]
    return dict(zip(('train', 'val', 'test'), normalized))


def parse_rgba_key(raw_key: str) -> Tuple[int, int, int, int]:
    values = [int(part.strip()) for part in raw_key.strip('()').split(',')]
    if len(values) != 4:
        raise ValueError(f'invalid RGBA key: {raw_key}')
    return tuple(values)


def pack_rgba(values: np.ndarray) -> np.ndarray:
    return (
        (values[..., 0].astype(np.uint32) << 24)
        | (values[..., 1].astype(np.uint32) << 16)
        | (values[..., 2].astype(np.uint32) << 8)
        | values[..., 3].astype(np.uint32)
    )


def packed_rgba_key(rgba: Tuple[int, int, int, int]) -> int:
    return (
        (int(rgba[0]) << 24)
        | (int(rgba[1]) << 16)
        | (int(rgba[2]) << 8)
        | int(rgba[3])
    )


def match_synthetic_label(source_folder: str, class_name: str) -> Optional[int]:
    name = class_name.lower()
    for rule in SYNTHETIC_LABEL_RULES.get(source_folder, []):
        if any(token in name for token in rule['contains']):
            return BLV_NAME_TO_ID[rule['blv_class']]
    return None


def build_synthetic_color_map(
    source_folder: str,
    label_mapping: Dict[str, Dict[str, str]],
) -> Dict[int, int]:
    color_map: Dict[int, int] = {}
    for rgba_text, metadata in label_mapping.items():
        class_name = metadata.get('class', '')
        matched = match_synthetic_label(source_folder, class_name)
        if matched is None:
            continue
        color_map[packed_rgba_key(parse_rgba_key(rgba_text))] = matched
    return color_map


def remap_rgba_mask(
    rgba_mask: np.ndarray,
    color_map: Dict[int, int],
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    output = np.full(rgba_mask.shape[:2], ignore_index, dtype=np.uint8)
    packed = pack_rgba(rgba_mask)
    for packed_color, class_id in color_map.items():
        output[packed == packed_color] = class_id
    return output


def load_index_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.uint8)


def save_index_mask(path: Path, mask: np.ndarray) -> None:
    ensure_dir(path.parent)
    Image.fromarray(mask.astype(np.uint8), mode='L').save(path)


def save_rgb_image(source_path: Path, destination_path: Path) -> None:
    ensure_dir(destination_path.parent)
    Image.open(source_path).convert('RGB').save(destination_path)


def encode_binary_mask(mask: np.ndarray, json_serializable: bool = False) -> Dict[str, object]:
    if mask_utils is None:
        return fallback_rle_encode(mask)

    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    if json_serializable and isinstance(encoded['counts'], bytes):
        encoded['counts'] = encoded['counts'].decode('ascii')
    return encoded


def fallback_rle_encode(mask: np.ndarray) -> Dict[str, object]:
    height, width = mask.shape
    flat = mask.astype(np.uint8).reshape(-1, order='F')
    counts: List[int] = []
    previous = 0
    run_length = 0

    for value in flat.tolist():
        if value != previous:
            counts.append(run_length)
            run_length = 1
            previous = value
        else:
            run_length += 1
    counts.append(run_length)

    return dict(size=[height, width], counts=counts)


def rle_to_binary_mask(rle: Dict[str, object]) -> np.ndarray:
    if mask_utils is not None and not isinstance(rle.get('counts'), list):
        decoded = mask_utils.decode(rle)
        if decoded.ndim == 3:
            decoded = decoded[..., 0]
        return decoded.astype(np.uint8)

    height, width = rle['size']
    flat = np.zeros(height * width, dtype=np.uint8)
    index = 0
    value = 0
    for count in rle['counts']:
        count = int(count)
        if count > 0 and value == 1:
            flat[index:index + count] = 1
        index += count
        value = 1 - value
    return flat.reshape((height, width), order='F')


def rle_iou(prediction_rle: Dict[str, object], ground_truth_rles: Sequence[Dict[str, object]]) -> np.ndarray:
    if not ground_truth_rles:
        return np.array([], dtype=np.float64)

    if mask_utils is not None and not isinstance(prediction_rle.get('counts'), list):
        return mask_utils.iou(
            [prediction_rle],
            list(ground_truth_rles),
            [0] * len(ground_truth_rles),
        )[0]

    pred_mask = rle_to_binary_mask(prediction_rle).astype(bool)
    ious = []
    for ground_truth_rle in ground_truth_rles:
        gt_mask = rle_to_binary_mask(ground_truth_rle).astype(bool)
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        ious.append(0.0 if union == 0 else float(intersection / union))
    return np.asarray(ious, dtype=np.float64)


def extract_semantic_instances(
    mask: np.ndarray,
    score_maps: Optional[np.ndarray] = None,
    ignore_index: int = IGNORE_INDEX,
    json_serializable: bool = False,
) -> List[Dict[str, object]]:
    ensure_cv2()
    instances: List[Dict[str, object]] = []
    valid_classes = [int(value) for value in np.unique(mask) if value != ignore_index]

    for class_id in valid_classes:
        binary = (mask == class_id).astype(np.uint8)
        if binary.sum() == 0:
            continue
        num_components, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        for component_idx in range(1, num_components):
            component_mask = labels == component_idx
            area = int(stats[component_idx, cv2.CC_STAT_AREA])
            x = int(stats[component_idx, cv2.CC_STAT_LEFT])
            y = int(stats[component_idx, cv2.CC_STAT_TOP])
            width = int(stats[component_idx, cv2.CC_STAT_WIDTH])
            height = int(stats[component_idx, cv2.CC_STAT_HEIGHT])
            instance = dict(
                category_id=class_id,
                area=area,
                bbox=[x, y, width, height],
                rle=encode_binary_mask(component_mask, json_serializable=json_serializable),
            )
            if score_maps is not None:
                instance['score'] = float(score_maps[class_id][component_mask].mean())
            instances.append(instance)
    return instances


def instances_to_coco_annotations(
    instances: Sequence[Dict[str, object]],
    image_id: int,
    ann_id_start: int,
) -> Tuple[List[Dict[str, object]], int]:
    annotations: List[Dict[str, object]] = []
    ann_id = ann_id_start
    for instance in instances:
        annotations.append(
            dict(
                id=ann_id,
                image_id=image_id,
                category_id=int(instance['category_id']),
                segmentation=instance['rle'],
                area=int(instance['area']),
                bbox=[int(value) for value in instance['bbox']],
                iscrowd=0,
            )
        )
        ann_id += 1
    return annotations, ann_id


def coco_payload(
    images: Sequence[Dict[str, object]],
    annotations: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    return dict(
        images=list(images),
        annotations=list(annotations),
        categories=BLV_CATEGORIES,
    )


def write_coco_json(
    path: Path,
    images: Sequence[Dict[str, object]],
    annotations: Sequence[Dict[str, object]],
) -> None:
    write_json(path, coco_payload(images, annotations))


def apply_class_lut(mask: np.ndarray, class_map: Optional[Dict[int, int]]) -> np.ndarray:
    if not class_map:
        return mask.astype(np.uint8)
    output = np.full(mask.shape, IGNORE_INDEX, dtype=np.uint8)
    for source_id, target_id in class_map.items():
        output[mask == source_id] = target_id
    return output


def class_pixel_counts(mask: np.ndarray) -> Counter:
    counts = Counter()
    values, pixel_counts = np.unique(mask, return_counts=True)
    for class_id, count in zip(values.tolist(), pixel_counts.tolist()):
        if class_id == IGNORE_INDEX:
            continue
        counts[int(class_id)] += int(count)
    return counts


def list_images(directory: Path) -> Dict[str, Path]:
    files = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file():
            files[path.stem] = path
    return files


def list_png_masks(directory: Path) -> Dict[str, Path]:
    files = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() == '.png' and path.is_file():
            files[path.stem] = path
    return files


def find_candidate_subdir(root: Path, names: Sequence[str]) -> Optional[Path]:
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def discover_real_source(root: Path) -> Dict[str, List[Dict[str, object]]]:
    img_dir = root / 'img_dir'
    ann_dir = root / 'ann_dir'
    if img_dir.is_dir() and ann_dir.is_dir():
        discovered: Dict[str, List[Dict[str, object]]] = {}
        for split in ('train', 'val', 'test'):
            split_imgs = img_dir / split
            split_masks = ann_dir / split
            if split_imgs.is_dir() and split_masks.is_dir():
                discovered[split] = collect_pairs(split_imgs, split_masks)
        if discovered:
            return discovered

    image_root = find_candidate_subdir(root, ('images', 'imgs', 'img', 'rgb'))
    mask_root = find_candidate_subdir(root, ('masks', 'mask', 'labels', 'annotations', 'ann'))
    if image_root is None or mask_root is None:
        raise FileNotFoundError(
            'Could not locate image and mask directories under '
            f'{root}. Expected either img_dir/ann_dir or common names like '
            'images/ and masks/.'
        )
    return {'all': collect_pairs(image_root, mask_root)}


def collect_pairs(image_root: Path, mask_root: Path) -> List[Dict[str, object]]:
    images = list_images(image_root)
    masks = list_png_masks(mask_root)
    shared = sorted(set(images) & set(masks))
    return [
        dict(stem=stem, image_path=images[stem], mask_path=masks[stem])
        for stem in shared
    ]


def stratified_split(
    pairs: Sequence[Dict[str, object]],
    ratios: Dict[str, float],
    seed: int = 42,
) -> Dict[str, List[Dict[str, object]]]:
    rng = random.Random(seed)
    splits = list(ratios)
    total_items = len(pairs)
    target_sizes = {split: math.floor(ratios[split] * total_items) for split in splits}
    remainder = total_items - sum(target_sizes.values())
    for split in sorted(splits, key=lambda item: ratios[item], reverse=True):
        if remainder <= 0:
            break
        target_sizes[split] += 1
        remainder -= 1

    class_presence: Dict[str, Tuple[int, ...]] = {}
    class_totals = Counter()
    for pair in pairs:
        mask = load_index_mask(Path(pair['mask_path']))
        present = tuple(
            int(class_id) for class_id in np.unique(mask) if class_id != IGNORE_INDEX
        )
        class_presence[pair['stem']] = present
        class_totals.update(present)

    split_targets = {
        split: {class_id: class_totals[class_id] * ratios[split] for class_id in class_totals}
        for split in splits
    }
    split_sizes = Counter()
    split_class_counts: Dict[str, Counter] = defaultdict(Counter)
    assignments: Dict[str, List[Dict[str, object]]] = {split: [] for split in splits}

    ordered_pairs = sorted(
        pairs,
        key=lambda pair: (len(class_presence[pair['stem']]), sum(class_totals[c] for c in class_presence[pair['stem']])),
        reverse=True,
    )

    for pair in ordered_pairs:
        present = class_presence[pair['stem']]
        best_split = None
        best_score = None
        for split in splits:
            remaining = target_sizes[split] - split_sizes[split]
            if remaining <= 0:
                continue
            score = remaining / max(target_sizes[split], 1)
            for class_id in present:
                deficit = split_targets[split][class_id] - split_class_counts[split][class_id]
                score += max(deficit, 0.0)
            score += rng.random() * 1e-6
            if best_score is None or score > best_score:
                best_score = score
                best_split = split
        if best_split is None:
            best_split = min(splits, key=lambda split: split_sizes[split])
        assignments[best_split].append(pair)
        split_sizes[best_split] += 1
        split_class_counts[best_split].update(present)

    return assignments


def load_class_map(path: Optional[str]) -> Optional[Dict[int, int]]:
    if not path:
        return None
    with Path(path).expanduser().open('r', encoding='utf-8') as handle:
        raw = json.load(handle)
    return {int(key): int(value) for key, value in raw.items()}


# ---------------------------------------------------------------------------
# Label Studio parsing
# ---------------------------------------------------------------------------


def _decode_ls_rle(rle_list: List[int], height: int, width: int) -> np.ndarray:
    """Decode a Label Studio brush-tool RLE into a binary mask.

    Label Studio uses a byte-level RLE where each pair of values gives
    (count, pixel_value) starting from the top-left in row-major order.
    The values are packed per-channel; for brushlabels the mask is single-
    channel so the total decoded length should equal height * width.
    """
    pixels: List[int] = []
    idx = 0
    while idx < len(rle_list):
        if idx + 1 >= len(rle_list):
            break
        count = rle_list[idx]
        value = rle_list[idx + 1]
        pixels.extend([value] * count)
        idx += 2

    flat = np.array(pixels, dtype=np.uint8)
    expected = height * width * 4
    if flat.size == expected:
        flat = flat.reshape(height, width, 4)
        return (flat[..., 3] > 127).astype(np.uint8)
    expected_single = height * width
    if flat.size >= expected_single:
        flat = flat[:expected_single]
    else:
        flat = np.pad(flat, (0, expected_single - flat.size))
    return (flat.reshape(height, width) > 127).astype(np.uint8)


def parse_label_studio_json(
    json_path: Path,
    image_root: Optional[Path] = None,
    label_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, object]]:
    """Parse a Label Studio JSON export into a list of (image_path, masks) dicts.

    Returns a list of dicts, each with keys:
      - ``image_path``: resolved path to the source image
      - ``masks``: list of ``(blv_class_name, binary_mask_np)`` tuples
      - ``width``, ``height``
    """
    if label_map is None:
        label_map = LABEL_STUDIO_TO_BLV

    with json_path.open('r', encoding='utf-8') as fh:
        raw = json.load(fh)

    if isinstance(raw, dict):
        raw = [raw]

    entries: List[Dict[str, object]] = []
    for task in raw:
        image_ref = task.get('data', {}).get('image', '')
        annotations = task.get('annotations', [])
        if not annotations:
            continue

        masks: List[Tuple[str, np.ndarray]] = []
        width = height = 0
        for annotation in annotations:
            for result in annotation.get('result', []):
                if result.get('type') != 'brushlabels':
                    continue
                value = result.get('value', {})
                rle = value.get('rle')
                labels = value.get('brushlabels', [])
                h = result.get('original_height', value.get('original_height', 0))
                w = result.get('original_width', value.get('original_width', 0))
                if not rle or not labels or not h or not w:
                    continue
                width, height = w, h
                for label in labels:
                    blv_name = label_map.get(label)
                    if blv_name is None:
                        continue
                    binary = _decode_ls_rle(rle, h, w)
                    masks.append((blv_name, binary))

        if not masks or not width:
            continue

        if image_root is not None:
            img_filename = Path(image_ref).name
            resolved = image_root / img_filename
        else:
            resolved = Path(image_ref)

        entries.append(dict(
            image_path=resolved,
            masks=masks,
            width=width,
            height=height,
        ))

    return entries


def label_studio_to_dataset(
    entries: Iterable[Dict[str, object]],
    out_dir: Path,
    split_name: str = 'all',
) -> List[Dict[str, object]]:
    """Convert parsed Label Studio entries into img_dir / ann_dir layout.

    Returns a list of ``{stem, image_path, mask_path}`` pair dicts compatible
    with the rest of the preprocessing pipeline.
    """
    img_out = ensure_dir(out_dir / 'img_dir' / split_name)
    mask_out = ensure_dir(out_dir / 'ann_dir' / split_name)
    pairs: List[Dict[str, object]] = []

    for entry in entries:
        src_path = Path(entry['image_path'])
        stem = sanitize_name(src_path.stem)
        output_name = f'{stem}.png'

        semantic = np.full(
            (entry['height'], entry['width']), IGNORE_INDEX, dtype=np.uint8,
        )
        for blv_name, binary in entry['masks']:
            class_id = BLV_NAME_TO_ID.get(blv_name)
            if class_id is None:
                continue
            semantic[binary > 0] = class_id

        if (semantic == IGNORE_INDEX).all():
            continue

        if src_path.is_file():
            save_rgb_image(src_path, img_out / output_name)
        save_index_mask(mask_out / output_name, semantic)
        pairs.append(dict(
            stem=stem,
            image_path=str(img_out / output_name),
            mask_path=str(mask_out / output_name),
        ))

    return pairs
