"""Shared constants for the BLV segmentation pipeline."""

from typing import Dict, List, Tuple

IGNORE_INDEX = 255

BLV_CLASSES: Tuple[str, ...] = (
    'elevator',
    'elevator_button',
    'door_button',
    'door',
    'crosswalk',
    'traffic_signal',
    'pedestrian_signal',
    'push_button',
    'bus_stop',
    'bus_stop_sign',
)

BLV_PALETTE: List[Tuple[int, int, int]] = [
    (214, 39, 40),
    (255, 127, 14),
    (148, 103, 189),
    (140, 86, 75),
    (44, 160, 44),
    (31, 119, 180),
    (23, 190, 207),
    (227, 119, 194),
    (188, 189, 34),
    (127, 127, 127),
]

BLV_NAME_TO_ID: Dict[str, int] = {
    class_name: idx for idx, class_name in enumerate(BLV_CLASSES)
}

BLV_CATEGORIES = [
    dict(id=class_id, name=class_name, supercategory='blv')
    for class_name, class_id in BLV_NAME_TO_ID.items()
]

# Natural-language class names for CLIP-based models (SAN).
BLV_VOCABULARY: Tuple[str, ...] = (
    'elevator', 'elevator button', 'door button', 'door',
    'crosswalk', 'traffic signal', 'pedestrian signal',
    'push button', 'bus stop', 'bus stop sign',
)

# ---------------------------------------------------------------------------
# V2 schema (NeurIPS 2026 release): 12-class, explicit background as id=0.
# Used for opensrc / new synthetic / new real datasets.
# ---------------------------------------------------------------------------
BLV_V2_CLASSES_ALL: Tuple[str, ...] = (
    'background',          # 0
    'elevator',            # 1
    'elevator_button',     # 2
    'door_button',         # 3
    'crosswalk',           # 4
    'pedestrian_signal',   # 5
    'aps_button',          # 6
    'bus_stop',             # 7
    'bus_stop_sign',       # 8
    'handrail',            # 9
    'escalator',           # 10
    'turnstile',           # 11
)

# Foreground-only view (after reduce_zero_label drops bg). Indices 0..10
# correspond to BLV_V2_CLASSES_ALL[1..11] respectively.
BLV_V2_CLASSES_FG: Tuple[str, ...] = BLV_V2_CLASSES_ALL[1:]

BLV_V2_PALETTE_ALL: List[Tuple[int, int, int]] = [
    (0, 0, 0),             # background — black
    (214, 39, 40),         # elevator
    (255, 127, 14),        # elevator_button
    (148, 103, 189),       # door_button
    (44, 160, 44),         # crosswalk
    (23, 190, 207),        # pedestrian_signal
    (227, 119, 194),       # aps_button
    (188, 189, 34),        # bus_stop
    (127, 127, 127),       # bus_stop_sign
    (140, 86, 75),         # handrail
    (31, 119, 180),        # escalator
    (174, 199, 232),       # turnstile
]

BLV_V2_PALETTE_FG: List[Tuple[int, int, int]] = BLV_V2_PALETTE_ALL[1:]

# Natural-language vocabulary for CLIP-based models (SAN), V2 fg-only.
BLV_V2_VOCABULARY_FG: Tuple[str, ...] = (
    'elevator', 'elevator button', 'door button',
    'crosswalk', 'pedestrian signal', 'accessible pedestrian signal button',
    'bus stop', 'bus stop sign', 'handrail', 'escalator', 'turnstile',
)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MMSEG_MODELS: Dict[str, Dict[str, str]] = {
    'mask2former': dict(
        checkpoint='mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.pth',
    ),
    'segformer': dict(
        checkpoint='segformer_mit-b5_640x640_160k_ade20k_20210801_121243-41d2845b.pth',
    ),
    'san': dict(
        checkpoint='san-vit-b16_20230906-fd0a7684.pth',
    ),
}

EXTERNAL_MODELS: Dict[str, str] = {
    'yolov11-seg-n': 'ultralytics',
    'yolov11-seg-x': 'ultralytics',
    'sam2.1-unet': 'custom',
    'dinov3-regcls': 'custom',
    'dinov3-eomt': 'custom',
}

ALL_MODELS = list(MMSEG_MODELS) + list(EXTERNAL_MODELS)

# ---------------------------------------------------------------------------
# Label Studio brush-label → BLV class name mapping
# ---------------------------------------------------------------------------
LABEL_STUDIO_TO_BLV: Dict[str, str] = {
    'Elevator_Button': 'elevator_button',
    'Elevator_Door': 'elevator',
    'Elevator': 'elevator',
    'Door_Button': 'door_button',
    'Door': 'door',
    'Crosswalk': 'crosswalk',
    'Traffic_Signal': 'traffic_signal',
    'Pedestrian_Signal': 'pedestrian_signal',
    'Push_Button': 'push_button',
    'Bus_Stop': 'bus_stop',
    'Bus_Stop_Sign': 'bus_stop_sign',
}

# ---------------------------------------------------------------------------
# Data paths  (override via env vars for portability across machines)
# ---------------------------------------------------------------------------
import os as _os
from pathlib import Path as _Path

PROJECT_ROOT = _os.environ.get(
    'BLV_PROJECT_ROOT',
    str(_Path(__file__).resolve().parents[1]),
)

_NAVABLE_ROOT = _os.environ.get(
    'NAVABLE_ROOT',
    str(_Path(PROJECT_ROOT).parent / 'NavAble'),
)

REAL_DATA_SOURCE = _os.environ.get(
    'BLV_REAL_DATA_SOURCE',
    _os.path.join(_NAVABLE_ROOT, 'RealData'),
)
SYNTHETIC_DATA_SOURCE = _os.environ.get(
    'BLV_SYNTHETIC_DATA_SOURCE',
    _os.path.join(_NAVABLE_ROOT, 'IsaacSim', 'data'),
)

PROJECT_DATA_ROOT = _os.environ.get(
    'BLV_PROJECT_DATA_ROOT',
    _os.path.join(PROJECT_ROOT, 'data'),
)
REAL_DATASET_ROOT = _os.path.join(PROJECT_DATA_ROOT, 'real')
SYNTHETIC_DATASET_ROOT = _os.path.join(PROJECT_DATA_ROOT, 'synthetic')

# ---------------------------------------------------------------------------
# Checkpoint paths
# ---------------------------------------------------------------------------
PRETRAINED_CKPT_DIR = _os.path.join(PROJECT_ROOT, 'checkpoints', 'pretrained')
FINETUNED_CKPT_DIR = _os.path.join(PROJECT_ROOT, 'checkpoints', 'finetuned')

PRETRAINED_CHECKPOINTS: Dict[str, str] = {
    'mask2former': (
        f'{PRETRAINED_CKPT_DIR}/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.pth'
    ),
    'segformer': (
        f'{PRETRAINED_CKPT_DIR}/segformer_mit-b5_640x640_160k_ade20k_20210801_121243-41d2845b.pth'
    ),
    'san_b16': (
        f'{PRETRAINED_CKPT_DIR}/san-vit-b16_20230906-fd0a7684.pth'
    ),
    'san_l14': (
        f'{PRETRAINED_CKPT_DIR}/san-vit-l14_20230907-a11e098f.pth'
    ),
}

CHECKPOINT_URLS: Dict[str, str] = {
    'mask2former': (
        'https://download.openmmlab.com/mmsegmentation/v0.5/mask2former/'
        'mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640/'
        'mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640'
        '_20221203_235933-7120c214.pth'
    ),
    'segformer': (
        'https://download.openmmlab.com/mmsegmentation/v0.5/segformer/'
        'segformer_mit-b5_640x640_160k_ade20k/'
        'segformer_mit-b5_640x640_160k_ade20k_20210801_121243-41d2845b.pth'
    ),
    'san_b16': (
        'https://download.openmmlab.com/mmsegmentation/v0.5/san/'
        'san-vit-b16_20230906-fd0a7684.pth'
    ),
    'san_l14': (
        'https://download.openmmlab.com/mmsegmentation/v0.5/san/'
        'san-vit-l14_20230907-a11e098f.pth'
    ),
    'san_clip_b16': (
        'https://download.openmmlab.com/mmsegmentation/v0.5/san/'
        'clip_vit-base-patch16-224_3rdparty-d08f8887.pth'
    ),
    'san_clip_l14': (
        'https://download.openmmlab.com/mmsegmentation/v0.5/san/'
        'clip-vit-large-patch14_3rdparty-d08f8887.pth'
    ),
}

# Folder-scoped substring rules for Isaac Sim semantic label strings.
# Exact class strings confirmed by Anh Nguyen (dataset owner, Apr 2026).
# Each token is checked as a substring of the lowercased class name from the
# per-frame semantic_segmentation_labels_*.json files.
# Within the 'elevator & elevator_button' folder the elevator_button rule MUST
# come before elevator (both share the 'elevator' substring; first match wins).
SYNTHETIC_LABEL_RULES = {
    'pedestrian_button': [
        dict(
            blv_class='push_button',
            # Run1-10: "nyccrosswalk,nyccrosswalkbodypivot"
            # Run31-45: "mesh"
            contains=(
                'nyccrosswalk',
                'mesh',
            ),
        ),
    ],
    'bus_stop': [
        dict(
            blv_class='bus_stop',
            contains=(
                'busstopshelterbusstopsheltermaterial',
                'busstopshelter2busstopshelter2material',
                'busstopshelter3busstopshelter3material',
                'busstopshelter4busstopshelter4material',
                'busstopshelter5busstopshelter5material',
                'busstopshelter6busstopshelter6material',
            ),
        ),
    ],
    'bus_stop_sign': [
        dict(
            blv_class='bus_stop_sign',
            # "section,stsigns" and "section,stsignsmetal"
            contains=(
                'section,stsigns',
                'section,stsignsmetal',
            ),
        ),
    ],
    'crosswalk': [
        dict(
            blv_class='crosswalk',
            # Run46,49-55: "marking,pedestriancrossing"
            # Run47,48: "pedestriancrossing"
            contains=('pedestriancrossing',),
        ),
    ],
    'elevator & elevator_button': [
        dict(
            blv_class='elevator_button',
            # Run56,57: "elevator_button,elevatorrequestbuttons"
            # Run58,59: "elevator_button,elevatorbutton"
            # Run60: "buttonlight,elevator_button"
            contains=(
                'elevator_button,elevatorrequestbuttons',
                'elevator_button,elevatorbutton',
                'buttonlight,elevator_button',
            ),
        ),
        dict(
            blv_class='elevator',
            # Run56,57: "brushedsteel,elevator"
            # Run58,59: "elevator,elevatordoor"
            # Run60: "elevator,multisurface4k"
            contains=(
                'brushedsteel,elevator',
                'elevator,elevatordoor',
                'elevator,multisurface4k',
            ),
        ),
    ],
}
