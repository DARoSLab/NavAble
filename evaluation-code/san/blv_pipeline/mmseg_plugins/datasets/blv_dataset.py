"""BLV dataset plugins for MMSegmentation.

Three dataset classes are exposed:

* ``BLVDataset`` — legacy 10-class schema (no explicit background, ignore=255).
  Kept for backward compatibility with old configs / old checkpoints.

* ``BLVDatasetV2All`` — new 12-class schema with explicit background as
  class 0 (V2). Used by SegFormer where bg is a trainable class.

* ``BLVDatasetV2Fg`` — V2 schema with ``reduce_zero_label=True`` to drop bg
  and shift fg classes down by 1. Yields 11 fg classes (indices 0..10) with
  bg pixels mapped to ignore_index=255. Used by Mask2Former / SAN to
  replicate the validated old training behavior on the new data.
"""

from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS

from blv_pipeline.constants import (
    BLV_CLASSES,
    BLV_PALETTE,
    BLV_V2_CLASSES_ALL,
    BLV_V2_CLASSES_FG,
    BLV_V2_PALETTE_ALL,
    BLV_V2_PALETTE_FG,
)


@DATASETS.register_module()
class BLVDataset(BaseSegDataset):
    """Legacy 10-class BLV dataset (no explicit background)."""

    METAINFO = dict(classes=BLV_CLASSES, palette=BLV_PALETTE)

    def __init__(self, img_suffix: str = '.png', seg_map_suffix: str = '.png', **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=False,
            **kwargs,
        )


@DATASETS.register_module()
class BLVDatasetV2All(BaseSegDataset):
    """V2 schema: 12 classes with explicit background as class 0.

    Used by SegFormer training where bg supervision is needed for the
    per-pixel classifier.
    """

    METAINFO = dict(classes=BLV_V2_CLASSES_ALL, palette=BLV_V2_PALETTE_ALL)

    def __init__(self, img_suffix: str = '.png', seg_map_suffix: str = '.png', **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=False,
            **kwargs,
        )


@DATASETS.register_module()
class BLVDatasetV2Fg(BaseSegDataset):
    """V2 schema with foreground-only labels (bg dropped via reduce_zero_label).

    With ``reduce_zero_label=True``, mmseg's LoadAnnotations applies:
        bg(0) → 255 (ignore), classes 1..11 → 0..10
    yielding 11 fg classes that match the validated Mask2Former / SAN
    training pipeline (which expects num_classes=11 + matcher no-object slot).
    """

    METAINFO = dict(classes=BLV_V2_CLASSES_FG, palette=BLV_V2_PALETTE_FG)

    def __init__(self, img_suffix: str = '.png', seg_map_suffix: str = '.png', **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=True,
            **kwargs,
        )
