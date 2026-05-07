"""Dataset transform to remap segmentation labels."""

from typing import Dict

import numpy as np
from mmcv.transforms import BaseTransform

from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RemapSegLabel(BaseTransform):
    """Remap one segmentation label id to another.

    This is used to turn ignored background pixels (255) into an explicit
    trainable background class index (e.g., 10) for SegFormer retraining.
    """

    def __init__(self, src_label: int = 255, dst_label: int = 10) -> None:
        self.src_label = int(src_label)
        self.dst_label = int(dst_label)

    def transform(self, results: Dict) -> Dict:
        gt_seg_map = results.get('gt_seg_map', None)
        if gt_seg_map is None:
            return results

        remapped = gt_seg_map.copy()
        remapped[remapped == self.src_label] = self.dst_label
        results['gt_seg_map'] = remapped.astype(np.uint8)
        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(src_label={self.src_label}, "
            f"dst_label={self.dst_label})"
        )
