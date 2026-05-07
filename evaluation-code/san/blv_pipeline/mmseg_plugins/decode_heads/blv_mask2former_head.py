"""Mask2Former decode head with per-query filtering for BLV inference.

Mask-classification heads use Q=100 queries with a Hungarian matcher. At
inference, ~98 queries match "no object" (background) while only ~2 match
actual fg objects. The standard einsum aggregation sums all queries, causing
the bg channel to dominate.

``BLVMask2FormerHead`` applies two fixes:

1. **Per-query filtering**: Zeros out bg-classified queries (those with
   max fg confidence below ``query_fg_threshold``) before the einsum,
   and strips the learned bg channel from the classification scores.

2. **Constant bg channel**: Appends a synthetic background channel with
   a fixed value ``bg_threshold`` after the einsum.  At pixels where no
   fg class exceeds this value, the bg channel wins argmax and the
   downstream metric maps those pixels to ``ignore_index``.  This
   prevents the "argmax-over-zeros" flooding where every pixel gets
   assigned to whichever fg class has the largest floating-point residual.

Set ``query_fg_threshold=0.0`` to disable per-query filtering.
Set ``bg_threshold=0.0`` to disable the background floor.
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from mmseg.models.decode_heads.mask2former_head import Mask2FormerHead
from mmseg.registry import MODELS
from mmseg.structures.seg_data_sample import SegDataSample
from mmseg.utils import ConfigType


@MODELS.register_module()
class BLVMask2FormerHead(Mask2FormerHead):
    """Mask2Former head with per-query fg filtering at inference."""

    def __init__(self, query_fg_threshold: float = 0.1,
                 bg_threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.query_fg_threshold = query_fg_threshold
        self.bg_threshold = bg_threshold

    def predict(self, x: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tuple[Tensor]:
        batch_data_samples = [
            SegDataSample(metainfo=metainfo) for metainfo in batch_img_metas
        ]

        all_cls_scores, all_mask_preds = self(x, batch_data_samples)
        mask_cls_results = all_cls_scores[-1]       # [B, Q, C+1]
        mask_pred_results = all_mask_preds[-1]      # [B, Q, H, W]

        if 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape']
        else:
            size = batch_img_metas[0]['img_shape']

        mask_pred_results = F.interpolate(
            mask_pred_results, size=size, mode='bilinear', align_corners=False)

        cls_score = F.softmax(mask_cls_results, dim=-1)  # [B, Q, C+1]
        fg_score = cls_score[..., :-1]                    # [B, Q, C] strip bg

        # Per-query fg confidence: zero out bg-classified queries
        if self.query_fg_threshold > 0.0:
            query_fg_conf = fg_score.max(dim=-1).values   # [B, Q]
            keep = (query_fg_conf > self.query_fg_threshold).unsqueeze(-1)
            fg_score = fg_score * keep.float()

        mask_pred = mask_pred_results.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', fg_score, mask_pred)

        # Append constant bg channel as confidence floor
        if self.bg_threshold > 0.0:
            bg = torch.full(
                (seg_logits.shape[0], 1, *seg_logits.shape[2:]),
                self.bg_threshold,
                dtype=seg_logits.dtype,
                device=seg_logits.device,
            )
            seg_logits = torch.cat([seg_logits, bg], dim=1)

        return seg_logits
