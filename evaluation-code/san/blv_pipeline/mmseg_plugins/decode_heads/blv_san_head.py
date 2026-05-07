"""SAN decode head with per-query filtering for BLV inference.

Same architecture-level fix as BLVMask2FormerHead: per-query filtering +
constant bg channel.  See ``blv_mask2former_head.py`` docstring for the
full rationale.

Set ``query_fg_threshold=0.0`` to disable per-query filtering.
Set ``bg_threshold=0.0`` to disable the background floor.
"""

from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor

from mmseg.models.decode_heads.san_head import SideAdapterCLIPHead
from mmseg.registry import MODELS


@MODELS.register_module()
class BLVSideAdapterCLIPHead(SideAdapterCLIPHead):
    """SAN head with per-query fg filtering at inference."""

    def __init__(self, query_fg_threshold: float = 0.1,
                 bg_threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.query_fg_threshold = query_fg_threshold
        self.bg_threshold = bg_threshold

    def predict_by_feat(self, seg_logits: List[Tensor],
                        batch_img_metas: List[dict]) -> Tensor:
        mask_pred = seg_logits[0]
        cls_score = seg_logits[1]

        if isinstance(batch_img_metas[0]['img_shape'], torch.Size):
            size = batch_img_metas[0]['img_shape']
        elif 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape'][:2]
        else:
            size = batch_img_metas[0]['img_shape']

        mask_pred = F.interpolate(
            mask_pred, size=size, mode='bilinear', align_corners=False)

        mask_cls = F.softmax(cls_score, dim=-1)       # [B, Q, C+1]
        fg_score = mask_cls[..., :-1]                  # [B, Q, C] strip bg

        # Per-query fg confidence: zero out bg-classified queries
        if self.query_fg_threshold > 0.0:
            query_fg_conf = fg_score.max(dim=-1).values  # [B, Q]
            keep = (query_fg_conf > self.query_fg_threshold).unsqueeze(-1)
            fg_score = fg_score * keep.float()

        mask_pred = mask_pred.sigmoid()
        seg_logits = torch.einsum('bqc,bqhw->bchw', fg_score, mask_pred)

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
