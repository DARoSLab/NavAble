"""SAM heuristic masking strategy -- original baseline approach.

Uses SAM's automatic mask generator with heuristic scoring
(size preference + center proximity) to select the best object mask.
No class awareness -- kept as a baseline for comparison.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from isaacnav.masking.base import BaseMaskingStrategy, MaskResult


class SamHeuristicMasking(BaseMaskingStrategy):
    """Heuristic-based masking using SAM automatic mask generator."""

    def __init__(self, config: dict):
        self.config = config
        self.checkpoint = config.get(
            "checkpoint", "extern/sam-3d-objects/checkpoints/sam_vit_h_4b8939.pth"
        )
        self.model_type = config.get("model_type", "vit_h")
        self.points_per_side = config.get("points_per_side", 32)
        self.pred_iou_thresh = config.get("pred_iou_thresh", 0.88)
        self.stability_score_thresh = config.get("stability_score_thresh", 0.95)
        self.min_mask_region_area = config.get("min_mask_region_area", 500)
        self.coverage_range = config.get("coverage_range", [0.05, 0.70])
        self.preferred_coverage = config.get("preferred_coverage", 0.30)
        self.center_weight = config.get("center_weight", 0.6)
        self.size_weight = config.get("size_weight", 0.4)
        self._model = None
        self._mask_generator = None

    def load_model(self) -> None:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        checkpoint = str(Path(self.checkpoint).resolve())

        # Download checkpoint if not exists
        if not os.path.exists(checkpoint):
            print(f"Downloading SAM checkpoint to {checkpoint}...")
            import urllib.request

            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
            urllib.request.urlretrieve(url, checkpoint)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = sam_model_registry[self.model_type](checkpoint=checkpoint)
        self._model.to(device)

        self._mask_generator = SamAutomaticMaskGenerator(
            self._model,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=self.min_mask_region_area,
        )

    def segment(
        self,
        image: np.ndarray,
        class_names: Optional[list[str]] = None,
        point_prompts: Optional[list[tuple[int, int]]] = None,
    ) -> list[MaskResult]:
        if self._mask_generator is None:
            self.load_model()

        img_area = image.shape[0] * image.shape[1]
        img_center = np.array([image.shape[1] / 2, image.shape[0] / 2])

        print("Generating masks with SAM (heuristic mode)...")
        masks = self._mask_generator.generate(image)

        if not masks:
            raise ValueError("No masks found in image")

        # Sort by area
        masks = sorted(masks, key=lambda x: x["area"], reverse=True)
        print(f"Found {len(masks)} masks")

        min_cov, max_cov = self.coverage_range
        best_mask = None
        best_score = -1

        for mask_data in masks:
            mask = mask_data["segmentation"]
            coverage = mask_data["area"] / img_area

            if coverage > max_cov or coverage < min_cov:
                continue

            y_indices, x_indices = np.where(mask)
            if len(x_indices) == 0:
                continue
            mask_center = np.array([x_indices.mean(), y_indices.mean()])

            size_score = 1.0 - abs(coverage - self.preferred_coverage)
            center_dist = np.linalg.norm(mask_center - img_center) / np.linalg.norm(
                img_center
            )
            center_score = 1.0 - min(center_dist, 1.0)

            score = size_score * self.size_weight + center_score * self.center_weight

            if score > best_score:
                best_score = score
                best_mask = mask_data

        # Fallback: largest non-background mask
        if best_mask is None:
            print("Warning: Could not find ideal object mask, using largest non-background mask")
            for mask_data in masks:
                if mask_data["area"] / img_area < 0.80:
                    best_mask = mask_data
                    break

        if best_mask is None:
            raise ValueError("Could not find suitable object mask")

        coverage = best_mask["area"] / img_area * 100
        print(f"Selected mask covering {coverage:.1f}% of image")

        # Use first class name if provided, otherwise None
        label = class_names[0] if class_names else None

        return [
            MaskResult(
                mask=best_mask["segmentation"],
                class_name=label,
                confidence=best_score,
                bbox=tuple(best_mask.get("bbox", ())),
                source_strategy="sam_heuristic",
            )
        ]
