"""Abstract base class for masking/segmentation strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


@dataclass
class MaskResult:
    """Result from a masking strategy."""

    mask: np.ndarray  # Boolean H x W array
    class_name: Optional[str]  # Detected/requested class label
    confidence: float  # Confidence score (0.0 - 1.0)
    bbox: Optional[tuple]  # (x, y, w, h) bounding box
    source_strategy: str  # Name of the strategy that produced this


class BaseMaskingStrategy(ABC):
    """Abstract base class for all masking/segmentation strategies."""

    @abstractmethod
    def __init__(self, config: dict):
        """Initialize with strategy-specific config from pipeline.yaml."""
        ...

    @abstractmethod
    def segment(
        self,
        image: np.ndarray,
        class_names: Optional[list[str]] = None,
        point_prompts: Optional[list[tuple[int, int]]] = None,
    ) -> list[MaskResult]:
        """
        Segment objects in the image.

        Args:
            image: RGB numpy array (H, W, 3), dtype uint8.
            class_names: Optional list of target class labels for
                         class-aware strategies (e.g., ["elevator", "door"]).
            point_prompts: Optional list of (x, y) pixel coordinates for
                           point-prompt strategies.

        Returns:
            List of MaskResult, one per detected object, sorted by
            confidence descending.
        """
        ...

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights. Separate from __init__ to allow lazy loading."""
        ...

    def save_masks(
        self, results: list[MaskResult], output_dir: Path, image_stem: str
    ) -> list[Path]:
        """Save mask results as PNG files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, result in enumerate(results):
            label = result.class_name or f"mask_{i}"
            # Sanitize label for filename
            safe_label = label.replace(" ", "_").replace("/", "_").lower()
            filename = f"{image_stem}_{safe_label}.mask.png"
            mask_path = output_dir / filename
            Image.fromarray((result.mask * 255).astype(np.uint8)).save(mask_path)
            paths.append(mask_path)
        return paths
