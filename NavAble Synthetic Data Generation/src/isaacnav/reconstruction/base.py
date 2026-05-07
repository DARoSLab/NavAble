"""Abstract base class for 3D reconstruction strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ReconstructionResult:
    """Result from a 3D reconstruction strategy."""

    ply_path: Optional[Path] = None  # Gaussian splat PLY
    glb_path: Optional[Path] = None  # Textured mesh GLB
    mesh_ply_path: Optional[Path] = None  # Mesh-only PLY
    metadata: dict = field(default_factory=dict)
    source_strategy: str = ""


class BaseReconstructionStrategy(ABC):
    """Abstract base class for image-to-3D reconstruction strategies."""

    @abstractmethod
    def __init__(self, config: dict):
        """Initialize with strategy-specific config."""
        ...

    @abstractmethod
    def reconstruct(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        output_dir: Path,
        stem: str,
        seed: int = 42,
    ) -> ReconstructionResult:
        """
        Reconstruct a 3D mesh from a masked image.

        Args:
            image: RGB numpy array (H, W, 3), dtype uint8.
            mask: Boolean numpy array (H, W).
            output_dir: Directory to write output files.
            stem: Base filename stem for output files.
            seed: Random seed for reproducibility.

        Returns:
            ReconstructionResult with paths to produced files.
        """
        ...

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights. Separate from __init__ to allow lazy loading."""
        ...
