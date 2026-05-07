"""Abstract base class for mesh format converters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ConversionResult:
    """Result from a format conversion."""

    output_path: Path
    format: str  # "usd", "usdz", "usda"
    has_texture: bool
    texture_path: Optional[Path] = None


class BaseConverter(ABC):
    """Abstract base class for mesh format converters."""

    @abstractmethod
    def __init__(self, config: dict):
        """Initialize with strategy-specific config."""
        ...

    @abstractmethod
    def convert(
        self,
        input_path: Path,
        output_path: Path,
        scale: float = 1.0,
    ) -> ConversionResult:
        """
        Convert a mesh file to the target format.

        Args:
            input_path: Path to input mesh file (PLY, GLB, etc.).
            output_path: Path for output file.
            scale: Scale factor for the mesh.

        Returns:
            ConversionResult with output path and metadata.
        """
        ...

    @abstractmethod
    def supported_input_formats(self) -> list[str]:
        """Return list of supported input file extensions (e.g., ['.ply', '.glb'])."""
        ...
