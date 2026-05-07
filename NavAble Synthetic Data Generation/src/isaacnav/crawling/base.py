"""Abstract base class for image crawlers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CrawlResult:
    """Result from downloading a single image."""

    image_path: Path
    metadata_path: Path
    source: str  # "pexels", "flickr", "wikimedia"
    license: str
    attribution: str


class BaseCrawler(ABC):
    """Abstract base class for image crawlers."""

    @abstractmethod
    def __init__(self, config: dict):
        """Initialize with strategy-specific config from pipeline.yaml."""
        ...

    @abstractmethod
    def search_images(self, query: str, limit: int) -> list[dict]:
        """Search for images matching query. Returns list of result metadata dicts."""
        ...

    @abstractmethod
    def download_images(
        self,
        query: str,
        output_dir: Path,
        num_images: int,
    ) -> list[CrawlResult]:
        """Download images for the query. Returns list of CrawlResults."""
        ...
