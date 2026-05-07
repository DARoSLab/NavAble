"""Multi-source crawler that runs multiple backends concurrently."""

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from isaacnav.crawling.base import BaseCrawler, CrawlResult


class MultiCrawler(BaseCrawler):
    """Run multiple crawlers concurrently for faster throughput and image diversity.

    Each backend gets an equal share of the total quota. Results are deduplicated
    by file size (to catch exact duplicates from different sources).

    Config example:
        multi:
          backends: ["bing", "wikimedia", "duckduckgo"]
    """

    def __init__(self, config: dict):
        self.config = config
        self.backend_names = config.get("backends", ["bing", "wikimedia", "duckduckgo"])
        self._full_crawl_config = config.get("_full_crawl_config", {})

    def _make_backend(self, name: str) -> BaseCrawler:
        from isaacnav.crawling.bing import BingCrawler
        from isaacnav.crawling.wikimedia import WikimediaCommonsCrawler
        from isaacnav.crawling.duckduckgo import DuckDuckGoCrawler

        registry = {
            "bing": BingCrawler,
            "wikimedia": WikimediaCommonsCrawler,
            "duckduckgo": DuckDuckGoCrawler,
        }
        if name not in registry:
            raise ValueError(f"Unknown crawler backend: {name}. Available: {list(registry.keys())}")

        backend_config = self._full_crawl_config.get(name, {})
        return registry[name](backend_config)

    def search_images(self, query: str, limit: int = 50) -> list[dict]:
        return []  # multi-crawler goes straight to download

    def download_images(
        self, query: str, output_dir: Path, num_images: int = 50
    ) -> list[CrawlResult]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        n_backends = len(self.backend_names)
        # Give each backend a proportional share, slightly over-request
        per_backend = max(10, (num_images // n_backends) + 5)

        print(f"Multi-crawler: {n_backends} backends × ~{per_backend} images each")
        print(f"  Backends: {', '.join(self.backend_names)}")

        def _run_backend(name: str) -> tuple[str, list[CrawlResult]]:
            try:
                backend = self._make_backend(name)
                results = backend.download_images(query, output_dir, per_backend)
                return name, results
            except Exception as e:
                print(f"  {name} crawler failed: {e}")
                return name, []

        all_results = []
        # Run backends concurrently — each has its own rate limiting internally
        with ThreadPoolExecutor(max_workers=n_backends) as executor:
            futures = {
                executor.submit(_run_backend, name): name
                for name in self.backend_names
            }
            for future in as_completed(futures):
                name, results = future.result()
                print(f"  {name}: got {len(results)} images")
                all_results.extend(results)

        # Deduplicate by file content hash (catches exact duplicates from different sources)
        seen_hashes = set()
        unique_results = []
        for cr in all_results:
            if not cr.image_path.exists():
                continue
            h = hashlib.md5(cr.image_path.read_bytes()).hexdigest()
            if h in seen_hashes:
                # Remove duplicate file + metadata
                cr.image_path.unlink(missing_ok=True)
                cr.metadata_path.unlink(missing_ok=True)
                continue
            seen_hashes.add(h)
            unique_results.append(cr)

        # Trim to requested count
        if len(unique_results) > num_images:
            for cr in unique_results[num_images:]:
                cr.image_path.unlink(missing_ok=True)
                cr.metadata_path.unlink(missing_ok=True)
            unique_results = unique_results[:num_images]

        dupes = len(all_results) - len(unique_results)
        print(f"\nMulti-crawler total: {len(unique_results)} unique images "
              f"({dupes} duplicates removed)")
        return unique_results
