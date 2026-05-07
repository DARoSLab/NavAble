"""Bing image crawler via icrawler. No API key required."""

import json
import logging
import shutil
import tempfile
from pathlib import Path

from isaacnav.crawling.base import BaseCrawler, CrawlResult

# Suppress icrawler's verbose logging (including per-URL download errors)
for _name in ("icrawler", "icrawler.crawler", "parser", "downloader", "feeder"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)


class BingCrawler(BaseCrawler):
    """Crawl images via Bing image search using icrawler.

    No API key needed. Fast and reliable.
    pip install icrawler
    """

    def __init__(self, config: dict):
        self.config = config
        self.min_size = config.get("min_size", 300)
        self.threads = config.get("threads", 4)

    def search_images(self, query: str, limit: int = 50) -> list[dict]:
        return []  # icrawler doesn't expose search-only API

    def download_images(
        self, query: str, output_dir: Path, num_images: int = 50
    ) -> list[CrawlResult]:
        try:
            from icrawler.builtin import BingImageCrawler
        except ImportError:
            raise ImportError(
                "icrawler package required. Install with: pip install icrawler"
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Count existing images to avoid overwriting
        existing = set(output_dir.glob("*.jpg")) | set(output_dir.glob("*.png")) | set(output_dir.glob("*.jpeg"))
        existing_count = len(existing)

        tmpdir = tempfile.mkdtemp(prefix="bing_crawl_")

        try:
            crawler = BingImageCrawler(
                storage={"root_dir": tmpdir},
                downloader_threads=self.threads,
            )
            # Fetch extra to compensate for download failures and small-file filtering
            crawler.crawl(
                keyword=query,
                max_num=int(num_images * 1.5),
                min_size=(self.min_size, self.min_size),
            )

            tmp_files = sorted(Path(tmpdir).glob("*"))
            query_clean = query.replace(" ", "_").replace("/", "_")[:30]

            downloaded = []
            for tmp_file in tmp_files:
                if len(downloaded) >= num_images:
                    break

                if tmp_file.stat().st_size < 10_000:
                    continue

                ext = tmp_file.suffix.lower()
                if ext not in (".jpg", ".jpeg", ".png", ".webp"):
                    ext = ".jpg"

                idx = existing_count + len(downloaded)
                filename = output_dir / f"{query_clean}_bing_{idx:04d}{ext}"

                shutil.move(str(tmp_file), str(filename))

                meta = {
                    "source": "bing",
                    "query": query,
                    "license": "Unknown",
                }
                meta_file = filename.with_suffix(".json")
                with open(meta_file, "w") as f:
                    json.dump(meta, f, indent=2)

                downloaded.append(
                    CrawlResult(
                        image_path=filename,
                        metadata_path=meta_file,
                        source="bing",
                        license="Unknown",
                        attribution=f"{query} via Bing Image Search",
                    )
                )
                print(f"  [bing {len(downloaded)}/{num_images}] {filename.name}")

            print(f"Bing: downloaded {len(downloaded)} images to {output_dir}")
            return downloaded

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
