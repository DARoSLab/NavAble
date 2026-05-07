"""DuckDuckGo image crawler. No API key required."""

import json
import time
from pathlib import Path

import requests

from isaacnav.crawling.base import BaseCrawler, CrawlResult


class DuckDuckGoCrawler(BaseCrawler):
    """Crawl images via DuckDuckGo image search.

    Uses the ddgs package. Can be rate-limited — uses backoff.
    pip install ddgs
    """

    def __init__(self, config: dict):
        self.config = config
        self.delay = config.get("delay", 1.0)
        self.min_size = config.get("min_size", 300)
        self.max_retries = config.get("max_retries", 3)

    def search_images(self, query: str, limit: int = 50) -> list[dict]:
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                raise ImportError("ddgs package required. Install with: pip install ddgs")

        for attempt in range(self.max_retries):
            try:
                time.sleep(self.delay * (attempt + 1))  # increasing delay
                with DDGS() as ddgs:
                    results = ddgs.images(query, max_results=limit)
                return results
            except Exception as e:
                err = str(e).lower()
                if "ratelimit" in err or "403" in err:
                    wait = 10 * (attempt + 1)
                    print(f"  DuckDuckGo rate-limited — waiting {wait}s (attempt {attempt+1})")
                    time.sleep(wait)
                else:
                    if attempt == self.max_retries - 1:
                        print(f"  DuckDuckGo search failed: {e}")
                        return []
                    time.sleep(5)
        return []

    def download_images(
        self, query: str, output_dir: Path, num_images: int = 50
    ) -> list[CrawlResult]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        existing_count = len(list(output_dir.glob("*_ddg_*")))

        results = self.search_images(query, limit=num_images * 2)
        print(f"DuckDuckGo search returned {len(results)} results")

        if not results:
            return []

        downloaded = []
        for result in results:
            if len(downloaded) >= num_images:
                break

            image_url = result.get("image")
            if not image_url:
                continue

            url_path = image_url.split("?")[0]
            ext = url_path.split(".")[-1].lower()
            if ext not in ("jpg", "jpeg", "png", "webp"):
                ext = "jpg"

            query_clean = query.replace(" ", "_").replace("/", "_")[:30]
            idx = existing_count + len(downloaded)
            filename = output_dir / f"{query_clean}_ddg_{idx:04d}.{ext}"

            try:
                time.sleep(self.delay)
                img_response = requests.get(
                    image_url, timeout=15,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                if img_response.status_code != 200:
                    continue

                content_type = img_response.headers.get("content-type", "")
                if "image" not in content_type:
                    continue

                content = img_response.content
                if len(content) < 10_000:
                    continue

                with open(filename, "wb") as f:
                    f.write(content)

                title = result.get("title", "")
                meta = {
                    "source": "duckduckgo",
                    "title": title,
                    "image_url": image_url,
                    "source_url": result.get("url", ""),
                    "license": result.get("license", "Unknown"),
                }
                meta_file = filename.with_suffix(".json")
                with open(meta_file, "w") as f:
                    json.dump(meta, f, indent=2)

                downloaded.append(
                    CrawlResult(
                        image_path=filename,
                        metadata_path=meta_file,
                        source="duckduckgo",
                        license=meta["license"],
                        attribution=f"{title} via DuckDuckGo",
                    )
                )
                print(f"  [ddg {len(downloaded)}/{num_images}] {filename.name}")

            except Exception:
                continue

        print(f"DuckDuckGo: downloaded {len(downloaded)} images to {output_dir}")
        return downloaded
