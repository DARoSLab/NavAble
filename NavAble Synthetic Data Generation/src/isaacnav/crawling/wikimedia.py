"""Wikimedia Commons image crawler. No API key required, CC licensed."""

import json
import time
from pathlib import Path

import requests

from isaacnav.crawling.base import BaseCrawler, CrawlResult


class WikimediaCommonsCrawler(BaseCrawler):
    """Crawl Creative Commons images from Wikimedia Commons.

    No API key needed. Rate-limited to ~1 req/sec (polite crawling).
    Images are CC-licensed with full attribution metadata.
    """

    BASE_URL = "https://commons.wikimedia.org/w/api.php"
    HEADERS = {
        "User-Agent": "NavAbleBot/2.1 (Academic Research; image-to-3d pipeline)"
    }

    def __init__(self, config: dict):
        self.config = config
        self.delay = config.get("delay", 1.5)  # conservative default
        self.max_retries = config.get("max_retries", 3)

    def search_images(self, query: str, limit: int = 50) -> list[dict]:
        all_pages = []
        continue_token = None

        while len(all_pages) < limit:
            batch = min(limit - len(all_pages), 50)
            params = {
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrnamespace": 6,
                "gsrsearch": f"{query} filetype:bitmap",
                "gsrlimit": batch,
                "prop": "imageinfo",
                "iiprop": "url|extmetadata|size|mime",
                "iiurlwidth": 1024,
            }
            if continue_token:
                params.update(continue_token)

            for attempt in range(self.max_retries):
                time.sleep(self.delay)
                try:
                    response = requests.get(
                        self.BASE_URL, params=params,
                        headers=self.HEADERS, timeout=30,
                    )
                    if response.status_code == 429:
                        wait = int(response.headers.get("Retry-After", 10 * (attempt + 1)))
                        print(f"  Wikimedia 429 — waiting {wait}s (attempt {attempt+1}/{self.max_retries})")
                        time.sleep(wait)
                        continue
                    response.raise_for_status()
                    break
                except requests.RequestException as e:
                    if attempt == self.max_retries - 1:
                        print(f"  Wikimedia search failed after {self.max_retries} attempts: {e}")
                        return all_pages
                    time.sleep(5 * (attempt + 1))
            else:
                return all_pages

            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            all_pages.extend(pages.values())

            cont = data.get("continue")
            if cont:
                continue_token = cont
            else:
                break

        return all_pages[:limit]

    def download_images(
        self, query: str, output_dir: Path, num_images: int = 50
    ) -> list[CrawlResult]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Count existing to avoid name collisions
        existing_count = len(list(output_dir.glob("*_wiki_*")))

        downloaded = []
        results = self.search_images(query, limit=num_images * 2)

        print(f"Wikimedia search returned {len(results)} results")

        if not results:
            return downloaded

        for page in results:
            if len(downloaded) >= num_images:
                break

            imageinfo = page.get("imageinfo", [{}])[0]
            image_url = imageinfo.get("thumburl") or imageinfo.get("url")

            if not image_url:
                continue

            mime = imageinfo.get("mime", "")
            if mime in ("image/svg+xml", "image/gif"):
                continue
            width = imageinfo.get("width", 0)
            height = imageinfo.get("height", 0)
            if width < 300 or height < 300:
                continue

            page_id = page.get("pageid", "unknown")
            title = page.get("title", "").replace("File:", "").replace(" ", "_")

            ext = image_url.split(".")[-1].split("?")[0].lower()
            if ext not in ("jpg", "jpeg", "png"):
                ext = "jpg"

            query_clean = query.replace(" ", "_").replace("/", "_")[:30]
            filename = output_dir / f"{query_clean}_wiki_{page_id}.{ext}"

            if filename.exists():
                continue  # skip duplicates

            for attempt in range(self.max_retries):
                try:
                    time.sleep(self.delay)
                    img_response = requests.get(
                        image_url, timeout=30, headers=self.HEADERS,
                    )
                    if img_response.status_code == 429:
                        wait = int(img_response.headers.get("Retry-After", 10 * (attempt + 1)))
                        print(f"  Wikimedia 429 on download — waiting {wait}s")
                        time.sleep(wait)
                        continue
                    if img_response.status_code != 200:
                        break  # non-retryable error

                    with open(filename, "wb") as f:
                        f.write(img_response.content)

                    extmeta = imageinfo.get("extmetadata", {})
                    license_short = extmeta.get("LicenseShortName", {}).get("value", "Unknown")
                    artist = extmeta.get("Artist", {}).get("value", "Unknown")

                    meta = {
                        "source": "wikimedia_commons",
                        "page_id": page_id,
                        "title": title,
                        "wikimedia_url": f"https://commons.wikimedia.org/wiki/File:{title}",
                        "license": license_short,
                        "artist": artist,
                        "attribution": f"{title} by {artist}, {license_short}, via Wikimedia Commons",
                    }
                    meta_file = filename.with_suffix(".json")
                    with open(meta_file, "w") as f:
                        json.dump(meta, f, indent=2)

                    downloaded.append(
                        CrawlResult(
                            image_path=filename,
                            metadata_path=meta_file,
                            source="wikimedia_commons",
                            license=license_short,
                            attribution=meta["attribution"],
                        )
                    )
                    print(f"  [wiki {len(downloaded)}/{num_images}] {filename.name}")
                    break

                except Exception as e:
                    if attempt == self.max_retries - 1:
                        print(f"  Failed {page_id}: {e}")

        print(f"Wikimedia: downloaded {len(downloaded)} images to {output_dir}")
        return downloaded
