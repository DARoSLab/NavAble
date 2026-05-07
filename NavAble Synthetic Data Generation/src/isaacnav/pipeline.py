"""Pipeline orchestrator -- ties all stages together via config.

Workflow: crawl → validate (delete bad) → segment → reconstruct → convert

Each image belongs to exactly ONE class (inferred from its parent directory
or passed explicitly). Segmentation only targets that single class.
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from PIL import Image

from isaacnav.conversion.base import BaseConverter  # noqa: F401
from isaacnav.masking.base import BaseMaskingStrategy  # noqa: F401
from isaacnav.reconstruction.base import BaseReconstructionStrategy  # noqa: F401
from isaacnav.data.output_layout import OutputLayout


# Strategy registries -- maps config names to classes
def _get_masking_registry():
    from isaacnav.masking.sam_heuristic import SamHeuristicMasking
    from isaacnav.masking.grounded_sam2 import GroundedSam2Masking
    return {
        "sam_heuristic": SamHeuristicMasking,
        "grounded_sam2": GroundedSam2Masking,
    }


def _get_reconstruction_registry():
    from isaacnav.reconstruction.sam3d import Sam3dReconstruction
    return {"sam3d": Sam3dReconstruction}


def _get_crawler_registry():
    from isaacnav.crawling.wikimedia import WikimediaCommonsCrawler
    from isaacnav.crawling.duckduckgo import DuckDuckGoCrawler
    from isaacnav.crawling.bing import BingCrawler
    from isaacnav.crawling.multi import MultiCrawler
    return {
        "wikimedia": WikimediaCommonsCrawler,
        "duckduckgo": DuckDuckGoCrawler,
        "bing": BingCrawler,
        "multi": MultiCrawler,
    }


def _get_validator_registry():
    from isaacnav.validation.gemini import GeminiValidator
    return {"gemini": GeminiValidator}


def _get_converter_registry():
    from isaacnav.conversion.glb_to_usd import GlbToUsdConverter
    from isaacnav.conversion.ply_to_usd import PlyToUsdConverter
    return {
        "glb_to_usd": GlbToUsdConverter,
        "ply_to_usd": PlyToUsdConverter,
    }


def _safe_label(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


class AssetPipeline:
    """Config-driven pipeline orchestrator.

    Each image is processed for exactly one class. The class is determined by:
    1. Explicit --classes argument
    2. Parent directory name (data/input/{class_name}/image.jpg)
    3. First class in config target_classes (fallback)
    """

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.config_dir = Path(config_path).parent

        # Lazily instantiated strategies
        self._masking: Optional[BaseMaskingStrategy] = None
        self._reconstruction: Optional[BaseReconstructionStrategy] = None
        self._validator = None
        self._converters: Optional[list[BaseConverter]] = None

        # Output layout
        output_cfg = self.config.get("output", {})
        self._layout = OutputLayout(output_cfg.get("base_dir", "output"))

    def _load_config(self, config_path: str) -> dict:
        with open(config_path) as f:
            return yaml.safe_load(f)

    @property
    def target_classes(self) -> list[str]:
        """Get the target class list from config."""
        tc = self.config.get("target_classes", [])
        if isinstance(tc, list):
            return tc
        if isinstance(tc, dict):
            if "classes" in tc:
                return tc["classes"]
            source = tc.get("source")
            if source:
                source_path = Path(source)
                if not source_path.is_absolute() and not source_path.exists():
                    source_path = self.config_dir / source_path
                with open(source_path) as f:
                    data = yaml.safe_load(f)
                categories = data.get("categories", [])
                return categories if isinstance(categories, list) else []
        return []

    def _infer_class(self, image_path: Path) -> str:
        """Infer the target class from the image's parent directory name."""
        parent = image_path.parent.name
        parent_norm = parent.lower().replace("_", " ").replace("-", " ")
        for cls in self.target_classes:
            if _safe_label(cls).replace("_", " ") == parent_norm:
                return cls
        return parent.replace("_", " ").replace("-", " ").title() if parent else "unknown"

    @property
    def masking_strategy(self) -> BaseMaskingStrategy:
        if self._masking is None:
            name = self.config["masking"]["strategy"]
            registry = _get_masking_registry()
            strategy_config = self.config["masking"].get(name, {})
            self._masking = registry[name](strategy_config)
        return self._masking

    @property
    def reconstruction_strategy(self) -> BaseReconstructionStrategy:
        if self._reconstruction is None:
            name = self.config["reconstruction"]["strategy"]
            registry = _get_reconstruction_registry()
            strategy_config = self.config["reconstruction"].get(name, {})
            strategy_config["seed"] = self.config["reconstruction"].get("seed", 42)
            self._reconstruction = registry[name](strategy_config)
        return self._reconstruction

    @property
    def validator(self):
        val_cfg = self.config.get("validation", {})
        if not val_cfg.get("enabled", False):
            return None
        if self._validator is None:
            name = val_cfg.get("strategy", "gemini")
            if name == "none":
                return None
            registry = _get_validator_registry()
            strategy_config = val_cfg.get(name, {})
            self._validator = registry[name](strategy_config)
        return self._validator

    # ----------------------------------------------------------------
    # Validate crawled images — delete failures from data/input
    # ----------------------------------------------------------------
    def _resolve_validation_prompt(self, class_name: str) -> str | None:
        """Resolve the validation prompt for a class, returning None for default behavior."""
        prompt_cfg = self.config.get("validation", {}).get("prompt", {})
        if not prompt_cfg:
            return None
        # Per-class override takes priority
        class_prompts = prompt_cfg.get("class_prompts", {})
        if class_name in class_prompts:
            return class_prompts[class_name]
        return prompt_cfg.get("default")

    def _validate_images(self, crawl_results: list, class_name: str) -> list:
        """Validate crawled images, deleting those that fail.

        Returns only the CrawlResults that passed validation.
        """
        if not self.validator:
            return crawl_results

        prompt = self._resolve_validation_prompt(class_name)

        print(f"\n[Validate] Checking {len(crawl_results)} images for '{class_name}'...")
        passed = []
        failed = 0

        for cr in crawl_results:
            try:
                val_result = self.validator.validate(cr.image_path, class_name, prompt=prompt)
                if val_result.contains_class:
                    passed.append(cr)
                    print(f"  ✓ {cr.image_path.name} (conf={val_result.confidence:.2f})")
                else:
                    # Delete the image and its metadata from data/input
                    cr.image_path.unlink(missing_ok=True)
                    cr.metadata_path.unlink(missing_ok=True)
                    failed += 1
                    actual = val_result.actual_content or "unknown"
                    print(f"  ✗ {cr.image_path.name} — not '{class_name}' (actual: {actual})")
            except Exception as e:
                # Keep image on validation error (don't delete what we can't verify)
                passed.append(cr)
                print(f"  ? {cr.image_path.name} — validation error: {e}")

        print(f"Validation: {len(passed)} passed, {failed} deleted")
        return passed

    # ----------------------------------------------------------------
    # Conversion subprocess (isolates pxr from SAM3D CUDA state)
    # ----------------------------------------------------------------
    def _run_conversion_subprocess(self, input_file: Path, output_path: Path) -> Optional[dict]:
        conv_config_json = json.dumps(self.config.get("conversion", {}))
        script = f"""
import sys, json
sys.path.insert(0, "src")
from pathlib import Path
from isaacnav.conversion.glb_to_usd import GlbToUsdConverter
from isaacnav.conversion.ply_to_usd import PlyToUsdConverter

input_file = Path("{input_file}")
output_path = Path("{output_path}")
config = json.loads('{conv_config_json}')

if input_file.suffix in (".glb", ".gltf"):
    converter = GlbToUsdConverter(config.get("glb_to_usd", {{}}))
elif input_file.suffix == ".ply":
    converter = PlyToUsdConverter(config.get("ply_to_usd", {{}}))
else:
    print(json.dumps({{"error": "unsupported format"}}))
    sys.exit(1)

result = converter.convert(input_file, output_path)
print(json.dumps({{
    "input": str(input_file),
    "output": str(result.output_path),
    "format": result.format,
    "has_texture": result.has_texture,
}}))
"""
        try:
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                for line in reversed(result.stdout.strip().splitlines()):
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        continue
                print(f"  Conversion output: {result.stdout.strip()}")
            else:
                print(f"  Conversion subprocess failed (exit {result.returncode})")
                if result.stderr:
                    for line in result.stderr.strip().splitlines()[-5:]:
                        print(f"    {line}")
        except subprocess.TimeoutExpired:
            print("  Conversion timed out (120s)")
        except Exception as e:
            print(f"  Conversion error: {e}")
        return None

    # ----------------------------------------------------------------
    # Core: process a single image for a single class
    # ----------------------------------------------------------------
    def run_single(
        self,
        image_path: str,
        class_name: str = None,
    ) -> dict:
        """Run the pipeline on a single image: segment → reconstruct → convert.

        Validation is NOT done here — it happens during crawl (see _validate_images).
        """
        image_path = Path(image_path)
        if class_name is None:
            class_name = self._infer_class(image_path)

        source_id = image_path.stem
        obj_dir = self._layout.create_object_dir(class_name, source_id)

        artifacts = {"object_dir": str(obj_dir), "class_name": class_name}
        start_time = time.time()

        image = np.array(Image.open(image_path).convert("RGB"))
        print(f"\n{'='*60}")
        print(f"Processing: {image_path.name}")
        print(f"Class: {class_name}")
        print(f"Output: {obj_dir}")
        print(f"{'='*60}")

        output_cfg = self.config.get("output", {})

        # Save original
        if output_cfg.get("save_original", True):
            dest = obj_dir / f"image{image_path.suffix}"
            shutil.copy2(str(image_path), str(dest))
            artifacts["original_image"] = str(dest)

        # --- Stage 1: Masking (single class only) ---
        print(f"\n[Segment] {self.config['masking']['strategy']} for '{class_name}'...")
        mask_results = self.masking_strategy.segment(image, [class_name])

        if not mask_results:
            print(f"  No '{class_name}' detected!")
            self._layout.write_metadata(
                obj_dir, class_name,
                masking_meta={"error": f"No {class_name} detected"},
            )
            return artifacts

        # Pick the single best mask (highest confidence)
        best_mask = max(mask_results, key=lambda m: m.confidence)
        print(f"  Best mask: confidence={best_mask.confidence:.3f} (from {len(mask_results)} detections)")

        masking_meta = {
            "class_name": best_mask.class_name,
            "confidence": best_mask.confidence,
            "bbox": best_mask.bbox,
            "strategy": best_mask.source_strategy,
            "total_detections": len(mask_results),
        }
        if output_cfg.get("save_masks", True):
            mask_path = obj_dir / "mask.png"
            Image.fromarray((best_mask.mask * 255).astype(np.uint8)).save(str(mask_path))
            artifacts["mask"] = str(mask_path)

        # --- Stage 2: Reconstruction ---
        print(f"\n[Reconstruct] {self.config['reconstruction']['strategy']}...")
        recon_result = self.reconstruction_strategy.reconstruct(
            image, best_mask.mask, obj_dir, source_id,
            seed=self.config["reconstruction"].get("seed", 42),
        )

        recon_meta = {"strategy": recon_result.source_strategy}
        if recon_result.ply_path:
            artifacts["ply"] = str(recon_result.ply_path)
            recon_meta["ply_path"] = str(recon_result.ply_path)
        if recon_result.glb_path:
            artifacts["glb"] = str(recon_result.glb_path)
            recon_meta["glb_path"] = str(recon_result.glb_path)

        # --- Stage 3: Conversion (subprocess) ---
        print(f"\n[Convert] GLB → USDZ...")
        conversion_meta = {}
        input_file = None
        if recon_result.glb_path and recon_result.glb_path.exists():
            input_file = recon_result.glb_path
        elif recon_result.mesh_ply_path and recon_result.mesh_ply_path.exists():
            input_file = recon_result.mesh_ply_path

        if input_file:
            out_path = obj_dir / "asset.usdz"
            conv_result = self._run_conversion_subprocess(input_file, out_path)
            if conv_result:
                artifacts["usdz"] = conv_result["output"]
                conversion_meta = conv_result

        elapsed = time.time() - start_time

        self._layout.write_metadata(
            obj_dir,
            class_name,
            masking_meta=masking_meta,
            reconstruction_meta=recon_meta,
            conversion_meta=conversion_meta,
        )

        print(f"\nDone! {class_name} processed in {elapsed:.1f}s")
        print(f"Output: {obj_dir}")
        artifacts["elapsed_seconds"] = elapsed
        return artifacts

    # ----------------------------------------------------------------
    # Process all images in a class directory
    # ----------------------------------------------------------------
    def run_class_dir(self, class_dir: Path, class_name: str = None) -> list[dict]:
        """Process all images in a class directory."""
        class_dir = Path(class_dir)
        if class_name is None:
            class_name = self._infer_class(class_dir / "dummy.jpg")

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in extensions)
        print(f"Found {len(images)} images in {class_dir} for class '{class_name}'")

        results = []
        for img_path in images:
            try:
                result = self.run_single(str(img_path), class_name=class_name)
                results.append(result)
            except Exception as e:
                print(f"Failed to process {img_path.name}: {e}")
        return results

    # ----------------------------------------------------------------
    # Crawl + validate images for target classes
    # ----------------------------------------------------------------
    def _crawl_classes(self, classes: list[str] = None) -> dict[str, list]:
        """Crawl images, then validate (deleting failures). Returns {class: [CrawlResult]}."""
        crawl_cfg = self.config.get("crawling", {})
        if not crawl_cfg.get("enabled", False):
            print("Crawling is disabled in config")
            return {}

        if classes is None:
            classes = self.target_classes
        if not classes:
            print("No target classes configured")
            return {}

        strategy_name = crawl_cfg.get("strategy", "bing")
        num_per_class = crawl_cfg.get("num_per_class", 50)
        data_dir = crawl_cfg.get("data_dir", "data/input")

        registry = _get_crawler_registry()
        crawler_config = crawl_cfg.get(strategy_name, {})
        if strategy_name == "multi":
            crawler_config["_full_crawl_config"] = crawl_cfg
        crawler = registry[strategy_name](crawler_config)

        results_by_class = {}
        for cls_name in classes:
            print(f"\n{'='*60}")
            print(f"Crawling images for: {cls_name}")
            print(f"{'='*60}")

            cls_dir = Path(data_dir) / _safe_label(cls_name)
            crawl_results = crawler.download_images(cls_name, cls_dir, num_per_class)
            print(f"Downloaded {len(crawl_results)} images to {cls_dir}")

            # Validate and delete bad images
            validated = self._validate_images(crawl_results, cls_name)
            results_by_class[cls_name] = validated

        return results_by_class

    # ----------------------------------------------------------------
    # Crawl only (no processing) — crawls + validates
    # ----------------------------------------------------------------
    def run_crawl_only(self, classes: list[str] = None) -> list:
        """Crawl and validate images without running segmentation/reconstruction."""
        results_by_class = self._crawl_classes(classes)
        all_results = []
        for crawl_results in results_by_class.values():
            all_results.extend(crawl_results)
        return all_results

    # ----------------------------------------------------------------
    # Full workflow: crawl → validate → process
    # ----------------------------------------------------------------
    def run_crawl_and_process(self, classes: list[str] = None) -> list[dict]:
        """Crawl images, validate, then process survivors."""
        results_by_class = self._crawl_classes(classes)

        all_results = []
        for cls_name, crawl_results in results_by_class.items():
            for cr in crawl_results:
                try:
                    result = self.run_single(str(cr.image_path), class_name=cls_name)
                    all_results.append(result)
                except Exception as e:
                    print(f"Failed to process {cr.image_path.name}: {e}")

        return all_results

    # ----------------------------------------------------------------
    # Process existing data/input/ directory tree
    # ----------------------------------------------------------------
    def run_all_classes(self, classes: list[str] = None) -> list[dict]:
        """Process all images already in data/input/{class_name}/ directories."""
        data_dir = Path(self.config.get("crawling", {}).get("data_dir", "data/input"))
        if classes is None:
            classes = self.target_classes

        all_results = []
        for cls_name in classes:
            cls_dir = data_dir / _safe_label(cls_name)
            if cls_dir.is_dir():
                results = self.run_class_dir(cls_dir, class_name=cls_name)
                all_results.extend(results)
            else:
                print(f"No images found for '{cls_name}' at {cls_dir}")

        return all_results
