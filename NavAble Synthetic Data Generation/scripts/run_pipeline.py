#!/usr/bin/env python3
"""Config-driven pipeline entry point.

Usage:
    # Crawl + validate (download images, validate with Gemini, delete bad ones)
    python scripts/run_pipeline.py --crawl-only
    python scripts/run_pipeline.py --crawl-only --classes "Elevator" "Door"

    # Crawl + validate + process (full pipeline)
    python scripts/run_pipeline.py --crawl
    python scripts/run_pipeline.py --crawl --classes "Elevator" "Door"

    # Process already-crawled images (skip crawling/validation)
    python scripts/run_pipeline.py --process-all
    python scripts/run_pipeline.py --process-all --classes "Elevator" "Door"

    # Single image (class inferred from parent directory name)
    python scripts/run_pipeline.py --image data/input/elevator/img1.jpg

    # Single image with explicit class
    python scripts/run_pipeline.py --image photo.jpg --class "Elevator"

    # Process a class directory
    python scripts/run_pipeline.py --image-dir data/input/pedestrian_signal/

    # Overrides
    python scripts/run_pipeline.py --crawl --crawl-strategy bing
    python scripts/run_pipeline.py --image img.jpg --masking-strategy sam_heuristic
    python scripts/run_pipeline.py --crawl-only --no-validation
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from isaacnav.pipeline import AssetPipeline


def main():
    parser = argparse.ArgumentParser(
        description="NavAble Image-to-3D Asset Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/pipeline.yaml",
        help="Pipeline config file (default: configs/pipeline.yaml)",
    )

    # Input modes (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--image", "-i",
        help="Single input image path",
    )
    input_group.add_argument(
        "--image-dir",
        help="Directory of images for one class (class inferred from dir name)",
    )
    input_group.add_argument(
        "--process-all", action="store_true",
        help="Process all class directories under data/input/",
    )
    input_group.add_argument(
        "--crawl", action="store_true",
        help="Crawl + validate + process (full pipeline)",
    )
    input_group.add_argument(
        "--crawl-only", action="store_true",
        help="Crawl + validate only (no segmentation/reconstruction)",
    )

    parser.add_argument(
        "--class", dest="class_name",
        help="Target class name for --image mode",
    )
    parser.add_argument(
        "--classes", nargs="+",
        help="Subset of classes for --crawl / --process-all (default: all from config)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--masking-strategy",
        choices=["sam_heuristic", "grounded_sam2"],
        help="Override masking strategy",
    )
    parser.add_argument(
        "--reconstruction-strategy",
        choices=["sam3d"],
        help="Override reconstruction strategy",
    )
    parser.add_argument(
        "--crawl-strategy",
        choices=["multi", "bing", "wikimedia", "duckduckgo"],
        help="Override crawl strategy",
    )
    parser.add_argument(
        "--no-validation", action="store_true",
        help="Disable Gemini validation (keep all crawled images)",
    )

    args = parser.parse_args()

    # Load pipeline
    pipeline = AssetPipeline(args.config)

    # Apply overrides
    if args.masking_strategy:
        pipeline.config["masking"]["strategy"] = args.masking_strategy
    if args.reconstruction_strategy:
        pipeline.config["reconstruction"]["strategy"] = args.reconstruction_strategy
    if args.no_validation:
        pipeline.config["validation"]["enabled"] = False
    if args.output:
        pipeline.config["output"]["base_dir"] = args.output
    if args.crawl_strategy:
        pipeline.config["crawling"]["strategy"] = args.crawl_strategy

    # --- Run ---
    if args.crawl_only:
        results = pipeline.run_crawl_only(classes=args.classes)
        print(f"\nCrawled and validated {len(results)} images (crawl-only, no processing)")

    elif args.crawl:
        results = pipeline.run_crawl_and_process(classes=args.classes)
        print(f"\nProcessed {len(results)} images total")

    elif args.process_all:
        results = pipeline.run_all_classes(classes=args.classes)
        print(f"\nProcessed {len(results)} images total")

    elif args.image:
        result = pipeline.run_single(args.image, class_name=args.class_name)
        print(f"\nArtifacts: {list(result.keys())}")

    elif args.image_dir:
        image_dir = Path(args.image_dir)
        results = pipeline.run_class_dir(image_dir, class_name=args.class_name)
        print(f"\nProcessed {len(results)} images from {image_dir}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
