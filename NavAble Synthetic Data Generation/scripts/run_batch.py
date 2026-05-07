#!/usr/bin/env python3
"""Batch processing entry point with manifest-based job tracking.

Usage:
    # Create manifest from a directory of images
    python scripts/run_batch.py create-manifest \
        --image-dir data/input/elevator \
        --classes "Elevator" \
        --manifest data/jobs/elevator_batch.json

    # Run batch processing
    python scripts/run_batch.py run \
        --config configs/pipeline.yaml \
        --manifest data/jobs/elevator_batch.json

    # Check status
    python scripts/run_batch.py status --manifest data/jobs/elevator_batch.json

    # Retry failed jobs
    python scripts/run_batch.py retry \
        --config configs/pipeline.yaml \
        --manifest data/jobs/elevator_batch.json
"""

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from isaacnav.data.manifest import BatchManifest
from isaacnav.pipeline import AssetPipeline


def cmd_create_manifest(args):
    manifest = BatchManifest.from_image_directory(
        args.image_dir, args.classes, args.manifest
    )
    summary = manifest.get_summary()
    print(f"Manifest created: {args.manifest}")
    print(f"  Jobs: {summary}")


def cmd_run(args):
    pipeline = AssetPipeline(args.config)
    manifest = BatchManifest(args.manifest)

    batch_cfg = pipeline.config.get("batch", {})
    max_retries = batch_cfg.get("max_retries", 3)

    pending = manifest.get_pending_jobs(limit=999999)
    total = len(pending)
    print(f"Processing {total} pending jobs...")

    for i, job in enumerate(pending):
        print(f"\n[{i+1}/{total}] Job {job.job_id}: {Path(job.image_path).name}")

        try:
            manifest.update_status(job.job_id, "masking")
            result = pipeline.run_single(
                job.image_path,
                class_names=job.class_names,
            )
            manifest.update_status(
                job.job_id, "done",
                output_dir=result.get("object_dir"),
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            if job.retry_count < max_retries:
                manifest.update_status(job.job_id, "failed", error=str(e))
            else:
                manifest.update_status(
                    job.job_id, "failed",
                    error=f"Max retries exceeded: {e}",
                )

    summary = manifest.get_summary()
    print(f"\nBatch complete: {summary}")


def cmd_status(args):
    manifest = BatchManifest(args.manifest)
    summary = manifest.get_summary()
    print(f"Manifest: {args.manifest}")
    for status, count in sorted(summary.items()):
        print(f"  {status}: {count}")


def cmd_retry(args):
    pipeline = AssetPipeline(args.config)
    manifest = BatchManifest(args.manifest)

    max_retries = pipeline.config.get("batch", {}).get("max_retries", 3)
    count = manifest.retry_failed(max_retries)
    print(f"Reset {count} failed jobs to pending")

    if count > 0:
        print("Running retries...")
        cmd_run(args)


def main():
    parser = argparse.ArgumentParser(description="Batch processing for NavAble pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # create-manifest
    create_p = subparsers.add_parser("create-manifest", help="Create job manifest from images")
    create_p.add_argument("--image-dir", required=True, help="Directory of input images")
    create_p.add_argument("--classes", nargs="+", required=True, help="Target class names")
    create_p.add_argument("--manifest", required=True, help="Output manifest path")

    # run
    run_p = subparsers.add_parser("run", help="Run batch processing")
    run_p.add_argument("--config", "-c", default="configs/pipeline.yaml")
    run_p.add_argument("--manifest", required=True, help="Manifest file path")

    # status
    status_p = subparsers.add_parser("status", help="Show batch status")
    status_p.add_argument("--manifest", required=True, help="Manifest file path")

    # retry
    retry_p = subparsers.add_parser("retry", help="Retry failed jobs")
    retry_p.add_argument("--config", "-c", default="configs/pipeline.yaml")
    retry_p.add_argument("--manifest", required=True, help="Manifest file path")

    args = parser.parse_args()

    if args.command == "create-manifest":
        cmd_create_manifest(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "retry":
        cmd_retry(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
