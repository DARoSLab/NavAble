"""Batch processing manifest for tracking job status.

JSON-backed job tracker that supports resume on interruption.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class JobItem:
    """Single item in a batch processing manifest."""

    def __init__(
        self,
        image_path: str,
        class_names: list[str],
        job_id: str = None,
        status: str = "pending",
        error: Optional[str] = None,
        output_dir: Optional[str] = None,
        retry_count: int = 0,
    ):
        self.job_id = job_id or uuid.uuid4().hex[:12]
        self.image_path = image_path
        self.class_names = class_names
        self.status = status  # pending | scoring | masking | reconstruction | conversion | done | failed
        self.error = error
        self.output_dir = output_dir
        self.retry_count = retry_count
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = self.created_at

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "image_path": self.image_path,
            "class_names": self.class_names,
            "status": self.status,
            "error": self.error,
            "output_dir": self.output_dir,
            "retry_count": self.retry_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "JobItem":
        item = cls(
            image_path=d["image_path"],
            class_names=d["class_names"],
            job_id=d.get("job_id"),
            status=d.get("status", "pending"),
            error=d.get("error"),
            output_dir=d.get("output_dir"),
            retry_count=d.get("retry_count", 0),
        )
        item.created_at = d.get("created_at", item.created_at)
        item.updated_at = d.get("updated_at", item.updated_at)
        return item


class BatchManifest:
    """JSON-backed job manifest for batch processing."""

    def __init__(self, manifest_path: str):
        self.manifest_path = Path(manifest_path)
        self.jobs: list[JobItem] = []
        if self.manifest_path.exists():
            self._load()

    def _load(self):
        with open(self.manifest_path) as f:
            data = json.load(f)
        self.jobs = [JobItem.from_dict(j) for j in data.get("jobs", [])]

    def _save(self):
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "2.0",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "summary": self.get_summary(),
            "jobs": [j.to_dict() for j in self.jobs],
        }
        with open(self.manifest_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_job(self, image_path: str, class_names: list[str]) -> str:
        """Add a job, return job_id."""
        item = JobItem(image_path=image_path, class_names=class_names)
        self.jobs.append(item)
        self._save()
        return item.job_id

    def get_pending_jobs(self, limit: int = 10) -> list[JobItem]:
        """Return next N pending jobs."""
        pending = [j for j in self.jobs if j.status == "pending"]
        return pending[:limit]

    def get_failed_jobs(self) -> list[JobItem]:
        """Return all failed jobs."""
        return [j for j in self.jobs if j.status == "failed"]

    def update_status(self, job_id: str, status: str, **kwargs) -> None:
        """Update job status and optional fields."""
        for job in self.jobs:
            if job.job_id == job_id:
                job.status = status
                job.updated_at = datetime.now(timezone.utc).isoformat()
                for k, v in kwargs.items():
                    if hasattr(job, k):
                        setattr(job, k, v)
                break
        self._save()

    def get_summary(self) -> dict:
        """Return counts by status."""
        summary = {}
        for job in self.jobs:
            summary[job.status] = summary.get(job.status, 0) + 1
        summary["total"] = len(self.jobs)
        return summary

    @classmethod
    def from_image_directory(
        cls,
        image_dir: str,
        class_names: list[str],
        manifest_path: str,
    ) -> "BatchManifest":
        """Create a manifest from all images in a directory."""
        manifest = cls(manifest_path)
        image_dir = Path(image_dir)

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = sorted(
            p for p in image_dir.iterdir() if p.suffix.lower() in extensions
        )

        for img_path in images:
            manifest.add_job(str(img_path), class_names)

        print(f"Created manifest with {len(images)} jobs from {image_dir}")
        return manifest

    def retry_failed(self, max_retries: int = 3) -> int:
        """Reset failed jobs to pending if under retry limit. Returns count."""
        count = 0
        for job in self.jobs:
            if job.status == "failed" and job.retry_count < max_retries:
                job.status = "pending"
                job.retry_count += 1
                job.error = None
                count += 1
        self._save()
        return count
