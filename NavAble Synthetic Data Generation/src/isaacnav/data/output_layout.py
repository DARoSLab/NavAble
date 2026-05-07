"""Structured output directory layout for per-object asset storage."""

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class OutputLayout:
    """Manages per-object output directory structure.

    Layout:
        output/{category}/{object_id}/
            ├── image.jpg
            ├── mask_{class}.png
            ├── mesh.glb
            ├── mesh_gs.ply
            ├── asset.usdz
            ├── textures/
            └── metadata.json
    """

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)

    def create_object_dir(
        self, category: str, source_id: str = None
    ) -> Path:
        """Create and return path for a new object's output directory."""
        safe_category = category.lower().replace(" ", "_").replace("-", "_")
        obj_id = source_id or uuid.uuid4().hex[:12]
        safe_id = f"{safe_category}_{obj_id}"

        obj_dir = self.base_dir / safe_category / safe_id
        obj_dir.mkdir(parents=True, exist_ok=True)
        (obj_dir / "textures").mkdir(exist_ok=True)
        return obj_dir

    def copy_original_image(self, image_path: Path, obj_dir: Path) -> Path:
        """Copy original image to the object directory."""
        dest = obj_dir / f"image{image_path.suffix}"
        shutil.copy2(str(image_path), str(dest))
        return dest

    def write_metadata(
        self,
        obj_dir: Path,
        category: str,
        source_meta: Optional[dict] = None,
        masking_meta: Optional[dict] = None,
        reconstruction_meta: Optional[dict] = None,
        conversion_meta=None,
    ) -> Path:
        """Write combined metadata.json for the object."""
        metadata = {
            "object_id": obj_dir.name,
            "category": category,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": "2.2",
            "source": source_meta or {},
            "masking": masking_meta or {},
            "reconstruction": reconstruction_meta or {},
            "conversion": conversion_meta or {},
        }

        meta_path = obj_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        return meta_path
