"""SAM 3D Objects reconstruction strategy.

Wraps Meta's SAM 3D Objects model for single-image 3D reconstruction.
Produces Gaussian splat PLY, textured GLB mesh, and optionally raw mesh PLY.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from isaacnav.reconstruction.base import BaseReconstructionStrategy, ReconstructionResult


class Sam3dReconstruction(BaseReconstructionStrategy):
    """3D reconstruction using Meta's SAM 3D Objects."""

    def __init__(self, config: dict):
        self.config = config
        self.sam3d_path = config.get("sam3d_path", "extern/sam-3d-objects")
        self.config_path = config.get(
            "config_path", "extern/sam-3d-objects/checkpoints/hf/pipeline.yaml"
        )
        self.compile = config.get("compile", False)
        self._inference = None

    def load_model(self) -> None:
        sam3d_abs = str(Path(self.sam3d_path).resolve())
        notebook_path = os.path.join(sam3d_abs, "notebook")

        # Add SAM3D to path if not already there
        if sam3d_abs not in sys.path:
            sys.path.insert(0, sam3d_abs)
        if notebook_path not in sys.path:
            sys.path.insert(0, notebook_path)

        os.environ.setdefault("CUDA_HOME", os.environ.get("CONDA_PREFIX", "/usr"))
        os.environ.setdefault("LIDRA_SKIP_INIT", "true")
        # RTX 5090 (Blackwell SM_120): spconv implicit_gemm kernels cause FPE
        os.environ.setdefault("SPCONV_ALGO", "native")

        from inference import Inference

        config_path = str(Path(self.config_path).resolve())
        print(f"Loading SAM-3D-objects model from {config_path}...")
        self._inference = Inference(config_path, compile=self.compile)

    def reconstruct(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        output_dir: Path,
        stem: str,
        seed: int = 42,
    ) -> ReconstructionResult:
        if self._inference is None:
            self.load_model()

        # SAM3D's load_image expects a file path, but we have an array.
        # We use the inference callable directly with image array + mask.
        from inference import load_image as sam3d_load_image

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Running 3D reconstruction with SAM-3D-objects...")
        output = self._inference(image, mask, seed=seed)

        result = ReconstructionResult(source_strategy="sam3d")
        result.metadata["seed"] = seed

        # Save gaussian splat as PLY
        if "gs" in output and output["gs"] is not None:
            ply_path = output_dir / f"{stem}_gs.ply"
            output["gs"].save_ply(str(ply_path))
            result.ply_path = ply_path
            print(f"Saved gaussian splat: {ply_path}")

        # Save GLB (mesh with texture)
        if "glb" in output and output["glb"] is not None:
            glb_path = output_dir / f"{stem}.glb"
            try:
                output["glb"].export(str(glb_path))
                result.glb_path = glb_path
                print(f"Saved GLB mesh: {glb_path}")
            except Exception as e:
                print(f"Could not save GLB: {e}")

        # Free the SAM3D output to avoid segfaults from stale CUDA tensors
        # (spconv native backend on Blackwell can crash when mesh tensors are accessed)
        # The GLB already contains the full textured mesh, so mesh PLY is redundant.
        del output

        return result
