"""Grounded SAM 2 masking strategy -- class-aware segmentation.

Uses Grounding DINO for text-to-box detection and SAM 2.1 for
box-to-mask segmentation. For composite objects (escalator, handrail),
multiple phrases can be configured per class and the resulting masks
are unioned into a single coherent mask.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from isaacnav.masking.base import BaseMaskingStrategy, MaskResult


class GroundedSam2Masking(BaseMaskingStrategy):
    """Class-aware masking using Grounding DINO + SAM 2.1 with prompt expansion."""

    def __init__(self, config: dict):
        self.config = config
        self.grounding_model_id = config.get(
            "grounding_model", "IDEA-Research/grounding-dino-base"
        )
        self.sam2_checkpoint = config.get(
            "sam2_checkpoint",
            "extern/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt",
        )
        self.sam2_model_cfg = config.get(
            "sam2_model_cfg", "configs/sam2.1/sam2.1_hiera_l.yaml"
        )
        self.box_threshold = config.get("box_threshold", 0.3)
        self.text_threshold = config.get("text_threshold", 0.25)
        self.nms_threshold = config.get("nms_threshold", 0.8)
        self.class_prompts = config.get("class_prompts", {})
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self._grounding_model = None
        self._sam2_predictor = None

    def load_model(self) -> None:
        gsam2_path = str(Path("extern/Grounded-SAM-2").resolve())
        if gsam2_path not in sys.path:
            sys.path.insert(0, gsam2_path)

        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            self._grounding_processor = AutoProcessor.from_pretrained(
                self.grounding_model_id
            )
            self._grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.grounding_model_id
            ).to(self.device)
            print(f"Loaded Grounding DINO: {self.grounding_model_id}")
        except ImportError:
            raise ImportError(
                "transformers package required for Grounded SAM 2. "
                "Install with: pip install transformers"
            )

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            sam2_checkpoint = str(Path(self.sam2_checkpoint).resolve())
            sam2_model_cfg = self.sam2_model_cfg

            sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=self.device)
            self._sam2_predictor = SAM2ImagePredictor(sam2_model)
            print(f"Loaded SAM 2.1 from {sam2_checkpoint}")
        except ImportError:
            raise ImportError(
                "sam2 package required. Install from extern/Grounded-SAM-2 or "
                "pip install sam-2"
            )

    def _phrases_for_class(self, class_name: str) -> list[str]:
        """Return the list of DINO prompt phrases for a class.

        Falls back to the lowercased class name if no override is configured.
        """
        phrases = self.class_prompts.get(class_name)
        if phrases is None:
            return [class_name.lower()]
        if isinstance(phrases, str):
            phrases = [phrases]
        # Normalize: lowercase, strip, dedup while preserving order.
        seen = set()
        out = []
        for p in phrases:
            p = str(p).strip().lower()
            if p and p not in seen:
                seen.add(p)
                out.append(p)
        return out or [class_name.lower()]

    def _match_class(self, label: str, class_phrases: dict[str, list[str]]) -> Optional[str]:
        """Assign a DINO label string to one of the requested classes."""
        if len(class_phrases) == 1:
            return next(iter(class_phrases.keys()))

        label_l = label.lower().strip()
        label_words = set(label_l.split())

        best_cls = None
        best_overlap = 0
        for cls, phrases in class_phrases.items():
            for phrase in phrases:
                phrase_words = set(phrase.split())
                overlap = len(phrase_words & label_words)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_cls = cls
        return best_cls

    def segment(
        self,
        image: np.ndarray,
        class_names: Optional[list[str]] = None,
        point_prompts: Optional[list[tuple[int, int]]] = None,
    ) -> list[MaskResult]:
        if self._grounding_model is None or self._sam2_predictor is None:
            self.load_model()

        if not class_names:
            raise ValueError(
                "grounded_sam2 strategy requires class_names. "
                "Pass a list of target classes like ['elevator', 'door']."
            )

        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(image)
        h, w = image.shape[:2]

        # Build expanded prompt from per-class phrase lists.
        class_phrases = {cls: self._phrases_for_class(cls) for cls in class_names}
        all_phrases = []
        seen = set()
        for phrases in class_phrases.values():
            for p in phrases:
                if p not in seen:
                    seen.add(p)
                    all_phrases.append(p)
        text_prompt = ". ".join(all_phrases) + "."
        print(f"Grounding DINO prompt: {text_prompt!r}")

        # Run Grounding DINO.
        inputs = self._grounding_processor(
            images=pil_image, text=text_prompt, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._grounding_model(**inputs)

        results = self._grounding_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(h, w)],
        )[0]

        boxes = results["boxes"]
        scores = results["scores"]
        labels = results.get("text_labels", results.get("labels", []))

        if len(boxes) == 0:
            print(f"No objects detected for classes: {class_names}")
            return []

        print(f"Grounding DINO detected {len(boxes)} boxes: {labels}")

        # NMS across all classes.
        from torchvision.ops import nms

        keep = nms(boxes, scores, self.nms_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = [labels[i] for i in keep]

        # Run SAM 2.1 with box prompts.
        self._sam2_predictor.set_image(image)
        input_boxes = boxes.cpu().numpy()
        masks, sam_scores, _ = self._sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        # Collect per-class detections (mask + box + score).
        per_class: dict[str, dict] = {}
        for i in range(len(masks)):
            mask = masks[i]
            if mask.ndim == 3:
                mask = mask[0]
            mask_bool = mask.astype(bool)

            cls = self._match_class(labels[i], class_phrases)
            if cls is None:
                continue  # label didn't map to any requested class

            entry = per_class.setdefault(
                cls,
                {
                    "mask": np.zeros((h, w), dtype=bool),
                    "boxes": [],
                    "scores": [],
                    "labels": [],
                },
            )
            entry["mask"] |= mask_bool
            entry["boxes"].append(input_boxes[i])
            entry["scores"].append(float(scores[i]))
            entry["labels"].append(labels[i])

        mask_results = []
        for cls, entry in per_class.items():
            boxes_arr = np.stack(entry["boxes"])
            x0, y0 = boxes_arr[:, :2].min(axis=0)
            x1, y1 = boxes_arr[:, 2:].max(axis=0)
            coverage = float(entry["mask"].sum()) / float(h * w)
            print(
                f"  [{cls}] unioned {len(entry['boxes'])} detections "
                f"({entry['labels']}), coverage={coverage:.1%}"
            )
            mask_results.append(
                MaskResult(
                    mask=entry["mask"],
                    class_name=cls,
                    confidence=max(entry["scores"]),
                    bbox=(
                        float(x0),
                        float(y0),
                        float(x1 - x0),
                        float(y1 - y0),
                    ),
                    source_strategy="grounded_sam2",
                )
            )

        mask_results.sort(key=lambda x: x.confidence, reverse=True)
        print(f"Produced {len(mask_results)} unioned masks via Grounded SAM 2")
        return mask_results
