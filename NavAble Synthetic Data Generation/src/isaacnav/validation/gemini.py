"""Gemini-based image validation.

Uses Google's Gemini VLM to verify that a crawled image
actually contains the expected object class.
"""

import os
from pathlib import Path

from isaacnav.validation.base import BaseValidator, ValidationResult


class GeminiValidator(BaseValidator):

    def __init__(self, config: dict):
        self.config = config
        self.model_name = config.get("model", "gemini-2.5-flash")
        self.api_key_env = config.get("api_key_env", "GOOGLE_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError:
                raise ImportError(
                    "google-genai package required for Gemini validation. "
                    "Install with: pip install google-genai"
                )

            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise ValueError(
                    f"Set {self.api_key_env} environment variable for Gemini API access"
                )
            self._client = genai.Client(api_key=api_key)
        return self._client

    def validate(
        self,
        image_path: Path,
        expected_class: str,
        prompt: str | None = None,
    ) -> ValidationResult:
        from PIL import Image

        client = self._get_client()
        image_path = Path(image_path)
        image = Image.open(image_path)

        if prompt is not None:
            question = prompt.format(class_name=expected_class)
        else:
            question = f"Does this image contain a '{expected_class}'?"

        prompt_text = (
            f"You are a strict data-curation assistant filtering images for a 3D object reconstruction pipeline (SAM-3D). "
            f"Your job is to reject any image that would fail to generate a clean, isolated 3D mesh.\n\n"
            f"CRITICAL REJECTION CRITERIA - Answer NO if the image has ANY of the following:\n"
            f"- It is a drawing, 3D render, CAD model, diagram, icon, or screenshot.\n"
            f"- The object is heavily occluded by people, vehicles, or other objects.\n"
            f"- The object is severely truncated (cut off by the edge of the frame).\n"
            f"- The image has heavy, intrusive stock-photo watermarks over the subject.\n"
            f"- The object is too distant or small to make out structural geometry.\n\n"
            f"SPECIFIC OBJECT EVALUATION:\n"
            f"{question}\n\n"
            f"Answer in this exact format:\n"
            f"CONTAINS: YES or NO\n"
            f"ACTUAL: <A brief 1-sentence factual description of what the image shows>\n"
            f"CONFIDENCE: <0.0 to 1.0>"
        )

        response = client.models.generate_content(
            model=self.model_name,
            contents=[prompt_text, image],
        )

        text = response.text.strip()
        return self._parse_response(text)

    def _parse_response(self, text: str) -> ValidationResult:
        contains = False
        actual = ""
        confidence = 0.0

        for line in text.splitlines():
            line = line.strip()
            if line.upper().startswith("CONTAINS:"):
                val = line.split(":", 1)[1].strip().upper()
                contains = val in ("YES", "TRUE", "1")
            elif line.upper().startswith("ACTUAL:"):
                actual = line.split(":", 1)[1].strip()
            elif line.upper().startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except ValueError:
                    confidence = 0.5

        return ValidationResult(
            contains_class=contains,
            actual_content=actual,
            confidence=confidence,
            details={"raw_response": text},
        )
