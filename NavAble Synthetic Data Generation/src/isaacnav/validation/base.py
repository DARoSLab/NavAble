"""Abstract base class for image validation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ValidationResult:
    """Result of validating whether an image contains the expected class."""

    contains_class: bool
    actual_content: str = ""  # What the image actually contains (if not the expected class)
    confidence: float = 0.0
    details: dict = field(default_factory=dict)


class BaseValidator(ABC):

    @abstractmethod
    def __init__(self, config: dict):
        ...

    @abstractmethod
    def validate(
        self,
        image_path: Path,
        expected_class: str,
        prompt: str | None = None,
    ) -> ValidationResult:
        """Check if image contains the expected class.

        Args:
            image_path: Path to the image file.
            expected_class: The class name the image should contain.
            prompt: Optional custom question prompt. If provided, this replaces
                the default question (output format instructions are still
                appended by the validator). May contain {class_name} placeholder.

        Returns ValidationResult with contains_class=True/False and
        optional actual_content string describing what's really in the image.
        """
        ...
