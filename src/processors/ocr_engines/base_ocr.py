"""Base OCR engine interface that all OCR engines implement."""

from abc import ABC, abstractmethod
from typing import Tuple


class BaseOCREngine(ABC):
    """Abstract base class for OCR engines.

    All OCR engines must implement extract_text(), is_available(), and get_name().
    """

    @abstractmethod
    def extract_text(self, image_path: str) -> Tuple[str, float]:
        """Extract text from an image.

        Args:
            image_path: Path to the image file.

        Returns:
            Tuple of (extracted text, confidence score 0-1).
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this engine's dependencies are installed and usable.

        Returns:
            True if the engine can be used.
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Get the human-readable name of this OCR engine.

        Returns:
            Engine name string.
        """
        ...
