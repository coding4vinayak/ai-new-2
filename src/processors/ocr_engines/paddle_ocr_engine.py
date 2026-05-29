"""PaddleOCR engine wrapper using the BaseOCREngine interface."""

import logging
from typing import Optional, Tuple

from src.processors.ocr_engines.base_ocr import BaseOCREngine

logger = logging.getLogger(__name__)

# Try importing paddleocr
try:
    from paddleocr import PaddleOCR

    _PADDLEOCR_AVAILABLE = True
except ImportError:
    _PADDLEOCR_AVAILABLE = False


class PaddleOCREngine(BaseOCREngine):
    """OCR engine using PaddleOCR.

    Supports multi-language text recognition with angle classification.
    """

    def __init__(self, lang: str = "en", use_angle_cls: bool = True) -> None:
        """Initialize the PaddleOCR engine.

        Args:
            lang: Language code for OCR (e.g., 'en', 'ch', 'fr').
            use_angle_cls: Whether to use angle classification for rotated text.
        """
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self._ocr: Optional[object] = None

    def _load_model(self) -> None:
        """Lazily initialize the PaddleOCR model."""
        if self._ocr is None:
            self._ocr = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                show_log=False,
            )

    def extract_text(self, image_path: str) -> Tuple[str, float]:
        """Extract text from an image using PaddleOCR.

        Args:
            image_path: Path to the image file.

        Returns:
            Tuple of (extracted text, confidence score 0-1).
        """
        self._load_model()

        result = self._ocr.ocr(image_path, cls=self.use_angle_cls)

        if not result or not result[0]:
            return "", 0.0

        texts = []
        confidences = []

        for line in result[0]:
            # Each line is [bbox, (text, confidence)]
            text_info = line[1]
            text = text_info[0]
            confidence = text_info[1]
            texts.append(text)
            confidences.append(confidence)

        combined_text = "\n".join(texts)
        avg_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        return combined_text, avg_confidence

    def is_available(self) -> bool:
        """Check if paddleocr is importable.

        Returns:
            True if paddleocr can be imported.
        """
        return _PADDLEOCR_AVAILABLE

    def get_name(self) -> str:
        """Get the engine name.

        Returns:
            Engine name string.
        """
        return "paddleocr"
