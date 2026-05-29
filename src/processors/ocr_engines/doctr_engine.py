"""DocTR (Mindee) OCR engine wrapper using the BaseOCREngine interface."""

import logging
from typing import Optional, Tuple

from src.processors.ocr_engines.base_ocr import BaseOCREngine

logger = logging.getLogger(__name__)

# Try importing doctr
try:
    from doctr.models import ocr_predictor

    _DOCTR_AVAILABLE = True
except ImportError:
    _DOCTR_AVAILABLE = False


class DocTREngine(BaseOCREngine):
    """OCR engine using DocTR (Mindee).

    Processes full document pages and returns structured text with
    word-level confidence scores.
    """

    def __init__(self, pretrained: bool = True) -> None:
        """Initialize the DocTR engine.

        Args:
            pretrained: Whether to use pretrained model weights.
        """
        self.pretrained = pretrained
        self._predictor: Optional[object] = None

    def _load_model(self) -> None:
        """Lazily load the DocTR predictor."""
        if self._predictor is None:
            self._predictor = ocr_predictor(pretrained=self.pretrained)

    def extract_text(self, image_path: str) -> Tuple[str, float]:
        """Extract text from an image using DocTR.

        Args:
            image_path: Path to the image file.

        Returns:
            Tuple of (extracted text, confidence score 0-1).
        """
        from doctr.io import DocumentFile

        self._load_model()

        doc = DocumentFile.from_images(image_path)
        result = self._predictor(doc)

        texts = []
        confidences = []

        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_text = []
                    for word in line.words:
                        line_text.append(word.value)
                        confidences.append(word.confidence)
                    texts.append(" ".join(line_text))

        combined_text = "\n".join(texts)
        avg_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        return combined_text, avg_confidence

    def is_available(self) -> bool:
        """Check if doctr is importable.

        Returns:
            True if doctr can be imported.
        """
        return _DOCTR_AVAILABLE

    def get_name(self) -> str:
        """Get the engine name.

        Returns:
            Engine name string.
        """
        return "doctr"
