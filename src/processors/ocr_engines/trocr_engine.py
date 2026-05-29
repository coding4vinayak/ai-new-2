"""TrOCR engine using Microsoft's transformer-based OCR model."""

import logging
from typing import Optional, Tuple

from src.processors.ocr_engines.base_ocr import BaseOCREngine

logger = logging.getLogger(__name__)

# Try importing transformers and torch
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

try:
    import torch  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class TrOCREngine(BaseOCREngine):
    """OCR engine using Microsoft TrOCR transformer model.

    Supports both printed text and handwritten text recognition.
    Uses 'microsoft/trocr-base-printed' for printed text and
    'microsoft/trocr-base-handwritten' for handwriting.
    """

    def __init__(self, mode: str = "printed") -> None:
        """Initialize the TrOCR engine.

        Args:
            mode: Either 'printed' or 'handwritten' to select the model variant.
        """
        self.mode = mode
        if mode == "handwritten":
            self.model_name = "microsoft/trocr-base-handwritten"
        else:
            self.model_name = "microsoft/trocr-base-printed"
        self._processor: Optional[object] = None
        self._model: Optional[object] = None

    def _load_model(self) -> None:
        """Lazily load the TrOCR model and processor."""
        if self._processor is None:
            self._processor = TrOCRProcessor.from_pretrained(self.model_name)
            self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name)

    def extract_text(self, image_path: str) -> Tuple[str, float]:
        """Extract text from an image using TrOCR.

        Splits the image into horizontal line segments and runs TrOCR on each.
        Returns concatenated text with confidence derived from model logits.

        Args:
            image_path: Path to the image file.

        Returns:
            Tuple of (extracted text, confidence score 0-1).
        """
        from PIL import Image

        self._load_model()

        image = Image.open(image_path).convert("RGB")
        lines = self._split_into_lines(image)

        all_text = []
        all_confidences = []

        for line_image in lines:
            text, confidence = self._process_line(line_image)
            if text.strip():
                all_text.append(text.strip())
                all_confidences.append(confidence)

        if not all_text:
            return "", 0.0

        combined_text = "\n".join(all_text)
        avg_confidence = sum(all_confidences) / len(all_confidences)

        return combined_text, avg_confidence

    def _process_line(self, image) -> Tuple[str, float]:
        """Process a single line image through TrOCR.

        Args:
            image: PIL Image of a text line.

        Returns:
            Tuple of (text, confidence).
        """
        import torch

        pixel_values = self._processor(
            images=image, return_tensors="pt"
        ).pixel_values

        with torch.no_grad():
            outputs = self._model.generate(
                pixel_values, output_scores=True, return_dict_in_generate=True
            )

        # Decode the generated text
        generated_ids = outputs.sequences
        text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        # Calculate confidence from scores
        if outputs.scores:
            scores = torch.stack(outputs.scores, dim=1)
            probs = torch.softmax(scores, dim=-1)
            max_probs = probs.max(dim=-1).values
            confidence = max_probs.mean().item()
        else:
            confidence = 0.5

        return text, confidence

    def _split_into_lines(self, image) -> list:
        """Split an image into horizontal line segments.

        Uses a simple projection-based approach to detect line boundaries.

        Args:
            image: PIL Image to split.

        Returns:
            List of PIL Images, one per detected line.
        """
        import numpy as np

        img_array = np.array(image.convert("L"))
        # Horizontal projection - sum of dark pixels per row
        projection = np.sum(img_array < 128, axis=1)

        # Find line boundaries based on gaps in the projection
        threshold = projection.max() * 0.1 if projection.max() > 0 else 0
        in_line = False
        lines = []
        start = 0

        for i, val in enumerate(projection):
            if val > threshold and not in_line:
                start = i
                in_line = True
            elif val <= threshold and in_line:
                if i - start > 5:  # Minimum line height
                    lines.append((start, i))
                in_line = False

        if in_line and len(projection) - start > 5:
            lines.append((start, len(projection)))

        # If no lines detected, return the whole image
        if not lines:
            return [image]

        # Crop each line
        width = image.width
        line_images = []
        for top, bottom in lines:
            line_img = image.crop((0, top, width, bottom))
            line_images.append(line_img)

        return line_images

    def is_available(self) -> bool:
        """Check if transformers and torch are importable.

        Returns:
            True if both dependencies are available.
        """
        return _TRANSFORMERS_AVAILABLE and _TORCH_AVAILABLE

    def get_name(self) -> str:
        """Get the engine name.

        Returns:
            Engine name string.
        """
        return f"trocr-{self.mode}"
