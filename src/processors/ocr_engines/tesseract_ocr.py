"""Tesseract OCR engine wrapper using the BaseOCREngine interface."""

from typing import Optional, Tuple

from PIL import Image, ImageEnhance, ImageFilter

from src.processors.ocr_engines.base_ocr import BaseOCREngine


class TesseractOCR(BaseOCREngine):
    """OCR engine using pytesseract (Tesseract).

    Wraps the existing OCR preprocessing logic into the BaseOCREngine interface.
    """

    def __init__(self, tesseract_path: Optional[str] = None) -> None:
        """Initialize the Tesseract OCR engine.

        Args:
            tesseract_path: Optional path to the tesseract binary.
        """
        self.tesseract_path = tesseract_path
        if tesseract_path:
            try:
                import pytesseract

                pytesseract.pytesseract.tesseract_cmd = tesseract_path
            except ImportError:
                pass

    def extract_text(self, image_path: str) -> Tuple[str, float]:
        """Extract text from an image using Tesseract OCR.

        Args:
            image_path: Path to the image file.

        Returns:
            Tuple of (extracted text, confidence score 0-1).
        """
        import pytesseract

        image = Image.open(image_path)
        preprocessed = self._preprocess_image(image)

        # Get detailed OCR data including confidence
        data = pytesseract.image_to_data(
            preprocessed, output_type=pytesseract.Output.DICT
        )

        # Extract text
        text = pytesseract.image_to_string(preprocessed)

        # Calculate average confidence from word-level confidences
        confidences = [
            int(conf)
            for conf in data["conf"]
            if conf != "-1" and str(conf).strip()
        ]
        avg_confidence = (
            sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
        )

        return text.strip(), avg_confidence

    def is_available(self) -> bool:
        """Check if pytesseract is importable.

        Returns:
            True if pytesseract can be imported.
        """
        try:
            import pytesseract  # noqa: F401

            return True
        except ImportError:
            return False

    def get_name(self) -> str:
        """Get the engine name.

        Returns:
            Engine name string.
        """
        return "tesseract"

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess an image for better OCR results.

        Applies grayscale conversion, contrast enhancement, sharpening,
        and threshold binarization.

        Args:
            image: PIL Image object.

        Returns:
            Preprocessed PIL Image.
        """
        # Convert to grayscale
        if image.mode != "L":
            image = image.convert("L")

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)

        # Apply sharpening filter for denoising
        image = image.filter(ImageFilter.SHARPEN)

        # Apply threshold for binarization
        image = image.point(lambda x: 0 if x < 128 else 255)

        return image
