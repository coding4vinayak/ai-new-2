"""OCR engine for text extraction from images using Tesseract."""

from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageEnhance, ImageFilter


class OCREngine:
    """Engine for optical character recognition using pytesseract."""

    def __init__(self, tesseract_path: Optional[str] = None) -> None:
        """Initialize the OCR engine.

        Args:
            tesseract_path: Path to the tesseract binary.
        """
        self.tesseract_path = tesseract_path
        if tesseract_path:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def extract_text(self, image_path: str) -> Tuple[str, float]:
        """Extract text from an image using OCR.

        Args:
            image_path: Path to the image file.

        Returns:
            Tuple of (extracted text, confidence score 0-1).
        """
        import pytesseract

        image = Image.open(image_path)
        preprocessed = self.preprocess_image(image)

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

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess an image for better OCR results.

        Applies deskew, threshold, and denoise operations.

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

    def detect_language(self, image_path: str) -> Optional[str]:
        """Detect the language of text in an image for OCR language hints.

        Args:
            image_path: Path to the image file.

        Returns:
            Detected language code or None.
        """
        import pytesseract

        image = Image.open(image_path)
        preprocessed = self.preprocess_image(image)

        try:
            osd = pytesseract.image_to_osd(preprocessed)
            for line in osd.split("\n"):
                if "Script:" in line:
                    script = line.split(":")[1].strip()
                    return self._script_to_lang(script)
        except Exception:
            pass

        return None

    def _script_to_lang(self, script: str) -> str:
        """Map a script name to a Tesseract language code.

        Args:
            script: Script name from OSD output.

        Returns:
            Tesseract language code.
        """
        script_map = {
            "Latin": "eng",
            "Cyrillic": "rus",
            "Arabic": "ara",
            "Han": "chi_sim",
            "Devanagari": "hin",
            "Japanese": "jpn",
            "Korean": "kor",
        }
        return script_map.get(script, "eng")
