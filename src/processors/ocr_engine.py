"""OCR engine for text extraction from images using Tesseract.

This module maintains backward compatibility with the original OCREngine class
while providing factory functions to access the new multi-engine OCR system.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def get_ocr_engine(engine_name: str, **kwargs) -> "BaseOCREngine":
    """Factory function to get an OCR engine by name.

    Args:
        engine_name: Name of the engine ('tesseract', 'trocr', 'paddleocr', 'doctr').
        **kwargs: Additional keyword arguments passed to the engine constructor.

    Returns:
        An instance of the requested OCR engine.

    Raises:
        ValueError: If the engine name is not recognized.
    """
    from src.processors.ocr_engines import (
        DocTREngine,
        PaddleOCREngine,
        TesseractOCR,
        TrOCREngine,
    )
    from src.processors.ocr_engines.base_ocr import BaseOCREngine

    engines: Dict[str, type] = {
        "tesseract": TesseractOCR,
        "trocr": TrOCREngine,
        "paddleocr": PaddleOCREngine,
        "doctr": DocTREngine,
    }

    engine_class = engines.get(engine_name)
    if engine_class is None:
        raise ValueError(
            f"Unknown OCR engine: {engine_name}. "
            f"Available: {list(engines.keys())}"
        )

    return engine_class(**kwargs)


def get_ocr_ensemble(
    engine_names: Optional[List[str]] = None,
    min_engines: int = 1,
    similarity_threshold: float = 0.7,
    fallback_strategy: str = "highest_confidence",
) -> "OCREnsemble":
    """Factory function to create a configured OCR ensemble.

    Args:
        engine_names: List of engine names to include. If None, uses all available.
        min_engines: Minimum number of engines that must succeed.
        similarity_threshold: Threshold for text similarity (0-1).
        fallback_strategy: Strategy when engines disagree.

    Returns:
        Configured OCREnsemble instance.
    """
    from src.processors.ocr_ensemble import OCREnsemble
    from src.processors.ocr_engines import (
        DocTREngine,
        PaddleOCREngine,
        TesseractOCR,
        TrOCREngine,
    )

    all_engines = {
        "tesseract": TesseractOCR,
        "trocr": TrOCREngine,
        "paddleocr": PaddleOCREngine,
        "doctr": DocTREngine,
    }

    if engine_names is None:
        engine_names = list(all_engines.keys())

    engines = []
    for name in engine_names:
        engine_class = all_engines.get(name)
        if engine_class:
            engines.append(engine_class())

    return OCREnsemble(
        engines=engines,
        min_engines=min_engines,
        similarity_threshold=similarity_threshold,
        fallback_strategy=fallback_strategy,
    )
