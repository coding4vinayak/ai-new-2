"""Image document processor using OCR for text extraction."""

from pathlib import Path
from typing import Optional
from uuid import uuid4

from PIL import Image, ImageEnhance

from src.models.document import Document, DocumentPage, FileType
from src.processors.ocr_engine import OCREngine


class ImageProcessor:
    """Processor for image documents (JPG, PNG, TIFF)."""

    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}

    def __init__(self, ocr_engine: Optional[OCREngine] = None) -> None:
        """Initialize the image processor.

        Args:
            ocr_engine: OCR engine instance for text extraction.
        """
        self.ocr_engine = ocr_engine or OCREngine()

    def process(self, file_path: str) -> Document:
        """Process an image file and extract text content via OCR.

        Args:
            file_path: Path to the image file.

        Returns:
            Document model with extracted content.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported image format: {path.suffix}. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )

        # Preprocess image for better OCR
        preprocessed_path = self._preprocess_for_ocr(path)
        target_path = preprocessed_path or str(path)

        # Extract text using OCR
        text, confidence = self.ocr_engine.extract_text(target_path)

        # Detect language
        language = self.ocr_engine.detect_language(target_path)

        page = DocumentPage(
            page_number=1,
            text=text,
            confidence=confidence,
        )

        return Document(
            id=uuid4(),
            filename=path.name,
            file_type=FileType.IMAGE,
            content=text,
            raw_text=text,
            page_count=1,
            language=language,
            pages=[page],
            file_path=str(path.absolute()),
            metadata={
                "original_format": path.suffix.lower(),
                "ocr_confidence": confidence,
            },
        )

    def _preprocess_for_ocr(self, image_path: Path) -> Optional[str]:
        """Preprocess image for better OCR results.

        Applies resizing and contrast enhancement.

        Args:
            image_path: Path to the original image.

        Returns:
            Path to preprocessed image, or None if preprocessing is not needed.
        """
        try:
            image = Image.open(image_path)

            # Resize if too small
            min_dimension = 300
            width, height = image.size
            if width < min_dimension or height < min_dimension:
                scale = max(min_dimension / width, min_dimension / height)
                new_size = (int(width * scale), int(height * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            # Enhance contrast
            if image.mode != "L":
                image = image.convert("L")
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)

            # Save preprocessed image
            preprocessed_path = image_path.parent / f"_preprocessed_{image_path.name}"
            image.save(str(preprocessed_path))
            return str(preprocessed_path)
        except Exception:
            return None
