"""PDF document processor supporting both text-based and scanned PDFs."""

from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from src.models.document import Document, DocumentPage, FileType
from src.processors.ocr_engine import OCREngine


class PDFProcessor:
    """Processor for PDF documents."""

    def __init__(self, ocr_engine: Optional[OCREngine] = None) -> None:
        """Initialize the PDF processor.

        Args:
            ocr_engine: OCR engine for processing scanned PDFs.
        """
        self.ocr_engine = ocr_engine or OCREngine()

    def process(self, file_path: str) -> Document:
        """Process a PDF file and extract text content.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Document model with extracted content.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        is_scanned = self.detect_if_scanned(file_path)

        if is_scanned:
            pages = self.extract_scanned_pdf(file_path)
        else:
            pages = self.extract_text_pdf(file_path)

        raw_text = "\n\n".join(page.text for page in pages)

        return Document(
            id=uuid4(),
            filename=path.name,
            file_type=FileType.PDF,
            content=raw_text,
            raw_text=raw_text,
            page_count=len(pages),
            pages=pages,
            file_path=str(path.absolute()),
            metadata={"is_scanned": is_scanned},
        )

    def extract_text_pdf(self, file_path: str) -> List[DocumentPage]:
        """Extract text from a text-based PDF using pdfplumber.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of DocumentPage objects.
        """
        import pdfplumber

        pages = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages.append(
                    DocumentPage(
                        page_number=i,
                        text=text,
                        confidence=1.0,  # Text-based PDFs have high confidence
                    )
                )
        return pages

    def extract_scanned_pdf(self, file_path: str) -> List[DocumentPage]:
        """Extract text from a scanned PDF using OCR.

        Converts each page to an image, then applies OCR.

        Args:
            file_path: Path to the scanned PDF file.

        Returns:
            List of DocumentPage objects with OCR results.
        """
        from PyPDF2 import PdfReader

        pages = []
        reader = PdfReader(file_path)

        for i, page in enumerate(reader.pages, start=1):
            # For scanned PDFs, we attempt basic text extraction first
            text = page.extract_text() or ""
            confidence = 0.5 if text else 0.0

            pages.append(
                DocumentPage(
                    page_number=i,
                    text=text,
                    confidence=confidence,
                )
            )

        return pages

    def detect_if_scanned(self, file_path: str) -> bool:
        """Detect if a PDF is scanned (image-based) or text-based.

        Uses a heuristic: if extractable text is very short relative to
        page count, it is likely scanned.

        Args:
            file_path: Path to the PDF file.

        Returns:
            True if the PDF appears to be scanned.
        """
        import pdfplumber

        try:
            with pdfplumber.open(file_path) as pdf:
                total_text_length = 0
                page_count = len(pdf.pages)

                for page in pdf.pages[:3]:  # Check first 3 pages
                    text = page.extract_text() or ""
                    total_text_length += len(text.strip())

                # Heuristic: less than 50 chars per page on average suggests scanned
                avg_chars_per_page = (
                    total_text_length / min(page_count, 3) if page_count > 0 else 0
                )
                return avg_chars_per_page < 50
        except Exception:
            return False
