"""Tests for core data models."""

from uuid import uuid4

import pytest

from src.models.confidence import ConfidenceReport, ConfidenceScore
from src.models.document import Document, DocumentPage, FileType
from src.models.extraction_result import (
    BatchExtractionResult,
    ExtractionMode,
    ExtractionResult,
)


class TestDocument:
    """Tests for the Document model."""

    def test_create_document(self):
        doc = Document(filename="test.pdf", file_type=FileType.PDF)
        assert doc.filename == "test.pdf"
        assert doc.file_type == FileType.PDF
        assert doc.page_count == 1
        assert doc.id is not None

    def test_document_with_pages(self):
        pages = [
            DocumentPage(page_number=1, text="Page 1 text", confidence=0.95),
            DocumentPage(page_number=2, text="Page 2 text", confidence=0.88),
        ]
        doc = Document(
            filename="multi.pdf",
            file_type=FileType.PDF,
            page_count=2,
            pages=pages,
        )
        assert len(doc.pages) == 2
        assert doc.pages[0].confidence == 0.95

    def test_file_type_enum(self):
        assert FileType.PDF == "pdf"
        assert FileType.IMAGE == "image"
        assert FileType.DOCX == "docx"
        assert FileType.TEXT == "text"


class TestConfidenceScore:
    """Tests for confidence scoring models."""

    def test_create_score(self):
        score = ConfidenceScore(
            score=0.85, method="local_ner", field_name="vendor_name"
        )
        assert score.score == 0.85
        assert score.method == "local_ner"
        assert score.field_name == "vendor_name"

    def test_score_bounds(self):
        with pytest.raises(Exception):
            ConfidenceScore(score=1.5, method="test", field_name="test")
        with pytest.raises(Exception):
            ConfidenceScore(score=-0.1, method="test", field_name="test")

    def test_confidence_report(self):
        scores = {
            "vendor_name": ConfidenceScore(
                score=0.9, method="api", field_name="vendor_name"
            ),
            "amount": ConfidenceScore(
                score=0.5, method="local", field_name="amount"
            ),
        }
        report = ConfidenceReport(
            scores=scores, overall_confidence=0.7, threshold=0.7
        )
        low = report.compute_low_confidence_fields()
        assert "amount" in low
        assert "vendor_name" not in low


class TestExtractionResult:
    """Tests for extraction result models."""

    def test_create_result(self):
        doc_id = uuid4()
        report = ConfidenceReport(overall_confidence=0.85)
        result = ExtractionResult(
            document_id=doc_id,
            extraction_mode=ExtractionMode.HYBRID,
            entities={"vendor": "Acme"},
            confidence_report=report,
            raw_text="Sample text",
            processing_time_ms=200.0,
        )
        assert result.document_id == doc_id
        assert result.extraction_mode == ExtractionMode.HYBRID
        assert result.entities["vendor"] == "Acme"

    def test_batch_result(self):
        batch = BatchExtractionResult(
            batch_id=uuid4(),
            total_documents=10,
            successful=8,
            failed=2,
        )
        assert batch.total_documents == 10
        assert batch.successful == 8
