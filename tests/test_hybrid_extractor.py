"""Tests for the hybrid extractor with mocked local and API components."""

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.extractors.hybrid_extractor import HybridExtractor
from src.models.confidence import ConfidenceReport, ConfidenceScore
from src.models.document import Document, FileType
from src.models.extraction_result import ExtractionMode, ExtractionResult


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document(
        id=uuid4(),
        filename="test.txt",
        file_type=FileType.TEXT,
        content="Contract between Acme Corp and Beta LLC for $50,000.",
        raw_text="Contract between Acme Corp and Beta LLC for $50,000.",
        page_count=1,
    )


@pytest.fixture
def high_confidence_local_result(sample_document):
    """Create a local extraction result with high confidence."""
    return ExtractionResult(
        document_id=sample_document.id,
        extraction_mode=ExtractionMode.LOCAL,
        entities={
            "organization_names": ["Acme Corp", "Beta LLC"],
            "monetary_amounts": ["$50,000"],
        },
        confidence_report=ConfidenceReport(
            scores={
                "organization_names": ConfidenceScore(
                    score=0.9, method="local_ner", field_name="organization_names"
                ),
                "monetary_amounts": ConfidenceScore(
                    score=0.85, method="local_regex", field_name="monetary_amounts"
                ),
            },
            overall_confidence=0.875,
            threshold=0.7,
        ),
        raw_text="Contract between Acme Corp and Beta LLC for $50,000.",
        processing_time_ms=50.0,
    )


@pytest.fixture
def low_confidence_local_result(sample_document):
    """Create a local extraction result with some low confidence fields."""
    return ExtractionResult(
        document_id=sample_document.id,
        extraction_mode=ExtractionMode.LOCAL,
        entities={
            "organization_names": ["Acme Corp"],
            "dates": ["unclear date reference"],
        },
        confidence_report=ConfidenceReport(
            scores={
                "organization_names": ConfidenceScore(
                    score=0.8, method="local_ner", field_name="organization_names"
                ),
                "dates": ConfidenceScore(
                    score=0.4, method="local_regex", field_name="dates"
                ),
            },
            overall_confidence=0.6,
            threshold=0.7,
        ),
        raw_text="Contract between Acme Corp and Beta LLC for $50,000.",
        processing_time_ms=45.0,
    )


@pytest.fixture
def api_result(sample_document):
    """Create an API extraction result."""
    return ExtractionResult(
        document_id=sample_document.id,
        extraction_mode=ExtractionMode.API,
        entities={
            "organization_names": ["Acme Corp", "Beta LLC"],
            "dates": ["2024-01-15"],
            "monetary_amounts": ["$50,000"],
        },
        confidence_report=ConfidenceReport(
            scores={
                "organization_names": ConfidenceScore(
                    score=0.92, method="api_openai", field_name="organization_names"
                ),
                "dates": ConfidenceScore(
                    score=0.9, method="api_openai", field_name="dates"
                ),
                "monetary_amounts": ConfidenceScore(
                    score=0.9, method="api_openai", field_name="monetary_amounts"
                ),
            },
            overall_confidence=0.9,
            threshold=0.7,
        ),
        raw_text="Contract between Acme Corp and Beta LLC for $50,000.",
        processing_time_ms=1200.0,
    )


@pytest.mark.asyncio
async def test_high_confidence_skips_api(
    sample_document, high_confidence_local_result
):
    """Test that high confidence local results skip API escalation."""
    mock_local = AsyncMock()
    mock_local.extract = AsyncMock(return_value=high_confidence_local_result)
    mock_local.validate_document = MagicMock()

    mock_api = AsyncMock()
    mock_api.extract = AsyncMock()

    hybrid = HybridExtractor(
        local_extractor=mock_local,
        api_extractor=mock_api,
        confidence_threshold=0.7,
    )

    result = await hybrid.extract(sample_document)

    # API should NOT have been called since all fields are above threshold
    mock_api.extract.assert_not_called()
    assert result.extraction_mode == ExtractionMode.HYBRID
    assert "organization_names" in result.entities


@pytest.mark.asyncio
async def test_low_confidence_triggers_api(
    sample_document, low_confidence_local_result, api_result
):
    """Test that low confidence fields trigger API escalation."""
    mock_local = AsyncMock()
    mock_local.extract = AsyncMock(return_value=low_confidence_local_result)
    mock_local.validate_document = MagicMock()

    mock_api = AsyncMock()
    mock_api.extract = AsyncMock(return_value=api_result)

    hybrid = HybridExtractor(
        local_extractor=mock_local,
        api_extractor=mock_api,
        confidence_threshold=0.7,
    )

    result = await hybrid.extract(sample_document)

    # API should have been called for the low-confidence 'dates' field
    mock_api.extract.assert_called_once()
    # Dates should come from API result
    assert result.entities["dates"] == ["2024-01-15"]


@pytest.mark.asyncio
async def test_result_merging(
    sample_document, low_confidence_local_result, api_result
):
    """Test that results are properly merged from local and API."""
    mock_local = AsyncMock()
    mock_local.extract = AsyncMock(return_value=low_confidence_local_result)
    mock_local.validate_document = MagicMock()

    mock_api = AsyncMock()
    mock_api.extract = AsyncMock(return_value=api_result)

    hybrid = HybridExtractor(
        local_extractor=mock_local,
        api_extractor=mock_api,
        confidence_threshold=0.7,
    )

    result = await hybrid.extract(sample_document)

    # Should have merged entities from both
    assert "organization_names" in result.entities
    assert "dates" in result.entities
    # monetary_amounts from API since it was a new field not in local
    assert "monetary_amounts" in result.entities


@pytest.mark.asyncio
async def test_api_failure_falls_back_to_local(
    sample_document, low_confidence_local_result
):
    """Test that API failure gracefully falls back to local results."""
    mock_local = AsyncMock()
    mock_local.extract = AsyncMock(return_value=low_confidence_local_result)
    mock_local.validate_document = MagicMock()

    mock_api = AsyncMock()
    mock_api.extract = AsyncMock(side_effect=Exception("API unavailable"))

    hybrid = HybridExtractor(
        local_extractor=mock_local,
        api_extractor=mock_api,
        confidence_threshold=0.7,
    )

    result = await hybrid.extract(sample_document)

    # Should still return local results on API failure
    assert result is not None
    assert result.entities == low_confidence_local_result.entities
    assert any("failed" in w.lower() for w in result.warnings)
