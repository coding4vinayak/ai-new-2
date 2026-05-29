"""Tests for the EnsembleExtractor with mocked OCR and NER ensembles."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.extractors.ensemble_extractor import EnsembleExtractor
from src.models.confidence import ConfidenceReport, ConfidenceScore
from src.models.document import Document, FileType
from src.models.extraction_result import ExtractionMode, ExtractionResult


@pytest.fixture
def sample_text_document():
    """Create a sample text document for testing."""
    return Document(
        id=uuid4(),
        filename="test.txt",
        file_type=FileType.TEXT,
        content="John Smith works at Acme Corp. Invoice total: $5,000.",
        raw_text="John Smith works at Acme Corp. Invoice total: $5,000.",
        page_count=1,
        metadata={},
    )


@pytest.fixture
def sample_image_document():
    """Create a sample image document for testing."""
    return Document(
        id=uuid4(),
        filename="scan.png",
        file_type=FileType.IMAGE,
        content=None,
        raw_text=None,
        page_count=1,
        file_path="/tmp/scan.png",
        metadata={},
    )


@pytest.fixture
def mock_ocr_ensemble():
    """Create a mock OCR ensemble."""
    ensemble = MagicMock()
    ensemble.extract_text.return_value = (
        "John Smith works at Acme Corp. Invoice total: $5,000.",
        0.92,
    )
    return ensemble


@pytest.fixture
def mock_ner_ensemble():
    """Create a mock NER ensemble."""
    ensemble = MagicMock()
    ensemble.extract_entities.return_value = {
        "person_names": [("John Smith", 0.9)],
        "organization_names": [("Acme Corp", 0.85)],
        "monetary_amounts": [("$5,000", 0.88)],
    }
    return ensemble


@pytest.fixture
def mock_layoutlm_engine():
    """Create a mock LayoutLM engine."""
    engine = MagicMock()
    engine.is_available.return_value = True
    engine.extract_entities.return_value = {
        "person_names": [("John Smith", 0.92)],
        "organization_names": [("Acme Corp", 0.90)],
    }
    return engine


@pytest.fixture
def mock_local_llm_extractor():
    """Create a mock local LLM extractor."""
    extractor = AsyncMock()
    extractor.extract.return_value = ExtractionResult(
        document_id=uuid4(),
        extraction_mode=ExtractionMode.LOCAL_LLM,
        entities={
            "person_names": "John Smith",
            "organization_names": "Acme Corp",
            "invoice_total": "$5,000",
        },
        confidence_report=ConfidenceReport(
            scores={
                "person_names": ConfidenceScore(
                    score=0.75, method="local_llm", field_name="person_names"
                ),
            },
            overall_confidence=0.75,
            threshold=0.6,
        ),
        raw_text="test",
        processing_time_ms=100.0,
    )
    return extractor


# --- Full pipeline tests ---


@pytest.mark.asyncio
async def test_full_pipeline_with_all_engines(
    sample_text_document,
    mock_ocr_ensemble,
    mock_ner_ensemble,
    mock_layoutlm_engine,
    mock_local_llm_extractor,
):
    """Test EnsembleExtractor with all engines mocked and active."""
    extractor = EnsembleExtractor(
        ocr_ensemble=mock_ocr_ensemble,
        ner_ensemble=mock_ner_ensemble,
        layoutlm_engine=mock_layoutlm_engine,
        local_llm_extractor=mock_local_llm_extractor,
        use_layoutlm=True,
        use_local_llm=True,
    )

    result = await extractor.extract(sample_text_document)

    assert result.extraction_mode == ExtractionMode.ENSEMBLE
    assert "person_names" in result.entities
    assert "organization_names" in result.entities
    assert result.processing_time_ms > 0
    assert result.confidence_report is not None
    assert result.confidence_report.overall_confidence > 0


@pytest.mark.asyncio
async def test_pipeline_ner_only(sample_text_document, mock_ner_ensemble):
    """Test ensemble with only NER ensemble (no OCR, no LayoutLM, no LLM)."""
    extractor = EnsembleExtractor(
        ocr_ensemble=MagicMock(extract_text=MagicMock(return_value=("", 0.0))),
        ner_ensemble=mock_ner_ensemble,
        layoutlm_engine=None,
        local_llm_extractor=None,
        use_layoutlm=False,
        use_local_llm=False,
    )

    result = await extractor.extract(sample_text_document)

    assert result.extraction_mode == ExtractionMode.ENSEMBLE
    assert "person_names" in result.entities
    assert "organization_names" in result.entities
    assert "monetary_amounts" in result.entities
    # OCR should not be called for text documents
    mock_ner_ensemble.extract_entities.assert_called_once()


@pytest.mark.asyncio
async def test_ocr_runs_for_image_documents(
    sample_image_document, mock_ocr_ensemble, mock_ner_ensemble
):
    """Test that OCR ensemble runs when document is an image."""
    extractor = EnsembleExtractor(
        ocr_ensemble=mock_ocr_ensemble,
        ner_ensemble=mock_ner_ensemble,
        layoutlm_engine=None,
        local_llm_extractor=None,
        use_layoutlm=False,
        use_local_llm=False,
    )

    result = await extractor.extract(sample_image_document)

    mock_ocr_ensemble.extract_text.assert_called_once_with("/tmp/scan.png")
    assert result.raw_text == "John Smith works at Acme Corp. Invoice total: $5,000."


# --- Graceful degradation tests ---


@pytest.mark.asyncio
async def test_graceful_degradation_ocr_failure(sample_image_document):
    """Test that ensemble handles OCR ensemble failure gracefully."""
    mock_ocr = MagicMock()
    mock_ocr.extract_text.side_effect = RuntimeError("OCR engine crashed")

    mock_ner = MagicMock()
    mock_ner.extract_entities.return_value = {}

    extractor = EnsembleExtractor(
        ocr_ensemble=mock_ocr,
        ner_ensemble=mock_ner,
        layoutlm_engine=None,
        local_llm_extractor=None,
        use_layoutlm=False,
        use_local_llm=False,
    )

    result = await extractor.extract(sample_image_document)

    assert result.extraction_mode == ExtractionMode.ENSEMBLE
    assert "OCR ensemble produced no results" in result.warnings


@pytest.mark.asyncio
async def test_graceful_degradation_ner_failure(sample_text_document):
    """Test that ensemble handles NER ensemble failure gracefully."""
    mock_ner = MagicMock()
    mock_ner.extract_entities.side_effect = RuntimeError("NER engine crashed")

    extractor = EnsembleExtractor(
        ocr_ensemble=None,
        ner_ensemble=mock_ner,
        layoutlm_engine=None,
        local_llm_extractor=None,
        use_layoutlm=False,
        use_local_llm=False,
    )

    result = await extractor.extract(sample_text_document)

    assert result.extraction_mode == ExtractionMode.ENSEMBLE
    assert "NER ensemble produced no results" in result.warnings


@pytest.mark.asyncio
async def test_graceful_degradation_layoutlm_unavailable(
    sample_text_document, mock_ner_ensemble
):
    """Test that ensemble handles unavailable LayoutLM gracefully."""
    mock_layoutlm = MagicMock()
    mock_layoutlm.is_available.return_value = False

    extractor = EnsembleExtractor(
        ocr_ensemble=None,
        ner_ensemble=mock_ner_ensemble,
        layoutlm_engine=mock_layoutlm,
        local_llm_extractor=None,
        use_layoutlm=True,
        use_local_llm=False,
    )

    result = await extractor.extract(sample_text_document)

    # Should still produce results from NER ensemble
    assert result.extraction_mode == ExtractionMode.ENSEMBLE
    assert "person_names" in result.entities


@pytest.mark.asyncio
async def test_graceful_degradation_local_llm_failure(
    sample_text_document, mock_ner_ensemble
):
    """Test that ensemble handles local LLM failure gracefully."""
    mock_llm = AsyncMock()
    mock_llm.extract.side_effect = RuntimeError("LLM endpoint down")

    extractor = EnsembleExtractor(
        ocr_ensemble=None,
        ner_ensemble=mock_ner_ensemble,
        layoutlm_engine=None,
        local_llm_extractor=mock_llm,
        use_layoutlm=False,
        use_local_llm=True,
    )

    result = await extractor.extract(sample_text_document)

    # Should still produce results from NER ensemble
    assert result.extraction_mode == ExtractionMode.ENSEMBLE
    assert "person_names" in result.entities


@pytest.mark.asyncio
async def test_graceful_degradation_all_engines_fail(sample_text_document):
    """Test that ensemble handles all engines failing gracefully."""
    mock_ner = MagicMock()
    mock_ner.extract_entities.side_effect = RuntimeError("NER failed")

    mock_llm = AsyncMock()
    mock_llm.extract.side_effect = RuntimeError("LLM failed")

    mock_layoutlm = MagicMock()
    mock_layoutlm.is_available.return_value = False

    extractor = EnsembleExtractor(
        ocr_ensemble=None,
        ner_ensemble=mock_ner,
        layoutlm_engine=mock_layoutlm,
        local_llm_extractor=mock_llm,
        use_layoutlm=True,
        use_local_llm=True,
    )

    result = await extractor.extract(sample_text_document)

    assert result.extraction_mode == ExtractionMode.ENSEMBLE
    assert result.entities == {}
    assert result.confidence_report.overall_confidence == 0.0


# --- Confidence-weighted merging tests ---


@pytest.mark.asyncio
async def test_confidence_weighted_merging_picks_best_source(sample_text_document):
    """Test that merging picks values from the highest-weight source."""
    mock_ner = MagicMock()
    mock_ner.extract_entities.return_value = {
        "person_names": [("John Smith", 0.9)],
        "organization_names": [("Acme Corp", 0.7)],
    }

    mock_layoutlm = MagicMock()
    mock_layoutlm.is_available.return_value = True
    mock_layoutlm.extract_entities.return_value = {
        "organization_names": [("Acme Corporation", 0.95)],
        "dates": [("2024-01-15", 0.88)],
    }

    extractor = EnsembleExtractor(
        ocr_ensemble=None,
        ner_ensemble=mock_ner,
        layoutlm_engine=mock_layoutlm,
        local_llm_extractor=None,
        use_layoutlm=True,
        use_local_llm=False,
    )

    result = await extractor.extract(sample_text_document)

    # LayoutLM has weight 0.85 vs NER 0.8, so for shared fields, LayoutLM wins
    assert result.entities["organization_names"] == "Acme Corporation"
    # NER-only fields are preserved
    assert result.entities["person_names"] == "John Smith"
    # LayoutLM-only fields are included
    assert result.entities["dates"] == "2024-01-15"


@pytest.mark.asyncio
async def test_confidence_boosted_by_multi_source_agreement(sample_text_document):
    """Test that confidence is higher when multiple sources agree on a field."""
    mock_ner = MagicMock()
    mock_ner.extract_entities.return_value = {
        "person_names": [("John Smith", 0.85)],
        "shared_field": [("shared_value", 0.7)],
    }

    mock_layoutlm = MagicMock()
    mock_layoutlm.is_available.return_value = True
    mock_layoutlm.extract_entities.return_value = {
        "shared_field": [("shared_value", 0.8)],
    }

    extractor = EnsembleExtractor(
        ocr_ensemble=None,
        ner_ensemble=mock_ner,
        layoutlm_engine=mock_layoutlm,
        local_llm_extractor=None,
        use_layoutlm=True,
        use_local_llm=False,
    )

    result = await extractor.extract(sample_text_document)

    # shared_field was extracted by 2/2 sources, should have higher confidence
    shared_conf = result.confidence_report.scores["shared_field"].score
    person_conf = result.confidence_report.scores["person_names"].score
    # Shared field gets agreement boost
    assert shared_conf > person_conf


# --- Configuration tests ---


@pytest.mark.asyncio
async def test_configuration_controls_engines_no_layoutlm(
    sample_text_document, mock_ner_ensemble, mock_layoutlm_engine
):
    """Test that use_layoutlm=False prevents LayoutLM from running."""
    extractor = EnsembleExtractor(
        ocr_ensemble=None,
        ner_ensemble=mock_ner_ensemble,
        layoutlm_engine=mock_layoutlm_engine,
        local_llm_extractor=None,
        use_layoutlm=False,
        use_local_llm=False,
    )

    result = await extractor.extract(sample_text_document)

    # LayoutLM should not be called
    mock_layoutlm_engine.extract_entities.assert_not_called()
    # NER results should still be present
    assert "person_names" in result.entities


@pytest.mark.asyncio
async def test_configuration_controls_engines_no_local_llm(
    sample_text_document, mock_ner_ensemble, mock_local_llm_extractor
):
    """Test that use_local_llm=False prevents local LLM from running."""
    extractor = EnsembleExtractor(
        ocr_ensemble=None,
        ner_ensemble=mock_ner_ensemble,
        layoutlm_engine=None,
        local_llm_extractor=mock_local_llm_extractor,
        use_layoutlm=False,
        use_local_llm=False,
    )

    result = await extractor.extract(sample_text_document)

    # Local LLM should not be called
    mock_local_llm_extractor.extract.assert_not_called()
    assert "person_names" in result.entities


@pytest.mark.asyncio
async def test_extraction_mode_is_ensemble(
    sample_text_document, mock_ner_ensemble
):
    """Test that the result always has ExtractionMode.ENSEMBLE."""
    extractor = EnsembleExtractor(
        ocr_ensemble=None,
        ner_ensemble=mock_ner_ensemble,
        layoutlm_engine=None,
        local_llm_extractor=None,
        use_layoutlm=False,
        use_local_llm=False,
    )

    result = await extractor.extract(sample_text_document)

    assert result.extraction_mode == ExtractionMode.ENSEMBLE
    assert result.extraction_mode.value == "ensemble"


# --- EnsembleExtractor initialization tests ---


@patch("src.extractors.ensemble_extractor.get_settings")
def test_ensemble_extractor_initializes_with_defaults(mock_settings):
    """Test that EnsembleExtractor can initialize with default settings."""
    mock_settings.return_value = MagicMock(
        ensemble={"confidence_threshold": 0.6, "ocr_engines": ["tesseract"], "ner_models": ["spacy_sm"]}
    )

    extractor = EnsembleExtractor(
        ocr_ensemble=MagicMock(),
        ner_ensemble=MagicMock(),
    )

    assert extractor.mode == ExtractionMode.ENSEMBLE
    assert extractor._confidence_threshold == 0.6


@pytest.mark.asyncio
async def test_ner_single_value_not_wrapped_in_list(sample_text_document):
    """Test that single NER entity values are not wrapped in a list."""
    mock_ner = MagicMock()
    mock_ner.extract_entities.return_value = {
        "person_names": [("John Smith", 0.9)],
    }

    extractor = EnsembleExtractor(
        ocr_ensemble=None,
        ner_ensemble=mock_ner,
        layoutlm_engine=None,
        local_llm_extractor=None,
        use_layoutlm=False,
        use_local_llm=False,
    )

    result = await extractor.extract(sample_text_document)

    # Single value should not be a list
    assert result.entities["person_names"] == "John Smith"


@pytest.mark.asyncio
async def test_ner_multiple_values_as_list(sample_text_document):
    """Test that multiple NER entity values are returned as a list."""
    mock_ner = MagicMock()
    mock_ner.extract_entities.return_value = {
        "person_names": [("John Smith", 0.9), ("Jane Doe", 0.85)],
    }

    extractor = EnsembleExtractor(
        ocr_ensemble=None,
        ner_ensemble=mock_ner,
        layoutlm_engine=None,
        local_llm_extractor=None,
        use_layoutlm=False,
        use_local_llm=False,
    )

    result = await extractor.extract(sample_text_document)

    # Multiple values should be a list
    assert result.entities["person_names"] == ["John Smith", "Jane Doe"]


@pytest.mark.asyncio
async def test_empty_text_document_skips_ner(sample_text_document):
    """Test that NER is skipped for documents with whitespace-only text."""
    sample_text_document.raw_text = "   "
    sample_text_document.content = "   "

    mock_ner = MagicMock()
    mock_ner.extract_entities.return_value = {}

    extractor = EnsembleExtractor(
        ocr_ensemble=None,
        ner_ensemble=mock_ner,
        layoutlm_engine=None,
        local_llm_extractor=None,
        use_layoutlm=False,
        use_local_llm=False,
    )

    result = await extractor.extract(sample_text_document)

    mock_ner.extract_entities.assert_not_called()
    assert result.entities == {}


# --- Orchestrator integration test ---


@patch("src.extractors.ensemble_extractor.get_settings")
def test_ensemble_mode_enum_value(mock_settings):
    """Test that ExtractionMode.ENSEMBLE has the correct string value."""
    assert ExtractionMode.ENSEMBLE.value == "ensemble"
    assert ExtractionMode.ENSEMBLE == "ensemble"
