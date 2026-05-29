"""Tests for the local extractor using spaCy NER and regex patterns."""

from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.extractors.local_extractor import LocalExtractor
from src.models.document import Document, FileType


@pytest.fixture
def sample_contract_text():
    """Load sample contract text for testing."""
    sample_path = Path(__file__).parent / "sample_docs" / "sample_contract.txt"
    return sample_path.read_text(encoding="utf-8")


@pytest.fixture
def sample_document(sample_contract_text):
    """Create a Document model from sample contract text."""
    return Document(
        id=uuid4(),
        filename="sample_contract.txt",
        file_type=FileType.TEXT,
        content=sample_contract_text,
        raw_text=sample_contract_text,
        page_count=1,
    )


@pytest.fixture
def mock_spacy_doc():
    """Create a mock spaCy doc with entities."""

    class MockEntity:
        def __init__(self, text, label_):
            self.text = text
            self.label_ = label_

    class MockDoc:
        def __init__(self):
            self.ents = [
                MockEntity("TechVision Solutions Inc.", "ORG"),
                MockEntity("GlobalRetail Corp.", "ORG"),
                MockEntity("Sarah Johnson", "PERSON"),
                MockEntity("Michael Chen", "PERSON"),
                MockEntity("January 15, 2024", "DATE"),
                MockEntity("$75,000.00", "MONEY"),
                MockEntity("San Francisco", "GPE"),
                MockEntity("New York", "GPE"),
                MockEntity("Delaware", "GPE"),
            ]

    return MockDoc()


@pytest.fixture
def extractor_with_mocked_spacy(mock_spacy_doc):
    """Create a LocalExtractor with mocked spaCy model."""
    with patch("src.extractors.local_extractor.LocalExtractor.nlp", new_callable=lambda: property(lambda self: MagicMock(return_value=mock_spacy_doc))):
        extractor = LocalExtractor(spacy_model="en_core_web_sm")
        # Directly patch the nlp property to return a callable mock
        mock_nlp = MagicMock(return_value=mock_spacy_doc)
        extractor._nlp = mock_nlp
        return extractor


@pytest.mark.asyncio
async def test_extraction_from_plain_text(extractor_with_mocked_spacy, sample_document):
    """Test that local extractor processes plain text and returns entities."""
    result = await extractor_with_mocked_spacy.extract(sample_document)

    assert result is not None
    assert result.document_id == sample_document.id
    assert result.extraction_mode.value == "local"
    assert result.processing_time_ms >= 0
    assert isinstance(result.entities, dict)
    assert len(result.entities) > 0


@pytest.mark.asyncio
async def test_entity_recognition(extractor_with_mocked_spacy, sample_document):
    """Test that entities like dates, names, and amounts are recognized."""
    result = await extractor_with_mocked_spacy.extract(sample_document)

    entities = result.entities
    # Check that some entities were found (either from spaCy mock or regex)
    assert len(entities) > 0

    # The mock spaCy doc should produce person_names and organization_names
    assert "person_names" in entities or "organization_names" in entities or "dates" in entities


@pytest.mark.asyncio
async def test_confidence_scoring(extractor_with_mocked_spacy, sample_document):
    """Test that confidence scores are computed for extracted fields."""
    result = await extractor_with_mocked_spacy.extract(sample_document)

    confidence = result.confidence_report
    assert confidence.overall_confidence > 0
    assert confidence.overall_confidence <= 1.0
    assert len(confidence.scores) > 0

    for field_name, score in confidence.scores.items():
        assert 0 <= score.score <= 1.0
        assert score.method in ("local_ner", "local_regex", "local_ner+regex", "local_unknown")


@pytest.mark.asyncio
async def test_empty_text_handling(extractor_with_mocked_spacy):
    """Test handling of empty or minimal text documents raises ValueError."""
    empty_doc = Document(
        id=uuid4(),
        filename="empty.txt",
        file_type=FileType.TEXT,
        content="",
        raw_text="",
        page_count=1,
    )

    # The base extractor validates that documents have text content
    with pytest.raises(ValueError, match="no text content"):
        await extractor_with_mocked_spacy.extract(empty_doc)


@pytest.mark.asyncio
async def test_minimal_text_handling(extractor_with_mocked_spacy):
    """Test handling of minimal text that produces no entities."""
    minimal_doc = Document(
        id=uuid4(),
        filename="minimal.txt",
        file_type=FileType.TEXT,
        content="x",
        raw_text="x",
        page_count=1,
    )

    # Mock nlp to return empty doc (no entities found)
    mock_empty_doc = MagicMock()
    mock_empty_doc.ents = []
    extractor_with_mocked_spacy._nlp = MagicMock(return_value=mock_empty_doc)

    result = await extractor_with_mocked_spacy.extract(minimal_doc)

    assert result is not None
    assert result.entities == {} or len(result.entities) == 0
    assert result.confidence_report.overall_confidence == 0.0
