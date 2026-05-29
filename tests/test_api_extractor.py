"""Tests for the API extractor using mocked LLM API calls."""

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.extractors.api_extractor import APIExtractor
from src.models.document import Document, FileType


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document(
        id=uuid4(),
        filename="contract.txt",
        file_type=FileType.TEXT,
        content="This is a contract between Acme Corp and Beta LLC effective January 1, 2024.",
        raw_text="This is a contract between Acme Corp and Beta LLC effective January 1, 2024.",
        page_count=1,
        metadata={"document_type": "contract"},
    )


@pytest.fixture
def extractor():
    """Create an API extractor with test API keys."""
    return APIExtractor(
        openai_api_key="test-key-openai",
        anthropic_api_key="test-key-anthropic",
    )


def test_prompt_construction(extractor, sample_document):
    """Test that extraction prompt is properly constructed with entity config."""
    prompt = extractor._build_prompt(sample_document, extractor._entity_config)

    assert "document entity extraction" in prompt.lower() or "extract" in prompt.lower()
    assert "contract" in prompt.lower()
    assert "JSON" in prompt


def test_response_parsing_valid_json(extractor):
    """Test parsing a valid JSON response from LLM."""
    raw_response = json.dumps({
        "party_names": ["Acme Corp", "Beta LLC"],
        "effective_date": "2024-01-01",
        "total_amount": "$100,000",
    })

    entities = extractor._parse_response(raw_response)

    assert entities["party_names"] == ["Acme Corp", "Beta LLC"]
    assert entities["effective_date"] == "2024-01-01"
    assert entities["total_amount"] == "$100,000"


def test_response_parsing_markdown_wrapped(extractor):
    """Test parsing JSON wrapped in markdown code blocks."""
    raw_response = '```json\n{"vendor": "Acme Corp", "amount": "$500"}\n```'

    entities = extractor._parse_response(raw_response)

    assert entities["vendor"] == "Acme Corp"
    assert entities["amount"] == "$500"


def test_response_parsing_invalid_json(extractor):
    """Test handling of invalid JSON response."""
    raw_response = "This is not valid JSON at all."
    entities = extractor._parse_response(raw_response)
    assert entities == {}


def test_chunking_short_text(extractor):
    """Test that short text is not chunked."""
    short_text = "Short document text."
    chunks = extractor._chunk_text(short_text, max_tokens=6000)
    assert len(chunks) == 1
    assert chunks[0] == short_text


def test_chunking_long_text(extractor):
    """Test that long text is properly chunked."""
    # Create text longer than the token limit
    long_text = "A" * 30000  # ~7500 tokens at 4 chars/token
    chunks = extractor._chunk_text(long_text, max_tokens=6000)
    assert len(chunks) > 1
    # Verify all text is covered
    combined = "".join(chunks)
    assert len(combined) == len(long_text)


@pytest.mark.asyncio
async def test_extraction_with_mocked_openai(extractor, sample_document):
    """Test full extraction with mocked OpenAI API call."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({
        "party_names": ["Acme Corp", "Beta LLC"],
        "effective_date": "2024-01-01",
    })

    with patch("openai.AsyncOpenAI") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        result = await extractor.extract(sample_document)

        assert result.extraction_mode.value == "api"
        assert "party_names" in result.entities
        assert result.entities["party_names"] == ["Acme Corp", "Beta LLC"]
        assert result.confidence_report.overall_confidence > 0


@pytest.mark.asyncio
async def test_error_handling_api_failure(sample_document):
    """Test error handling when both API providers fail."""
    extractor = APIExtractor(
        openai_api_key="",
        anthropic_api_key="",
    )

    result = await extractor.extract(sample_document)

    # Should still return a result, but with empty entities and warnings
    assert result is not None
    assert result.extraction_mode.value == "api"
    assert len(result.warnings) > 0
