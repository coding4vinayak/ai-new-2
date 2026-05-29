"""Tests for the OpenAI-compatible local LLM extractor."""

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.extractors.openai_compat_extractor import (
    DEFAULT_MODELS,
    PROVIDER_DEFAULTS,
    OpenAICompatExtractor,
)
from src.models.document import Document, FileType
from src.models.extraction_result import ExtractionMode


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document(
        id=uuid4(),
        filename="invoice.txt",
        file_type=FileType.TEXT,
        content="Invoice #12345 from Acme Corp. Total: $1,500.00. Due: 2024-03-15.",
        raw_text="Invoice #12345 from Acme Corp. Total: $1,500.00. Due: 2024-03-15.",
        page_count=1,
        metadata={"document_type": "invoice"},
    )


@pytest.fixture
def extractor():
    """Create an OpenAI-compatible extractor with test settings."""
    return OpenAICompatExtractor(
        base_url="http://localhost:11434/v1",
        model_name="llama3",
        api_key="not-needed",
        provider="ollama",
    )


@pytest.fixture
def lmstudio_extractor():
    """Create an extractor configured for LM Studio."""
    return OpenAICompatExtractor(
        base_url="http://localhost:1234/v1",
        model_name="local-model",
        api_key="not-needed",
        provider="lmstudio",
    )


def test_extractor_initialization_ollama(extractor):
    """Test extractor initializes correctly for Ollama provider."""
    assert extractor._base_url == "http://localhost:11434/v1"
    assert extractor._model_name == "llama3"
    assert extractor._api_key == "not-needed"
    assert extractor._provider == "ollama"
    assert extractor.mode == ExtractionMode.LOCAL_LLM


def test_extractor_initialization_lmstudio(lmstudio_extractor):
    """Test extractor initializes correctly for LM Studio provider."""
    assert lmstudio_extractor._base_url == "http://localhost:1234/v1"
    assert lmstudio_extractor._model_name == "local-model"
    assert lmstudio_extractor._provider == "lmstudio"


def test_provider_defaults():
    """Test that provider defaults are correctly defined."""
    assert PROVIDER_DEFAULTS["ollama"] == "http://localhost:11434/v1"
    assert PROVIDER_DEFAULTS["lmstudio"] == "http://localhost:1234/v1"
    assert PROVIDER_DEFAULTS["localai"] == "http://localhost:8080/v1"
    assert PROVIDER_DEFAULTS["vllm"] == "http://localhost:8000/v1"


def test_default_models():
    """Test that default model names are defined for each provider."""
    assert DEFAULT_MODELS["ollama"] == "llama3"
    assert DEFAULT_MODELS["lmstudio"] == "local-model"
    assert DEFAULT_MODELS["localai"] == "gpt-3.5-turbo"
    assert DEFAULT_MODELS["vllm"] == "mistral"


def test_prompt_construction(extractor, sample_document):
    """Test that extraction prompt is properly constructed."""
    prompt = extractor._build_prompt(sample_document, extractor._entity_config)

    assert "extract" in prompt.lower()
    assert "invoice" in prompt.lower()
    assert "JSON" in prompt


def test_response_parsing_valid_json(extractor):
    """Test parsing a valid JSON response."""
    raw_response = json.dumps({
        "vendor": "Acme Corp",
        "invoice_number": "12345",
        "total": "$1,500.00",
    })

    entities = extractor._parse_response(raw_response)

    assert entities["vendor"] == "Acme Corp"
    assert entities["invoice_number"] == "12345"
    assert entities["total"] == "$1,500.00"


def test_response_parsing_markdown_wrapped(extractor):
    """Test parsing JSON wrapped in markdown code blocks."""
    raw_response = '```json\n{"vendor": "Acme Corp", "amount": "$500"}\n```'

    entities = extractor._parse_response(raw_response)

    assert entities["vendor"] == "Acme Corp"
    assert entities["amount"] == "$500"


def test_response_parsing_invalid_json(extractor):
    """Test handling of invalid JSON response."""
    raw_response = "This is not valid JSON."
    entities = extractor._parse_response(raw_response)
    assert entities == {}


def test_confidence_report_building(extractor):
    """Test confidence report is built correctly for local LLM results."""
    entities = {"vendor": "Acme Corp", "total": "$1,500.00", "empty_field": ""}

    report = extractor._build_confidence_report(entities)

    assert report.overall_confidence > 0
    assert "vendor" in report.scores
    assert report.scores["vendor"].score == 0.75
    assert report.scores["vendor"].method == "local_llm_ollama"
    # Empty values get lower confidence
    assert report.scores["empty_field"].score == 0.4


@pytest.mark.asyncio
async def test_extraction_with_mocked_openai(extractor, sample_document):
    """Test full extraction with mocked OpenAI-compatible API call."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({
        "vendor": "Acme Corp",
        "invoice_number": "12345",
        "total": "$1,500.00",
        "due_date": "2024-03-15",
    })

    with patch("openai.AsyncOpenAI") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        result = await extractor.extract(sample_document)

        assert result.extraction_mode == ExtractionMode.LOCAL_LLM
        assert result.entities["vendor"] == "Acme Corp"
        assert result.entities["invoice_number"] == "12345"
        assert result.confidence_report.overall_confidence > 0
        assert len(result.warnings) == 0

        # Verify client was created with correct params
        mock_client_class.assert_called_once_with(
            base_url="http://localhost:11434/v1",
            api_key="not-needed",
            timeout=120,
        )


@pytest.mark.asyncio
async def test_extraction_endpoint_failure(extractor, sample_document):
    """Test graceful handling when the endpoint is down."""
    with patch("openai.AsyncOpenAI") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("Connection refused")
        )
        mock_client_class.return_value = mock_client

        result = await extractor.extract(sample_document)

        # Should return result with empty entities and warnings
        assert result is not None
        assert result.extraction_mode == ExtractionMode.LOCAL_LLM
        assert result.entities == {}
        assert len(result.warnings) > 0
        assert "Connection refused" in result.warnings[0]


@pytest.mark.asyncio
async def test_health_check_passes():
    """Test health check when endpoint is available."""
    extractor = OpenAICompatExtractor(
        base_url="http://localhost:11434/v1",
        model_name="llama3",
        provider="ollama",
    )

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        result = await extractor.health_check()
        assert result is True


@pytest.mark.asyncio
async def test_health_check_fails():
    """Test health check when endpoint is not available."""
    extractor = OpenAICompatExtractor(
        base_url="http://localhost:11434/v1",
        model_name="llama3",
        provider="ollama",
    )

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        result = await extractor.health_check()
        assert result is False


@pytest.mark.asyncio
async def test_extraction_retries_on_failure(extractor, sample_document):
    """Test that extraction retries on transient failures."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps({"vendor": "Acme"})

    call_count = 0

    async def mock_create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Temporary error")
        return mock_response

    with patch("openai.AsyncOpenAI") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = mock_create
        mock_client_class.return_value = mock_client

        result = await extractor.extract(sample_document)

        assert result.entities["vendor"] == "Acme"
        assert call_count == 3  # Failed twice, succeeded on third
