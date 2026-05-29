"""Tests for the FreeModelRegistry singleton."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.model_registry import (
    FreeModelRegistry,
    ModelStatus,
    ModelType,
    RegisteredModel,
)


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the singleton registry before each test."""
    FreeModelRegistry.reset_instance()
    yield
    FreeModelRegistry.reset_instance()


@pytest.fixture
def registry():
    """Create a fresh registry instance."""
    return FreeModelRegistry()


def test_singleton_pattern():
    """Test that FreeModelRegistry is a singleton."""
    registry1 = FreeModelRegistry()
    registry2 = FreeModelRegistry()
    assert registry1 is registry2


def test_register_model(registry):
    """Test registering a new model."""
    model = registry.register_model(
        name="llama3",
        model_type=ModelType.LLM,
        endpoint="http://localhost:11434/v1",
        config={"temperature": 0.0},
    )

    assert model.name == "llama3"
    assert model.model_type == ModelType.LLM
    assert model.status == ModelStatus.UNAVAILABLE
    assert model.endpoint == "http://localhost:11434/v1"
    assert model.config == {"temperature": 0.0}


def test_get_model(registry):
    """Test retrieving a registered model by name."""
    registry.register_model(name="tesseract", model_type=ModelType.OCR)

    model = registry.get_model("tesseract")
    assert model is not None
    assert model.name == "tesseract"
    assert model.model_type == ModelType.OCR


def test_get_model_not_found(registry):
    """Test retrieving a non-existent model returns None."""
    model = registry.get_model("nonexistent")
    assert model is None


def test_get_available_models_empty(registry):
    """Test getting available models when none are available."""
    registry.register_model(name="model1", model_type=ModelType.LLM)
    available = registry.get_available_models()
    assert available == []


def test_get_available_models_with_available(registry):
    """Test getting available models after setting status."""
    registry.register_model(name="model1", model_type=ModelType.LLM)
    registry.register_model(name="model2", model_type=ModelType.OCR)

    registry.set_model_status("model1", ModelStatus.AVAILABLE)

    available = registry.get_available_models()
    assert len(available) == 1
    assert available[0].name == "model1"


def test_get_available_models_filter_by_type(registry):
    """Test filtering available models by type."""
    registry.register_model(name="llama3", model_type=ModelType.LLM)
    registry.register_model(name="tesseract", model_type=ModelType.OCR)
    registry.register_model(name="spacy_sm", model_type=ModelType.NER)

    registry.set_model_status("llama3", ModelStatus.AVAILABLE)
    registry.set_model_status("tesseract", ModelStatus.AVAILABLE)
    registry.set_model_status("spacy_sm", ModelStatus.AVAILABLE)

    llm_models = registry.get_available_models(ModelType.LLM)
    assert len(llm_models) == 1
    assert llm_models[0].name == "llama3"

    ocr_models = registry.get_available_models(ModelType.OCR)
    assert len(ocr_models) == 1
    assert ocr_models[0].name == "tesseract"


def test_set_model_status(registry):
    """Test manually setting model status."""
    registry.register_model(name="model1", model_type=ModelType.LLM)

    registry.set_model_status("model1", ModelStatus.AVAILABLE)
    model = registry.get_model("model1")
    assert model.status == ModelStatus.AVAILABLE

    registry.set_model_status("model1", ModelStatus.UNAVAILABLE)
    model = registry.get_model("model1")
    assert model.status == ModelStatus.UNAVAILABLE


def test_set_model_status_nonexistent(registry):
    """Test setting status for non-existent model is handled gracefully."""
    # Should not raise, just log a warning
    registry.set_model_status("nonexistent", ModelStatus.AVAILABLE)


def test_list_all_models(registry):
    """Test listing all registered models."""
    registry.register_model(name="model1", model_type=ModelType.LLM)
    registry.register_model(name="model2", model_type=ModelType.OCR)
    registry.register_model(name="model3", model_type=ModelType.NER)

    all_models = registry.list_all_models()
    assert len(all_models) == 3
    names = {m.name for m in all_models}
    assert names == {"model1", "model2", "model3"}


def test_clear_registry(registry):
    """Test clearing all models from registry."""
    registry.register_model(name="model1", model_type=ModelType.LLM)
    registry.register_model(name="model2", model_type=ModelType.OCR)

    registry.clear()
    assert registry.list_all_models() == []


@pytest.mark.asyncio
async def test_health_check_llm_endpoint_success(registry):
    """Test health check for LLM endpoint that responds successfully."""
    registry.register_model(
        name="llama3",
        model_type=ModelType.LLM,
        endpoint="http://localhost:11434/v1",
    )

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        result = await registry.check_health("llama3")
        assert result is True

        model = registry.get_model("llama3")
        assert model.status == ModelStatus.AVAILABLE


@pytest.mark.asyncio
async def test_health_check_llm_endpoint_failure(registry):
    """Test health check for LLM endpoint that is down."""
    registry.register_model(
        name="llama3",
        model_type=ModelType.LLM,
        endpoint="http://localhost:11434/v1",
    )

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = mock_client

        result = await registry.check_health("llama3")
        assert result is False

        model = registry.get_model("llama3")
        assert model.status == ModelStatus.UNAVAILABLE


@pytest.mark.asyncio
async def test_health_check_unregistered_model(registry):
    """Test health check for a model that is not registered."""
    result = await registry.check_health("nonexistent")
    assert result is False


@pytest.mark.asyncio
async def test_health_check_local_ner_model(registry):
    """Test health check for a local NER model (checks importability)."""
    registry.register_model(name="spacy_ner", model_type=ModelType.NER)

    # spaCy should be importable in our test environment
    result = await registry.check_health("spacy_ner")
    assert result is True

    model = registry.get_model("spacy_ner")
    assert model.status == ModelStatus.AVAILABLE


def test_graceful_fallback_unavailable_models(registry):
    """Test graceful handling when models are unavailable."""
    registry.register_model(name="model1", model_type=ModelType.LLM)
    registry.register_model(name="model2", model_type=ModelType.LLM)

    # None are available
    available = registry.get_available_models(ModelType.LLM)
    assert available == []

    # Make one available
    registry.set_model_status("model1", ModelStatus.AVAILABLE)
    available = registry.get_available_models(ModelType.LLM)
    assert len(available) == 1
    assert available[0].name == "model1"
