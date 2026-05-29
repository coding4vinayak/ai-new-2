"""Central model registry for tracking available free models and endpoints."""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Types of models the registry can track."""

    OCR = "ocr"
    NER = "ner"
    LLM = "llm"
    LAYOUT = "layout"


class ModelStatus(str, Enum):
    """Status of a registered model."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    CHECKING = "checking"


class RegisteredModel(BaseModel):
    """A model tracked by the registry."""

    name: str = Field(..., description="Unique model identifier")
    model_type: ModelType = Field(..., description="Type of model")
    status: ModelStatus = Field(
        default=ModelStatus.UNAVAILABLE, description="Current availability status"
    )
    endpoint: Optional[str] = Field(None, description="Endpoint URL if applicable")
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Additional model configuration"
    )


class FreeModelRegistry:
    """Singleton registry for tracking available models.

    Tracks OCR engines, NER models, LLM endpoints, and layout models.
    Provides health checking and graceful fallback when models are unavailable.

    NOTE: This registry is currently informational and used for health-check UI
    purposes. The EnsembleExtractor builds engines directly based on configuration
    rather than consulting the registry for engine selection. Integration into the
    engine selection pipeline is planned for a future iteration.
    """

    _instance: Optional["FreeModelRegistry"] = None
    _initialized: bool = False

    def __new__(cls) -> "FreeModelRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._models: Dict[str, RegisteredModel] = {}
        self._initialized = True

    def register_model(
        self,
        name: str,
        model_type: ModelType,
        endpoint: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> RegisteredModel:
        """Register a model with the registry.

        Args:
            name: Unique model identifier.
            model_type: Type of model (ocr, ner, llm, layout).
            endpoint: Optional endpoint URL for remote models.
            config: Optional additional configuration.

        Returns:
            The registered model instance.
        """
        model = RegisteredModel(
            name=name,
            model_type=model_type,
            status=ModelStatus.UNAVAILABLE,
            endpoint=endpoint,
            config=config or {},
        )
        self._models[name] = model
        logger.info(f"Registered model: {name} (type={model_type.value})")
        return model

    def get_model(self, name: str) -> Optional[RegisteredModel]:
        """Get a registered model by name.

        Args:
            name: The model name to look up.

        Returns:
            The registered model or None if not found.
        """
        model = self._models.get(name)
        if model is None:
            logger.warning(f"Model not found: {name}")
        return model

    def get_available_models(
        self, model_type: Optional[ModelType] = None
    ) -> List[RegisteredModel]:
        """Get all available models, optionally filtered by type.

        Args:
            model_type: Optional type filter.

        Returns:
            List of available registered models.
        """
        models = list(self._models.values())
        if model_type is not None:
            models = [m for m in models if m.model_type == model_type]
        return [m for m in models if m.status == ModelStatus.AVAILABLE]

    async def check_health(self, model_name: str) -> bool:
        """Check health of a specific model.

        For LLM models with endpoints, pings the /v1/models endpoint.
        For other models, checks if the model is importable/available.

        Args:
            model_name: Name of the model to check.

        Returns:
            True if the model is healthy/available.
        """
        model = self._models.get(model_name)
        if model is None:
            logger.warning(f"Cannot check health for unregistered model: {model_name}")
            return False

        model.status = ModelStatus.CHECKING

        if model.model_type == ModelType.LLM and model.endpoint:
            healthy = await self._check_llm_endpoint(model.endpoint)
        else:
            healthy = self._check_local_model(model)

        model.status = ModelStatus.AVAILABLE if healthy else ModelStatus.UNAVAILABLE
        logger.info(
            f"Health check for {model_name}: "
            f"{'available' if healthy else 'unavailable'}"
        )
        return healthy

    async def _check_llm_endpoint(self, endpoint: str) -> bool:
        """Check if an LLM endpoint is responding.

        Args:
            endpoint: Base URL of the endpoint.

        Returns:
            True if the endpoint responds to /v1/models or /models.
        """
        try:
            import httpx

            # Try /v1/models first, then /models
            urls_to_try = []
            if endpoint.endswith("/v1"):
                urls_to_try.append(f"{endpoint}/models")
            else:
                urls_to_try.append(f"{endpoint.rstrip('/')}/v1/models")
                urls_to_try.append(f"{endpoint.rstrip('/')}/models")

            async with httpx.AsyncClient(timeout=5.0) as client:
                for url in urls_to_try:
                    try:
                        response = await client.get(url)
                        if response.status_code == 200:
                            return True
                    except Exception:
                        continue
            return False
        except ImportError:
            logger.warning("httpx not available for health checks")
            return False
        except Exception as e:
            logger.warning(f"Health check failed for endpoint {endpoint}: {e}")
            return False

    def _check_local_model(self, model: RegisteredModel) -> bool:
        """Check if a local model is available.

        Args:
            model: The registered model to check.

        Returns:
            True if the model appears available locally.
        """
        # For local models, check if the relevant package is importable
        if model.model_type == ModelType.NER:
            try:
                import spacy
                return True
            except ImportError:
                return False
        elif model.model_type == ModelType.OCR:
            try:
                import pytesseract
                return True
            except ImportError:
                return False
        # Default: assume available if registered
        return True

    def set_model_status(self, name: str, status: ModelStatus) -> None:
        """Manually set a model's status.

        Args:
            name: Model name.
            status: New status to set.
        """
        model = self._models.get(name)
        if model:
            model.status = status
        else:
            logger.warning(f"Cannot set status for unregistered model: {name}")

    def list_all_models(self) -> List[RegisteredModel]:
        """List all registered models regardless of status.

        Returns:
            List of all registered models.
        """
        return list(self._models.values())

    def clear(self) -> None:
        """Clear all registered models. Useful for testing."""
        self._models.clear()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance. Useful for testing."""
        cls._instance = None
        cls._initialized = False
