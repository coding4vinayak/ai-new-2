"""Application configuration management using pydantic-settings."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


def _load_yaml_config(filename: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    config_dir = Path(__file__).parent.parent.parent / "config"
    config_path = config_dir / filename
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


class ExtractionSettings(BaseSettings):
    """Extraction-related settings."""

    modes: List[str] = ["local", "api", "hybrid", "local_llm"]
    default_mode: str = "hybrid"
    hybrid_escalation_threshold: float = 0.7
    minimum_acceptable_threshold: float = 0.5
    high_confidence_threshold: float = 0.9


class BatchSettings(BaseSettings):
    """Batch processing settings."""

    max_concurrent: int = 5
    max_batch_size: int = 100
    timeout_per_document_seconds: int = 120
    retry_attempts: int = 3


class APIKeysSettings(BaseSettings):
    """API key settings loaded from environment variables."""

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    class Config:
        populate_by_name = True


class LocalLLMSettings(BaseSettings):
    """Local LLM endpoint settings for OpenAI-compatible providers."""

    base_url: str = Field(
        default="http://localhost:11434/v1", alias="LOCAL_LLM_BASE_URL"
    )
    model_name: str = Field(default="llama3", alias="LOCAL_LLM_MODEL")
    api_key: str = Field(default="not-needed", alias="LOCAL_LLM_API_KEY")
    provider: str = "ollama"
    timeout_seconds: int = 120
    max_retries: int = 3

    class Config:
        populate_by_name = True


class FreeModelsSettings(BaseSettings):
    """Settings for free/open-source model configurations."""

    ocr_engines: List[Dict[str, Any]] = Field(default_factory=list)
    ner_models: List[Dict[str, Any]] = Field(default_factory=list)
    layout_models: List[Dict[str, Any]] = Field(default_factory=list)


class EnsembleSettings(BaseSettings):
    """Settings for ensemble/multi-model extraction strategies."""

    ocr_engines: List[str] = Field(default_factory=lambda: ["tesseract"])
    ner_models: List[str] = Field(default_factory=lambda: ["spacy_sm"])
    voting_strategy: str = "confidence_weighted"
    confidence_threshold: float = 0.6


class WebhookSettings(BaseSettings):
    """Webhook configuration settings."""

    webhook_secret: str = Field(default="", alias="WEBHOOK_SECRET")

    class Config:
        populate_by_name = True


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""

    database_url: str = Field(
        default="sqlite:///data/audit.db", alias="DATABASE_URL"
    )

    class Config:
        populate_by_name = True


class Settings(BaseSettings):
    """Main application settings combining all configuration sources."""

    # Environment variable settings
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    tesseract_path: str = Field(default="/usr/bin/tesseract", alias="TESSERACT_PATH")
    spacy_model: str = Field(default="en_core_web_sm", alias="SPACY_MODEL")
    database_url: str = Field(default="sqlite:///data/audit.db", alias="DATABASE_URL")
    webhook_secret: str = Field(default="", alias="WEBHOOK_SECRET")

    # Local LLM settings from environment
    local_llm_base_url: str = Field(
        default="http://localhost:11434/v1", alias="LOCAL_LLM_BASE_URL"
    )
    local_llm_model: str = Field(default="llama3", alias="LOCAL_LLM_MODEL")
    local_llm_api_key: str = Field(default="not-needed", alias="LOCAL_LLM_API_KEY")

    # Extraction settings (from YAML)
    extraction: Optional[Dict[str, Any]] = None
    batch_processing: Optional[Dict[str, Any]] = None
    api: Optional[Dict[str, Any]] = None
    local_llm: Optional[Dict[str, Any]] = None
    free_models: Optional[Dict[str, Any]] = None
    ensemble: Optional[Dict[str, Any]] = None
    supported_file_types: List[str] = [
        "pdf", "png", "jpg", "jpeg", "tiff", "docx", "txt"
    ]
    logging_config: Optional[Dict[str, Any]] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        populate_by_name = True
        extra = "ignore"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._load_yaml_settings()

    def _load_yaml_settings(self) -> None:
        """Load and merge YAML configuration."""
        yaml_config = _load_yaml_config("settings.yaml")
        if yaml_config:
            if self.extraction is None:
                self.extraction = yaml_config.get("extraction")
            if self.batch_processing is None:
                self.batch_processing = yaml_config.get("batch_processing")
            if self.api is None:
                self.api = yaml_config.get("api")
            if self.local_llm is None:
                self.local_llm = yaml_config.get("local_llm")
            if self.free_models is None:
                self.free_models = yaml_config.get("free_models")
            if self.ensemble is None:
                self.ensemble = yaml_config.get("ensemble")
            if "supported_file_types" in yaml_config:
                self.supported_file_types = yaml_config["supported_file_types"]
            if self.logging_config is None:
                self.logging_config = yaml_config.get("logging")

    @property
    def confidence_threshold(self) -> float:
        """Get the hybrid escalation confidence threshold."""
        if self.extraction and "confidence_thresholds" in self.extraction:
            return self.extraction["confidence_thresholds"].get(
                "hybrid_escalation", 0.7
            )
        return 0.7

    @property
    def max_concurrent(self) -> int:
        """Get maximum concurrent batch processing limit."""
        if self.batch_processing:
            return self.batch_processing.get("max_concurrent", 5)
        return 5

    @property
    def max_batch_size(self) -> int:
        """Get maximum batch size."""
        if self.batch_processing:
            return self.batch_processing.get("max_batch_size", 100)
        return 100

    def get_local_llm_settings(self) -> LocalLLMSettings:
        """Get local LLM settings from YAML config and env vars.

        Returns:
            LocalLLMSettings instance with merged configuration.
        """
        local_llm_config = self.local_llm or {}
        return LocalLLMSettings(
            base_url=self.local_llm_base_url or local_llm_config.get(
                "base_url", "http://localhost:11434/v1"
            ),
            model_name=self.local_llm_model or local_llm_config.get(
                "model_name", "llama3"
            ),
            api_key=self.local_llm_api_key or local_llm_config.get(
                "api_key", "not-needed"
            ),
            provider=local_llm_config.get("provider", "ollama"),
            timeout_seconds=local_llm_config.get("timeout_seconds", 120),
            max_retries=local_llm_config.get("max_retries", 3),
        )

    def get_ensemble_settings(self) -> EnsembleSettings:
        """Get ensemble settings from YAML config.

        Returns:
            EnsembleSettings instance.
        """
        ensemble_config = self.ensemble or {}
        return EnsembleSettings(
            ocr_engines=ensemble_config.get("ocr_engines", ["tesseract"]),
            ner_models=ensemble_config.get("ner_models", ["spacy_sm"]),
            voting_strategy=ensemble_config.get(
                "voting_strategy", "confidence_weighted"
            ),
            confidence_threshold=ensemble_config.get("confidence_threshold", 0.6),
        )


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings singleton."""
    return Settings()
