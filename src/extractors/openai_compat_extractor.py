"""OpenAI-compatible extractor for local LLM endpoints (Ollama, LM Studio, etc.)."""

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import yaml

from src.extractors.base import BaseExtractor, ExtractionMode
from src.models.confidence import ConfidenceReport, ConfidenceScore
from src.models.document import Document
from src.models.extraction_result import ExtractionResult
from src.utils.config import get_settings

logger = logging.getLogger(__name__)


# Default endpoints for known providers
PROVIDER_DEFAULTS = {
    "ollama": "http://localhost:11434/v1",
    "lmstudio": "http://localhost:1234/v1",
    "localai": "http://localhost:8080/v1",
    "vllm": "http://localhost:8000/v1",
    "text-generation-webui": "http://localhost:5000/v1",
    "custom": None,
}

# Common model names
DEFAULT_MODELS = {
    "ollama": "llama3",
    "lmstudio": "local-model",
    "localai": "gpt-3.5-turbo",
    "vllm": "mistral",
    "text-generation-webui": "model",
    "custom": "default",
}


def _load_entity_config() -> Dict[str, Any]:
    """Load entity configuration from YAML."""
    from pathlib import Path

    config_path = Path(__file__).parent.parent.parent / "config" / "entities.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


class OpenAICompatExtractor(BaseExtractor):
    """Extractor that works with any OpenAI-compatible API endpoint.

    Supports Ollama, LM Studio, LocalAI, vLLM, text-generation-webui,
    and any other service exposing an OpenAI-compatible chat completions API.
    """

    MAX_TOKENS = 4000
    CHARS_PER_TOKEN = 4

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        timeout_seconds: int = 120,
        max_retries: int = 3,
    ) -> None:
        """Initialize the OpenAI-compatible extractor.

        Args:
            base_url: Base URL of the OpenAI-compatible endpoint.
            model_name: Name of the model to use.
            api_key: Optional API key (many local endpoints don't need one).
            provider: Provider name (ollama, lmstudio, localai, vllm, custom).
            timeout_seconds: Request timeout in seconds.
            max_retries: Number of retry attempts on failure.
        """
        super().__init__(mode=ExtractionMode.LOCAL_LLM)
        settings = get_settings()

        # Resolve provider
        self._provider = provider or self._resolve_provider(settings)

        # Resolve base_url
        self._base_url = base_url or self._resolve_base_url(settings)

        # Resolve model name
        self._model_name = model_name or self._resolve_model_name(settings)

        # Resolve API key (many local endpoints use "not-needed" or empty)
        self._api_key = api_key or self._resolve_api_key(settings)

        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._entity_config = _load_entity_config()

    def _resolve_provider(self, settings) -> str:
        """Resolve provider from settings."""
        local_llm = getattr(settings, "local_llm", None)
        if local_llm and isinstance(local_llm, dict):
            return local_llm.get("provider", "ollama")
        return "ollama"

    def _resolve_base_url(self, settings) -> str:
        """Resolve base URL from settings or provider defaults."""
        local_llm = getattr(settings, "local_llm", None)
        if local_llm and isinstance(local_llm, dict):
            url = local_llm.get("base_url")
            if url:
                return url
        return PROVIDER_DEFAULTS.get(self._provider) or "http://localhost:11434/v1"

    def _resolve_model_name(self, settings) -> str:
        """Resolve model name from settings or provider defaults."""
        local_llm = getattr(settings, "local_llm", None)
        if local_llm and isinstance(local_llm, dict):
            model = local_llm.get("model_name")
            if model:
                return model
        return DEFAULT_MODELS.get(self._provider, "llama3")

    def _resolve_api_key(self, settings) -> str:
        """Resolve API key from settings."""
        local_llm = getattr(settings, "local_llm", None)
        if local_llm and isinstance(local_llm, dict):
            key = local_llm.get("api_key")
            if key:
                return key
        return "not-needed"

    async def extract(self, document: Document) -> ExtractionResult:
        """Extract entities from a document using a local LLM endpoint.

        Args:
            document: The document to extract entities from.

        Returns:
            ExtractionResult with extracted entities and confidence scores.
        """
        self.validate_document(document)

        start_time = time.time()
        text = document.raw_text or document.content or ""

        # Build extraction prompt
        prompt = self._build_prompt(document, self._entity_config)

        # Call the OpenAI-compatible endpoint
        raw_response = None
        warnings: List[str] = []

        try:
            raw_response = await self._call_endpoint(prompt, text)
        except Exception as e:
            logger.warning(f"Local LLM extraction failed: {e}")
            warnings.append(f"Local LLM extraction failed: {str(e)}")
            raw_response = "{}"

        # Parse response
        entities = self._parse_response(raw_response)

        # Build confidence report
        confidence_report = self._build_confidence_report(entities)

        processing_time = (time.time() - start_time) * 1000

        return ExtractionResult(
            document_id=document.id,
            extraction_mode=ExtractionMode.LOCAL_LLM,
            entities=entities,
            confidence_report=confidence_report,
            raw_text=text,
            processing_time_ms=processing_time,
            extracted_at=datetime.utcnow(),
            extractor_version=self._get_version(),
            warnings=warnings,
        )

    async def get_confidence(self, result: ExtractionResult) -> ConfidenceReport:
        """Return the confidence report from an extraction result.

        Args:
            result: The extraction result to evaluate.

        Returns:
            ConfidenceReport with per-field and overall confidence scores.
        """
        return result.confidence_report

    async def health_check(self) -> bool:
        """Check if the configured endpoint is responding.

        Pings the /models endpoint to verify connectivity.

        Returns:
            True if the endpoint is healthy.
        """
        try:
            import httpx

            models_url = f"{self._base_url.rstrip('/')}/models"
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(models_url)
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed for {self._base_url}: {e}")
            return False

    def _build_prompt(
        self, document: Document, entity_config: Dict[str, Any]
    ) -> str:
        """Build the extraction prompt with entity definitions.

        Args:
            document: The document being processed.
            entity_config: Entity definitions from config/entities.yaml.

        Returns:
            Formatted prompt string for the LLM.
        """
        doc_type = document.metadata.get("document_type", "general")

        entities_section = entity_config.get("entities", {})
        target_entities = entities_section.get(
            doc_type, entities_section.get("general", [])
        )

        entity_descriptions = []
        for entity_def in target_entities:
            name = entity_def.get("name", "")
            etype = entity_def.get("type", "string")
            desc = entity_def.get("description", "")
            entity_descriptions.append(f"- {name} ({etype}): {desc}")

        entities_text = (
            "\n".join(entity_descriptions)
            if entity_descriptions
            else "- Extract all relevant entities"
        )

        prompt = f"""You are a document entity extraction system. Extract structured data from the following document.

Document type: {doc_type}
Document filename: {document.filename}

Extract the following entities:
{entities_text}

Return your response as a valid JSON object where keys are entity names and values are the extracted data.
For list types, use JSON arrays. For dates, use ISO format (YYYY-MM-DD) when possible.
If an entity is not found in the document, omit it from the response.
Only return the JSON object, no additional text or explanation."""

        return prompt

    async def _call_endpoint(self, prompt: str, text: str) -> str:
        """Call the OpenAI-compatible endpoint for extraction.

        Args:
            prompt: System prompt with extraction instructions.
            text: Document text to extract from.

        Returns:
            Raw response string.

        Raises:
            Exception: If the API call fails after all retries.
        """
        import openai

        client = openai.AsyncOpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout_seconds,
        )

        # Truncate text if too long
        max_chars = self.MAX_TOKENS * self.CHARS_PER_TOKEN
        if len(text) > max_chars:
            text = text[:max_chars]

        last_error = None
        for attempt in range(self._max_retries):
            try:
                response = await client.chat.completions.create(
                    model=self._model_name,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": text},
                    ],
                    temperature=0.0,
                )
                content = response.choices[0].message.content or "{}"
                return content
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1}/{self._max_retries} failed: {e}"
                )
                if attempt < self._max_retries - 1:
                    continue

        raise last_error  # type: ignore[misc]

    def _parse_response(self, raw_response: str) -> Dict[str, Any]:
        """Parse structured JSON from LLM response.

        Handles markdown code block wrapping.

        Args:
            raw_response: Raw text response from the LLM.

        Returns:
            Parsed entity dictionary.
        """
        text = raw_response.strip()

        # Remove markdown code block wrapping if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            logger.warning("Failed to parse local LLM response as JSON")

        return {}

    def _build_confidence_report(
        self, entities: Dict[str, Any]
    ) -> ConfidenceReport:
        """Build confidence report for local LLM extraction results.

        Local LLM extractions generally have moderate confidence.

        Args:
            entities: Extracted entities.

        Returns:
            ConfidenceReport with scores for each field.
        """
        settings = get_settings()
        threshold = settings.confidence_threshold

        scores: Dict[str, ConfidenceScore] = {}

        for field_name, value in entities.items():
            # Local LLMs have moderate confidence
            score = 0.75 if value else 0.4
            scores[field_name] = ConfidenceScore(
                score=score,
                method=f"local_llm_{self._provider}",
                explanation=f"Extracted via local LLM ({self._model_name} on {self._provider})",
                field_name=field_name,
            )

        overall = (
            sum(s.score for s in scores.values()) / len(scores) if scores else 0.0
        )

        report = ConfidenceReport(
            scores=scores,
            overall_confidence=overall,
            threshold=threshold,
        )
        report.compute_low_confidence_fields()

        return report
