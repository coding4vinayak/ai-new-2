"""API extractor using OpenAI GPT-4 and Anthropic Claude for entity extraction."""

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


def _load_entity_config() -> Dict[str, Any]:
    """Load entity configuration from YAML."""
    from pathlib import Path

    config_path = Path(__file__).parent.parent.parent / "config" / "entities.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


class APIExtractor(BaseExtractor):
    """Extractor using LLM APIs (OpenAI GPT-4 / Anthropic Claude) for extraction.

    Provides high-quality entity extraction by sending document text to
    large language models with structured extraction prompts. Includes
    retry logic, token limit handling, and provider fallback.
    """

    # Approximate token limits for chunking (conservative estimates)
    MAX_TOKENS_OPENAI = 6000
    MAX_TOKENS_ANTHROPIC = 6000
    CHARS_PER_TOKEN = 4  # Approximate characters per token

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
    ) -> None:
        """Initialize the API extractor.

        Args:
            openai_api_key: OpenAI API key. Falls back to config/env.
            anthropic_api_key: Anthropic API key. Falls back to config/env.
        """
        super().__init__(mode=ExtractionMode.API)
        settings = get_settings()
        self._openai_api_key = openai_api_key or settings.openai_api_key
        self._anthropic_api_key = anthropic_api_key or settings.anthropic_api_key
        self._entity_config = _load_entity_config()
        self._api_settings = settings.api or {}

    async def extract(self, document: Document) -> ExtractionResult:
        """Extract entities from a document using LLM APIs.

        Tries OpenAI first, falls back to Anthropic on failure.

        Args:
            document: The document to extract entities from.

        Returns:
            ExtractionResult with extracted entities and confidence scores.
        """
        self.validate_document(document)

        start_time = time.time()
        text = document.raw_text or document.content or ""

        # Build the extraction prompt
        prompt = self._build_prompt(document, self._entity_config)

        # Try OpenAI first, fall back to Anthropic
        raw_response = None
        provider = "openai"
        warnings: List[str] = []

        if self._openai_api_key:
            try:
                raw_response = await self._call_openai(prompt, text)
            except Exception as e:
                logger.warning(f"OpenAI extraction failed: {e}")
                warnings.append(f"OpenAI failed: {str(e)}")
                raw_response = None

        if raw_response is None and self._anthropic_api_key:
            provider = "anthropic"
            try:
                raw_response = await self._call_anthropic(prompt, text)
            except Exception as e:
                logger.warning(f"Anthropic extraction failed: {e}")
                warnings.append(f"Anthropic failed: {str(e)}")
                raw_response = None

        if raw_response is None:
            # Both providers failed or no keys available
            warnings.append("No API provider available or all providers failed")
            raw_response = "{}"
            provider = "none"

        # Parse the response
        entities = self._parse_response(raw_response)

        # Build confidence report
        confidence_report = self._build_confidence_report(entities, provider)

        processing_time = (time.time() - start_time) * 1000

        return ExtractionResult(
            document_id=document.id,
            extraction_mode=ExtractionMode.API,
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

    def _build_prompt(
        self, document: Document, entity_config: Dict[str, Any]
    ) -> str:
        """Build the extraction prompt with entity definitions from config.

        Args:
            document: The document being processed.
            entity_config: Entity definitions from config/entities.yaml.

        Returns:
            Formatted prompt string for the LLM.
        """
        # Determine document type from metadata or filename
        doc_type = document.metadata.get("document_type", "general")

        # Get entity definitions for this document type
        entities_section = entity_config.get("entities", {})
        target_entities = entities_section.get(doc_type, entities_section.get("general", []))

        # Build entity descriptions for the prompt
        entity_descriptions = []
        for entity_def in target_entities:
            name = entity_def.get("name", "")
            etype = entity_def.get("type", "string")
            desc = entity_def.get("description", "")
            entity_descriptions.append(f"- {name} ({etype}): {desc}")

        entities_text = "\n".join(entity_descriptions) if entity_descriptions else "- Extract all relevant entities"

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

    async def _call_openai(self, prompt: str, text: str) -> str:
        """Call OpenAI API for extraction.

        Args:
            prompt: System prompt with extraction instructions.
            text: Document text to extract from.

        Returns:
            Raw response string from OpenAI.

        Raises:
            Exception: If the API call fails after retries.
        """
        import openai

        openai_settings = self._api_settings.get("openai", {})
        model = openai_settings.get("model", "gpt-4")
        max_retries = openai_settings.get("max_retries", 3)
        temperature = openai_settings.get("temperature", 0.0)

        # Chunk text if too long
        chunks = self._chunk_text(text, self.MAX_TOKENS_OPENAI)

        client = openai.AsyncOpenAI(api_key=self._openai_api_key)

        all_responses = []
        for chunk in chunks:
            last_error = None
            for attempt in range(max_retries):
                try:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": chunk},
                        ],
                        temperature=temperature,
                        response_format={"type": "json_object"},
                    )
                    content = response.choices[0].message.content or "{}"
                    all_responses.append(content)
                    break
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        continue
                    raise last_error

        # Merge chunk responses if multiple
        if len(all_responses) == 1:
            return all_responses[0]
        return self._merge_chunk_responses(all_responses)

    async def _call_anthropic(self, prompt: str, text: str) -> str:
        """Call Anthropic API for extraction.

        Args:
            prompt: System prompt with extraction instructions.
            text: Document text to extract from.

        Returns:
            Raw response string from Anthropic.

        Raises:
            Exception: If the API call fails after retries.
        """
        import anthropic

        anthropic_settings = self._api_settings.get("anthropic", {})
        model = anthropic_settings.get("model", "claude-3-sonnet-20240229")
        max_retries = anthropic_settings.get("max_retries", 3)
        temperature = anthropic_settings.get("temperature", 0.0)

        # Chunk text if too long
        chunks = self._chunk_text(text, self.MAX_TOKENS_ANTHROPIC)

        client = anthropic.AsyncAnthropic(api_key=self._anthropic_api_key)

        all_responses = []
        for chunk in chunks:
            last_error = None
            for attempt in range(max_retries):
                try:
                    response = await client.messages.create(
                        model=model,
                        max_tokens=4096,
                        system=prompt,
                        messages=[
                            {"role": "user", "content": chunk},
                        ],
                        temperature=temperature,
                    )
                    content = response.content[0].text if response.content else "{}"
                    all_responses.append(content)
                    break
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        continue
                    raise last_error

        # Merge chunk responses if multiple
        if len(all_responses) == 1:
            return all_responses[0]
        return self._merge_chunk_responses(all_responses)

    def _parse_response(self, raw_response: str) -> Dict[str, Any]:
        """Parse structured JSON from LLM response.

        Handles cases where the LLM wraps JSON in markdown code blocks.

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
            logger.warning("Failed to parse LLM response as JSON")

        return {}

    def _chunk_text(self, text: str, max_tokens: int) -> List[str]:
        """Split text into chunks that fit within token limits.

        Args:
            text: Full document text.
            max_tokens: Maximum tokens per chunk.

        Returns:
            List of text chunks.
        """
        max_chars = max_tokens * self.CHARS_PER_TOKEN

        if len(text) <= max_chars:
            return [text]

        chunks = []
        # Split on paragraph boundaries
        paragraphs = text.split("\n\n")
        current_chunk = ""

        for para in paragraphs:
            if len(para) > max_chars:
                # Single paragraph exceeds limit, split by sentence/position
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                for i in range(0, len(para), max_chars):
                    chunks.append(para[i : i + max_chars])
            elif len(current_chunk) + len(para) + 2 > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text[:max_chars]]

    def _merge_chunk_responses(self, responses: List[str]) -> str:
        """Merge multiple chunk responses into a single JSON response.

        Args:
            responses: List of JSON response strings from different chunks.

        Returns:
            Merged JSON string.
        """
        merged: Dict[str, Any] = {}

        for response in responses:
            try:
                parsed = json.loads(response)
                if isinstance(parsed, dict):
                    for key, value in parsed.items():
                        if key in merged:
                            # Merge lists
                            if isinstance(merged[key], list) and isinstance(value, list):
                                merged[key].extend(value)
                            elif isinstance(merged[key], list):
                                merged[key].append(value)
                            # Keep the first non-empty value for scalars
                        else:
                            merged[key] = value
            except json.JSONDecodeError:
                continue

        return json.dumps(merged)

    def _build_confidence_report(
        self, entities: Dict[str, Any], provider: str
    ) -> ConfidenceReport:
        """Build confidence report for API extraction results.

        API extractions generally have high confidence when successful.

        Args:
            entities: Extracted entities.
            provider: Which API provider was used.

        Returns:
            ConfidenceReport with scores for each field.
        """
        settings = get_settings()
        threshold = settings.confidence_threshold

        scores: Dict[str, ConfidenceScore] = {}

        for field_name, value in entities.items():
            if provider == "none":
                score = 0.0
                method = "api_failed"
                explanation = "No API provider available"
            else:
                # API extractions are generally high confidence
                score = 0.9 if value else 0.5
                method = f"api_{provider}"
                explanation = f"Extracted via {provider} LLM"

            scores[field_name] = ConfidenceScore(
                score=score,
                method=method,
                explanation=explanation,
                field_name=field_name,
            )

        overall = sum(s.score for s in scores.values()) / len(scores) if scores else 0.0

        report = ConfidenceReport(
            scores=scores,
            overall_confidence=overall,
            threshold=threshold,
        )
        report.compute_low_confidence_fields()

        return report
