"""Local extractor using spaCy NER and regex patterns for entity extraction."""

import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import yaml

from src.extractors.base import BaseExtractor, ExtractionMode
from src.models.confidence import ConfidenceReport, ConfidenceScore
from src.models.document import Document
from src.models.extraction_result import ExtractionResult
from src.utils.config import get_settings


def _load_entity_config() -> Dict[str, Any]:
    """Load entity configuration from YAML."""
    from pathlib import Path

    config_path = Path(__file__).parent.parent.parent / "config" / "entities.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


class LocalExtractor(BaseExtractor):
    """Extractor using spaCy NER and regex patterns for local entity extraction.

    This extractor does not require any external API calls and processes
    documents entirely on the local machine using NLP models and pattern matching.
    """

    def __init__(
        self, spacy_model: Optional[str] = None, use_ensemble: bool = False
    ) -> None:
        """Initialize the local extractor.

        Args:
            spacy_model: Name of the spaCy model to load. Defaults to config value.
            use_ensemble: If True, use NER ensemble instead of just spaCy.
        """
        super().__init__(mode=ExtractionMode.LOCAL)
        settings = get_settings()
        self._model_name = spacy_model or settings.spacy_model
        self._nlp = None
        self._use_ensemble = use_ensemble
        self._entity_config = _load_entity_config()
        self._ensemble = None

    @property
    def nlp(self):
        """Lazy-load the spaCy model."""
        if self._nlp is None:
            import spacy

            self._nlp = spacy.load(self._model_name)
        return self._nlp

    def _get_ensemble(self):
        """Get or create the NER ensemble instance."""
        if self._ensemble is None:
            from src.extractors.ner_engines import HuggingFaceNER, SpaCyNER
            from src.extractors.ner_ensemble import NERensemble

            engines = [SpaCyNER(model_name=self._model_name), HuggingFaceNER()]
            self._ensemble = NERensemble(engines=engines)
        return self._ensemble

    def _extract_with_ensemble(self, text: str) -> Dict[str, Any]:
        """Extract entities using the NER ensemble.

        Args:
            text: Text to process.

        Returns:
            Dictionary of extracted entities grouped by type.
        """
        ensemble = self._get_ensemble()
        results = ensemble.extract_entities(text)

        # Convert from ensemble format (value, confidence) tuples to flat lists
        entities: Dict[str, Any] = {}
        for entity_type, values in results.items():
            entities[entity_type] = [v for v, _ in values]

        return entities

    async def extract(self, document: Document) -> ExtractionResult:
        """Extract entities from a document using spaCy and regex patterns.

        Args:
            document: The document to extract entities from.

        Returns:
            ExtractionResult with extracted entities and confidence scores.
        """
        self.validate_document(document)

        start_time = time.time()
        text = document.raw_text or document.content or ""

        # Extract entities using NER (ensemble or spaCy only)
        if self._use_ensemble:
            spacy_entities = self._extract_with_ensemble(text)
        else:
            spacy_entities = self._extract_with_spacy(text)

        # Extract entities using regex patterns from entity config
        regex_entities = self._extract_with_regex(text, self._entity_config)

        # Merge results (NER entities take precedence for overlapping fields)
        merged_entities: Dict[str, Any] = {}
        merged_entities.update(regex_entities)
        merged_entities.update(spacy_entities)

        # Compute confidence scores
        confidence_report = self._compute_confidence(merged_entities, spacy_entities, regex_entities)

        processing_time = (time.time() - start_time) * 1000

        return ExtractionResult(
            document_id=document.id,
            extraction_mode=ExtractionMode.LOCAL,
            entities=merged_entities,
            confidence_report=confidence_report,
            raw_text=text,
            processing_time_ms=processing_time,
            extracted_at=datetime.utcnow(),
            extractor_version=self._get_version(),
            warnings=[],
        )

    async def get_confidence(self, result: ExtractionResult) -> ConfidenceReport:
        """Return the confidence report from an extraction result.

        Args:
            result: The extraction result to evaluate.

        Returns:
            ConfidenceReport with per-field and overall confidence scores.
        """
        return result.confidence_report

    def _extract_with_spacy(self, text: str) -> Dict[str, Any]:
        """Extract named entities using spaCy NER pipeline.

        Args:
            text: Text to process.

        Returns:
            Dictionary of extracted entities grouped by type.
        """
        doc = self.nlp(text)

        entities: Dict[str, List[str]] = {
            "person_names": [],
            "organization_names": [],
            "dates": [],
            "monetary_amounts": [],
            "locations": [],
        }

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                if ent.text not in entities["person_names"]:
                    entities["person_names"].append(ent.text)
            elif ent.label_ == "ORG":
                if ent.text not in entities["organization_names"]:
                    entities["organization_names"].append(ent.text)
            elif ent.label_ == "DATE":
                if ent.text not in entities["dates"]:
                    entities["dates"].append(ent.text)
            elif ent.label_ == "MONEY":
                if ent.text not in entities["monetary_amounts"]:
                    entities["monetary_amounts"].append(ent.text)
            elif ent.label_ == "GPE":
                if ent.text not in entities["locations"]:
                    entities["locations"].append(ent.text)

        # Remove empty entries
        return {k: v for k, v in entities.items() if v}

    def _extract_with_regex(
        self, text: str, entity_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract entities using regex patterns based on entity configuration.

        Args:
            text: Text to process.
            entity_config: Entity definitions from config/entities.yaml.

        Returns:
            Dictionary of extracted entities from regex matching.
        """
        entities: Dict[str, Any] = {}

        # Date patterns (various formats)
        date_patterns = [
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
            r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
        ]
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        if dates:
            entities["dates"] = list(set(dates))

        # Monetary amount patterns
        money_patterns = [
            r"\$[\d,]+(?:\.\d{2})?",
            r"USD\s*[\d,]+(?:\.\d{2})?",
            r"[\d,]+(?:\.\d{2})?\s*(?:USD|EUR|GBP|dollars|pounds|euros)",
        ]
        amounts = []
        for pattern in money_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            amounts.extend(matches)
        if amounts:
            entities["monetary_amounts"] = list(set(amounts))

        # Invoice number patterns
        invoice_patterns = [
            r"(?:Invoice|Inv|INV)[\s#:]*([A-Z0-9][\w-]{3,20})",
            r"(?:Invoice Number|Invoice No\.?)[\s:]*([A-Z0-9][\w-]{3,20})",
        ]
        invoice_numbers = []
        for pattern in invoice_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            invoice_numbers.extend(matches)
        if invoice_numbers:
            entities["invoice_number"] = invoice_numbers[0] if len(invoice_numbers) == 1 else invoice_numbers

        # Email patterns
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        emails = re.findall(email_pattern, text)
        if emails:
            entities["emails"] = list(set(emails))

        # Phone number patterns
        phone_pattern = r"(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
        phones = re.findall(phone_pattern, text)
        if phones:
            entities["phone_numbers"] = list(set(phones))

        return entities

    def _compute_confidence(
        self,
        merged_entities: Dict[str, Any],
        spacy_entities: Dict[str, Any],
        regex_entities: Dict[str, Any],
    ) -> ConfidenceReport:
        """Compute confidence scores for each extracted field.

        Confidence is based on:
        - Whether the field was found by spaCy NER (higher confidence)
        - Whether the field was found by regex (moderate confidence)
        - Whether both methods agree (highest confidence)

        Args:
            merged_entities: The final merged entity dictionary.
            spacy_entities: Entities found by spaCy.
            regex_entities: Entities found by regex.

        Returns:
            ConfidenceReport with per-field scores.
        """
        settings = get_settings()
        threshold = settings.confidence_threshold

        scores: Dict[str, ConfidenceScore] = {}

        for field_name in merged_entities:
            in_spacy = field_name in spacy_entities and bool(spacy_entities[field_name])
            in_regex = field_name in regex_entities and bool(regex_entities[field_name])

            if in_spacy and in_regex:
                # Both methods found this entity - high confidence
                score = 0.95
                method = "local_ner+regex"
                explanation = "Confirmed by both NER and pattern matching"
            elif in_spacy:
                # Only spaCy found it - moderate-high confidence
                score = 0.8
                method = "local_ner"
                explanation = "Extracted by spaCy NER model"
            elif in_regex:
                # Only regex found it - moderate confidence
                score = 0.7
                method = "local_regex"
                explanation = "Matched by regex pattern"
            else:
                score = 0.5
                method = "local_unknown"
                explanation = "Source undetermined"

            scores[field_name] = ConfidenceScore(
                score=score,
                method=method,
                explanation=explanation,
                field_name=field_name,
            )

        # Calculate overall confidence
        if scores:
            overall = sum(s.score for s in scores.values()) / len(scores)
        else:
            overall = 0.0

        report = ConfidenceReport(
            scores=scores,
            overall_confidence=overall,
            threshold=threshold,
        )
        report.compute_low_confidence_fields()

        return report
