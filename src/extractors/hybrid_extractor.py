"""Hybrid extractor orchestrating local-first extraction with API fallback."""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Set

from src.extractors.api_extractor import APIExtractor
from src.extractors.base import BaseExtractor, ExtractionMode
from src.extractors.local_extractor import LocalExtractor
from src.models.confidence import ConfidenceReport, ConfidenceScore
from src.models.document import Document
from src.models.extraction_result import ExtractionResult
from src.utils.config import get_settings

logger = logging.getLogger(__name__)


class HybridExtractor(BaseExtractor):
    """Extractor that combines local and API extraction for optimal results.

    Runs local extraction first, then evaluates confidence per field.
    Fields with confidence below the threshold are escalated to the API
    extractor for higher-quality extraction. Results are merged with
    API values preferred for escalated fields.
    """

    def __init__(
        self,
        local_extractor: LocalExtractor = None,
        api_extractor: APIExtractor = None,
        confidence_threshold: float = None,
    ) -> None:
        """Initialize the hybrid extractor.

        Args:
            local_extractor: Local extractor instance. Created if not provided.
            api_extractor: API extractor instance. Created if not provided.
            confidence_threshold: Threshold for API escalation. Defaults to config.
        """
        super().__init__(mode=ExtractionMode.HYBRID)
        self._local_extractor = local_extractor or LocalExtractor()
        self._api_extractor = api_extractor or APIExtractor()

        settings = get_settings()
        self._threshold = confidence_threshold or settings.confidence_threshold

    async def extract(self, document: Document) -> ExtractionResult:
        """Extract entities using local-first approach with API fallback.

        Process:
        1. Run LocalExtractor to get initial results
        2. Evaluate confidence for each field
        3. Identify fields below threshold
        4. If low-confidence fields exist, call APIExtractor for those fields
        5. Merge results, preferring API values for escalated fields

        Args:
            document: The document to extract entities from.

        Returns:
            ExtractionResult with merged entities and confidence scores.
        """
        self.validate_document(document)

        start_time = time.time()
        warnings: List[str] = []

        # Step 1: Run local extraction
        local_result = await self._local_extractor.extract(document)

        # Step 2 & 3: Identify low-confidence fields
        escalated_fields = self._should_escalate(
            local_result.confidence_report, self._threshold
        )

        # Step 4: Call API extractor if there are low-confidence fields
        api_result = None
        if escalated_fields:
            logger.info(
                f"Escalating {len(escalated_fields)} fields to API: {escalated_fields}"
            )
            warnings.append(
                f"Escalated {len(escalated_fields)} low-confidence fields to API"
            )
            try:
                api_result = await self._api_extractor.extract(document)
            except Exception as e:
                logger.warning(f"API extraction failed during hybrid: {e}")
                warnings.append(f"API escalation failed: {str(e)}")
                api_result = None

        # Step 5: Merge results
        if api_result:
            merged_entities = self._merge_results(
                local_result, api_result, escalated_fields
            )
            confidence_report = self._build_merged_confidence(
                local_result, api_result, escalated_fields
            )
        else:
            merged_entities = local_result.entities
            confidence_report = local_result.confidence_report

        processing_time = (time.time() - start_time) * 1000

        return ExtractionResult(
            document_id=document.id,
            extraction_mode=ExtractionMode.HYBRID,
            entities=merged_entities,
            confidence_report=confidence_report,
            raw_text=local_result.raw_text,
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

    def _should_escalate(
        self, confidence_report: ConfidenceReport, threshold: float
    ) -> List[str]:
        """Identify fields that need API escalation based on confidence threshold.

        Args:
            confidence_report: Confidence report from local extraction.
            threshold: Minimum confidence threshold for a field to be accepted.

        Returns:
            List of field names that should be escalated to API.
        """
        low_confidence_fields = []
        for field_name, score in confidence_report.scores.items():
            if score.score < threshold:
                low_confidence_fields.append(field_name)
        return low_confidence_fields

    def _merge_results(
        self,
        local_result: ExtractionResult,
        api_result: ExtractionResult,
        escalated_fields: List[str],
    ) -> Dict[str, Any]:
        """Merge local and API extraction results.

        For escalated fields, prefer API values. For all other fields,
        keep local values. Also include any new fields from API that
        local extraction missed entirely.

        Args:
            local_result: Result from local extraction.
            api_result: Result from API extraction.
            escalated_fields: Fields that were escalated to API.

        Returns:
            Merged entity dictionary.
        """
        merged = dict(local_result.entities)

        escalated_set: Set[str] = set(escalated_fields)

        for field_name, value in api_result.entities.items():
            if field_name in escalated_set:
                # Replace local value with API value for escalated fields
                merged[field_name] = value
            elif field_name not in merged:
                # Add new fields that API found but local missed
                merged[field_name] = value

        return merged

    def _build_merged_confidence(
        self,
        local_result: ExtractionResult,
        api_result: ExtractionResult,
        escalated_fields: List[str],
    ) -> ConfidenceReport:
        """Build a merged confidence report from local and API results.

        Args:
            local_result: Result from local extraction.
            api_result: Result from API extraction.
            escalated_fields: Fields that were escalated.

        Returns:
            ConfidenceReport reflecting the merged state.
        """
        scores: Dict[str, ConfidenceScore] = {}
        escalated_set: Set[str] = set(escalated_fields)

        # Start with local scores
        for field_name, score in local_result.confidence_report.scores.items():
            if field_name in escalated_set:
                # Use API score for escalated fields if available
                if field_name in api_result.confidence_report.scores:
                    api_score = api_result.confidence_report.scores[field_name]
                    scores[field_name] = ConfidenceScore(
                        score=api_score.score,
                        method="hybrid_api",
                        explanation=f"Escalated to API due to low local confidence ({score.score:.2f})",
                        field_name=field_name,
                    )
                else:
                    scores[field_name] = score
            else:
                scores[field_name] = ConfidenceScore(
                    score=score.score,
                    method="hybrid_local",
                    explanation="Accepted from local extraction",
                    field_name=field_name,
                )

        # Add any new fields from API
        for field_name, score in api_result.confidence_report.scores.items():
            if field_name not in scores:
                scores[field_name] = ConfidenceScore(
                    score=score.score,
                    method="hybrid_api_new",
                    explanation="New field discovered by API extraction",
                    field_name=field_name,
                )

        overall = sum(s.score for s in scores.values()) / len(scores) if scores else 0.0

        report = ConfidenceReport(
            scores=scores,
            overall_confidence=overall,
            threshold=self._threshold,
        )
        report.compute_low_confidence_fields()

        return report
