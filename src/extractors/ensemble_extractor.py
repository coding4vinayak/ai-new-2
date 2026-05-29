"""Ensemble extractor orchestrating OCR ensemble + NER ensemble + optional engines."""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.extractors.base import BaseExtractor, ExtractionMode
from src.models.confidence import ConfidenceReport, ConfidenceScore
from src.models.document import Document, FileType
from src.models.extraction_result import ExtractionResult
from src.utils.config import get_settings

logger = logging.getLogger(__name__)


class EnsembleExtractor(BaseExtractor):
    """Extractor that orchestrates the full enhanced local extraction pipeline.

    Combines:
    - OCR ensemble for image/scanned documents
    - NER ensemble for entity extraction
    - Optional LayoutLM for structured document extraction
    - Optional local LLM for additional extraction

    Configuration-driven: reads from settings.yaml which models to include.
    Merges all results with confidence-weighted scoring.
    """

    def __init__(
        self,
        ocr_ensemble=None,
        ner_ensemble=None,
        layoutlm_engine=None,
        local_llm_extractor=None,
        use_layoutlm: bool = False,
        use_local_llm: bool = False,
    ) -> None:
        """Initialize the ensemble extractor.

        Args:
            ocr_ensemble: OCREnsemble instance for OCR processing.
            ner_ensemble: NERensemble instance for NER extraction.
            layoutlm_engine: LayoutLMEngine instance for layout-aware extraction.
            local_llm_extractor: OpenAICompatExtractor for local LLM extraction.
            use_layoutlm: Whether to include LayoutLM in the pipeline.
            use_local_llm: Whether to include local LLM in the pipeline.
        """
        super().__init__(mode=ExtractionMode.ENSEMBLE)
        self._ocr_ensemble = ocr_ensemble
        self._ner_ensemble = ner_ensemble
        self._layoutlm_engine = layoutlm_engine
        self._local_llm_extractor = local_llm_extractor
        self._use_layoutlm = use_layoutlm
        self._use_local_llm = use_local_llm

        settings = get_settings()
        ensemble_config = settings.ensemble or {}
        self._confidence_threshold = ensemble_config.get("confidence_threshold", 0.6)

        # Auto-configure from settings if instances not provided
        if self._ocr_ensemble is None:
            self._ocr_ensemble = self._build_ocr_ensemble(settings)
        if self._ner_ensemble is None:
            self._ner_ensemble = self._build_ner_ensemble(settings)
        if self._layoutlm_engine is None and self._use_layoutlm:
            self._layoutlm_engine = self._build_layoutlm_engine()
        if self._local_llm_extractor is None and self._use_local_llm:
            self._local_llm_extractor = self._build_local_llm_extractor()

    def _build_ocr_ensemble(self, settings):
        """Build OCR ensemble from configuration.

        Args:
            settings: Application settings.

        Returns:
            OCREnsemble instance or None if not available.
        """
        try:
            from src.processors.ocr_ensemble import OCREnsemble
            from src.processors.ocr_engines.tesseract_ocr import TesseractOCR

            ensemble = OCREnsemble()
            ensemble_config = settings.ensemble or {}
            configured_engines = ensemble_config.get("ocr_engines", ["tesseract"])

            engine_map = {"tesseract": TesseractOCR}

            # Try to import optional engines
            try:
                from src.processors.ocr_engines.trocr_engine import TrOCREngine
                engine_map["trocr"] = TrOCREngine
            except ImportError:
                pass

            try:
                from src.processors.ocr_engines.paddle_ocr_engine import PaddleOCREngine
                engine_map["paddleocr"] = PaddleOCREngine
            except ImportError:
                pass

            try:
                from src.processors.ocr_engines.doctr_engine import DocTREngine
                engine_map["doctr"] = DocTREngine
            except ImportError:
                pass

            for engine_name in configured_engines:
                engine_cls = engine_map.get(engine_name)
                if engine_cls:
                    try:
                        engine = engine_cls()
                        ensemble.add_engine(engine)
                    except Exception as e:
                        logger.warning(f"Failed to create OCR engine '{engine_name}': {e}")

            return ensemble
        except Exception as e:
            logger.warning(f"Failed to build OCR ensemble: {e}")
            return None

    def _build_ner_ensemble(self, settings):
        """Build NER ensemble from configuration.

        Args:
            settings: Application settings.

        Returns:
            NERensemble instance or None if not available.
        """
        try:
            from src.extractors.ner_ensemble import NERensemble
            from src.extractors.ner_engines.spacy_ner import SpaCyNER

            ensemble = NERensemble(confidence_threshold=self._confidence_threshold)
            ensemble_config = settings.ensemble or {}
            configured_models = ensemble_config.get("ner_models", ["spacy_sm"])

            model_map = {
                "spacy_sm": lambda: SpaCyNER(model_name="en_core_web_sm"),
                "spacy_lg": lambda: SpaCyNER(model_name="en_core_web_lg"),
            }

            # Try to import optional NER engines
            try:
                from src.extractors.ner_engines.huggingface_ner import HuggingFaceNER
                model_map["huggingface_ner"] = lambda: HuggingFaceNER()
            except ImportError:
                pass

            for model_name in configured_models:
                factory = model_map.get(model_name)
                if factory:
                    try:
                        engine = factory()
                        ensemble.add_engine(engine)
                    except Exception as e:
                        logger.warning(f"Failed to create NER engine '{model_name}': {e}")

            return ensemble
        except Exception as e:
            logger.warning(f"Failed to build NER ensemble: {e}")
            return None

    def _build_layoutlm_engine(self):
        """Build LayoutLM engine if available.

        Returns:
            LayoutLMEngine instance or None.
        """
        try:
            from src.extractors.ner_engines.layoutlm_engine import LayoutLMEngine

            engine = LayoutLMEngine()
            if engine.is_available():
                return engine
            logger.info("LayoutLM dependencies not available, skipping")
            return None
        except Exception as e:
            logger.warning(f"Failed to create LayoutLM engine: {e}")
            return None

    def _build_local_llm_extractor(self):
        """Build local LLM extractor if configured.

        Returns:
            OpenAICompatExtractor instance or None.
        """
        try:
            from src.extractors.openai_compat_extractor import OpenAICompatExtractor

            return OpenAICompatExtractor()
        except Exception as e:
            logger.warning(f"Failed to create local LLM extractor: {e}")
            return None

    async def extract(self, document: Document) -> ExtractionResult:
        """Extract entities using the full ensemble pipeline.

        Pipeline:
        1. If document is image/scanned, run OCR ensemble to get text
        2. Run NER ensemble on text for entity extraction
        3. Optionally run LayoutLM for structure-aware extraction
        4. Optionally run local LLM for additional extraction
        5. Merge all results with confidence-weighted scoring

        Args:
            document: The document to extract entities from.

        Returns:
            ExtractionResult with merged entities from all engines.
        """
        # For ensemble mode, we allow documents without text if they have
        # a file_path (OCR will extract text from image/scanned docs)
        if not (document.raw_text or document.content or document.file_path):
            self.validate_document(document)
        if not document.filename:
            raise ValueError("Document must have a filename.")

        start_time = time.time()
        warnings: List[str] = []
        all_entities: List[Tuple[Dict[str, Any], float, str]] = []

        text = document.raw_text or document.content or ""

        # Step 1: OCR ensemble for image/scanned documents
        if self._should_run_ocr(document):
            ocr_text, ocr_confidence = self._run_ocr(document)
            if ocr_text:
                text = ocr_text
                logger.info(f"OCR ensemble extracted text (confidence: {ocr_confidence:.2f})")
            else:
                warnings.append("OCR ensemble produced no results")

        # Step 2: NER ensemble for entity extraction
        ner_entities = self._run_ner(text)
        if ner_entities:
            all_entities.append((ner_entities, 0.8, "ner_ensemble"))
        else:
            warnings.append("NER ensemble produced no results")

        # Step 3: Optional LayoutLM extraction
        if self._use_layoutlm and self._layoutlm_engine:
            layoutlm_entities = self._run_layoutlm(text)
            if layoutlm_entities:
                all_entities.append((layoutlm_entities, 0.85, "layoutlm"))

        # Step 4: Optional local LLM extraction
        if self._use_local_llm and self._local_llm_extractor:
            llm_entities = await self._run_local_llm(document)
            if llm_entities:
                all_entities.append((llm_entities, 0.75, "local_llm"))

        # Step 5: Merge results with confidence weighting
        merged_entities = self._merge_all_results(all_entities)

        # Build confidence report
        confidence_report = self._build_confidence_report(merged_entities, all_entities)

        processing_time = (time.time() - start_time) * 1000

        return ExtractionResult(
            document_id=document.id,
            extraction_mode=ExtractionMode.ENSEMBLE,
            entities=merged_entities,
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

    def _should_run_ocr(self, document: Document) -> bool:
        """Determine if OCR should be run on this document.

        Args:
            document: The document to check.

        Returns:
            True if OCR should be applied.
        """
        if document.file_type == FileType.IMAGE:
            return True
        # If document has no text but has a file path, try OCR
        if not (document.raw_text or document.content) and document.file_path:
            return True
        return False

    def _run_ocr(self, document: Document) -> Tuple[str, float]:
        """Run OCR ensemble on a document.

        Args:
            document: The document to OCR.

        Returns:
            Tuple of (extracted text, confidence score).
        """
        if self._ocr_ensemble is None:
            return "", 0.0

        file_path = document.file_path or ""
        if not file_path:
            return "", 0.0

        try:
            text, confidence = self._ocr_ensemble.extract_text(file_path)
            return text, confidence
        except Exception as e:
            logger.warning(f"OCR ensemble failed: {e}")
            return "", 0.0

    def _run_ner(self, text: str) -> Dict[str, Any]:
        """Run NER ensemble on text.

        Args:
            text: Text to extract entities from.

        Returns:
            Dictionary of extracted entities.
        """
        if self._ner_ensemble is None:
            return {}

        if not text.strip():
            return {}

        try:
            # NER ensemble returns Dict[str, List[Tuple[str, float]]]
            raw_entities = self._ner_ensemble.extract_entities(text)

            # Convert to flat entity format
            entities: Dict[str, Any] = {}
            for entity_type, values in raw_entities.items():
                if len(values) == 1:
                    entities[entity_type] = values[0][0]
                else:
                    entities[entity_type] = [v for v, _ in values]

            return entities
        except Exception as e:
            logger.warning(f"NER ensemble failed: {e}")
            return {}

    def _run_layoutlm(self, text: str) -> Dict[str, Any]:
        """Run LayoutLM engine on text.

        Args:
            text: Text to process.

        Returns:
            Dictionary of extracted entities.
        """
        if self._layoutlm_engine is None:
            return {}

        if not text.strip():
            return {}

        try:
            if not self._layoutlm_engine.is_available():
                return {}

            raw_entities = self._layoutlm_engine.extract_entities(text)

            # Convert to flat entity format
            entities: Dict[str, Any] = {}
            for entity_type, values in raw_entities.items():
                if len(values) == 1:
                    entities[entity_type] = values[0][0]
                else:
                    entities[entity_type] = [v for v, _ in values]

            return entities
        except Exception as e:
            logger.warning(f"LayoutLM engine failed: {e}")
            return {}

    async def _run_local_llm(self, document: Document) -> Dict[str, Any]:
        """Run local LLM extractor on a document.

        Args:
            document: The document to extract from.

        Returns:
            Dictionary of extracted entities.
        """
        if self._local_llm_extractor is None:
            return {}

        try:
            result = await self._local_llm_extractor.extract(document)
            return result.entities
        except Exception as e:
            logger.warning(f"Local LLM extraction failed: {e}")
            return {}

    def _merge_all_results(
        self,
        all_entities: List[Tuple[Dict[str, Any], float, str]],
    ) -> Dict[str, Any]:
        """Merge results from all engines with confidence-weighted scoring.

        For each field, picks the value from the source with highest weight.
        If multiple sources agree on a value, confidence is boosted.

        Args:
            all_entities: List of (entities_dict, weight, source_name) tuples.

        Returns:
            Merged entity dictionary.
        """
        if not all_entities:
            return {}

        # Collect all values per field with their weights
        field_values: Dict[str, List[Tuple[Any, float, str]]] = {}

        for entities, weight, source in all_entities:
            for field, value in entities.items():
                if field not in field_values:
                    field_values[field] = []
                field_values[field].append((value, weight, source))

        # For each field, select the best value
        merged: Dict[str, Any] = {}
        for field, values in field_values.items():
            if len(values) == 1:
                merged[field] = values[0][0]
            else:
                # Pick value from highest-weight source
                best = max(values, key=lambda x: x[1])
                merged[field] = best[0]

        return merged

    def _build_confidence_report(
        self,
        merged_entities: Dict[str, Any],
        all_entities: List[Tuple[Dict[str, Any], float, str]],
    ) -> ConfidenceReport:
        """Build confidence report based on ensemble agreement.

        Fields extracted by multiple engines get higher confidence.

        Args:
            merged_entities: The final merged entities.
            all_entities: All source results with weights.

        Returns:
            ConfidenceReport with per-field confidence scores.
        """
        scores: Dict[str, ConfidenceScore] = {}
        num_sources = len(all_entities)

        for field_name in merged_entities:
            # Count how many sources extracted this field
            sources_with_field = []
            for entities, weight, source in all_entities:
                if field_name in entities:
                    sources_with_field.append((weight, source))

            # Base confidence from best source weight
            if sources_with_field:
                base_confidence = max(w for w, _ in sources_with_field)
                # Boost for multi-source agreement
                agreement_ratio = len(sources_with_field) / max(num_sources, 1)
                confidence = min(base_confidence + (0.15 * agreement_ratio), 1.0)
            else:
                confidence = 0.5

            source_names = [s for _, s in sources_with_field]
            scores[field_name] = ConfidenceScore(
                score=confidence,
                method="ensemble",
                explanation=f"Extracted by: {', '.join(source_names)}",
                field_name=field_name,
            )

        overall = (
            sum(s.score for s in scores.values()) / len(scores) if scores else 0.0
        )

        report = ConfidenceReport(
            scores=scores,
            overall_confidence=overall,
            threshold=self._confidence_threshold,
        )
        report.compute_low_confidence_fields()

        return report
