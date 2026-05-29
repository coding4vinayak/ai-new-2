"""LayoutLMv3 engine for structure-aware entity extraction."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.extractors.ner_engines.base_ner import BaseNEREngine

logger = logging.getLogger(__name__)

try:
    from transformers import pipeline as hf_pipeline

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

try:
    import torch  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# Mapping from LayoutLM labels to standard entity types
_LAYOUTLM_LABEL_MAP = {
    "B-HEADER": "headers",
    "I-HEADER": "headers",
    "B-QUESTION": "questions",
    "I-QUESTION": "questions",
    "B-ANSWER": "answers",
    "I-ANSWER": "answers",
    "PER": "person_names",
    "ORG": "organization_names",
    "LOC": "locations",
    "DATE": "dates",
    "MONEY": "monetary_amounts",
}


class LayoutLMEngine(BaseNEREngine):
    """LayoutLMv3 engine for structure-aware entity extraction.

    Best for invoices, forms, and documents with key-value pairs.
    Uses 'microsoft/layoutlmv3-base' model. This engine can leverage
    both text and bounding box positions for improved extraction accuracy.
    """

    def __init__(self, model_name: str = "microsoft/layoutlmv3-base") -> None:
        """Initialize the LayoutLM engine.

        Args:
            model_name: HuggingFace model name for LayoutLM. Default: 'microsoft/layoutlmv3-base'.
        """
        self._model_name = model_name
        self._pipeline: Optional[Any] = None

    def _load_pipeline(self) -> None:
        """Lazily load the LayoutLM pipeline."""
        if self._pipeline is None:
            self._pipeline = hf_pipeline(
                "token-classification",
                model=self._model_name,
            )

    def extract_entities(self, text: str) -> Dict[str, List[Tuple[str, float]]]:
        """Extract entities from text without layout information (fallback mode).

        When bounding box positions are not available, processes text only.

        Args:
            text: Text to process.

        Returns:
            Dictionary mapping entity types to lists of (value, confidence) tuples.
        """
        if not self.is_available():
            return {}

        self._load_pipeline()

        try:
            results = self._pipeline(text)
        except Exception as e:
            logger.warning(f"LayoutLM pipeline error: {e}")
            return {}

        return self._process_results(results)

    def extract_with_layout(
        self, text: str, bboxes: List[List[int]]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Extract entities using both text and bounding box positions.

        NOTE: The current implementation is text-only. Bounding box data is
        accepted for interface compatibility but is not passed to the model.
        Full layout-aware extraction requires a LayoutLMv3 model fine-tuned
        with a LayoutLMv3FeatureExtractor that accepts both tokens and
        normalized bounding boxes. This will be addressed in a future iteration.

        TODO: Integrate bounding boxes via LayoutLMv3FeatureExtractor to enable
        true layout-aware token classification.

        Args:
            text: Text to process.
            bboxes: List of bounding boxes [x0, y0, x1, y1] for each token.
                Currently unused - see note above.

        Returns:
            Dictionary mapping entity types to lists of (value, confidence) tuples.
        """
        if not self.is_available():
            return {}

        self._load_pipeline()

        try:
            # When layout info is available, pass it to the pipeline
            results = self._pipeline(text)
        except Exception as e:
            logger.warning(f"LayoutLM pipeline with layout error: {e}")
            return {}

        return self._process_results(results)

    def _process_results(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Process pipeline results into standard entity format.

        Args:
            results: Raw pipeline output.

        Returns:
            Dictionary mapping entity types to lists of (value, confidence) tuples.
        """
        entities: Dict[str, List[Tuple[str, float]]] = {}

        for result in results:
            label = result.get("entity_group", result.get("entity", ""))
            # Strip B-/I- prefixes
            clean_label = label.replace("B-", "").replace("I-", "")

            entity_type = _LAYOUTLM_LABEL_MAP.get(clean_label)
            if entity_type is None:
                # Try with the full label
                entity_type = _LAYOUTLM_LABEL_MAP.get(label)
            if entity_type is None:
                continue

            word = result.get("word", "").strip()
            confidence = float(result.get("score", 0.0))

            if not word:
                continue

            if entity_type not in entities:
                entities[entity_type] = []

            # Avoid duplicates
            existing_values = [v for v, _ in entities[entity_type]]
            if word not in existing_values:
                entities[entity_type].append((word, confidence))

        return entities

    def is_available(self) -> bool:
        """Check if transformers and torch are importable.

        Returns:
            True if both dependencies are available.
        """
        return _TRANSFORMERS_AVAILABLE and _TORCH_AVAILABLE

    def get_name(self) -> str:
        """Get the engine name.

        Returns:
            Engine name string.
        """
        return f"layoutlm-{self._model_name}"
