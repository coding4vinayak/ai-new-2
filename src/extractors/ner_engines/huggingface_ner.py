"""HuggingFace transformer NER engine using pipeline('ner')."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.extractors.ner_engines.base_ner import BaseNEREngine

logger = logging.getLogger(__name__)

try:
    from transformers import pipeline as hf_pipeline

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


# Mapping from HuggingFace BIO entity labels to standard entity types
_HF_LABEL_MAP = {
    "PER": "person_names",
    "LOC": "locations",
    "ORG": "organization_names",
    "MISC": "miscellaneous",
}


class HuggingFaceNER(BaseNEREngine):
    """NER engine using HuggingFace transformers pipeline.

    Supports any HuggingFace NER model. Default model is 'dslim/bert-base-NER'.
    Maps HF entity labels (B-PER, I-PER, B-ORG, etc.) to standard types.
    Uses aggregation_strategy='simple' to merge sub-word tokens.
    """

    def __init__(self, model_name: str = "dslim/bert-base-NER") -> None:
        """Initialize the HuggingFace NER engine.

        Args:
            model_name: HuggingFace model name for NER. Default: 'dslim/bert-base-NER'.
        """
        self._model_name = model_name
        self._pipeline: Optional[Any] = None

    def _load_pipeline(self) -> None:
        """Lazily load the NER pipeline."""
        if self._pipeline is None:
            self._pipeline = hf_pipeline(
                "ner",
                model=self._model_name,
                aggregation_strategy="simple",
            )

    def extract_entities(self, text: str) -> Dict[str, List[Tuple[str, float]]]:
        """Extract named entities using HuggingFace NER pipeline.

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
            logger.warning(f"HuggingFace NER pipeline error: {e}")
            return {}

        entities: Dict[str, List[Tuple[str, float]]] = {}

        for result in results:
            # Handle aggregated entity group labels
            label = result.get("entity_group", result.get("entity", ""))
            # Strip B-/I- prefixes if present
            clean_label = label.replace("B-", "").replace("I-", "")

            entity_type = _HF_LABEL_MAP.get(clean_label)
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
        """Check if transformers is importable.

        Returns:
            True if transformers is available.
        """
        return _TRANSFORMERS_AVAILABLE

    def get_name(self) -> str:
        """Get the engine name.

        Returns:
            Engine name string.
        """
        return f"huggingface-{self._model_name}"
