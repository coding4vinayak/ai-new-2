"""SpaCy NER engine wrapping existing spaCy NER logic."""

import logging
from typing import Dict, List, Optional, Tuple

from src.extractors.ner_engines.base_ner import BaseNEREngine

logger = logging.getLogger(__name__)

try:
    import spacy

    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False


# Mapping from spaCy entity labels to standard entity types
_SPACY_LABEL_MAP = {
    "PERSON": "person_names",
    "ORG": "organization_names",
    "DATE": "dates",
    "MONEY": "monetary_amounts",
    "GPE": "locations",
}


class SpaCyNER(BaseNEREngine):
    """NER engine using spaCy's built-in NER pipeline.

    Maps spaCy labels (PERSON, ORG, DATE, MONEY, GPE) to standard
    entity types (person_names, organization_names, dates, monetary_amounts, locations).
    """

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        """Initialize the spaCy NER engine.

        Args:
            model_name: Name of the spaCy model to load.
        """
        self._model_name = model_name
        self._nlp: Optional[object] = None

    @property
    def nlp(self):
        """Lazy-load the spaCy model."""
        if self._nlp is None:
            self._nlp = spacy.load(self._model_name)
        return self._nlp

    def extract_entities(self, text: str) -> Dict[str, List[Tuple[str, float]]]:
        """Extract named entities using spaCy NER pipeline.

        Args:
            text: Text to process.

        Returns:
            Dictionary mapping entity types to lists of (value, confidence) tuples.
        """
        if not self.is_available():
            return {}

        doc = self.nlp(text)

        entities: Dict[str, List[Tuple[str, float]]] = {}

        for ent in doc.ents:
            entity_type = _SPACY_LABEL_MAP.get(ent.label_)
            if entity_type is None:
                continue

            # Use spaCy's entity score if available, otherwise default to 0.8
            confidence = 0.8
            if hasattr(ent, "kb_id_") and ent.kb_id_:
                confidence = 0.9

            if entity_type not in entities:
                entities[entity_type] = []

            # Avoid duplicates
            existing_values = [v for v, _ in entities[entity_type]]
            if ent.text not in existing_values:
                entities[entity_type].append((ent.text, confidence))

        return entities

    def is_available(self) -> bool:
        """Check if spaCy is importable.

        Returns:
            True if spaCy is available.
        """
        return _SPACY_AVAILABLE

    def get_name(self) -> str:
        """Get the engine name.

        Returns:
            Engine name string.
        """
        return f"spacy-{self._model_name}"
