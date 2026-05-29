"""Base NER engine interface that all NER engines implement."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class BaseNEREngine(ABC):
    """Abstract base class for NER engines.

    All NER engines must implement extract_entities(), is_available(), and get_name().
    The return format maps entity types to lists of (extracted_value, confidence_score) tuples.
    """

    @abstractmethod
    def extract_entities(self, text: str) -> Dict[str, List[Tuple[str, float]]]:
        """Extract named entities from text.

        Args:
            text: The text to extract entities from.

        Returns:
            Dictionary mapping entity types to lists of (value, confidence) tuples.
            Example: {"person_names": [("John Doe", 0.95)], "dates": [("2024-01-15", 0.8)]}
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this engine's dependencies are installed and usable.

        Returns:
            True if the engine can be used.
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Get the human-readable name of this NER engine.

        Returns:
            Engine name string.
        """
        ...
