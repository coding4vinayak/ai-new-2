"""Base extractor interface defining the contract for all extraction modes."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional

from src.models.confidence import ConfidenceReport
from src.models.document import Document
from src.models.extraction_result import ExtractionResult


class ExtractionMode(str, Enum):
    """Available extraction modes."""

    LOCAL = "local"
    API = "api"
    HYBRID = "hybrid"


class BaseExtractor(ABC):
    """Abstract base class for document entity extractors.

    All extraction implementations (local, API, hybrid) must inherit
    from this class and implement the required abstract methods.
    """

    def __init__(self, mode: ExtractionMode) -> None:
        """Initialize the base extractor.

        Args:
            mode: The extraction mode this extractor operates in.
        """
        self.mode = mode

    @abstractmethod
    async def extract(self, document: Document) -> ExtractionResult:
        """Extract entities from a document.

        This is the primary extraction method that must be implemented
        by all concrete extractors.

        Args:
            document: The document to extract entities from.

        Returns:
            ExtractionResult containing extracted entities and metadata.
        """
        ...

    @abstractmethod
    async def get_confidence(self, result: ExtractionResult) -> ConfidenceReport:
        """Calculate confidence scores for an extraction result.

        Args:
            result: The extraction result to evaluate.

        Returns:
            ConfidenceReport with per-field and overall scores.
        """
        ...

    def validate_document(self, document: Document) -> bool:
        """Validate that a document is suitable for extraction.

        Args:
            document: The document to validate.

        Returns:
            True if the document is valid for extraction.

        Raises:
            ValueError: If the document is invalid.
        """
        if not document.raw_text and not document.content:
            raise ValueError(
                "Document has no text content. Ensure it has been processed first."
            )
        if not document.filename:
            raise ValueError("Document must have a filename.")
        return True

    def format_output(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format raw extraction output into a standardized structure.

        Args:
            raw_result: Raw extraction output from the extraction method.

        Returns:
            Standardized output dictionary.
        """
        return {
            "entities": raw_result.get("entities", {}),
            "metadata": {
                "mode": self.mode.value,
                "version": self._get_version(),
            },
            "raw": raw_result,
        }

    def _get_version(self) -> str:
        """Get the extractor version string.

        Returns:
            Version string.
        """
        return "1.0.0"
