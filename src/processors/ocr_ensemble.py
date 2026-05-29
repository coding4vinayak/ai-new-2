"""OCR ensemble orchestrator with confidence-weighted result merging."""

import logging
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from src.processors.ocr_engines.base_ocr import BaseOCREngine

logger = logging.getLogger(__name__)


class OCREnsemble:
    """Orchestrates multiple OCR engines and merges results.

    Runs all available engines on an image and uses confidence-weighted
    merging to produce the best possible text extraction result.
    """

    def __init__(
        self,
        engines: Optional[List[BaseOCREngine]] = None,
        min_engines: int = 1,
        similarity_threshold: float = 0.7,
        fallback_strategy: str = "highest_confidence",
    ) -> None:
        """Initialize the OCR ensemble.

        Args:
            engines: List of OCR engine instances to use.
            min_engines: Minimum number of engines that must succeed.
            similarity_threshold: Threshold for considering texts as similar (0-1).
            fallback_strategy: Strategy when engines disagree.
                Options: 'highest_confidence', 'longest_text', 'first_available'.
        """
        self.engines = engines or []
        self.min_engines = min_engines
        self.similarity_threshold = similarity_threshold
        self.fallback_strategy = fallback_strategy

    def add_engine(self, engine: BaseOCREngine) -> None:
        """Add an OCR engine to the ensemble.

        Args:
            engine: An OCR engine instance implementing BaseOCREngine.
        """
        self.engines.append(engine)

    def get_available_engines(self) -> List[BaseOCREngine]:
        """Get list of currently available engines.

        Returns:
            List of engines that report themselves as available.
        """
        return [e for e in self.engines if e.is_available()]

    def extract_text(self, image_path: str) -> Tuple[str, float]:
        """Extract text using all available engines and merge results.

        Runs all available engines, collects results, and applies
        confidence-weighted merging to produce the best result.

        Args:
            image_path: Path to the image file.

        Returns:
            Tuple of (merged text, combined confidence score 0-1).
        """
        available = self.get_available_engines()

        if not available:
            logger.warning("No OCR engines available in ensemble")
            return "", 0.0

        # Collect results from all available engines
        results: List[Tuple[str, float, str]] = []
        for engine in available:
            try:
                text, confidence = engine.extract_text(image_path)
                if text.strip():
                    results.append((text, confidence, engine.get_name()))
                    logger.debug(
                        f"Engine '{engine.get_name()}' returned "
                        f"{len(text)} chars with confidence {confidence:.3f}"
                    )
            except Exception as e:
                logger.warning(
                    f"Engine '{engine.get_name()}' failed: {e}"
                )
                continue

        if not results:
            logger.warning("All OCR engines produced empty results")
            return "", 0.0

        if len(results) < self.min_engines:
            logger.warning(
                f"Only {len(results)} engines succeeded, "
                f"minimum required is {self.min_engines}"
            )

        # If only one result, return it directly
        if len(results) == 1:
            return results[0][0], results[0][1]

        return self._merge_results(results)

    def _merge_results(
        self, results: List[Tuple[str, float, str]]
    ) -> Tuple[str, float]:
        """Merge results from multiple engines using confidence weighting.

        If texts are similar (high overlap), picks the highest confidence one.
        If texts differ significantly, uses the fallback strategy.

        Args:
            results: List of (text, confidence, engine_name) tuples.

        Returns:
            Tuple of (merged text, combined confidence).
        """
        # Sort by confidence descending
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

        # Check similarity between top results
        best_text, best_confidence, best_engine = sorted_results[0]

        # Calculate pairwise similarities with the best result
        similar_results = [(best_text, best_confidence)]
        for text, confidence, engine_name in sorted_results[1:]:
            similarity = self._text_similarity(best_text, text)
            if similarity >= self.similarity_threshold:
                similar_results.append((text, confidence))

        # If most results are similar, use confidence-weighted selection
        if len(similar_results) > len(sorted_results) / 2:
            # Texts agree - return highest confidence with boosted score
            agreement_bonus = min(0.1, 0.05 * (len(similar_results) - 1))
            merged_confidence = min(1.0, best_confidence + agreement_bonus)
            return best_text, merged_confidence

        # Texts disagree - use fallback strategy
        return self._apply_fallback(sorted_results)

    def _apply_fallback(
        self, sorted_results: List[Tuple[str, float, str]]
    ) -> Tuple[str, float]:
        """Apply fallback strategy when engines disagree.

        Args:
            sorted_results: Results sorted by confidence descending.

        Returns:
            Tuple of (selected text, confidence).
        """
        if self.fallback_strategy == "longest_text":
            longest = max(sorted_results, key=lambda x: len(x[0]))
            return longest[0], longest[1]
        elif self.fallback_strategy == "first_available":
            return sorted_results[0][0], sorted_results[0][1]
        else:
            # Default: highest_confidence (already sorted)
            return sorted_results[0][0], sorted_results[0][1]

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two texts.

        Args:
            text1: First text string.
            text2: Second text string.

        Returns:
            Similarity ratio between 0 and 1.
        """
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        return SequenceMatcher(None, text1, text2).ratio()

    def get_engine_status(self) -> Dict[str, bool]:
        """Get availability status of all engines.

        Returns:
            Dictionary mapping engine names to availability status.
        """
        return {
            engine.get_name(): engine.is_available() for engine in self.engines
        }
