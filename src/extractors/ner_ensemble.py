"""NER ensemble orchestrator with confidence-weighted merging."""

import logging
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from src.extractors.ner_engines.base_ner import BaseNEREngine

logger = logging.getLogger(__name__)


class NERensemble:
    """NER ensemble that runs multiple NER engines and merges results.

    For each text: runs all available NER engines, collects entity results,
    and implements confidence-weighted merging. For each entity type, collects
    all extracted values across engines, groups similar values (fuzzy matching),
    computes weighted confidence for each group, and returns the highest-confidence
    unique values.
    """

    def __init__(
        self,
        engines: Optional[List[BaseNEREngine]] = None,
        weights: Optional[Dict[str, float]] = None,
        confidence_threshold: float = 0.3,
    ) -> None:
        """Initialize the NER ensemble.

        Args:
            engines: List of BaseNEREngine instances to use.
            weights: Optional weights per engine name. Higher weight = more influence.
            confidence_threshold: Minimum confidence to include an entity in results.
        """
        self._engines = engines or []
        self._weights = weights or {}
        self._confidence_threshold = confidence_threshold

    @property
    def engines(self) -> List[BaseNEREngine]:
        """Get list of engines."""
        return self._engines

    def add_engine(self, engine: BaseNEREngine) -> None:
        """Add an engine to the ensemble.

        Args:
            engine: NER engine to add.
        """
        self._engines.append(engine)

    def extract_entities(self, text: str) -> Dict[str, List[Tuple[str, float]]]:
        """Run all available engines and merge results with confidence weighting.

        Args:
            text: Text to extract entities from.

        Returns:
            Dictionary mapping entity types to lists of (value, confidence) tuples.
        """
        all_results: List[Tuple[str, Dict[str, List[Tuple[str, float]]]]] = []

        for engine in self._engines:
            if not engine.is_available():
                logger.debug(f"Skipping unavailable engine: {engine.get_name()}")
                continue

            try:
                result = engine.extract_entities(text)
                all_results.append((engine.get_name(), result))
            except Exception as e:
                logger.warning(
                    f"Engine {engine.get_name()} failed: {e}"
                )
                continue

        if not all_results:
            logger.warning("No NER engines produced results")
            return {}

        return self._merge_results(all_results)

    def _merge_results(
        self,
        all_results: List[Tuple[str, Dict[str, List[Tuple[str, float]]]]],
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Merge results from multiple engines with confidence-weighted voting.

        Args:
            all_results: List of (engine_name, entities_dict) tuples.

        Returns:
            Merged dictionary of entity types to (value, confidence) tuples.
        """
        # Collect all values by entity type
        type_values: Dict[str, List[Tuple[str, float, str]]] = {}

        for engine_name, entities in all_results:
            weight = self._weights.get(engine_name, 1.0)

            for entity_type, values in entities.items():
                if entity_type not in type_values:
                    type_values[entity_type] = []

                for value, confidence in values:
                    weighted_confidence = confidence * weight
                    type_values[entity_type].append(
                        (value, weighted_confidence, engine_name)
                    )

        # Group similar values and compute final confidence
        merged: Dict[str, List[Tuple[str, float]]] = {}

        for entity_type, values in type_values.items():
            grouped = self._group_similar_values(values)
            filtered = [
                (v, c) for v, c in grouped if c >= self._confidence_threshold
            ]
            # Sort by confidence descending
            filtered.sort(key=lambda x: x[1], reverse=True)
            if filtered:
                merged[entity_type] = filtered

        return merged

    def _group_similar_values(
        self, values: List[Tuple[str, float, str]]
    ) -> List[Tuple[str, float]]:
        """Group similar entity values and compute combined confidence.

        Uses simple string similarity to group values that likely refer to
        the same entity. Combined confidence is the weighted average of
        all values in the group, boosted by the number of engines that agree.

        Args:
            values: List of (value, weighted_confidence, engine_name) tuples.

        Returns:
            List of (canonical_value, combined_confidence) tuples.
        """
        if not values:
            return []

        groups: List[List[Tuple[str, float, str]]] = []

        for value, confidence, engine_name in values:
            matched = False
            for group in groups:
                # Check if this value is similar to any in the group
                representative = group[0][0]
                if self._is_similar(value, representative):
                    group.append((value, confidence, engine_name))
                    matched = True
                    break

            if not matched:
                groups.append([(value, confidence, engine_name)])

        # For each group, pick the best representative and compute confidence
        result: List[Tuple[str, float]] = []
        num_engines = len(set(e for _, _, e in values))

        for group in groups:
            # Use the value with highest individual confidence as canonical
            best_value = max(group, key=lambda x: x[1])[0]

            # Combine confidences: average weighted confidence, boosted by agreement
            confidences = [c for _, c, _ in group]
            avg_confidence = sum(confidences) / len(confidences)

            # Boost for multi-engine agreement
            unique_engines = len(set(e for _, _, e in group))
            agreement_boost = min(unique_engines / max(num_engines, 1), 1.0)
            combined = min(avg_confidence * (1.0 + 0.2 * agreement_boost), 1.0)

            result.append((best_value, combined))

        return result

    def _is_similar(self, a: str, b: str) -> bool:
        """Check if two entity strings are similar enough to be grouped.

        Uses case-insensitive containment and SequenceMatcher ratio for
        sequence-aware similarity comparison.

        Args:
            a: First string.
            b: Second string.

        Returns:
            True if the strings are considered similar.
        """
        a_lower = a.lower().strip()
        b_lower = b.lower().strip()

        # Exact match
        if a_lower == b_lower:
            return True

        # One contains the other
        if a_lower in b_lower or b_lower in a_lower:
            return True

        # Sequence-aware similarity (SequenceMatcher ratio)
        similarity = SequenceMatcher(None, a_lower, b_lower).ratio()

        return similarity > 0.8
