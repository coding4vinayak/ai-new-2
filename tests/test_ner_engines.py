"""Tests for NER engines with mocked dependencies."""

from unittest.mock import MagicMock, patch

import pytest

from src.extractors.ner_engines.base_ner import BaseNEREngine
from src.extractors.ner_engines.huggingface_ner import HuggingFaceNER
from src.extractors.ner_engines.layoutlm_engine import LayoutLMEngine
from src.extractors.ner_engines.spacy_ner import SpaCyNER
from src.extractors.ner_ensemble import NERensemble


# --- BaseNEREngine interface tests ---


class TestBaseNEREngine:
    """Test that BaseNEREngine interface is properly defined."""

    def test_cannot_instantiate_abstract_class(self):
        """BaseNEREngine cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseNEREngine()

    def test_subclass_must_implement_methods(self):
        """Subclasses must implement all abstract methods."""

        class IncompleteEngine(BaseNEREngine):
            pass

        with pytest.raises(TypeError):
            IncompleteEngine()


# --- SpaCyNER tests ---


class TestSpaCyNER:
    """Test SpaCyNER engine with mocked spaCy model."""

    def _make_mock_doc(self):
        """Create a mock spaCy doc with entities."""

        class MockEntity:
            def __init__(self, text, label_):
                self.text = text
                self.label_ = label_
                self.kb_id_ = ""

        mock_doc = MagicMock()
        mock_doc.ents = [
            MockEntity("John Doe", "PERSON"),
            MockEntity("Acme Corp", "ORG"),
            MockEntity("January 15, 2024", "DATE"),
            MockEntity("$50,000", "MONEY"),
            MockEntity("New York", "GPE"),
        ]
        return mock_doc

    @patch("src.extractors.ner_engines.spacy_ner._SPACY_AVAILABLE", True)
    def test_extract_entities_correct_format(self):
        """SpaCyNER produces correct output format: Dict[str, List[Tuple[str, float]]]."""
        engine = SpaCyNER()
        mock_doc = self._make_mock_doc()
        mock_nlp = MagicMock(return_value=mock_doc)
        engine._nlp = mock_nlp

        result = engine.extract_entities("Sample text about John Doe at Acme Corp")

        assert isinstance(result, dict)
        for entity_type, values in result.items():
            assert isinstance(values, list)
            for value, confidence in values:
                assert isinstance(value, str)
                assert isinstance(confidence, float)
                assert 0 <= confidence <= 1.0

    @patch("src.extractors.ner_engines.spacy_ner._SPACY_AVAILABLE", True)
    def test_maps_spacy_labels_correctly(self):
        """SpaCyNER maps PERSON->person_names, ORG->organization_names, etc."""
        engine = SpaCyNER()
        mock_doc = self._make_mock_doc()
        mock_nlp = MagicMock(return_value=mock_doc)
        engine._nlp = mock_nlp

        result = engine.extract_entities("Sample text")

        assert "person_names" in result
        assert "organization_names" in result
        assert "dates" in result
        assert "monetary_amounts" in result
        assert "locations" in result

        assert ("John Doe", 0.8) in result["person_names"]
        assert ("Acme Corp", 0.8) in result["organization_names"]
        assert ("January 15, 2024", 0.8) in result["dates"]
        assert ("$50,000", 0.8) in result["monetary_amounts"]
        assert ("New York", 0.8) in result["locations"]

    @patch("src.extractors.ner_engines.spacy_ner._SPACY_AVAILABLE", True)
    def test_deduplicates_entities(self):
        """SpaCyNER does not produce duplicate values for same entity type."""

        class MockEntity:
            def __init__(self, text, label_):
                self.text = text
                self.label_ = label_
                self.kb_id_ = ""

        mock_doc = MagicMock()
        mock_doc.ents = [
            MockEntity("John Doe", "PERSON"),
            MockEntity("John Doe", "PERSON"),
        ]

        engine = SpaCyNER()
        engine._nlp = MagicMock(return_value=mock_doc)

        result = engine.extract_entities("John Doe met with John Doe")

        assert len(result["person_names"]) == 1

    @patch("src.extractors.ner_engines.spacy_ner._SPACY_AVAILABLE", False)
    def test_is_available_false_when_spacy_missing(self):
        """is_available() returns False when spaCy is not installed."""
        engine = SpaCyNER()
        assert engine.is_available() is False

    @patch("src.extractors.ner_engines.spacy_ner._SPACY_AVAILABLE", False)
    def test_returns_empty_when_unavailable(self):
        """extract_entities returns empty dict when engine is unavailable."""
        engine = SpaCyNER()
        result = engine.extract_entities("Some text")
        assert result == {}

    @patch("src.extractors.ner_engines.spacy_ner._SPACY_AVAILABLE", True)
    def test_get_name(self):
        """get_name() returns descriptive engine name."""
        engine = SpaCyNER(model_name="en_core_web_lg")
        assert engine.get_name() == "spacy-en_core_web_lg"


# --- HuggingFaceNER tests ---


class TestHuggingFaceNER:
    """Test HuggingFaceNER engine with mocked transformers pipeline."""

    def _make_mock_pipeline_output(self):
        """Create mock HuggingFace NER pipeline output."""
        return [
            {"entity_group": "PER", "word": "John Smith", "score": 0.98},
            {"entity_group": "ORG", "word": "Microsoft", "score": 0.95},
            {"entity_group": "LOC", "word": "Seattle", "score": 0.92},
            {"entity_group": "PER", "word": "Jane Doe", "score": 0.88},
            {"entity_group": "MISC", "word": "Python", "score": 0.75},
        ]

    @patch("src.extractors.ner_engines.huggingface_ner._TRANSFORMERS_AVAILABLE", True)
    def test_extract_entities_maps_labels_correctly(self):
        """HuggingFaceNER maps B-PER->person_names, B-ORG->organization_names, etc."""
        engine = HuggingFaceNER()
        mock_output = self._make_mock_pipeline_output()
        engine._pipeline = MagicMock(return_value=mock_output)

        result = engine.extract_entities("John Smith works at Microsoft in Seattle")

        assert "person_names" in result
        assert "organization_names" in result
        assert "locations" in result

        person_values = [v for v, _ in result["person_names"]]
        assert "John Smith" in person_values
        assert "Jane Doe" in person_values

        org_values = [v for v, _ in result["organization_names"]]
        assert "Microsoft" in org_values

        loc_values = [v for v, _ in result["locations"]]
        assert "Seattle" in loc_values

    @patch("src.extractors.ner_engines.huggingface_ner._TRANSFORMERS_AVAILABLE", True)
    def test_uses_pipeline_confidence_scores(self):
        """HuggingFaceNER uses confidence scores from pipeline output."""
        engine = HuggingFaceNER()
        engine._pipeline = MagicMock(
            return_value=[
                {"entity_group": "PER", "word": "Alice", "score": 0.97},
            ]
        )

        result = engine.extract_entities("Alice went home")

        assert result["person_names"][0] == ("Alice", 0.97)

    @patch("src.extractors.ner_engines.huggingface_ner._TRANSFORMERS_AVAILABLE", True)
    def test_handles_bio_prefixed_labels(self):
        """HuggingFaceNER strips B-/I- prefixes from labels."""
        engine = HuggingFaceNER()
        engine._pipeline = MagicMock(
            return_value=[
                {"entity_group": "B-PER", "word": "Bob", "score": 0.9},
                {"entity_group": "I-ORG", "word": "Corp", "score": 0.85},
            ]
        )

        result = engine.extract_entities("Bob at Corp")

        # After stripping B-/I-, PER maps to person_names, ORG to organization_names
        assert "person_names" in result
        assert "organization_names" in result

    @patch("src.extractors.ner_engines.huggingface_ner._TRANSFORMERS_AVAILABLE", False)
    def test_is_available_false_when_transformers_missing(self):
        """is_available() returns False when transformers is not installed."""
        engine = HuggingFaceNER()
        assert engine.is_available() is False

    @patch("src.extractors.ner_engines.huggingface_ner._TRANSFORMERS_AVAILABLE", False)
    def test_returns_empty_when_unavailable(self):
        """extract_entities returns empty dict when engine is unavailable."""
        engine = HuggingFaceNER()
        result = engine.extract_entities("Some text")
        assert result == {}

    @patch("src.extractors.ner_engines.huggingface_ner._TRANSFORMERS_AVAILABLE", True)
    def test_handles_pipeline_error_gracefully(self):
        """HuggingFaceNER handles pipeline errors without crashing."""
        engine = HuggingFaceNER()
        engine._pipeline = MagicMock(side_effect=RuntimeError("Model error"))

        result = engine.extract_entities("Some text")
        assert result == {}

    @patch("src.extractors.ner_engines.huggingface_ner._TRANSFORMERS_AVAILABLE", True)
    def test_get_name(self):
        """get_name() returns descriptive engine name."""
        engine = HuggingFaceNER(model_name="dslim/bert-base-NER")
        assert engine.get_name() == "huggingface-dslim/bert-base-NER"


# --- LayoutLMEngine tests ---


class TestLayoutLMEngine:
    """Test LayoutLMEngine with mocked transformers."""

    def _make_mock_pipeline_output(self):
        """Create mock LayoutLM pipeline output."""
        return [
            {"entity_group": "PER", "word": "Alice Johnson", "score": 0.93},
            {"entity_group": "ORG", "word": "Widgets Inc", "score": 0.89},
            {"entity_group": "B-ANSWER", "word": "$1000.00", "score": 0.95},
        ]

    @patch("src.extractors.ner_engines.layoutlm_engine._TRANSFORMERS_AVAILABLE", True)
    @patch("src.extractors.ner_engines.layoutlm_engine._TORCH_AVAILABLE", True)
    def test_extract_entities_text_only_fallback(self):
        """LayoutLMEngine handles text-only fallback without layout info."""
        engine = LayoutLMEngine()
        mock_output = self._make_mock_pipeline_output()
        engine._pipeline = MagicMock(return_value=mock_output)

        result = engine.extract_entities("Alice Johnson at Widgets Inc owes $1000.00")

        assert isinstance(result, dict)
        assert "person_names" in result
        assert "organization_names" in result

    @patch("src.extractors.ner_engines.layoutlm_engine._TRANSFORMERS_AVAILABLE", True)
    @patch("src.extractors.ner_engines.layoutlm_engine._TORCH_AVAILABLE", True)
    def test_extract_with_layout(self):
        """LayoutLMEngine can process text with bounding box info."""
        engine = LayoutLMEngine()
        mock_output = [
            {"entity_group": "PER", "word": "Bob", "score": 0.91},
        ]
        engine._pipeline = MagicMock(return_value=mock_output)

        bboxes = [[0, 0, 100, 50]]
        result = engine.extract_with_layout("Bob", bboxes)

        assert "person_names" in result
        assert ("Bob", 0.91) in result["person_names"]

    @patch("src.extractors.ner_engines.layoutlm_engine._TRANSFORMERS_AVAILABLE", True)
    @patch("src.extractors.ner_engines.layoutlm_engine._TORCH_AVAILABLE", False)
    def test_is_available_false_when_torch_missing(self):
        """is_available() returns False when torch is not installed."""
        engine = LayoutLMEngine()
        assert engine.is_available() is False

    @patch("src.extractors.ner_engines.layoutlm_engine._TRANSFORMERS_AVAILABLE", False)
    @patch("src.extractors.ner_engines.layoutlm_engine._TORCH_AVAILABLE", True)
    def test_is_available_false_when_transformers_missing(self):
        """is_available() returns False when transformers is not installed."""
        engine = LayoutLMEngine()
        assert engine.is_available() is False

    @patch("src.extractors.ner_engines.layoutlm_engine._TRANSFORMERS_AVAILABLE", False)
    @patch("src.extractors.ner_engines.layoutlm_engine._TORCH_AVAILABLE", False)
    def test_returns_empty_when_unavailable(self):
        """extract_entities returns empty dict when engine is unavailable."""
        engine = LayoutLMEngine()
        result = engine.extract_entities("Some text")
        assert result == {}

    @patch("src.extractors.ner_engines.layoutlm_engine._TRANSFORMERS_AVAILABLE", True)
    @patch("src.extractors.ner_engines.layoutlm_engine._TORCH_AVAILABLE", True)
    def test_handles_pipeline_error_gracefully(self):
        """LayoutLMEngine handles pipeline errors without crashing."""
        engine = LayoutLMEngine()
        engine._pipeline = MagicMock(side_effect=RuntimeError("Model error"))

        result = engine.extract_entities("Some text")
        assert result == {}

    @patch("src.extractors.ner_engines.layoutlm_engine._TRANSFORMERS_AVAILABLE", True)
    @patch("src.extractors.ner_engines.layoutlm_engine._TORCH_AVAILABLE", True)
    def test_get_name(self):
        """get_name() returns descriptive engine name."""
        engine = LayoutLMEngine()
        assert "layoutlm" in engine.get_name()


# --- NERensemble tests ---


class TestNERensemble:
    """Test NER ensemble with confidence-weighted merging."""

    def _make_mock_engine(self, name, available, entities):
        """Create a mock NER engine with specified behavior."""
        engine = MagicMock(spec=BaseNEREngine)
        engine.get_name.return_value = name
        engine.is_available.return_value = available
        engine.extract_entities.return_value = entities
        return engine

    def test_merges_results_from_multiple_engines(self):
        """NERensemble merges results from multiple engines."""
        engine1 = self._make_mock_engine(
            "engine1",
            True,
            {"person_names": [("John Doe", 0.9)]},
        )
        engine2 = self._make_mock_engine(
            "engine2",
            True,
            {"person_names": [("John Doe", 0.85)], "locations": [("NYC", 0.8)]},
        )

        ensemble = NERensemble(engines=[engine1, engine2])
        result = ensemble.extract_entities("John Doe in NYC")

        assert "person_names" in result
        assert "locations" in result

        # Both engines found John Doe - should be merged
        person_values = [v for v, _ in result["person_names"]]
        assert "John Doe" in person_values

    def test_confidence_weighted_merging(self):
        """Ensemble uses confidence weighting for merged results."""
        engine1 = self._make_mock_engine(
            "engine1",
            True,
            {"person_names": [("Alice", 0.95)]},
        )
        engine2 = self._make_mock_engine(
            "engine2",
            True,
            {"person_names": [("Alice", 0.6)]},
        )

        ensemble = NERensemble(
            engines=[engine1, engine2],
            weights={"engine1": 1.5, "engine2": 0.5},
        )
        result = ensemble.extract_entities("Alice went home")

        assert "person_names" in result
        # Alice should appear with confidence boosted by multi-engine agreement
        alice_conf = [c for v, c in result["person_names"] if v == "Alice"]
        assert len(alice_conf) == 1
        assert alice_conf[0] > 0.5

    def test_handles_unavailable_engines_gracefully(self):
        """Ensemble skips engines that are unavailable."""
        available_engine = self._make_mock_engine(
            "available",
            True,
            {"person_names": [("Bob", 0.9)]},
        )
        unavailable_engine = self._make_mock_engine(
            "unavailable",
            False,
            {},
        )

        ensemble = NERensemble(engines=[available_engine, unavailable_engine])
        result = ensemble.extract_entities("Bob was here")

        assert "person_names" in result
        unavailable_engine.extract_entities.assert_not_called()

    def test_handles_engine_failure_gracefully(self):
        """Ensemble handles engines that throw exceptions."""
        good_engine = self._make_mock_engine(
            "good",
            True,
            {"person_names": [("Charlie", 0.85)]},
        )
        bad_engine = MagicMock(spec=BaseNEREngine)
        bad_engine.get_name.return_value = "bad"
        bad_engine.is_available.return_value = True
        bad_engine.extract_entities.side_effect = RuntimeError("Crash!")

        ensemble = NERensemble(engines=[good_engine, bad_engine])
        result = ensemble.extract_entities("Charlie is fine")

        assert "person_names" in result
        person_values = [v for v, _ in result["person_names"]]
        assert "Charlie" in person_values

    def test_respects_confidence_threshold(self):
        """Ensemble filters out entities below confidence threshold."""
        engine = self._make_mock_engine(
            "engine",
            True,
            {
                "person_names": [("Alice", 0.9)],
                "locations": [("X", 0.1)],
            },
        )

        ensemble = NERensemble(engines=[engine], confidence_threshold=0.5)
        result = ensemble.extract_entities("Alice in X")

        assert "person_names" in result
        # Low confidence entity should be filtered out
        assert "locations" not in result

    def test_returns_empty_when_no_engines_available(self):
        """Ensemble returns empty dict when no engines are available."""
        engine = self._make_mock_engine("unavailable", False, {})
        ensemble = NERensemble(engines=[engine])
        result = ensemble.extract_entities("Some text")
        assert result == {}

    def test_returns_empty_when_no_engines(self):
        """Ensemble returns empty dict when initialized with no engines."""
        ensemble = NERensemble(engines=[])
        result = ensemble.extract_entities("Some text")
        assert result == {}

    def test_groups_similar_values(self):
        """Ensemble groups similar entity values from different engines."""
        engine1 = self._make_mock_engine(
            "engine1",
            True,
            {"person_names": [("John Doe", 0.9)]},
        )
        engine2 = self._make_mock_engine(
            "engine2",
            True,
            {"person_names": [("John Doe", 0.85)]},
        )

        ensemble = NERensemble(engines=[engine1, engine2])
        result = ensemble.extract_entities("John Doe")

        # Same name from two engines should be merged into one entry
        assert len(result["person_names"]) == 1

    def test_add_engine(self):
        """Can add engines to ensemble after initialization."""
        ensemble = NERensemble(engines=[])
        engine = self._make_mock_engine(
            "added", True, {"person_names": [("Dave", 0.9)]}
        )
        ensemble.add_engine(engine)

        result = ensemble.extract_entities("Dave")
        assert "person_names" in result


# --- Integration tests with LocalExtractor ---


class TestLocalExtractorEnsemble:
    """Test LocalExtractor ensemble mode integration."""

    @patch("src.extractors.local_extractor.get_settings")
    def test_default_does_not_use_ensemble(self, mock_settings):
        """LocalExtractor defaults to use_ensemble=False."""
        mock_settings.return_value = MagicMock(
            spacy_model="en_core_web_sm", confidence_threshold=0.7
        )
        from src.extractors.local_extractor import LocalExtractor

        extractor = LocalExtractor()
        assert extractor._use_ensemble is False

    @patch("src.extractors.local_extractor.get_settings")
    def test_ensemble_mode_can_be_enabled(self, mock_settings):
        """LocalExtractor can be initialized with use_ensemble=True."""
        mock_settings.return_value = MagicMock(
            spacy_model="en_core_web_sm", confidence_threshold=0.7
        )
        from src.extractors.local_extractor import LocalExtractor

        extractor = LocalExtractor(use_ensemble=True)
        assert extractor._use_ensemble is True


# --- Registration tests ---


class TestNEREngineRegistration:
    """Test NER engine registration in FreeModelRegistry."""

    def test_register_ner_engines(self):
        """All NER engines can be registered in the model registry."""
        from src.models.model_registry import FreeModelRegistry, ModelType

        # Reset registry for clean test
        FreeModelRegistry.reset_instance()
        registry = FreeModelRegistry()

        from src.extractors.ner_engines import register_ner_engines

        register_ner_engines()

        # Check engines were registered
        all_models = registry.list_all_models()
        model_names = [m.name for m in all_models]

        assert "spacy-ner" in model_names
        assert "huggingface-ner" in model_names
        assert "layoutlmv3" in model_names

        # Check types
        spacy_model = registry.get_model("spacy-ner")
        assert spacy_model.model_type == ModelType.NER

        layoutlm_model = registry.get_model("layoutlmv3")
        assert layoutlm_model.model_type == ModelType.LAYOUT

        # Cleanup
        FreeModelRegistry.reset_instance()
