"""Tests for OCR engines, ensemble, and registry integration."""

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.processors.ocr_engines.base_ocr import BaseOCREngine
from src.processors.ocr_engines.tesseract_ocr import TesseractOCR
from src.processors.ocr_engines.trocr_engine import TrOCREngine
from src.processors.ocr_engines.paddle_ocr_engine import PaddleOCREngine
from src.processors.ocr_engines.doctr_engine import DocTREngine
from src.processors.ocr_ensemble import OCREnsemble


# --- BaseOCREngine interface tests ---


class ConcreteOCREngine(BaseOCREngine):
    """Concrete implementation for testing the abstract base class."""

    def extract_text(self, image_path: str):
        return "test text", 0.95

    def is_available(self) -> bool:
        return True

    def get_name(self) -> str:
        return "concrete_test"


def test_base_ocr_engine_cannot_be_instantiated():
    """Test that BaseOCREngine cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseOCREngine()


def test_concrete_engine_implements_interface():
    """Test that a concrete implementation works properly."""
    engine = ConcreteOCREngine()
    text, confidence = engine.extract_text("test.png")
    assert text == "test text"
    assert confidence == 0.95
    assert engine.is_available() is True
    assert engine.get_name() == "concrete_test"


# --- TesseractOCR tests ---


def test_tesseract_get_name():
    """Test TesseractOCR returns correct name."""
    engine = TesseractOCR()
    assert engine.get_name() == "tesseract"


@patch("src.processors.ocr_engines.tesseract_ocr.pytesseract", create=True)
def test_tesseract_is_available_when_importable(mock_pytesseract):
    """Test TesseractOCR.is_available() when pytesseract is importable."""
    engine = TesseractOCR()
    # The is_available method tries to import pytesseract
    with patch.dict("sys.modules", {"pytesseract": MagicMock()}):
        result = engine.is_available()
    assert result is True


def test_tesseract_is_available_when_not_importable():
    """Test TesseractOCR.is_available() when pytesseract is not importable."""
    engine = TesseractOCR()
    with patch.dict("sys.modules", {"pytesseract": None}):
        with patch("builtins.__import__", side_effect=ImportError):
            result = engine.is_available()
    assert result is False


@patch("pytesseract.image_to_string", return_value="Hello World")
@patch(
    "pytesseract.image_to_data",
    return_value={"conf": ["95", "88", "92", "-1"]},
)
@patch("PIL.Image.open")
def test_tesseract_extract_text(mock_open, mock_data, mock_string):
    """Test TesseractOCR.extract_text() returns proper tuple."""
    mock_image = MagicMock()
    mock_image.mode = "RGB"
    mock_image.convert.return_value = mock_image
    mock_image.filter.return_value = mock_image
    mock_image.point.return_value = mock_image
    mock_open.return_value = mock_image

    # Mock ImageEnhance
    with patch("PIL.ImageEnhance.Contrast") as mock_contrast:
        mock_enhancer = MagicMock()
        mock_enhancer.enhance.return_value = mock_image
        mock_contrast.return_value = mock_enhancer

        engine = TesseractOCR()
        text, confidence = engine.extract_text("test.png")

    assert text == "Hello World"
    # (95 + 88 + 92) / 3 / 100 = 0.9166...
    assert abs(confidence - 0.9167) < 0.01


# --- TrOCREngine tests ---


def test_trocr_get_name_printed():
    """Test TrOCREngine returns correct name for printed mode."""
    engine = TrOCREngine(mode="printed")
    assert engine.get_name() == "trocr-printed"


def test_trocr_get_name_handwritten():
    """Test TrOCREngine returns correct name for handwritten mode."""
    engine = TrOCREngine(mode="handwritten")
    assert engine.get_name() == "trocr-handwritten"


def test_trocr_model_name_printed():
    """Test TrOCREngine selects correct model for printed mode."""
    engine = TrOCREngine(mode="printed")
    assert engine.model_name == "microsoft/trocr-base-printed"


def test_trocr_model_name_handwritten():
    """Test TrOCREngine selects correct model for handwritten mode."""
    engine = TrOCREngine(mode="handwritten")
    assert engine.model_name == "microsoft/trocr-base-handwritten"


def test_trocr_is_available_when_deps_present():
    """Test TrOCREngine.is_available() with both deps available."""
    with patch(
        "src.processors.ocr_engines.trocr_engine._TRANSFORMERS_AVAILABLE", True
    ):
        with patch(
            "src.processors.ocr_engines.trocr_engine._TORCH_AVAILABLE", True
        ):
            engine = TrOCREngine()
            assert engine.is_available() is True


def test_trocr_is_available_without_transformers():
    """Test TrOCREngine.is_available() without transformers."""
    with patch(
        "src.processors.ocr_engines.trocr_engine._TRANSFORMERS_AVAILABLE", False
    ):
        with patch(
            "src.processors.ocr_engines.trocr_engine._TORCH_AVAILABLE", True
        ):
            engine = TrOCREngine()
            assert engine.is_available() is False


def test_trocr_is_available_without_torch():
    """Test TrOCREngine.is_available() without torch."""
    with patch(
        "src.processors.ocr_engines.trocr_engine._TRANSFORMERS_AVAILABLE", True
    ):
        with patch(
            "src.processors.ocr_engines.trocr_engine._TORCH_AVAILABLE", False
        ):
            engine = TrOCREngine()
            assert engine.is_available() is False


@patch("src.processors.ocr_engines.trocr_engine.TrOCRProcessor")
@patch("src.processors.ocr_engines.trocr_engine.VisionEncoderDecoderModel")
@patch("PIL.Image.open")
def test_trocr_extract_text(mock_open, mock_model_class, mock_processor_class):
    """Test TrOCREngine.extract_text() with mocked transformers."""
    import torch

    # Setup mock image
    mock_image = MagicMock()
    mock_image.convert.return_value = mock_image
    mock_image.width = 100
    mock_image.crop.return_value = mock_image
    mock_open.return_value = mock_image

    # Mock numpy for line splitting - return whole image as one line
    with patch("numpy.array") as mock_np_array:
        mock_np_array.return_value = MagicMock()
        mock_np_array.return_value.__lt__ = MagicMock(return_value=MagicMock())

        # Mock processor
        mock_processor = MagicMock()
        mock_pixel_values = MagicMock()
        mock_pixel_values.pixel_values = torch.zeros(1, 3, 16, 16)
        mock_processor.return_value = mock_pixel_values
        mock_processor.batch_decode.return_value = ["Hello World"]
        mock_processor_class.from_pretrained.return_value = mock_processor

        # Mock model
        mock_model = MagicMock()
        mock_scores = [torch.randn(1, 100) for _ in range(5)]
        mock_outputs = MagicMock()
        mock_outputs.sequences = torch.tensor([[1, 2, 3]])
        mock_outputs.scores = mock_scores
        mock_model.generate.return_value = mock_outputs
        mock_model_class.from_pretrained.return_value = mock_model

        engine = TrOCREngine(mode="printed")
        # Directly test _process_line since line splitting is complex
        engine._processor = mock_processor
        engine._model = mock_model

        text, confidence = engine._process_line(mock_image)

    assert text == "Hello World"
    assert 0.0 <= confidence <= 1.0


# --- PaddleOCREngine tests ---


def test_paddleocr_get_name():
    """Test PaddleOCREngine returns correct name."""
    engine = PaddleOCREngine()
    assert engine.get_name() == "paddleocr"


def test_paddleocr_is_available_when_importable():
    """Test PaddleOCREngine.is_available() when paddleocr is importable."""
    with patch(
        "src.processors.ocr_engines.paddle_ocr_engine._PADDLEOCR_AVAILABLE", True
    ):
        engine = PaddleOCREngine()
        assert engine.is_available() is True


def test_paddleocr_is_available_when_not_importable():
    """Test PaddleOCREngine.is_available() when paddleocr is not importable."""
    with patch(
        "src.processors.ocr_engines.paddle_ocr_engine._PADDLEOCR_AVAILABLE", False
    ):
        engine = PaddleOCREngine()
        assert engine.is_available() is False


@patch("src.processors.ocr_engines.paddle_ocr_engine.PaddleOCR", create=True)
def test_paddleocr_extract_text(mock_paddleocr_class):
    """Test PaddleOCREngine.extract_text() with mocked PaddleOCR."""
    # Mock PaddleOCR result format
    mock_result = [
        [
            [[[0, 0], [100, 0], [100, 20], [0, 20]], ("Hello", 0.95)],
            [[[0, 25], [100, 25], [100, 45], [0, 45]], ("World", 0.88)],
        ]
    ]

    mock_ocr_instance = MagicMock()
    mock_ocr_instance.ocr.return_value = mock_result
    mock_paddleocr_class.return_value = mock_ocr_instance

    with patch(
        "src.processors.ocr_engines.paddle_ocr_engine._PADDLEOCR_AVAILABLE", True
    ):
        engine = PaddleOCREngine()
        engine._ocr = mock_ocr_instance
        text, confidence = engine.extract_text("test.png")

    assert "Hello" in text
    assert "World" in text
    # (0.95 + 0.88) / 2 = 0.915
    assert abs(confidence - 0.915) < 0.01


@patch("src.processors.ocr_engines.paddle_ocr_engine.PaddleOCR", create=True)
def test_paddleocr_extract_text_empty_result(mock_paddleocr_class):
    """Test PaddleOCREngine.extract_text() with empty result."""
    mock_ocr_instance = MagicMock()
    mock_ocr_instance.ocr.return_value = [None]
    mock_paddleocr_class.return_value = mock_ocr_instance

    with patch(
        "src.processors.ocr_engines.paddle_ocr_engine._PADDLEOCR_AVAILABLE", True
    ):
        engine = PaddleOCREngine()
        engine._ocr = mock_ocr_instance
        text, confidence = engine.extract_text("test.png")

    assert text == ""
    assert confidence == 0.0


# --- DocTREngine tests ---


def test_doctr_get_name():
    """Test DocTREngine returns correct name."""
    engine = DocTREngine()
    assert engine.get_name() == "doctr"


def test_doctr_is_available_when_importable():
    """Test DocTREngine.is_available() when doctr is importable."""
    with patch(
        "src.processors.ocr_engines.doctr_engine._DOCTR_AVAILABLE", True
    ):
        engine = DocTREngine()
        assert engine.is_available() is True


def test_doctr_is_available_when_not_importable():
    """Test DocTREngine.is_available() when doctr is not importable."""
    with patch(
        "src.processors.ocr_engines.doctr_engine._DOCTR_AVAILABLE", False
    ):
        engine = DocTREngine()
        assert engine.is_available() is False


@patch("src.processors.ocr_engines.doctr_engine.ocr_predictor", create=True)
def test_doctr_extract_text(mock_predictor_fn):
    """Test DocTREngine.extract_text() with mocked doctr."""
    # Build mock result structure
    mock_word1 = MagicMock()
    mock_word1.value = "Hello"
    mock_word1.confidence = 0.92

    mock_word2 = MagicMock()
    mock_word2.value = "World"
    mock_word2.confidence = 0.89

    mock_line = MagicMock()
    mock_line.words = [mock_word1, mock_word2]

    mock_block = MagicMock()
    mock_block.lines = [mock_line]

    mock_page = MagicMock()
    mock_page.blocks = [mock_block]

    mock_result = MagicMock()
    mock_result.pages = [mock_page]

    mock_predictor = MagicMock()
    mock_predictor.return_value = mock_result
    mock_predictor_fn.return_value = mock_predictor

    with patch(
        "src.processors.ocr_engines.doctr_engine._DOCTR_AVAILABLE", True
    ):
        # Mock doctr.io.DocumentFile which is imported inside extract_text
        mock_doctr_io = MagicMock()
        mock_doctr_io.DocumentFile.from_images.return_value = "mock_doc"
        with patch.dict("sys.modules", {"doctr": MagicMock(), "doctr.io": mock_doctr_io}):
            engine = DocTREngine()
            engine._predictor = mock_predictor
            text, confidence = engine.extract_text("test.png")

    assert "Hello" in text
    assert "World" in text
    # (0.92 + 0.89) / 2 = 0.905
    assert abs(confidence - 0.905) < 0.01


# --- OCREnsemble tests ---


class MockEngine(BaseOCREngine):
    """Mock OCR engine for ensemble testing."""

    def __init__(self, name: str, text: str, confidence: float, available: bool = True):
        self._name = name
        self._text = text
        self._confidence = confidence
        self._available = available

    def extract_text(self, image_path: str):
        return self._text, self._confidence

    def is_available(self) -> bool:
        return self._available

    def get_name(self) -> str:
        return self._name


class FailingEngine(BaseOCREngine):
    """Mock OCR engine that raises an exception."""

    def extract_text(self, image_path: str):
        raise RuntimeError("Engine failed")

    def is_available(self) -> bool:
        return True

    def get_name(self) -> str:
        return "failing_engine"


def test_ensemble_no_engines():
    """Test ensemble with no engines returns empty result."""
    ensemble = OCREnsemble(engines=[])
    text, confidence = ensemble.extract_text("test.png")
    assert text == ""
    assert confidence == 0.0


def test_ensemble_no_available_engines():
    """Test ensemble when no engines are available."""
    engine = MockEngine("unavailable", "text", 0.9, available=False)
    ensemble = OCREnsemble(engines=[engine])
    text, confidence = ensemble.extract_text("test.png")
    assert text == ""
    assert confidence == 0.0


def test_ensemble_single_engine():
    """Test ensemble with a single engine returns its result."""
    engine = MockEngine("test", "Hello World", 0.9)
    ensemble = OCREnsemble(engines=[engine])
    text, confidence = ensemble.extract_text("test.png")
    assert text == "Hello World"
    assert confidence == 0.9


def test_ensemble_similar_results_boost_confidence():
    """Test ensemble boosts confidence when engines agree."""
    engine1 = MockEngine("engine1", "Hello World", 0.9)
    engine2 = MockEngine("engine2", "Hello World", 0.85)
    engine3 = MockEngine("engine3", "Hello World", 0.8)

    ensemble = OCREnsemble(engines=[engine1, engine2, engine3])
    text, confidence = ensemble.extract_text("test.png")

    assert text == "Hello World"
    # Confidence should be boosted above best single engine
    assert confidence > 0.9


def test_ensemble_different_results_uses_highest_confidence():
    """Test ensemble uses highest confidence when texts differ significantly."""
    engine1 = MockEngine("engine1", "Hello World", 0.9)
    engine2 = MockEngine("engine2", "completely different text here", 0.5)

    ensemble = OCREnsemble(
        engines=[engine1, engine2], similarity_threshold=0.9
    )
    text, confidence = ensemble.extract_text("test.png")

    assert text == "Hello World"
    assert confidence == 0.9


def test_ensemble_fallback_longest_text():
    """Test ensemble with longest_text fallback strategy."""
    engine1 = MockEngine("engine1", "Short", 0.95)
    engine2 = MockEngine("engine2", "This is a much longer text result", 0.7)

    ensemble = OCREnsemble(
        engines=[engine1, engine2],
        similarity_threshold=0.9,
        fallback_strategy="longest_text",
    )
    text, confidence = ensemble.extract_text("test.png")

    assert text == "This is a much longer text result"
    assert confidence == 0.7


def test_ensemble_handles_engine_failure_gracefully():
    """Test ensemble handles engine exceptions gracefully."""
    failing = FailingEngine()
    working = MockEngine("working", "Hello", 0.8)

    ensemble = OCREnsemble(engines=[failing, working])
    text, confidence = ensemble.extract_text("test.png")

    assert text == "Hello"
    assert confidence == 0.8


def test_ensemble_all_engines_fail():
    """Test ensemble when all engines fail."""
    failing1 = FailingEngine()
    failing2 = FailingEngine()

    ensemble = OCREnsemble(engines=[failing1, failing2])
    text, confidence = ensemble.extract_text("test.png")

    assert text == ""
    assert confidence == 0.0


def test_ensemble_get_available_engines():
    """Test get_available_engines filters unavailable ones."""
    available = MockEngine("available", "text", 0.9, available=True)
    unavailable = MockEngine("unavailable", "text", 0.9, available=False)

    ensemble = OCREnsemble(engines=[available, unavailable])
    engines = ensemble.get_available_engines()

    assert len(engines) == 1
    assert engines[0].get_name() == "available"


def test_ensemble_get_engine_status():
    """Test get_engine_status returns status dict."""
    engine1 = MockEngine("engine1", "text", 0.9, available=True)
    engine2 = MockEngine("engine2", "text", 0.9, available=False)

    ensemble = OCREnsemble(engines=[engine1, engine2])
    status = ensemble.get_engine_status()

    assert status == {"engine1": True, "engine2": False}


def test_ensemble_add_engine():
    """Test adding an engine to the ensemble."""
    ensemble = OCREnsemble(engines=[])
    engine = MockEngine("new_engine", "text", 0.9)
    ensemble.add_engine(engine)

    assert len(ensemble.engines) == 1
    assert ensemble.engines[0].get_name() == "new_engine"


def test_ensemble_min_engines_warning():
    """Test ensemble logs warning when fewer than min_engines succeed."""
    engine = MockEngine("engine1", "Hello", 0.9)
    ensemble = OCREnsemble(engines=[engine], min_engines=3)

    # Should still return the result even with fewer than min_engines
    text, confidence = ensemble.extract_text("test.png")
    assert text == "Hello"
    assert confidence == 0.9


def test_ensemble_empty_text_excluded():
    """Test that engines returning empty text are excluded from merging."""
    empty_engine = MockEngine("empty", "", 0.9)
    working_engine = MockEngine("working", "Real text", 0.8)

    ensemble = OCREnsemble(engines=[empty_engine, working_engine])
    text, confidence = ensemble.extract_text("test.png")

    assert text == "Real text"
    assert confidence == 0.8


# --- Factory function tests ---


def test_get_ocr_engine_tesseract():
    """Test factory function returns TesseractOCR."""
    from src.processors.ocr_engine import get_ocr_engine

    engine = get_ocr_engine("tesseract")
    assert isinstance(engine, TesseractOCR)


def test_get_ocr_engine_trocr():
    """Test factory function returns TrOCREngine."""
    from src.processors.ocr_engine import get_ocr_engine

    engine = get_ocr_engine("trocr")
    assert isinstance(engine, TrOCREngine)


def test_get_ocr_engine_paddleocr():
    """Test factory function returns PaddleOCREngine."""
    from src.processors.ocr_engine import get_ocr_engine

    engine = get_ocr_engine("paddleocr")
    assert isinstance(engine, PaddleOCREngine)


def test_get_ocr_engine_doctr():
    """Test factory function returns DocTREngine."""
    from src.processors.ocr_engine import get_ocr_engine

    engine = get_ocr_engine("doctr")
    assert isinstance(engine, DocTREngine)


def test_get_ocr_engine_invalid():
    """Test factory function raises ValueError for unknown engine."""
    from src.processors.ocr_engine import get_ocr_engine

    with pytest.raises(ValueError, match="Unknown OCR engine"):
        get_ocr_engine("nonexistent")


def test_get_ocr_ensemble():
    """Test ensemble factory function creates ensemble with engines."""
    from src.processors.ocr_engine import get_ocr_ensemble

    ensemble = get_ocr_ensemble(engine_names=["tesseract", "trocr"])
    assert isinstance(ensemble, OCREnsemble)
    assert len(ensemble.engines) == 2


def test_get_ocr_ensemble_default():
    """Test ensemble factory function with default engines."""
    from src.processors.ocr_engine import get_ocr_ensemble

    ensemble = get_ocr_ensemble()
    assert isinstance(ensemble, OCREnsemble)
    assert len(ensemble.engines) == 4  # all engines


# --- Registry integration tests ---


def test_register_ocr_engines():
    """Test that OCR engines are registered in FreeModelRegistry."""
    from src.models.model_registry import FreeModelRegistry, ModelType

    FreeModelRegistry.reset_instance()

    from src.processors.ocr_engines import register_ocr_engines

    registry = FreeModelRegistry()
    register_ocr_engines()

    ocr_models = registry.list_all_models()
    ocr_names = {m.name for m in ocr_models if m.model_type == ModelType.OCR}

    assert "tesseract" in ocr_names
    assert "trocr-printed" in ocr_names
    assert "trocr-handwritten" in ocr_names
    assert "paddleocr" in ocr_names
    assert "doctr" in ocr_names

    FreeModelRegistry.reset_instance()


# --- Backward compatibility tests ---


def test_original_ocr_engine_still_works():
    """Test that the original OCREngine class is still importable and functional."""
    from src.processors.ocr_engine import OCREngine

    engine = OCREngine()
    assert hasattr(engine, "extract_text")
    assert hasattr(engine, "preprocess_image")
    assert hasattr(engine, "detect_language")
