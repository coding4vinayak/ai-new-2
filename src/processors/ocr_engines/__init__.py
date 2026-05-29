"""OCR engines package with multiple backend support.

Exports all OCR engine classes for use throughout the application.
"""

from src.processors.ocr_engines.base_ocr import BaseOCREngine
from src.processors.ocr_engines.doctr_engine import DocTREngine
from src.processors.ocr_engines.paddle_ocr_engine import PaddleOCREngine
from src.processors.ocr_engines.tesseract_ocr import TesseractOCR
from src.processors.ocr_engines.trocr_engine import TrOCREngine

__all__ = [
    "BaseOCREngine",
    "TesseractOCR",
    "TrOCREngine",
    "PaddleOCREngine",
    "DocTREngine",
    "register_ocr_engines",
]


def register_ocr_engines() -> None:
    """Register all OCR engines in the FreeModelRegistry.

    Checks each engine's availability and sets the appropriate status.
    """
    from src.models.model_registry import (
        FreeModelRegistry,
        ModelStatus,
        ModelType,
    )

    registry = FreeModelRegistry()

    engines = [
        ("tesseract", TesseractOCR()),
        ("trocr-printed", TrOCREngine(mode="printed")),
        ("trocr-handwritten", TrOCREngine(mode="handwritten")),
        ("paddleocr", PaddleOCREngine()),
        ("doctr", DocTREngine()),
    ]

    for name, engine in engines:
        model = registry.register_model(
            name=name,
            model_type=ModelType.OCR,
            config={"engine_class": engine.__class__.__name__},
        )
        if engine.is_available():
            model.status = ModelStatus.AVAILABLE
        else:
            model.status = ModelStatus.UNAVAILABLE

