"""NER engines package with multiple backend support.

Exports all NER engine classes for use throughout the application.
"""

from src.extractors.ner_engines.base_ner import BaseNEREngine
from src.extractors.ner_engines.huggingface_ner import HuggingFaceNER
from src.extractors.ner_engines.layoutlm_engine import LayoutLMEngine
from src.extractors.ner_engines.spacy_ner import SpaCyNER

__all__ = [
    "BaseNEREngine",
    "SpaCyNER",
    "HuggingFaceNER",
    "LayoutLMEngine",
    "register_ner_engines",
]


def register_ner_engines() -> None:
    """Register all NER engines in the FreeModelRegistry.

    Checks each engine's availability and sets the appropriate status.
    """
    from src.models.model_registry import (
        FreeModelRegistry,
        ModelStatus,
        ModelType,
    )

    registry = FreeModelRegistry()

    engines = [
        ("spacy-ner", SpaCyNER(), ModelType.NER),
        ("huggingface-ner", HuggingFaceNER(), ModelType.NER),
        ("layoutlmv3", LayoutLMEngine(), ModelType.LAYOUT),
    ]

    for name, engine, model_type in engines:
        model = registry.register_model(
            name=name,
            model_type=model_type,
            config={"engine_class": engine.__class__.__name__},
        )
        if engine.is_available():
            model.status = ModelStatus.AVAILABLE
        else:
            model.status = ModelStatus.UNAVAILABLE
