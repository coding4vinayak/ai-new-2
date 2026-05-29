"""Document agent orchestrator coordinating the full extraction pipeline."""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.extractors.api_extractor import APIExtractor
from src.extractors.base import BaseExtractor, ExtractionMode
from src.extractors.hybrid_extractor import HybridExtractor
from src.extractors.local_extractor import LocalExtractor
from src.models.document import Document, FileType
from src.models.extraction_result import ExtractionResult
from src.processors.docx_processor import DocxProcessor
from src.processors.image_processor import ImageProcessor
from src.processors.pdf_processor import PDFProcessor
from src.utils.config import get_settings

logger = logging.getLogger(__name__)


class DocumentAgent:
    """Main orchestrator coordinating the document intelligence pipeline.

    This is the primary entry point for processing documents. It coordinates:
    1. Document type detection and processor selection
    2. Extraction mode selection and entity extraction
    3. Industry-specific analysis (if applicable)
    4. Action triggering based on extraction results
    """

    def __init__(self) -> None:
        """Initialize the document agent with processors and extractors."""
        self._settings = get_settings()

        # Processors
        self._pdf_processor = PDFProcessor()
        self._image_processor = ImageProcessor()
        self._docx_processor = DocxProcessor()

        # Extractors (lazy initialization)
        self._local_extractor: Optional[LocalExtractor] = None
        self._api_extractor: Optional[APIExtractor] = None
        self._hybrid_extractor: Optional[HybridExtractor] = None

        # Action engine (lazy import to avoid circular dependencies)
        self._action_engine = None

    def _get_local_extractor(self) -> LocalExtractor:
        """Get or create the local extractor instance."""
        if self._local_extractor is None:
            self._local_extractor = LocalExtractor()
        return self._local_extractor

    def _get_api_extractor(self) -> APIExtractor:
        """Get or create the API extractor instance."""
        if self._api_extractor is None:
            self._api_extractor = APIExtractor()
        return self._api_extractor

    def _get_hybrid_extractor(self) -> HybridExtractor:
        """Get or create the hybrid extractor instance."""
        if self._hybrid_extractor is None:
            self._hybrid_extractor = HybridExtractor(
                local_extractor=self._get_local_extractor(),
                api_extractor=self._get_api_extractor(),
            )
        return self._hybrid_extractor

    def _get_action_engine(self):
        """Get or create the action engine instance."""
        if self._action_engine is None:
            from src.agent.action_engine import ActionEngine

            self._action_engine = ActionEngine()
        return self._action_engine

    async def process_document(
        self,
        file_path: str,
        mode: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResult:
        """Process a document through the full extraction pipeline.

        Orchestrates: processor selection -> extraction -> industry analysis -> actions.

        Args:
            file_path: Path to the document file.
            mode: Extraction mode ('local', 'api', 'hybrid'). Defaults to config.
            options: Additional processing options (document_type, skip_actions, etc.)

        Returns:
            ExtractionResult with all extracted entities and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file type is not supported.
        """
        options = options or {}
        start_time = time.time()

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        # Step 1: Detect file type and select processor
        file_type = self._detect_file_type(path)
        processor = self._select_processor(file_type)

        # Step 2: Process the document to extract text
        logger.info(f"Processing document: {path.name} (type: {file_type.value})")
        document = processor.process(str(path))

        # Apply document type from options if provided
        if "document_type" in options:
            document.metadata["document_type"] = options["document_type"]

        # Step 3: Select extractor based on mode
        extraction_mode = mode or self._settings.extraction.get(
            "default_mode", "hybrid"
        ) if self._settings.extraction else "hybrid"
        extractor = self._select_extractor(extraction_mode)

        # Step 4: Extract entities
        logger.info(f"Extracting entities using mode: {extraction_mode}")
        result = await extractor.extract(document)

        # Step 5: Run industry analysis if applicable
        if not options.get("skip_analysis", False):
            result = await self._run_industry_analysis(result, options.get("document_type"))

        # Step 6: Trigger actions based on results
        if not options.get("skip_actions", False):
            await self._trigger_actions(result)

        logger.info(
            f"Document processed in {(time.time() - start_time) * 1000:.1f}ms"
        )

        return result

    def _detect_file_type(self, path: Path) -> FileType:
        """Detect the file type from the file extension.

        Args:
            path: Path to the file.

        Returns:
            FileType enum value.

        Raises:
            ValueError: If the file type is not supported.
        """
        extension = path.suffix.lower().lstrip(".")

        extension_map = {
            "pdf": FileType.PDF,
            "png": FileType.IMAGE,
            "jpg": FileType.IMAGE,
            "jpeg": FileType.IMAGE,
            "tiff": FileType.IMAGE,
            "tif": FileType.IMAGE,
            "docx": FileType.DOCX,
            "txt": FileType.TEXT,
        }

        file_type = extension_map.get(extension)
        if file_type is None:
            supported = ", ".join(self._settings.supported_file_types)
            raise ValueError(
                f"Unsupported file type: .{extension}. Supported types: {supported}"
            )

        return file_type

    def _select_processor(self, file_type: FileType):
        """Select the appropriate document processor based on file type.

        Args:
            file_type: The detected file type.

        Returns:
            Processor instance for the file type.
        """
        processors = {
            FileType.PDF: self._pdf_processor,
            FileType.IMAGE: self._image_processor,
            FileType.DOCX: self._docx_processor,
        }

        processor = processors.get(file_type)
        if processor is None:
            # For text files, use a simple text processor
            return _TextProcessor()

        return processor

    def _select_extractor(self, mode: str) -> BaseExtractor:
        """Select the appropriate extractor based on extraction mode.

        Args:
            mode: Extraction mode string ('local', 'api', 'hybrid').

        Returns:
            BaseExtractor instance for the specified mode.
        """
        mode_map = {
            "local": self._get_local_extractor,
            "api": self._get_api_extractor,
            "hybrid": self._get_hybrid_extractor,
        }

        factory = mode_map.get(mode.lower())
        if factory is None:
            logger.warning(f"Unknown mode '{mode}', falling back to hybrid")
            return self._get_hybrid_extractor()

        return factory()

    async def _run_industry_analysis(
        self, result: ExtractionResult, doc_type: Optional[str] = None
    ) -> ExtractionResult:
        """Run industry-specific analysis on the extraction result.

        This is a hook for industry analyzers (contract, invoice, compliance, etc.)
        that will be implemented in FEAT-003.

        Args:
            result: The extraction result to analyze.
            doc_type: Optional document type hint.

        Returns:
            ExtractionResult potentially enriched with industry analysis.
        """
        # Industry analysis modules will be implemented in FEAT-003
        # For now, return the result unchanged
        return result

    async def _trigger_actions(self, result: ExtractionResult) -> List[Dict[str, Any]]:
        """Trigger actions based on extraction results using the action engine.

        Args:
            result: The extraction result to evaluate rules against.

        Returns:
            List of triggered actions.
        """
        try:
            action_engine = self._get_action_engine()
            actions = action_engine.evaluate_rules(result)
            return actions
        except Exception as e:
            logger.warning(f"Action triggering failed: {e}")
            return []


class _TextProcessor:
    """Simple text file processor for .txt files."""

    def process(self, file_path: str) -> Document:
        """Process a plain text file.

        Args:
            file_path: Path to the text file.

        Returns:
            Document model with text content.
        """
        from uuid import uuid4

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")

        text = path.read_text(encoding="utf-8")

        return Document(
            id=uuid4(),
            filename=path.name,
            file_type=FileType.TEXT,
            content=text,
            raw_text=text,
            page_count=1,
            file_path=str(path.absolute()),
            metadata={},
        )
