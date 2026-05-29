"""Document agent orchestrator coordinating the full extraction pipeline."""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.extractors.api_extractor import APIExtractor
from src.extractors.base import BaseExtractor
from src.extractors.hybrid_extractor import HybridExtractor
from src.extractors.local_extractor import LocalExtractor
from src.models.extraction_result import ExtractionMode
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

        Invokes the appropriate industry analyzers based on document type:
        - contract: ContractAnalyzer
        - invoice: InvoiceProcessor
        - All types: PIIDetector, ComplianceChecker

        The document classifier is used when no explicit doc_type is provided.
        Each module is wrapped in its own error handler so a failure in one
        does not prevent the others from running.

        Args:
            result: The extraction result to analyze.
            doc_type: Optional document type hint.

        Returns:
            ExtractionResult enriched with industry analysis metadata.
        """
        from src.industry.compliance_checker import ComplianceChecker
        from src.industry.contract_analyzer import ContractAnalyzer
        from src.industry.document_classifier import DocumentClassifier
        from src.industry.invoice_processor import InvoiceProcessor
        from src.industry.pii_detector import PIIDetector

        text = result.raw_text or ""

        # Classify document type if not provided
        effective_type = doc_type
        if not effective_type:
            try:
                classifier = DocumentClassifier()
                if text.strip():
                    classified_type, confidence = classifier.classify(text)
                    effective_type = classified_type.value
                    result.entities.setdefault("_classification", {
                        "type": classified_type.value,
                        "confidence": confidence,
                    })
            except Exception as e:
                logger.warning(f"Document classification failed: {e}")
                result.warnings.append(f"Document classification failed: {str(e)}")

        # Run PII detection on all document types
        try:
            pii_detector = PIIDetector()
            if text.strip():
                pii_entities = pii_detector.detect_pii(text)
                if pii_entities:
                    pii_report = pii_detector.generate_pii_report(pii_entities)
                    result.entities.setdefault("_pii", {
                        "total_entities": pii_report.total_entities,
                        "risk_level": pii_report.risk_level,
                        "counts_by_type": pii_report.counts_by_type,
                    })
        except Exception as e:
            logger.warning(f"PII detection failed: {e}")
            result.warnings.append(f"PII detection failed: {str(e)}")

        # Run type-specific analysis
        if effective_type and effective_type.upper() == "CONTRACT":
            try:
                analyzer = ContractAnalyzer()
                analysis = analyzer.analyze(result)
                result.entities.setdefault("_contract_analysis", {
                    "clauses_found": len(analysis.clauses),
                    "obligations_found": len(analysis.obligations),
                    "key_dates": len(analysis.key_dates),
                    "risk_score": analysis.risk_score,
                    "risk_factors": analysis.risk_factors,
                    "recommendations": analysis.recommendations,
                })
            except Exception as e:
                logger.warning(f"Contract analysis failed: {e}")
                result.warnings.append(f"Contract analysis failed: {str(e)}")

        elif effective_type and effective_type.upper() == "INVOICE":
            try:
                processor = InvoiceProcessor()
                invoice_data = processor.process(result)
                result.entities.setdefault("_invoice", {
                    "vendor": invoice_data.vendor,
                    "invoice_number": invoice_data.invoice_number,
                    "total": invoice_data.total,
                    "currency": invoice_data.currency,
                    "line_items_count": len(invoice_data.line_items),
                    "validation_status": invoice_data.validation_status.value,
                    "validation_errors": invoice_data.validation_errors,
                })
            except Exception as e:
                logger.warning(f"Invoice processing failed: {e}")
                result.warnings.append(f"Invoice processing failed: {str(e)}")

        # Run compliance check for contracts and legal docs
        if effective_type and effective_type.upper() in ("CONTRACT", "LEGAL"):
            try:
                checker = ComplianceChecker()
                compliance_report = checker.check_compliance(
                    document_text=text,
                    doc_type=effective_type.upper(),
                )
                result.entities.setdefault("_compliance", {
                    "compliant": compliance_report.compliant,
                    "risk_level": compliance_report.risk_level.value,
                    "findings_count": len(compliance_report.findings),
                    "missing_clauses": compliance_report.missing_clauses,
                    "recommendations": compliance_report.recommendations,
                })
            except Exception as e:
                logger.warning(f"Compliance check failed: {e}")
                result.warnings.append(f"Compliance check failed: {str(e)}")

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
