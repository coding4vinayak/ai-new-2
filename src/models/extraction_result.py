"""Extraction result data models."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from src.models.confidence import ConfidenceReport


class ExtractionMode(str, Enum):
    """Available extraction modes."""

    LOCAL = "local"
    API = "api"
    HYBRID = "hybrid"
    LOCAL_LLM = "local_llm"
    ENSEMBLE = "ensemble"


class ExtractionResult(BaseModel):
    """Result of a document extraction operation."""

    document_id: UUID = Field(..., description="ID of the source document")
    extraction_mode: ExtractionMode = Field(
        ..., description="Mode used for extraction"
    )
    entities: Dict[str, Any] = Field(
        default_factory=dict, description="Extracted entities"
    )
    confidence_report: ConfidenceReport = Field(
        ..., description="Confidence scores for extracted fields"
    )
    raw_text: str = Field(default="", description="Raw text that was processed")
    processing_time_ms: float = Field(
        ..., ge=0, description="Processing time in milliseconds"
    )
    extracted_at: datetime = Field(
        default_factory=datetime.utcnow, description="Extraction timestamp"
    )
    extractor_version: str = Field(
        default="1.0.0", description="Version of the extractor used"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Warnings generated during extraction"
    )


class BatchExtractionResult(BaseModel):
    """Result of a batch extraction operation."""

    batch_id: UUID = Field(..., description="Unique batch identifier")
    results: List[ExtractionResult] = Field(
        default_factory=list, description="Individual extraction results"
    )
    total_documents: int = Field(default=0, description="Total documents in batch")
    successful: int = Field(default=0, description="Successfully processed documents")
    failed: int = Field(default=0, description="Failed documents")
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Error details for failed documents"
    )
    started_at: Optional[datetime] = Field(None, description="Batch start time")
    completed_at: Optional[datetime] = Field(None, description="Batch completion time")
    total_processing_time_ms: float = Field(
        default=0, description="Total processing time in milliseconds"
    )
