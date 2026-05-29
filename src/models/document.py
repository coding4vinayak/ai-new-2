"""Document data models."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class FileType(str, Enum):
    """Supported document file types."""

    PDF = "pdf"
    IMAGE = "image"
    DOCX = "docx"
    TEXT = "text"


class DocumentPage(BaseModel):
    """Represents a single page of a document."""

    page_number: int = Field(..., ge=1, description="Page number (1-indexed)")
    text: str = Field(default="", description="Extracted text content of the page")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence of text extraction"
    )


class Document(BaseModel):
    """Core document model representing a processed document."""

    id: UUID = Field(default_factory=uuid4, description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_type: FileType = Field(..., description="Type of the document file")
    content: Optional[str] = Field(None, description="Processed/structured content")
    raw_text: Optional[str] = Field(None, description="Raw extracted text")
    page_count: int = Field(default=1, ge=1, description="Number of pages")
    language: Optional[str] = Field(None, description="Detected language code")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Document creation timestamp"
    )
    file_path: Optional[str] = Field(None, description="Path to the source file")
    pages: List[DocumentPage] = Field(
        default_factory=list, description="Individual page data"
    )
