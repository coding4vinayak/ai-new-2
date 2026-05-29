"""Confidence scoring models for extraction results."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ConfidenceScore(BaseModel):
    """Confidence score for a single extracted field."""

    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    method: str = Field(..., description="Method used for extraction (e.g., 'local_ner', 'api_gpt4', 'hybrid')")
    explanation: Optional[str] = Field(None, description="Explanation of how confidence was determined")
    field_name: str = Field(..., description="Name of the field this score applies to")


class ConfidenceReport(BaseModel):
    """Aggregated confidence report for an extraction result."""

    scores: Dict[str, ConfidenceScore] = Field(
        default_factory=dict,
        description="Mapping of field names to their confidence scores",
    )
    overall_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Overall confidence for the extraction"
    )
    low_confidence_fields: List[str] = Field(
        default_factory=list,
        description="Fields with confidence below the threshold",
    )
    threshold: float = Field(
        default=0.7, description="Threshold used to determine low confidence fields"
    )

    def compute_low_confidence_fields(self) -> List[str]:
        """Compute and return fields with confidence below the threshold."""
        self.low_confidence_fields = [
            name for name, score in self.scores.items() if score.score < self.threshold
        ]
        return self.low_confidence_fields
