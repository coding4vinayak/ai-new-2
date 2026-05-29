"""Document classification module for auto-classifying document types."""

import re
from enum import Enum
from typing import Dict, List, Tuple

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Supported document type classifications."""

    CONTRACT = "CONTRACT"
    INVOICE = "INVOICE"
    LEGAL = "LEGAL"
    FINANCIAL = "FINANCIAL"
    HR = "HR"
    MEDICAL = "MEDICAL"
    GENERAL = "GENERAL"


class ClassificationResult(BaseModel):
    """Result of document classification."""

    document_type: DocumentType = Field(..., description="Classified document type")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence of the classification"
    )
    scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Scores for all document types considered",
    )


class DocumentClassifier:
    """Classifies documents into types using keyword scoring and structural analysis.

    Uses a combination of keyword-based scoring and structural pattern
    detection to determine the most likely document type.
    """

    # Keywords indicative of each document type
    _KEYWORDS: Dict[DocumentType, List[str]] = {
        DocumentType.CONTRACT: [
            "agreement",
            "whereas",
            "hereby",
            "party",
            "parties",
            "terms and conditions",
            "effective date",
            "termination",
            "obligations",
            "governing law",
            "indemnify",
            "confidential",
            "covenant",
            "execute",
            "binding",
            "shall",
            "herein",
            "hereto",
            "witnesseth",
        ],
        DocumentType.INVOICE: [
            "invoice",
            "bill to",
            "ship to",
            "total due",
            "subtotal",
            "tax",
            "payment terms",
            "due date",
            "invoice number",
            "quantity",
            "unit price",
            "amount due",
            "remit to",
            "purchase order",
            "line item",
        ],
        DocumentType.LEGAL: [
            "court",
            "plaintiff",
            "defendant",
            "jurisdiction",
            "statute",
            "regulation",
            "law",
            "legal",
            "ruling",
            "verdict",
            "counsel",
            "attorney",
            "litigation",
            "appeal",
            "motion",
        ],
        DocumentType.FINANCIAL: [
            "balance sheet",
            "income statement",
            "revenue",
            "profit",
            "loss",
            "assets",
            "liabilities",
            "equity",
            "cash flow",
            "fiscal year",
            "quarterly",
            "earnings",
            "dividend",
            "investment",
            "depreciation",
        ],
        DocumentType.HR: [
            "employee",
            "salary",
            "benefits",
            "vacation",
            "performance review",
            "hire",
            "termination",
            "onboarding",
            "job description",
            "compensation",
            "leave",
            "payroll",
            "handbook",
            "policy",
            "human resources",
        ],
        DocumentType.MEDICAL: [
            "patient",
            "diagnosis",
            "treatment",
            "prescription",
            "medical",
            "clinical",
            "symptoms",
            "physician",
            "hospital",
            "health",
            "dosage",
            "medication",
            "prognosis",
            "lab results",
            "vitals",
        ],
    }

    # Structural patterns for each document type
    _STRUCTURAL_PATTERNS: Dict[DocumentType, List[str]] = {
        DocumentType.CONTRACT: [
            r"^\d+\.\s+[A-Z]",  # Numbered clauses
            r"ARTICLE\s+[IVXLCDM\d]+",  # Article numbering
            r"Section\s+\d+",  # Section numbering
            r"IN WITNESS WHEREOF",
        ],
        DocumentType.INVOICE: [
            r"\$[\d,]+\.\d{2}",  # Dollar amounts
            r"\d+\s*x\s*\$?[\d,]+",  # Quantity x price
            r"(?:Qty|Quantity)\s*[\|:]\s*\d+",  # Quantity columns
            r"Total[\s:]+\$?[\d,]+\.\d{2}",  # Total line
        ],
        DocumentType.LEGAL: [
            r"Case\s+No\.\s*\w+",  # Case numbers
            r"v\.\s+",  # Plaintiff v. Defendant
            r"ORDER|JUDGMENT|RULING",
        ],
        DocumentType.FINANCIAL: [
            r"\(\$?[\d,]+\)",  # Negative numbers in parentheses
            r"FY\s*\d{4}",  # Fiscal year
            r"Q[1-4]\s+\d{4}",  # Quarterly references
        ],
        DocumentType.HR: [
            r"Position:\s*",
            r"Department:\s*",
            r"Start Date:\s*",
            r"Annual Salary:\s*",
        ],
        DocumentType.MEDICAL: [
            r"DOB:\s*\d",
            r"Patient\s+ID:\s*",
            r"ICD-\d+",
            r"Rx:\s*",
        ],
    }

    def classify(self, text: str) -> Tuple[DocumentType, float]:
        """Classify a document based on its text content.

        Uses keyword scoring and structural analysis to determine
        the document type.

        Args:
            text: The document text to classify.

        Returns:
            Tuple of (document_type, confidence) where confidence is 0.0-1.0.
        """
        if not text or not text.strip():
            return (DocumentType.GENERAL, 0.5)

        keyword_scores = self._keyword_scores(text)
        structural_scores = self._structural_analysis(text)

        # Combine scores: keywords weighted 0.6, structural 0.4
        combined_scores: Dict[DocumentType, float] = {}
        for doc_type in DocumentType:
            if doc_type == DocumentType.GENERAL:
                continue
            kw_score = keyword_scores.get(doc_type, 0.0)
            st_score = structural_scores.get(doc_type, 0.0)
            combined_scores[doc_type] = (kw_score * 0.6) + (st_score * 0.4)

        if not combined_scores:
            return (DocumentType.GENERAL, 0.5)

        # Find the best classification
        best_type = max(combined_scores, key=combined_scores.get)  # type: ignore
        best_score = combined_scores[best_type]

        # If best score is too low, classify as GENERAL
        if best_score < 0.03:
            return (DocumentType.GENERAL, 0.5)

        # Calculate confidence based on:
        # 1. Absolute score strength (how many keywords matched)
        # 2. Relative dominance (how far ahead of second-best)
        sorted_scores = sorted(combined_scores.values(), reverse=True)
        second_best = sorted_scores[1] if len(sorted_scores) > 1 else 0.0

        # Dominance: how much better is the top score vs the rest
        dominance = (best_score - second_best) / best_score if best_score > 0 else 0.0

        # Base confidence from absolute score (scaled so 0.1 -> 0.7, 0.3 -> 0.95)
        base_confidence = min(1.0, 0.5 + (best_score * 4.0))

        # Boost by dominance factor
        confidence = base_confidence * (0.7 + 0.3 * dominance)
        confidence = min(1.0, max(0.0, confidence))

        return (best_type, round(confidence, 3))

    def _keyword_scores(self, text: str) -> Dict[DocumentType, float]:
        """Score text against keyword lists for each document type.

        Args:
            text: Document text to analyze.

        Returns:
            Dictionary mapping document types to their keyword scores (0.0-1.0).
        """
        text_lower = text.lower()
        scores: Dict[DocumentType, float] = {}

        for doc_type, keywords in self._KEYWORDS.items():
            matches = 0
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    matches += 1

            # Normalize by total keywords for this type
            scores[doc_type] = matches / len(keywords) if keywords else 0.0

        return scores

    def _structural_analysis(self, text: str) -> Dict[DocumentType, float]:
        """Analyze text structure for patterns indicative of document types.

        Args:
            text: Document text to analyze.

        Returns:
            Dictionary mapping document types to structural pattern scores (0.0-1.0).
        """
        scores: Dict[DocumentType, float] = {}

        for doc_type, patterns in self._STRUCTURAL_PATTERNS.items():
            matches = 0
            for pattern in patterns:
                if re.search(pattern, text, re.MULTILINE):
                    matches += 1

            scores[doc_type] = matches / len(patterns) if patterns else 0.0

        return scores
