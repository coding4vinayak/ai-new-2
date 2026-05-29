"""PII detection and redaction module."""

import hashlib
import re
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PIIType(str, Enum):
    """Types of PII that can be detected."""

    SSN = "ssn"
    EMAIL = "email"
    PHONE = "phone"
    CREDIT_CARD = "credit_card"
    ADDRESS = "address"
    DOB = "date_of_birth"
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    IP_ADDRESS = "ip_address"
    NAME = "name"


class RedactionMethod(str, Enum):
    """Available redaction methods."""

    MASK = "mask"
    HASH = "hash"
    REMOVE = "remove"


class PIIEntity(BaseModel):
    """A detected PII entity in the text."""

    pii_type: PIIType = Field(..., description="Type of PII detected")
    value: str = Field(..., description="The PII value found")
    start_pos: int = Field(..., description="Start position in text")
    end_pos: int = Field(..., description="End position in text")
    confidence: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Detection confidence"
    )


class PIIReport(BaseModel):
    """Report summarizing PII detection results."""

    total_entities: int = Field(default=0, description="Total PII entities found")
    counts_by_type: Dict[str, int] = Field(
        default_factory=dict, description="Count of entities by PII type"
    )
    risk_level: str = Field(
        default="low", description="Overall PII risk level"
    )
    entities: List[PIIEntity] = Field(
        default_factory=list, description="All detected PII entities"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations for PII handling"
    )


class PIIDetector:
    """Detects and redacts PII from document text.

    Uses regex-based pattern matching to identify various types of
    personally identifiable information.
    """

    # Regex patterns for each PII type
    _PATTERNS: Dict[PIIType, List[tuple]] = {
        PIIType.SSN: [
            (r"\b\d{3}-\d{2}-\d{4}\b", 0.95),
            (r"\b\d{3}\s\d{2}\s\d{4}\b", 0.85),
        ],
        PIIType.EMAIL: [
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", 0.95),
        ],
        PIIType.PHONE: [
            (r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", 0.90),
            (r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", 0.90),
        ],
        PIIType.CREDIT_CARD: [
            (r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6011)[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", 0.90),
            (r"\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b", 0.80),
        ],
        PIIType.DOB: [
            (r"\b(?:DOB|Date of Birth|Born)[:\s]+(\d{1,2}/\d{1,2}/\d{2,4})\b", 0.90),
            (r"\b(?:DOB|Date of Birth|Born)[:\s]+(\w+\s+\d{1,2},?\s+\d{4})\b", 0.85),
        ],
        PIIType.PASSPORT: [
            (r"\b(?:Passport\s*(?:#|No\.?|Number)?)[:\s]*([A-Z0-9]{6,9})\b", 0.85),
        ],
        PIIType.IP_ADDRESS: [
            (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", 0.80),
        ],
        PIIType.DRIVERS_LICENSE: [
            (r"\b(?:DL|Driver'?s?\s*License)\s*(?:#|No\.?|Number)?[:\s]*([A-Z0-9]{5,15})\b", 0.80),
        ],
    }

    # Mask replacements for each PII type
    _MASKS: Dict[PIIType, str] = {
        PIIType.SSN: "***-**-****",
        PIIType.EMAIL: "[EMAIL REDACTED]",
        PIIType.PHONE: "[PHONE REDACTED]",
        PIIType.CREDIT_CARD: "****-****-****-****",
        PIIType.ADDRESS: "[ADDRESS REDACTED]",
        PIIType.DOB: "[DOB REDACTED]",
        PIIType.PASSPORT: "[PASSPORT REDACTED]",
        PIIType.DRIVERS_LICENSE: "[DL REDACTED]",
        PIIType.IP_ADDRESS: "[IP REDACTED]",
        PIIType.NAME: "[NAME REDACTED]",
    }

    def detect_pii(self, text: str) -> List[PIIEntity]:
        """Detect PII entities in text.

        Args:
            text: Text to scan for PII.

        Returns:
            List of detected PII entities with positions and types.
        """
        entities: List[PIIEntity] = []
        if not text:
            return entities

        for pii_type, patterns in self._PATTERNS.items():
            for pattern, confidence in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Use group(1) if there's a capturing group, otherwise full match
                    if match.lastindex and match.lastindex >= 1:
                        value = match.group(1)
                        start = match.start(1)
                        end = match.end(1)
                    else:
                        value = match.group(0)
                        start = match.start()
                        end = match.end()

                    # Avoid duplicate overlapping matches
                    if not self._overlaps_existing(entities, start, end):
                        entities.append(
                            PIIEntity(
                                pii_type=pii_type,
                                value=value,
                                start_pos=start,
                                end_pos=end,
                                confidence=confidence,
                            )
                        )

        # Sort by position
        entities.sort(key=lambda e: e.start_pos)
        return entities

    def redact(
        self,
        text: str,
        pii_entities: List[PIIEntity],
        method: str = "mask",
    ) -> str:
        """Redact PII from text using the specified method.

        Args:
            text: Original text containing PII.
            pii_entities: List of PII entities to redact.
            method: Redaction method ('mask', 'hash', or 'remove').

        Returns:
            Text with PII redacted.
        """
        if not pii_entities:
            return text

        # Sort entities by position in reverse to preserve positions
        sorted_entities = sorted(
            pii_entities, key=lambda e: e.start_pos, reverse=True
        )

        result = text
        for entity in sorted_entities:
            replacement = self._get_replacement(entity, method)
            result = result[:entity.start_pos] + replacement + result[entity.end_pos:]

        return result

    def generate_pii_report(self, pii_entities: List[PIIEntity]) -> PIIReport:
        """Generate a summary report of detected PII.

        Args:
            pii_entities: List of detected PII entities.

        Returns:
            PIIReport with counts, risk assessment, and recommendations.
        """
        counts_by_type: Dict[str, int] = {}
        for entity in pii_entities:
            type_name = entity.pii_type.value
            counts_by_type[type_name] = counts_by_type.get(type_name, 0) + 1

        # Assess risk level
        risk_level = self._assess_risk(pii_entities)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            pii_entities, counts_by_type
        )

        return PIIReport(
            total_entities=len(pii_entities),
            counts_by_type=counts_by_type,
            risk_level=risk_level,
            entities=pii_entities,
            recommendations=recommendations,
        )

    def _overlaps_existing(
        self, entities: List[PIIEntity], start: int, end: int
    ) -> bool:
        """Check if a position range overlaps with existing entities.

        Args:
            entities: Existing entities.
            start: Start position to check.
            end: End position to check.

        Returns:
            True if there is an overlap.
        """
        for entity in entities:
            if start < entity.end_pos and end > entity.start_pos:
                return True
        return False

    def _get_replacement(self, entity: PIIEntity, method: str) -> str:
        """Get the replacement string for a PII entity.

        Args:
            entity: The PII entity to replace.
            method: Redaction method.

        Returns:
            Replacement string.
        """
        if method == "remove":
            return ""
        elif method == "hash":
            hash_val = hashlib.sha256(entity.value.encode()).hexdigest()[:8]
            return f"[{entity.pii_type.value.upper()}:{hash_val}]"
        else:
            # mask (default)
            return self._MASKS.get(entity.pii_type, "[REDACTED]")

    def _assess_risk(self, entities: List[PIIEntity]) -> str:
        """Assess overall PII risk level.

        Args:
            entities: Detected PII entities.

        Returns:
            Risk level string.
        """
        if not entities:
            return "low"

        high_risk_types = {PIIType.SSN, PIIType.CREDIT_CARD, PIIType.PASSPORT}
        has_high_risk = any(e.pii_type in high_risk_types for e in entities)

        if has_high_risk or len(entities) > 10:
            return "high"
        elif len(entities) > 5:
            return "medium"
        return "low"

    def _generate_recommendations(
        self,
        entities: List[PIIEntity],
        counts: Dict[str, int],
    ) -> List[str]:
        """Generate PII handling recommendations.

        Args:
            entities: Detected entities.
            counts: Counts by type.

        Returns:
            List of recommendations.
        """
        recommendations: List[str] = []

        if not entities:
            return ["No PII detected - document appears safe for sharing"]

        recommendations.append(
            "Redact PII before sharing or storing in unprotected systems"
        )

        high_risk_types = {PIIType.SSN, PIIType.CREDIT_CARD, PIIType.PASSPORT}
        found_high_risk = [
            e.pii_type.value for e in entities if e.pii_type in high_risk_types
        ]
        if found_high_risk:
            recommendations.append(
                f"High-risk PII detected ({', '.join(set(found_high_risk))})"
                " - ensure encryption at rest and in transit"
            )

        if counts.get("email", 0) > 0:
            recommendations.append(
                "Email addresses found - verify consent for data processing"
            )

        return recommendations
