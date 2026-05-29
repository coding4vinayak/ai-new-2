"""Contract analysis module for deep contract document processing."""

import re
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from src.models.extraction_result import ExtractionResult


class ClauseType(str, Enum):
    """Types of contract clauses."""

    TERMINATION = "termination"
    INDEMNITY = "indemnity"
    CONFIDENTIALITY = "confidentiality"
    FORCE_MAJEURE = "force_majeure"
    LIMITATION_OF_LIABILITY = "limitation_of_liability"
    GOVERNING_LAW = "governing_law"
    PAYMENT = "payment"
    IP_RIGHTS = "ip_rights"
    NON_COMPETE = "non_compete"
    DISPUTE_RESOLUTION = "dispute_resolution"


class Clause(BaseModel):
    """A contract clause extracted from the document."""

    clause_type: ClauseType = Field(..., description="Type of the clause")
    text: str = Field(..., description="Full text of the clause")
    section_number: Optional[str] = Field(
        None, description="Section/article number if found"
    )
    confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Detection confidence"
    )


class Obligation(BaseModel):
    """A contractual obligation identified in the document."""

    party: str = Field(..., description="Party responsible for the obligation")
    description: str = Field(..., description="Description of the obligation")
    deadline: Optional[str] = Field(None, description="Deadline or timeframe if found")
    clause_reference: Optional[str] = Field(
        None, description="Reference to the clause containing this obligation"
    )


class KeyDate(BaseModel):
    """A key date identified in the contract."""

    description: str = Field(..., description="What the date represents")
    date_str: str = Field(..., description="Date string as found in the document")
    days_until: Optional[int] = Field(
        None, description="Days until this date (negative if past)"
    )


class ContractAnalysis(BaseModel):
    """Complete analysis result for a contract document."""

    clauses: List[Clause] = Field(
        default_factory=list, description="Identified clauses"
    )
    obligations: List[Obligation] = Field(
        default_factory=list, description="Identified obligations"
    )
    key_dates: List[KeyDate] = Field(
        default_factory=list, description="Key dates extracted"
    )
    risk_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall risk score"
    )
    risk_factors: List[str] = Field(
        default_factory=list, description="Identified risk factors"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations for review"
    )


class ContractAnalyzer:
    """Performs deep analysis of contract documents.

    Extracts clauses, obligations, key dates, and computes risk scores
    for contract documents.
    """

    # Patterns for clause detection
    _CLAUSE_PATTERNS: Dict[ClauseType, List[str]] = {
        ClauseType.TERMINATION: [
            r"terminat(?:ion|e|ing)",
            r"cancel(?:lation)?",
            r"end\s+(?:of\s+)?(?:this\s+)?agreement",
        ],
        ClauseType.INDEMNITY: [
            r"indemnif(?:y|ication|ied)",
            r"hold\s+harmless",
            r"defend\s+and\s+indemnify",
        ],
        ClauseType.CONFIDENTIALITY: [
            r"confidential(?:ity)?",
            r"non-disclosure",
            r"proprietary\s+information",
            r"trade\s+secret",
        ],
        ClauseType.FORCE_MAJEURE: [
            r"force\s+majeure",
            r"act\s+of\s+god",
            r"unforeseeable\s+circumstances",
        ],
        ClauseType.LIMITATION_OF_LIABILITY: [
            r"limitation\s+of\s+liability",
            r"limit(?:ed)?\s+liab(?:le|ility)",
            r"in\s+no\s+event\s+shall.*(?:liable|exceed)",
        ],
        ClauseType.GOVERNING_LAW: [
            r"governing\s+law",
            r"governed\s+by\s+(?:the\s+)?laws?\s+of",
            r"jurisdiction",
        ],
        ClauseType.PAYMENT: [
            r"payment\s+terms?",
            r"(?:net\s+)?\d+\s+days",
            r"compensation",
            r"fees?\s+(?:shall|will)\s+be",
        ],
        ClauseType.IP_RIGHTS: [
            r"intellectual\s+property",
            r"patent|copyright|trademark",
            r"ownership\s+of\s+(?:work|deliverables)",
        ],
        ClauseType.NON_COMPETE: [
            r"non-compete",
            r"non-competition",
            r"restrictive\s+covenant",
        ],
        ClauseType.DISPUTE_RESOLUTION: [
            r"dispute\s+resolution",
            r"arbitration",
            r"mediation",
        ],
    }

    # Standard clauses that should be present in a well-formed contract
    _STANDARD_CLAUSES = [
        ClauseType.TERMINATION,
        ClauseType.CONFIDENTIALITY,
        ClauseType.GOVERNING_LAW,
        ClauseType.LIMITATION_OF_LIABILITY,
        ClauseType.PAYMENT,
    ]

    def analyze(self, extraction_result: ExtractionResult) -> ContractAnalysis:
        """Perform deep analysis of a contract document.

        Args:
            extraction_result: The extraction result containing contract text.

        Returns:
            ContractAnalysis with clauses, obligations, dates, and risk assessment.
        """
        text = extraction_result.raw_text or ""

        clauses = self.extract_clauses(text)
        obligations = self.extract_obligations(text)
        key_dates = self.track_expiry_dates(extraction_result)
        risk_score = self.calculate_risk_score(clauses, obligations)

        # Generate risk factors and recommendations
        risk_factors = self._identify_risk_factors(clauses, obligations)
        recommendations = self._generate_recommendations(clauses, risk_factors)

        return ContractAnalysis(
            clauses=clauses,
            obligations=obligations,
            key_dates=key_dates,
            risk_score=risk_score,
            risk_factors=risk_factors,
            recommendations=recommendations,
        )

    def extract_clauses(self, text: str) -> List[Clause]:
        """Identify and categorize clauses in the contract text.

        Args:
            text: The contract text to analyze.

        Returns:
            List of identified clauses with their types.
        """
        clauses: List[Clause] = []
        if not text:
            return clauses

        # Split text into sections/paragraphs
        sections = self._split_into_sections(text)

        for section_num, section_text in sections:
            for clause_type, patterns in self._CLAUSE_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, section_text, re.IGNORECASE):
                        # Avoid duplicates of same type in same section
                        if not any(
                            c.clause_type == clause_type
                            and c.section_number == section_num
                            for c in clauses
                        ):
                            clauses.append(
                                Clause(
                                    clause_type=clause_type,
                                    text=section_text.strip()[:500],
                                    section_number=section_num,
                                    confidence=0.85,
                                )
                            )
                        break

        return clauses

    def extract_obligations(self, text: str) -> List[Obligation]:
        """Extract obligations from contract text.

        Args:
            text: The contract text to analyze.

        Returns:
            List of identified obligations.
        """
        obligations: List[Obligation] = []
        if not text:
            return obligations

        # Patterns indicating obligations
        obligation_patterns = [
            r"(?P<party>(?:Party\s+[AB]|(?:the\s+)?(?:Seller|Buyer|Vendor|Client|Provider|Licensee|Licensor)))\s+(?:shall|must|agrees?\s+to|is\s+obligated\s+to)\s+(?P<desc>[^.]+)",
            r"(?P<party>(?:Company|Contractor|Employee|Employer))\s+(?:shall|must|agrees?\s+to)\s+(?P<desc>[^.]+)",
        ]

        for pattern in obligation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                party = match.group("party").strip()
                description = match.group("desc").strip()

                # Look for deadline in nearby text
                deadline = self._find_nearby_deadline(
                    text, match.start(), match.end()
                )

                obligations.append(
                    Obligation(
                        party=party,
                        description=description[:200],
                        deadline=deadline,
                    )
                )

        return obligations

    def track_expiry_dates(self, result: ExtractionResult) -> List[KeyDate]:
        """Identify key dates and compute days until expiry.

        Args:
            result: The extraction result to analyze.

        Returns:
            List of key dates with days-until-expiry calculations.
        """
        key_dates: List[KeyDate] = []
        text = result.raw_text or ""

        # Date patterns to look for
        date_contexts = [
            (r"(?:effective|start)\s+date[:\s]+(\w+\s+\d{1,2},?\s+\d{4})", "Effective Date"),
            (r"(?:expir(?:y|ation)|end)\s+date[:\s]+(\w+\s+\d{1,2},?\s+\d{4})", "Expiry Date"),
            (r"(?:terminat(?:ion|e))\s+(?:date|by)[:\s]+(\w+\s+\d{1,2},?\s+\d{4})", "Termination Date"),
            (r"(?:renewal|renew)\s+(?:date|by)[:\s]+(\w+\s+\d{1,2},?\s+\d{4})", "Renewal Date"),
            (r"(\d{1,2}/\d{1,2}/\d{4})", "Date Reference"),
            (r"(\d{4}-\d{2}-\d{2})", "Date Reference"),
        ]

        for pattern, description in date_contexts:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_str = match.group(1)
                days_until = self._compute_days_until(date_str)
                key_dates.append(
                    KeyDate(
                        description=description,
                        date_str=date_str,
                        days_until=days_until,
                    )
                )

        return key_dates

    def calculate_risk_score(
        self, clauses: List[Clause], obligations: List[Obligation]
    ) -> float:
        """Calculate an overall risk score for the contract.

        Scoring based on:
        - Missing standard clauses (higher risk)
        - One-sided obligations (higher risk)
        - Presence of protective clauses (lower risk)

        Args:
            clauses: Extracted clauses.
            obligations: Extracted obligations.

        Returns:
            Risk score between 0.0 (low risk) and 1.0 (high risk).
        """
        risk_points = 0.0
        max_points = 10.0

        # Check for missing standard clauses
        found_types = {c.clause_type for c in clauses}
        for standard_clause in self._STANDARD_CLAUSES:
            if standard_clause not in found_types:
                risk_points += 1.5

        # Check for one-sided obligations
        if obligations:
            parties = [o.party.lower() for o in obligations]
            unique_parties = set(parties)
            if len(unique_parties) == 1 and len(obligations) > 2:
                # All obligations on one party
                risk_points += 2.0

        # Missing force majeure is a moderate risk
        if ClauseType.FORCE_MAJEURE not in found_types:
            risk_points += 0.5

        # Missing dispute resolution
        if ClauseType.DISPUTE_RESOLUTION not in found_types:
            risk_points += 0.5

        # Normalize to 0-1 range
        return min(1.0, risk_points / max_points)

    def _split_into_sections(self, text: str) -> List[tuple]:
        """Split text into numbered sections or paragraphs.

        Args:
            text: Full contract text.

        Returns:
            List of (section_number, section_text) tuples.
        """
        # Try splitting by numbered sections
        section_pattern = r"(?:^|\n)(\d+(?:\.\d+)*\.?\s)"
        parts = re.split(section_pattern, text)

        sections = []
        if len(parts) > 2:
            # We got section splits
            for i in range(1, len(parts) - 1, 2):
                section_num = parts[i].strip().rstrip(".")
                section_text = parts[i + 1] if i + 1 < len(parts) else ""
                sections.append((section_num, section_text))
        else:
            # Fall back to paragraph splitting
            paragraphs = text.split("\n\n")
            for idx, para in enumerate(paragraphs, 1):
                if para.strip():
                    sections.append((str(idx), para))

        return sections

    def _find_nearby_deadline(self, text: str, start: int, end: int) -> Optional[str]:
        """Find a deadline mentioned near an obligation.

        Args:
            text: Full text.
            start: Start position of the obligation.
            end: End position of the obligation.

        Returns:
            Deadline string if found, None otherwise.
        """
        # Look in a window around the obligation
        window_start = max(0, start - 50)
        window_end = min(len(text), end + 200)
        window = text[window_start:window_end]

        deadline_patterns = [
            r"within\s+(\d+\s+(?:days?|months?|years?))",
            r"by\s+(\w+\s+\d{1,2},?\s+\d{4})",
            r"no\s+later\s+than\s+(\w+\s+\d{1,2},?\s+\d{4})",
            r"(?:due|deadline)[:\s]+([^.;]+)",
        ]

        for pattern in deadline_patterns:
            match = re.search(pattern, window, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _compute_days_until(self, date_str: str) -> Optional[int]:
        """Compute days until a given date.

        Args:
            date_str: Date string to parse.

        Returns:
            Number of days until the date (negative if past), or None if unparseable.
        """
        date_formats = [
            "%B %d, %Y",
            "%B %d %Y",
            "%m/%d/%Y",
            "%Y-%m-%d",
            "%b %d, %Y",
            "%b %d %Y",
        ]

        for fmt in date_formats:
            try:
                date = datetime.strptime(date_str.strip(), fmt)
                delta = date - datetime.utcnow()
                return delta.days
            except ValueError:
                continue

        return None

    def _identify_risk_factors(
        self, clauses: List[Clause], obligations: List[Obligation]
    ) -> List[str]:
        """Identify specific risk factors in the contract.

        Args:
            clauses: Extracted clauses.
            obligations: Extracted obligations.

        Returns:
            List of risk factor descriptions.
        """
        risk_factors = []
        found_types = {c.clause_type for c in clauses}

        for standard in self._STANDARD_CLAUSES:
            if standard not in found_types:
                risk_factors.append(
                    f"Missing standard clause: {standard.value}"
                )

        if obligations:
            parties = [o.party.lower() for o in obligations]
            unique_parties = set(parties)
            if len(unique_parties) == 1 and len(obligations) > 2:
                risk_factors.append(
                    "One-sided obligations detected - all obligations fall on one party"
                )

        if ClauseType.FORCE_MAJEURE not in found_types:
            risk_factors.append("No force majeure clause - risk in unforeseen events")

        return risk_factors

    def _generate_recommendations(
        self, clauses: List[Clause], risk_factors: List[str]
    ) -> List[str]:
        """Generate recommendations based on the analysis.

        Args:
            clauses: Extracted clauses.
            risk_factors: Identified risk factors.

        Returns:
            List of recommendation strings.
        """
        recommendations = []

        found_types = {c.clause_type for c in clauses}

        if ClauseType.LIMITATION_OF_LIABILITY not in found_types:
            recommendations.append(
                "Add a limitation of liability clause to cap potential damages"
            )

        if ClauseType.DISPUTE_RESOLUTION not in found_types:
            recommendations.append(
                "Include a dispute resolution mechanism (arbitration or mediation)"
            )

        if ClauseType.FORCE_MAJEURE not in found_types:
            recommendations.append(
                "Add a force majeure clause to address unforeseeable events"
            )

        if risk_factors:
            recommendations.append(
                "Review identified risk factors with legal counsel before signing"
            )

        return recommendations
