"""Compliance checking module for regulatory analysis."""

import re
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """Compliance risk levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceFinding(BaseModel):
    """A single compliance finding."""

    category: str = Field(..., description="Category of the finding")
    description: str = Field(..., description="Description of the finding")
    severity: RiskLevel = Field(..., description="Severity of the finding")
    recommendation: str = Field(
        default="", description="Recommended action to address the finding"
    )
    clause_reference: Optional[str] = Field(
        None, description="Reference to related clause"
    )


class ComplianceReport(BaseModel):
    """Complete compliance analysis report."""

    compliant: bool = Field(..., description="Whether the document is compliant")
    findings: List[ComplianceFinding] = Field(
        default_factory=list, description="Compliance findings"
    )
    missing_clauses: List[str] = Field(
        default_factory=list, description="Required clauses that are missing"
    )
    risk_level: RiskLevel = Field(
        default=RiskLevel.LOW, description="Overall risk level"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Compliance recommendations"
    )
    jurisdiction: Optional[str] = Field(
        None, description="Jurisdiction checked against"
    )


class ComplianceChecker:
    """Checks documents for regulatory compliance.

    Analyzes documents against regulatory requirements based on
    document type and jurisdiction.
    """

    # Required clauses by document type
    _REQUIRED_CLAUSES: Dict[str, List[Dict[str, str]]] = {
        "CONTRACT": [
            {
                "name": "Data Processing Terms",
                "patterns": "data process|data protection|personal data",
                "category": "GDPR",
            },
            {
                "name": "Termination Rights",
                "patterns": "terminat|cancel|end of agreement",
                "category": "General",
            },
            {
                "name": "Governing Law",
                "patterns": "governing law|governed by|jurisdiction",
                "category": "General",
            },
            {
                "name": "Dispute Resolution",
                "patterns": "dispute resolution|arbitration|mediation",
                "category": "General",
            },
            {
                "name": "Limitation of Liability",
                "patterns": "limitation of liability|limit.*liab",
                "category": "General",
            },
            {
                "name": "Confidentiality",
                "patterns": "confidential|non-disclosure|proprietary",
                "category": "General",
            },
        ],
        "INVOICE": [
            {
                "name": "Tax Identification",
                "patterns": "tax id|tin|vat|ein|gst",
                "category": "Tax",
            },
            {
                "name": "Payment Terms",
                "patterns": "payment terms|net \\d+|due date",
                "category": "Financial",
            },
        ],
        "HR": [
            {
                "name": "Equal Opportunity Statement",
                "patterns": "equal opportunity|non-discrimination|diversity",
                "category": "Employment Law",
            },
            {
                "name": "At-Will Employment",
                "patterns": "at-will|at will",
                "category": "Employment Law",
            },
        ],
    }

    # GDPR-specific requirements
    _GDPR_REQUIREMENTS = [
        {
            "name": "Data Processing Purpose",
            "patterns": "purpose of processing|processing purpose|lawful basis",
            "description": "Must specify the purpose of data processing",
        },
        {
            "name": "Data Retention Period",
            "patterns": "retention period|data retention|stored for|retain.*data",
            "description": "Must specify data retention periods",
        },
        {
            "name": "Data Subject Rights",
            "patterns": "right to access|right to erasure|right to rectification|data subject rights|right to be forgotten",
            "description": "Must reference data subject rights",
        },
        {
            "name": "Data Breach Notification",
            "patterns": "data breach|breach notification|notify.*breach",
            "description": "Must include data breach notification procedures",
        },
        {
            "name": "Data Transfer",
            "patterns": "data transfer|cross-border|international transfer|standard contractual clauses",
            "description": "Must address international data transfers",
        },
    ]

    def check_compliance(
        self,
        document_text: str,
        doc_type: str = "CONTRACT",
        jurisdiction: Optional[str] = None,
    ) -> ComplianceReport:
        """Check a document for compliance with regulations.

        Args:
            document_text: The document text to check.
            doc_type: Document type (CONTRACT, INVOICE, HR, etc.).
            jurisdiction: Jurisdiction to check against (e.g., "EU", "US").

        Returns:
            ComplianceReport with findings and recommendations.
        """
        findings: List[ComplianceFinding] = []
        missing_clauses: List[str] = []
        recommendations: List[str] = []

        # Check required clauses for document type
        type_findings, type_missing = self._check_required_clauses(
            document_text, doc_type
        )
        findings.extend(type_findings)
        missing_clauses.extend(type_missing)

        # Check contract-specific compliance
        if doc_type.upper() == "CONTRACT":
            contract_findings = self._check_contract_compliance(document_text)
            findings.extend(contract_findings)

        # Check GDPR compliance if applicable
        if jurisdiction and jurisdiction.upper() in ("EU", "GDPR", "EEA"):
            gdpr_findings = self._check_gdpr_compliance(document_text)
            findings.extend(gdpr_findings)

        # Determine overall risk level
        risk_level = self._assess_risk_level(findings, missing_clauses)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            findings, missing_clauses, doc_type
        )

        # Determine compliance
        compliant = risk_level in (RiskLevel.LOW,) and not missing_clauses

        return ComplianceReport(
            compliant=compliant,
            findings=findings,
            missing_clauses=missing_clauses,
            risk_level=risk_level,
            recommendations=recommendations,
            jurisdiction=jurisdiction,
        )

    def _check_required_clauses(
        self, text: str, doc_type: str
    ) -> tuple:
        """Check for required clauses based on document type.

        Args:
            text: Document text.
            doc_type: Document type string.

        Returns:
            Tuple of (findings, missing_clause_names).
        """
        findings: List[ComplianceFinding] = []
        missing: List[str] = []

        required = self._REQUIRED_CLAUSES.get(doc_type.upper(), [])

        for clause_req in required:
            found = self._detect_clause(text, clause_req["patterns"])
            if not found:
                missing.append(clause_req["name"])
                findings.append(
                    ComplianceFinding(
                        category=clause_req["category"],
                        description=f"Required clause missing: {clause_req['name']}",
                        severity=RiskLevel.HIGH,
                        recommendation=f"Add {clause_req['name']} clause to the document",
                    )
                )

        return findings, missing

    def _check_contract_compliance(self, text: str) -> List[ComplianceFinding]:
        """Check contract-specific compliance requirements.

        Args:
            text: Contract text to analyze.

        Returns:
            List of compliance findings.
        """
        findings: List[ComplianceFinding] = []

        # Check for clear party identification
        party_patterns = [
            r"(?:between|by and between)\s+.*?(?:and|&)\s+",
            r"Party\s+[AB]",
            r"(?:Buyer|Seller|Licensor|Licensee)",
        ]
        party_found = any(
            re.search(p, text, re.IGNORECASE) for p in party_patterns
        )
        if not party_found:
            findings.append(
                ComplianceFinding(
                    category="Contract Formation",
                    description="Parties to the agreement are not clearly identified",
                    severity=RiskLevel.HIGH,
                    recommendation="Clearly identify all parties to the agreement",
                )
            )

        # Check for effective date
        date_patterns = [
            r"effective\s+(?:date|as\s+of)",
            r"dated\s+(?:this|as\s+of)",
            r"commenc(?:e|ing)\s+on",
        ]
        date_found = any(
            re.search(p, text, re.IGNORECASE) for p in date_patterns
        )
        if not date_found:
            findings.append(
                ComplianceFinding(
                    category="Contract Formation",
                    description="No effective date specified",
                    severity=RiskLevel.MEDIUM,
                    recommendation="Specify the effective date of the agreement",
                )
            )

        # Check for signature block
        sig_patterns = [
            r"(?:signed|executed)\s+by",
            r"signature",
            r"IN WITNESS WHEREOF",
            r"authorized\s+representative",
        ]
        sig_found = any(
            re.search(p, text, re.IGNORECASE) for p in sig_patterns
        )
        if not sig_found:
            findings.append(
                ComplianceFinding(
                    category="Contract Formation",
                    description="No signature block or execution clause found",
                    severity=RiskLevel.MEDIUM,
                    recommendation="Include proper signature blocks for all parties",
                )
            )

        return findings

    def _check_gdpr_compliance(self, text: str) -> List[ComplianceFinding]:
        """Check GDPR-specific compliance requirements.

        Args:
            text: Document text to analyze.

        Returns:
            List of GDPR compliance findings.
        """
        findings: List[ComplianceFinding] = []

        for req in self._GDPR_REQUIREMENTS:
            found = self._detect_clause(text, req["patterns"])
            if not found:
                findings.append(
                    ComplianceFinding(
                        category="GDPR",
                        description=f"Missing: {req['description']}",
                        severity=RiskLevel.HIGH,
                        recommendation=f"Add provisions for: {req['name']}",
                    )
                )

        return findings

    def _detect_clause(self, text: str, patterns: str) -> bool:
        """Detect if a clause pattern exists in the text.

        Args:
            text: Document text.
            patterns: Pipe-separated regex patterns.

        Returns:
            True if any pattern matches.
        """
        pattern_list = patterns.split("|")
        for pattern in pattern_list:
            if re.search(pattern.strip(), text, re.IGNORECASE):
                return True
        return False

    def _assess_risk_level(
        self,
        findings: List[ComplianceFinding],
        missing_clauses: List[str],
    ) -> RiskLevel:
        """Assess overall compliance risk level.

        Args:
            findings: All compliance findings.
            missing_clauses: Missing required clauses.

        Returns:
            Overall risk level.
        """
        critical_count = sum(
            1 for f in findings if f.severity == RiskLevel.CRITICAL
        )
        high_count = sum(
            1 for f in findings if f.severity == RiskLevel.HIGH
        )

        if critical_count > 0:
            return RiskLevel.CRITICAL
        if high_count >= 3 or len(missing_clauses) >= 3:
            return RiskLevel.HIGH
        if high_count >= 1 or len(missing_clauses) >= 1:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _generate_recommendations(
        self,
        findings: List[ComplianceFinding],
        missing_clauses: List[str],
        doc_type: str,
    ) -> List[str]:
        """Generate compliance recommendations.

        Args:
            findings: All findings.
            missing_clauses: Missing clauses.
            doc_type: Document type.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        if missing_clauses:
            recommendations.append(
                f"Add the following required clauses: {', '.join(missing_clauses)}"
            )

        high_findings = [
            f for f in findings if f.severity in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        ]
        if high_findings:
            recommendations.append(
                "Address high-severity findings before finalizing the document"
            )

        if doc_type.upper() == "CONTRACT":
            recommendations.append(
                "Have legal counsel review the document for jurisdiction-specific requirements"
            )

        return recommendations
