"""Tests for the contract analyzer module."""

from pathlib import Path
from uuid import uuid4

import pytest

from src.industry.contract_analyzer import (
    ClauseType,
    ContractAnalyzer,
)
from src.models.confidence import ConfidenceReport
from src.models.extraction_result import ExtractionMode, ExtractionResult


@pytest.fixture
def sample_contract_text():
    """Load sample contract text for testing."""
    sample_path = Path(__file__).parent / "sample_docs" / "sample_contract.txt"
    return sample_path.read_text(encoding="utf-8")


@pytest.fixture
def extraction_result(sample_contract_text):
    """Create an extraction result from the sample contract."""
    return ExtractionResult(
        document_id=uuid4(),
        extraction_mode=ExtractionMode.LOCAL,
        entities={
            "party_names": ["TechVision Solutions Inc.", "GlobalRetail Corp."],
            "effective_date": "January 15, 2024",
        },
        confidence_report=ConfidenceReport(overall_confidence=0.85, threshold=0.7),
        raw_text=sample_contract_text,
        processing_time_ms=100.0,
    )


@pytest.fixture
def analyzer():
    """Create a ContractAnalyzer instance."""
    return ContractAnalyzer()


def test_clause_extraction(analyzer, sample_contract_text):
    """Test that clauses are properly extracted from the contract."""
    clauses = analyzer.extract_clauses(sample_contract_text)

    assert len(clauses) > 0

    # Check for expected clause types
    clause_types = {c.clause_type for c in clauses}
    assert ClauseType.TERMINATION in clause_types
    assert ClauseType.CONFIDENTIALITY in clause_types
    assert ClauseType.GOVERNING_LAW in clause_types
    assert ClauseType.PAYMENT in clause_types
    assert ClauseType.INDEMNITY in clause_types


def test_risk_scoring(analyzer, extraction_result):
    """Test risk score calculation for the sample contract."""
    analysis = analyzer.analyze(extraction_result)

    # Risk score should be between 0 and 1
    assert 0.0 <= analysis.risk_score <= 1.0

    # Contract has most standard clauses, so risk should be relatively low
    # (the sample contract is fairly complete)
    assert analysis.risk_score < 0.8


def test_obligation_detection(analyzer, sample_contract_text):
    """Test that obligations are detected from the contract text."""
    obligations = analyzer.extract_obligations(sample_contract_text)

    # The sample contract contains obligations for both parties
    assert len(obligations) > 0

    # Check that obligation fields are populated
    for obligation in obligations:
        assert obligation.party != ""
        assert obligation.description != ""


def test_expiry_date_tracking(analyzer, extraction_result):
    """Test that key dates are identified and tracked."""
    key_dates = analyzer.track_expiry_dates(extraction_result)

    # The sample contract has effective and expiry dates
    assert len(key_dates) > 0

    # Check that date entries have descriptions and date strings
    for date_entry in key_dates:
        assert date_entry.description != ""
        assert date_entry.date_str != ""


def test_full_analysis(analyzer, extraction_result):
    """Test the complete analysis pipeline."""
    analysis = analyzer.analyze(extraction_result)

    assert analysis.clauses is not None
    assert analysis.obligations is not None
    assert analysis.key_dates is not None
    assert analysis.risk_score >= 0.0
    assert isinstance(analysis.risk_factors, list)
    assert isinstance(analysis.recommendations, list)
