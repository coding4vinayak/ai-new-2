"""Tests for the batch processor module."""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.batch.batch_processor import BatchProcessor, BatchResult
from src.batch.progress_tracker import JobStatus, ProgressTracker


@pytest.fixture
def temp_text_files():
    """Create temporary text files for batch testing."""
    files = []
    for i in range(3):
        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w") as f:
            f.write(f"Test document {i}. This is sample content for testing.")
        files.append(path)
    yield files
    # Cleanup
    for path in files:
        if os.path.exists(path):
            os.remove(path)


@pytest.fixture
def tracker(tmp_path):
    """Create a ProgressTracker with a temporary database."""
    db_path = str(tmp_path / "test_progress.db")
    return ProgressTracker(database_path=db_path)


@pytest.fixture
def processor(tracker):
    """Create a BatchProcessor with custom tracker."""
    return BatchProcessor(max_concurrent=2, tracker=tracker)


def test_batch_job_creation(tracker):
    """Test that batch jobs can be created in the progress tracker."""
    job_id = str(uuid4())
    job = tracker.create_job(job_id, total_items=5)

    assert job.id == job_id
    assert job.total_items == 5
    assert job.status == JobStatus.RUNNING


def test_progress_tracking(tracker):
    """Test progress tracking updates correctly."""
    job_id = str(uuid4())
    tracker.create_job(job_id, total_items=3)

    tracker.update_progress(job_id, "item_0", "completed")
    tracker.update_progress(job_id, "item_1", "completed")
    tracker.update_progress(job_id, "item_2", "failed", error="test error")

    progress = tracker.get_progress(job_id)
    assert progress.completed_items == 2
    assert progress.failed_items == 1
    assert progress.progress_pct > 0


@pytest.mark.asyncio
async def test_batch_processing_with_mock(processor, temp_text_files):
    """Test batch processing with mocked orchestrator."""
    from src.models.confidence import ConfidenceReport
    from src.models.extraction_result import ExtractionMode, ExtractionResult

    mock_result = ExtractionResult(
        document_id=uuid4(),
        extraction_mode=ExtractionMode.LOCAL,
        entities={"test": "data"},
        confidence_report=ConfidenceReport(overall_confidence=0.8, threshold=0.7),
        raw_text="test",
        processing_time_ms=10.0,
    )

    with patch("src.agent.orchestrator.DocumentAgent") as mock_agent_class:
        mock_instance = AsyncMock()
        mock_instance.process_document = AsyncMock(return_value=mock_result)
        mock_agent_class.return_value = mock_instance

        result = await processor.process_batch(
            file_paths=temp_text_files,
            mode="local",
        )

        assert isinstance(result, BatchResult)
        assert result.total_documents == 3
        assert result.successful == 3
        assert result.failed == 0


@pytest.mark.asyncio
async def test_batch_cancellation(processor):
    """Test that batch jobs can be cancelled."""
    job_id = "test-cancel-job"
    processor._cancellation_flags[job_id] = False

    # Cancel the job
    processor.cancel_batch(job_id)

    assert processor._cancellation_flags[job_id] is True


@pytest.mark.asyncio
async def test_batch_error_handling(processor, temp_text_files):
    """Test that individual failures in a batch are handled gracefully."""
    call_count = 0

    async def mock_process(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise ValueError("Simulated processing error")
        from src.models.confidence import ConfidenceReport
        from src.models.extraction_result import ExtractionMode, ExtractionResult

        return ExtractionResult(
            document_id=uuid4(),
            extraction_mode=ExtractionMode.LOCAL,
            entities={},
            confidence_report=ConfidenceReport(overall_confidence=0.8, threshold=0.7),
            raw_text="test",
            processing_time_ms=10.0,
        )

    with patch("src.agent.orchestrator.DocumentAgent") as mock_agent_class:
        mock_instance = MagicMock()
        mock_instance.process_document = mock_process
        mock_agent_class.return_value = mock_instance

        result = await processor.process_batch(
            file_paths=temp_text_files,
            mode="local",
        )

        assert result.total_documents == 3
        assert result.successful == 2
        assert result.failed == 1
        assert len(result.errors) == 1
