"""Batch document processing module with concurrency control."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from src.batch.progress_tracker import (
    BatchProgress,
    ItemStatus,
    JobStatus,
    ProgressTracker,
)
from src.utils.config import get_settings

logger = logging.getLogger(__name__)


class BatchResult(BaseModel):
    """Result of a batch processing operation."""

    job_id: str = Field(..., description="Batch job identifier")
    total_documents: int = Field(default=0, description="Total documents processed")
    successful: int = Field(default=0, description="Successfully processed")
    failed: int = Field(default=0, description="Failed to process")
    results: List[Dict[str, Any]] = Field(
        default_factory=list, description="Individual results"
    )
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Error details"
    )
    processing_time_ms: float = Field(
        default=0.0, description="Total processing time in milliseconds"
    )


class BatchProcessor:
    """Processes multiple documents concurrently with progress tracking.

    Uses asyncio.Semaphore for concurrency control and integrates
    with ProgressTracker for persistent progress state.
    """

    def __init__(
        self,
        max_concurrent: Optional[int] = None,
        tracker: Optional[ProgressTracker] = None,
    ) -> None:
        """Initialize the batch processor.

        Args:
            max_concurrent: Maximum concurrent document processing tasks.
                           Defaults to value from config.
            tracker: Optional ProgressTracker instance. Creates one if not provided.
        """
        settings = get_settings()
        batch_config = settings.batch_processing or {}

        self._max_concurrent = max_concurrent or batch_config.get("max_concurrent", 5)
        self._tracker = tracker or ProgressTracker()
        self._cancellation_flags: Dict[str, bool] = {}

    async def process_batch(
        self,
        file_paths: List[str],
        mode: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
    ) -> BatchResult:
        """Process a batch of documents concurrently.

        Args:
            file_paths: List of file paths to process.
            mode: Extraction mode ('local', 'api', 'hybrid').
            options: Additional processing options.
            job_id: Optional pre-created job ID. If not provided, a new one is generated.

        Returns:
            BatchResult with aggregated results.
        """
        options = options or {}
        if job_id is None:
            job_id = str(uuid4())
            # Create the job in the tracker only if we generated the ID
            self._tracker.create_job(job_id, len(file_paths))
        start_time = time.time()
        self._cancellation_flags[job_id] = False

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self._max_concurrent)

        # Process all documents concurrently
        tasks = [
            self._process_single(file_path, mode, options, semaphore, job_id, idx)
            for idx, file_path in enumerate(file_paths)
        ]

        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        successful = 0
        failed = 0
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        for idx, result in enumerate(results_list):
            file_path = file_paths[idx]
            if isinstance(result, Exception):
                failed += 1
                errors.append(
                    {
                        "file_path": file_path,
                        "error": str(result),
                        "error_type": type(result).__name__,
                    }
                )
            elif isinstance(result, dict) and result.get("error"):
                failed += 1
                errors.append(result)
            else:
                successful += 1
                results.append(result if isinstance(result, dict) else {"file_path": file_path, "status": "success"})

        # Mark job complete or failed
        if failed == len(file_paths):
            self._tracker.mark_failed(job_id, "All documents failed processing")
        else:
            self._tracker.mark_complete(job_id)

        processing_time_ms = (time.time() - start_time) * 1000

        # Clean up cancellation flag
        self._cancellation_flags.pop(job_id, None)

        return BatchResult(
            job_id=job_id,
            total_documents=len(file_paths),
            successful=successful,
            failed=failed,
            results=results,
            errors=errors,
            processing_time_ms=processing_time_ms,
        )

    async def _process_single(
        self,
        file_path: str,
        mode: Optional[str],
        options: Dict[str, Any],
        semaphore: asyncio.Semaphore,
        job_id: str,
        idx: int,
    ) -> Dict[str, Any]:
        """Process a single document with semaphore-based concurrency control.

        Args:
            file_path: Path to the document.
            mode: Extraction mode.
            options: Processing options.
            semaphore: Concurrency control semaphore.
            job_id: Job identifier for progress tracking.
            idx: Index of this item in the batch.

        Returns:
            Dictionary with processing result or error info.
        """
        item_id = f"item_{idx}_{Path(file_path).name}"

        # Check for cancellation
        if self._cancellation_flags.get(job_id, False):
            self._tracker.update_progress(
                job_id, item_id, ItemStatus.SKIPPED.value
            )
            return {"file_path": file_path, "status": "skipped", "reason": "cancelled"}

        async with semaphore:
            try:
                # Update progress to processing
                self._tracker.update_progress(
                    job_id, item_id, ItemStatus.PROCESSING.value
                )

                # Import orchestrator here to avoid circular imports
                from src.agent.orchestrator import DocumentAgent

                agent = DocumentAgent()
                result = await agent.process_document(
                    file_path=file_path,
                    mode=mode,
                    options=options,
                )

                # Mark as completed
                self._tracker.update_progress(
                    job_id, item_id, ItemStatus.COMPLETED.value
                )

                return {
                    "file_path": file_path,
                    "status": "success",
                    "document_id": str(result.document_id),
                    "entities_count": len(result.entities),
                    "confidence": result.confidence_report.overall_confidence,
                }

            except Exception as e:
                logger.error(f"Batch item failed: {file_path} - {e}")
                self._tracker.update_progress(
                    job_id, item_id, ItemStatus.FAILED.value, error=str(e)
                )
                return {
                    "file_path": file_path,
                    "status": "failed",
                    "error": str(e),
                    "error_type": type(e).__name__,
                }

    def get_batch_status(self, job_id: str) -> BatchProgress:
        """Get the current status of a batch job.

        Args:
            job_id: Job identifier.

        Returns:
            BatchProgress with current state.
        """
        return self._tracker.get_progress(job_id)

    def cancel_batch(self, job_id: str) -> None:
        """Cancel a running batch job.

        Sets a cancellation flag that will be checked by pending items.

        Args:
            job_id: Job identifier to cancel.
        """
        self._cancellation_flags[job_id] = True
        logger.info(f"Batch job {job_id} cancellation requested")
