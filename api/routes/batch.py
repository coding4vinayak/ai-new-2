"""Batch processing endpoints."""

import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile

from src.batch.batch_processor import BatchProcessor
from src.batch.progress_tracker import JobStatus

router = APIRouter()

# Module-level processor instance (shared across requests)
_processor = BatchProcessor()

# Store job metadata for tracking
_job_metadata: Dict[str, Dict[str, Any]] = {}

# Maximum upload size per file (50 MB)
_MAX_BATCH_FILE_SIZE = 50 * 1024 * 1024


async def _run_batch_in_background(
    file_paths: List[str], mode: str, job_id: str
) -> None:
    """Run batch processing as a background task.

    Args:
        file_paths: List of saved file paths to process.
        mode: Extraction mode.
        job_id: Pre-created job ID for tracking.
    """
    result = await _processor.process_batch(
        file_paths=file_paths,
        mode=mode,
    )

    # Store result metadata
    _job_metadata[job_id] = {
        "file_paths": file_paths,
        "mode": mode,
        "result": result.model_dump(),
    }


@router.post("/process")
async def start_batch_processing(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    mode: Optional[str] = Query("hybrid", description="Extraction mode"),
):
    """Start a batch processing job for multiple documents.

    Accepts multiple file uploads, saves them, and starts background processing.
    Returns a job_id immediately that can be used to track progress.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Save uploaded files with size validation
    os.makedirs("uploads", exist_ok=True)
    file_paths = []
    for file in files:
        content = await file.read()
        if len(content) > _MAX_BATCH_FILE_SIZE:
            # Clean up already-saved files
            for saved_path in file_paths:
                if os.path.exists(saved_path):
                    os.remove(saved_path)
            raise HTTPException(
                status_code=413,
                detail=f"File '{file.filename}' exceeds maximum size of {_MAX_BATCH_FILE_SIZE // (1024 * 1024)} MB",
            )
        suffix = os.path.splitext(file.filename or "document.txt")[1]
        temp_path = os.path.join("uploads", f"{uuid4()}{suffix}")
        with open(temp_path, "wb") as f:
            f.write(content)
        file_paths.append(temp_path)

    # Create a job in the tracker immediately so status can be polled
    job_id = str(uuid4())
    _processor._tracker.create_job(job_id, len(file_paths))

    # Initialize metadata entry so we can track that the job exists
    _job_metadata[job_id] = {
        "file_paths": file_paths,
        "mode": mode,
        "result": None,
    }

    # Launch processing as a background task
    background_tasks.add_task(_run_batch_in_background, file_paths, mode, job_id)

    return {
        "job_id": job_id,
        "status": "accepted",
        "total_documents": len(file_paths),
        "message": "Batch job accepted. Poll GET /batch/status/{job_id} for progress.",
    }


@router.get("/status/{job_id}")
async def get_batch_status(job_id: str):
    """Get the status of a batch processing job."""
    if job_id not in _job_metadata:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    try:
        progress = _processor.get_batch_status(job_id)
        return {
            "job_id": progress.job_id,
            "status": progress.status.value,
            "total_items": progress.total_items,
            "completed_items": progress.completed_items,
            "failed_items": progress.failed_items,
            "progress_pct": progress.progress_pct,
            "elapsed_seconds": progress.elapsed_seconds,
            "estimated_remaining_seconds": progress.estimated_remaining_seconds,
        }
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")


@router.get("/result/{job_id}")
async def get_batch_result(job_id: str):
    """Get the results of a completed batch processing job."""
    if job_id not in _job_metadata:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    metadata = _job_metadata[job_id]
    result = metadata.get("result")
    if result is None:
        raise HTTPException(
            status_code=202,
            detail="Job is still processing. Poll GET /batch/status/{job_id} for progress.",
        )
    return result


@router.delete("/{job_id}")
async def cancel_batch(job_id: str):
    """Cancel a running batch job."""
    try:
        _processor.cancel_batch(job_id)
        return {"job_id": job_id, "status": "cancellation_requested"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
