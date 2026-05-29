"""Batch processing endpoints."""

import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from src.batch.batch_processor import BatchProcessor
from src.batch.progress_tracker import JobStatus

router = APIRouter()

# Module-level processor instance (shared across requests)
_processor = BatchProcessor()

# Store job metadata for tracking
_job_metadata: Dict[str, Dict[str, Any]] = {}


@router.post("/process")
async def start_batch_processing(
    files: List[UploadFile] = File(...),
    mode: Optional[str] = Query("hybrid", description="Extraction mode"),
):
    """Start a batch processing job for multiple documents.

    Accepts multiple file uploads and starts asynchronous batch processing.
    Returns a job_id that can be used to track progress.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Save uploaded files
    os.makedirs("uploads", exist_ok=True)
    file_paths = []
    for file in files:
        suffix = os.path.splitext(file.filename or "document.txt")[1]
        temp_path = os.path.join("uploads", f"{uuid4()}{suffix}")
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        file_paths.append(temp_path)

    # Start batch processing
    result = await _processor.process_batch(
        file_paths=file_paths,
        mode=mode,
    )

    # Store metadata
    _job_metadata[result.job_id] = {
        "file_paths": file_paths,
        "mode": mode,
        "result": result.model_dump(),
    }

    return {
        "job_id": result.job_id,
        "status": "completed",
        "total_documents": result.total_documents,
        "successful": result.successful,
        "failed": result.failed,
        "processing_time_ms": round(result.processing_time_ms, 2),
    }


@router.get("/status/{job_id}")
async def get_batch_status(job_id: str):
    """Get the status of a batch processing job."""
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
    return metadata.get("result", {})


@router.delete("/{job_id}")
async def cancel_batch(job_id: str):
    """Cancel a running batch job."""
    try:
        _processor.cancel_batch(job_id)
        return {"job_id": job_id, "status": "cancellation_requested"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
