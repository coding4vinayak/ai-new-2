"""Extraction endpoints for document processing."""

import os
import tempfile
import time
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

router = APIRouter()

# Maximum upload size (50 MB)
_MAX_FILE_SIZE = 50 * 1024 * 1024


async def _save_upload(file: UploadFile) -> str:
    """Save an uploaded file to a temporary location and return the path.

    Raises:
        HTTPException: If the file exceeds the maximum allowed size.
    """
    os.makedirs("uploads", exist_ok=True)
    content = await file.read()
    if len(content) > _MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File size ({len(content)} bytes) exceeds maximum allowed size of {_MAX_FILE_SIZE // (1024 * 1024)} MB",
        )
    suffix = os.path.splitext(file.filename or "document.txt")[1]
    temp_path = os.path.join("uploads", f"{uuid4()}{suffix}")
    with open(temp_path, "wb") as f:
        f.write(content)
    return temp_path


async def _extract_with_mode(file: UploadFile, mode: str, document_type: Optional[str] = None) -> Dict[str, Any]:
    """Common extraction logic for all modes."""
    from src.agent.orchestrator import DocumentAgent

    file_path = await _save_upload(file)

    try:
        agent = DocumentAgent()
        options = {"skip_actions": False}
        if document_type:
            options["document_type"] = document_type

        result = await agent.process_document(
            file_path=file_path,
            mode=mode,
            options=options,
        )

        return {
            "document_id": str(result.document_id),
            "extraction_mode": result.extraction_mode.value,
            "entities": result.entities,
            "confidence": {
                "overall": result.confidence_report.overall_confidence,
                "scores": {
                    k: {"score": v.score, "method": v.method}
                    for k, v in result.confidence_report.scores.items()
                },
                "low_confidence_fields": result.confidence_report.low_confidence_fields,
            },
            "processing_time_ms": round(result.processing_time_ms, 2),
            "warnings": result.warnings,
        }
    finally:
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)


@router.post("/local")
async def extract_local(
    file: UploadFile = File(...),
    document_type: Optional[str] = Query(None, description="Document type hint"),
):
    """Extract entities using local NLP mode (spaCy + regex)."""
    return await _extract_with_mode(file, "local", document_type)


@router.post("/api")
async def extract_api(
    file: UploadFile = File(...),
    document_type: Optional[str] = Query(None, description="Document type hint"),
):
    """Extract entities using API mode (OpenAI GPT-4 / Anthropic Claude)."""
    return await _extract_with_mode(file, "api", document_type)


@router.post("/hybrid")
async def extract_hybrid(
    file: UploadFile = File(...),
    document_type: Optional[str] = Query(None, description="Document type hint"),
):
    """Extract entities using hybrid mode (local-first with API fallback)."""
    return await _extract_with_mode(file, "hybrid", document_type)


@router.post("/local-llm")
async def extract_local_llm(
    file: UploadFile = File(...),
    document_type: Optional[str] = Query(None, description="Document type hint"),
):
    """Extract entities using a local LLM via OpenAI-compatible endpoint.

    Supports Ollama, LM Studio, LocalAI, vLLM, and any OpenAI-compatible service.
    Configure the endpoint via LOCAL_LLM_BASE_URL and LOCAL_LLM_MODEL env vars.
    """
    return await _extract_with_mode(file, "local_llm", document_type)


@router.post("/auto")
async def extract_auto(
    file: UploadFile = File(...),
    document_type: Optional[str] = Query(None, description="Document type hint"),
):
    """Extract entities with auto-detected best mode based on document type.

    Uses hybrid mode by default for optimal accuracy/cost balance.
    For simple text files, uses local mode. For complex documents, uses hybrid.
    """
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()

    # Auto-select mode based on file type
    if ext in (".txt",):
        mode = "local"
    else:
        mode = "hybrid"

    return await _extract_with_mode(file, mode, document_type)
