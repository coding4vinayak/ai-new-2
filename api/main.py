"""FastAPI application for the Document Intelligence Agent."""

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes.actions import router as actions_router
from api.routes.batch import router as batch_router
from api.routes.extract import router as extract_router
from api.routes.health import router as health_router
from api.routes.webhooks import router as webhooks_router

# Track app start time for uptime reporting
_start_time = time.time()


def get_uptime() -> float:
    """Get application uptime in seconds."""
    return time.time() - _start_time


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    # Startup: ensure data directories exist
    Path("data").mkdir(parents=True, exist_ok=True)
    Path("uploads").mkdir(parents=True, exist_ok=True)

    # Initialize audit database
    from src.utils.audit_logger import AuditLogger

    AuditLogger()

    yield

    # Shutdown: cleanup if needed


app = FastAPI(
    title="Document Intelligence Agent API",
    description=(
        "AI-powered document intelligence system supporting multiple extraction modes "
        "(Local NLP, API LLM, Hybrid), batch processing, industry-specific analysis "
        "(contracts, invoices, compliance, PII detection), and configurable action rules."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router, tags=["Health"])
app.include_router(extract_router, prefix="/extract", tags=["Extraction"])
app.include_router(batch_router, prefix="/batch", tags=["Batch Processing"])
app.include_router(actions_router, prefix="/actions", tags=["Actions"])
app.include_router(webhooks_router, prefix="/webhooks", tags=["Webhooks"])


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    """Handle file not found errors."""
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle value errors (e.g., unsupported file types)."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )
