"""Health and metrics endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Check application health status."""
    from api.main import get_uptime

    return {
        "status": "healthy",
        "version": "1.0.0",
        "uptime_seconds": round(get_uptime(), 1),
    }


@router.get("/metrics")
async def get_metrics():
    """Get basic application metrics from audit log."""
    try:
        from src.utils.audit_logger import AuditLogger

        logger = AuditLogger()
        recent = logger.get_recent_extractions(limit=1000)

        total_extractions = len(recent)
        avg_processing_time = 0.0
        extractions_by_mode = {}

        if recent:
            durations = [r["duration_ms"] for r in recent if r.get("duration_ms")]
            if durations:
                avg_processing_time = sum(durations) / len(durations)

            for record in recent:
                mode = record.get("mode", "unknown")
                extractions_by_mode[mode] = extractions_by_mode.get(mode, 0) + 1

        return {
            "total_extractions": total_extractions,
            "avg_processing_time_ms": round(avg_processing_time, 2),
            "extractions_by_mode": extractions_by_mode,
        }
    except Exception:
        return {
            "total_extractions": 0,
            "avg_processing_time_ms": 0.0,
            "extractions_by_mode": {},
        }
