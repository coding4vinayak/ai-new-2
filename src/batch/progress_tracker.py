"""Progress tracking module for batch processing jobs."""

import json
import sqlite3
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Status of a batch job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ItemStatus(str, Enum):
    """Status of a single item in a batch job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class BatchJob(BaseModel):
    """Represents a batch processing job."""

    id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(default=JobStatus.PENDING, description="Job status")
    total_items: int = Field(default=0, description="Total items in the batch")
    completed_items: int = Field(default=0, description="Successfully completed items")
    failed_items: int = Field(default=0, description="Failed items")
    start_time: Optional[str] = Field(None, description="Job start time (ISO format)")
    end_time: Optional[str] = Field(None, description="Job end time (ISO format)")
    progress_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Progress percentage"
    )
    error: Optional[str] = Field(None, description="Error message if job failed")


class BatchProgress(BaseModel):
    """Current progress information for a batch job."""

    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Current job status")
    total_items: int = Field(default=0, description="Total items")
    completed_items: int = Field(default=0, description="Completed items")
    failed_items: int = Field(default=0, description="Failed items")
    progress_pct: float = Field(default=0.0, description="Progress percentage")
    elapsed_seconds: Optional[float] = Field(
        None, description="Elapsed time in seconds"
    )
    estimated_remaining_seconds: Optional[float] = Field(
        None, description="Estimated remaining time"
    )
    items: Dict[str, str] = Field(
        default_factory=dict,
        description="Item statuses (item_id -> status)",
    )


class ProgressTracker:
    """Tracks batch job progress using SQLite for persistence.

    Stores job and item state in SQLite so progress survives restarts.
    """

    def __init__(self, database_path: str = "data/batch_progress.db") -> None:
        """Initialize the progress tracker.

        Args:
            database_path: Path to the SQLite database file.
        """
        self.db_path = Path(database_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS batch_jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'pending',
                    total_items INTEGER NOT NULL DEFAULT 0,
                    completed_items INTEGER NOT NULL DEFAULT 0,
                    failed_items INTEGER NOT NULL DEFAULT 0,
                    start_time TEXT,
                    end_time TEXT,
                    error TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS batch_items (
                    job_id TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    error TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, item_id),
                    FOREIGN KEY (job_id) REFERENCES batch_jobs(id)
                )
            """)
            conn.commit()

    def create_job(self, job_id: str, total_items: int) -> BatchJob:
        """Create a new batch job.

        Args:
            job_id: Unique identifier for the job.
            total_items: Total number of items to process.

        Returns:
            BatchJob model representing the created job.
        """
        start_time = datetime.utcnow().isoformat()

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO batch_jobs
                (id, status, total_items, completed_items, failed_items, start_time)
                VALUES (?, ?, ?, 0, 0, ?)
                """,
                (job_id, JobStatus.RUNNING.value, total_items, start_time),
            )
            conn.commit()

        return BatchJob(
            id=job_id,
            status=JobStatus.RUNNING,
            total_items=total_items,
            completed_items=0,
            failed_items=0,
            start_time=start_time,
            progress_pct=0.0,
        )

    def update_progress(
        self, job_id: str, item_id: str, status: str, error: Optional[str] = None
    ) -> None:
        """Update progress for a specific item in a batch job.

        Args:
            job_id: Job identifier.
            item_id: Item identifier.
            status: New status for the item.
            error: Error message if the item failed.
        """
        now = datetime.utcnow().isoformat()

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO batch_items (job_id, item_id, status, error, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (job_id, item_id, status, error, now),
            )

            # Update job counters
            if status == ItemStatus.COMPLETED.value:
                conn.execute(
                    "UPDATE batch_jobs SET completed_items = completed_items + 1 WHERE id = ?",
                    (job_id,),
                )
            elif status == ItemStatus.FAILED.value:
                conn.execute(
                    "UPDATE batch_jobs SET failed_items = failed_items + 1 WHERE id = ?",
                    (job_id,),
                )

            conn.commit()

    def get_progress(self, job_id: str) -> BatchProgress:
        """Get current progress for a batch job.

        Args:
            job_id: Job identifier.

        Returns:
            BatchProgress with current state.

        Raises:
            ValueError: If the job is not found.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM batch_jobs WHERE id = ?", (job_id,)
            )
            row = cursor.fetchone()

            if not row:
                raise ValueError(f"Job not found: {job_id}")

            job_data = dict(row)

            # Get item statuses
            items_cursor = conn.execute(
                "SELECT item_id, status FROM batch_items WHERE job_id = ?",
                (job_id,),
            )
            items = {r["item_id"]: r["status"] for r in items_cursor.fetchall()}

        total = job_data["total_items"]
        completed = job_data["completed_items"]
        failed = job_data["failed_items"]
        progress_pct = ((completed + failed) / total * 100.0) if total > 0 else 0.0

        # Compute elapsed time
        elapsed_seconds = None
        estimated_remaining = None
        if job_data["start_time"]:
            start = datetime.fromisoformat(job_data["start_time"])
            elapsed = (datetime.utcnow() - start).total_seconds()
            elapsed_seconds = elapsed

            processed = completed + failed
            if processed > 0:
                rate = elapsed / processed
                remaining = total - processed
                estimated_remaining = rate * remaining

        return BatchProgress(
            job_id=job_id,
            status=JobStatus(job_data["status"]),
            total_items=total,
            completed_items=completed,
            failed_items=failed,
            progress_pct=round(progress_pct, 1),
            elapsed_seconds=elapsed_seconds,
            estimated_remaining_seconds=estimated_remaining,
            items=items,
        )

    def mark_complete(self, job_id: str) -> None:
        """Mark a batch job as completed.

        Args:
            job_id: Job identifier.
        """
        end_time = datetime.utcnow().isoformat()
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "UPDATE batch_jobs SET status = ?, end_time = ? WHERE id = ?",
                (JobStatus.COMPLETED.value, end_time, job_id),
            )
            conn.commit()

    def mark_failed(self, job_id: str, error: str) -> None:
        """Mark a batch job as failed.

        Args:
            job_id: Job identifier.
            error: Error description.
        """
        end_time = datetime.utcnow().isoformat()
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "UPDATE batch_jobs SET status = ?, end_time = ?, error = ? WHERE id = ?",
                (JobStatus.FAILED.value, end_time, error, job_id),
            )
            conn.commit()
