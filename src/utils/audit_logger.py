"""Audit logging using SQLite for tracking extraction operations."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID


class AuditLogger:
    """SQLite-based audit logger for tracking document extractions and actions."""

    def __init__(self, database_url: str = "sqlite:///data/audit.db") -> None:
        """Initialize the audit logger.

        Args:
            database_url: SQLite database URL (format: sqlite:///path/to/db).
        """
        # Parse the SQLite URL to get the file path
        if database_url.startswith("sqlite:///"):
            db_path = database_url[len("sqlite:///"):]
        else:
            db_path = database_url

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS extractions (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_summary TEXT,
                    timestamp TEXT NOT NULL,
                    duration_ms REAL,
                    user_id TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS actions (
                    id TEXT PRIMARY KEY,
                    extraction_id TEXT,
                    action_type TEXT NOT NULL,
                    action_name TEXT NOT NULL,
                    payload TEXT,
                    status TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (extraction_id) REFERENCES extractions(id)
                )
            """)
            conn.commit()

    def log_extraction(
        self,
        extraction_id: str,
        document_id: str,
        mode: str,
        status: str,
        result_summary: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """Log an extraction event.

        Args:
            extraction_id: Unique extraction identifier.
            document_id: Document identifier.
            mode: Extraction mode used (local, api, hybrid).
            status: Status of the extraction (success, failed, partial).
            result_summary: Summary of the extraction result.
            duration_ms: Processing duration in milliseconds.
            user_id: Optional user identifier.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO extractions (id, document_id, mode, status, result_summary, timestamp, duration_ms, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    extraction_id,
                    document_id,
                    mode,
                    status,
                    json.dumps(result_summary) if result_summary else None,
                    datetime.utcnow().isoformat(),
                    duration_ms,
                    user_id,
                ),
            )
            conn.commit()

    def log_action(
        self,
        action_id: str,
        extraction_id: Optional[str],
        action_type: str,
        action_name: str,
        payload: Optional[Dict[str, Any]] = None,
        status: str = "triggered",
    ) -> None:
        """Log an action event.

        Args:
            action_id: Unique action identifier.
            extraction_id: Related extraction identifier.
            action_type: Type of action (webhook, notification, etc.).
            action_name: Name of the action rule.
            payload: Action payload data.
            status: Status of the action.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO actions (id, extraction_id, action_type, action_name, payload, status, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    action_id,
                    extraction_id,
                    action_type,
                    action_name,
                    json.dumps(payload) if payload else None,
                    status,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

    def get_audit_trail(self, document_id: str) -> List[Dict[str, Any]]:
        """Get the full audit trail for a document.

        Args:
            document_id: Document identifier.

        Returns:
            List of audit records for the document.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM extractions WHERE document_id = ? ORDER BY timestamp DESC
                """,
                (document_id,),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_recent_extractions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent extraction records.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of recent extraction records.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM extractions ORDER BY timestamp DESC LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
