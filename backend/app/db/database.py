from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from app.core.config import settings


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        settings.ensure_directories()
        connection = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    content_type TEXT,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    current_agent TEXT,
                    input_path TEXT NOT NULL,
                    result_json TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def create_task(self, task_id: str, filename: str, content_type: str | None, input_path: str) -> None:
        now = utcnow()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO tasks (
                    id, filename, content_type, status, stage, progress,
                    current_agent, input_path, result_json, error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    filename,
                    content_type,
                    "queued",
                    "queued",
                    0.0,
                    None,
                    input_path,
                    None,
                    None,
                    now,
                    now,
                ),
            )
            connection.commit()

    def update_task(self, task_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = utcnow()
        keys = list(fields.keys())
        values = [fields[key] for key in keys]
        assignments = ", ".join(f"{key} = ?" for key in keys)
        with self.connect() as connection:
            connection.execute(
                f"UPDATE tasks SET {assignments} WHERE id = ?",
                (*values, task_id),
            )
            connection.commit()

    def save_result(self, task_id: str, result: dict[str, Any]) -> None:
        self.update_task(
            task_id,
            result_json=json.dumps(result, ensure_ascii=False),
            status="completed",
            stage="completed",
            progress=1.0,
            current_agent="decision_agent",
            error_message=None,
        )

    def fail_task(self, task_id: str, error_message: str) -> None:
        self.update_task(
            task_id,
            status="failed",
            stage="failed",
            progress=1.0,
            error_message=error_message,
        )

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return dict(row) if row else None

    def list_tasks(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]


database = Database(str(settings.database_path))
database.initialize()
