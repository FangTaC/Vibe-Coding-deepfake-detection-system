from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from app.db.database import database
from app.schemas.task import (
    DetectionResult,
    ReviewLabel,
    TaskStage,
    TaskStatus,
    TaskStatusResponse,
    TaskSummaryResponse,
)


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


class TaskRepository:
    def create(self, task_id: str, filename: str, content_type: str | None, input_path: str) -> None:
        database.create_task(task_id, filename, content_type, input_path)

    def update_progress(
        self,
        task_id: str,
        *,
        status: TaskStatus,
        stage: TaskStage,
        progress: float,
        current_agent: str | None,
    ) -> None:
        database.update_task(
            task_id,
            status=status.value,
            stage=stage.value,
            progress=progress,
            current_agent=current_agent,
        )

    def save_result(self, task_id: str, result: DetectionResult) -> None:
        database.save_result(task_id, result.model_dump(mode="json"))

    def fail(self, task_id: str, error_message: str) -> None:
        database.fail_task(task_id, error_message)

    def get_raw(self, task_id: str) -> dict[str, Any] | None:
        return database.get_task(task_id)

    def get_status(self, task_id: str) -> TaskStatusResponse | None:
        record = database.get_task(task_id)
        if not record:
            return None
        return TaskStatusResponse(
            task_id=record["id"],
            status=TaskStatus(record["status"]),
            stage=TaskStage(record["stage"]),
            progress=float(record["progress"]),
            current_agent=record["current_agent"],
            created_at=parse_datetime(record["created_at"]),
            updated_at=parse_datetime(record["updated_at"]),
            error_message=record["error_message"],
        )

    def get_result(self, task_id: str) -> DetectionResult | None:
        record = database.get_task(task_id)
        if not record or not record["result_json"]:
            return None
        payload = json.loads(record["result_json"])
        return DetectionResult.model_validate(payload)

    def list_tasks(self, limit: int = 20) -> list[TaskSummaryResponse]:
        records = database.list_tasks(limit=limit)
        summaries: list[TaskSummaryResponse] = []
        for record in records:
            label: ReviewLabel | None = None
            confidence: float | None = None
            if record["result_json"]:
                payload = json.loads(record["result_json"])
                label = ReviewLabel(payload["label"])
                confidence = payload.get("confidence")
            summaries.append(
                TaskSummaryResponse(
                    task_id=record["id"],
                    filename=record["filename"],
                    status=TaskStatus(record["status"]),
                    stage=TaskStage(record["stage"]),
                    progress=float(record["progress"]),
                    current_agent=record["current_agent"],
                    created_at=parse_datetime(record["created_at"]),
                    updated_at=parse_datetime(record["updated_at"]),
                    error_message=record["error_message"],
                    label=label,
                    confidence=confidence,
                )
            )
        return summaries


task_repository = TaskRepository()

