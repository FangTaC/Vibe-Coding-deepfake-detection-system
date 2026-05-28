from __future__ import annotations

from pathlib import Path

from app.core.config import settings


class StorageService:
    def __init__(self) -> None:
        settings.ensure_directories()

    def create_task_directories(self, task_id: str) -> tuple[Path, Path]:
        upload_dir = settings.uploads_dir / task_id
        artifact_dir = settings.artifacts_dir / task_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return upload_dir, artifact_dir

    def save_upload(self, task_id: str, filename: str, content: bytes) -> Path:
        upload_dir, _ = self.create_task_directories(task_id)
        file_path = upload_dir / filename
        file_path.write_bytes(content)
        return file_path

    def artifact_path(self, task_id: str, name: str) -> Path:
        _, artifact_dir = self.create_task_directories(task_id)
        return artifact_dir / Path(name).name

    def artifact_url(self, task_id: str, name: str) -> str:
        return f"{settings.api_prefix}/tasks/{task_id}/artifacts/{Path(name).name}"


storage_service = StorageService()

