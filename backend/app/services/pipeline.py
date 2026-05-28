from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import traceback
import uuid
from pathlib import Path

from fastapi import HTTPException

from app.agents.decision_agent import DecisionAgent
from app.agents.feature_extraction_agent import FeatureExtractionAgent
from app.core.config import settings
from app.schemas.task import CreateTaskResponse, TaskStage, TaskStatus
from app.services.algorithm_router import algorithm_router
from app.services.repository import task_repository
from app.services.storage import storage_service


class DetectionOrchestrator:
    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(
            max_workers=max(1, settings.task_worker_count),
            thread_name_prefix="deepfake-task",
        )
        self.futures: dict[str, Future] = {}

    def validate_upload(self, filename: str, content: bytes) -> None:
        extension = Path(filename).suffix.lower()
        if extension not in settings.allowed_extensions:
            raise HTTPException(status_code=400, detail="仅支持 jpg/jpeg/png/webp 格式图像。")
        if len(content) > settings.max_upload_size_mb * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"图片大小不能超过 {settings.max_upload_size_mb}MB。")

    def create_task(self, filename: str, content_type: str | None, content: bytes) -> CreateTaskResponse:
        self.validate_upload(filename, content)
        task_id = uuid.uuid4().hex
        input_path = storage_service.save_upload(task_id, filename, content)
        task_repository.create(task_id, filename, content_type, str(input_path))
        return CreateTaskResponse(task_id=task_id, status=TaskStatus.queued)

    def run_task(self, task_id: str) -> None:
        record = task_repository.get_raw(task_id)
        if not record:
            return
        image_path = record["input_path"]
        feature_agent = FeatureExtractionAgent()
        decision_agent = DecisionAgent()
        try:
            task_repository.update_progress(
                task_id,
                status=TaskStatus.running,
                stage=TaskStage.preprocessing,
                progress=0.1,
                current_agent="feature_agent",
            )
            task_repository.update_progress(
                task_id,
                status=TaskStatus.running,
                stage=TaskStage.feature_extraction,
                progress=0.45,
                current_agent="feature_agent",
            )
            routing = algorithm_router.route(image_path)
            features = feature_agent.run(task_id, image_path, routing=routing)
            task_repository.update_progress(
                task_id,
                status=TaskStatus.running,
                stage=TaskStage.decision,
                progress=0.75,
                current_agent="decision_agent",
            )
            result = decision_agent.run(task_id, image_path, features, routing=routing)
            task_repository.save_result(task_id, result)
        except Exception as exc:
            task_repository.fail(task_id, f"{exc}\n{traceback.format_exc(limit=3)}")
        finally:
            self.futures.pop(task_id, None)

    def submit_task(self, task_id: str) -> None:
        future = self.futures.get(task_id)
        if future and not future.done():
            return
        self.futures[task_id] = self.executor.submit(self.run_task, task_id)


detection_orchestrator = DetectionOrchestrator()
