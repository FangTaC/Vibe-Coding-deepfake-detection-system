from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from app.core.config import settings
from app.schemas.task import CreateTaskResponse, DetectionResult, TaskStatusResponse, TaskSummaryResponse
from app.services.model_registry import model_registry
from app.services.pipeline import detection_orchestrator
from app.services.repository import task_repository
from app.services.storage import storage_service


router = APIRouter(prefix=settings.api_prefix)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/status")
def system_status() -> dict:
    return model_registry.build_runtime_status()


@router.get("/tasks", response_model=list[TaskSummaryResponse])
def list_tasks(limit: int = Query(default=20, ge=1, le=100)) -> list[TaskSummaryResponse]:
    return task_repository.list_tasks(limit=limit)


@router.post("/tasks", response_model=CreateTaskResponse)
async def create_task(
    file: UploadFile = File(...),
) -> CreateTaskResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="上传文件缺少文件名。")
    content = await file.read()
    response = detection_orchestrator.create_task(file.filename, file.content_type, content)
    detection_orchestrator.submit_task(response.task_id)
    return response


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str) -> TaskStatusResponse:
    status = task_repository.get_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="任务不存在。")
    return status


@router.get("/tasks/{task_id}/result", response_model=DetectionResult)
def get_task_result(task_id: str) -> DetectionResult:
    result = task_repository.get_result(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="任务结果不存在或尚未生成。")
    return result


@router.get("/tasks/{task_id}/artifacts/{name}")
def get_task_artifact(task_id: str, name: str) -> FileResponse:
    artifact_path = storage_service.artifact_path(task_id, name)
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="产物文件不存在。")
    return FileResponse(path=artifact_path)
