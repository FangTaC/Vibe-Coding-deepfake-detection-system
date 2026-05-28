from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient


BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.main import app  # noqa: E402


client = TestClient(app)


@pytest.fixture(autouse=True)
def disable_insightface(monkeypatch):
    monkeypatch.setattr("app.services.backends.InsightFaceDetector.is_available", lambda self: False)


def build_face_image(fake: bool = False) -> bytes:
    image = Image.new("RGB", (512, 512), color=(242, 238, 231))
    draw = ImageDraw.Draw(image)
    draw.ellipse((128, 88, 384, 390), fill=(240, 204, 176), outline=(151, 112, 94), width=4)
    draw.ellipse((188, 190, 230, 226), fill=(40, 58, 67))
    draw.ellipse((282, 190, 324, 226), fill=(40, 58, 67))
    draw.arc((206, 252, 306, 320), start=10, end=170, fill=(146, 72, 78), width=5)
    draw.rectangle((228, 222, 284, 286), fill=(223, 174, 150))
    if fake:
        for row in range(0, 512, 24):
            color = (242, 118, 93) if (row // 24) % 2 == 0 else (71, 190, 214)
            draw.rectangle((0, row, 512, row + 12), fill=color)
        for col in range(0, 512, 32):
            draw.rectangle((col, 120, col + 6, 410), fill=(255, 240, 60))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def wait_for_completion(task_id: str, timeout_seconds: int = 30) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = client.get(f"/api/tasks/{task_id}")
        response.raise_for_status()
        payload = response.json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.2)
    raise AssertionError("task did not finish in time")


def test_create_task_and_get_result() -> None:
    response = client.post(
        "/api/tasks",
        files={"file": ("portrait.png", build_face_image(fake=True), "image/png")},
    )
    assert response.status_code == 200
    task_id = response.json()["task_id"]

    status_payload = wait_for_completion(task_id)
    assert status_payload["status"] == "completed"

    result_response = client.get(f"/api/tasks/{task_id}/result")
    assert result_response.status_code == 200
    result_payload = result_response.json()
    assert result_payload["task_id"] == task_id
    assert "evidence_chain" in result_payload
    assert result_payload["artifacts"]
    assert "threshold_source" in result_payload["model_versions"]
    assert "visual" in result_payload["model_versions"]
    assert "fusion" in result_payload["model_versions"]


def test_blank_image_returns_review_required() -> None:
    image = Image.new("RGB", (320, 320), color=(128, 128, 128))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    response = client.post(
        "/api/tasks",
        files={"file": ("blank.png", buffer.getvalue(), "image/png")},
    )
    assert response.status_code == 200
    task_id = response.json()["task_id"]

    status_payload = wait_for_completion(task_id)
    assert status_payload["status"] == "completed"

    result_response = client.get(f"/api/tasks/{task_id}/result")
    payload = result_response.json()
    assert payload["label"] == "需人工复核"
    assert payload["faces"] == []
    assert "超出系统适用范围" in payload["review_reason"]
    assert payload["model_versions"]["threshold_source"] in {"default_config", "visual_metadata", "fusion_metadata"}
