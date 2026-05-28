from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import httpx

import sys


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import settings  # noqa: E402
from app.services.model_registry import model_registry  # noqa: E402


API_BASE = "http://127.0.0.1:8000/api"
FRONTEND_BASE = "http://127.0.0.1:5173"


def find_edge() -> str | None:
    candidates = [
        shutil.which("msedge"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def wait_for_completion(client: httpx.Client, task_id: str, timeout_seconds: int = 120) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            response = client.get(f"{API_BASE}/tasks/{task_id}", timeout=20.0)
            response.raise_for_status()
            payload = response.json()
        except httpx.ReadTimeout:
            time.sleep(1.0)
            continue
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.8)
    raise TimeoutError(f"task {task_id} did not finish in time")


def take_screenshot(task_id: str, file_name: str) -> str | None:
    edge_path = find_edge()
    if not edge_path:
        return None
    screenshot_path = settings.demo_reports_dir / file_name
    url = f"{FRONTEND_BASE}/?task={task_id}"
    subprocess.run(
        [
            edge_path,
            "--headless=new",
            "--disable-gpu",
            "--window-size=1600,1400",
            "--virtual-time-budget=6000",
            f"--screenshot={screenshot_path}",
            url,
        ],
        check=False,
        timeout=30,
    )
    return str(screenshot_path) if screenshot_path.exists() else None


def main() -> None:
    manifest = json.loads(Path(settings.demo_manifest_path).read_text(encoding="utf-8"))
    report_rows = []
    required_steps = {"input_loaded", "feature_extraction", "strategy_selection", "final_decision"}
    with httpx.Client() as client:
        for item in manifest["samples"]:
            sample_path = settings.demo_samples_dir / item["file"]
            with sample_path.open("rb") as file_handle:
                response = client.post(
                    f"{API_BASE}/tasks",
                    files={"file": (sample_path.name, file_handle, "image/png")},
                    timeout=60.0,
                )
            response.raise_for_status()
            task_id = response.json()["task_id"]
            started_at = time.time()
            status_payload = wait_for_completion(client, task_id)
            elapsed = round(time.time() - started_at, 2)
            result_payload = client.get(f"{API_BASE}/tasks/{task_id}/result", timeout=20.0).json()
            steps = {entry["step"] for entry in result_payload["evidence_chain"]}
            screenshot = take_screenshot(task_id, f"{item['category']}_{task_id}.png")
            report_rows.append(
                {
                    "category": item["category"],
                    "file": item["file"],
                    "expected_hint": item["expected_hint"],
                    "task_id": task_id,
                    "status": status_payload["status"],
                    "label": result_payload["label"],
                    "confidence": result_payload["confidence"],
                    "elapsed_seconds": elapsed,
                    "required_steps_ok": required_steps.issubset(steps),
                    "evidence_steps": sorted(steps),
                    "model_versions": result_payload.get("model_versions", {}),
                    "semantic_flags": [flag["code"] for flag in result_payload.get("semantic_flags", [])],
                    "result_json_path": str(settings.demo_reports_dir / f"{item['category']}_{task_id}.json"),
                    "frontend_screenshot_path": screenshot,
                }
            )
            Path(report_rows[-1]["result_json_path"]).write_text(
                json.dumps(result_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "api_base": API_BASE,
        "frontend_base": FRONTEND_BASE,
        "integration_status": model_registry.build_runtime_status(),
        "items": report_rows,
    }
    json_path = settings.demo_reports_dir / "demo_checklist_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown_lines = [
        "# Demo Checklist Report",
        "",
        model_registry.build_status_table_markdown(),
        "",
        "| category | task_id | label | elapsed_seconds | screenshot |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report_rows:
        screenshot_text = row["frontend_screenshot_path"] or "not_captured"
        markdown_lines.append(
            f"| {row['category']} | {row['task_id']} | {row['label']} | {row['elapsed_seconds']} | {screenshot_text} |"
        )
    markdown_path = settings.demo_reports_dir / "demo_checklist_report.md"
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    print(f"Saved demo report to {json_path}")


if __name__ == "__main__":
    main()
