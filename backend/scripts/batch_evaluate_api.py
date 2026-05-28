#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib import error, request


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CSV_FIELDS = [
    "filename",
    "path",
    "task_id",
    "ground_truth",
    "predicted_label",
    "predicted_class",
    "confidence",
    "fusion_score",
    "status",
    "review_required",
    "correct",
    "error_message",
]
FINAL_STATUSES = {"completed", "failed", "timeout", "http_error", "client_error"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate a labeled image folder by calling the existing single-image detection API."
    )
    parser.add_argument("--input-dir", required=True, help="Directory that contains evaluation images.")
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:8000/api",
        help="API base URL, for example http://127.0.0.1:8000/api",
    )
    parser.add_argument(
        "--output-csv",
        default="batch_eval_results.csv",
        help="Where to save per-image detection results.",
    )
    parser.add_argument(
        "--output-json",
        default="batch_eval_summary.json",
        help="Where to save summary statistics and raw rows.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan subdirectories.")
    parser.add_argument("--max-files", type=int, default=0, help="Optional limit for quick experiments.")
    parser.add_argument("--poll-interval", type=float, default=1.2, help="Polling interval in seconds.")
    parser.add_argument("--timeout-seconds", type=int, default=300, help="Per-image timeout.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output CSV and skip files already written with a final status.",
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Ignore existing outputs and start over. Cannot be combined with --resume.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=1,
        help="Rewrite summary JSON every N newly processed images. CSV is always appended per image.",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle candidate images before evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used with --shuffle.")
    parser.add_argument("--max-real", type=int, default=0, help="Maximum number of real images to evaluate.")
    parser.add_argument("--max-fake", type=int, default=0, help="Maximum number of fake images to evaluate.")
    return parser.parse_args()


def iter_images(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    files = [path for path in root.glob(pattern) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files)


def infer_ground_truth(path: Path) -> str:
    name = path.name.lower()
    if "real" in name:
        return "real"
    if any(token in name for token in ("fake", "deepfake", "ai", "generated")):
        return "fake"
    return ""


def select_images(
    images: list[Path],
    *,
    max_files: int,
    max_real: int,
    max_fake: int,
    shuffle: bool,
    seed: int,
) -> list[Path]:
    real_images = [path for path in images if infer_ground_truth(path) == "real"]
    fake_images = [path for path in images if infer_ground_truth(path) == "fake"]
    unlabeled_images = [path for path in images if infer_ground_truth(path) == ""]

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(real_images)
        rng.shuffle(fake_images)
        rng.shuffle(unlabeled_images)

    selected: list[Path] = []
    if max_real > 0 or max_fake > 0:
        selected.extend(real_images[: max_real or len(real_images)])
        selected.extend(fake_images[: max_fake or len(fake_images)])
        if max_files > 0 and len(selected) < max_files:
            remaining_slots = max_files - len(selected)
            selected.extend(unlabeled_images[:remaining_slots])
    else:
        selected = list(images)
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(selected)
        if max_files > 0:
            selected = selected[:max_files]

    if max_files > 0 and len(selected) > max_files:
        selected = selected[:max_files]

    return selected


def normalize_prediction(label: str) -> str:
    normalized = (label or "").lower()
    if "fake" in normalized or "\u4f2a" in normalized:
        return "fake"
    if "real" in normalized or "\u771f" in normalized:
        return "real"
    return "review"


def encode_multipart(file_path: Path, field_name: str = "file") -> tuple[bytes, str]:
    boundary = f"----AgentDeepfake{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    file_bytes = file_path.read_bytes()
    parts = [
        f"--{boundary}\r\n".encode("utf-8"),
        (
            f'Content-Disposition: form-data; name="{field_name}"; filename="{file_path.name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8"),
        file_bytes,
        b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    body = b"".join(parts)
    return body, f"multipart/form-data; boundary={boundary}"


def request_json(url: str, *, data: bytes | None = None, headers: dict[str, str] | None = None) -> dict[str, Any]:
    req = request.Request(url, data=data, headers=headers or {}, method="POST" if data is not None else "GET")
    with request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def create_task(api_base: str, image_path: Path) -> dict[str, Any]:
    body, content_type = encode_multipart(image_path)
    return request_json(
        f"{api_base}/tasks",
        data=body,
        headers={"Content-Type": content_type, "Content-Length": str(len(body))},
    )


def get_task_status(api_base: str, task_id: str) -> dict[str, Any]:
    return request_json(f"{api_base}/tasks/{task_id}")


def get_task_result(api_base: str, task_id: str) -> dict[str, Any]:
    return request_json(f"{api_base}/tasks/{task_id}/result")


def evaluate_one(
    api_base: str,
    image_path: Path,
    *,
    poll_interval: float,
    timeout_seconds: int,
) -> dict[str, Any]:
    ground_truth = infer_ground_truth(image_path)
    created = create_task(api_base, image_path)
    task_id = created["task_id"]
    start = time.time()

    while True:
        status_payload = get_task_status(api_base, task_id)
        status = status_payload["status"]
        if status == "completed":
            result = get_task_result(api_base, task_id)
            predicted_label = result.get("label", "")
            predicted_class = normalize_prediction(predicted_label)
            return {
                "filename": image_path.name,
                "path": str(image_path),
                "task_id": task_id,
                "ground_truth": ground_truth,
                "predicted_label": predicted_label,
                "predicted_class": predicted_class,
                "confidence": result.get("confidence"),
                "fusion_score": result.get("fusion_score"),
                "status": status,
                "review_required": result.get("review_required"),
                "correct": predicted_class == ground_truth if ground_truth else None,
                "error_message": "",
            }

        if status == "failed":
            return {
                "filename": image_path.name,
                "path": str(image_path),
                "task_id": task_id,
                "ground_truth": ground_truth,
                "predicted_label": "",
                "predicted_class": "",
                "confidence": None,
                "fusion_score": None,
                "status": status,
                "review_required": None,
                "correct": None,
                "error_message": status_payload.get("error_message", ""),
            }

        if time.time() - start > timeout_seconds:
            return {
                "filename": image_path.name,
                "path": str(image_path),
                "task_id": task_id,
                "ground_truth": ground_truth,
                "predicted_label": "",
                "predicted_class": "",
                "confidence": None,
                "fusion_score": None,
                "status": "timeout",
                "review_required": None,
                "correct": None,
                "error_message": f"Timed out after {timeout_seconds}s",
            }

        time.sleep(poll_interval)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [row for row in rows if row["ground_truth"]]
    correct_rows = [row for row in labeled if row["correct"] is True]
    incorrect_rows = [row for row in labeled if row["correct"] is False]
    real_rows = [row for row in labeled if row["ground_truth"] == "real"]
    fake_rows = [row for row in labeled if row["ground_truth"] == "fake"]
    real_correct = [row for row in real_rows if row["correct"] is True]
    fake_correct = [row for row in fake_rows if row["correct"] is True]
    false_positive = [row for row in labeled if row["ground_truth"] == "real" and row["predicted_class"] == "fake"]
    false_negative = [row for row in labeled if row["ground_truth"] == "fake" and row["predicted_class"] == "real"]
    review_required = [row for row in rows if row["predicted_class"] == "review"]
    failed = [row for row in rows if row["status"] in {"failed", "timeout", "http_error", "client_error"}]
    predicted_real = [row for row in rows if row["predicted_class"] == "real"]
    predicted_fake = [row for row in rows if row["predicted_class"] == "fake"]

    return {
        "total_images": len(rows),
        "labeled_images": len(labeled),
        "unlabeled_images": len(rows) - len(labeled),
        "correct": len(correct_rows),
        "incorrect": len(incorrect_rows),
        "accuracy": round(len(correct_rows) / len(labeled), 4) if labeled else None,
        "real_count": len(real_rows),
        "fake_count": len(fake_rows),
        "predicted_real": len(predicted_real),
        "predicted_fake": len(predicted_fake),
        "real_accuracy": round(len(real_correct) / len(real_rows), 4) if real_rows else None,
        "fake_accuracy": round(len(fake_correct) / len(fake_rows), 4) if fake_rows else None,
        "false_positive": len(false_positive),
        "false_negative": len(false_negative),
        "review_required": len(review_required),
        "failed_or_timeout": len(failed),
    }


def load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        return [dict(row) for row in reader]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def write_summary_json(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


def should_skip(row: dict[str, Any]) -> bool:
    return str(row.get("status", "")).lower() in FINAL_STATUSES


def normalize_loaded_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {field: row.get(field, "") for field in CSV_FIELDS}
    text = str(normalized.get("correct", "")).strip().lower()
    if text == "true":
        normalized["correct"] = True
    elif text == "false":
        normalized["correct"] = False
    else:
        normalized["correct"] = None
    for key in ("confidence", "fusion_score"):
        value = str(normalized.get(key, "")).strip()
        normalized[key] = float(value) if value not in {"", "None", "null"} else None
    review_required = str(normalized.get("review_required", "")).strip().lower()
    if review_required == "true":
        normalized["review_required"] = True
    elif review_required == "false":
        normalized["review_required"] = False
    else:
        normalized["review_required"] = None
    return normalized


def main() -> int:
    args = parse_args()
    if args.resume and args.rewrite:
        print("[error] --resume and --rewrite cannot be used together", file=sys.stderr)
        return 1

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"[error] input directory not found: {input_dir}", file=sys.stderr)
        return 1

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)

    images = iter_images(input_dir, args.recursive)
    images = select_images(
        images,
        max_files=args.max_files,
        max_real=args.max_real,
        max_fake=args.max_fake,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    if not images:
        print("[error] no images found", file=sys.stderr)
        return 1

    existing_rows: list[dict[str, Any]] = []
    completed_paths: set[str] = set()
    if args.resume and output_csv.exists():
        existing_rows = [normalize_loaded_row(row) for row in load_existing_rows(output_csv)]
        completed_paths = {str(row["path"]) for row in existing_rows if should_skip(row)}
        print(f"[resume] loaded {len(existing_rows)} existing rows, {len(completed_paths)} completed paths")
    elif args.rewrite:
        if output_csv.exists():
            output_csv.unlink()
        if output_json.exists():
            output_json.unlink()

    rows: list[dict[str, Any]] = list(existing_rows)
    pending_images = [path for path in images if str(path) not in completed_paths]

    selected_real = sum(1 for path in images if infer_ground_truth(path) == "real")
    selected_fake = sum(1 for path in images if infer_ground_truth(path) == "fake")
    selected_unlabeled = len(images) - selected_real - selected_fake
    print(
        f"[start] total_images={len(images)} pending={len(pending_images)} api={args.api_base} "
        f"(real={selected_real}, fake={selected_fake}, unlabeled={selected_unlabeled})"
    )
    if args.resume:
        print("[resume] CSV and JSON will be updated incrementally")
    processed_since_flush = 0

    if not args.resume and output_csv.exists() and not args.rewrite:
        print("[warn] output CSV already exists and will be appended to. Use --rewrite to start clean.", file=sys.stderr)

    for index, image_path in enumerate(pending_images, start=1):
        print(f"[{index}/{len(pending_images)}] {image_path.name}")
        try:
            row = evaluate_one(
                args.api_base.rstrip("/"),
                image_path,
                poll_interval=args.poll_interval,
                timeout_seconds=args.timeout_seconds,
            )
        except error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="ignore")
            row = {
                "filename": image_path.name,
                "path": str(image_path),
                "task_id": "",
                "ground_truth": infer_ground_truth(image_path),
                "predicted_label": "",
                "predicted_class": "",
                "confidence": None,
                "fusion_score": None,
                "status": "http_error",
                "review_required": None,
                "correct": None,
                "error_message": f"{exc.code}: {message}",
            }
        except Exception as exc:  # noqa: BLE001
            row = {
                "filename": image_path.name,
                "path": str(image_path),
                "task_id": "",
                "ground_truth": infer_ground_truth(image_path),
                "predicted_label": "",
                "predicted_class": "",
                "confidence": None,
                "fusion_score": None,
                "status": "client_error",
                "review_required": None,
                "correct": None,
                "error_message": f"{type(exc).__name__}: {exc}",
            }

        rows.append(row)
        append_csv_row(output_csv, row)
        processed_since_flush += 1

        print(
            f"    status={row['status']} gt={row['ground_truth'] or 'unlabeled'} "
            f"pred={row['predicted_class'] or '--'} conf={row['confidence']}"
        )

        if processed_since_flush >= max(1, args.flush_every):
            write_summary_json(output_json, summarize(rows), rows)
            processed_since_flush = 0

    summary = summarize(rows)
    write_summary_json(output_json, summary, rows)

    print("[done] summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] csv={output_csv}")
    print(f"[done] json={output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
