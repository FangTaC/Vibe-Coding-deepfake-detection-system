from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Iterator


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.schemas.task import EvidenceEntry, FaceResult, ReviewLabel  # noqa: E402
from app.services.backends import FaceDetectorManager, FusionBackendManager, PublicFigureGallery, VisualBackendManager  # noqa: E402
from app.services.model_registry import model_registry  # noqa: E402
from app.utils.image_ops import crop_face, load_image  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LABEL_TO_ID = {"real": 0, "fake": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


@dataclass(slots=True)
class OfflineExample:
    image_path: Path
    label_name: str
    label_id: int


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iter_examples(data_root: Path, split: str) -> Iterator[OfflineExample]:
    split_root = resolve_split_root(data_root, split)
    for label_name, label_id in LABEL_TO_ID.items():
        class_dir = split_root / label_name
        if not class_dir.exists():
            raise FileNotFoundError(f"expected class directory: {class_dir}")
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield OfflineExample(image_path=image_path, label_name=label_name, label_id=label_id)


def resolve_split_root(data_root: Path, split: str) -> Path:
    direct_root = data_root
    nested_root = data_root / split

    if all((nested_root / name).exists() for name in LABEL_TO_ID):
        return nested_root
    if all((direct_root / name).exists() for name in LABEL_TO_ID):
        return direct_root
    raise FileNotFoundError(
        "expected dataset layout '<data_root>/<split>/{real,fake}' or '<data_root>/{real,fake}'"
    )


def dataset_class_distribution(data_root: Path, split: str) -> dict[str, int]:
    split_root = resolve_split_root(data_root, split)
    distribution: dict[str, int] = {}
    for label_name in LABEL_TO_ID:
        class_dir = split_root / label_name
        distribution[label_name] = sum(
            1 for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return distribution


def aggregate_fusion_features(faces: Iterable[FaceResult], semantic_score: float = 0.0) -> dict[str, float]:
    face_list = list(faces)
    if not face_list:
        return {
            "max_face_score": 0.0,
            "mean_face_score": 0.0,
            "dct_mean": 0.0,
            "quality_mean": 0.0,
            "consistency_mean": 0.0,
            "compression_mean": 0.0,
            "face_count": 0.0,
            "semantic_score": float(semantic_score),
        }
    return {
        "max_face_score": float(max(face.deepfake_score for face in face_list)),
        "mean_face_score": float(mean(face.deepfake_score for face in face_list)),
        "dct_mean": float(mean(face.dct_score for face in face_list)),
        "quality_mean": float(mean(face.quality_score for face in face_list)),
        "consistency_mean": float(mean(face.augmentation_consistency_score for face in face_list)),
        "compression_mean": float(mean(face.compression_score for face in face_list)),
        "face_count": float(len(face_list)),
        "semantic_score": float(semantic_score),
    }


def fusion_feature_vector(feature_row: dict[str, float]) -> list[float]:
    return [
        float(feature_row["max_face_score"]),
        float(feature_row["mean_face_score"]),
        float(feature_row["dct_mean"]),
        float(feature_row["quality_mean"]),
        float(feature_row["consistency_mean"]),
        float(feature_row["compression_mean"]),
        float(feature_row["face_count"]),
        float(feature_row["semantic_score"]),
    ]


def build_offline_feature_payload(
    image_path: Path,
    detector: FaceDetectorManager,
    visual_backend: VisualBackendManager,
    gallery: PublicFigureGallery,
    debug: bool = False,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    if debug:
        print(f"[feature] load_image path={image_path}", flush=True)
    image = load_image(image_path)
    if debug:
        print(f"[feature] load_image_done sec={elapsed_seconds(started_at):.3f}", flush=True)
        print(f"[feature] detect_faces path={image_path}", flush=True)
    detect_started = time.perf_counter()
    detections = detector.detect(image)
    if debug:
        print(
            f"[feature] detect_faces_done faces={len(detections)} sec={elapsed_seconds(detect_started):.3f}",
            flush=True,
        )
    evidence = [
        EvidenceEntry(
            step="input_loaded",
            actor="offline_feature_builder",
            timestamp=now_utc(),
            summary="离线评测已加载输入图像。",
            details={"width": image.width, "height": image.height, "mode": image.mode},
        )
    ]

    if not detections:
        evidence.append(
            EvidenceEntry(
                step="feature_extraction",
                actor="offline_feature_builder",
                timestamp=now_utc(),
                summary="未检测到有效人脸，离线评测保留空特征。",
                details={"face_count": 0, "detector_backend": "none", "classifier_backend": "none"},
            )
        )
        return {
            "faces": [],
            "artifacts": [],
            "evidence": evidence,
            "input_summary": {"width": image.width, "height": image.height, "mode": image.mode, "face_count": 0},
            "model_versions": {"detector": "none", "visual": "none"},
        }

    face_results: list[FaceResult] = []
    for index, detected_face in enumerate(detections):
        face_id = f"face_{index + 1}"
        if debug:
            print(
                f"[feature] analyze_face face_id={face_id} detector_backend={detected_face.backend}",
                flush=True,
            )
        analyze_started = time.perf_counter()
        face_crop = crop_face(image, detected_face.bbox)
        analysis = visual_backend.analyze(face_crop, image.size, detected_face.bbox)
        if debug:
            print(
                f"[feature] analyze_face_done face_id={face_id} visual_backend={analysis['backend_name']} "
                f"sec={elapsed_seconds(analyze_started):.3f}",
                flush=True,
            )
            print(f"[feature] gallery_match face_id={face_id}", flush=True)
        gallery_started = time.perf_counter()
        public_figures = gallery.match(detected_face.embedding or analysis["feature_embedding"])
        if debug:
            print(
                f"[feature] gallery_match_done face_id={face_id} matches={len(public_figures)} "
                f"sec={elapsed_seconds(gallery_started):.3f}",
                flush=True,
            )
        face_results.append(
            FaceResult(
                face_id=face_id,
                bbox=detected_face.bbox,
                landmarks=[[round(x, 2), round(y, 2)] for x, y in detected_face.landmarks],
                deepfake_score=round(float(analysis["deepfake_score"]), 4),
                quality_score=round(float(analysis["quality_score"]), 4),
                dct_score=round(float(analysis["dct_score"]), 4),
                blur_score=round(float(analysis["blur_score"]), 4),
                compression_score=round(float(analysis["compression_score"]), 4),
                face_size_ratio=round(float(analysis["face_size_ratio"]), 4),
                augmentation_consistency_score=round(float(analysis["augmentation_consistency_score"]), 4),
                feature_embedding=list(analysis["feature_embedding"]),
                detector_backend=detected_face.backend,
                classifier_backend=str(analysis["backend_name"]),
                public_figure_candidates=public_figures,
                artifacts=[],
            )
        )

    evidence.append(
        EvidenceEntry(
            step="feature_extraction",
            actor="offline_feature_builder",
            timestamp=now_utc(),
            summary=f"离线评测完成 {len(face_results)} 张人脸的特征提取。",
            details={
                "face_count": len(face_results),
                "detector_backend": face_results[0].detector_backend,
                "detector_provider": detections[0].provider,
                "classifier_backend": face_results[0].classifier_backend,
            },
        )
    )

    model_versions = {
        "detector": face_results[0].detector_backend,
        "detector_provider": detections[0].provider,
    }
    model_versions.update(visual_backend.describe(face_results[0].classifier_backend))
    if debug:
        print(
            f"[feature] payload_done path={image_path} face_count={len(face_results)} "
            f"total_sec={elapsed_seconds(started_at):.3f}",
            flush=True,
        )
    return {
        "faces": face_results,
        "artifacts": [],
        "evidence": evidence,
        "input_summary": {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "face_count": len(face_results),
        },
        "model_versions": model_versions,
    }


def calibrate_thresholds(
    scores: list[float],
    labels: list[int],
    default_true_threshold: float,
    default_fake_threshold: float,
) -> tuple[dict[str, float], dict[str, float]]:
    if not scores or not labels or len(scores) != len(labels):
        return (
            {"true_threshold": default_true_threshold, "fake_threshold": default_fake_threshold},
            {"best_accuracy": 0.0, "review_rate": 0.0},
        )

    best_state: tuple[float, float, float, float] | None = None
    for true_step in range(5, 51):
        true_threshold = round(true_step / 100.0, 2)
        for fake_step in range(max(true_step + 5, 55), 96):
            fake_threshold = round(fake_step / 100.0, 2)
            predicted = []
            review_count = 0
            for score, truth in zip(scores, labels):
                if score <= true_threshold:
                    predicted.append(0)
                elif score >= fake_threshold:
                    predicted.append(1)
                else:
                    predicted.append(1 - truth)
                    review_count += 1

            accuracy = sum(int(pred == truth) for pred, truth in zip(predicted, labels)) / max(len(labels), 1)
            review_rate = review_count / max(len(labels), 1)
            state = (accuracy, -review_rate, true_threshold, fake_threshold)
            if best_state is None or state > best_state:
                best_state = state

    if best_state is None:
        return (
            {"true_threshold": default_true_threshold, "fake_threshold": default_fake_threshold},
            {"best_accuracy": 0.0, "review_rate": 0.0},
        )
    accuracy, negative_review_rate, true_threshold, fake_threshold = best_state
    return (
        {"true_threshold": true_threshold, "fake_threshold": fake_threshold},
        {"best_accuracy": round(accuracy, 4), "review_rate": round(-negative_review_rate, 4)},
    )


def compute_binary_metrics(
    true_labels: list[int],
    predicted_labels: list[int | None],
    scores: list[float],
) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # type: ignore

    transformed_predictions = [
        prediction if prediction is not None else 1 - truth for truth, prediction in zip(true_labels, predicted_labels)
    ]
    accuracy = float(accuracy_score(true_labels, transformed_predictions)) if true_labels else 0.0
    f1 = float(f1_score(true_labels, transformed_predictions, zero_division=0)) if true_labels else 0.0

    auc = 0.0
    if true_labels and len(set(true_labels)) > 1:
        auc = float(roc_auc_score(true_labels, scores))

    review_rate = (
        sum(1 for prediction in predicted_labels if prediction is None) / max(len(predicted_labels), 1)
        if predicted_labels
        else 0.0
    )
    return {
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4),
        "review_rate": round(review_rate, 4),
    }


def prediction_from_thresholds(score: float, true_threshold: float, fake_threshold: float) -> tuple[int | None, str]:
    if score <= true_threshold:
        return 0, ReviewLabel.likely_real.value
    if score >= fake_threshold:
        return 1, ReviewLabel.likely_fake.value
    return None, ReviewLabel.review_required.value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_task_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def elapsed_seconds(started_at: float) -> float:
    return round(time.perf_counter() - started_at, 4)
