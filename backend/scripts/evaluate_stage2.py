from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import median

from stage2_common import (
    aggregate_fusion_features,
    build_offline_feature_payload,
    compute_binary_metrics,
    elapsed_seconds,
    fusion_feature_vector,
    iter_examples,
    prediction_from_thresholds,
)

import sys


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.agents.decision_agent import DecisionAgent  # noqa: E402
from app.schemas.task import ReviewLabel  # noqa: E402
from app.services.backends import FaceDetectorManager, FusionBackendManager, PublicFigureGallery, VisualBackendManager  # noqa: E402
from app.services.model_registry import model_registry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate stage-2 pipelines on a labeled dataset split.")
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset root with <split>/{real,fake} layout.")
    parser.add_argument("--split", type=str, default="val", help="Dataset split name, for example val / test.")
    parser.add_argument("--dataset-version", type=str, required=True, help="Frozen dataset version string.")
    parser.add_argument("--output-json", type=Path, help="Optional JSON output path.")
    parser.add_argument("--limit-per-class", type=int, default=0, help="Optional cap per class, 0 means no limit.")
    return parser.parse_args()


def label_to_binary(label: str, true_label: int) -> int | None:
    if label == ReviewLabel.likely_real.value:
        return 0
    if label == ReviewLabel.likely_fake.value:
        return 1
    return None


def summarize_pipeline(true_labels: list[int], predictions: list[int | None], scores: list[float], latencies: list[float]) -> dict[str, float]:
    metrics = compute_binary_metrics(true_labels, predictions, scores)
    metrics["latency"] = round(float(median(latencies)) if latencies else 0.0, 4)
    return metrics


def main() -> None:
    args = parse_args()
    detector = FaceDetectorManager()
    visual_backend = VisualBackendManager()
    fusion_backend = FusionBackendManager()
    gallery = PublicFigureGallery()
    decision_agent = DecisionAgent()
    threshold_policy = model_registry.resolve_threshold_policy()

    counts: dict[str, int] = {"real": 0, "fake": 0}
    visual_true: list[int] = []
    visual_pred: list[int | None] = []
    visual_scores: list[float] = []
    visual_latencies: list[float] = []

    fusion_true: list[int] = []
    fusion_pred: list[int | None] = []
    fusion_scores: list[float] = []
    fusion_latencies: list[float] = []

    full_true: list[int] = []
    full_pred: list[int | None] = []
    full_scores: list[float] = []
    full_latencies: list[float] = []

    for example in iter_examples(args.data_root, args.split):
        if args.limit_per_class > 0 and counts[example.label_name] >= args.limit_per_class:
            continue

        started_at = time.perf_counter()
        payload = build_offline_feature_payload(example.image_path, detector, visual_backend, gallery)
        feature_elapsed = elapsed_seconds(started_at)
        feature_row = aggregate_fusion_features(payload["faces"], semantic_score=0.0)

        visual_score = float(feature_row["max_face_score"]) if payload["faces"] else 0.5
        visual_prediction, _visual_label = prediction_from_thresholds(
            visual_score,
            threshold_policy["true_threshold"],
            threshold_policy["fake_threshold"],
        )
        visual_true.append(example.label_id)
        visual_pred.append(visual_prediction)
        visual_scores.append(visual_score)
        visual_latencies.append(feature_elapsed)

        fusion_score, _fusion_backend_name = fusion_backend.predict(fusion_feature_vector(feature_row))
        fusion_prediction, _fusion_label = prediction_from_thresholds(
            fusion_score,
            threshold_policy["true_threshold"],
            threshold_policy["fake_threshold"],
        )
        fusion_true.append(example.label_id)
        fusion_pred.append(fusion_prediction)
        fusion_scores.append(float(fusion_score))
        fusion_latencies.append(elapsed_seconds(started_at))

        result = decision_agent.run(f"eval_{example.image_path.stem}", str(example.image_path), payload)
        full_true.append(example.label_id)
        full_pred.append(label_to_binary(result.label.value, example.label_id))
        full_scores.append(float(result.fusion_score))
        full_latencies.append(elapsed_seconds(started_at))

        counts[example.label_name] += 1

    if sum(counts.values()) == 0:
        raise ValueError("no labeled images were found for evaluation")

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_version": args.dataset_version,
        "data_root": str(args.data_root.resolve()),
        "split": args.split,
        "counts": counts,
        "threshold_policy": threshold_policy,
        "runtime_status": model_registry.build_runtime_status(),
        "pipelines": {
            "visual_only": summarize_pipeline(visual_true, visual_pred, visual_scores, visual_latencies),
            "visual_plus_fusion": summarize_pipeline(fusion_true, fusion_pred, fusion_scores, fusion_latencies),
            "full_dual_agent": summarize_pipeline(full_true, full_pred, full_scores, full_latencies),
        },
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved evaluation report to {args.output_json}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
