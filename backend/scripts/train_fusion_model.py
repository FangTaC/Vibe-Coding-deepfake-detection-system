from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from stage2_common import calibrate_thresholds


FEATURE_KEYS = [
    "max_face_score",
    "mean_face_score",
    "dct_mean",
    "quality_mean",
    "consistency_mean",
    "compression_mean",
    "face_count",
    "semantic_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost fusion model from extracted feature records.")
    parser.add_argument("--train-jsonl", type=Path, required=True, help="JSONL file with features and label.")
    parser.add_argument("--val-jsonl", type=Path, help="Optional validation JSONL used for metrics and threshold freeze.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-version", type=str, required=True, help="Frozen dataset version string for metadata.")
    return parser.parse_args()


def load_jsonl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    features = []
    labels = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        item = json.loads(line)
        missing_keys = [key for key in FEATURE_KEYS + ["label"] if key not in item]
        if missing_keys:
            raise KeyError(f"{path}:{line_number} missing keys: {', '.join(missing_keys)}")
        features.append([float(item[key]) for key in FEATURE_KEYS])
        labels.append(int(item["label"]))
    if not features:
        raise ValueError(f"{path} does not contain any training rows")
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score  # type: ignore

    if len(set(labels.tolist())) < 2:
        return 0.0
    return float(roc_auc_score(labels, scores))


def main() -> None:
    args = parse_args()
    import xgboost as xgb  # type: ignore
    from sklearn.metrics import accuracy_score, f1_score  # type: ignore

    train_features, train_labels = load_jsonl(args.train_jsonl)
    booster = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
    )
    booster.fit(train_features, train_labels)

    if args.val_jsonl:
        metric_split = "val"
        metric_features, metric_labels = load_jsonl(args.val_jsonl)
        threshold_source = "validated_grid_search"
    else:
        metric_split = "train_only"
        metric_features, metric_labels = train_features, train_labels
        threshold_source = "default_config_no_validation"

    metric_scores = booster.predict_proba(metric_features)[:, 1]
    metric_predictions = (metric_scores >= 0.5).astype(np.int32)
    accuracy = float(accuracy_score(metric_labels, metric_predictions))
    f1 = float(f1_score(metric_labels, metric_predictions, zero_division=0))
    auc = safe_auc(metric_labels, metric_scores)

    if args.val_jsonl:
        thresholds, calibration = calibrate_thresholds(
            metric_scores.tolist(),
            metric_labels.tolist(),
            default_true_threshold=0.30,
            default_fake_threshold=0.70,
        )
    else:
        thresholds = {"true_threshold": 0.30, "fake_threshold": 0.70}
        calibration = {"best_accuracy": round(accuracy, 4), "review_rate": 0.0}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "xgboost_fusion.json"
    booster.save_model(model_path)

    metadata = {
        "model_name": "xgboost_fusion",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_version": args.dataset_version,
        "feature_keys": FEATURE_KEYS,
        "train_rows": int(train_labels.shape[0]),
        "val_rows": int(metric_labels.shape[0]) if args.val_jsonl else 0,
        "metrics": {
            "split": metric_split,
            "accuracy": round(accuracy, 4),
            "f1": round(f1, 4),
            "auc": round(auc, 4),
        },
        "thresholds": thresholds,
        "threshold_source": threshold_source,
        "calibration": calibration,
    }
    metadata_path = args.output_dir / "xgboost_fusion.meta.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved fusion model to {model_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
