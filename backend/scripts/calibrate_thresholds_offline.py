#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline threshold calibration using an existing batch evaluation CSV."
    )
    parser.add_argument("--input-csv", required=True, help="Path to batch evaluation CSV.")
    parser.add_argument(
        "--output-json",
        default="threshold_calibration_results.json",
        help="Where to save all scanned threshold results and recommended candidates.",
    )
    parser.add_argument(
        "--output-csv",
        default="threshold_calibration_results.csv",
        help="Where to save flattened threshold scan results.",
    )
    parser.add_argument("--true-min", type=float, default=0.0)
    parser.add_argument("--true-max", type=float, default=0.5)
    parser.add_argument("--true-step", type=float, default=0.05)
    parser.add_argument("--fake-min", type=float, default=0.5)
    parser.add_argument("--fake-max", type=float, default=1.0)
    parser.add_argument("--fake-step", type=float, default=0.05)
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of recommended threshold pairs to keep.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.2,
        help="Minimum auto-decision coverage (non-review ratio) for recommended candidates.",
    )
    return parser.parse_args()


def frange(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 4))
        current += step
    return values


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            ground_truth = (row.get("ground_truth") or "").strip().lower()
            if ground_truth not in {"real", "fake"}:
                continue
            score_text = (row.get("fusion_score") or "").strip()
            if not score_text:
                continue
            try:
                fusion_score = float(score_text)
            except ValueError:
                continue
            rows.append(
                {
                    "filename": row.get("filename", ""),
                    "ground_truth": ground_truth,
                    "fusion_score": fusion_score,
                }
            )
    return rows


def predict_label(score: float, true_threshold: float, fake_threshold: float) -> str:
    if score <= true_threshold:
        return "real"
    if score >= fake_threshold:
        return "fake"
    return "review"


def evaluate_thresholds(rows: list[dict[str, Any]], true_threshold: float, fake_threshold: float) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        raise ValueError("No labeled rows with fusion_score available.")

    predicted_real = 0
    predicted_fake = 0
    predicted_review = 0
    correct = 0
    incorrect = 0
    false_positive = 0
    false_negative = 0

    real_count = sum(1 for row in rows if row["ground_truth"] == "real")
    fake_count = total - real_count
    real_correct = 0
    fake_correct = 0

    for row in rows:
        predicted = predict_label(row["fusion_score"], true_threshold, fake_threshold)
        gt = row["ground_truth"]

        if predicted == "real":
            predicted_real += 1
            if gt == "real":
                correct += 1
                real_correct += 1
            else:
                incorrect += 1
                false_negative += 1
        elif predicted == "fake":
            predicted_fake += 1
            if gt == "fake":
                correct += 1
                fake_correct += 1
            else:
                incorrect += 1
                false_positive += 1
        else:
            predicted_review += 1
            incorrect += 1

    coverage = (predicted_real + predicted_fake) / total
    accuracy = correct / total
    real_accuracy = real_correct / real_count if real_count else None
    fake_accuracy = fake_correct / fake_count if fake_count else None
    review_rate = predicted_review / total

    return {
        "true_threshold": round(true_threshold, 4),
        "fake_threshold": round(fake_threshold, 4),
        "total_images": total,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": round(accuracy, 4),
        "coverage": round(coverage, 4),
        "review_rate": round(review_rate, 4),
        "real_count": real_count,
        "fake_count": fake_count,
        "predicted_real": predicted_real,
        "predicted_fake": predicted_fake,
        "predicted_review": predicted_review,
        "real_accuracy": round(real_accuracy, 4) if real_accuracy is not None else None,
        "fake_accuracy": round(fake_accuracy, 4) if fake_accuracy is not None else None,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def recommend_candidates(
    results: list[dict[str, Any]],
    *,
    top_k: int,
    min_coverage: float,
) -> list[dict[str, Any]]:
    eligible = [row for row in results if row["coverage"] >= min_coverage]
    eligible.sort(
        key=lambda row: (
            row["accuracy"],
            row["coverage"],
            -row["review_rate"],
            -(row["false_positive"] + row["false_negative"]),
        ),
        reverse=True,
    )
    return eligible[:top_k]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = load_rows(Path(args.input_csv))
    if not rows:
        print("[error] no usable labeled rows found in CSV")
        return 1

    true_values = frange(args.true_min, args.true_max, args.true_step)
    fake_values = frange(args.fake_min, args.fake_max, args.fake_step)

    results: list[dict[str, Any]] = []
    for true_threshold in true_values:
        for fake_threshold in fake_values:
            if true_threshold >= fake_threshold:
                continue
            results.append(evaluate_thresholds(rows, true_threshold, fake_threshold))

    recommendations = recommend_candidates(
        results,
        top_k=max(1, args.top_k),
        min_coverage=max(0.0, min(1.0, args.min_coverage)),
    )

    payload = {
        "input_csv": str(Path(args.input_csv)),
        "samples_used": len(rows),
        "search_space": {
            "true_values": true_values,
            "fake_values": fake_values,
        },
        "recommendations": recommendations,
        "all_results": results,
    }

    Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(Path(args.output_csv), results)

    print(f"[done] scanned {len(results)} threshold pairs using {len(rows)} labeled samples")
    print("[done] top recommendations:")
    for item in recommendations:
        print(
            f"  true={item['true_threshold']:.2f} fake={item['fake_threshold']:.2f} "
            f"acc={item['accuracy']:.4f} coverage={item['coverage']:.4f} review={item['review_rate']:.4f} "
            f"real_acc={item['real_accuracy']} fake_acc={item['fake_accuracy']}"
        )
    print(f"[done] json={args.output_json}")
    print(f"[done] csv={args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
