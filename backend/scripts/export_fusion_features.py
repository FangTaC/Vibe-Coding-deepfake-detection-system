from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from stage2_common import aggregate_fusion_features, build_offline_feature_payload, iter_examples

import sys


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.backends import FaceDetectorManager, PublicFigureGallery, VisualBackendManager  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export image-level fusion features to JSONL.")
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset root with <split>/{real,fake} layout.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split name, for example train / val / test.")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Target JSONL file path.")
    parser.add_argument("--dataset-version", type=str, required=True, help="Frozen dataset version string.")
    parser.add_argument("--limit-per-class", type=int, default=0, help="Optional cap per class, 0 means no limit.")
    parser.add_argument("--log-every", type=int, default=5, help="Print progress every N exported samples.")
    return parser.parse_args()


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    detector = FaceDetectorManager()
    visual_backend = VisualBackendManager()
    gallery = PublicFigureGallery()

    counts = defaultdict(int)
    exported = 0
    started_at = time.time()

    if args.output_jsonl.exists():
        args.output_jsonl.unlink()

    print(
        f"[start] split={args.split} data_root={args.data_root} "
        f"output_jsonl={args.output_jsonl} limit_per_class={args.limit_per_class}",
        flush=True,
    )

    for example in iter_examples(args.data_root, args.split):
        if args.limit_per_class > 0 and counts[example.label_name] >= args.limit_per_class:
            continue

        item_started = time.time()
        try:
            print(
                f"[processing] #{exported + 1} label={example.label_name} path={example.image_path}",
                flush=True,
            )
            payload = build_offline_feature_payload(
                example.image_path,
                detector,
                visual_backend,
                gallery,
                debug=True,
            )
            feature_row = aggregate_fusion_features(payload["faces"], semantic_score=0.0)
            feature_row.update(
                {
                    "label": example.label_id,
                    "label_name": example.label_name,
                    "image_path": str(example.image_path.resolve()),
                    "dataset_version": args.dataset_version,
                    "split": args.split,
                    "detector_backend": payload["model_versions"].get("detector", "none"),
                    "visual_backend": payload["model_versions"].get("visual", "none"),
                    "face_count": int(feature_row["face_count"]),
                }
            )
            append_jsonl(args.output_jsonl, feature_row)
            exported += 1
            counts[example.label_name] += 1

            if exported % args.log_every == 0:
                elapsed = time.time() - started_at
                print(
                    f"[progress] exported={exported} counts={dict(counts)} "
                    f"last_item_sec={time.time() - item_started:.2f} total_sec={elapsed:.2f}",
                    flush=True,
                )
        except Exception as exc:
            print(
                f"[error] failed_on={example.image_path} label={example.label_name} exc={exc}",
                flush=True,
            )
            raise

    total_elapsed = time.time() - started_at
    print(f"[done] exported={exported} output={args.output_jsonl}", flush=True)
    print(f"[done] per_class_counts={dict(counts)} total_sec={total_elapsed:.2f}", flush=True)


if __name__ == "__main__":
    main()
