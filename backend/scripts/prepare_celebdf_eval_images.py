#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import cv2  # type: ignore


REAL_DIRS = {"Celeb-real", "YouTube-real"}
FAKE_DIRS = {"Celeb-synthesis"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract labeled evaluation images from Celeb-DF-v2 testing videos."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to Celeb-DF-v2 root, e.g. E:\\Celeb-DF\\Celeb-DF-v2",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where to save extracted labeled images.",
    )
    parser.add_argument(
        "--every-n-frames",
        type=int,
        default=24,
        help="Extract one frame every N frames.",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=5,
        help="Maximum extracted frames per video.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=0,
        help="Optional square resize target, 0 keeps original resolution.",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality for saved images.",
    )
    return parser.parse_args()


def infer_label_from_relative_path(relative_path: str) -> str:
    normalized = relative_path.replace("\\", "/").strip()
    top_level = normalized.split("/", 1)[0]
    if top_level in REAL_DIRS:
        return "real"
    if top_level in FAKE_DIRS:
        return "fake"
    raise ValueError(f"Unknown Celeb-DF category for path: {relative_path}")


def sanitize_stem(relative_path: str) -> str:
    normalized = relative_path.replace("\\", "/").strip()
    stem = normalized.replace("/", "__")
    if stem.lower().endswith(".mp4"):
        stem = stem[:-4]
    return stem


def load_testing_list(dataset_root: Path) -> list[str]:
    list_path = dataset_root / "List_of_testing_videos.txt"
    if not list_path.exists():
        raise FileNotFoundError(f"Testing list not found: {list_path}")
    items: list[str] = []
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Celeb-DF official testing list lines may look like:
        # "1 YouTube-real/00170.mp4"
        # "0 Celeb-synthesis/id1_id2_0003.mp4"
        if " " in line:
            _, candidate = line.split(" ", 1)
            line = candidate.strip()
        items.append(line)
    return items


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    *,
    label: str,
    file_stem: str,
    every_n_frames: int,
    max_frames_per_video: int,
    resize: int,
    jpg_quality: int,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[warn] skip unreadable video: {video_path}")
        return 0

    saved = 0
    frame_index = 0
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)]

    try:
        while saved < max_frames_per_video:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % every_n_frames == 0:
                if resize and resize > 0:
                    frame = cv2.resize(frame, (resize, resize), interpolation=cv2.INTER_AREA)
                output_name = f"{label}__{file_stem}__frame_{frame_index:06d}.jpg"
                output_path = output_dir / output_name
                cv2.imwrite(str(output_path), frame, encode_params)
                saved += 1

            frame_index += 1
    finally:
        cap.release()

    return saved


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    testing_videos = load_testing_list(dataset_root)
    total_saved = 0

    print(f"[start] processing {len(testing_videos)} Celeb-DF testing videos")
    print(f"[config] every_n_frames={args.every_n_frames} max_frames_per_video={args.max_frames_per_video}")

    for index, relative_path in enumerate(testing_videos, start=1):
        video_path = dataset_root / relative_path
        if not video_path.exists():
            print(f"[warn] missing video: {video_path}")
            continue

        label = infer_label_from_relative_path(relative_path)
        file_stem = sanitize_stem(relative_path)
        saved = extract_frames_from_video(
            video_path,
            output_dir,
            label=label,
            file_stem=file_stem,
            every_n_frames=max(1, args.every_n_frames),
            max_frames_per_video=max(1, args.max_frames_per_video),
            resize=max(0, args.resize),
            jpg_quality=args.jpg_quality,
        )
        total_saved += saved
        print(f"[{index}/{len(testing_videos)}] {relative_path} -> label={label} saved={saved}")

    print(f"[done] saved {total_saved} images to {output_dir}")
    print("[done] filenames now include real__/fake__ prefixes and can be used by batch_evaluate_api.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
