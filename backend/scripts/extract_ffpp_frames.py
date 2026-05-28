from __future__ import annotations

import argparse
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract training frames from downloaded FF++ videos. "
            "By default this performs uniform sparse sampling, which is the most practical choice "
            "for this project."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Root directory that contains downloaded videos.")
    parser.add_argument("--output-root", type=Path, required=True, help="Root directory for extracted image frames.")
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Target extraction rate. 1 fps is a practical default for initial training data.",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=40,
        help="Hard cap per video to avoid exploding dataset size.",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="Frame image format.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality when image-format=jpg.",
    )
    return parser.parse_args()


def iter_videos(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: float,
    max_frames_per_video: int,
    image_format: str,
    quality: int,
) -> int:
    import cv2  # type: ignore

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return 0

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    source_fps = max(float(source_fps), 1.0)
    frame_interval = max(int(round(source_fps / max(fps, 1e-6))), 1)

    saved = 0
    frame_idx = 0
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx % frame_interval == 0:
            output_path = output_dir / f"frame_{frame_idx:06d}.{image_format}"
            if image_format == "jpg":
                cv2.imwrite(str(output_path), frame, jpeg_params)
            else:
                cv2.imwrite(str(output_path), frame)
            saved += 1
            if saved >= max_frames_per_video:
                break
        frame_idx += 1

    capture.release()
    return saved


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    videos = iter_videos(args.input_root)
    if not videos:
        raise FileNotFoundError(f"No video files found under {args.input_root}")

    total_saved = 0
    for video_path in videos:
        relative_parent = video_path.parent.relative_to(args.input_root)
        output_dir = args.output_root / relative_parent / video_path.stem
        saved = extract_frames(
            video_path=video_path,
            output_dir=output_dir,
            fps=args.fps,
            max_frames_per_video=args.max_frames_per_video,
            image_format=args.image_format,
            quality=args.quality,
        )
        total_saved += saved
        print(f"[ok] {video_path.name}: saved {saved} frames -> {output_dir}")

    print(f"Finished. Saved {total_saved} frames from {len(videos)} videos.")


if __name__ == "__main__":
    main()
