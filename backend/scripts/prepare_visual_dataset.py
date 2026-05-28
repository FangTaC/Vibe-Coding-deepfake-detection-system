from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
REAL_DIR_NAMES = {"original"}
FAKE_DIR_NAMES = {"Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "DeepFakeDetection", "FaceShifter"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a visual training dataset from extracted FF++ frames. "
            "Frames under 'original' become real; frames under manipulation folders become fake."
        )
    )
    parser.add_argument("--frames-root", type=Path, required=True, help="Root directory of extracted frames.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output dataset root.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of hard-linking them. Use this if hard links are unsupported.",
    )
    return parser.parse_args()


def iter_label_images(frames_root: Path, label_dirs: set[str]) -> list[Path]:
    images: list[Path] = []
    for top_dir in label_dirs:
        source_root = frames_root / top_dir
        if not source_root.exists():
            continue
        for path in source_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(path)
    return sorted(images)


def split_paths(paths: list[Path], val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    val_count = int(len(shuffled) * val_ratio)
    val_paths = shuffled[:val_count]
    train_paths = shuffled[val_count:]
    return train_paths, val_paths


def safe_name(path: Path) -> str:
    parent_tag = "__".join(path.parts[-3:-1]) if len(path.parts) >= 3 else "sample"
    return f"{parent_tag}__{path.name}"


def materialize(paths: list[Path], target_dir: Path, use_copy: bool) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        target = target_dir / safe_name(path)
        if target.exists():
            continue
        if use_copy:
            shutil.copy2(path, target)
        else:
            try:
                target.hardlink_to(path)
            except OSError:
                shutil.copy2(path, target)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    real_images = iter_label_images(args.frames_root, REAL_DIR_NAMES)
    fake_images = iter_label_images(args.frames_root, FAKE_DIR_NAMES)

    if not real_images:
        raise FileNotFoundError(f"No real images found under {args.frames_root / 'original'}")
    if not fake_images:
        raise FileNotFoundError("No fake images found under expected manipulation directories.")

    train_real, val_real = split_paths(real_images, args.val_ratio, args.seed)
    train_fake, val_fake = split_paths(fake_images, args.val_ratio, args.seed)

    materialize(train_real, args.output_root / "train" / "real", args.copy)
    materialize(val_real, args.output_root / "val" / "real", args.copy)
    materialize(train_fake, args.output_root / "train" / "fake", args.copy)
    materialize(val_fake, args.output_root / "val" / "fake", args.copy)

    print("Prepared visual dataset:")
    print(f"  train/real: {len(train_real)}")
    print(f"  val/real:   {len(val_real)}")
    print(f"  train/fake: {len(train_fake)}")
    print(f"  val/fake:   {len(val_fake)}")
    print(f"  output:     {args.output_root}")


if __name__ == "__main__":
    main()
