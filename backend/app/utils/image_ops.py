from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageFile, ImageFilter

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def image_to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image).astype(np.float32)


def array_to_image(array: np.ndarray) -> Image.Image:
    clipped = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(clipped)


def normalize_score(value: float, low: float, high: float) -> float:
    if high == low:
        return 0.0
    clipped = min(max(value, low), high)
    return float((clipped - low) / (high - low))


def compute_grayscale(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L")).astype(np.float32)


def compute_blur_score(image: Image.Image) -> float:
    gray = compute_grayscale(image)
    lap = (
        -4 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    variance = float(np.var(lap))
    return normalize_score(variance, 20.0, 1200.0)


def compute_dct_high_freq_score(image: Image.Image) -> float:
    gray = compute_grayscale(image)
    freq = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.abs(freq)
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) * 0.18
    yy, xx = np.ogrid[:h, :w]
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) >= radius**2
    high_freq = magnitude[mask].mean()
    low_freq = magnitude[~mask].mean() + 1e-6
    ratio = high_freq / low_freq
    return normalize_score(ratio, 0.15, 1.10)


def compute_compression_score(image: Image.Image) -> float:
    gray = compute_grayscale(image)
    if gray.shape[0] < 16 or gray.shape[1] < 16:
        return 0.0
    vertical_boundaries = []
    horizontal_boundaries = []
    for col in range(8, gray.shape[1], 8):
        vertical_boundaries.append(np.mean(np.abs(gray[:, col] - gray[:, col - 1])))
    for row in range(8, gray.shape[0], 8):
        horizontal_boundaries.append(np.mean(np.abs(gray[row, :] - gray[row - 1, :])))
    blockiness = 0.0
    if vertical_boundaries or horizontal_boundaries:
        blockiness = float(np.mean(vertical_boundaries + horizontal_boundaries))
    return normalize_score(blockiness, 4.0, 25.0)


def compute_face_size_ratio(bbox: Iterable[int], image_size: tuple[int, int]) -> float:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    image_width, image_height = image_size
    return float((width * height) / max(1, image_width * image_height))


def compute_exposure_score(image: Image.Image) -> float:
    gray = compute_grayscale(image)
    mean_value = float(np.mean(gray))
    std_value = float(np.std(gray))
    mean_component = 1.0 - min(abs(mean_value - 128.0) / 128.0, 1.0)
    std_component = normalize_score(std_value, 18.0, 75.0)
    return float((mean_component * 0.55) + (std_component * 0.45))


def compute_quality_score(image: Image.Image) -> float:
    blur = compute_blur_score(image)
    exposure = compute_exposure_score(image)
    return float((blur * 0.6) + (exposure * 0.4))


def _estimate_fake_probability(
    image: Image.Image,
    image_size: tuple[int, int],
    bbox: Iterable[int],
) -> tuple[float, dict[str, float]]:
    blur_score = compute_blur_score(image)
    dct_score = compute_dct_high_freq_score(image)
    compression_score = compute_compression_score(image)
    face_size_ratio = compute_face_size_ratio(bbox, image_size)
    quality_score = compute_quality_score(image)

    fake_score = (
        dct_score * 0.36
        + compression_score * 0.24
        + (1.0 - blur_score) * 0.18
        + min(face_size_ratio / 0.35, 1.0) * 0.08
        + (1.0 - quality_score) * 0.14
    )
    fake_score = float(min(max(fake_score, 0.0), 1.0))
    metrics = {
        "blur_score": blur_score,
        "dct_score": dct_score,
        "compression_score": compression_score,
        "face_size_ratio": face_size_ratio,
        "quality_score": quality_score,
    }
    return fake_score, metrics


def compute_augmentation_consistency_score(
    image: Image.Image,
    image_size: tuple[int, int],
    bbox: Iterable[int],
) -> float:
    base_score, _ = _estimate_fake_probability(image, image_size, bbox)
    variants = [
        image.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        ImageEnhance.Brightness(image).enhance(1.15),
        ImageEnhance.Contrast(image).enhance(1.10),
    ]
    scores = [base_score]
    for variant in variants:
        score, _ = _estimate_fake_probability(variant, image_size, bbox)
        scores.append(score)
    variation = float(np.std(scores))
    return float(1.0 - min(variation / 0.25, 1.0))


def compute_histogram_embedding(image: Image.Image, bins: int = 8) -> list[float]:
    array = image_to_array(image)
    features: list[float] = []
    for channel_index in range(3):
        hist, _ = np.histogram(array[:, :, channel_index], bins=bins, range=(0, 255), density=True)
        features.extend(hist.tolist())
    return [round(float(value), 6) for value in features]


def crop_face(image: Image.Image, bbox: Iterable[int], size: int = 256) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = bbox
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    margin_x = int(box_width * 0.18)
    margin_y = int(box_height * 0.22)
    crop = image.crop(
        (
            max(0, x1 - margin_x),
            max(0, y1 - margin_y),
            min(width, x2 + margin_x),
            min(height, y2 + margin_y),
        )
    )
    return crop.resize((size, size))


def generate_anomaly_map(image: Image.Image) -> np.ndarray:
    gray = compute_grayscale(image)
    blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=3)).convert("L")).astype(np.float32)
    high_freq = np.abs(gray - blurred)
    lap = np.abs(
        -4 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    anomaly = high_freq * 0.65 + lap * 0.35
    anomaly -= anomaly.min()
    if anomaly.max() > 0:
        anomaly = anomaly / anomaly.max()
    return anomaly


def overlay_heatmap(image: Image.Image, anomaly_map: np.ndarray) -> Image.Image:
    base = image_to_array(image)
    heat = np.zeros_like(base)
    heat[:, :, 0] = anomaly_map * 255
    heat[:, :, 1] = anomaly_map * 128
    overlay = (base * 0.62) + (heat * 0.38)
    return array_to_image(overlay)


def to_jpeg_bytes(image: Image.Image, quality: int = 90) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def simple_face_like_probability(image: Image.Image) -> float:
    gray = compute_grayscale(image)
    variance = float(np.var(gray))
    mean_value = float(np.mean(gray))
    ratio = normalize_score(variance, 90.0, 2000.0) * 0.75 + normalize_score(mean_value, 40.0, 200.0) * 0.25
    return min(max(ratio, 0.0), 1.0)


def build_default_landmarks(bbox: Iterable[int]) -> list[list[float]]:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return [
        [x1 + width * 0.3, y1 + height * 0.38],
        [x1 + width * 0.7, y1 + height * 0.38],
        [x1 + width * 0.5, y1 + height * 0.55],
        [x1 + width * 0.35, y1 + height * 0.72],
        [x1 + width * 0.65, y1 + height * 0.72],
    ]


def build_center_bbox(image_size: tuple[int, int]) -> list[int]:
    width, height = image_size
    side = int(min(width, height) * 0.54)
    cx = width // 2
    cy = height // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    return [x1, y1, min(width, x1 + side), min(height, y1 + side)]

