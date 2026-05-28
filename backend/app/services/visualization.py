from __future__ import annotations

from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from app.services.storage import storage_service
from app.utils.image_ops import generate_anomaly_map, overlay_heatmap


def _get_font(size: int = 16):
    """Return a PIL font; fall back to the built-in default if no TTF is available."""
    try:
        # Try common system font locations
        import os
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/msyh.ttc",  # Microsoft YaHei
        ]
        for path in candidates:
            if os.path.exists(path):
                return ImageFont.truetype(path, size)
    except Exception:
        pass
    return ImageFont.load_default()


def save_annotated_image(task_id: str, image: Image.Image, boxes: list[Iterable[int]]) -> str:
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size

    if not boxes:
        # Draw a semi-transparent overlay notice
        font = _get_font(max(14, min(24, w // 20)))
        msg = "No face detected"
        try:
            bbox_text = draw.textbbox((0, 0), msg, font=font)
            tw = bbox_text[2] - bbox_text[0]
            th = bbox_text[3] - bbox_text[1]
        except AttributeError:
            tw, th = draw.textsize(msg, font=font)  # Pillow < 9
        # Draw a dark banner at the bottom
        banner_h = th + 16
        draw.rectangle([0, h - banner_h, w, h], fill=(30, 30, 30, 200))
        draw.text(
            ((w - tw) // 2, h - banner_h + 8),
            msg,
            fill=(255, 200, 80),
            font=font,
        )
        # Draw a thin border around the image
        draw.rectangle([0, 0, w - 1, h - 1], outline=(200, 160, 40), width=2)
    else:
        font = _get_font(max(13, min(20, w // 30)))
        for index, bbox in enumerate(boxes):
            x1, y1, x2, y2 = bbox
            # Draw filled semi-transparent label background
            label = f"Face {index + 1}"
            try:
                tb = draw.textbbox((0, 0), label, font=font)
                lw = tb[2] - tb[0] + 8
                lh = tb[3] - tb[1] + 4
            except AttributeError:
                lw, lh = draw.textsize(label, font=font)
                lw += 8
                lh += 4
            draw.rectangle([x1, y1, x2, y2], outline=(18, 224, 182), width=3)
            draw.rectangle([x1, y1, x1 + lw, y1 + lh], fill=(18, 224, 182))
            draw.text((x1 + 4, y1 + 2), label, fill=(20, 40, 35), font=font)

    file_name = "annotated_original.jpg"
    annotated.save(storage_service.artifact_path(task_id, file_name), format="JPEG", quality=92)
    return file_name


def save_face_crop(task_id: str, face_id: str, image: Image.Image) -> str:
    file_name = f"{face_id}_aligned.jpg"
    image.save(storage_service.artifact_path(task_id, file_name), format="JPEG", quality=94)
    return file_name


def save_heatmap(task_id: str, face_id: str, image: Image.Image) -> str:
    anomaly_map = generate_anomaly_map(image)
    heatmap = overlay_heatmap(image, anomaly_map)
    file_name = f"{face_id}_gradcam.jpg"
    heatmap.save(storage_service.artifact_path(task_id, file_name), format="JPEG", quality=90)
    return file_name

