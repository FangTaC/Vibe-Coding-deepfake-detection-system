from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

import sys


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import settings  # noqa: E402


def draw_face(draw: ImageDraw.ImageDraw, center: tuple[int, int], size: int, skin: tuple[int, int, int], hair: tuple[int, int, int]) -> None:
    cx, cy = center
    w = size
    h = int(size * 1.18)
    draw.ellipse((cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2), fill=skin, outline=(92, 70, 60), width=3)
    draw.ellipse((cx - w // 2, cy - h // 2 - 16, cx + w // 2, cy - h // 6), fill=hair)
    draw.ellipse((cx - w // 4, cy - h // 8, cx - w // 8, cy + h // 20), fill=(30, 38, 49))
    draw.ellipse((cx + w // 8, cy - h // 8, cx + w // 4, cy + h // 20), fill=(30, 38, 49))
    draw.rectangle((cx - 10, cy - 2, cx + 10, cy + h // 6), fill=(223, 174, 150))
    draw.arc((cx - w // 5, cy + h // 8, cx + w // 5, cy + h // 3), start=5, end=175, fill=(140, 56, 70), width=4)


def save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def generate_gallery_images() -> None:
    for name, skin, hair, bg in (
        ("特朗普", (240, 202, 170), (215, 156, 59), (242, 235, 220)),
        ("马杜罗", (196, 152, 126), (42, 38, 35), (229, 232, 238)),
    ):
        image = Image.new("RGB", (420, 420), color=bg)
        draw = ImageDraw.Draw(image)
        draw_face(draw, (210, 210), 180, skin, hair)
        save(image, settings.gallery_dir / f"{name}.jpg")


def generate_demo_samples() -> None:
    samples_dir = settings.demo_samples_dir

    real_single = Image.new("RGB", (640, 640), color=(238, 235, 229))
    draw = ImageDraw.Draw(real_single)
    draw_face(draw, (320, 320), 240, (236, 200, 173), (80, 52, 44))
    save(real_single, samples_dir / "single_real.png")

    fake_single = real_single.copy()
    fake_draw = ImageDraw.Draw(fake_single)
    for row in range(0, 640, 28):
        tone = (255, 181, 76) if (row // 28) % 2 == 0 else (69, 200, 228)
        fake_draw.rectangle((0, row, 640, row + 9), fill=tone)
    for col in range(0, 640, 40):
        fake_draw.rectangle((col, 90, col + 5, 560), fill=(255, 240, 77))
    save(fake_single, samples_dir / "single_fake.png")

    no_face = Image.new("RGB", (640, 640), color=(118, 138, 168))
    draw = ImageDraw.Draw(no_face)
    draw.rectangle((90, 90, 550, 550), outline=(242, 246, 250), width=12)
    draw.text((180, 290), "NO FACE", fill=(242, 246, 250))
    save(no_face, samples_dir / "no_face.png")

    multi_face = Image.new("RGB", (960, 640), color=(236, 236, 232))
    draw = ImageDraw.Draw(multi_face)
    draw_face(draw, (260, 320), 200, (236, 198, 170), (82, 58, 46))
    draw_face(draw, (700, 320), 190, (196, 154, 128), (40, 40, 36))
    save(multi_face, samples_dir / "multi_face.png")

    low_quality = real_single.copy().resize((240, 240)).resize((640, 640)).filter(ImageFilter.GaussianBlur(radius=7))
    save(low_quality, samples_dir / "low_quality.png")

    public_figure = Image.new("RGB", (960, 640), color=(232, 234, 239))
    draw = ImageDraw.Draw(public_figure)
    draw.rectangle((0, 0, 960, 88), fill=(22, 52, 46))
    draw.text((36, 30), "Demo: public figure context / identity swap risk", fill=(244, 255, 252))
    trump = Image.open(settings.gallery_dir / "特朗普.jpg").convert("RGB").resize((320, 320))
    maduro = Image.open(settings.gallery_dir / "马杜罗.jpg").convert("RGB").resize((320, 320))
    public_figure.paste(maduro, (100, 180))
    public_figure.paste(trump, (540, 180))
    draw.text((170, 520), "TRUMP", fill=(30, 38, 49))
    draw.text((640, 520), "MADURO", fill=(30, 38, 49))
    save(public_figure, samples_dir / "public_figure_context.png")

    manifest = {
        "generated_at": "auto",
        "samples": [
            {"category": "single_real", "file": "single_real.png", "expected_hint": "疑似真实"},
            {"category": "single_fake", "file": "single_fake.png", "expected_hint": "疑似伪造"},
            {"category": "no_face", "file": "no_face.png", "expected_hint": "需人工复核"},
            {"category": "multi_face", "file": "multi_face.png", "expected_hint": "需人工复核或触发语义分析"},
            {"category": "low_quality", "file": "low_quality.png", "expected_hint": "需人工复核"},
            {"category": "public_figure_context", "file": "public_figure_context.png", "expected_hint": "触发公共人物候选或语义标记"},
        ],
    }
    Path(settings.demo_manifest_path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    settings.ensure_directories()
    generate_gallery_images()
    generate_demo_samples()
    print(f"Demo assets generated under {settings.demo_dir}")


if __name__ == "__main__":
    main()

