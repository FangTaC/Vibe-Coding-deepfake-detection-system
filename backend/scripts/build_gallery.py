from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

import sys


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import settings  # noqa: E402
from app.services.backends import FaceDetectorManager  # noqa: E402
from app.utils.image_ops import compute_histogram_embedding  # noqa: E402
from app.utils.image_ops import crop_face  # noqa: E402


def build_embedding(image_path: Path) -> list[float]:
    image = Image.open(image_path).convert("RGB")
    detections = FaceDetectorManager().detect(image)
    if detections:
        image = crop_face(image, detections[0].bbox)
    else:
        image = image.resize((256, 256))
    return compute_histogram_embedding(image)


def main() -> None:
    gallery_dir = settings.gallery_dir
    manifest_path = gallery_dir / "gallery.json"
    manifest = {"people": []}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    updated_people = []
    for person in manifest.get("people", []):
        name = person.get("name")
        if not name:
            continue
        image_path = gallery_dir / f"{name}.jpg"
        if not image_path.exists():
            updated_people.append(person)
            continue
        embedding = build_embedding(image_path)
        updated_people.append({**person, "embedding": embedding})

    manifest["people"] = updated_people
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Updated {len(updated_people)} gallery entries -> {manifest_path}")


if __name__ == "__main__":
    main()
