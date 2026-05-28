from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import settings
from app.services.model_registry import VISUAL_MODEL_SPECS, model_registry
from app.utils.image_ops import (
    _estimate_fake_probability,
    build_center_bbox,
    build_default_landmarks,
    compute_augmentation_consistency_score,
    compute_histogram_embedding,
    crop_face,
    image_to_array,
    simple_face_like_probability,
)


@dataclass(slots=True)
class DetectedFace:
    bbox: list[int]
    landmarks: list[list[float]]
    embedding: list[float]
    backend: str
    provider: str = ""


class InsightFaceDetector:
    def __init__(self) -> None:
        self._backend = None
        self._available = None
        self.backend_name = "insightface"
        self.provider_name = "uninitialized"

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        model_dir = settings.insightface_root_dir / "models" / settings.insightface_name
        if not model_dir.exists() and not settings.insightface_allow_download:
            self.provider_name = "assets_missing"
            self._available = False
            return False
        try:
            import onnxruntime as ort  # type: ignore
            from insightface.app import FaceAnalysis  # type: ignore
        except Exception:
            self.provider_name = "unavailable"
            self._available = False
            return False

        candidate_providers = []
        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in available_providers:
            candidate_providers.append("CUDAExecutionProvider")
        if "CPUExecutionProvider" in available_providers:
            candidate_providers.append("CPUExecutionProvider")

        for provider in candidate_providers:
            try:
                backend = FaceAnalysis(
                    name=settings.insightface_name,
                    root=str(settings.insightface_root_dir),
                    providers=[provider],
                )
                ctx_id = 0 if provider == "CUDAExecutionProvider" else -1
                backend.prepare(ctx_id=ctx_id, det_size=(640, 640))
                self._backend = backend
                self.provider_name = provider
                self._available = True
                return True
            except Exception:
                continue

        self.provider_name = "unavailable"
        self._available = False
        return False

    def detect(self, image: Image.Image) -> list[DetectedFace]:
        if not self.is_available():
            return []
        array = image_to_array(image)[:, :, ::-1].astype(np.uint8)
        raw_faces = self._backend.get(array)
        detections: list[DetectedFace] = []
        for raw_face in raw_faces[: settings.max_faces]:
            bbox = [int(value) for value in raw_face.bbox.tolist()]
            landmarks = [[float(x), float(y)] for x, y in raw_face.kps.tolist()]
            embedding = raw_face.embedding.tolist() if hasattr(raw_face, "embedding") and raw_face.embedding is not None else []
            detections.append(
                DetectedFace(
                    bbox=bbox,
                    landmarks=landmarks,
                    embedding=embedding,
                    backend=self.backend_name,
                    provider=self.provider_name,
                )
            )
        return detections


class OpenCVHaarDetector:
    def __init__(self) -> None:
        self._cascade = None
        self._available = None
        self.backend_name = "opencv_haar"

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import cv2  # type: ignore

            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(str(cascade_path))
            self._cascade = cascade if not cascade.empty() else None
            self._available = self._cascade is not None
        except Exception:
            self._available = False
        return self._available

    def detect(self, image: Image.Image) -> list[DetectedFace]:
        if not self.is_available():
            return []
        # Resize large images to speed up detection and improve recall
        w_orig, h_orig = image.size
        scale = 1.0
        if max(w_orig, h_orig) > 1200:
            scale = 1200.0 / max(w_orig, h_orig)
            resized = image.resize((int(w_orig * scale), int(h_orig * scale)))
        else:
            resized = image
        gray = np.asarray(resized.convert("L")).astype(np.uint8)
        detections: list[DetectedFace] = []
        # Try progressively more permissive parameters to increase recall
        param_sets = [
            dict(scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)),
            dict(scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)),
        ]
        for params in param_sets:
            boxes = self._cascade.detectMultiScale(gray, **params)
            if len(boxes) > 0:
                break
        if len(boxes) == 0:
            return []
        for x, y, w, h in boxes[: settings.max_faces]:
            # Scale bbox back to original image coordinates
            bbox = [
                int(x / scale), int(y / scale),
                int((x + w) / scale), int((y + h) / scale),
            ]
            crop = crop_face(image, bbox)
            detections.append(
                DetectedFace(
                    bbox=bbox,
                    landmarks=build_default_landmarks(bbox),
                    embedding=compute_histogram_embedding(crop),
                    backend=self.backend_name,
                    provider="CPUExecutionProvider",
                )
            )
        return detections


class HeuristicFaceDetector:
    backend_name = "heuristic_detector"

    def detect(self, image: Image.Image) -> list[DetectedFace]:
        candidates: list[tuple[float, list[int]]] = []
        width, height = image.size
        candidate_boxes = [
            build_center_bbox(image.size),
            [0, 0, width // 2, height // 2],
            [width // 2, 0, width, height // 2],
            [0, height // 2, width // 2, height],
            [width // 2, height // 2, width, height],
            [0, 0, width // 2, height],
            [width // 2, 0, width, height],
        ]
        for bbox in candidate_boxes:
            face_crop = crop_face(image, bbox)
            probability = simple_face_like_probability(face_crop)
            candidates.append((probability, bbox))

        if not candidates or max(score for score, _ in candidates) < 0.60:
            return []

        selected: list[list[int]] = []
        for probability, bbox in sorted(candidates, key=lambda item: item[0], reverse=True):
            if len(selected) >= settings.max_faces:
                break
            if probability < 0.23:
                continue
            if any(iou(bbox, existing) > 0.35 for existing in selected):
                continue
            selected.append(bbox)

        detections: list[DetectedFace] = []
        for bbox in selected:
            face_crop = crop_face(image, bbox)
            detections.append(
                DetectedFace(
                    bbox=bbox,
                    landmarks=build_default_landmarks(bbox),
                    embedding=compute_histogram_embedding(face_crop),
                    backend=self.backend_name,
                    provider="heuristic",
                )
            )
        return detections


class FaceDetectorManager:
    def __init__(self) -> None:
        self.backends = [
            InsightFaceDetector(),
            OpenCVHaarDetector(),
            HeuristicFaceDetector(),
        ]
        self._backend_map = {backend.backend_name: backend for backend in self.backends}

    def detect(self, image: Image.Image) -> list[DetectedFace]:
        for backend in self.backends:
            detections = backend.detect(image)
            if detections:
                return detections[: settings.max_faces]
        return []

    def detect_with_backend(self, backend_name: str, image: Image.Image) -> tuple[list[DetectedFace], str]:
        backend = self._backend_map.get(backend_name)
        if backend is not None:
            detections = backend.detect(image)
            if detections:
                return detections[: settings.max_faces], backend.backend_name
        detections = self.detect(image)
        actual_backend = detections[0].backend if detections else (backend_name or "none")
        return detections, actual_backend


class TorchVisionVisualBackend:
    def __init__(self, backend_name: str) -> None:
        self.backend_name = backend_name
        self.spec = VISUAL_MODEL_SPECS[backend_name]
        self.model = None
        self.available = None
        self.metadata = model_registry.visual_metadata(backend_name)

    def _load_model(self) -> bool:
        if self.available is not None:
            return self.available
        weights_path = model_registry.visual_model_path(self.backend_name)
        if not weights_path or not Path(weights_path).exists() or not model_registry.visual_metadata_exists(self.backend_name):
            self.available = False
            return False
        try:
            import torch  # type: ignore
            import torchvision.models as models  # type: ignore

            model = self._build_model(models, torch)
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()
            self.model = model
            self.available = True
        except Exception:
            self.available = False
        return self.available

    def _build_model(self, models: Any, torch: Any) -> Any:
        arch = self.spec["arch"]
        if arch == "efficientnet_b0":
            model = models.efficientnet_b0(weights=None)
            classifier_in = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Linear(classifier_in, 2)
            return model
        if arch == "resnet50":
            model = models.resnet50(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, 2)
            return model
        if arch == "mobilenet_v3_large":
            model = models.mobilenet_v3_large(weights=None)
            classifier_in = model.classifier[3].in_features
            model.classifier[3] = torch.nn.Linear(classifier_in, 2)
            return model
        raise ValueError(f"unsupported visual architecture: {arch}")

    def analyze(self, face_image: Image.Image, image_size: tuple[int, int], bbox: list[int]) -> dict[str, Any]:
        if not self._load_model():
            raise RuntimeError("optional visual model is unavailable")
        import torch  # type: ignore
        from torchvision import transforms  # type: ignore

        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        tensor = transform(face_image).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        fake_score, metrics = _estimate_fake_probability(face_image, image_size, bbox)
        consistency = compute_augmentation_consistency_score(face_image, image_size, bbox)
        calibrated = float(np.clip((float(probs[1]) * 0.7) + (fake_score * 0.3), 0.0, 1.0))
        return {
            "deepfake_score": calibrated,
            "feature_embedding": compute_histogram_embedding(face_image),
            "quality_score": metrics["quality_score"],
            "dct_score": metrics["dct_score"],
            "blur_score": metrics["blur_score"],
            "compression_score": metrics["compression_score"],
            "face_size_ratio": metrics["face_size_ratio"],
            "augmentation_consistency_score": consistency,
            "backend_name": self.backend_name,
            "model_tag": self.metadata.get("model_name", self.spec["model_name"]),
        }


class EfficientNetVisualBackend(TorchVisionVisualBackend):
    def __init__(self) -> None:
        super().__init__("efficientnet_b0_optional")


class HeuristicVisualBackend:
    backend_name = "heuristic_visual"

    def analyze(self, face_image: Image.Image, image_size: tuple[int, int], bbox: list[int]) -> dict[str, Any]:
        deepfake_score, metrics = _estimate_fake_probability(face_image, image_size, bbox)
        consistency = compute_augmentation_consistency_score(face_image, image_size, bbox)
        deepfake_score = float(np.clip((deepfake_score * 0.78) + ((1.0 - consistency) * 0.22), 0.0, 1.0))
        return {
            "deepfake_score": deepfake_score,
            "feature_embedding": compute_histogram_embedding(face_image),
            "quality_score": metrics["quality_score"],
            "dct_score": metrics["dct_score"],
            "blur_score": metrics["blur_score"],
            "compression_score": metrics["compression_score"],
            "face_size_ratio": metrics["face_size_ratio"],
            "augmentation_consistency_score": consistency,
            "backend_name": self.backend_name,
            "model_tag": "heuristic_visual_baseline",
        }


class VisualBackendManager:
    def __init__(self) -> None:
        self.optional_backend = EfficientNetVisualBackend()
        self.resnet_backend = TorchVisionVisualBackend("resnet50_optional")
        self.mobilenet_backend = TorchVisionVisualBackend("mobilenet_v3_large_optional")
        self.fallback_backend = HeuristicVisualBackend()
        self._backend_map = {
            self.optional_backend.backend_name: self.optional_backend,
            self.resnet_backend.backend_name: self.resnet_backend,
            self.mobilenet_backend.backend_name: self.mobilenet_backend,
            self.fallback_backend.backend_name: self.fallback_backend,
        }

    def analyze(self, face_image: Image.Image, image_size: tuple[int, int], bbox: list[int]) -> dict[str, Any]:
        return self.analyze_with_backend(settings.visual_default_backend, face_image, image_size, bbox)

    def _select_auto_backend(
        self,
        face_image: Image.Image,
        image_size: tuple[int, int],
        bbox: list[int],
    ) -> tuple[str, list[str]]:
        _score, metrics = _estimate_fake_probability(face_image, image_size, bbox)
        reason_codes: list[str] = []
        face_size_ratio = float(metrics.get("face_size_ratio", 0.0))
        quality_score = float(metrics.get("quality_score", 0.0))
        blur_score = float(metrics.get("blur_score", 0.0))

        if quality_score >= 0.65 and face_size_ratio >= 0.05 and blur_score <= 0.65:
            reason_codes.append("VISUAL_EFFICIENTNET_FOR_CLEAR_SINGLE_FACE")
            preferred = "efficientnet_b0_optional"
        elif quality_score < 0.25 or face_size_ratio < 0.05 or (quality_score < 0.45 and blur_score > 0.65):
            reason_codes.append("VISUAL_RESNET_FOR_LOW_QUALITY_OR_SMALL_FACE")
            preferred = "resnet50_optional"
        else:
            reason_codes.append("VISUAL_EFFICIENTNET_DEFAULT")
            preferred = "efficientnet_b0_optional"

        if model_registry.visual_backend_ready(preferred):
            return preferred, reason_codes
        for fallback in ("efficientnet_b0_optional", "resnet50_optional", "mobilenet_v3_large_optional"):
            if model_registry.visual_backend_ready(fallback):
                reason_codes.append(f"VISUAL_FALLBACK_READY_{fallback.upper()}")
                return fallback, reason_codes
        reason_codes.append("VISUAL_HEURISTIC_ONLY")
        return "heuristic_visual", reason_codes

    def _analyze_primary_and_optional_review(
        self,
        primary_backend_name: str,
        face_image: Image.Image,
        image_size: tuple[int, int],
        bbox: list[int],
        reason_codes: list[str],
    ) -> dict[str, Any]:
        primary_backend = self._backend_map.get(primary_backend_name, self.optional_backend)
        try:
            primary = primary_backend.analyze(face_image, image_size, bbox)
        except Exception:
            primary = self.fallback_backend.analyze(face_image, image_size, bbox)
            primary_backend_name = self.fallback_backend.backend_name

        model_scores = {primary["backend_name"]: float(primary["deepfake_score"])}
        review_backend_name = ""
        primary_score = float(primary["deepfake_score"])
        if settings.visual_gray_low <= primary_score <= settings.visual_gray_high:
            review_backend_name = "resnet50_optional" if primary_backend_name != "resnet50_optional" else "efficientnet_b0_optional"
            review_backend = self._backend_map.get(review_backend_name)
            if review_backend is not None:
                try:
                    review = review_backend.analyze(face_image, image_size, bbox)
                    review_score = float(review["deepfake_score"])
                    model_scores[review["backend_name"]] = review_score
                    disagreement = abs(primary_score - review_score)
                    primary["deepfake_score"] = float(np.clip((primary_score + review_score) / 2.0, 0.0, 1.0))
                    primary["visual_model_disagreement"] = disagreement
                    reason_codes.append("VISUAL_GRAY_ZONE_SECOND_MODEL_REVIEW")
                    if disagreement >= settings.visual_disagreement_threshold:
                        reason_codes.append("VISUAL_MODEL_DISAGREEMENT")
                except Exception:
                    reason_codes.append("VISUAL_SECOND_MODEL_UNAVAILABLE")

        primary["visual_model_scores"] = model_scores
        primary["visual_selection_reason"] = reason_codes
        return primary

    def analyze_with_backend(
        self,
        backend_name: str,
        face_image: Image.Image,
        image_size: tuple[int, int],
        bbox: list[int],
    ) -> dict[str, Any]:
        if not backend_name or backend_name in {"auto", "auto_visual"}:
            backend_name, reason_codes = self._select_auto_backend(face_image, image_size, bbox)
            return self._analyze_primary_and_optional_review(backend_name, face_image, image_size, bbox, reason_codes)

        reason_codes = [f"VISUAL_REQUESTED_{backend_name.upper()}"]
        backend = self._backend_map.get(backend_name)
        if backend is not None:
            if backend_name != self.fallback_backend.backend_name and not model_registry.visual_backend_ready(backend_name):
                return self.analyze_with_backend("auto", face_image, image_size, bbox)
            return self._analyze_primary_and_optional_review(backend_name, face_image, image_size, bbox, reason_codes)
        return self.analyze_with_backend("auto", face_image, image_size, bbox)

    def describe(self, backend_name: str) -> dict[str, str]:
        metadata = model_registry.visual_metadata(backend_name)
        weights_path = model_registry.visual_model_path(backend_name)
        metadata_path = model_registry.visual_metadata_path_for(backend_name)
        return {
            "visual": backend_name,
            "visual_weights_path": str(weights_path or settings.vision_model_path or ""),
            "visual_metadata_path": str(metadata_path or settings.visual_metadata_path or ""),
            "visual_model_name": str(metadata.get("model_name", "heuristic_visual_baseline" if backend_name == "heuristic_visual" else backend_name)),
        }


class XGBoostFusionBackend:
    backend_name = "xgboost_optional"

    def __init__(self) -> None:
        self.model = None
        self.available = None
        self.metadata = model_registry.fusion_metadata()

    def _load(self) -> bool:
        if self.available is not None:
            return self.available
        try:
            import xgboost as xgb  # type: ignore

            model_path = settings.fusion_model_path
            if not model_path or not Path(model_path).exists() or not model_registry.fusion_metadata_exists():
                self.available = False
                return False
            booster = xgb.Booster()
            booster.load_model(model_path)
            self.model = booster
            self.available = True
        except Exception:
            self.available = False
        return self.available

    def predict(self, features: list[float]) -> float:
        if not self._load():
            raise RuntimeError("optional xgboost model is unavailable")
        import xgboost as xgb  # type: ignore

        data = xgb.DMatrix(np.array([features], dtype=np.float32))
        return float(np.clip(self.model.predict(data)[0], 0.0, 1.0))


class HeuristicFusionBackend:
    backend_name = "heuristic_fusion"

    def predict(self, features: list[float]) -> float:
        max_face, mean_face, dct_mean, quality_mean, consistency_mean, compression_mean, face_count, semantic_score = features
        raw = (
            max_face * 0.38
            + mean_face * 0.18
            + dct_mean * 0.10
            + (1.0 - quality_mean) * 0.12
            + (1.0 - consistency_mean) * 0.08
            + compression_mean * 0.06
            + min(face_count / 3.0, 1.0) * 0.03
            + semantic_score * 0.05
        )
        return float(np.clip(raw, 0.0, 1.0))


class FusionBackendManager:
    def __init__(self) -> None:
        self.optional_backend = XGBoostFusionBackend()
        self.fallback_backend = HeuristicFusionBackend()
        self._backend_map = {
            self.optional_backend.backend_name: self.optional_backend,
            self.fallback_backend.backend_name: self.fallback_backend,
        }

    def predict(self, features: list[float]) -> tuple[float, str]:
        try:
            return self.optional_backend.predict(features), self.optional_backend.backend_name
        except Exception:
            return self.fallback_backend.predict(features), self.fallback_backend.backend_name

    def predict_with_backend(self, backend_name: str, features: list[float]) -> tuple[float, str]:
        backend = self._backend_map.get(backend_name)
        if backend is not None:
            try:
                return backend.predict(features), backend.backend_name
            except Exception:
                pass
        return self.predict(features)

    def describe(self, backend_name: str) -> dict[str, str]:
        metadata = model_registry.fusion_metadata()
        return {
            "fusion": backend_name,
            "fusion_model_path": settings.fusion_model_path or "",
            "fusion_metadata_path": settings.fusion_metadata_path or "",
            "fusion_model_name": str(metadata.get("model_name", "heuristic_fusion_baseline" if backend_name == "heuristic_fusion" else "xgboost_fusion")),
        }


class PublicFigureGallery:
    def __init__(self) -> None:
        self.gallery_manifest = settings.gallery_dir / "gallery.json"

    def load(self) -> list[dict[str, Any]]:
        if not self.gallery_manifest.exists():
            return []
        try:
            return json.loads(self.gallery_manifest.read_text(encoding="utf-8")).get("people", [])
        except Exception:
            return []

    def match(self, face_embedding: list[float]) -> list[str]:
        if not face_embedding:
            return []
        candidates: list[tuple[str, float]] = []
        for item in self.load():
            reference_embedding = item.get("embedding", [])
            if not reference_embedding or len(reference_embedding) != len(face_embedding):
                continue
            similarity = cosine_similarity(face_embedding, reference_embedding)
            if similarity >= 0.93:
                candidates.append((item.get("name", "未知人物"), similarity))
        candidates.sort(key=lambda item: item[1], reverse=True)
        return [name for name, _ in candidates[:3]]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    left_array = np.asarray(left, dtype=np.float32)
    right_array = np.asarray(right, dtype=np.float32)
    denom = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    if denom == 0:
        return 0.0
    return float(np.dot(left_array, right_array) / denom)


def iou(left: list[int], right: list[int]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    left_area = float(max(1, lx2 - lx1) * max(1, ly2 - ly1))
    right_area = float(max(1, rx2 - rx1) * max(1, ry2 - ry1))
    return inter / max(left_area + right_area - inter, 1.0)


class LocalQwenSemanticBackend:
    backend_name = "qwen3_vl_local"

    def __init__(self) -> None:
        self._pipeline = None
        self._available = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        if settings.semantic_backend == "heuristic":
            self._available = False
            return False
        if settings.semantic_force_offline and settings.semantic_model_path is None:
            self._available = False
            return False
        try:
            from transformers import pipeline  # type: ignore

            model_id = settings.semantic_model_path or settings.semantic_model_name
            self._pipeline = pipeline(
                "image-text-to-text",
                model=model_id,
                trust_remote_code=True,
                device_map="auto",
            )
            self._available = True
        except Exception:
            self._available = False
        return self._available

    def analyze(self, image_path: str, context: dict[str, Any]) -> dict[str, Any]:
        if not self.is_available():
            raise RuntimeError("local Qwen backend is unavailable")
        prompt = (
            "你是 deepfake 图像检测系统中的语义辅助模块。"
            "请只根据输入图像和上下文输出 JSON，字段必须包含 semantic_flags、semantic_score、summary。"
            "semantic_flags 中的 code 只能从 "
            "[PUBLIC_FIGURE_CONTEXT, SCENE_COMMONSENSE_CONFLICT, IDENTITY_SWAP_RISK, CROWD_CONTEXT, LOW_INFORMATION] "
            "中选择。不要输出最终真假结论。"
        )
        result = self._pipeline(image=image_path, text=prompt, max_new_tokens=192)
        generated = ""
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict):
                generated = first.get("generated_text", "")
        payload = extract_json_object(generated)
        return payload


class RuleBasedSemanticBackend:
    backend_name = "rule_based_semantic"

    def analyze(self, image_path: str, context: dict[str, Any]) -> dict[str, Any]:
        flags: list[dict[str, Any]] = []
        score = 0.0
        public_figure_candidates = context.get("public_figure_candidates", [])
        face_count = context.get("face_count", 0)
        conflict_level = context.get("conflict_level", 0.0)
        quality_score = context.get("quality_score", 0.0)
        max_face_score = context.get("max_face_score", 0.0)

        if public_figure_candidates:
            flags.append(
                {
                    "code": "PUBLIC_FIGURE_CONTEXT",
                    "label": f"检测到公共人物候选：{', '.join(public_figure_candidates[:2])}",
                    "severity": "warning",
                    "confidence": 0.72,
                    "evidence": "公共人物图库候选命中，需要结合语义上下文复核身份与场景。",
                }
            )
            score += 0.18
        if face_count > 1:
            flags.append(
                {
                    "code": "CROWD_CONTEXT",
                    "label": "检测到多人脸场景",
                    "severity": "info",
                    "confidence": 0.68,
                    "evidence": "多人脸场景更容易出现换脸或局部篡改，已提升复核优先级。",
                }
            )
            score += 0.08
        if conflict_level >= 0.35:
            flags.append(
                {
                    "code": "SCENE_COMMONSENSE_CONFLICT",
                    "label": "视觉证据存在冲突",
                    "severity": "warning",
                    "confidence": min(0.95, 0.55 + conflict_level),
                    "evidence": "多项视觉指标不一致，建议触发语义辅助判断。",
                }
            )
            score += 0.16
        if quality_score < settings.low_quality_threshold:
            flags.append(
                {
                    "code": "LOW_INFORMATION",
                    "label": "图像信息量不足",
                    "severity": "warning",
                    "confidence": 0.82,
                    "evidence": "图像质量过低时语义与视觉都更容易不稳定。",
                }
            )
            score += 0.06
        if max_face_score >= 0.55 and public_figure_candidates:
            flags.append(
                {
                    "code": "IDENTITY_SWAP_RISK",
                    "label": "存在身份置换风险",
                    "severity": "high",
                    "confidence": 0.77,
                    "evidence": "公共人物候选与视觉伪造分数同时偏高，适合作为答辩中的语义异常案例。",
                }
            )
            score += 0.24

        summary = "未触发显著语义异常。"
        if flags:
            summary = "；".join(flag["label"] for flag in flags)
        return {
            "semantic_flags": flags,
            "semantic_score": float(np.clip(score, 0.0, 1.0)),
            "summary": summary,
        }


class SemanticBackendManager:
    def __init__(self) -> None:
        self.qwen_api_backend = QwenVisionSemanticBackend()
        self.optional_backend = LocalQwenSemanticBackend()
        self.fallback_backend = RuleBasedSemanticBackend()
        self._backend_map = {
            self.qwen_api_backend.backend_name: self.qwen_api_backend,
            self.optional_backend.backend_name: self.optional_backend,
            self.fallback_backend.backend_name: self.fallback_backend,
        }

    def analyze(self, image_path: str, context: dict[str, Any]) -> tuple[dict[str, Any], str]:
        # 优先使用通义千问 API 进行视觉语义分析（一眼假检测）
        if self.qwen_api_backend.is_available():
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self.qwen_api_backend.analyze, image_path, context)
                    payload = future.result(timeout=settings.llm_timeout_seconds)
                return payload, self.qwen_api_backend.backend_name
            except (FutureTimeoutError, Exception):
                pass
        # 次选本地 Qwen VL 模型
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.optional_backend.analyze, image_path, context)
                payload = future.result(timeout=settings.semantic_timeout_seconds)
            return payload, self.optional_backend.backend_name
        except FutureTimeoutError:
            timeout_payload = self.fallback_backend.analyze(image_path, context)
            timeout_payload["semantic_flags"] = timeout_payload.get("semantic_flags", []) + [
                {
                    "code": "SEMANTIC_TIMEOUT",
                    "label": "语义模型超时，已回退到规则模式",
                    "severity": "warning",
                    "confidence": 0.9,
                    "evidence": f"语义模块超过 {settings.semantic_timeout_seconds}s 未返回。",
                }
            ]
            timeout_payload["summary"] = "语义模型超时，系统已自动回退为规则式语义分析。"
            return timeout_payload, f"{self.fallback_backend.backend_name}_after_timeout"
        except Exception:
            return self.fallback_backend.analyze(image_path, context), self.fallback_backend.backend_name

    def analyze_with_backend(
        self,
        backend_name: str,
        image_path: str,
        context: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        backend = self._backend_map.get(backend_name)
        if backend is not None:
            try:
                if backend is self.qwen_api_backend:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(backend.analyze, image_path, context)
                        return future.result(timeout=settings.llm_timeout_seconds), backend.backend_name
                if backend is self.optional_backend:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(backend.analyze, image_path, context)
                        return future.result(timeout=settings.semantic_timeout_seconds), backend.backend_name
                return backend.analyze(image_path, context), backend.backend_name
            except Exception:
                pass
        return self.analyze(image_path, context)


class QwenStrategyBackend:
    """通义千问 LLM 策略选择后端：使用大模型分析任务画像并选择检测策略，输出完整思维决策链路。"""

    backend_name = "qwen_strategy"

    SYSTEM_PROMPT = (
        "你是 deepfake 检测系统中的决策智能体，负责根据特征提取结果选择最合适的检测策略。\n\n"
        "可用策略及适用场景：\n"
        "- visual_only：仅使用视觉深度学习模型（EfficientNet）判断。适用于分数明确（低于真图阈值或高于伪造阈值）、"
        "图像质量良好、无公众人物候选、视觉指标一致的简单案例。\n"
        "- visual_plus_semantic：视觉+语义联合分析。适用于分数落入灰区（介于两个阈值之间）、检测到公众人物候选"
        "（存在身份置换风险）、多人脸场景、视觉证据指标存在冲突（conflict_level≥0.35）等复杂案例。\n"
        "- review_only：转人工复核。适用于图像质量过低（quality_score<0.35）或输入不在系统适用范围内。\n\n"
        "你必须严格输出 JSON 格式（不含任何额外文本）：\n"
        '{"selected_strategy": "...", "reason_codes": [...], "semantic_required": true/false, '
        '"expected_risk_level": "low|medium|high", "thinking_chain": "详细推理过程"}\n\n'
        "reason_codes 可选值：DEFAULT_VISUAL_PATH, GRAY_ZONE_SCORE, MULTI_FACE_SCENE, "
        "PUBLIC_FIGURE_CANDIDATE, VISUAL_CONFLICT, LOW_QUALITY_INPUT, NEED_SEMANTIC_CHECK\n\n"
        "thinking_chain 字段必须详细说明：你分析了哪些指标、各指标的含义、为什么选择该策略、预期风险等级的理由。"
    )

    def __init__(self) -> None:
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        if not settings.llm_api_key or not settings.llm_strategy_enabled:
            self._available = False
            return False
        self._available = True
        return self._available

    def select_strategy(self, task_profile: dict[str, Any], threshold_policy: dict[str, Any]) -> dict[str, Any]:
        if not self.is_available():
            raise RuntimeError("Qwen strategy backend is unavailable")
        import httpx

        user_message = (
            "请根据以下任务画像选择最合适的检测策略，并在 thinking_chain 字段中详细说明你的推理过程：\n\n"
            f"任务画像：\n{json.dumps(task_profile, ensure_ascii=False, indent=2)}\n\n"
            f"当前阈值策略：真图阈值={threshold_policy['true_threshold']}，"
            f"伪造阈值={threshold_policy['fake_threshold']}"
        )
        response = httpx.post(
            f"{settings.llm_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.llm_api_key}", "Content-Type": "application/json"},
            json={
                "model": settings.llm_text_model,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "temperature": 0.1,
                "max_tokens": 800,
            },
            timeout=float(settings.llm_timeout_seconds),
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        payload = extract_json_object(content)
        if not payload.get("thinking_chain"):
            payload["thinking_chain"] = content
        return payload


class QwenVisionSemanticBackend:
    """通义千问 VL API 语义后端：利用多模态大模型识别"一眼假"图像。"""

    backend_name = "qwen_vl_api"

    SYSTEM_PROMPT = (
        "你是 deepfake 检测系统的语义辅助智能体，专门识别人眼可直接判断为虚假的图像（\"一眼假\"）。\n\n"
        "请从以下几个维度分析图像的语义真实性：\n"
        "1. 场景常识冲突：人物与场景组合是否违反常识（例如历史上从未发生的政治场景、不相干人物的奇异组合）\n"
        "2. 身份置换风险：知名公众人物是否以异常方式出现（面部特征与典型形象不符、身份上下文可疑）\n"
        "3. 视觉拼接异常：背景、光照、面部边缘是否存在明显不自然的拼接痕迹\n"
        "4. 上下文语义异常：图像整体叙事是否合理，是否存在\"一眼看出来是假的\"特征\n\n"
        "你必须严格输出 JSON 格式：\n"
        '{"semantic_flags": [{"code": "...", "label": "...", "severity": "info|warning|high", '
        '"confidence": 0.0~1.0, "evidence": "..."}], "semantic_score": 0.0~1.0, '
        '"summary": "一句话摘要", "thinking_chain": "详细分析推理过程"}\n\n'
        "code 只能从以下选择：PUBLIC_FIGURE_CONTEXT, SCENE_COMMONSENSE_CONFLICT, IDENTITY_SWAP_RISK, "
        "CROWD_CONTEXT, LOW_INFORMATION\n"
        "不要给出最终真假结论，只提供语义层面的证据与推理。"
    )

    def __init__(self) -> None:
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        if not settings.llm_api_key:
            self._available = False
            return False
        self._available = True
        return self._available

    def analyze(self, image_path: str, context: dict[str, Any]) -> dict[str, Any]:
        if not self.is_available():
            raise RuntimeError("Qwen VL API semantic backend is unavailable")
        import base64
        import httpx
        from pathlib import Path as _Path

        image_bytes = _Path(image_path).read_bytes()
        image_b64 = base64.standard_b64encode(image_bytes).decode()
        suffix = _Path(image_path).suffix.lower().lstrip(".")
        media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(
            suffix, "image/jpeg"
        )

        response = httpx.post(
            f"{settings.llm_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.llm_api_key}", "Content-Type": "application/json"},
            json={
                "model": settings.llm_vision_model,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_b64}"}},
                            {
                                "type": "text",
                                "text": (
                                    "请对这张图像进行语义层面真实性分析，重点识别是否存在\"一眼假\"特征，"
                                    "并在 thinking_chain 字段中详细记录你的推理过程。\n"
                                    f"视觉分析上下文：{json.dumps(context, ensure_ascii=False)}"
                                ),
                            },
                        ],
                    },
                ],
                "temperature": 0.1,
                "max_tokens": 800,
            },
            timeout=float(settings.llm_timeout_seconds),
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        payload = extract_json_object(content)
        if not payload.get("thinking_chain"):
            payload["thinking_chain"] = content
        return payload


def extract_json_object(text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("empty model response")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in response")
    return json.loads(text[start : end + 1])
