from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.schemas.task import ArtifactRef, EvidenceEntry, FaceResult
from app.services.backends import FaceDetectorManager, PublicFigureGallery, VisualBackendManager
from app.services.storage import storage_service
from app.services.visualization import save_annotated_image, save_face_crop, save_heatmap
from app.utils.image_ops import crop_face, load_image


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class FeatureExtractionAgent:
    name = "feature_agent"

    def __init__(self) -> None:
        self.detector = FaceDetectorManager()
        self.visual_backend = VisualBackendManager()
        self.gallery = PublicFigureGallery()

    def run(self, task_id: str, image_path: str, routing: dict[str, Any] | None = None) -> dict[str, Any]:
        image = load_image(image_path)
        routing = routing or {}
        requested_detector = routing.get("detector_backend", "")
        requested_visual = routing.get("visual_backend", "")
        detections, detector_backend_used = self.detector.detect_with_backend(requested_detector, image)
        evidence: list[EvidenceEntry] = [
            EvidenceEntry(
                step="input_loaded",
                actor=self.name,
                timestamp=now_utc(),
                summary="已读取输入图像并开始预处理。",
                details={
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                    "algorithm_routing": routing,
                },
            )
        ]

        if not detections:
            evidence.append(
                EvidenceEntry(
                    step="face_detection",
                    actor=self.name,
                    timestamp=now_utc(),
                    summary="未检测到有效人脸，任务将转入人工复核。",
                    details={
                        "face_count": 0,
                        "detector_backend": "none",
                        "detector_backend_requested": requested_detector or "default_chain",
                    },
                )
            )
            evidence.append(
                EvidenceEntry(
                    step="feature_extraction",
                    actor=self.name,
                    timestamp=now_utc(),
                    summary="未提取到可用人脸特征，任务将进入受限决策路径。",
                    details={
                        "face_count": 0,
                        "detector_backend": "none",
                        "classifier_backend": "none",
                        "visual_backend_requested": requested_visual or "default_chain",
                    },
                )
            )
            annotated_name = save_annotated_image(task_id, image, [])
            return {
                "faces": [],
                "artifacts": [
                    ArtifactRef(
                        name=annotated_name,
                        kind="annotated_original",
                        url=storage_service.artifact_url(task_id, annotated_name),
                        description="原图标注结果",
                    )
                ],
                "evidence": evidence,
                "input_summary": {
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                    "face_count": 0,
                },
                "model_versions": {
                    "detector": "none",
                    "visual": "none",
                    "detector_requested": requested_detector or "default_chain",
                    "visual_requested": requested_visual or "default_chain",
                },
                "routing": routing,
            }

        face_results: list[FaceResult] = []
        boxes = [detected_face.bbox for detected_face in detections]
        annotated_name = save_annotated_image(task_id, image, boxes)
        detector_provider = detections[0].provider if detections else "unknown"

        evidence.append(
            EvidenceEntry(
                step="face_detection",
                actor=self.name,
                timestamp=now_utc(),
                summary=f"检测到 {len(detections)} 张人脸，使用后端：{detections[0].backend}。",
                details={
                    "face_count": len(detections),
                    "detector_backend": detector_backend_used,
                    "detector_backend_requested": requested_detector or "default_chain",
                    "detector_provider": detector_provider,
                    "bboxes": [d.bbox for d in detections],
                },
            )
        )

        for index, detected_face in enumerate(detections):
            face_id = f"face_{index + 1}"
            face_crop = crop_face(image, detected_face.bbox)
            crop_name = save_face_crop(task_id, face_id, face_crop)
            heatmap_name = save_heatmap(task_id, face_id, face_crop)
            visual_backend_for_face = requested_visual
            if requested_visual in {"", "auto", "auto_visual"} and len(detections) >= 2:
                visual_backend_for_face = "mobilenet_v3_large_optional"
            analysis = self.visual_backend.analyze_with_backend(
                visual_backend_for_face,
                face_crop,
                image.size,
                detected_face.bbox,
            )
            public_figures = self.gallery.match(detected_face.embedding or analysis["feature_embedding"])
            artifacts = [
                ArtifactRef(
                    name=crop_name,
                    kind="aligned_face",
                    url=storage_service.artifact_url(task_id, crop_name),
                    description="对齐后的人脸裁剪图",
                ),
                ArtifactRef(
                    name=heatmap_name,
                    kind="gradcam",
                    url=storage_service.artifact_url(task_id, heatmap_name),
                    description="伪造痕迹热力图",
                ),
            ]
            face_results.append(
                FaceResult(
                    face_id=face_id,
                    bbox=detected_face.bbox,
                    landmarks=[[round(x, 2), round(y, 2)] for x, y in detected_face.landmarks],
                    deepfake_score=round(analysis["deepfake_score"], 4),
                    quality_score=round(analysis["quality_score"], 4),
                    dct_score=round(analysis["dct_score"], 4),
                    blur_score=round(analysis["blur_score"], 4),
                    compression_score=round(analysis["compression_score"], 4),
                    face_size_ratio=round(analysis["face_size_ratio"], 4),
                    augmentation_consistency_score=round(analysis["augmentation_consistency_score"], 4),
                    feature_embedding=analysis["feature_embedding"],
                    detector_backend=detected_face.backend,
                    classifier_backend=analysis["backend_name"],
                    visual_model_scores={
                        key: round(float(value), 4)
                        for key, value in analysis.get("visual_model_scores", {}).items()
                    },
                    visual_selection_reason=list(analysis.get("visual_selection_reason", [])),
                    visual_model_disagreement=(
                        round(float(analysis["visual_model_disagreement"]), 4)
                        if analysis.get("visual_model_disagreement") is not None
                        else None
                    ),
                    public_figure_candidates=public_figures,
                    artifacts=artifacts,
                )
            )

        evidence.append(
            EvidenceEntry(
                step="feature_extraction",
                actor=self.name,
                timestamp=now_utc(),
                summary=f"已完成 {len(face_results)} 张人脸的视觉证据提取。",
                details={
                    "face_count": len(face_results),
                    "detector_backend": face_results[0].detector_backend if face_results else "unknown",
                    "detector_provider": detector_provider,
                    "classifier_backend": face_results[0].classifier_backend if face_results else "unknown",
                    "visual_backend_requested": requested_visual or "default_chain",
                    "visual_selection_reason": face_results[0].visual_selection_reason if face_results else [],
                    "visual_model_scores": face_results[0].visual_model_scores if face_results else {},
                },
            )
        )

        all_artifacts = [
            ArtifactRef(
                name=annotated_name,
                kind="annotated_original",
                url=storage_service.artifact_url(task_id, annotated_name),
                description="带人脸框的原图",
            )
        ]
        for face in face_results:
            all_artifacts.extend(face.artifacts)

        model_versions = {
            "detector": face_results[0].detector_backend if face_results else "none",
            "detector_provider": detector_provider,
            "detector_requested": requested_detector or "default_chain",
            "visual_requested": requested_visual or "default_chain",
            "visual_selected": face_results[0].classifier_backend if face_results else "none",
        }
        model_versions.update(self.visual_backend.describe(face_results[0].classifier_backend if face_results else "none"))

        return {
            "faces": face_results,
            "artifacts": all_artifacts,
            "evidence": evidence,
            "input_summary": {
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "face_count": len(face_results),
            },
            "model_versions": model_versions,
            "routing": routing,
        }
