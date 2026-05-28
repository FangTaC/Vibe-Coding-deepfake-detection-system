from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.model_registry import model_registry
from app.utils.image_ops import load_image


@dataclass(slots=True)
class RoutingDecision:
    detector_backend: str
    visual_backend: str
    fusion_backend: str
    semantic_backend: str
    reason_codes: list[str]
    image_width: int
    image_height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_backend": self.detector_backend,
            "visual_backend": self.visual_backend,
            "fusion_backend": self.fusion_backend,
            "semantic_backend": self.semantic_backend,
            "reason_codes": list(self.reason_codes),
            "image_width": self.image_width,
            "image_height": self.image_height,
        }


class AlgorithmRouter:
    """Selects preferred backends from a constrained whitelist."""

    def route(self, image_path: str) -> dict[str, Any]:
        image = load_image(image_path)
        status = model_registry.build_runtime_status()
        reason_codes: list[str] = []

        detector_backend = self._select_detector(status, image.size, reason_codes)
        visual_backend = self._select_visual(status, reason_codes)
        fusion_backend = self._select_fusion(status, reason_codes)
        semantic_backend = self._select_semantic(status, image.size, reason_codes)

        return RoutingDecision(
            detector_backend=detector_backend,
            visual_backend=visual_backend,
            fusion_backend=fusion_backend,
            semantic_backend=semantic_backend,
            reason_codes=reason_codes,
            image_width=image.width,
            image_height=image.height,
        ).to_dict()

    def _select_detector(
        self,
        status: dict[str, Any],
        image_size: tuple[int, int],
        reason_codes: list[str],
    ) -> str:
        width, height = image_size
        detector_status = status["detector"]["effective_status"]
        if detector_status == "ready":
            reason_codes.append("DETECTOR_INSIGHTFACE_READY")
            return "insightface"
        if self._module_ready("cv2"):
            if max(width, height) >= 320:
                reason_codes.append("DETECTOR_OPENCV_FALLBACK")
                return "opencv_haar"
        reason_codes.append("DETECTOR_HEURISTIC_ONLY")
        return "heuristic_detector"

    def _select_visual(self, status: dict[str, Any], reason_codes: list[str]) -> str:
        ready_backends = status["visual"].get("ready_backends", [])
        if ready_backends:
            reason_codes.append("VISUAL_AUTO_MULTI_MODEL_READY")
            return "auto_visual"
        reason_codes.append("VISUAL_HEURISTIC_FALLBACK")
        return "heuristic_visual"

    def _select_fusion(self, status: dict[str, Any], reason_codes: list[str]) -> str:
        fusion_ready = status["fusion"]["effective_status"] == "ready" and status["fusion"]["xgboost_installed"]
        if fusion_ready:
            reason_codes.append("FUSION_XGBOOST_READY")
            return "xgboost_optional"
        reason_codes.append("FUSION_HEURISTIC_FALLBACK")
        return "heuristic_fusion"

    def _select_semantic(
        self,
        status: dict[str, Any],
        image_size: tuple[int, int],
        reason_codes: list[str],
    ) -> str:
        width, height = image_size
        if status["llm"]["effective_status"] == "ready" and max(width, height) <= 4096:
            reason_codes.append("SEMANTIC_QWEN_API_READY")
            return "qwen_vl_api"
        if status["semantic"]["effective_status"] == "ready" and status["semantic"]["transformers_installed"]:
            reason_codes.append("SEMANTIC_LOCAL_QWEN_READY")
            return "qwen3_vl_local"
        reason_codes.append("SEMANTIC_RULE_BASED_FALLBACK")
        return "rule_based_semantic"

    @staticmethod
    def _module_ready(module_name: str) -> bool:
        return model_registry.module_installed(module_name)


algorithm_router = AlgorithmRouter()
