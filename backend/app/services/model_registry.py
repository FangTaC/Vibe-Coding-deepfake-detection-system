from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from app.core.config import settings

VISUAL_MODEL_SPECS: dict[str, dict[str, str]] = {
    "efficientnet_b0_optional": {
        "arch": "efficientnet_b0",
        "model_name": "efficientnet_b0_deepfake",
        "folder": "efficientnet_b0",
        "legacy_weights": "vision_model_path",
        "legacy_metadata": "visual_metadata_path",
    },
    "resnet50_optional": {
        "arch": "resnet50",
        "model_name": "resnet50_deepfake",
        "folder": "resnet50",
    },
    "mobilenet_v3_large_optional": {
        "arch": "mobilenet_v3_large",
        "model_name": "mobilenet_v3_large_deepfake",
        "folder": "mobilenet_v3_large",
    },
}


def _safe_load_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


class ModelRegistry:
    def __init__(self) -> None:
        settings.ensure_directories()

    def visual_model_path(self, backend_name: str = "efficientnet_b0_optional") -> Path | None:
        spec = VISUAL_MODEL_SPECS.get(backend_name)
        if not spec:
            return None
        legacy_key = spec.get("legacy_weights")
        if legacy_key:
            legacy_path = getattr(settings, legacy_key, None)
            if legacy_path and Path(legacy_path).exists():
                return Path(legacy_path)
        return settings.visual_model_dir / spec["folder"] / "model.pt"

    def visual_metadata_path_for(self, backend_name: str = "efficientnet_b0_optional") -> Path | None:
        spec = VISUAL_MODEL_SPECS.get(backend_name)
        if not spec:
            return None
        legacy_key = spec.get("legacy_metadata")
        if legacy_key:
            legacy_path = getattr(settings, legacy_key, None)
            if legacy_path and Path(legacy_path).exists():
                return Path(legacy_path)
        return settings.visual_model_dir / spec["folder"] / "meta.json"

    def visual_model_exists(self, backend_name: str = "efficientnet_b0_optional") -> bool:
        path = self.visual_model_path(backend_name)
        return bool(path and path.exists())

    def fusion_model_exists(self) -> bool:
        return bool(settings.fusion_model_path and Path(settings.fusion_model_path).exists())

    def visual_metadata_exists(self, backend_name: str = "efficientnet_b0_optional") -> bool:
        path = self.visual_metadata_path_for(backend_name)
        return bool(path and path.exists())

    def fusion_metadata_exists(self) -> bool:
        return bool(settings.fusion_metadata_path and Path(settings.fusion_metadata_path).exists())

    def visual_backend_ready(self, backend_name: str = "efficientnet_b0_optional") -> bool:
        return self.visual_model_exists(backend_name) and self.visual_metadata_exists(backend_name)

    def fusion_backend_ready(self) -> bool:
        return self.fusion_model_exists() and self.fusion_metadata_exists()

    def visual_metadata(self, backend_name: str = "efficientnet_b0_optional") -> dict[str, Any]:
        return _safe_load_json(self.visual_metadata_path_for(backend_name))

    def available_visual_models(self) -> dict[str, dict[str, Any]]:
        models: dict[str, dict[str, Any]] = {}
        for backend_name, spec in VISUAL_MODEL_SPECS.items():
            weights_path = self.visual_model_path(backend_name)
            metadata_path = self.visual_metadata_path_for(backend_name)
            weights_present = bool(weights_path and weights_path.exists())
            metadata_present = bool(metadata_path and metadata_path.exists())
            status = "ready"
            if not weights_present:
                status = "weights_missing"
            elif not metadata_present:
                status = "metadata_missing"
            models[backend_name] = {
                "arch": spec["arch"],
                "model_name": spec["model_name"],
                "weights_present": weights_present,
                "metadata_present": metadata_present,
                "weights_path": str(weights_path) if weights_path else "",
                "metadata_path": str(metadata_path) if metadata_path else "",
                "effective_status": status,
            }
        return models

    def fusion_metadata(self) -> dict[str, Any]:
        return _safe_load_json(settings.fusion_metadata_path)

    def resolve_threshold_policy(self) -> dict[str, Any]:
        fusion_meta = self.fusion_metadata()
        visual_meta = self.visual_metadata()
        threshold_block = fusion_meta.get("thresholds") or visual_meta.get("thresholds") or {}

        true_threshold = float(threshold_block.get("true_threshold", settings.true_threshold))
        fake_threshold = float(threshold_block.get("fake_threshold", settings.fake_threshold))
        source = "default_config"
        if fusion_meta.get("thresholds"):
            source = "fusion_metadata"
        elif visual_meta.get("thresholds"):
            source = "visual_metadata"

        return {
            "source": source,
            "true_threshold": round(true_threshold, 4),
            "fake_threshold": round(fake_threshold, 4),
            "visual_metadata_path": settings.visual_metadata_path,
            "fusion_metadata_path": settings.fusion_metadata_path,
        }

    def build_runtime_status(self) -> dict[str, Any]:
        try:
            import onnxruntime as ort  # type: ignore

            providers = ort.get_available_providers()
        except Exception:
            providers = []
        insightface_model_dir = settings.insightface_root_dir / "models" / settings.insightface_name
        insightface_model_ready = insightface_model_dir.exists()
        insightface_installed = self.module_installed("insightface")
        detector_status = "ready"
        if not insightface_installed:
            detector_status = "dependency_missing"
        elif not insightface_model_ready and not settings.insightface_allow_download:
            detector_status = "assets_missing"
        elif not insightface_model_ready and settings.insightface_allow_download:
            detector_status = "download_pending"

        visual_models = self.available_visual_models()
        ready_visual_models = [
            name for name, model_status in visual_models.items()
            if model_status["effective_status"] == "ready"
        ]
        default_visual = (
            "efficientnet_b0_optional"
            if "efficientnet_b0_optional" in ready_visual_models
            else (ready_visual_models[0] if ready_visual_models else "none")
        )
        fusion_model_present = self.fusion_model_exists()
        fusion_metadata_present = self.fusion_metadata_exists()
        local_qwen_cached = bool(settings.semantic_model_path and Path(settings.semantic_model_path).exists())

        visual_weights_present = self.visual_model_exists()
        visual_metadata_present = self.visual_metadata_exists()
        visual_status = "ready" if ready_visual_models else "weights_missing"
        if not ready_visual_models and visual_weights_present and not visual_metadata_present:
            visual_status = "metadata_missing"
        elif not ready_visual_models and not visual_weights_present:
            visual_status = "weights_missing"

        fusion_status = "ready"
        if not fusion_model_present:
            fusion_status = "model_missing"
        elif not fusion_metadata_present:
            fusion_status = "metadata_missing"

        semantic_status = "ready" if local_qwen_cached else "not_configured"

        return {
            "detector": {
                "preferred_backend": "insightface",
                "preferred_provider": "CUDAExecutionProvider" if "CUDAExecutionProvider" in providers else "CPUExecutionProvider",
                "available_providers": providers,
                "insightface_installed": insightface_installed,
                "model_cache_ready": insightface_model_ready,
                "model_cache_path": str(insightface_model_dir),
                "allow_download": settings.insightface_allow_download,
                "effective_status": detector_status,
            },
            "visual": {
                "weights_present": visual_weights_present,
                "metadata_present": visual_metadata_present,
                "weights_path": settings.vision_model_path,
                "metadata_path": settings.visual_metadata_path,
                "model_dir": str(settings.visual_model_dir),
                "default_backend": default_visual,
                "ready_backends": ready_visual_models,
                "models": visual_models,
                "effective_status": visual_status,
            },
            "fusion": {
                "model_present": fusion_model_present,
                "metadata_present": fusion_metadata_present,
                "model_path": settings.fusion_model_path,
                "metadata_path": settings.fusion_metadata_path,
                "xgboost_installed": self.module_installed("xgboost"),
                "effective_status": fusion_status,
            },
            "semantic": {
                "backend_mode": settings.semantic_backend,
                "model_path": settings.semantic_model_path,
                "local_qwen_cached": local_qwen_cached,
                "transformers_installed": self.module_installed("transformers"),
                "effective_status": semantic_status,
            },
            "llm": {
                "api_key_configured": bool(settings.llm_api_key),
                "strategy_enabled": settings.llm_strategy_enabled,
                "base_url": settings.llm_base_url,
                "text_model": settings.llm_text_model,
                "vision_model": settings.llm_vision_model,
                "effective_status": "ready" if settings.llm_api_key else "not_configured",
            },
        }

    def build_status_table_markdown(self) -> str:
        status = self.build_runtime_status()
        rows = [
            (
                "detector",
                "yes" if status["detector"]["effective_status"] == "ready" else f"no ({status['detector']['effective_status']})",
                status["detector"]["preferred_provider"],
                "fallback to opencv_haar / heuristic_detector",
            ),
            (
                "visual",
                "yes" if status["visual"]["effective_status"] == "ready" else f"no ({status['visual']['effective_status']})",
                settings.vision_model_path,
                "fallback to heuristic_visual",
            ),
            (
                "fusion",
                "yes" if status["fusion"]["effective_status"] == "ready" else f"no ({status['fusion']['effective_status']})",
                settings.fusion_model_path,
                "fallback to heuristic_fusion",
            ),
            (
                "semantic",
                "yes" if status["semantic"]["effective_status"] == "ready" else f"no ({status['semantic']['effective_status']})",
                settings.semantic_model_path or "not configured",
                "fallback to rule_based_semantic",
            ),
        ]
        lines = [
            "| module | ready | source | fallback |",
            "| --- | --- | --- | --- |",
        ]
        for module, ready, source, fallback in rows:
            lines.append(f"| {module} | {ready} | {source} | {fallback} |")
        return "\n".join(lines)

    @staticmethod
    def module_installed(module_name: str) -> bool:
        return importlib.util.find_spec(module_name) is not None


model_registry = ModelRegistry()
