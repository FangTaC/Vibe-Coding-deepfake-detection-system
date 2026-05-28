from __future__ import annotations

import json
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.model_registry import ModelRegistry  # noqa: E402
from app.core.config import settings  # noqa: E402


def test_threshold_policy_uses_defaults_when_metadata_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "visual_metadata_path", str(tmp_path / "missing_visual.json"))
    monkeypatch.setattr(settings, "fusion_metadata_path", str(tmp_path / "missing_fusion.json"))
    registry = ModelRegistry()
    policy = registry.resolve_threshold_policy()
    assert policy["source"] == "default_config"
    assert policy["true_threshold"] == settings.true_threshold
    assert policy["fake_threshold"] == settings.fake_threshold


def test_threshold_policy_prefers_fusion_metadata(tmp_path, monkeypatch) -> None:
    visual_meta = tmp_path / "visual.json"
    fusion_meta = tmp_path / "fusion.json"
    visual_meta.write_text(
        json.dumps({"thresholds": {"true_threshold": 0.22, "fake_threshold": 0.81}}, ensure_ascii=False),
        encoding="utf-8",
    )
    fusion_meta.write_text(
        json.dumps({"thresholds": {"true_threshold": 0.18, "fake_threshold": 0.77}}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "visual_metadata_path", str(visual_meta))
    monkeypatch.setattr(settings, "fusion_metadata_path", str(fusion_meta))
    registry = ModelRegistry()
    policy = registry.resolve_threshold_policy()
    assert policy["source"] == "fusion_metadata"
    assert policy["true_threshold"] == 0.18
    assert policy["fake_threshold"] == 0.77


def test_runtime_status_marks_visual_and_fusion_as_metadata_missing(tmp_path, monkeypatch) -> None:
    visual_weights = tmp_path / "efficientnet_b0_deepfake.pt"
    fusion_model = tmp_path / "xgboost_fusion.json"
    visual_weights.write_text("weights", encoding="utf-8")
    fusion_model.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(settings, "vision_model_path", str(visual_weights))
    monkeypatch.setattr(settings, "visual_metadata_path", str(tmp_path / "missing_visual.meta.json"))
    monkeypatch.setattr(settings, "fusion_model_path", str(fusion_model))
    monkeypatch.setattr(settings, "fusion_metadata_path", str(tmp_path / "missing_fusion.meta.json"))

    registry = ModelRegistry()
    status = registry.build_runtime_status()
    assert status["visual"]["effective_status"] == "metadata_missing"
    assert status["fusion"]["effective_status"] == "metadata_missing"


def test_runtime_status_marks_visual_and_fusion_as_ready_when_model_and_metadata_exist(tmp_path, monkeypatch) -> None:
    visual_weights = tmp_path / "efficientnet_b0_deepfake.pt"
    visual_meta = tmp_path / "efficientnet_b0_deepfake.meta.json"
    fusion_model = tmp_path / "xgboost_fusion.json"
    fusion_meta = tmp_path / "xgboost_fusion.meta.json"
    visual_weights.write_text("weights", encoding="utf-8")
    visual_meta.write_text("{}", encoding="utf-8")
    fusion_model.write_text("{}", encoding="utf-8")
    fusion_meta.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(settings, "vision_model_path", str(visual_weights))
    monkeypatch.setattr(settings, "visual_metadata_path", str(visual_meta))
    monkeypatch.setattr(settings, "fusion_model_path", str(fusion_model))
    monkeypatch.setattr(settings, "fusion_metadata_path", str(fusion_meta))

    registry = ModelRegistry()
    status = registry.build_runtime_status()
    assert status["visual"]["effective_status"] == "ready"
    assert status["fusion"]["effective_status"] == "ready"
