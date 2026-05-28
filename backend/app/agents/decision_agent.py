from __future__ import annotations

from datetime import datetime, timezone
from statistics import mean
from typing import Any

from app.core.config import settings
from app.schemas.task import (
    DetectionResult,
    EvidenceEntry,
    FaceResult,
    ReviewLabel,
    SemanticFlag,
    StrategyDecision,
    StrategyName,
    TaskProfile,
)
from app.services.backends import FusionBackendManager, QwenStrategyBackend, SemanticBackendManager
from app.services.model_registry import model_registry
from app.utils.json_validation import validate_payload


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class DecisionAgent:
    name = "decision_agent"

    def __init__(self) -> None:
        self.fusion_backend = FusionBackendManager()
        self.semantic_backend = SemanticBackendManager()
        self.strategy_backend = QwenStrategyBackend()

    def run(
        self,
        task_id: str,
        image_path: str,
        features: dict[str, Any],
        routing: dict[str, Any] | None = None,
    ) -> DetectionResult:
        faces: list[FaceResult] = features["faces"]
        evidence: list[EvidenceEntry] = list(features["evidence"])
        artifacts = features["artifacts"]
        input_summary = features["input_summary"]
        model_versions = dict(features["model_versions"])
        threshold_policy = model_registry.resolve_threshold_policy()
        routing = routing or features.get("routing") or {}
        requested_fusion = routing.get("fusion_backend", "")
        requested_semantic = routing.get("semantic_backend", "")

        if not faces:
            return self._build_no_face_result(
                task_id=task_id,
                evidence=evidence,
                artifacts=artifacts,
                input_summary=input_summary,
                model_versions=model_versions,
                threshold_policy=threshold_policy,
            )

        task_profile = self._build_task_profile(faces, threshold_policy)
        strategy_decision, strategy_backend_used, strategy_error = self._select_strategy(task_profile, threshold_policy)
        strategy_detail: dict[str, Any] = {
            "task_profile": task_profile.model_dump(),
            "selected_strategy": strategy_decision.selected_strategy.value,
            "reason_codes": strategy_decision.reason_codes,
            "threshold_policy": threshold_policy,
            "strategy_backend": strategy_backend_used,
            "algorithm_routing": routing,
        }
        if strategy_decision.thinking_chain:
            strategy_detail["thinking_chain"] = strategy_decision.thinking_chain
        if strategy_error:
            strategy_detail["llm_error"] = strategy_error
        evidence.append(
            EvidenceEntry(
                step="strategy_selection",
                actor=self.name,
                timestamp=now_utc(),
                summary=(
                    f"[{strategy_backend_used}] 已根据任务画像选定策略：{strategy_decision.selected_strategy.value}。"
                    + ("（含 LLM 思维链）" if strategy_decision.thinking_chain else "")
                ),
                details=strategy_detail,
            )
        )

        semantic_flags: list[SemanticFlag] = []
        semantic_summary = "本次任务未触发语义分析。"
        semantic_score = 0.0
        semantic_backend_name = "disabled"
        if strategy_decision.semantic_required:
            semantic_context = task_profile.model_dump()
            semantic_payload, semantic_backend_name = self.semantic_backend.analyze_with_backend(
                requested_semantic,
                image_path,
                semantic_context,
            )
            semantic_flags = [SemanticFlag.model_validate(item) for item in semantic_payload.get("semantic_flags", [])]
            semantic_summary = semantic_payload.get("summary", "语义模块已完成辅助分析。")
            semantic_score = float(semantic_payload.get("semantic_score", 0.0))
            semantic_detail: dict[str, Any] = {
                "semantic_backend": semantic_backend_name,
                "semantic_backend_requested": requested_semantic or "default_chain",
                "semantic_flags": [flag.model_dump() for flag in semantic_flags],
                "semantic_score": round(semantic_score, 4),
            }
            if semantic_payload.get("thinking_chain"):
                semantic_detail["thinking_chain"] = semantic_payload["thinking_chain"]
            evidence.append(
                EvidenceEntry(
                    step="semantic_analysis",
                    actor=self.name,
                    timestamp=now_utc(),
                    summary=(
                        f"[{semantic_backend_name}] 已完成语义辅助判断。"
                        + ("（含 LLM 视觉推理链）" if semantic_payload.get("thinking_chain") else "")
                    ),
                    details=semantic_detail,
                )
            )

        fusion_score, fusion_backend_name = self._fuse_scores(faces, semantic_score, requested_fusion)
        model_versions.update(self.fusion_backend.describe(fusion_backend_name))
        model_versions["semantic"] = semantic_backend_name if strategy_decision.semantic_required else "disabled"
        model_versions["fusion_requested"] = requested_fusion or "default_chain"
        model_versions["semantic_requested"] = requested_semantic or "default_chain"
        model_versions["threshold_source"] = str(threshold_policy["source"])
        model_versions["true_threshold"] = str(threshold_policy["true_threshold"])
        model_versions["fake_threshold"] = str(threshold_policy["fake_threshold"])

        label, review_reason = self._finalize_label(
            faces,
            task_profile,
            semantic_flags,
            fusion_score,
            strategy_decision,
            threshold_policy,
        )
        confidence = self._compute_confidence(fusion_score, label)
        evidence.append(
            EvidenceEntry(
                step="final_decision",
                actor=self.name,
                timestamp=now_utc(),
                summary="已生成最终检测结论与证据链。",
                details={
                    "fusion_score": round(fusion_score, 4),
                    "confidence": round(confidence, 4),
                    "label": label.value,
                    "review_reason": review_reason,
                    "fusion_backend": fusion_backend_name,
                    "fusion_backend_requested": requested_fusion or "default_chain",
                    "threshold_policy": threshold_policy,
                    "gray_zone_hit": str(
                        threshold_policy["true_threshold"] < fusion_score < threshold_policy["fake_threshold"]
                    ).lower(),
                },
            )
        )

        visual_summary = self._build_visual_summary(faces, fusion_score)
        return DetectionResult(
            task_id=task_id,
            label=label,
            image_label=label,
            confidence=round(confidence, 4),
            fusion_score=round(fusion_score, 4),
            review_required=label == ReviewLabel.review_required,
            review_reason=review_reason,
            visual_summary=visual_summary,
            semantic_summary=semantic_summary,
            current_agent=self.name,
            faces=faces,
            semantic_flags=semantic_flags,
            evidence_chain=evidence,
            artifacts=artifacts,
            agent_trace=[
                {"agent": "feature_agent", "status": "completed"},
                {"agent": self.name, "status": "completed", "strategy": strategy_decision.selected_strategy.value},
            ],
            task_profile=task_profile,
            strategy_decision=strategy_decision,
            input_summary=input_summary,
            model_versions=model_versions,
        )

    def _build_no_face_result(
        self,
        *,
        task_id: str,
        evidence: list[EvidenceEntry],
        artifacts: list[Any],
        input_summary: dict[str, Any],
        model_versions: dict[str, str],
        threshold_policy: dict[str, Any],
    ) -> DetectionResult:
        task_profile = TaskProfile(
            face_count=0,
            max_face_score=0.0,
            mean_face_score=0.0,
            quality_score=0.0,
            public_figure_candidates=[],
            need_semantic_check=False,
            conflict_level=1.0,
        )
        strategy_decision = StrategyDecision(
            selected_strategy=StrategyName.review_only,
            reason_codes=["NO_FACE_DETECTED", "OUT_OF_SCOPE_IMAGE"],
            semantic_required=False,
            expected_risk_level="high",
        )
        evidence.append(
            EvidenceEntry(
                step="strategy_selection",
                actor=self.name,
                timestamp=now_utc(),
                summary="未检测到有效人脸，已切换到 review_only 受限策略。",
                details={
                    "task_profile": task_profile.model_dump(),
                    "selected_strategy": strategy_decision.selected_strategy.value,
                    "reason_codes": strategy_decision.reason_codes,
                    "threshold_policy": threshold_policy,
                },
            )
        )
        evidence.append(
            EvidenceEntry(
                step="final_decision",
                actor=self.name,
                timestamp=now_utc(),
                summary="输入超出系统适用范围，最终结论为需人工复核。",
                details={
                    "fusion_score": 0.0,
                    "confidence": 0.18,
                    "label": ReviewLabel.review_required.value,
                    "review_reason": "未检测到人脸，超出系统适用范围。",
                    "threshold_policy": threshold_policy,
                    "gray_zone_hit": "false",
                },
            )
        )
        model_versions["threshold_source"] = str(threshold_policy["source"])
        return DetectionResult(
            task_id=task_id,
            label=ReviewLabel.review_required,
            image_label=ReviewLabel.review_required,
            confidence=0.18,
            fusion_score=0.0,
            review_required=True,
            review_reason="未检测到人脸，超出系统适用范围。",
            visual_summary="未检测到可用于 Deepfake 检测的人脸。",
            semantic_summary="未触发语义分析。",
            current_agent=self.name,
            faces=[],
            semantic_flags=[],
            evidence_chain=evidence,
            artifacts=artifacts,
            agent_trace=[
                {"agent": "feature_agent", "status": "completed"},
                {"agent": self.name, "status": "completed", "strategy": strategy_decision.selected_strategy.value},
            ],
            task_profile=task_profile,
            strategy_decision=strategy_decision,
            input_summary=input_summary,
            model_versions=model_versions,
        )

    def _build_task_profile(self, faces: list[FaceResult], threshold_policy: dict[str, Any]) -> TaskProfile:
        max_face_score = max(face.deepfake_score for face in faces)
        mean_face_score = mean(face.deepfake_score for face in faces)
        quality_score = mean(face.quality_score for face in faces)
        public_figures: list[str] = []
        for face in faces:
            public_figures.extend(face.public_figure_candidates)
        public_figures = sorted(set(public_figures))
        conflict_level = max(
            abs(face.deepfake_score - face.dct_score) + (1.0 - face.augmentation_consistency_score) * 0.5
            for face in faces
        )
        need_semantic_check = (
            threshold_policy["true_threshold"] < max_face_score < threshold_policy["fake_threshold"]
            or len(faces) > 1
            or bool(public_figures)
            or conflict_level >= 0.35
        )
        return TaskProfile(
            face_count=len(faces),
            max_face_score=round(max_face_score, 4),
            mean_face_score=round(mean_face_score, 4),
            quality_score=round(quality_score, 4),
            public_figure_candidates=public_figures,
            need_semantic_check=need_semantic_check,
            conflict_level=round(min(conflict_level, 1.0), 4),
        )

    def _select_strategy(
        self, task_profile: TaskProfile, threshold_policy: dict[str, Any]
    ) -> tuple[StrategyDecision, str, str | None]:
        """优先使用 Qwen LLM 选择策略（输出思维决策链路），失败时回退到规则逻辑。"""
        if self.strategy_backend.is_available():
            try:
                payload = self.strategy_backend.select_strategy(task_profile.model_dump(), threshold_policy)
                decision = validate_payload(StrategyDecision, payload, retries=1)
                return decision, self.strategy_backend.backend_name, None
            except Exception as exc:
                fallback_payload = self._rule_based_strategy(task_profile, threshold_policy)
                fallback_decision = validate_payload(StrategyDecision, fallback_payload, retries=1)
                return fallback_decision, "rule_based", f"{type(exc).__name__}: {exc}"
        # 规则回退
        payload = self._rule_based_strategy(task_profile, threshold_policy)
        decision = validate_payload(StrategyDecision, payload, retries=1)
        return decision, "rule_based", "strategy backend unavailable"

    def _rule_based_strategy(self, task_profile: TaskProfile, threshold_policy: dict[str, Any]) -> dict[str, Any]:
        if task_profile.quality_score < settings.low_quality_threshold:
            return {
                "selected_strategy": StrategyName.review_only.value,
                "reason_codes": ["LOW_QUALITY_INPUT"],
                "semantic_required": False,
                "expected_risk_level": "high",
            }
        if task_profile.need_semantic_check:
            reason_codes: list[str] = []
            if threshold_policy["true_threshold"] < task_profile.max_face_score < threshold_policy["fake_threshold"]:
                reason_codes.append("GRAY_ZONE_SCORE")
            if task_profile.face_count > 1:
                reason_codes.append("MULTI_FACE_SCENE")
            if task_profile.public_figure_candidates:
                reason_codes.append("PUBLIC_FIGURE_CANDIDATE")
            if task_profile.conflict_level >= 0.35:
                reason_codes.append("VISUAL_CONFLICT")
            return {
                "selected_strategy": StrategyName.visual_plus_semantic.value,
                "reason_codes": reason_codes or ["NEED_SEMANTIC_CHECK"],
                "semantic_required": True,
                "expected_risk_level": "high" if task_profile.public_figure_candidates else "medium",
            }
        return {
            "selected_strategy": StrategyName.visual_only.value,
            "reason_codes": ["DEFAULT_VISUAL_PATH"],
            "semantic_required": False,
            "expected_risk_level": "medium",
        }

    def _fuse_scores(
        self,
        faces: list[FaceResult],
        semantic_score: float,
        requested_fusion_backend: str = "",
    ) -> tuple[float, str]:
        features = [
            max(face.deepfake_score for face in faces),
            mean(face.deepfake_score for face in faces),
            mean(face.dct_score for face in faces),
            mean(face.quality_score for face in faces),
            mean(face.augmentation_consistency_score for face in faces),
            mean(face.compression_score for face in faces),
            float(len(faces)),
            semantic_score,
        ]
        return self.fusion_backend.predict_with_backend(requested_fusion_backend, features)

    def _finalize_label(
        self,
        faces: list[FaceResult],
        task_profile: TaskProfile,
        semantic_flags: list[SemanticFlag],
        fusion_score: float,
        strategy_decision: StrategyDecision,
        threshold_policy: dict[str, Any],
    ) -> tuple[ReviewLabel, str | None]:
        true_threshold = float(threshold_policy["true_threshold"])
        fake_threshold = float(threshold_policy["fake_threshold"])
        if strategy_decision.selected_strategy == StrategyName.review_only:
            return ReviewLabel.review_required, "图像质量过低或输入不满足系统适用范围。"
        if task_profile.quality_score < settings.low_quality_threshold:
            return ReviewLabel.review_required, "图像质量过低，系统主动降级为人工复核。"
        if any(flag.code == "SEMANTIC_TIMEOUT" for flag in semantic_flags):
            return ReviewLabel.review_required, "语义模型超时，系统已降级为人工复核。"
        if semantic_flags and task_profile.conflict_level >= 0.45 and true_threshold < fusion_score < 0.85:
            return ReviewLabel.review_required, "视觉与语义证据存在冲突，建议人工复核。"
        if fusion_score <= true_threshold:
            return ReviewLabel.likely_real, None
        if fusion_score >= fake_threshold:
            return ReviewLabel.likely_fake, None
        return ReviewLabel.review_required, "检测分数落入灰区，需要人工复核。"

    def _compute_confidence(self, fusion_score: float, label: ReviewLabel) -> float:
        confidence = abs(fusion_score - 0.5) * 2.0
        if label == ReviewLabel.review_required:
            confidence *= 0.65
        return float(max(min(confidence, 0.99), 0.05))

    def _build_visual_summary(self, faces: list[FaceResult], fusion_score: float) -> str:
        top_face = max(faces, key=lambda face: face.deepfake_score)
        return (
            f"系统共分析 {len(faces)} 张人脸；最高风险人脸为 {top_face.face_id}，"
            f"视觉伪造分数 {top_face.deepfake_score:.2f}，图像级融合分数 {fusion_score:.2f}。"
        )
