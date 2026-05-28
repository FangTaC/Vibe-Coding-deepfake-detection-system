from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class TaskStage(str, Enum):
    queued = "queued"
    preprocessing = "preprocessing"
    feature_extraction = "feature_extraction"
    decision = "decision"
    completed = "completed"
    failed = "failed"


class ReviewLabel(str, Enum):
    likely_real = "疑似真实"
    likely_fake = "疑似伪造"
    review_required = "需人工复核"


class StrategyName(str, Enum):
    visual_only = "visual_only"
    visual_plus_semantic = "visual_plus_semantic"
    review_only = "review_only"


class ArtifactRef(BaseModel):
    name: str
    kind: str
    url: str
    description: str | None = None


class SemanticFlag(BaseModel):
    code: str
    label: str
    severity: str = "info"
    confidence: float = 0.0
    evidence: str | None = None


class FaceResult(BaseModel):
    face_id: str
    bbox: list[int]
    landmarks: list[list[float]] = Field(default_factory=list)
    deepfake_score: float
    quality_score: float
    dct_score: float
    blur_score: float
    compression_score: float
    face_size_ratio: float
    augmentation_consistency_score: float
    feature_embedding: list[float] = Field(default_factory=list)
    detector_backend: str
    classifier_backend: str
    visual_model_scores: dict[str, float] = Field(default_factory=dict)
    visual_selection_reason: list[str] = Field(default_factory=list)
    visual_model_disagreement: float | None = None
    public_figure_candidates: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)


class TaskProfile(BaseModel):
    face_count: int
    max_face_score: float
    mean_face_score: float
    quality_score: float
    public_figure_candidates: list[str] = Field(default_factory=list)
    need_semantic_check: bool
    conflict_level: float


class StrategyDecision(BaseModel):
    selected_strategy: StrategyName
    reason_codes: list[str] = Field(default_factory=list)
    semantic_required: bool
    expected_risk_level: str
    thinking_chain: str | None = None


class EvidenceEntry(BaseModel):
    step: str
    actor: str
    timestamp: datetime
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)


class DetectionResult(BaseModel):
    task_id: str
    label: ReviewLabel
    image_label: ReviewLabel
    confidence: float
    fusion_score: float
    review_required: bool
    review_reason: str | None = None
    visual_summary: str
    semantic_summary: str
    current_agent: str
    faces: list[FaceResult] = Field(default_factory=list)
    semantic_flags: list[SemanticFlag] = Field(default_factory=list)
    evidence_chain: list[EvidenceEntry] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    agent_trace: list[dict[str, Any]] = Field(default_factory=list)
    task_profile: TaskProfile
    strategy_decision: StrategyDecision
    input_summary: dict[str, Any] = Field(default_factory=dict)
    model_versions: dict[str, str] = Field(default_factory=dict)


class CreateTaskResponse(BaseModel):
    task_id: str
    status: TaskStatus


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    stage: TaskStage
    progress: float
    current_agent: str | None = None
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None


class TaskSummaryResponse(TaskStatusResponse):
    filename: str
    label: ReviewLabel | None = None
    confidence: float | None = None
