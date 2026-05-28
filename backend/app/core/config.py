from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]

# Load .env file if present
_env_file = BACKEND_ROOT / ".env"
if _env_file.is_file():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            os.environ.setdefault(_key.strip(), _val.strip())
REPO_ROOT = BACKEND_ROOT.parent
DEFAULT_RUNTIME_DIR = Path(
    os.getenv(
        "AGENT_DEEPFAKE_RUNTIME_DIR",
        str(Path(os.getenv("LOCALAPPDATA", str(BACKEND_ROOT / "storage"))) / "AgentDeepfakeFaceDet"),
    )
)


@dataclass(slots=True)
class Settings:
    app_name: str = "基于智能体的 Deepfake 人脸图像检测系统"
    api_prefix: str = "/api"
    backend_root: Path = BACKEND_ROOT
    repo_root: Path = REPO_ROOT
    storage_dir: Path = field(default_factory=lambda: BACKEND_ROOT / "storage")
    runtime_dir: Path = field(default_factory=lambda: DEFAULT_RUNTIME_DIR)
    models_dir: Path = field(default_factory=lambda: BACKEND_ROOT / "storage" / "models")
    uploads_dir: Path = field(default_factory=lambda: DEFAULT_RUNTIME_DIR / "uploads")
    artifacts_dir: Path = field(default_factory=lambda: DEFAULT_RUNTIME_DIR / "artifacts")
    gallery_dir: Path = field(default_factory=lambda: BACKEND_ROOT / "storage" / "gallery")
    insightface_root_dir: Path = field(default_factory=lambda: BACKEND_ROOT / "storage" / "insightface")
    demo_dir: Path = field(default_factory=lambda: BACKEND_ROOT / "storage" / "demo")
    demo_samples_dir: Path = field(default_factory=lambda: BACKEND_ROOT / "storage" / "demo" / "samples")
    demo_reports_dir: Path = field(default_factory=lambda: BACKEND_ROOT / "storage" / "demo" / "reports")
    database_path: Path = field(default_factory=lambda: DEFAULT_RUNTIME_DIR / "app.db")
    frontend_dist_dir: Path = field(default_factory=lambda: REPO_ROOT / "frontend" / "dist")
    max_upload_size_mb: int = 10
    max_faces: int = 5
    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
    true_threshold: float = 0.30
    fake_threshold: float = 0.70
    low_quality_threshold: float = 0.35
    semantic_timeout_seconds: int = 20
    task_worker_count: int = int(os.getenv("TASK_WORKER_COUNT", "1"))
    cors_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    )
    semantic_backend: str = os.getenv("SEMANTIC_BACKEND", "auto")
    semantic_model_name: str = os.getenv("SEMANTIC_MODEL_NAME", "Qwen/Qwen3-VL-2B-Instruct")
    semantic_model_path: str | None = os.getenv("SEMANTIC_MODEL_PATH")
    visual_model_dir: Path = field(default_factory=lambda: Path(os.getenv(
        "VISUAL_MODEL_DIR",
        str(BACKEND_ROOT / "storage" / "models" / "visual"),
    )))
    visual_default_backend: str = os.getenv("VISUAL_DEFAULT_BACKEND", "auto")
    visual_gray_low: float = float(os.getenv("VISUAL_GRAY_LOW", "0.45"))
    visual_gray_high: float = float(os.getenv("VISUAL_GRAY_HIGH", "0.60"))
    visual_disagreement_threshold: float = float(os.getenv("VISUAL_DISAGREEMENT_THRESHOLD", "0.25"))
    vision_model_path: str | None = os.getenv(
        "VISION_MODEL_PATH",
        str(BACKEND_ROOT / "storage" / "models" / "efficientnet_b0_deepfake.pt"),
    )
    fusion_model_path: str | None = os.getenv(
        "FUSION_MODEL_PATH",
        str(BACKEND_ROOT / "storage" / "models" / "xgboost_fusion.json"),
    )
    visual_metadata_path: str | None = os.getenv(
        "VISUAL_METADATA_PATH",
        str(BACKEND_ROOT / "storage" / "models" / "efficientnet_b0_deepfake.meta.json"),
    )
    fusion_metadata_path: str | None = os.getenv(
        "FUSION_METADATA_PATH",
        str(BACKEND_ROOT / "storage" / "models" / "xgboost_fusion.meta.json"),
    )
    insightface_name: str = os.getenv("INSIGHTFACE_MODEL_NAME", "buffalo_l")
    insightface_allow_download: bool = os.getenv("INSIGHTFACE_ALLOW_DOWNLOAD", "0") == "1"
    semantic_force_offline: bool = os.getenv("SEMANTIC_FORCE_OFFLINE", "1") == "1"
    llm_api_key: str | None = os.getenv("DASHSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    llm_text_model: str = os.getenv("LLM_TEXT_MODEL", "qwen-plus")
    llm_vision_model: str = os.getenv("LLM_VISION_MODEL", "qwen-vl-max")
    llm_strategy_enabled: bool = os.getenv("LLM_STRATEGY_ENABLED", "1") == "1"
    llm_timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    demo_manifest_path: str = os.getenv(
        "DEMO_MANIFEST_PATH",
        str(BACKEND_ROOT / "storage" / "demo" / "demo_samples_manifest.json"),
    )
    integration_status_path: str = os.getenv(
        "INTEGRATION_STATUS_PATH",
        str(BACKEND_ROOT / "storage" / "demo" / "reports" / "integration_status.json"),
    )

    def ensure_directories(self) -> None:
        for directory in (
            self.storage_dir,
            self.runtime_dir,
            self.models_dir,
            self.visual_model_dir,
            self.uploads_dir,
            self.artifacts_dir,
            self.gallery_dir,
            self.insightface_root_dir,
            self.demo_dir,
            self.demo_samples_dir,
            self.demo_reports_dir,
            self.database_path.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
