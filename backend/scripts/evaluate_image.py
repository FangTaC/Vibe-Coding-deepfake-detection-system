from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.agents.decision_agent import DecisionAgent  # noqa: E402
from app.agents.feature_extraction_agent import FeatureExtractionAgent  # noqa: E402


def main(image_path: str) -> None:
    path = Path(image_path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    task_id = uuid.uuid4().hex
    feature_agent = FeatureExtractionAgent()
    decision_agent = DecisionAgent()
    features = feature_agent.run(task_id, str(path))
    result = decision_agent.run(task_id, str(path), features)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python backend/scripts/evaluate_image.py <image_path>")
        raise SystemExit(1)
    main(sys.argv[1])

