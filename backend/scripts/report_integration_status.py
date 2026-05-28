from __future__ import annotations

import json
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import settings  # noqa: E402
from app.services.model_registry import model_registry  # noqa: E402


def main() -> None:
    settings.ensure_directories()
    status = model_registry.build_runtime_status()
    status_path = Path(settings.integration_status_path)
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown_path = status_path.with_suffix(".md")
    markdown = [
        "# 当前真实模型接入状态",
        "",
        model_registry.build_status_table_markdown(),
        "",
        "```json",
        json.dumps(status, ensure_ascii=False, indent=2),
        "```",
    ]
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")
    print(f"Saved runtime status to {status_path}")


if __name__ == "__main__":
    main()
