from __future__ import annotations

import os
from pathlib import Path

import uvicorn


if __name__ == "__main__":
    backend_root = Path(__file__).resolve().parent
    reload_enabled = os.getenv("BACKEND_RELOAD", "0") == "1"
    uvicorn.run(
        "app.main:app",
        app_dir=str(backend_root),
        host="127.0.0.1",
        port=8000,
        reload=reload_enabled,
        reload_dirs=[str(backend_root / "app")] if reload_enabled else None,
        reload_excludes=[
            str(backend_root / "storage" / "*"),
            str(backend_root / "logs" / "*"),
            str(backend_root / "scripts" / "*"),
        ]
        if reload_enabled
        else None,
    )
