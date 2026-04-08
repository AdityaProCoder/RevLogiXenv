"""Single-command launcher for local OpenEnv API server."""

from __future__ import annotations

import os
from importlib import import_module

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    os.environ.setdefault("ENV_README_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "README.md")))
    port = int(os.getenv("PORT", "8000"))

    app_module = import_module("server.app")
    app = getattr(app_module, "build_app")()
    print(f"API: http://localhost:{port}/")

    uvicorn = import_module("uvicorn")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
