"""
Root server app wrapper for AutonomousReturns-v0.

This file exists to support OpenEnv layouts that expect `server.app:app`
at repository root. It delegates to the package implementation at
`autonomous_returns_v0.server.app`.
"""

import os
from importlib import import_module


def _build_runtime_app():
    module = import_module("autonomous_returns_v0.server.app")
    build_app = getattr(module, "build_app")
    return build_app()


app = _build_runtime_app()


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """
    Root entrypoint expected by OpenEnv local validators and local UI launch.
    """
    os.environ["ENABLE_WEB_INTERFACE"] = "true"
    run_port = int(os.getenv("PORT", str(port)))
    uvicorn = import_module("uvicorn")
    uvicorn.run(_build_runtime_app(), host=host, port=run_port)


__all__ = ["app", "main"]

if __name__ == "__main__":
    main()
