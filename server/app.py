"""FastAPI/OpenEnv application entrypoint for RevLogiXenv."""

from __future__ import annotations

import os
from importlib import import_module
from typing import Any

from RevLogiXenv_v0 import AutonomousReturnsEnv, ReturnsAction, ReturnsObservation

_openenv_env_server = import_module("openenv.core.env_server")
create_fastapi_app = getattr(_openenv_env_server, "create_fastapi_app")


def _env_factory() -> AutonomousReturnsEnv:
    task = os.getenv("REVLOGIXENV_TASK", os.getenv("AUTONOMOUS_RETURNS_TASK", "easy")).strip().lower() or "easy"
    return AutonomousReturnsEnv(task=task)


def build_app() -> Any:
    """Build the OpenEnv-compatible API-only app."""
    return create_fastapi_app(
        env=_env_factory,
        action_cls=ReturnsAction,
        observation_cls=ReturnsObservation,
    )


app = build_app()


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    run_port = int(os.getenv("PORT", str(port)))
    uvicorn = import_module("uvicorn")
    uvicorn.run(build_app(), host=host, port=run_port)


__all__ = ["app", "build_app", "main"]

if __name__ == "__main__":
    main()
