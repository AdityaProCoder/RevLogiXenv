"""FastAPI/OpenEnv application entrypoint for AutonomousReturns-v0."""

from __future__ import annotations

import os
from importlib import import_module, util
from typing import Any

from autonomous_returns_v0 import AutonomousReturnsEnv, ReturnsAction, ReturnsObservation

_openenv_env_server = import_module("openenv.core.env_server")
create_app = getattr(_openenv_env_server, "create_app")
create_fastapi_app = getattr(_openenv_env_server, "create_fastapi_app")


def _web_requested() -> bool:
    return os.getenv("ENABLE_WEB_INTERFACE", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _gradio_available() -> bool:
    return util.find_spec("gradio") is not None


def _env_factory() -> AutonomousReturnsEnv:
    task = os.getenv("AUTONOMOUS_RETURNS_TASK", "easy").strip().lower() or "easy"
    return AutonomousReturnsEnv(task=task)


def build_app() -> Any:
    """
    Build the OpenEnv-compatible app.

    If web UI is requested but Gradio is not installed, gracefully fall back to API-only.
    """
    if _web_requested() and _gradio_available():
        from server.gradio_builder import gradio_builder

        return create_app(
            env=_env_factory,
            action_cls=ReturnsAction,
            observation_cls=ReturnsObservation,
            env_name="autonomous_returns_v0",
            gradio_builder=gradio_builder,
        )

    if _web_requested() and not _gradio_available():
        print(
            "ENABLE_WEB_INTERFACE=true but gradio is not installed; serving API-only "
            "(install optional dependency with `.[web]`)."
        )

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
