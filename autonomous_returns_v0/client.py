"""
Typed EnvClient implementation for AutonomousReturns-v0.

Designed for OpenEnv interoperability with:
- Strongly typed action/observation/state models
- Async-first API with optional sync wrapper via `.sync()`
- Compatibility with local servers, Docker runtime, and remote deployments
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict

from .models import DispositionAction, ReturnsAction, ReturnsObservation, ReturnsState

_openenv_env_client = import_module("openenv.core.env_client")
EnvClient = getattr(_openenv_env_client, "EnvClient")
StepResult = getattr(_openenv_env_client, "StepResult")


class AutonomousReturnsClient(EnvClient):
    """
    Typed client for AutonomousReturns-v0.

    Example (async):
        async with AutonomousReturnsClient(base_url="http://localhost:8000") as env:
            result = await env.reset(seed=42)
            result = await env.step(ReturnsAction(action=DispositionAction.RESELL_DISCOUNT_30))

    Example (sync wrapper):
        with AutonomousReturnsClient(base_url="http://localhost:8000").sync() as env:
            result = env.reset(seed=42)
            result = env.step(ReturnsAction(action=DispositionAction.WAIT))
    """

    def __init__(self, base_url: str = "http://localhost:8000", **kwargs: Any):
        super().__init__(base_url=base_url, **kwargs)

    def _step_payload(self, action: ReturnsAction) -> Dict[str, Any]:
        """
        Convert typed action model to transport payload.
        """
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ReturnsObservation]:
        """
        Parse server response into typed observation + step metadata.
        """
        observation = ReturnsObservation(**payload["observation"])
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> ReturnsState:
        """
        Parse state response into typed ReturnsState.
        """
        return ReturnsState(**payload)

    @staticmethod
    def action(action: DispositionAction) -> ReturnsAction:
        """
        Convenience helper to build a typed action object.
        """
        return ReturnsAction(action=action)
