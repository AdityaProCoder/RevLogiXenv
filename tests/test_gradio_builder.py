from __future__ import annotations

import pytest
from openenv.core.env_server.types import EnvironmentMetadata

gr = pytest.importorskip("gradio")

from server.gradio_builder import gradio_builder


class _DummyEpisodeState:
    def __init__(self) -> None:
        self.current_observation = None
        self.action_logs = []


class _DummyWebManager:
    def __init__(self) -> None:
        self.episode_state = _DummyEpisodeState()

    async def reset_environment(self, reset_kwargs=None):
        del reset_kwargs
        return {"observation": {}, "reward": 0.0, "done": False}

    async def step_environment(self, action_data):
        del action_data
        return {"observation": {}, "reward": 0.0, "done": False}

    def get_state(self):
        return {
            "episode_id": "test-episode",
            "step_count": 0,
            "task": "easy",
            "episode_clock": 0,
            "total_ledger": 0.0,
            "item_queue_size": 0,
            "pending_queue_size": 0,
            "resolution_history_size": 0,
            "total_processed": 0,
            "total_flagged_fraud": 0,
            "metadata": {},
        }


def test_gradio_builder_returns_blocks() -> None:
    manager = _DummyWebManager()
    metadata = EnvironmentMetadata(
        name="autonomous_returns_v0",
        description="test",
        version="1.0.0",
    )

    blocks = gradio_builder(
        web_manager=manager,
        action_fields=[],
        metadata=metadata,
        is_chat_env=False,
        title="AutonomousReturns-v0",
        quick_start_md="quick start",
    )

    assert isinstance(blocks, gr.Blocks)
