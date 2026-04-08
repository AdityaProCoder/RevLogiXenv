from __future__ import annotations

import pytest

from autonomous_returns_v0 import (
    AutonomousReturnsEnv,
    DispositionAction,
    Grader,
    HeuristicPolicy,
    ReturnsAction,
)


def _agent_fn(obs):
    return HeuristicPolicy()(obs)


def test_grader_is_bounded_and_deterministic() -> None:
    grader_1 = Grader("hard")
    grader_2 = Grader("hard")

    result_1 = grader_1.grade(_agent_fn, seed=2025)
    result_2 = grader_2.grade(_agent_fn, seed=2025)

    assert result_1 == result_2
    assert 0.0 <= result_1["margin_score"] <= 1.0
    assert 0.0 <= result_1["fraud_metrics"]["f1"] <= 1.0
    assert 0.0 <= result_1["final_score"] <= 1.0


def test_score_completed_episode_requires_done() -> None:
    env = AutonomousReturnsEnv("medium")
    env.reset(seed=42)

    grader = Grader("medium")
    with pytest.raises(RuntimeError, match="incomplete episode"):
        grader.score_completed_episode(env=env)


def test_score_completed_episode_after_rollout() -> None:
    env = AutonomousReturnsEnv("medium")
    obs = env.reset(seed=42)
    while not obs.done:
        if obs.current_item is None:
            action = ReturnsAction(action=DispositionAction.WAIT)
        else:
            action = ReturnsAction(action=DispositionAction.RESELL_DISCOUNT_30)
        obs = env.step(action)

    grader = Grader("medium")
    result = grader.score_completed_episode(env=env)
    assert result["task"] == "medium"
    assert 0.0 <= result["final_score"] <= 1.0
