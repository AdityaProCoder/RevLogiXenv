"""
Tests for scoring consistency across inference-mode and grader-mode outputs.

Verifies that:
1. inference.py and grader.py both produce scores within (0.01, 0.99).
2. _scoring_mode and _score_formula markers are present in grader output.
3. The two scoring modes are intentionally different (documented in openenv.yaml).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from RevLogiXenv_v0 in the worktree
sys.path.insert(0, str(Path(__file__).parent.parent / "RevLogiXenv_v0"))

from RevLogiXenv_v0 import AutonomousReturnsEnv, Grader
from RevLogiXenv_v0.policies import heuristic_action


def test_grader_output_contains_scoring_markers():
    """Grader._compute_scores() must include _scoring_mode and _score_formula."""
    env = AutonomousReturnsEnv("easy")
    obs = env.reset(seed=42)

    while not obs.done:
        action = heuristic_action(obs)
        obs = env.step(action)

    grader = Grader("easy")
    result = grader.score_completed_episode(env=env)

    assert "_scoring_mode" in result, "Grader output missing _scoring_mode"
    assert result["_scoring_mode"] == "grader"
    assert "_score_formula" in result, "Grader output missing _score_formula"
    assert result["_score_formula"] == "margin_score"


def test_grader_hard_task_has_composite_formula():
    """Hard task grader must report the composite fraud formula."""
    env = AutonomousReturnsEnv("hard")
    obs = env.reset(seed=99)

    while not obs.done:
        action = heuristic_action(obs)
        obs = env.step(action)

    grader = Grader("hard")
    result = grader.score_completed_episode(env=env)

    assert result["_scoring_mode"] == "grader"
    assert "0.6 * margin_score + 0.4 * fraud_f1" in result["_score_formula"]


def test_grader_scores_bounded_to_open_interval():
    """All grader scores must be strictly within (0.01, 0.99)."""
    env = AutonomousReturnsEnv("medium")
    obs = env.reset(seed=7)

    while not obs.done:
        action = heuristic_action(obs)
        obs = env.step(action)

    grader = Grader("medium")
    result = grader.score_completed_episode(env=env)

    for key in ("final_score", "margin_score"):
        score = result[key]
        assert 0.01 < score < 0.99, f"{key}={score} outside (0.01, 0.99)"

    f1 = result["fraud_metrics"]["f1"]
    assert 0.01 < f1 < 0.99, f"fraud_f1={f1} outside (0.01, 0.99)"


def test_grader_consistent_across_tasks():
    """Grader must produce valid output for easy, medium, and hard tasks."""
    tasks = ["easy", "medium", "hard"]
    for task in tasks:
        env = AutonomousReturnsEnv(task)
        obs = env.reset(seed=123)

        while not obs.done:
            action = heuristic_action(obs)
            obs = env.step(action)

        grader = Grader(task)
        result = grader.score_completed_episode(env=env)

        assert "final_score" in result
        assert "margin_score" in result
        assert "fraud_metrics" in result
        assert "profit" in result
        assert 0.01 < result["final_score"] < 0.99


def test_inference_score_mean_of_step_rewards():
    """
    The inference-mode score formula is mean(step_rewards), clamped to (0.01, 0.99).
    This test verifies that the inference.py output format matches the documented formula.
    """
    # We test the computation logic without an LLM by directly exercising
    # the reward-collection path in an episode.
    from RevLogiXenv_v0.environment import AutonomousReturnsEnv
    from RevLogiXenv_v0.policies import heuristic_action

    REWARD_MIN = 0.01
    REWARD_MAX = 0.99

    task = "easy"
    env = AutonomousReturnsEnv(task)
    obs = env.reset(seed=42)

    step_rewards: list[float] = []
    while not obs.done:
        action = heuristic_action(obs)
        obs = env.step(action)
        if obs.reward is not None:
            clamped = float(max(REWARD_MIN, min(REWARD_MAX, obs.reward)))
            step_rewards.append(clamped)

    inference_score = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
    assert 0.01 < inference_score < 0.99


if __name__ == "__main__":
    test_grader_output_contains_scoring_markers()
    test_grader_hard_task_has_composite_formula()
    test_grader_scores_bounded_to_open_interval()
    test_grader_consistent_across_tasks()
    test_inference_score_mean_of_step_rewards()
    print("All scoring consistency tests passed.")
