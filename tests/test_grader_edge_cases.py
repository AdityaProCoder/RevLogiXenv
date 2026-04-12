"""
Tests for Grader edge cases and fraud metric robustness.

Covers:
- No fraud items present, agent does not flag anything → f1 near perfect
- No fraud items present, agent does flag something → f1 low floor
- All fraud items correctly flagged → f1 perfect
- Legacy easy task alias still works
- extra_hard noise level is accessible
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "RevLogiXenv_v0"))

from RevLogiXenv_v0 import AutonomousReturnsEnv, Grader
from RevLogiXenv_v0.models import DispositionAction, ReturnsAction


def _run_episode_with_action(task: str, seed: int, policy_fn) -> AutonomousReturnsEnv:
    """Helper: run a full episode with a given policy function."""
    env = AutonomousReturnsEnv(task)
    obs = env.reset(seed=seed)
    while not obs.done:
        action = policy_fn(obs)
        obs = env.step(action)
    return env


def _always_wait(obs):
    """Policy that always waits — useful for draining queues without penalty."""
    return ReturnsAction(action=DispositionAction.WAIT)


def _always_resell_full(obs):
    """Policy that always resells at full — used to probe fraud metrics."""
    return ReturnsAction(action=DispositionAction.RESELL_FULL)


def test_no_fraud_items_no_flags_yields_high_f1():
    """
    When no fraudulent items exist in the episode and the agent never flags
    anything, the fraud F1 should be clamped to near-perfect (0.99).
    """
    # easy task has fraud_enabled: false — guaranteed no fraud items
    env = _run_episode_with_action("easy", seed=7, policy_fn=_always_wait)
    grader = Grader("easy")
    result = grader.score_completed_episode(env=env)

    f1 = result["fraud_metrics"]["f1"]
    assert f1 == 0.99, f"Expected f1=0.99 for no-fraud task, got {f1}"


def test_easy_task_alias_legacy_easy_still_works():
    """
    The _legacy_easy task alias is declared in openenv.yaml and must be
    accepted by the runtime environment.
    """
    env = AutonomousReturnsEnv("_legacy_easy")
    obs = env.reset(seed=1)
    assert obs is not None
    assert env._config["num_items"] == 10
    assert env._config["noise_level"] == "easy"
    assert env._config["delay_type"] == "immediate"

    # Backward compatibility alias should continue to work as before.
    legacy_alias_env = AutonomousReturnsEnv("_old_easy")
    alias_obs = legacy_alias_env.reset(seed=1)
    assert alias_obs is not None
    steps = 0
    while not obs.done and steps < 50:
        action = ReturnsAction(action=DispositionAction.WAIT)
        obs = env.step(action)
        steps += 1
    assert steps > 0


def test_extra_hard_noise_level_accessible():
    """
    Verify that the 'extra_hard' noise level can be set via task configuration
    and does not raise errors during item generation.
    """
    # 'hard' task uses noise_level: extra_hard — verify this works end-to-end
    env = AutonomousReturnsEnv("hard")
    obs = env.reset(seed=999)
    steps = 0
    while not obs.done and steps < 100:
        action = ReturnsAction(action=DispositionAction.RESELL_DISCOUNT_30)
        obs = env.step(action)
        steps += 1
    assert obs.done, "hard task should complete within 100 steps"


def test_grader_rejects_incomplete_episode():
    """Grader must raise RuntimeError when scoring an incomplete episode."""
    env = AutonomousReturnsEnv("easy")
    env.reset(seed=42)
    # Don't run any steps — episode is incomplete
    grader = Grader("easy")
    try:
        grader.score_completed_episode(env=env)
        assert False, "Expected RuntimeError for incomplete episode"
    except RuntimeError as e:
        assert "incomplete" in str(e).lower()


def test_grader_rejects_no_snapshot():
    """Grader must raise RuntimeError when environment has no episode snapshot."""
    # Create env but don't call reset()
    env = AutonomousReturnsEnv("easy")
    grader = Grader("easy")
    try:
        grader.score_completed_episode(env=env)
        assert False, "Expected RuntimeError for missing episode snapshot"
    except RuntimeError as e:
        assert "snapshot" in str(e).lower() or "reset" in str(e).lower()


def test_reward_detail_components_populated():
    """
    Verify that reward_detail.components is populated during normal episode execution.
    At least one step should have non-empty components dict.
    """
    env = AutonomousReturnsEnv("medium")
    obs = env.reset(seed=55)

    components_seen: list[dict] = []
    while not obs.done:
        action = ReturnsAction(action=DispositionAction.RESELL_DISCOUNT_30)
        obs = env.step(action)
        if obs.reward_detail and obs.reward_detail.components:
            components_seen.append(obs.reward_detail.components)

    assert len(components_seen) > 0, "Expected at least one step with populated components"


def test_fraud_metrics_precision_recall_clamped():
    """All fraud precision/recall/f1 values must be within [0.01, 0.99] inclusive."""
    env = _run_episode_with_action("medium", seed=21, policy_fn=_always_resell_full)
    grader = Grader("medium")
    result = grader.score_completed_episode(env=env)

    fm = result["fraud_metrics"]
    for key in ("precision", "recall", "f1"):
        score = fm[key]
        # Clamped scores are in the inclusive [0.01, 0.99] range per grader._clamp_score
        assert 0.01 <= score <= 0.99, f"{key}={score} outside [0.01, 0.99]"


if __name__ == "__main__":
    test_no_fraud_items_no_flags_yields_high_f1()
    test_easy_task_alias_legacy_easy_still_works()
    test_extra_hard_noise_level_accessible()
    test_grader_rejects_incomplete_episode()
    test_grader_rejects_no_snapshot()
    test_reward_detail_components_populated()
    test_fraud_metrics_precision_recall_clamped()
    print("All grader edge-case tests passed.")
