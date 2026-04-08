from __future__ import annotations

from autonomous_returns_v0 import (
    AutonomousReturnsEnv,
    DispositionAction,
    HeuristicPolicy,
    ReturnsAction,
)


def _run_episode(task: str, seed: int) -> tuple[float, list[float], list[str]]:
    env = AutonomousReturnsEnv(task)
    obs = env.reset(seed=seed)
    policy = HeuristicPolicy()
    rewards: list[float] = []

    while not obs.done:
        action = policy(obs)
        obs = env.step(action)
        if obs.reward_detail is not None:
            rewards.append(obs.reward_detail.value)

    history_actions = [r.action.value for r in env._full_resolution_history]
    return env.state.total_ledger, rewards, history_actions


def test_reward_bounds_across_tasks() -> None:
    policy = HeuristicPolicy()
    for task in ("easy", "medium", "hard"):
        env = AutonomousReturnsEnv(task)
        obs = env.reset(seed=42)
        while not obs.done:
            obs = env.step(policy(obs))
            if obs.reward_detail is not None:
                assert 0.0 <= obs.reward_detail.value <= 1.0


def test_repeatability_for_fixed_seed() -> None:
    result_1 = _run_episode(task="hard", seed=1337)
    result_2 = _run_episode(task="hard", seed=1337)
    assert result_1 == result_2


def test_wait_on_active_item_is_penalized() -> None:
    env = AutonomousReturnsEnv("easy")
    obs = env.reset(seed=42)
    assert obs.current_item is not None

    obs = env.step(ReturnsAction(action=DispositionAction.WAIT))
    assert obs.reward_detail is not None
    assert obs.reward_detail.raw_step_reward < 0.0
    assert "wait_with_active_item" in obs.reward_detail.penalties
    assert any(r.action == DispositionAction.WAIT for r in env._full_resolution_history)


def test_wait_on_tail_has_no_invalid_penalty() -> None:
    env = AutonomousReturnsEnv("medium")
    obs = env.reset(seed=42)

    while obs.current_item is not None and not obs.done:
        obs = env.step(ReturnsAction(action=DispositionAction.RESELL_DISCOUNT_30))

    assert obs.current_item is None
    assert env.pending_queue

    obs = env.step(ReturnsAction(action=DispositionAction.WAIT))
    assert obs.reward_detail is not None
    assert "invalid_action_when_idle" not in obs.reward_detail.penalties
    assert "non_wait_on_tail" not in obs.reward_detail.penalties


def test_step_after_done_is_guarded() -> None:
    env = AutonomousReturnsEnv("easy")
    obs = env.reset(seed=7)
    policy = HeuristicPolicy()
    while not obs.done:
        obs = env.step(policy(obs))

    ledger_before = env.state.total_ledger
    obs_after = env.step(ReturnsAction(action=DispositionAction.RESELL_FULL))

    assert obs_after.done is True
    assert obs_after.reward_detail is not None
    assert obs_after.reward_detail.raw_step_reward == 0.0
    assert "post_done_step" in obs_after.reward_detail.penalties
    assert env.state.total_ledger == ledger_before


def test_reset_task_override_changes_active_task() -> None:
    env = AutonomousReturnsEnv("easy")
    obs = env.reset(seed=11, task="hard")
    assert env.task == "hard"
    assert env.state.task == "hard"
    assert obs.remaining_items == 30
