"""
Oracle for AutonomousReturns-v0.

This implementation is deterministic and robust:
- Uses immutable episode snapshots captured at environment reset time
- Never reads mutable/depleted internal queues during or after rollout
- Provides bounded margin scoring in [0.01, 0.99]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .models import Condition, DispositionAction


@dataclass(frozen=True)
class OracleRecord:
    item_id: int
    hidden_condition: Condition
    price: float


class Oracle:
    MIN_SCORE = 0.01
    MAX_SCORE = 0.99

    def __init__(self, env):
        self._env = env
        self._snapshot: list[OracleRecord] = self._build_snapshot()
        self._optimal_actions: list[DispositionAction] = []
        self._optimal_rewards: list[float] = []
        self._precompute()

    def _build_snapshot(self) -> list[OracleRecord]:
        raw = getattr(self._env, "_episode_snapshot", None)
        if not raw:
            raise ValueError(
                "Environment snapshot is empty. Reset the environment before creating Oracle."
            )

        snapshot: list[OracleRecord] = []
        for record in raw:
            snapshot.append(
                OracleRecord(
                    item_id=int(record.item_id),
                    hidden_condition=record.hidden_condition,
                    price=float(record.price),
                )
            )
        return snapshot

    def _iter_candidate_actions(self) -> Iterable[DispositionAction]:
        for action in DispositionAction:
            if action not in (DispositionAction.WAIT, DispositionAction.INSPECT):
                yield action

    def _precompute(self) -> None:
        self._optimal_actions.clear()
        self._optimal_rewards.clear()

        for rec in self._snapshot:
            best_action = DispositionAction.RESELL_DISCOUNT_30
            best_reward = float("-inf")

            for action in self._iter_candidate_actions():
                raw, _components = self._env._compute_raw_reward(
                    action, rec.hidden_condition, rec.price
                )
                if raw > best_reward:
                    best_reward = raw
                    best_action = action

            self._optimal_actions.append(best_action)
            self._optimal_rewards.append(best_reward)

    def get_optimal_action(self, item_index: int) -> DispositionAction:
        return self._optimal_actions[item_index]

    def optimal_action_plan(self) -> Sequence[DispositionAction]:
        return tuple(self._optimal_actions)

    def compute_optimal_profit(self) -> float:
        return float(sum(self._optimal_rewards))

    def compute_margin_score(self, agent_profit: float) -> float:
        """
        Bounded margin score in [0.01, 0.99].

        - 0.99 when agent reaches or exceeds oracle
        - 0.01 when agent_profit <= 0 and oracle > 0
        - smooth linear scaling in-between
        """
        optimal = self.compute_optimal_profit()
        if optimal <= 0:
            return self.MAX_SCORE if agent_profit >= 0 else self.MIN_SCORE

        ratio = agent_profit / optimal
        return max(self.MIN_SCORE, min(self.MAX_SCORE, float(ratio)))
