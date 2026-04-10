"""
Grader for revLogiXenv-v0 tasks.

This grader provides deterministic, bounded scoring in the strict interval
[0.01, 0.99] and robust hard-task handling for fraud metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .environment import AutonomousReturnsEnv
from .models import Condition, DispositionAction, ReturnsAction, ReturnsObservation
from .oracle import Oracle


@dataclass(frozen=True)
class FraudMetrics:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


class Grader:
    """
    Evaluation utility for RevLogiXenv policies.
    
    Computes profit, margins, and fraud metrics (F1) across multiple episodes.
    Final scores are strictly clamped to the (0.01, 0.99) range to comply
    with hackathon validation requirements.

    Scoring:
    - easy:  final_score = margin_score
    - medium: final_score = margin_score
    - hard: final_score = 0.6 * margin_score + 0.4 * fraud_f1
    """
    MIN_SCORE = 0.01
    MAX_SCORE = 0.99
    SCORE_DECIMALS = 3

    def __init__(self, task: str):
        self.task = task
        self.env = AutonomousReturnsEnv(task)
        self.oracle: Oracle | None = None

    def grade(
        self,
        agent_fn: Callable[
            [ReturnsObservation], ReturnsAction | DispositionAction | str
        ],
        seed: int,
    ) -> dict:
        """
        Run one deterministic episode and return bounded scores.
        """
        obs = self.env.reset(seed=seed)
        self.oracle = Oracle(self.env)

        while not obs.done:
            action = agent_fn(obs)
            obs = self.env.step(action)

        self._ensure_episode_completed()
        return self._compute_scores()

    def score_completed_episode(self, env: AutonomousReturnsEnv | None = None) -> dict:
        """
        Public API to score an already-completed episode.

        This is useful for baseline integrations that already rolled out an
        episode and want official grading without re-running agent actions.

        Args:
            env: Optional completed environment instance. If None, uses self.env.

        Returns:
            dict with bounded scoring fields.
        """
        if env is not None:
            self.env = env
            self.task = env.task
        self._ensure_episode_completed()
        self.oracle = Oracle(self.env)
        return self._compute_scores()

    def _ensure_episode_completed(self) -> None:
        if self.env.item_queue or self.env.pending_queue:
            raise RuntimeError(
                "Cannot score an incomplete episode. Roll out until observation.done == True."
            )
        if not getattr(self.env, "_episode_snapshot", None):
            raise RuntimeError(
                "Cannot score episode before reset. Call env.reset(...) and run at least one rollout."
            )

    def _compute_scores(self) -> dict:
        if self.oracle is None:
            raise RuntimeError("Oracle not initialized. Call grade() first.")

        agent_profit = float(self.env.state.total_ledger)
        optimal_profit = float(self.oracle.compute_optimal_profit())
        margin_score = self._clamp_score(
            self.oracle.compute_margin_score(agent_profit)
        )

        fraud = self._compute_fraud_metrics()

        if self.task == "hard":
            final_score = self._clamp_score(0.6 * margin_score + 0.4 * fraud.f1)
        else:
            final_score = self._clamp_score(margin_score)

        return {
            "task": self.task,
            "profit": agent_profit,
            "optimal_profit": optimal_profit,
            "margin_score": margin_score,
            "fraud_metrics": {
                "tp": fraud.tp,
                "fp": fraud.fp,
                "fn": fraud.fn,
                "precision": fraud.precision,
                "recall": fraud.recall,
                "f1": fraud.f1,
            },
            "final_score": final_score,
        }

    def _compute_fraud_metrics(self) -> FraudMetrics:
        """
        Compute robust fraud metrics from full resolution history.

        Special handling:
        - If there are no fraud cases and no fraud flags, treat as near-perfect (f1=0.99 after clamp).
        - If there are no fraud cases but agent flags non-fraud items, clamp low (f1=0.01 floor).
        """
        tp = fp = fn = 0
        positives = 0
        predicted = 0

        snapshot_by_item = {
            int(record.item_id): record.hidden_condition
            for record in getattr(self.env, "_episode_snapshot", [])
        }

        for resolution in self.env._full_resolution_history:
            hidden = snapshot_by_item.get(int(resolution.item_id))
            is_fraud = hidden == Condition.FRAUDULENT
            flagged = resolution.action == DispositionAction.FLAG_FRAUD

            if is_fraud:
                positives += 1
            if flagged:
                predicted += 1

            if is_fraud and flagged:
                tp += 1
            elif (not is_fraud) and flagged:
                fp += 1
            elif is_fraud and (not flagged):
                fn += 1

        if positives == 0 and predicted == 0:
            # Nothing to detect and no false alarms: perfect.
            return FraudMetrics(
                tp=0,
                fp=0,
                fn=0,
                precision=self._clamp_score(1.0),
                recall=self._clamp_score(1.0),
                f1=self._clamp_score(1.0),
            )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return FraudMetrics(
            tp=tp,
            fp=fp,
            fn=fn,
            precision=self._clamp_score(precision),
            recall=self._clamp_score(recall),
            f1=self._clamp_score(f1),
        )

    @classmethod
    def _clamp_score(cls, score: float) -> float:
        """
        Clamps score strictly to (0.01, 0.99).
        Ensures derived task scores in validation never hit 0.0 or 1.0.
        """
        clamped = max(cls.MIN_SCORE, min(cls.MAX_SCORE, float(score)))
        return round(clamped, cls.SCORE_DECIMALS)
