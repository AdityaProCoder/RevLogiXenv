from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable

from .grader import Grader
from .models import ReturnsAction, ReturnsObservation
from .policies import HeuristicPolicy, PolicyConfig


@dataclass(frozen=True)
class LocalBaselineRun:
    task: str
    seed: int
    final_score: float
    margin_score: float
    profit: float
    optimal_profit: float
    fraud_f1: float


@dataclass(frozen=True)
class LocalBaselineSummary:
    task: str
    runs: int
    avg_final_score: float
    avg_margin_score: float
    avg_profit: float
    avg_fraud_f1: float


class LocalBaselineRunner:
    """
    Deterministic, local-only baseline runner.

    - No external APIs
    - Fully reproducible with explicit seeds
    - Uses the same grader outputs as submission scoring
    """

    def __init__(self, policy: HeuristicPolicy | None = None):
        self.policy = policy or HeuristicPolicy(PolicyConfig())

    def _agent_fn(self, obs: ReturnsObservation) -> ReturnsAction:
        return self.policy(obs)

    def run_once(self, task: str, seed: int) -> LocalBaselineRun:
        grader = Grader(task)
        result = grader.grade(self._agent_fn, seed=seed)
        fraud = result.get("fraud_metrics", {}) or {}

        return LocalBaselineRun(
            task=task,
            seed=seed,
            final_score=float(result["final_score"]),
            margin_score=float(result["margin_score"]),
            profit=float(result["profit"]),
            optimal_profit=float(result["optimal_profit"]),
            fraud_f1=float(fraud.get("f1", 0.0)),
        )

    def run(
        self,
        tasks: Iterable[str] = ("easy", "medium", "hard"),
        seeds: Iterable[int] = (42,),
    ) -> dict:
        tasks = list(tasks)
        seeds = list(seeds)

        runs: list[LocalBaselineRun] = []
        for task in tasks:
            for seed in seeds:
                runs.append(self.run_once(task=task, seed=seed))

        summaries: list[LocalBaselineSummary] = []
        for task in tasks:
            task_runs = [r for r in runs if r.task == task]
            summaries.append(
                LocalBaselineSummary(
                    task=task,
                    runs=len(task_runs),
                    avg_final_score=mean(r.final_score for r in task_runs),
                    avg_margin_score=mean(r.margin_score for r in task_runs),
                    avg_profit=mean(r.profit for r in task_runs),
                    avg_fraud_f1=mean(r.fraud_f1 for r in task_runs),
                )
            )

        overall_score = mean(s.avg_final_score for s in summaries) if summaries else 0.0

        return {
            "runner": "local_deterministic_heuristic",
            "tasks": tasks,
            "seeds": seeds,
            "runs": [r.__dict__ for r in runs],
            "summary": [s.__dict__ for s in summaries],
            "overall_score": overall_score,
        }


def run_local_baseline(
    tasks: Iterable[str] = ("easy", "medium", "hard"),
    seeds: Iterable[int] = (42,),
) -> dict:
    """
    Convenience function for quick local baseline execution.
    """
    return LocalBaselineRunner().run(tasks=tasks, seeds=seeds)


if __name__ == "__main__":
    report = run_local_baseline(tasks=("easy", "medium", "hard"), seeds=(42, 1337))
    print(report)
