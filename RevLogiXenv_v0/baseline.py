"""
Baseline agent for AutonomousReturns-v0.

Features:
- OpenEnv-compatible typed actions (`ReturnsAction`)
- Robust handling of delayed-reward tail steps (`WAIT`)
- Reproducible benchmark runner across easy/medium/hard tasks
- OpenAI-compatible model support with environment variable credentials
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import import_module
from typing import Iterable

from .environment import AutonomousReturnsEnv
from .grader import Grader
from .models import DispositionAction, ReturnsAction, ReturnsObservation
from .prompting import (
    FULL_SYSTEM_PROMPT as LLM_SYSTEM_PROMPT,
    observation_to_prompt_baseline as observation_to_prompt,
    parse_action_baseline as parse_action,
)


@dataclass(frozen=True)
class BaselineRunConfig:
    provider: str = "openai"
    model: str | None = None
    temperature: float = 0.15
    max_tokens: int = 64


class BaselineAgent:
    def __init__(self, config: BaselineRunConfig | None = None):
        self.config = config or BaselineRunConfig()
        self.provider = self.config.provider.lower()
        self.model = self.config.model

        if self.provider != "openai":
            raise ValueError("Only 'openai' provider is supported. Set provider='openai'.")

        try:
            openai_mod = import_module("openai")
            OpenAI = getattr(openai_mod, "OpenAI")
        except Exception as exc:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            ) from exc

        api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("HF_TOKEN or OPENAI_API_KEY environment variable not set")
        base_url = os.environ.get("API_BASE_URL") or os.environ.get("OPENAI_API_BASE_URL")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        if not self.model:
            self.model = os.environ.get("MODEL_NAME", "gpt-4.1-mini")

    def _call_llm(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        text = response.choices[0].message.content or ""
        return text.strip()

    def observation_to_prompt(self, obs: ReturnsObservation) -> str:
        return observation_to_prompt(obs)

    def parse_action(
        self, response_text: str, obs: ReturnsObservation
    ) -> ReturnsAction:
        return parse_action(response_text, obs)

    def act(self, obs: ReturnsObservation) -> ReturnsAction:
        if obs.current_item is None:
            return ReturnsAction(action=DispositionAction.WAIT)
        prompt = self.observation_to_prompt(obs)
        raw = self._call_llm(prompt)
        return self.parse_action(raw, obs)

    def run_episode(self, task: str, seed: int = 42, verbose: bool = False) -> dict:
        env = AutonomousReturnsEnv(task)
        obs = env.reset(seed=seed)

        if verbose:
            print(f"[{task}] provider={self.provider} model={self.model} seed={seed}")

        steps = 0
        while not obs.done:
            action = self.act(obs)
            if verbose:
                name = obs.current_item.product_title if obs.current_item else "<tail>"
                print(f"  step={steps + 1} item={name} action={action.action.value}")
            obs = env.step(action)
            steps += 1

        grader = Grader(task)
        graded = grader.score_completed_episode(env=env)

        return {
            "task": task,
            "provider": self.provider,
            "model": self.model,
            "seed": seed,
            "steps": steps,
            "profit": graded["profit"],
            "optimal_profit": graded["optimal_profit"],
            "margin_score": graded["margin_score"],
            "fraud_metrics": graded["fraud_metrics"],
            "final_score": graded["final_score"],
        }

    def run_benchmark(
        self,
        seeds: Iterable[int] = (42,),
        tasks: Iterable[str] = ("easy", "medium", "hard"),
        verbose: bool = False,
    ) -> dict:
        runs: list[dict] = []
        for seed in seeds:
            for task in tasks:
                runs.append(self.run_episode(task=task, seed=seed, verbose=verbose))

        by_task: dict[str, list[dict]] = {}
        for r in runs:
            by_task.setdefault(r["task"], []).append(r)

        summary: dict[str, dict] = {}
        overall_scores: list[float] = []

        for task, task_runs in by_task.items():
            avg_final = sum(x["final_score"] for x in task_runs) / len(task_runs)
            avg_margin = sum(x["margin_score"] for x in task_runs) / len(task_runs)
            avg_profit = sum(x["profit"] for x in task_runs) / len(task_runs)
            summary[task] = {
                "num_runs": len(task_runs),
                "avg_final_score": avg_final,
                "avg_margin_score": avg_margin,
                "avg_profit": avg_profit,
            }
            overall_scores.append(avg_final)

        overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0

        return {
            "provider": self.provider,
            "model": self.model,
            "seeds": list(seeds),
            "tasks": list(tasks),
            "runs": runs,
            "summary": summary,
            "overall_score": overall,
        }


def run_openai_baseline(
    model: str = "gpt-4.1-mini",
    seeds: Iterable[int] = (42,),
    verbose: bool = False,
) -> dict:
    agent = BaselineAgent(BaselineRunConfig(provider="openai", model=model))
    return agent.run_benchmark(seeds=seeds, verbose=verbose)


if __name__ == "__main__":
    result = run_openai_baseline(verbose=True)
    print(result)
