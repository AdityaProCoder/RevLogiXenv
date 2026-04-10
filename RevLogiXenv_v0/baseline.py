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

ACTION_STRINGS = [a.value for a in DispositionAction]

LLM_SYSTEM_PROMPT = """You are a reverse-logistics triage agent for AutonomousReturns-v0.

NOISE MODEL (CRITICAL):
- Items have HIDDEN TRUE CONDITION: perfect, lightly_used, damaged, fraudulent
- You observe NOISY condition scores (0-10), NOT ground truth
- Fraudulent items score HIGH (mean=7.8, std=0.70) to mimic quality items
- Perfect items score highest (mean=8.9), damaged score lowest (mean=3.0)
- High score + fraud history = SUSPICIOUS (may be fraudulent mimicking quality)
- High score + clean history = likely good

CUSTOMER PROFILE INTERPRETATION:
- RETURNS: number of previous returns (0-4+)
- HIST_FRAUD_FLAGS: past fraud indicators (0 = clean, 2+ = high risk)
- RECENT_ABUSE: True if customer shows abuse patterns

FRAUD INDICATORS (strongest to weakest):
1. 'wrong item' or 'missing parts' in return reason
2. high HIST_FRAUD_FLAGS (>0)
3. RECENT_ABUSE = True
4. 'mismatch'/'suspect'/'serial' in inspection note
5. 'missing' packaging

DISPOSITION VALUE GUIDE:
- RESELL_FULL: Best for PERFECT ($0.98x). Bad for damaged ($0.22x) or fraud ($0.10x)
- RESELL_DISCOUNT_15: Good for LIGHTLY_USED ($0.82x). Avoid on fraud ($0.22x)
- RESELL_DISCOUNT_30: General lightly-used fallback
- RESELL_DISCOUNT_50: Damaged items, low-value situations
- REFURBISH: High-price damaged items only ($18 fixed + 15% variable)
- DISPOSE: Low-price (<$90) damaged items
- FLAG_FRAUD: Only with 3+ fraud signals. Costs $6+$0.03x. Recovers $0.88x if correct
- INSPECT: Uncertain expensive items ($250+). Reduces noise 14x but costs $5

RULES:
- score > 8.0 AND no fraud history → likely PERFECT → RESELL_FULL or RESELL_DISCOUNT_15
- score 5.0-7.5 AND fraud signals → INSPECT before deciding
- score < 4.0 → likely damaged → REFURBISH (high$) or DISPOSE (low$)
- NEVER flag fraud with < 3 signals (causes expensive false positives)

Choose EXACTLY ONE action: RESELL_FULL, RESELL_DISCOUNT_15, RESELL_DISCOUNT_30, RESELL_DISCOUNT_50, REFURBISH, DISPOSE, FLAG_FRAUD, INSPECT, WAIT
Return ONLY the action name. No explanation. No punctuation."""


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
            self.model = os.environ.get("MODEL_NAME", "gpt-4o-mini")

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
        if obs.current_item is None:
            return (
                "No current item is available. "
                f"Pending resolutions: {obs.pending_resolution_count}. "
                "Output exactly one action token: wait"
            )

        item = obs.current_item
        lines: list[str] = [
            f"Product: {item.product_title}",
            f"Category: {item.category}",
            f"Price: {item.price:.2f}",
            f"Noisy condition score: {item.noisy_condition_score:.2f}/10",
            f"Return reason: {item.stated_return_reason}",
            f"Customer profile: RETURNS={item.customer_profile.total_returns}, "
            f"HIST_FRAUD_FLAGS={item.customer_profile.historical_fraud_flags}, "
            f"RECENT_ABUSE={item.customer_profile.recent_abuse_signals}",
            f"Packaging: {item.packaging_condition}",
        ]
        if item.damage_flags:
            lines.append(f"Damage flags: {', '.join(item.damage_flags)}")
        if item.inspection_note:
            lines.append(f"Inspection note: {item.inspection_note}")

        lines.append(f"Remaining items: {obs.remaining_items}")
        lines.append(f"Pending resolutions: {obs.pending_resolution_count}")
        lines.append(f"Running ledger: {obs.running_ledger:.4f}")

        if obs.last_three_resolutions:
            lines.append("Recent outcomes:")
            for r in obs.last_three_resolutions:
                lines.append(
                    f"- action={r.action.value} normalized_reward={r.normalized_reward:.4f}"
                )

        lines.append("")
        lines.append("Choose one action token exactly from:")
        lines.append(", ".join(ACTION_STRINGS))
        return "\n".join(lines)

    def parse_action(
        self, response_text: str, obs: ReturnsObservation
    ) -> ReturnsAction:
        text = (response_text or "").strip().lower()

        if obs.current_item is None:
            return ReturnsAction(action=DispositionAction.WAIT)

        for act in DispositionAction:
            if act.value == text:
                return ReturnsAction(action=act)

        for act in DispositionAction:
            if act.value in text:
                return ReturnsAction(action=act)

        return ReturnsAction(action=DispositionAction.RESELL_DISCOUNT_30)

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
    model: str = "gpt-4o-mini",
    seeds: Iterable[int] = (42,),
    verbose: bool = False,
) -> dict:
    agent = BaselineAgent(BaselineRunConfig(provider="openai", model=model))
    return agent.run_benchmark(seeds=seeds, verbose=verbose)


if __name__ == "__main__":
    result = run_openai_baseline(verbose=True)
    print(result)
