"""Hackathon-compliant inference script for RevLogiXenv."""
from __future__ import annotations

import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from RevLogiXenv_v0 import AutonomousReturnsEnv
from RevLogiXenv_v0.models import DispositionAction, ReturnsAction, ReturnsObservation

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

TASKS = os.getenv("TASKS", "easy,medium,hard")
BENCHMARK = os.getenv("BENCHMARK", "revlogixenv_v0")

# Backwards-compatible label retained by older evaluators (if they hardcode it).
LEGACY_BENCHMARK_NAME = os.getenv("LEGACY_BENCHMARK_NAME", "revlogixenv_v0")
MAX_STEPS = 100
MAX_TOTAL_REWARD = 1.0
SUCCESS_SCORE_THRESHOLD = 0.5
TEMPERATURE = 0.15
MAX_TOKENS = 64

LLM_SYSTEM_PROMPT = (
    "You are a reverse-logistics triage agent. Output EXACTLY ONE WORD from this list: "
    "RESELL_FULL, RESELL_DISCOUNT_15, RESELL_DISCOUNT_30, RESELL_DISCOUNT_50, REFURBISH, "
    "DISPOSE, FLAG_FRAUD, INSPECT, WAIT. No explanation. No punctuation. Just the single word."
)


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={_bool_str(done)} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={_bool_str(success)} steps={steps} rewards={rewards_str}", flush=True)


def observation_to_prompt(obs: ReturnsObservation) -> str:
    if obs.current_item is None:
        return "No items left. Output exactly: wait"

    item = obs.current_item
    lines: list[str] = [
        f"Product: {item.product_title}",
        f"Price: ${item.price:.2f}",
        f"Score: {item.noisy_condition_score:.1f}/10",
        f"Reason: {item.stated_return_reason}",
        f"Returns: {item.customer_profile.total_returns}",
        f"Fraud flags: {item.customer_profile.historical_fraud_flags}",
        f"Abuse: {item.customer_profile.recent_abuse_signals}",
        f"Packaging: {item.packaging_condition}",
    ]
    if item.damage_flags:
        lines.append(f"Damage: {', '.join(item.damage_flags)}")
    if item.inspection_note:
        lines.append(f"Note: {item.inspection_note}")

    lines.append(f"Remaining: {obs.remaining_items}, Pending: {obs.pending_resolution_count}")
    lines.append(f"Ledger: ${obs.running_ledger:.2f}")
    if obs.last_three_resolutions:
        lines.append("Recent:")
        for resolution in obs.last_three_resolutions:
            lines.append(f"{resolution.action.value}={resolution.normalized_reward:.2f}")
    lines.append("")
    lines.append(
        "Output ONE word: RESELL_FULL, RESELL_DISCOUNT_15, RESELL_DISCOUNT_30, "
        "RESELL_DISCOUNT_50, REFURBISH, DISPOSE, FLAG_FRAUD, INSPECT, or WAIT"
    )
    return "\n".join(lines)


def parse_action(response_text: str, obs: ReturnsObservation) -> ReturnsAction:
    if obs.current_item is None:
        return ReturnsAction(action=DispositionAction.WAIT)

    text = (response_text or "").lower().strip()
    actions = [
        "resell_full",
        "resell_discount_15",
        "resell_discount_30",
        "resell_discount_50",
        "refurbish",
        "dispose",
        "flag_fraud",
        "inspect",
        "wait",
    ]
    pattern = r"\b(" + "|".join(sorted(actions, key=len, reverse=True)) + r")\b"

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    for line in reversed(lines):
        line_match = re.fullmatch(pattern, line)
        if line_match:
            matched = line_match.group(1)
            for action in DispositionAction:
                if action.value == matched:
                    return ReturnsAction(action=action)

    matches = re.findall(pattern, text)
    if matches:
        matched = matches[-1]
        for action in DispositionAction:
            if action.value == matched:
                return ReturnsAction(action=action)

    return ReturnsAction(action=DispositionAction.WAIT)


def get_model_response(client: OpenAI, prompt: str) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS + 1024,
            stream=False,
            extra_body={"reasoning_split": True},
        )
        content = (completion.choices[0].message.content or "").strip()
        if content:
            return content
        reasoning_details = getattr(completion.choices[0].message, "reasoning_details", None)
        if isinstance(reasoning_details, list):
            parts: list[str] = []
            for chunk in reasoning_details:
                if isinstance(chunk, dict) and "text" in chunk:
                    parts.append(chunk["text"] or "")
            return "".join(parts)
    except Exception:
        return "wait"
    return "wait"


def run_episode(client: OpenAI, task_name: str) -> tuple[bool, int, float, list[float]]:
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    env = AutonomousReturnsEnv(task_name)
    try:
        obs = env.reset(seed=42)
        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            prompt = observation_to_prompt(obs)
            message = get_model_response(client, prompt)
            action = parse_action(message, obs)

            obs = env.step(action)
            reward = float(obs.reward) if obs.reward is not None else 0.0
            done = bool(obs.done)
            error = None
            if isinstance(obs.metadata, dict):
                raw_error = obs.metadata.get("last_action_error")
                error = raw_error if isinstance(raw_error, str) and raw_error else None

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action.action.value, reward=reward, done=done, error=error)

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
    finally:
        try:
            env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return success, steps_taken, score, rewards


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    task_names = [t.strip() for t in TASKS.split(",") if t.strip()]
    for task_name in task_names:
        run_episode(client, task_name)


if __name__ == "__main__":
    main()

