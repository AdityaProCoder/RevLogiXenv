"""
Hackathon-compliant inference script for AutonomousReturns-v0.
"""
from __future__ import annotations

import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

from autonomous_returns_v0 import AutonomousReturnsEnv
from autonomous_returns_v0.models import DispositionAction, ReturnsAction, ReturnsObservation
from autonomous_returns_v0.grader import Grader

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")
TASK_NAME = os.environ.get("TASK_NAME", "hard")
BENCHMARK = "autonomous_returns_v0"
IMAGE_NAME = os.environ.get("IMAGE_NAME", "")
MAX_STEPS = 100
MAX_TOTAL_REWARD = 1.0
SUCCESS_SCORE_THRESHOLD = 0.5
TEMPERATURE = 0.15
MAX_TOKENS = 64

LLM_SYSTEM_PROMPT = """You are a reverse-logistics triage agent. Output EXACTLY ONE WORD from this list: RESELL_FULL, RESELL_DISCOUNT_15, RESELL_DISCOUNT_30, RESELL_DISCOUNT_50, REFURBISH, DISPOSE, FLAG_FRAUD, INSPECT, WAIT. No explanation. No punctuation. Just the single word."""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_str = f" error={error}" if error else ""
    print(f"[STEP] step={step} action={action} reward={reward:.4f} done={done}{error_str}", flush=True)


def log_end(success: bool, steps: int, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success} steps={steps} rewards={rewards_str}")


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
        for r in obs.last_three_resolutions:
            lines.append(f"{r.action.value}={r.normalized_reward:.2f}")

    lines.append("")
    lines.append("Output ONE word: RESELL_FULL, RESELL_DISCOUNT_15, RESELL_DISCOUNT_30, RESELL_DISCOUNT_50, REFURBISH, DISPOSE, FLAG_FRAUD, INSPECT, or WAIT")
    return "\n".join(lines)


def parse_action(response_text: str, obs: ReturnsObservation) -> ReturnsAction:
    if obs.current_item is None:
        return ReturnsAction(action=DispositionAction.WAIT)

    import re
    # Clean up the response to handle reasoning vs final output
    # If the model uses a structured format or just outputs the word at the end
    text = (response_text or "").lower().strip()
    
    # Try searching from the end first to find the most likely final decision
    actions = ["resell_full", "resell_discount_15", "resell_discount_30",
               "resell_discount_50", "refurbish", "dispose", "flag_fraud", "inspect", "wait"]
    
    # Look for the last line that contains EXACTLY one of the action words, 
    # or the last occurrence of an action word in the entire text.
    pattern = r'\b(' + '|'.join(sorted(actions, key=len, reverse=True)) + r')\b'
    
    # Strategy 1: Look for a word on a line by itself (most likely the final answer)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        for line in reversed(lines):
            line_match = re.fullmatch(pattern, line)
            if line_match:
                matched = line_match.group(1)
                for act in DispositionAction:
                    if act.value == matched:
                        return ReturnsAction(action=act)

    # Strategy 2: Fallback to the last keyword found anywhere
    matches = re.findall(pattern, text)
    if matches:
        matched = matches[-1]
        for act in DispositionAction:
            if act.value == matched:
                return ReturnsAction(action=act)

    return ReturnsAction(action=DispositionAction.WAIT)


def get_model_response(client: OpenAI, step: int, prompt: str) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS + 1024, # Allow space for reasoning
            stream=False,
            extra_body={"reasoning_split": True},
        )
        
        # Collect all reasoning details
        reasoning_parts = []
        rd = getattr(completion.choices[0].message, "reasoning_details", None)
        if rd and isinstance(rd, list):
            for chunk in rd:
                if isinstance(chunk, dict) and "text" in chunk:
                    reasoning_parts.append(chunk["text"] or "")
        
        raw_reasoning = "".join(reasoning_parts)
        content = (completion.choices[0].message.content or "").strip()
        
        # Structured output: if content looks like a clean answer, return it.
        # Otherwise, combine with reasoning so the parser can find the best match.
        if content:
            # Check if content has any of our keywords
            # If it's just the keyword (as requested), this is perfect.
            return content
            
        print(f"[DEBUG] Step {step} fallback to reasoning (length: {len(raw_reasoning)})", flush=True)
        return raw_reasoning
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
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
            message = get_model_response(client, step, prompt)

            action = parse_action(message, obs)
            result = env.step(action)
            obs = result

            reward = obs.reward if obs.reward is not None else 0.0
            done = obs.done

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action.action.value, reward=reward, done=done, error=None)

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return success, steps_taken, score, rewards


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    for task_name in ["easy", "medium", "hard"]:
        run_episode(client, task_name)


if __name__ == "__main__":
    main()

