"""
Shared LLM prompting utilities for RevLogiXenv.

Provides unified prompt construction and action parsing for both the
hackathon inference entrypoint (inference.py) and the baseline agent
(baseline.py).

Two system prompts are exported to allow callers to choose the level of
detail:
- MINIMAL_SYSTEM_PROMPT: concise, single-sentence instruction (inference.py style)
- FULL_SYSTEM_PROMPT: detailed domain briefing with noise model and rules (baseline.py style)
"""

from __future__ import annotations

import re

from .models import DispositionAction, ReturnsAction, ReturnsObservation

# Minimal prompt — used by inference.py for hackathon Phase 2 validation.
MINIMAL_SYSTEM_PROMPT = (
    "You are a reverse-logistics triage agent. Output EXACTLY ONE WORD from this list: "
    "RESELL_FULL, RESELL_DISCOUNT_15, RESELL_DISCOUNT_30, RESELL_DISCOUNT_50, REFURBISH, "
    "DISPOSE, FLAG_FRAUD, INSPECT, WAIT. No explanation. No punctuation. Just the single word."
)

# Full prompt — used by baseline.py for richer LLM reasoning.
FULL_SYSTEM_PROMPT = """You are a reverse-logistics triage agent for AutonomousReturns-v0.

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


def observation_to_prompt_inference(obs: ReturnsObservation) -> str:
    """
    Build a concise LLM prompt from an observation.
    Used by inference.py for hackathon Phase 2 validation.
    """
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


def observation_to_prompt_baseline(obs: ReturnsObservation) -> str:
    """
    Build a detailed LLM prompt from an observation.
    Used by baseline.py for richer LLM reasoning.
    """
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
    action_strings = [a.value for a in DispositionAction]
    lines.append(", ".join(action_strings))
    return "\n".join(lines)


def parse_action_inference(response_text: str, obs: ReturnsObservation) -> ReturnsAction:
    """
    Parse an LLM response into a ReturnsAction using regex.
    Searches all lines and returns the last match found.
    Falls back to WAIT if no match.
    Used by inference.py.
    """
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


def parse_action_baseline(response_text: str, obs: ReturnsObservation) -> ReturnsAction:
    """
    Parse an LLM response into a ReturnsAction using exact + substring matching.
    Tries exact match first, then substring match, then falls back to default.
    Used by baseline.py.
    """
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
