from __future__ import annotations

from dataclasses import dataclass

from .models import DispositionAction, ReturnsAction, ReturnsObservation


@dataclass(frozen=True)
class PolicyConfig:
    """
    Tunable thresholds for the deterministic heuristic policy.

    Key changes from v1:
    - Removed fraud_score_threshold (high score = LESS fraud risk, not more)
    - Lowered fraud_return_count_threshold (be more aggressive on history)
    - Added explicit INSPECT score range to avoid inspecting obvious items
    """

    fraud_return_count_threshold: int = 2

    perfect_score_threshold: float = 8.3
    lightly_used_score_threshold: float = 6.0
    damaged_score_threshold: float = 4.0

    high_price_threshold: float = 300.0
    low_price_threshold: float = 90.0

    inspect_score_low: float = 5.0
    inspect_score_high: float = 7.5
    inspect_min_price: float = 250.0


class HeuristicPolicy:
    """
    Reusable deterministic policy for AutonomousReturns-v0.

    Goals:
    - Avoid API drift by always returning typed `ReturnsAction`
    - Be deterministic and reproducible
    - Provide a stronger non-LLM baseline for benchmarking
    """

    def __init__(self, config: PolicyConfig | None = None):
        self.config = config or PolicyConfig()

    def __call__(self, obs: ReturnsObservation) -> ReturnsAction:
        # Tail-drain steps (no item available): explicit WAIT.
        if obs.current_item is None:
            return ReturnsAction(action=DispositionAction.WAIT)

        item = obs.current_item
        score = float(item.noisy_condition_score)
        price = float(item.price)
        reason = (item.stated_return_reason or "").lower()
        returns = int(item.customer_profile.total_returns)
        hist_flags = int(item.customer_profile.historical_fraud_flags)
        recent_abuse = item.customer_profile.recent_abuse_signals

        damage_flags = [d.lower() for d in (item.damage_flags or [])]
        packaging = (item.packaging_condition or "").lower()
        note = (item.inspection_note or "").lower()

        # Fraud suspicion: requires multiple weak signals to reduce false positives.
        # CRITICAL: High score does NOT indicate fraud. In hard mode, FRAUDULENT items
        # score ~7.8 (std=0.70) to mimic quality, while PERFECT items score ~8.9.
        # High score + clean history = likely good item.
        fraud_signal_count = 0
        if returns >= self.config.fraud_return_count_threshold:
            fraud_signal_count += 1
        if any(k in reason for k in ("wrong item", "missing parts")):
            fraud_signal_count += 2  # Strong fraud indicator
        if "not as described" in reason:
            fraud_signal_count += 1
        if "mismatch" in note or "suspect" in note or "serial" in note:
            fraud_signal_count += 2  # Strong indicator
        if "missing" in packaging:
            fraud_signal_count += 1
        # REMOVED: score >= fraud_score_threshold (causes false positives)

        # Customer history is the most reliable fraud predictor
        if hist_flags > 0:
            fraud_signal_count += 1
        if recent_abuse:
            fraud_signal_count += 2

        if fraud_signal_count >= 3:
            return ReturnsAction(action=DispositionAction.FLAG_FRAUD)

        # Active Sensing: Inspect expensive items with moderate uncertainty
        # that haven't been inspected yet. Avoids inspecting obvious items.
        has_been_inspected = "detailed inspection confirms" in note
        if not has_been_inspected and price >= self.config.inspect_min_price:
            if self.config.inspect_score_low <= score <= self.config.inspect_score_high:
                if fraud_signal_count >= 1 or score < 6.0:
                    return ReturnsAction(action=DispositionAction.INSPECT)

        # Strong physical damage indicators.
        severe_damage = any(
            k in flag
            for flag in damage_flags
            for k in ("broken", "water", "missing parts")
        )
        moderate_damage = any(
            k in flag for flag in damage_flags for k in ("scratched", "wear", "used")
        ) or packaging in {"damaged", "missing"}

        # Damaged branch: price-sensitive refurbish/dispose/resell.
        if severe_damage or score < self.config.damaged_score_threshold:
            if price >= self.config.high_price_threshold:
                return ReturnsAction(action=DispositionAction.REFURBISH)
            if price <= self.config.low_price_threshold:
                return ReturnsAction(action=DispositionAction.DISPOSE)
            return ReturnsAction(action=DispositionAction.RESELL_DISCOUNT_50)

        # Lightly used branch.
        if score < self.config.lightly_used_score_threshold or moderate_damage:
            if price >= self.config.high_price_threshold:
                return ReturnsAction(action=DispositionAction.RESELL_DISCOUNT_15)
            if price <= self.config.low_price_threshold:
                return ReturnsAction(action=DispositionAction.RESELL_DISCOUNT_30)
            return ReturnsAction(action=DispositionAction.RESELL_DISCOUNT_30)

        # Near-perfect branch.
        if score >= self.config.perfect_score_threshold:
            if any(
                k in reason for k in ("changed mind", "wrong size", "no longer needed")
            ):
                return ReturnsAction(action=DispositionAction.RESELL_FULL)
            return ReturnsAction(action=DispositionAction.RESELL_DISCOUNT_15)

        # Default conservative choice.
        return ReturnsAction(action=DispositionAction.RESELL_DISCOUNT_30)


def heuristic_action(obs: ReturnsObservation) -> ReturnsAction:
    """
    Functional wrapper for quick integration with existing grader signatures.
    """
    return HeuristicPolicy()(obs)
