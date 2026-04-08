from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Optional
from uuid import uuid4

import numpy as np

from .models import (
    Action,
    Condition,
    DispositionAction,
    EnvState,
    Item,
    ResolutionSummary,
    ReturnsAction,
    ReturnsObservation,
    Reward,
    CustomerProfile,
)

_openenv_env_server = import_module("openenv.core.env_server")
Environment = getattr(_openenv_env_server, "Environment")

HIDDEN_CONDITION_PRIORS: dict[
    tuple[str, int] | tuple[str, str], dict[Condition, float]
] = {
    ("electronics", 3): {
        Condition.FRAUDULENT: 0.40,
        Condition.DAMAGED: 0.30,
        Condition.LIGHTLY_USED: 0.22,
        Condition.PERFECT: 0.08,
    },
    ("electronics", 2): {
        Condition.FRAUDULENT: 0.28,
        Condition.DAMAGED: 0.26,
        Condition.LIGHTLY_USED: 0.30,
        Condition.PERFECT: 0.16,
    },
    ("electronics", 1): {
        Condition.FRAUDULENT: 0.14,
        Condition.DAMAGED: 0.22,
        Condition.LIGHTLY_USED: 0.40,
        Condition.PERFECT: 0.24,
    },
    ("electronics", 0): {
        Condition.FRAUDULENT: 0.05,
        Condition.DAMAGED: 0.12,
        Condition.LIGHTLY_USED: 0.38,
        Condition.PERFECT: 0.45,
    },
    ("clothing", 3): {
        Condition.FRAUDULENT: 0.10,
        Condition.DAMAGED: 0.20,
        Condition.LIGHTLY_USED: 0.50,
        Condition.PERFECT: 0.20,
    },
    ("clothing", 0): {
        Condition.FRAUDULENT: 0.02,
        Condition.DAMAGED: 0.08,
        Condition.LIGHTLY_USED: 0.22,
        Condition.PERFECT: 0.68,
    },
    ("default", "high"): {
        Condition.FRAUDULENT: 0.20,
        Condition.DAMAGED: 0.25,
        Condition.LIGHTLY_USED: 0.35,
        Condition.PERFECT: 0.20,
    },
    ("default", "low"): {
        Condition.FRAUDULENT: 0.02,
        Condition.DAMAGED: 0.08,
        Condition.LIGHTLY_USED: 0.22,
        Condition.PERFECT: 0.68,
    },
}

CONDITION_MEANS = {
    Condition.PERFECT: 8.9,
    Condition.LIGHTLY_USED: 6.3,
    Condition.DAMAGED: 3.0,
    Condition.FRAUDULENT: 7.8,
}

NOISE_STD = {"easy": 0.12, "medium": 0.45, "hard": 0.70, "extra_hard": 1.50}

STATED_REASONS = {
    Condition.PERFECT: [
        "no longer needed",
        "wrong size",
        "changed mind",
        "better price found",
    ],
    Condition.LIGHTLY_USED: [
        "not as expected",
        "too complicated",
        "better price found",
        "no longer needed",
    ],
    Condition.DAMAGED: ["defective", "broken", "not working", "arrived damaged"],
    Condition.FRAUDULENT: [
        "defective",
        "not as described",
        "wrong item",
        "missing parts",
    ],
}

PRODUCT_TITLES = {
    "electronics": [
        "Wireless Headphones",
        "USB-C Cable",
        "Laptop Charger",
        "Smartphone Case",
        "Bluetooth Speaker",
    ],
    "clothing": [
        "Cotton T-Shirt",
        "Denim Jeans",
        "Running Shoes",
        "Winter Jacket",
        "Wool Sweater",
    ],
    "home": [
        "Coffee Maker",
        "Toaster Oven",
        "Desk Lamp",
        "Throw Pillow",
        "Kitchen Knife Set",
    ],
    "books": [
        "Python Programming",
        "Data Science Guide",
        "Machine Learning Basics",
        "Web Development",
        "Algorithm Design",
    ],
}

PACKAGING_CONDITIONS = ["good", "opened", "damaged", "missing"]
INSPECTION_NOTES = [
    "looks fine",
    "minor wear",
    "visible damage",
    "repackaged",
    "suspect",
]

TASK_CONFIG = {
    "easy": {
        "num_items": 20,
        "fraud_enabled": False,
        "noise_level": "medium",
        "delay_type": "fixed_3",
    },
    "medium": {
        "num_items": 30,
        "fraud_enabled": True,
        "noise_level": "hard",
        "delay_type": "variable",
    },
    "hard": {
        "num_items": 40,
        "fraud_enabled": True,
        "noise_level": "extra_hard",
        "delay_type": "variable",
    },
    "_old_easy": {
        "num_items": 20,
        "fraud_enabled": False,
        "noise_level": "medium",
        "delay_type": "fixed_3",
    },
    "_old_medium": {
        "num_items": 30,
        "fraud_enabled": True,
        "noise_level": "hard",
        "delay_type": "variable",
    },
    "_old_hard": {
        "num_items": 40,
        "fraud_enabled": True,
        "noise_level": "extra_hard",
        "delay_type": "variable",
    },
}


@dataclass(frozen=True)
class EpisodeRecord:
    item_id: int
    hidden_condition: Condition
    price: float


@dataclass
class PendingRecord:
    item_id: int
    action: DispositionAction
    hidden_condition: Condition
    price: float
    delay_remaining: int


class AutonomousReturnsEnv(Environment[ReturnsAction, ReturnsObservation, EnvState]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, task: str = "easy"):
        super().__init__()
        if task not in TASK_CONFIG:
            raise ValueError(f"Task must be one of {list(TASK_CONFIG.keys())}")

        self.task = task
        self._config = TASK_CONFIG[task]
        self.action_space = list(Action)
        self._rng: np.random.Generator = np.random.default_rng()
        self._item_counter = 0

        self._episode_snapshot: list[EpisodeRecord] = []
        self._current_index = 0
        self._reset_state()

    def _reset_state(self) -> None:
        self.item_queue: list[Item] = []
        self.pending_queue: list[PendingRecord] = []
        self.resolution_history: list[ResolutionSummary] = []
        self._full_resolution_history: list[ResolutionSummary] = []
        self.total_ledger = 0.0

        self._state = EnvState(
            episode_id=str(uuid4()),
            step_count=0,
            task=self.task,
            episode_clock=0,
            total_ledger=0.0,
            item_queue_size=0,
            pending_queue_size=0,
            resolution_history_size=0,
            total_processed=0,
            total_flagged_fraud=0,
            metadata={
                "reveal_hidden_conditions": False,
            },
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        reveal_hidden_conditions: bool = False,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> ReturnsObservation:
        if task is not None:
            self._set_task(task)

        self._rng = np.random.default_rng(seed)
        self._item_counter = 0
        self._current_index = 0
        self._episode_snapshot = []
        self._reset_state()

        if episode_id:
            self._state.episode_id = episode_id

        self._state.metadata["reveal_hidden_conditions"] = bool(
            reveal_hidden_conditions
        )

        self._generate_items()
        self._state.item_queue_size = len(self.item_queue)
        self._state.pending_queue_size = 0
        self._state.resolution_history_size = 0

        return self._build_observation(done=False, reward_detail=None)

    def step(
        self,
        action: ReturnsAction | DispositionAction | str,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ReturnsObservation:
        coerced_action = self._coerce_action(action)

        if self._is_episode_done():
            self._state.step_count += 1
            self._state.episode_clock += 1

            penalties = {"post_done_step": 0.35}
            components = {"post_done_step_attempt": -1.0}
            shaped = self._shape_reward(
                raw_step_reward=0.0,
                action=coerced_action.action,
                penalties=penalties,
                components=components,
            )
            obs = self._build_observation(done=True, reward_detail=shaped)
            obs.reward = shaped.value
            obs.metadata.update(
                {
                    "pending_count": len(self.pending_queue),
                    "resolved_this_step": 0,
                    "raw_step_reward": 0.0,
                    "post_done_step_ignored": True,
                }
            )
            return obs

        self._state.step_count += 1
        self._state.episode_clock += 1

        penalties: dict[str, float] = {}
        adjustment_components: dict[str, float] = {}
        raw_adjustment = 0.0
        resolved_count = 0

        if self.item_queue:
            chosen_action = coerced_action.action
            current_item = self.item_queue[0]
            record = self._episode_snapshot[self._current_index]

            if chosen_action == DispositionAction.INSPECT:
                inspect_cost = 5.0
                raw_adjustment -= inspect_cost
                penalties["inspection_cost"] = 0.05
                adjustment_components["inspection_fee"] = -inspect_cost
                
                # Update item to have less noise
                # Safe to import CONDITION_MEANS here or reference it directly
                true_mean = CONDITION_MEANS[record.hidden_condition]
                better_score = float(np.clip(true_mean + float(self._rng.normal(0, 0.05)), 0.0, 10.0))
                
                updated_item = current_item.model_copy(update={
                    "noisy_condition_score": better_score,
                    "inspection_note": f"Detailed inspection confirms {record.hidden_condition.value} state."
                })
                self.item_queue[0] = updated_item
                # Do not advance index or pop
            else:
                self.item_queue.pop(0)
                self._current_index += 1

                if chosen_action == DispositionAction.WAIT:
                    # Exponential Backlog Penalty
                    queue_size = len(self.item_queue) + 1
                    backlog_multiplier = 1.0 + (queue_size * 0.15)
                    wait_penalty = max(5.0, 0.08 * float(current_item.price)) * backlog_multiplier
                    raw_adjustment -= wait_penalty
                    penalties["wait_with_active_item"] = 0.18 + (queue_size * 0.02)
                    adjustment_components["active_wait_penalty"] = -wait_penalty

                self.pending_queue.append(
                    PendingRecord(
                        item_id=current_item.item_id,
                        action=chosen_action,
                        hidden_condition=record.hidden_condition,
                        price=record.price,
                        delay_remaining=self._sample_delay(),
                    )
                )
        else:
            if coerced_action.action != DispositionAction.WAIT:
                idle_penalty = 8.0
                raw_adjustment -= idle_penalty
                penalties["invalid_action_when_idle"] = 0.22
                adjustment_components["invalid_idle_action_penalty"] = -idle_penalty
                if self.pending_queue:
                    tail_penalty = 6.0
                    raw_adjustment -= tail_penalty
                    penalties["non_wait_on_tail"] = 0.12
                    adjustment_components["tail_action_penalty"] = -tail_penalty

        step_raw_reward, resolved_count, resolved_components = self._decrement_and_resolve()
        if step_raw_reward != 0.0:
            self.total_ledger += step_raw_reward
        if raw_adjustment != 0.0:
            self.total_ledger += raw_adjustment

        merged_raw = raw_adjustment + step_raw_reward
        step_components = dict(resolved_components)
        self._merge_components(step_components, adjustment_components)
        shaped = self._shape_reward(
            raw_step_reward=merged_raw,
            action=coerced_action.action,
            penalties=penalties,
            components=step_components,
        )

        done = self._is_episode_done()
        obs = self._build_observation(done=done, reward_detail=shaped)
        obs.reward = shaped.value
        obs.metadata.update(
            {
                "pending_count": len(self.pending_queue),
                "resolved_this_step": resolved_count,
                "raw_step_reward": merged_raw,
                "penalties_applied": penalties,
            }
        )
        return obs

    @property
    def state(self) -> EnvState:
        self._state.total_ledger = self.total_ledger
        self._state.item_queue_size = len(self.item_queue)
        self._state.pending_queue_size = len(self.pending_queue)
        self._state.resolution_history_size = len(self.resolution_history)
        return self._state

    def close(self) -> None:
        return None

    def _set_task(self, task: str) -> None:
        if not isinstance(task, str):
            raise TypeError("Task override must be a string.")
        task_name = task.strip().lower()
        if task_name not in TASK_CONFIG:
            raise ValueError(f"Task must be one of {list(TASK_CONFIG.keys())}")
        self.task = task_name
        self._config = TASK_CONFIG[task_name]

    def _is_episode_done(self) -> bool:
        return len(self.item_queue) == 0 and len(self.pending_queue) == 0

    @staticmethod
    def _merge_components(
        target: dict[str, float],
        source: dict[str, float],
    ) -> None:
        for key, value in source.items():
            target[key] = float(target.get(key, 0.0) + float(value))

    def _generate_items(self) -> None:
        categories = list(PRODUCT_TITLES.keys())
        is_extra_hard = self._config["noise_level"] == "extra_hard"

        for _ in range(self._config["num_items"]):
            category = str(self._rng.choice(categories))
            price = float(self._rng.integers(15, 650))
            
            # Generate base customer profile
            return_count = int(self._rng.integers(0, 5))
            
            # Fraud deception for extra_hard: fraud users mimic legitimate patterns
            if is_extra_hard and self._config["fraud_enabled"]:
                # 50% of fraud items appear as first-time returners with clean history
                is_deceptive_fraud = self._rng.random() < 0.50
                if is_deceptive_fraud:
                    return_count = 0
                    hist_fraud = 0
                else:
                    hist_fraud = int(self._rng.integers(0, 3))
            else:
                hist_fraud = int(self._rng.integers(0, 3)) if return_count > 0 and self._config["fraud_enabled"] else 0
            
            # Legitimate deception for extra_hard: some legit users look suspicious
            if is_extra_hard and self._rng.random() < 0.45:
                # Fake suspicious profile on legitimate item
                hist_fraud = int(self._rng.integers(1, 3))
                return_count = int(self._rng.integers(2, 5))
            
            profile = CustomerProfile(
                customer_id=f"CUST_{self._rng.integers(1000, 9999)}",
                total_returns=return_count,
                historical_fraud_flags=hist_fraud,
                recent_abuse_signals=bool(hist_fraud > 0 and self._rng.random() < 0.3)
            )

            hidden = self._sample_hidden_condition(category, profile.total_returns)
            if not self._config["fraud_enabled"] and hidden == Condition.FRAUDULENT:
                hidden = Condition.DAMAGED

            noisy_score = self._generate_noisy_score(hidden)
            
            # Deceptive score injection for extra_hard
            if is_extra_hard:
                noisy_score = self._apply_deceptive_score(hidden, noisy_score)
            
            reason = str(self._rng.choice(STATED_REASONS[hidden]))
            packaging, damage_flags = self._sample_packaging_and_damage(hidden)
            
            # Deceptive packaging/reason injection for extra_hard
            if is_extra_hard:
                packaging, damage_flags, reason = self._apply_deceptive_signals(
                    hidden, packaging, damage_flags, reason
                )
            
            inspection_note = None
            if self._config["noise_level"] in ("hard", "extra_hard"):
                inspection_note = self._sample_inspection_note(hidden)
                # Extra hard: fraud inspection notes sometimes look clean
                if is_extra_hard and hidden == Condition.FRAUDULENT and self._rng.random() < 0.30:
                    inspection_note = "looks fine"

            item = Item(
                item_id=self._item_counter,
                product_title=str(self._rng.choice(PRODUCT_TITLES[category])),
                category=category,
                price=price,
                customer_profile=profile,
                stated_return_reason=reason,
                noisy_condition_score=noisy_score,
                packaging_condition=packaging,
                damage_flags=damage_flags,
                inspection_note=inspection_note,
            )
            self.item_queue.append(item)
            self._episode_snapshot.append(
                EpisodeRecord(
                    item_id=item.item_id, hidden_condition=hidden, price=price
                )
            )
            self._item_counter += 1

    def _sample_hidden_condition(self, category: str, return_count: int) -> Condition:
        key: tuple[str, int] | tuple[str, str] = (category.lower(), return_count)
        if key not in HIDDEN_CONDITION_PRIORS:
            key = ("default", "high" if return_count >= 2 else "low")
        prior = HIDDEN_CONDITION_PRIORS[key]
        conditions = list(prior.keys())
        probs = list(prior.values())
        idx = int(self._rng.choice(len(conditions), p=probs))
        return conditions[idx]

    def _sample_packaging_and_damage(self, hidden: Condition) -> tuple[str, list[str]]:
        if hidden == Condition.PERFECT:
            packaging_probs = [0.70, 0.24, 0.05, 0.01]
            damage_options = [["none"], ["minor scuff"]]
            damage_probs = [0.9, 0.1]
        elif hidden == Condition.LIGHTLY_USED:
            packaging_probs = [0.28, 0.52, 0.16, 0.04]
            damage_options = [["none"], ["scratched"], ["used signs"]]
            damage_probs = [0.35, 0.35, 0.30]
        elif hidden == Condition.DAMAGED:
            packaging_probs = [0.08, 0.26, 0.42, 0.24]
            damage_options = [
                ["broken"],
                ["missing parts"],
                ["water damage"],
                ["scratched"],
            ]
            damage_probs = [0.30, 0.26, 0.18, 0.26]
        else:
            packaging_probs = [0.25, 0.35, 0.20, 0.20]
            damage_options = [
                ["none"],
                ["missing parts"],
                ["wrong item marker"],
                ["repackaged mismatch"],
            ]
            damage_probs = [0.34, 0.22, 0.24, 0.20]

        packaging = str(self._rng.choice(PACKAGING_CONDITIONS, p=packaging_probs))
        damage = list(self._rng.choice(damage_options, p=damage_probs))
        return packaging, damage

    def _sample_inspection_note(self, hidden: Condition) -> str:
        if hidden == Condition.PERFECT:
            notes = ["looks fine", "factory seal likely broken", "minor cosmetic wear"]
        elif hidden == Condition.LIGHTLY_USED:
            notes = ["minor wear", "repackaged", "light usage traces"]
        elif hidden == Condition.DAMAGED:
            notes = ["visible damage", "functional test failed", "parts missing"]
        else:
            notes = ["serial mismatch suspect", "accessory mismatch", "suspect"]
        return str(self._rng.choice(notes))

    def _apply_deceptive_score(self, hidden: Condition, base_score: float) -> float:
        r = self._rng.random()
        if hidden == Condition.PERFECT and r < 0.50:
            return float(np.clip(self._rng.normal(4.5, 0.8), 0.0, 10.0))
        elif hidden == Condition.LIGHTLY_USED and r < 0.40:
            if r < 0.20:
                return float(np.clip(self._rng.normal(8.5, 0.5), 0.0, 10.0))
            else:
                return float(np.clip(self._rng.normal(3.5, 0.8), 0.0, 10.0))
        elif hidden == Condition.DAMAGED and r < 0.50:
            return float(np.clip(self._rng.normal(7.5, 0.8), 0.0, 10.0))
        elif hidden == Condition.FRAUDULENT and r < 0.55:
            return float(np.clip(self._rng.normal(8.8, 0.3), 0.0, 10.0))
        return base_score

    def _apply_deceptive_signals(
        self, hidden: Condition, packaging: str, damage_flags: list[str], reason: str
    ) -> tuple[str, list[str], str]:
        r = self._rng.random()
        if r < 0.30:
            packaging, damage_flags = self._invert_packaging_damage(packaging, damage_flags)
        if r < 0.25:
            reason = self._invert_reason(hidden)
        if r < 0.15:
            packaging = "good"
            damage_flags = ["none"]
        return packaging, damage_flags, reason

    def _invert_packaging_damage(self, packaging: str, damage_flags: list[str]) -> tuple[str, list[str]]:
        if packaging == "good" and self._rng.random() < 0.4:
            packaging = "damaged"
            damage_flags = ["none"]
        elif packaging == "damaged" and self._rng.random() < 0.4:
            packaging = "good"
        elif packaging in ("opened", "missing") and self._rng.random() < 0.3:
            if damage_flags == ["none"]:
                damage_flags = ["minor scuff"]
        return packaging, damage_flags

    def _invert_reason(self, hidden: Condition) -> str:
        if hidden == Condition.PERFECT:
            return self._rng.choice(["defective", "not working", "broken"])
        elif hidden == Condition.DAMAGED:
            return self._rng.choice(["changed mind", "wrong size", "no longer needed"])
        elif hidden == Condition.FRAUDULENT:
            return self._rng.choice(["changed mind", "wrong size", "better price found"])
        return "no longer needed"

    def _generate_noisy_score(self, hidden: Condition) -> float:
        mean = CONDITION_MEANS[hidden]
        noise = float(self._rng.normal(0, NOISE_STD[self._config["noise_level"]]))
        return float(np.clip(mean + noise, 0.0, 10.0))

    def _sample_delay(self) -> int:
        t = self._config["delay_type"]
        if t == "immediate":
            return 0
        if t == "fixed_3":
            return 3
        return int(self._rng.integers(1, 6))

    def _coerce_action(
        self,
        action: ReturnsAction | DispositionAction | str,
    ) -> ReturnsAction:
        if isinstance(action, ReturnsAction):
            return action
        if isinstance(action, DispositionAction):
            return ReturnsAction(action=action)
        if isinstance(action, str):
            try:
                return ReturnsAction(action=DispositionAction(action))
            except ValueError as exc:
                valid = ", ".join(a.value for a in DispositionAction)
                raise ValueError(
                    f"Unknown action string '{action}'. Valid actions: {valid}"
                ) from exc
        raise TypeError("Action must be ReturnsAction, DispositionAction, or str.")

    def _decrement_and_resolve(self) -> tuple[float, int, dict[str, float]]:
        for record in self.pending_queue:
            record.delay_remaining -= 1

        resolved = [r for r in self.pending_queue if r.delay_remaining <= 0]
        self.pending_queue = [r for r in self.pending_queue if r.delay_remaining > 0]

        step_raw = 0.0
        aggregated_components: dict[str, float] = {}
        for record in resolved:
            raw, components = self._compute_raw_reward(
                record.action, record.hidden_condition, record.price
            )
            step_raw += raw
            self._merge_components(aggregated_components, components)
            if record.action == DispositionAction.FLAG_FRAUD:
                self._state.total_flagged_fraud += 1

            reveal_hidden = bool(
                self._state.metadata.get("reveal_hidden_conditions", False)
            )
            summary = ResolutionSummary(
                item_id=record.item_id,
                action=record.action,
                normalized_reward=self._normalize_raw_reward(raw, record.price),
                reward_components=components,
                hidden_condition_revealed=record.hidden_condition
                if reveal_hidden
                else None,
            )
            self.resolution_history.append(summary)
            self._full_resolution_history.append(summary)
            if len(self.resolution_history) > 3:
                self.resolution_history.pop(0)

            self._state.total_processed += 1

        return step_raw, len(resolved), aggregated_components

    def _compute_raw_reward(
        self,
        action: DispositionAction,
        hidden: Condition,
        price: float,
    ) -> tuple[float, dict[str, float]]:
        processing = 0.05 * price
        inspect_cost = 1.5
        storage = 0.01 * price
        components: dict[str, float] = {
            "processing": -processing,
            "storage": -storage,
            "inspect": -inspect_cost,
        }

        if action == DispositionAction.RESELL_FULL:
            # Highest upside on pristine goods, highest downside on misclassified bad returns.
            if hidden == Condition.PERFECT:
                revenue = 0.98 * price
            elif hidden == Condition.LIGHTLY_USED:
                revenue = 0.80 * price
                components["quality_slippage"] = -0.05 * price
            elif hidden == Condition.DAMAGED:
                revenue = 0.22 * price
                components["secondary_return"] = -0.24 * price
            else:
                revenue = 0.10 * price
                components["chargeback"] = -0.62 * price
            components["revenue"] = revenue

        elif action == DispositionAction.RESELL_DISCOUNT_15:
            # Moderate markdown is best for many mid/high-ticket lightly used items.
            if hidden == Condition.PERFECT:
                revenue = 0.89 * price
            elif hidden == Condition.LIGHTLY_USED:
                revenue = 0.82 * price
            elif hidden == Condition.DAMAGED:
                revenue = 0.34 * price
                components["secondary_return"] = -0.10 * price
            else:
                revenue = 0.22 * price
                components["chargeback"] = -0.40 * price
            components["revenue"] = revenue
            if price >= 300 and hidden in (Condition.PERFECT, Condition.LIGHTLY_USED):
                components["high_value_demand_bonus"] = 0.03 * price

        elif action == DispositionAction.RESELL_DISCOUNT_30:
            # Wider markdown works better for low/mid ticket inventory.
            if hidden == Condition.PERFECT:
                revenue = 0.74 * price
            elif hidden == Condition.LIGHTLY_USED:
                revenue = 0.70 * price
            elif hidden == Condition.DAMAGED:
                revenue = 0.47 * price
                components["secondary_return"] = -0.05 * price
            else:
                revenue = 0.26 * price
                components["chargeback"] = -0.28 * price
            components["revenue"] = revenue
            if price < 120 and hidden in (Condition.PERFECT, Condition.LIGHTLY_USED):
                components["low_ticket_turnover_bonus"] = 0.05 * price

        elif action == DispositionAction.RESELL_DISCOUNT_50:
            # Deep markdown can be rational for very low-ticket goods and some damaged stock.
            if hidden == Condition.PERFECT:
                revenue = 0.56 * price
            elif hidden == Condition.LIGHTLY_USED:
                revenue = 0.57 * price
            elif hidden == Condition.DAMAGED:
                revenue = 0.46 * price
            else:
                revenue = 0.31 * price
                components["chargeback"] = -0.16 * price
            components["revenue"] = revenue
            if price < 60:
                components["clearance_speed_bonus"] = 0.06 * price

        elif action == DispositionAction.REFURBISH:
            # Refurbish has fixed+variable costs: better for expensive damaged items.
            refurb_fixed = 18.0
            refurb_variable = 0.15 * price
            components["refurb_fixed_cost"] = -refurb_fixed
            components["refurb_variable_cost"] = -refurb_variable
            if hidden == Condition.DAMAGED:
                recovered = 0.76 * price
            elif hidden == Condition.LIGHTLY_USED:
                recovered = 0.40 * price
            elif hidden == Condition.PERFECT:
                recovered = 0.22 * price
            else:
                recovered = 0.12 * price
                components["fraud_leak"] = -0.18 * price
            components["recovered_value"] = recovered

        elif action == DispositionAction.DISPOSE:
            # Disposal is expensive for high-value mistakes, but can be optimal for low-value damaged/fraud.
            disposal_fixed = 3.0
            components["disposal_processing"] = -disposal_fixed
            if hidden == Condition.DAMAGED:
                salvage = 0.10 * price
            elif hidden == Condition.FRAUDULENT:
                salvage = 0.03 * price
                components["containment_credit"] = 0.04 * price
            else:
                salvage = 0.01 * price
                components["waste_penalty"] = -0.14 * price
            components["salvage"] = salvage

        elif action == DispositionAction.FLAG_FRAUD:
            # Fraud flagging has review cost and false-positive cost, but strong recovery on true fraud.
            review_fixed = 6.0
            investigation_cost = 0.03 * price
            components["review_fixed_cost"] = -review_fixed
            components["investigation"] = -investigation_cost
            if hidden == Condition.FRAUDULENT:
                recovered = 0.88 * price
                components["fraud_recovery"] = recovered
            else:
                recovered = -0.30 * price
                components["false_flag_penalty"] = recovered

        else:  # WAIT
            components["idle_penalty"] = -0.01 * price

        is_extra_hard = self._config["noise_level"] == "extra_hard"
        if is_extra_hard and hidden == Condition.FRAUDULENT and action != DispositionAction.FLAG_FRAUD:
            components["missed_fraud_penalty"] = -0.70 * price

        if is_extra_hard and hidden != Condition.FRAUDULENT and action == DispositionAction.FLAG_FRAUD:
            components["false_fraud_flag_penalty"] = -0.55 * price

        total = sum(components.values())
        return total, components

    def _normalize_raw_reward(self, raw: float, price: float) -> float:
        if price <= 0:
            return 0.0
        return float(np.clip(raw / price, -2.0, 2.0))

    def _shape_reward(
        self,
        raw_step_reward: float,
        action: DispositionAction,
        penalties: Optional[dict[str, float]] = None,
        components: Optional[dict[str, float]] = None,
    ) -> Reward:
        del action  # Action-specific penalties are injected by caller.

        penalties = dict(penalties or {})
        components = dict(components or {})

        # Use arctan instead of logistic to avoid early saturation on high-ticket batches.
        base_value = 0.5 + (float(np.arctan(raw_step_reward / 80.0)) / float(np.pi))
        penalty_total = sum(max(0.0, float(v)) for v in penalties.values())
        value = float(np.clip(base_value - penalty_total, 0.0, 1.0))

        margin_component = float(
            np.clip(0.5 + (float(np.arctan(raw_step_reward / 120.0)) / float(np.pi)), 0.0, 1.0)
        )
        fraud_signal = (
            float(components.get("fraud_recovery", 0.0))
            + float(components.get("containment_credit", 0.0))
            + float(components.get("false_flag_penalty", 0.0))
        )
        fraud_component = float(
            np.clip(0.5 + (float(np.arctan(fraud_signal / 60.0)) / float(np.pi)), 0.0, 1.0)
        )

        return Reward(
            value=value,
            raw_step_reward=raw_step_reward,
            margin_component=margin_component,
            fraud_component=fraud_component,
            penalties=penalties,
            components=components,
        )

    def _build_observation(
        self,
        done: bool,
        reward_detail: Optional[Reward],
    ) -> ReturnsObservation:
        obs = ReturnsObservation(
            current_item=self.item_queue[0] if self.item_queue else None,
            remaining_items=len(self.item_queue),
            pending_resolution_count=len(self.pending_queue),
            running_ledger=self.total_ledger,
            last_three_resolutions=self.resolution_history[-3:]
            if self.resolution_history
            else [],
            reward_detail=reward_detail,
            done=done,
            reward=reward_detail.value if reward_detail else None,
            metadata={"task": self.task},
        )
        return obs
