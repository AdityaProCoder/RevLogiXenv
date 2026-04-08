from __future__ import annotations

from enum import Enum
from importlib import import_module
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

_openenv_env_server = import_module("openenv.core.env_server")
OpenEnvAction = getattr(_openenv_env_server, "Action")
OpenEnvObservation = getattr(_openenv_env_server, "Observation")
OpenEnvState = getattr(_openenv_env_server, "State")


class Condition(str, Enum):
    PERFECT = "perfect"
    LIGHTLY_USED = "lightly_used"
    DAMAGED = "damaged"
    FRAUDULENT = "fraudulent"


class DispositionAction(str, Enum):
    RESELL_FULL = "resell_full"
    RESELL_DISCOUNT_15 = "resell_discount_15"
    RESELL_DISCOUNT_30 = "resell_discount_30"
    RESELL_DISCOUNT_50 = "resell_discount_50"
    REFURBISH = "refurbish"
    DISPOSE = "dispose"
    FLAG_FRAUD = "flag_fraud"
    INSPECT = "inspect"
    WAIT = "wait"


class ReturnsAction(OpenEnvAction):
    """
    OpenEnv-compatible action payload.
    """

    action: DispositionAction = Field(
        ...,
        description="Disposition decision for current item. Use 'inspect' to gather more info. Use 'wait' when no current item is available.",
    )


class CustomerProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    customer_id: str
    total_returns: int = Field(ge=0)
    historical_fraud_flags: int = Field(ge=0)
    recent_abuse_signals: bool = False



class Item(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: int
    product_title: str
    category: str
    price: float = Field(gt=0)
    customer_profile: CustomerProfile
    stated_return_reason: str
    noisy_condition_score: float = Field(ge=0.0, le=10.0)
    packaging_condition: Optional[str] = None
    damage_flags: Optional[list[str]] = None
    inspection_note: Optional[str] = None


class ResolutionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: int
    action: DispositionAction
    normalized_reward: float
    reward_components: dict[str, float] = Field(default_factory=dict)
    hidden_condition_revealed: Optional[Condition] = Field(
        default=None,
        description="Optional reveal channel for analysis/debugging. May be omitted in strict deployments.",
    )


class Reward(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float = Field(ge=0.0, le=1.0)
    raw_step_reward: float = 0.0
    margin_component: float = Field(ge=0.0, le=1.0, default=0.0)
    fraud_component: float = Field(ge=0.0, le=1.0, default=0.0)
    penalties: dict[str, float] = Field(default_factory=dict)
    components: dict[str, float] = Field(
        default_factory=dict,
        description="Aggregated step economics and adjustment components.",
    )


class ReturnsObservation(OpenEnvObservation):
    """
    OpenEnv-compatible observation. Inherits reward/done/metadata fields from OpenEnvObservation.
    """

    current_item: Optional[Item] = None
    remaining_items: int = Field(ge=0)
    pending_resolution_count: int = Field(ge=0)
    running_ledger: float = 0.0
    last_three_resolutions: list[ResolutionSummary] = Field(default_factory=list)

    # Structured reward payload for typed consumers.
    reward_detail: Optional[Reward] = None


class ReturnsState(OpenEnvState):
    """
    OpenEnv-compatible environment state.
    """

    task: str
    episode_clock: int = Field(ge=0, default=0)
    total_ledger: float = 0.0
    item_queue_size: int = Field(ge=0, default=0)
    pending_queue_size: int = Field(ge=0, default=0)
    resolution_history_size: int = Field(ge=0, default=0)

    # Extra operational diagnostics.
    total_processed: int = Field(ge=0, default=0)
    total_flagged_fraud: int = Field(ge=0, default=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


# Backward-compatible aliases used by existing modules.
Action = DispositionAction
Observation = ReturnsObservation
EnvState = ReturnsState
