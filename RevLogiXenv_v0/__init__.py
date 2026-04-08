"""
AutonomousReturns-v0 package exports.

This package exposes:
- OpenEnv-compatible models (`ReturnsAction`, `ReturnsObservation`, `ReturnsState`)
- Environment/server runtime (`AutonomousReturnsEnv`)
- Client (`AutonomousReturnsClient`)
- Evaluation utilities (`Oracle`, `Grader`, `BaselineAgent`)
"""

from .baseline import BaselineAgent, BaselineRunConfig
from .baseline_local import (
    LocalBaselineRun,
    LocalBaselineRunner,
    LocalBaselineSummary,
    run_local_baseline,
)
from .client import AutonomousReturnsClient
from .environment import AutonomousReturnsEnv
from .grader import Grader
from .models import (
    Action,
    Condition,
    DispositionAction,
    EnvState,
    Item,
    ResolutionSummary,
    ReturnsAction,
    ReturnsObservation,
    ReturnsState,
    Reward,
)
from .oracle import Oracle
from .policies import HeuristicPolicy, PolicyConfig, heuristic_action

__all__ = [
    "Action",
    "Condition",
    "DispositionAction",
    "Item",
    "ResolutionSummary",
    "Reward",
    "ReturnsAction",
    "ReturnsObservation",
    "ReturnsState",
    "EnvState",
    "AutonomousReturnsEnv",
    "AutonomousReturnsClient",
    "Oracle",
    "Grader",
    "BaselineAgent",
    "BaselineRunConfig",
    "HeuristicPolicy",
    "PolicyConfig",
    "heuristic_action",
    "LocalBaselineRun",
    "LocalBaselineSummary",
    "LocalBaselineRunner",
    "run_local_baseline",
]

__version__ = "0.2.0"
