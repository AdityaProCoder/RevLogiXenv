# RevLogiXenv Core Package

This package (`RevLogiXenv_v0`) implements the core environment logic and reference policies for the RevLogiXenv reverse logistics benchmark.

## Environment Overview

RevLogiXenv is a Partially Observable Markov Decision Process (POMDP) that simulates the operational realities of e-commerce returns.

### Core Dynamics
- **Condition States**: Every item has a ground-truth condition: `perfect`, `lightly_used`, `damaged`, or `fraudulent`.
- **Noisy Observations**: Agents see condition scores (0-10) and packaging status, which have high variance (especially for fraudulent items).
- **Delayed Resolution**: Most actions enqueue items into a "pending" queue. Financial results are only "resolved" after a task-specific delay.
- **Operational Costs**: Actions like `inspect` or `refurbish` carry fixed and variable costs that must be balanced against potential recovery values.

### Reward Shaping
Rewards are derived from actual economic outcomes (resale price - costs). Final rewards are shaped into the `(0.01, 0.99)` range:
- `0.99`: Reached or exceeded the theoretical optimal (Oracle).
- `0.01`: Significant loss or failure to act.
- Linear scaling used between bounds.

## API Contract

The environment implements the standard OpenEnv API:

```python
from RevLogiXenv_v0 import AutonomousReturnsEnv

env = AutonomousReturnsEnv(task="medium")
obs = env.reset(seed=42)

# Action space is discrete (ReturnsAction)
action = get_agent_decision(obs)
obs = env.step(action)
```

## Reference Policies

1. **Heuristic**: A robust deterministic policy in `RevLogiXenv_v0/policies.py` that handles threshold-based decision making.
2. **Baseline LLM**: An OpenAI-compatible agent in `RevLogiXenv_v0/baseline.py` that uses prompt-based reasoning.

## Grading and Evaluation

The `Grader` class provides precise evaluation of agent behavior:
- **Profit**: Raw monetary outcome.
- **Margin Score**: Normalised profit relative to an Oracle.
- **Fraud Metrics**: Precision, Recall, and F1 for identifying fraudulent items.
- **Composite Score**: Weighted combination of margin and fraud metrics (used for Hard difficulty).
