# RevLogiXenv_v0

This directory contains the core Python environment package for the `RevLogiXenv/` repository:

- Environment dynamics (POMDP + delayed resolution)
- Typed action/observation/state models (OpenEnv-compatible)
- Reference policies (deterministic + optional LLM baseline runner)
- Grading utilities (oracle margin, fraud metrics, composite scores)

For a quickstart (setup + run commands), see the root `README.md`.

## Imports (Python)

```python
from RevLogiXenv_v0 import AutonomousReturnsEnv, ReturnsAction, DispositionAction
```

## Formal Framing (POMDP)

- Hidden state: true condition of each item (`perfect`, `lightly_used`, `damaged`, `fraudulent`).
- Observation: noisy proxies (condition score, packaging, return reason, damage flags, inspection notes, customer signals).
- Action: one operational decision per step (resell/discount/refurbish/dispose/flag fraud/inspect/wait).
- Transition: non-`inspect` actions enqueue items into a pending resolution queue with task-dependent delay.
- Reward: raw economics + explicit penalties, shaped to `[0, 1]` for stable RL evaluation.

## OpenEnv API Contract

Required methods/properties:

- `reset(seed=None, episode_id=None, reveal_hidden_conditions=False, task=None, **kwargs) -> ReturnsObservation`
- `step(action: ReturnsAction) -> ReturnsObservation`
- `state -> ReturnsState`

Typed action space (`ReturnsAction.action`):

- `resell_full`
- `resell_discount_15`
- `resell_discount_30`
- `resell_discount_50`
- `refurbish`
- `dispose`
- `flag_fraud`
- `inspect`
- `wait`

## Tasks

Task tiers are configured in `RevLogiXenv_v0/environment.py`:

- `easy`: 20 items, fraud disabled, fixed delay
- `medium`: 30 items, fraud enabled, variable delay
- `hard`: 40 items, fraud enabled, extra-hard noise, variable delay

## Example: Direct Environment Loop

```python
from RevLogiXenv_v0 import AutonomousReturnsEnv, ReturnsAction, DispositionAction

env = AutonomousReturnsEnv(task="hard")
obs = env.reset(seed=42)

while not obs.done:
    if obs.current_item is None:
        action = ReturnsAction(action=DispositionAction.WAIT)
    else:
        action = ReturnsAction(action=DispositionAction.RESELL_DISCOUNT_30)
    obs = env.step(action)

print(env.state.total_ledger)
```

## Example: Grading a Policy

```python
from RevLogiXenv_v0 import Grader
from RevLogiXenv_v0.policies import HeuristicPolicy

grader = Grader("hard")
result = grader.grade(HeuristicPolicy(), seed=42)
print(result["final_score"])
```

