# RevLogiXenv_v0 Core Internals

`RevLogiXenv_v0` contains the environment runtime, decision policies, and evaluators used by RevLogiXenv.

## Current Technical Problem

Core benchmarking reliability depends on three parts staying in sync:

1. Manifest contract (`openenv.yaml`).
2. Runtime behavior (`environment.py`).
3. Evaluator semantics (`grader.py` and `oracle.py`).

If these drift, benchmark outputs become hard to trust and hackathon validation can fail unexpectedly.

## What This Version Does

1. Introduces shared prompting utility module:
   - `prompting.py` centralizes prompt generation and action parsing.
2. Aligns task contract behavior:
   - Supports `_legacy_easy` at runtime.
3. Keeps scoring behavior explicit:
   - Inference and grader contexts are both documented and test-covered.

## Environment Model

RevLogiXenv is a POMDP with delayed financial resolution.

### Hidden State

Each item has a hidden true condition:

- `perfect`
- `lightly_used`
- `damaged`
- `fraudulent`

### Observable Signals

Agent receives noisy signals:

- Condition score
- Packaging condition
- Damage flags
- Optional inspection note
- Customer profile fraud/abuse history

### Action Set

- `resell_full`
- `resell_discount_15`
- `resell_discount_30`
- `resell_discount_50`
- `refurbish`
- `dispose`
- `flag_fraud`
- `inspect`
- `wait`

### Delayed Outcome Dynamics

Most actions enqueue pending resolutions; reward may appear after task-specific delay.

## Task Configurations

| Task | Items | Fraud | Noise | Delay |
|---|---:|---|---|---|
| `easy` | 20 | no | medium | fixed_3 |
| `medium` | 30 | yes | hard | variable |
| `hard` | 40 | yes | extra_hard | variable |
| `_legacy_easy` | 10 | no | easy | immediate |
| `_old_easy` | 20 | no | medium | fixed_3 |

Compatibility aliases `_old_medium` and `_old_hard` are also supported.

## Scoring Details

### Inference Path

- Per-step reward is clamped in `[0.01, 0.99]`.
- Final output includes step logs and final score field.

### Grader Path

`Grader` returns:

- `profit`
- `optimal_profit`
- `margin_score`
- `fraud_metrics` (`precision`, `recall`, `f1`, plus counts)
- `final_score`

Task formulas:

1. `easy`: margin-based.
2. `medium`: margin-based.
3. `hard`: weighted margin + fraud F1.

## Public API Surface

Main exported interfaces:

- `AutonomousReturnsEnv`
- `ReturnsAction`, `ReturnsObservation`, `ReturnsState`
- `Grader`
- `Oracle`
- `HeuristicPolicy`
- `BaselineAgent`

## Example Usage

```python
from RevLogiXenv_v0 import AutonomousReturnsEnv, Grader
from RevLogiXenv_v0.models import ReturnsAction, DispositionAction

env = AutonomousReturnsEnv("medium")
obs = env.reset(seed=42)
while not obs.done:
    obs = env.step(ReturnsAction(action=DispositionAction.RESELL_DISCOUNT_30))

grader = Grader("medium")
result = grader.score_completed_episode(env=env)
print(result["final_score"])
```

## Test Coverage Focus

- Scoring consistency across paths.
- Fraud metric edge cases.
- Policy behavior regressions.
- Incomplete episode guardrails.

## Related Files

- `environment.py`: transition and reward logic.
- `prompting.py`: shared prompt and parse utilities.
- `grader.py`: policy evaluation metrics.
- `oracle.py`: optimal reference policy/economics.
- `baseline.py`: LLM baseline runner.
- `policies.py`: deterministic heuristic baseline.
