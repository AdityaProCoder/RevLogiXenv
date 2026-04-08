# AutonomousReturns-v0

OpenEnv-compatible reverse-logistics environment for training and evaluating agents on real e-commerce returns triage decisions.

## Why This Environment Exists

AutonomousReturns-v0 models a real operational workflow: deciding what to do with returned inventory under uncertainty (resell/refurbish/dispose/fraud-flag).  
This is not a toy game loop; it is intended as an agent-evaluation environment for economically grounded decision-making.

## Formal Framing (POMDP)

- Hidden state: true condition of each item (`perfect`, `lightly_used`, `damaged`, `fraudulent`).
- Observation: noisy proxies (condition score, packaging, return reason, damage flags, inspection notes).
- Action: one disposition decision per step.
- Transition: item enters pending resolution queue with task-dependent delay.
- Reward:
  - raw economics from item/action outcome plus explicit behavioral penalties,
  - shaped per-step reward in `[0, 1]` for stable RL training.

### Step Reward Math

For each step:

1. `raw_step_reward = sum(resolved_item_raw_rewards) + raw_adjustments`
2. `base = 0.5 + atan(raw_step_reward / 80.0) / pi`
3. `reward = clip(base - sum(nonnegative_penalties), 0.0, 1.0)`

Penalty channels include:
- `wait_with_active_item`
- `invalid_action_when_idle`
- `non_wait_on_tail`
- `post_done_step`

The structured reward payload (`reward_detail`) also carries aggregated step economics in `components`.

## OpenEnv API Contract

### Required Methods

- `reset(seed=None, episode_id=None, reveal_hidden_conditions=False, task=None, **kwargs) -> ReturnsObservation`
- `step(action) -> ReturnsObservation`
- `state -> ReturnsState`

### Typed Action Space

`ReturnsAction.action` enum:

- `resell_full`
- `resell_discount_15`
- `resell_discount_30`
- `resell_discount_50`
- `refurbish`
- `dispose`
- `flag_fraud`
- `inspect`
- `wait`

`wait` is valid and useful for tail-drain steps when no current item is available.  
Using `wait` while an active item exists is explicitly penalized.

### Typed Observation Space (`ReturnsObservation`)

- `current_item` (nullable)
- `remaining_items`
- `pending_resolution_count`
- `running_ledger`
- `last_three_resolutions`
- `reward_detail` (typed `Reward`)
- inherited OpenEnv fields: `done`, `reward`, `metadata`

### Typed State Space (`ReturnsState`)

- `episode_id`
- `step_count`
- `task`
- `episode_clock`
- `total_ledger`
- `item_queue_size`
- `pending_queue_size`
- `resolution_history_size`
- `total_processed`
- `total_flagged_fraud`
- `metadata`

## Tasks and Difficulty Ladder

| Task | Items | Fraud | Noise | Delay | Description |
|---|---:|---|---|---|---|---|
| `easy` | 20 | disabled | medium | fixed (3-step) | Moderate decision complexity |
| `medium` | 30 | enabled | high | variable (1-5) | High uncertainty |
| `hard` | 40 | enabled | extra_hard | variable (1-5) | Extreme uncertainty + adversarial noise |

**Note:** Legacy task configs (`_legacy_easy`, `_old_medium`, `_old_hard`) are preserved for backward compatibility but not used in normal task selection.

## Grading

Grader outputs deterministic, bounded metrics in `[0, 1]`:

- `margin_score`
- `fraud_metrics.precision`, `fraud_metrics.recall`, `fraud_metrics.f1`
- `final_score`

Task scoring:

- `easy`: `final_score = margin_score` (margin-only, like previous medium)
- `medium`: `final_score = margin_score` (margin-only, like previous hard)
- `hard`: `final_score = 0.6 * margin_score + 0.4 * fraud_f1` (composite with fraud)

## Project Layout

```text
autonomous_returns_v0/
├── autonomous_returns_v0/
│   ├── models.py
│   ├── environment.py
│   ├── oracle.py
│   ├── grader.py
│   ├── policies.py
│   ├── baseline_local.py
│   ├── baseline.py
│   ├── client.py
│   └── server/app.py
├── server/app.py
├── openenv.yaml
├── run_baseline.py
├── pyproject.toml
└── tests/
```

## Setup

```bash
pip install .
```

Optional:

```bash
pip install ".[openai]"
pip install ".[vertex]"
pip install ".[web]"
pip install ".[all]"
```

## Local Validation and Tests

```bash
openenv validate
pytest -q
```

## Run Web UI (Single Command)

Either command below enables the web interface automatically and serves:
`http://localhost:8000/web/`

```bash
python run_server.py
```

```bash
uv run run-server
```

Runtime contract validation:

```bash
uvicorn server.app:app --host 127.0.0.1 --port 8000
openenv validate --url http://127.0.0.1:8000
```

## Usage

### Direct Environment Loop

```python
from autonomous_returns_v0 import AutonomousReturnsEnv, ReturnsAction, DispositionAction

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

### Grader

```python
from autonomous_returns_v0 import Grader
from autonomous_returns_v0.policies import HeuristicPolicy

policy = HeuristicPolicy()
grader = Grader("hard")
result = grader.grade(policy, seed=42)
print(result["final_score"])
```

## Reproducible Baseline (Official Reporting Path)

Command:

```bash
python run_baseline.py --mode local --tasks easy,medium,hard --seeds 42,1337,2025 --pretty
```

Deterministic local baseline (this branch, current implementation):

| Task | Seeds | Avg Final Score | Avg Margin | Avg Profit | Avg Fraud F1 |
|---|---|---:|---:|---:|---:|
| easy | 42,1337,2025 | ~0.90-0.95 | ~0.90-0.95 | ~4000 | 1.0000 |
| medium | 42,1337,2025 | ~0.75-0.85 | ~0.75-0.85 | ~7000 | ~0.65 |
| hard | 42,1337,2025 | ~0.65-0.75 | ~0.65-0.75 | ~5000 | ~0.70 |

**Note:** Scores reflect swapped difficulty. medium is now the hardest (extra_hard noise, 40 items). hard uses previous hard config.

## Optional LLM Baseline

OpenAI/Google baseline runner is available but non-deterministic over time:

- `OPENAI_API_KEY`
- `VERTEX_PROJECT_ID`

Example:

```bash
python run_baseline.py --mode llm --provider openai --model gpt-4o-mini --tasks easy,medium,hard --seeds 42 --pretty
```

## OpenEnv Manifest

`openenv.yaml` includes:

- `spec_version`, `name`, `version`
- runtime config (`fastapi`, `app`, `port`)
- task metadata
- structured action/observation/state descriptors
- reward/grading metadata

## Notes

- Episode post-termination `step()` is explicitly guarded and non-rewarding.
- Hard-mode fraud grading is deterministic and robust to no-positive edge cases.
- Task can be overridden at reset: `reset(task="easy|medium|hard")`.

## License

Apache-2.0
