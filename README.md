# AutonomousReturns-v0

OpenEnv-compatible reverse-logistics environment for e-commerce returns triage.

This repo provides:
- A sequential decision-making environment (`autonomous_returns_v0/`) with delayed outcomes, adversarial fraud, and economically grounded rewards
- A FastAPI server (`server.app:app`) to serve the environment over HTTP for `openenv validate` and container/Space runtimes
- Deterministic and LLM baselines (`run_baseline.py`)
- A hackathon-compliant `inference.py` with strict stdout formatting

Key files:
- Environment implementation: `autonomous_returns_v0/environment.py`
- Typed models: `autonomous_returns_v0/models.py`
- Scoring: `autonomous_returns_v0/grader.py`, `autonomous_returns_v0/oracle.py`
- Manifest: `openenv.yaml`

## What This Is

AutonomousReturns-v0 models returns processing as a sequential decision problem.

At each step, an agent observes partial, noisy evidence about the current return (condition score, reasons, packaging, damage flags, customer history) and chooses an operational action from a fixed menu (resell/discount/refurbish/dispose/flag fraud/inspect/wait).

This is not a one-shot classification task. Actions change what information becomes available, carry costs, and can resolve with delays.

## Why It's Hard (Realism)

Reverse logistics is hard because decisions must be made under uncertainty and adversarial pressure:
- Observations are noisy proxies, not ground truth
- Fraud is adversarial and can mimic "good" items (high condition score, clean history)
- Economic trade-offs dominate (inspection time and costs vs recovered value)
- Outcomes can be delayed (pending resolution queue) so good policies must reason beyond the current step

## Formal Framing (POMDP)

AutonomousReturns-v0 is naturally framed as a partially observable Markov decision process (POMDP):

- Hidden state
  - `Condition` in `autonomous_returns_v0/models.py` (`perfect`, `lightly_used`, `damaged`, `fraudulent`)
  - The environment samples a hidden condition per item and stores an immutable episode snapshot at reset
- Observation
  - `ReturnsObservation` and `Item` in `autonomous_returns_v0/models.py`
  - Includes noisy condition score, stated return reason, packaging condition, damage flags, inspection note, customer profile signals, plus queue/ledger context
- Action
  - `DispositionAction` in `autonomous_returns_v0/models.py`
  - Discrete, operational actions: `resell_full`, discount tiers, `refurbish`, `dispose`, `flag_fraud`, `inspect`, `wait`
- Transition
  - `pending_queue` and delay mechanics in `autonomous_returns_v0/environment.py`
  - Actions can enqueue items for delayed resolution; each step decrements delays and resolves eligible items
- Reward
  - Raw economics + penalties + shaping in `autonomous_returns_v0/environment.py`
  - Returned as a shaped, bounded `reward_detail.value` in `[0, 1]` for stable RL-style training/evaluation

## Dynamics That Make It Sequential

Delayed resolution (the main sequential ingredient):
- Non-`inspect` actions on an active item move it into a pending queue with a sampled delay (fixed or variable by task)
- Each `step()` also resolves pending items whose delay has expired, emitting reward that reflects those outcomes

Tail-drain steps (why `WAIT` exists):
- Once all items are acted upon, there can still be pending resolutions
- The environment continues until both the item queue and pending queue are empty; `WAIT` is the correct tail action

Active sensing via `INSPECT`:
- `INSPECT` has an explicit cost and reduces observation noise for the current item without advancing the queue
- It updates the item with a tighter condition score estimate and a deterministic-style inspection note

Extra-hard deception mechanisms (hard task):
- Fraud can mimic legitimate customer profiles and high condition scores
- Legitimate items can be generated with suspicious-looking histories (to stress false-positive control)
- Packaging/reason/inspection notes can be deceptively "clean" for fraud in extra-hard noise

## Reward and Scoring

### Step reward (conceptual)

Each step aggregates:
- Raw economics from resolved items (recovered value, processing costs, fraud recovery/penalties)
- Adjustment terms for the chosen action (inspection fee, active wait penalty, invalid tail actions, etc.)
- Shaping into `[0, 1]` via an arctan-based transform (to avoid early saturation on high-ticket episodes)

The structured reward is returned in `ReturnsObservation.reward_detail` (`autonomous_returns_v0/models.py`).

### Oracle and margin scoring

The oracle is a perfect-information reference used for grading (not training):
- `Oracle` in `autonomous_returns_v0/oracle.py` snapshots hidden conditions at reset
- It precomputes the best action per item among all disposition actions excluding `WAIT` and `INSPECT`
- `margin_score` is computed as a bounded ratio of realized profit to oracle optimal profit in `autonomous_returns_v0/grader.py`

### Hard-tier composite score

`autonomous_returns_v0/grader.py` defines:
- `easy`: `final_score = margin_score`
- `medium`: `final_score = margin_score`
- `hard`: `final_score = 0.6 * margin_score + 0.4 * fraud_f1`

Fraud metrics are computed from the full resolution history (robust to edge cases like "no fraud items").

## Baselines / Smart Methods Included

This repo intentionally focuses on an RL-ready environment plus reference policies (it does not claim to train a learned RL policy).

Deterministic baseline (recommended for reproducibility):
- `HeuristicPolicy` in `autonomous_returns_v0/policies.py`
- Uses signal counting to reduce false positives for fraud
- Uses price-aware thresholds for refurbish/dispose/discount decisions
- Uses inspection gating for expensive, uncertain items
- Handles tail-drain explicitly via `WAIT`

Optional LLM baseline (non-deterministic over time):
- `BaselineAgent` in `autonomous_returns_v0/baseline.py`
- The system prompt encodes the environment's noise model and rough action economics
- Provider support as implemented: OpenAI-compatible via `openai` SDK, and Google via Vertex AI SDK

## Hackathon `inference.py` Contract

`inference.py` (repo root) is designed to satisfy strict evaluator checks:
- Environment variables
  - `API_BASE_URL` (default: `https://api.openai.com/v1`)
  - `MODEL_NAME` (default: `gpt-4.1-mini`)
  - `HF_TOKEN` (required; used as the OpenAI client API key)
- Stdout format
  - Emits exactly three line types: `[START]`, `[STEP]`, `[END]` in that order
  - Rewards printed with 2 decimal places
  - Booleans printed as lowercase `true`/`false`
  - Always includes `error=<...>` per `[STEP]` (or `null`)

`inference.py` uses the OpenAI SDK against any OpenAI-compatible endpoint via `API_BASE_URL`.

## Server: What It's For

The server (`server.app:app`) is the OpenEnv/FastAPI wrapper for serving the environment over HTTP.

Notes:
- It is API-only (no custom web UI)
- Visiting `http://localhost:8000/` may return `{"detail":"Not Found"}` which is normal
- Use `http://localhost:8000/health` and `http://localhost:8000/docs`

This server is what enables:
- `openenv validate` against a running URL
- Running the environment inside Docker / a Hugging Face Space runtime

## How To Run (Local + Docker)

Local server:
```bash
python run_server.py
openenv validate --url http://127.0.0.1:8000
```

Deterministic baseline:
```bash
python run_baseline.py --mode local --tasks easy,medium,hard --seeds 42 --pretty
```

Docker (canonical spec is the root `Dockerfile`):
```bash
docker build -t autonomous-returns-v0:latest .
docker run --rm -p 8000:8000 autonomous-returns-v0:latest
```

Windows note:
- Prefer Docker Desktop with the WSL2 backend for Linux parity with the evaluation runtime.

## Evaluation Notes (From Attached Writeups)

The attached analyses highlight a common pattern when using LLM policies under increasing entropy:
- In hard settings, some models shift toward conservative strategies (high fraud-flagging, more waiting) that preserve completion but reduce economic efficiency
- Provider instability (e.g., intermittent API errors) can introduce "dead steps" where the agent defaults to a safe action and loses reward density

These results are illustrative examples from a particular run/model/provider configuration and are not guaranteed to reproduce exactly across time.

Reproduce locally:
```bash
# Deterministic, fully reproducible scoring
python run_baseline.py --mode local --tasks easy,medium,hard --seeds 42 --pretty

# LLM policy smoke run (requires HF_TOKEN; optional API_BASE_URL/MODEL_NAME)
python inference.py
```

## More Documentation

- Package/API details: `autonomous_returns_v0/README.md`
