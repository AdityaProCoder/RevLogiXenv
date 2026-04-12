---
title: RevLogiXenv
emoji: "📦"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - reverse-logistics
  - fraud-detection
  - decision-making
---

# RevLogiXenv

RevLogiXenv is a high-fidelity reinforcement learning environment designed to benchmark agentic reasoning in **reverse logistics**. 

Unlike simple classification tasks, reverse logistics involves complex, sequential decisions under significant uncertainty. Every returned item in an e-commerce ecosystem requires a triage decision that impacts long-term profitability and system integrity.

## The Challenge

In RevLogiXenv, an agent acts as a returns triage specialist. For every item, the agent must decide:
- **Resell**: At full price or various discount tiers (15%, 30%, 50%).
- **Refurbish**: Invest in quality restoration for high-value items.
- **Dispose**: Efficiently handle low-value or severely damaged goods.
- **Flag Fraud**: Identify adversarial returns without overwhelming the system with false positives.
- **Inspect**: Spend time and capital to reduce noise and gain higher signal.
- **Wait**: Manage the flow of delayed resolutions.

### Real-World Uncertainty
- **Noisy Signals**: Condition scores are imperfect proxies for ground truth.
- **Adversarial Fraud**: Fraudulent items are designed to mimic high-quality returns.
- **Delayed Outcomes**: The financial impact of a decision is often only known several steps later.
- **Operational Costs**: Every action (inspection, flagging, refurbishing) has a tangible cost.

---

## Project Structure

- `RevLogiXenv_v0/`: Core Python environment package. Includes POMDP dynamics, models, and reference policies.
- `server/`: FastAPI implementation providing an OpenEnv-compliant API.
- `inference.py`: Hackathon-compliant inference script for Phase 2 validation.
- `run_baseline.py`: CLI tool for running deterministic or LLM-based baselines.
- `openenv.yaml`: Environment manifest including task definitions and scoring formulas.

---

## Quickstart

### 1. Requirements

This project uses [uv](https://astral.sh/uv/) for high-performance dependency management.

```bash
# Install uv if you haven't already
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Setup

```bash
# Sync dependencies and create virtual environment
uv sync

# Configure environment variables
cp .env.example .env
# Edit .env and add your HF_TOKEN or OPENAI_API_KEY
```

### 3. Execution

#### Direct Inference (Hackathon Format)
The `inference.py` script is optimized for the Meta OpenEnv Hackathon Phase 2 validation. It emits standardized logs to stdout.

```bash
uv run python inference.py
```

#### Running Baselines
Use the `run_baseline.py` script for local testing and developer iteration.

```bash
# Run local deterministic baseline
uv run python run_baseline.py --mode local

# Run LLM-based baseline (requires OpenAI-compatible API)
uv run python run_baseline.py --mode llm --model gpt-4.1-mini
```

---

## OpenEnv API Contract

This environment adheres to the OpenEnv standard. You can interact with it via the local server or the provided `AutonomousReturnsEnv` class.

- **Observation Space**: Structured data including `current_item`, `noisy_condition_score`, `customer_profile`, and `running_ledger`.
- **Action Space**: Discrete actions (`resell_full`, `inspect`, `flag_fraud`, etc.).
- **Rewards**: Economics-driven rewards shaped strictly to the `(0.01, 0.99)` range.

---

## Hackathon Output Format

The `inference.py` script outputs the following standardized format for automated validation:

```text
[START] task=easy env=revlogixenv_v0 model=gpt-4.1-mini
[STEP] step=1 action=inspect reward=0.43 done=false error=null
...
[END] success=true steps=20 score=0.717 rewards=0.43,0.50,...
```

> [!IMPORTANT]
> To pass Phase 2 validation, all per-step rewards and the final task score are strictly bounded within the open interval (0, 1).

---

## Task Configurations

The environment supports three difficulty tiers plus legacy compatibility aliases:

| Task | Items | Fraud | Noise | Delay | Scoring |
|------|-------|-------|-------|-------|---------|
| `easy` | 20 | No | Medium | Fixed (3 steps) | Oracle-margin |
| `medium` | 30 | Yes | Hard | Variable | Oracle-margin |
| `hard` | 40 | Yes | Extra-hard | Variable | 0.6×margin + 0.4×fraud-F1 |
| `_legacy_easy` | 10 | No | Easy | Immediate | Oracle-margin |

The `_legacy_easy` task is preserved for backward compatibility with early experiments.
Internal environment code may also reference `_old_easy`, `_old_medium`, and `_old_hard`
keys — these are functionally identical to `easy`, `medium`, and `hard` respectively.

## Scoring Contexts

RevLogiXenv has **two independent scoring modes** that target different use cases:

1. **Inference mode** (hackathon Phase 2): `score = mean(step_rewards)`, each reward clamped to `(0.01, 0.99)`. This is what `inference.py` emits and what the hackathon validator parses.

2. **Grader mode** (local Python evaluation): Oracle-normalized margin score, with a fraud-F1 composite for `hard` tasks. Computed by the `Grader` class. This is used for local policy comparison and benchmarking.

Both modes enforce the same `(0.01, 0.99)` bounded range. The `openenv.yaml` manifest declares the inference-mode formula.

## Observation Fields

The `ReturnsObservation.reward_detail` field exposes a structured `Reward` payload with detailed economics:

- `value`: The shaped reward (clamped, used by inference)
- `raw_step_reward`: Unclamped raw economics in dollars
- `margin_component`: Shaped margin signal (0.01–0.99)
- `fraud_component`: Shaped fraud signal (0.01–0.99)
- `components`: Dict of line-item economics (`processing`, `revenue`, `fraud_recovery`, etc.)
- `penalties`: Dict of applied penalties and their strength

## Documentation

- [Technical Details](RevLogiXenv_v0/README.md): Deep dive into environment dynamics and models.
- [Task Manifest](openenv.yaml): Configuration for easy, medium, and hard tiers.
