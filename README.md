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

RevLogiXenv is an OpenEnv-compatible reverse-logistics benchmark where an agent must triage returned items under noise, fraud risk, delayed outcomes, and cost pressure.

## Current Problem (Today)

Reverse logistics is not a single-step classification task. It is a sequential decision system with delayed financial consequences.

Current operational pain points:

1. Partial observability: true condition is hidden; observed signals are noisy.
2. Adversarial behavior: fraud can mimic high-quality returns.
3. Delayed outcomes: action impact often appears several steps later.
4. Cost asymmetry: false fraud flags and missed fraud both create expensive failure modes.
5. Contract drift risk: manifest docs, runtime behavior, and evaluator logic can diverge if not tested together.

## What Is Being Solved Now

This repo version focuses on making the system safer and more maintainable for hackathon execution and local benchmarking:

1. Shared prompting architecture:
   - Common prompt-building and parsing logic is centralized in `RevLogiXenv_v0/prompting.py`.
   - `inference.py` and `RevLogiXenv_v0/baseline.py` consume the same shared logic.
2. Contract alignment:
   - Runtime supports `_legacy_easy` task key as declared in the manifest.
   - Task-level scoring metadata for `medium` is aligned with grader behavior.
3. Evaluator robustness:
   - Grader/oracle interactions are clearer and less dependent on private internals.
4. Better test coverage:
   - Added scoring and edge-case tests around grader behavior and fraud metrics.

## Current Limitations

Known limitations that still exist:

1. Inference and grader represent different scoring contexts by design, which can confuse new users.
2. Baseline provider support is currently openai-focused in code path assumptions.
3. Some historical local test expectations may not reflect current provider constraints.
4. Full production hardening (telemetry, CI policy gates, strict manifest-runtime lints) is still incremental.

## Detailed Solution Architecture

### 1) System Layers

1. Contract layer:
   - `openenv.yaml` defines tasks, action/observation models, and scoring metadata.
2. Runtime layer:
   - `RevLogiXenv_v0/environment.py` implements POMDP dynamics, queues, delays, reward shaping.
3. Policy layer:
   - `RevLogiXenv_v0/policies.py` deterministic baseline.
   - `RevLogiXenv_v0/baseline.py` LLM baseline.
   - `RevLogiXenv_v0/prompting.py` shared prompt/parse utilities.
4. Evaluation layer:
   - `RevLogiXenv_v0/grader.py` computes policy metrics and final scores.
   - `RevLogiXenv_v0/oracle.py` computes optimal reference profit and margin baseline.
5. API layer:
   - `server/app.py` exposes OpenEnv-compatible FastAPI server.

### 2) Scoring Contexts

There are two contexts and both are intentional:

1. Inference context (`inference.py`):
   - Emits strict `[START]`, `[STEP]`, `[END]` log lines.
   - Per-step rewards are clamped to `0.01-0.99`.
   - End line includes `score` and rewards CSV.
2. Grader context (`Grader`):
   - `easy` and `medium`: margin-based final score.
   - `hard`: weighted composite using margin and fraud F1.

### 3) Why This Architecture

1. Shared prompting removes duplicate logic drift.
2. Manifest/runtime alignment reduces submission failures.
3. Explicit grader/oracle contract improves local evaluation trust.
4. Layered modules make changes safer to test and review.

## Project Structure

- `RevLogiXenv_v0/`: Core package with environment, models, policies, grader, oracle, prompting.
- `server/`: API layer for OpenEnv-compatible interaction.
- `inference.py`: Hackathon inference entrypoint in required root location.
- `run_baseline.py`: Local and LLM baseline runner.
- `run_server.py`: Local API launcher.
- `openenv.yaml`: Manifest and task/scoring contract.
- `tests/`: Unit and regression tests.

## Quick Start

### A) Install (Primary: uv)

```bash
# Install uv (Windows PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install deps from lockfile
uv sync
```

### B) Install (Fallback: pip + venv)

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# Linux/macOS
# source .venv/bin/activate

pip install --upgrade pip
pip install -e .
pip install -e ".[dev]"
```

### C) Environment Setup

```bash
# Linux/macOS
cp .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```

Important vars:

- `API_BASE_URL` (has default)
- `MODEL_NAME` (has default)
- `HF_TOKEN` (required for inference)
- `TASKS` (optional list like `easy,medium,hard`)

## Run Instructions

### 1) Run Inference (Hackathon path)

```bash
uv run python inference.py
```

Expected log structure:

```text
[START] task=<task> env=<benchmark> model=<model>
[STEP] step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
```

### 2) Run Local Deterministic Baseline

```bash
uv run python run_baseline.py --mode local --tasks easy,medium,hard --seeds 42 --pretty
```

### 3) Run LLM Baseline

```bash
uv run python run_baseline.py --mode llm --provider auto --model gpt-4.1-mini --tasks easy --seeds 42 --pretty
```

### 4) Run API Server

```bash
uv run python run_server.py
```

Health check:

```bash
# Linux/macOS
curl http://localhost:8000/health

# Windows PowerShell
Invoke-WebRequest http://localhost:8000/health
```

## Docker

### Build

```bash
docker build -t revlogixenv:local .
```

### Run API

```bash
docker run --rm -p 8000:8000 -e PORT=8000 -e REVLOGIXENV_TASK=easy revlogixenv:local
```

### Health Check

```bash
curl http://localhost:8000/health
```

### Run Inference in Container

```bash
docker run --rm \
  -e HF_TOKEN=<your_token> \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=gpt-4.1-mini \
  -e TASKS=easy \
  --entrypoint uv revlogixenv:local run python inference.py
```

## Test Matrix

Core checks:

```bash
uv run pytest -q tests/test_scoring_consistency.py tests/test_grader_edge_cases.py tests/test_policies_improvements.py
uv run pytest -q tests/test_grader_and_baseline.py tests/test_grader_edge_cases.py
```

## Troubleshooting

1. `HF_TOKEN environment variable is required`:
   - Set `HF_TOKEN` in `.env` or shell environment.
2. Task key error:
   - Use valid tasks (`easy`, `medium`, `hard`, `_legacy_easy`, `_old_easy`, `_old_medium`, `_old_hard`).
3. API health failure:
   - Confirm server is running on port `8000` and port is free.
4. Baseline provider mismatch tests:
   - Some historical tests may assume unsupported providers.

## Additional Docs

- Core package internals: `RevLogiXenv_v0/README.md`
- Manifest contract: `openenv.yaml`
