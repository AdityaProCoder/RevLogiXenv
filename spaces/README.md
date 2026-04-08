---
title: AutonomousReturns-v0
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - reverse-logistics
  - fraud-detection
---

# AutonomousReturns-v0 (OpenEnv FastAPI Space)

AutonomousReturns-v0 is a real-world reverse-logistics environment for training and evaluating agent policies on e-commerce returns triage.

This Space serves the environment through an OpenEnv-compatible FastAPI runtime.
The interactive Gradio interface is available at `/web`.

## Environment Summary

You operate a returns-processing workflow where each step requires selecting a disposition action under uncertainty:

- `resell_full`
- `resell_discount_15`
- `resell_discount_30`
- `resell_discount_50`
- `refurbish`
- `dispose`
- `flag_fraud`
- `inspect`
- `wait`

Difficulty tiers:

- `easy`: 10 items, immediate resolution, low noise
- `medium`: 20 items, delayed resolution, medium noise
- `hard`: 30 items, delayed variable resolution, fraud enabled, high noise

## OpenEnv Metadata

The project includes an OpenEnv manifest at `openenv.yaml` with:

- `spec_version: 1`
- runtime: `fastapi`
- app: `server.app:app`
- typed action/observation/state model references
- task definitions and scoring metadata

## Setup (Local)

From the project root (`returnenv/autonomous_returns_v0`):

1. Install package and dependencies:
   - `pip install .`

2. Run the server:
   - `uvicorn server.app:app --host 0.0.0.0 --port 8000`

3. Health check:
   - `GET /health`

## Space/Docker Runtime

This Space uses the Dockerfile in `spaces/Dockerfile` and launches:

- `uvicorn server.app:app --host 0.0.0.0 --port 7860`

The container exposes and serves on port `7860`, matching Hugging Face Space configuration.
`ENABLE_WEB_INTERFACE=true` is enabled for this Space runtime, so both:

- OpenEnv API endpoints (programmatic usage), and
- Gradio judge-facing interface at `/web`

are available.

## Environment Variable Configuration

Optional runtime variable:

- `AUTONOMOUS_RETURNS_TASK` in `{easy, medium, hard}`

If unset or invalid, the server defaults to `easy`.

## Client Usage

The package provides a typed client:

- `AutonomousReturnsClient`
- `ReturnsAction`
- `DispositionAction`

It supports async usage and sync wrapper usage through the OpenEnv client base.

## Baselines

Two baselines are included:

1. `BaselineAgent` (LLM-based)
   - Supports OpenAI (`OPENAI_API_KEY`) and Google Vertex (`VERTEX_PROJECT_ID`).

2. `LocalBaselineRunner` (deterministic, no external API)
   - Reproducible task scoring over fixed seeds.

## Judge Playground (`/web`)

The custom Gradio tab (`Custom`) includes:

- Manual action controls for all 8 disposition actions
- Heuristic autoplay (deterministic, no API key required)
- LLM autoplay (`openai` or `google`) using runtime env vars only

LLM autoplay requires:

- `OPENAI_API_KEY` for OpenAI mode
- `VERTEX_PROJECT_ID` for Google mode

No API keys are entered in the UI.

## Notes

- Scoring is bounded to `[0.0, 1.0]` in grader outputs.
- Environment rewards include trajectory-level shaping and delayed outcome resolution.
- The deployment is designed for OpenEnv-style HTTP/WebSocket interaction via FastAPI.
