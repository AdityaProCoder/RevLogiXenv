---
title: AutonomousReturns-v0
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - reverse-logistics
  - fraud-detection
---

# AutonomousReturns-v0 (OpenEnv FastAPI Space)

AutonomousReturns-v0 is a real-world reverse-logistics environment for training and evaluating agent policies on e-commerce returns triage.

This Space serves the environment through an OpenEnv-compatible FastAPI runtime.

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

This Space uses the root Dockerfile and launches:

- `uvicorn server.app:app --host 0.0.0.0 --port 7860`

The container exposes and serves on port `7860`, matching Hugging Face Space configuration.
This deployment exposes OpenEnv API endpoints (programmatic usage) only.

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

## Notes

- Scoring is bounded to `[0.0, 1.0]` in grader outputs.
- Environment rewards include trajectory-level shaping and delayed outcome resolution.
- The deployment is designed for OpenEnv-style HTTP/WebSocket interaction via FastAPI.
