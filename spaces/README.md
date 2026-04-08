---
title: RevLogiXenv
emoji: "📦"
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

# RevLogiXenv (OpenEnv FastAPI Space)

This Space serves the `RevLogiXenv` environment through an OpenEnv-compatible FastAPI runtime.

It is intended for API access (no custom web UI).

## What Runs in the Container

- App: `server.app:app`
- Command: `uvicorn server.app:app --host 0.0.0.0 --port 7860`
- Health: `GET /health`
- Docs: `GET /docs`

## Local Setup (for Space parity)

From the repo root (`RevLogiXenv/`):

1. Install:
   - `pip install .`

2. Run:
   - `uvicorn server.app:app --host 0.0.0.0 --port 7860`

## Notes

- The OpenEnv manifest lives at `openenv.yaml`.
- Difficulty tiers are `easy`, `medium`, and `hard`.

