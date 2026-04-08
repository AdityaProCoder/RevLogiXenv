# AutonomousReturns-v0

OpenEnv-compatible reverse-logistics environment for e-commerce returns triage.

- Package docs: `autonomous_returns_v0/README.md`
- Server entrypoint: `server.app:app`
- CLI scripts: `server`, `run-server`, `baseline`

## Local Build Standard

- Canonical container spec: root `Dockerfile` only.
- Build command:
  - `docker build -t autonomous-returns-v0:latest .`

## Hackathon Notes

- `inference.py` is in the repo root and follows the required logging/env contract.
- No Hugging Face Space submission is performed by this repository workflow.
