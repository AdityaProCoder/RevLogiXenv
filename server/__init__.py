"""
Root server package marker for OpenEnv app module resolution.

This package enables import paths such as:
- `server.app:app`

The actual FastAPI app is provided in `server/app.py` (root-level wrapper)
and delegates to `autonomous_returns_v0.server.app`.
"""
