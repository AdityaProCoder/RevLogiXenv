# OpenEnv/FastAPI runtime image for RevLogiXenv
FROM python:3.12-slim

WORKDIR /app

# Runtime safety + cleaner logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# Minimal OS deps + healthcheck utility
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv for lockfile-based environment sync
RUN pip install --upgrade pip uv

# Copy metadata/lock first to maximise Docker layer caching
COPY pyproject.toml uv.lock README.md ./

# Copy package source and entry-points
COPY RevLogiXenv_v0 ./RevLogiXenv_v0
COPY server ./server
COPY openenv.yaml ./openenv.yaml
COPY run_baseline.py run_server.py inference.py ./

# Install dependencies exactly from lockfile (no dev extras, no google deps)
RUN uv sync --frozen --no-dev

EXPOSE 8000

# Basic liveness check expected by OpenEnv/FastAPI deployments
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

# Start OpenEnv-compatible FastAPI app
CMD ["uv", "run", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
