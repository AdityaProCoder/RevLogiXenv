# OpenEnv/FastAPI runtime image for RevLogiXenv
FROM python:3.12-slim

WORKDIR /app

# Runtime safety + cleaner logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}" \
    PIP_NO_CACHE_DIR=1 \
    ENABLE_WEB_INTERFACE=false \
    PORT=8000

# Minimal OS deps + healthcheck utility
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv for lockfile-based environment sync
RUN pip install --upgrade pip setuptools wheel uv

# Copy metadata/lock first to maximize Docker layer caching
COPY pyproject.toml uv.lock README.md ./

# Copy package source
COPY RevLogiXenv_v0 ./RevLogiXenv_v0
COPY server ./server
COPY openenv.yaml ./openenv.yaml
COPY run_baseline.py run_server.py inference.py ./

# Install dependencies and project exactly from lockfile
RUN uv sync --frozen --no-dev

EXPOSE 8000

# Basic liveness check expected by OpenEnv/FastAPI deployments
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

# Start OpenEnv-compatible FastAPI app
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
