# OpenEnv/FastAPI runtime image for AutonomousReturns-v0
FROM python:3.10-slim

WORKDIR /app

# Runtime safety + cleaner logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ENABLE_WEB_INTERFACE=false \
    PORT=8000

# Minimal OS deps + healthcheck utility
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy package metadata + source before install (required for successful build)
COPY pyproject.toml README.md ./
COPY autonomous_returns_v0 ./autonomous_returns_v0
COPY server ./server
COPY openenv.yaml ./openenv.yaml

# Install package and dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install .

EXPOSE 8000

# Basic liveness check expected by OpenEnv/FastAPI deployments
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

# Start OpenEnv-compatible FastAPI app
CMD ["uvicorn", "autonomous_returns_v0.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
