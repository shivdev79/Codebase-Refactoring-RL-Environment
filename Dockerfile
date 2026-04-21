# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Multi-stage build — compatible with openenv-base
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Install system dependencies (git for VCS deps, build tools for native wheels)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

ARG BUILD_MODE=in-repo
ARG ENV_NAME=rlproj

# Copy project code into the builder stage
COPY . /app/env

WORKDIR /app/env

# Ensure uv is available
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install all dependencies (with lock file if present, otherwise resolve)
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

# Install the project itself into the venv
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# Also ensure the extra sandbox deps are present in the venv
RUN /app/env/.venv/bin/pip install --quiet \
    "pytest>=8.0.0" \
    "pytest-json-report>=1.5.0" \
    "flake8>=7.0.0" \
    "openai>=1.0.0"

# ---------------------------------------------------------------------------
# Final runtime stage
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/env/.venv /app/.venv

# Copy environment source code
COPY --from=builder /app/env /app/env

# Activate the virtual environment for all subsequent commands
ENV PATH="/app/.venv/bin:$PATH"

# Make sure Python can find the rlproj package
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Health check — the FastAPI server exposes /health via openenv-core
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the FastAPI server
# Workers=1 keeps episode state consistent; scale via max_concurrent_envs instead
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 1"]
