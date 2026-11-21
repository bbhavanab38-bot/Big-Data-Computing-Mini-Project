# syntax=docker/dockerfile:1
# ==============================================================================
# Multi-stage production Dockerfile for Banking Retention Intelligence Platform
# ==============================================================================

FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install security packages and curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root system user and group for runtime security
RUN groupadd -g 1001 appgroup && \
    useradd -u 1001 -g appgroup -m -s /bin/bash appuser

WORKDIR /app

# Install Python dependencies first for optimal Docker layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code with non-root ownership
COPY --chown=appuser:appgroup . /app

# Ensure data and models directory exist and have write permissions for appuser
RUN mkdir -p /app/data /app/models /app/data/campaigns && \
    chown -R appuser:appgroup /app/data /app/models

USER appuser

# ------------------------------------------------------------------------------
# Target: api (FastAPI v2 Microservice on port 8000)
# ------------------------------------------------------------------------------
FROM base AS api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["uvicorn", "src.api_v2.app:app", "--host", "0.0.0.0", "--port", "8000"]

# ------------------------------------------------------------------------------
# Target: ui (Streamlit Executive Retention Platform on port 8501)
# ------------------------------------------------------------------------------
FROM base AS ui

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "dashboard/platform_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
