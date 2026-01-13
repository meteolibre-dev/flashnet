# FlashNet Backend Docker Image
# Uses official PyTorch images with CUDA support

# ============================================
# Build Stage
# ============================================
FROM python:3.10-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python build dependencies
RUN pip install --no-cache-dir wheel setuptools packaging

# Copy and install project dependencies
WORKDIR /build

# Copy main project requirements
COPY pyproject.toml meteolibre_model/config/configs.yml ./

# Copy and install meteolibre-model (the main ML library)
COPY meteolibre_model/ ./meteolibre_model/
RUN pip install --no-cache-dir -e .

# Copy backend-specific requirements
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# ============================================
# GPU Runtime Stage - PyTorch CUDA Image
# ============================================
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS runtime-cuda

# Install additional system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libhdf5-serial-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy backend application
COPY --chown=appuser:appuser backend/ ./backend/

# Switch to non-root user
USER appuser

# Health check (for Cloud Run)
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Environment variables
ENV MODE=cli
ENV LOG_LEVEL=INFO
ENV CACHE_DIR=/tmp/flashnet_cache
ENV MODEL_PATH=/tmp/flashnet_cache/model.safetensors
ENV MODEL_GCS_PATH=gs://eumetsat_mtg_preprocess/assets/model_v15_mtg_world_lightning_shortcut_jit_e160.safetensors

# Default: Run CLI mode for Cloud Run Job
# Use --mode=web for continuous server
CMD ["python3", "backend/main.py", "--mode=cli"]


# ============================================
# Alternative CPU-only Runtime Stage
# ============================================
FROM python:3.10-slim-bookworm AS runtime-cpu

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libhdf5-serial-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy backend application
COPY --chown=appuser:appuser backend/ ./backend/

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Environment variables
ENV MODE=cli
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8080
ENV LOG_LEVEL=INFO
ENV CACHE_DIR=/tmp/flashnet_cache
ENV MODEL_PATH=/tmp/flashnet_cache/model.safetensors
ENV MODEL_GCS_PATH=gs://eumetsat_mtg_preprocess/assets/model_v15_mtg_world_lightning_shortcut_jit_e160.safetensors

# Default: Run CLI mode for Cloud Run Job
CMD ["python3", "backend/main.py", "--mode=cli"]


# ============================================
# Development Stage
# ============================================
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel AS runtime-dev

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    nano \
    htop \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy backend and model code
COPY --chown=appuser:appuser backend/ ./backend/
COPY --chown=appuser:appuser meteolibre_model/ ./meteolibre_model/

# Switch to non-root user
USER appuser

# Development mode with auto-reload
ENV MODE=dev
ENV LOG_LEVEL=DEBUG
ENV SERVER_RELOAD=true

CMD ["python3", "-c", "
import uvicorn
import sys
sys.path.insert(0, '/app')
from backend.main import web_app
uvicorn.run(web_app, host='0.0.0.0', port=8080, reload=True)
"]