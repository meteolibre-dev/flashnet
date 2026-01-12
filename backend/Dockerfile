# FlashNet Backend Docker Image
# Supports both CPU and GPU (CUDA) inference

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
# Runtime Stage
# ============================================
FROM nvidia/cuda:12.4-runtime-ubuntu22.04 AS runtime-cuda

# Set CUDA environment variables for PyTorch
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
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

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Default environment variables
ENV MODEL_PATH="/app/models/latest.safetensors"
ENV SERVER_HOST="0.0.0.0"
ENV SERVER_PORT="8080"
ENV LOG_LEVEL="INFO"
ENV CACHE_DIR="/tmp/flashnet_cache"

# Run the backend service
CMD ["python", "-m", "backend.main"]


# ============================================
# Alternative CPU-only Runtime Stage
# ============================================
# Use this stage if GPU is not available
# Build with: --target runtime-cpu
FROM python:3.10-slim-bookworm AS runtime-cpu

# Install system dependencies for headless rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
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

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Default environment variables
ENV MODEL_PATH="/app/models/latest.safetensors"
ENV SERVER_HOST="0.0.0.0"
ENV SERVER_PORT="8080"
ENV LOG_LEVEL="INFO"
ENV CACHE_DIR="/tmp/flashnet_cache"

# Run the backend service
CMD ["python", "-m", "backend.main"]