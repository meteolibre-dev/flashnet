# FlashNet Backend Docker Image
# Single-stage build for CUDA runtime

FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

# Install system dependencies
# build-essential and git are needed for installing some python packages
# libgl1... are needed for opencv/image processing libraries if used
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libhdf5-serial-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency definition files first to leverage Docker cache
COPY pyproject.toml meteolibre_model/config/configs.yml ./
COPY backend/requirements.txt ./backend/

# Install Python dependencies
# Upgrade pip first to avoid issues with older versions
RUN pip install --no-cache-dir --break-system-packages --upgrade pip && \
    pip install --no-cache-dir --break-system-packages -r backend/requirements.txt

# Copy and install the local meteolibre-model package
COPY meteolibre_model/ ./meteolibre_model/
RUN pip install --no-cache-dir --break-system-packages -e .

# Copy the rest of the backend application
COPY backend/ ./backend/

# Create a non-root user for security
RUN useradd -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment configuration
ENV MODE=cli
ENV LOG_LEVEL=INFO
ENV CACHE_DIR=/tmp/flashnet_cache
ENV MODEL_PATH=/tmp/flashnet_cache/model.safetensors
ENV MODEL_GCS_PATH=gs://eumetsat_mtg_preprocess/assets/radar_finetune_v3_sigma0dot2.safetensors

# Default command
CMD ["python3", "backend/main.py", "--mode=cli"]
