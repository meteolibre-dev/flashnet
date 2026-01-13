# FlashNet Backend

CLI-based weather forecast inference pipeline for Cloud Run Jobs.

## Quick Start

### Prerequisites

- Docker with NVIDIA Container Toolkit (for GPU support)
- Docker Compose v2
- GCP credentials (if using cloud storage)

### 1. Setup Environment

```bash
# Copy environment template
cp backend/.env.example backend/.env

# Edit with your settings
nano backend/.env
```

### 2. Run Locally (CLI Mode)

**With GPU:**
```bash
# Build
docker build -f backend/Dockerfile --target runtime-cuda -t flashnet-backend .

# Run single pipeline execution
docker run --rm --gpus all \
  -v $(pwd)/models:/app/models:ro \
  -e MODE=cli \
  flashnet-backend
```

**With specific date:**
```bash
docker run --rm --gpus all \
  -v $(pwd)/models:/app/models:ro \
  -e MODE=cli \
  flashnet-backend \
  python3 backend/main.py --mode=cli --date="2026-01-12 13:40:00"
```

### 3. Run Web Server (Optional)

For health checks only:
```bash
docker run --rm --gpus all \
  -v $(pwd)/models:/app/models:ro \
  -p 8080:8080 \
  flashnet-backend \
  python3 backend/main.py --mode=web
```

## Usage

### CLI Mode (for Cloud Run Jobs)

```bash
# Run with default date (now - 4 hours)
python3 backend/main.py --mode=cli

# Run with specific date
python3 backend/main.py --mode=cli --date="2026-01-12 13:40:00"

# Run with custom model path
python3 backend/main.py --mode=cli --model-path=/path/to/model.safetensors
```

### Web Mode (for health checks)

```bash
python3 backend/main.py --mode=web
```

Endpoints:
- `GET /` - Service info
- `GET /health` - Health check
- `POST /pipeline/run` - Trigger pipeline

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODE` | `cli` | Run mode: `cli` or `web` |
| `MODEL_PATH` | `models/latest.safetensors` | Path to model weights |
| `MODEL_TYPE` | `jit` | Model architecture (`jit` or `unet`) |
| `PATCH_SIZE` | `128` | Tiled inference patch size |
| `DENOISING_STEPS` | `128` | Number of denoising steps |
| `BATCH_SIZE` | `64` | Inference batch size |
| `FORECAST_STEPS` | `18` | Total forecast frames |
| `NB_FORECAST` | `3` | Frames per batch |
| `GCP_SOURCE_BUCKET` | `eumetsat_mtg_preprocess` | Input bucket |
| `GCP_DEST_BUCKET` | `inference_result` | Output bucket |
| `CACHE_DIR` | `/tmp/flashnet_cache` | Temp cache directory |

### Model Files

Place your model weights in the `models/` directory:

```bash
ls models/
latest.safetensors
```

## Docker Commands

| Command | Description |
|---------|-------------|
| `make build` | Build CUDA image |
| `make build-cpu` | Build CPU-only image |
| `make run-detached` | Run in background (CLI mode) |
| `make run-web` | Run web server |
| `make logs-follow` | Follow logs |
| `make stop` | Stop container |

## Cloud Run Job Deployment

```bash
# Build and push
docker build -f backend/Dockerfile --target runtime-cuda -t flashnet-backend .
docker tag flashnet-backend gcr.io/PROJECT_ID/flashnet-repo/flashnet-backend:latest
docker push gcr.io/PROJECT_ID/flashnet-repo/flashnet-backend:latest

# Deploy Cloud Run Job
gcloud run jobs deploy flashnet-backend-job \
  --image gcr.io/PROJECT_ID/flashnet-repo/flashnet-backend:latest \
  --region europe-west3 \
  --cpu 4 \
  --memory 16Gi \
  --gpu 1 \
  --gpu-type nvidia-tesla-t4 \
  --task-timeout 3600 \
  --max-retries 2

# Execute job
gcloud run jobs execute flashnet-backend-job --region europe-west3
```

## File Locations

- **Input:** `gs://eumetsat_mtg_preprocess/inference_h5/`
- **Pattern:** `YYYY-MM-DD_HH-MM_region.h5`
- **Output:** `gs://inference_result/forecasts/YYYY-MM-DD/`

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  GCP Storage    │────▶│  CLI Pipeline│────▶│  GCP Storage    │
│  (Input H5)     │     │  Download    │     │  (Output NPZ)   │
└─────────────────┘     │  Inference   │     └─────────────────┘
                        │  Upload      │
                        └──────────────┘
```

## Troubleshooting

### GPU Not Available

```bash
# Verify NVIDIA drivers
nvidia-smi

# Install NVIDIA Container Toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Out of Memory

Reduce batch size in `.env`:
```env
BATCH_SIZE=32
FORECAST_STEPS=9
```

### GCP Authentication

```bash
# Set credentials path
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

Or use GitHub secret `GCP_CREDENTIALS` for CI/CD.