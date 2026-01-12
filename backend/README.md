# FlashNet Backend

REST API backend for weather forecasting using rectified flow models.

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

### 2. Build and Run

**With GPU (CUDA):**
```bash
make build
make run-detached
```

**CPU-only:**
```bash
make build-cpu
make run-cpu
```

### 3. Verify

```bash
# Check health
curl http://localhost:8080/health

# Submit inference task
curl -X POST http://localhost:8080/infer

# Check task status
curl http://localhost:8080/tasks/{task_id}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_HOST` | `0.0.0.0` | Server bind address |
| `SERVER_PORT` | `8080` | Server port |
| `MODEL_PATH` | `models/latest.safetensors` | Path to model weights |
| `MODEL_TYPE` | `jit` | Model architecture (`jit` or `unet`) |
| `PATCH_SIZE` | `128` | Tiled inference patch size |
| `DENOISING_STEPS` | `128` | Number of denoising steps |
| `BATCH_SIZE` | `64` | Inference batch size |
| `FORECAST_STEPS` | `18` | Total forecast frames to generate |
| `NB_FORECAST` | `3` | Frames per forecast batch |
| `GCP_SOURCE_BUCKET` | `eumetsat_mtg_preprocess` | Input H5 files bucket |
| `GCP_DEST_BUCKET` | `inference_result` | Output results bucket |

### Model Files

Place your model weights in the `models/` directory:

```bash
ls models/
latest.safetensors
```

## API Endpoints

### Health Check
```
GET /health
```
Returns service health status.

### List Files
```
GET /files?pattern=*.h5&limit=10
```
List available H5 files in source bucket.

### Get Latest File
```
GET /files/latest
```
Get the most recent H5 file.

### Submit Inference
```
POST /infer
{
  "file_pattern": "2026-01-12_*.h5",  // optional
  "forecast_steps": 18,                // optional
  "nb_forecast": 3                     // optional
}
```
Returns task ID for tracking.

### Task Status
```
GET /tasks/{task_id}
```
Returns task status and results.

### List Tasks
```
GET /tasks?limit=10
```
List recent tasks.

### Model Info
```
GET /models/info
```
Returns loaded model configuration.

## Docker Commands

| Command | Description |
|---------|-------------|
| `make build` | Build CUDA image |
| `make build-cpu` | Build CPU-only image |
| `make run-detached` | Run in background |
| `make logs-follow` | Follow logs |
| `make stop` | Stop container |
| `make clean` | Remove container |
| `make shell` | Open bash shell |

## Production Deployment

### GCP Cloud Run

```bash
# Build and push to Container Registry
make build
docker tag flashnet-backend:cuda gcr.io/PROJECT_ID/flashnet-backend
docker push gcr.io/PROJECT_ID/flashnet-backend

# Deploy
make deploy-cloud-run
```

### Docker Compose (Production)

```bash
# Override for production settings
docker compose -f backend/docker-compose.yml up -d
```

## File Locations

- **Input:** `gs://eumetsat_mtg_preprocess/inference_h5/`
- **Pattern:** `YYYY-MM-DD_HH-MM_region.h5`
- **Output:** `gs://inference_result/forecasts/YYYY-MM-DD/`

## Development

```bash
# Build with debug mode
make build-debug

# Run with hot reload
cd backend
python -m main
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

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  GCP Storage    │────▶│  Backend     │────▶│  GCP Storage    │
│  (Input H5)     │     │  API Server  │     │  (Output NPZ)   │
└─────────────────┘     └──────────────┘     └─────────────────┘
                              │
                              ▼
                       ┌──────────────┐
                       │   Inference  │
                       │    Engine    │
                       └──────────────┘
                              │
                              ▼
                       ┌──────────────┐
                       │   Tiled      │
                       │  Diffusion   │
                       └──────────────┘
```