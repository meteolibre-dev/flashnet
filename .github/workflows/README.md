# GitHub Workflows

## backend-gpu.yml

This workflow builds and deploys the FlashNet backend to Cloud Run with GPU support.

### Trigger
- Push to `main` branch when backend or model files change

### Jobs

1. **build-and-push** - Builds Docker image and pushes to Google Artifact Registry
2. **deploy-cloud-run-gpu-job** - Deploys Cloud Run Job with GPU
3. **execute-cloud-run-gpu-job** - Executes the job and waits for completion
4. **cleanup-old-jobs** - Cleans up old job executions (keeps last 5)

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `GCP_CREDENTIALS` | Service account JSON with required permissions |

### Required IAM Permissions

The service account needs:
- `Artifact Registry Reader/Writer`
- `Cloud Run Admin`
- `Storage Object Admin` (if reading from GCS)

### GPU Configuration

The workflow uses:
- GPU Type: `nvidia-tesla-t4`
- GPU Count: `1`
- Memory: `16Gi`
- CPU: `4`

### Environment Variables in Workflow

| Variable | Value |
|----------|-------|
| `PROJECT_ID` | meteoforecast |
| `GAR_LOCATION` | europe-west3 |
| `REPOSITORY` | flashnet-repo |
| `SERVICE` | flashnet-backend |
| `JOB_NAME` | flashnet-backend-gpu-job |
