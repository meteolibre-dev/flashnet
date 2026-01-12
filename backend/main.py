"""
FlashNet weather forecasting backend service.

This service provides a REST API for:
1. Retrieving the latest H5 files from GCP bucket
2. Running tiled inference for weather forecasting
3. Pushing results back to GCP bucket
"""

import os
import sys
import logging
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import aiofiles
import numpy as np

# Add project root to sys.path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List

from backend.config import get_config, Config
from backend.gcp_client import get_gcs_client, GCPStorageClient, GCSFileInfo
from backend.inference_engine import InferenceEngine, InferenceResult, InferenceStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class InferenceTask(BaseModel):
    """Model for inference task submission."""
    file_pattern: Optional[str] = Field(
        None,
        description="Pattern to match files (e.g., '2026-01-12_*.h5'). Uses latest if not provided."
    )
    forecast_steps: int = Field(18, ge=1, le=100, description="Number of forecast steps")
    nb_forecast: int = Field(3, ge=1, le=10, description="Frames per forecast batch")


class InferenceTaskResponse(BaseModel):
    """Response for task submission."""
    task_id: str
    status: str
    message: str
    file_info: Optional[Dict[str, Any]] = None


class TaskStatusResponse(BaseModel):
    """Response for task status query."""
    task_id: str
    status: str
    message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    output_files: Optional[List[str]] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class HealthResponse(BaseModel):
    """Response for health check."""
    status: str
    model_loaded: bool
    gcp_connected: bool
    timestamp: datetime


# Global state
_executor = ThreadPoolExecutor(max_workers=1)
_running_tasks: Dict[str, InferenceResult] = {}
_inference_engine: Optional[InferenceEngine] = None


def get_inference_engine() -> InferenceEngine:
    """Get or create the inference engine."""
    global _inference_engine
    if _inference_engine is None:
        config = get_config()
        _inference_engine = InferenceEngine(
            model_path=config.model.model_path,
            model_type=config.model.model_type,
            patch_size=config.model.patch_size,
            denoising_steps=config.model.denoising_steps,
            batch_size=config.model.batch_size,
            context_frames=config.model.context_frames,
            use_residual=config.model.use_residual
        )
    return _inference_engine


def cleanup_old_tasks(max_age_hours: int = 24) -> None:
    """Clean up old task results."""
    global _running_tasks
    cutoff = datetime.now() - datetime.timedelta(hours=max_age_hours)
    expired_ids = [
        task_id for task_id, result in _running_tasks.items()
        if result.created_at and result.created_at < cutoff
    ]
    for task_id in expired_ids:
        del _running_tasks[task_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting FlashNet backend service...")
    config = get_config()
    logger.info(f"Configuration loaded: {config}")

    # Pre-load model to check availability
    try:
        engine = get_inference_engine()
        logger.info("Inference engine initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize inference engine: {e}")

    # Test GCP connection
    try:
        gcs_client = get_gcs_client()
        files = gcs_client.list_files(max_results=1)
        logger.info(f"GCP connection successful, found {len(files)} files in source bucket")
    except Exception as e:
        logger.warning(f"GCP connection failed: {e}")

    yield

    # Shutdown
    logger.info("Shutting down FlashNet backend service...")
    if _inference_engine:
        _inference_engine.cleanup()
    _executor.shutdown(wait=False)


app = FastAPI(
    title="FlashNet Weather Forecasting API",
    description="Backend service for weather forecasting using rectified flow models",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def run_inference_task(
    task_id: str,
    gcs_path: str,
    forecast_steps: int,
    nb_forecast: int,
    output_dir: str
) -> None:
    """Run inference task in background."""
    global _running_tasks

    try:
        config = get_config()
        gcs_client = get_gcs_client()
        engine = get_inference_engine()

        # Download file from GCS
        local_data_path = os.path.join(output_dir, "input.h5")
        gcs_client.download_file(gcs_path, local_data_path)

        # Run inference
        result = engine.run_inference(
            data_path=local_data_path,
            output_dir=output_dir,
            forecast_steps=forecast_steps,
            nb_forecast=nb_forecast
        )

        if result.status == InferenceStatus.COMPLETED:
            # Upload results to GCS
            dest_prefix = f"{config.gcp.dest_prefix}/{datetime.now().strftime('%Y-%m-%d')}"
            for filename in Path(output_dir).glob("*.npz"):
                dest_path = f"{config.gcp.dest_bucket}/{dest_prefix}/{filename.name}"
                gcs_client.upload_file(str(filename), dest_path)

        _running_tasks[task_id] = result

    except Exception as e:
        logger.exception(f"Task {task_id} failed")
        _running_tasks[task_id] = InferenceResult(
            status=InferenceStatus.FAILED,
            error_message=str(e),
            created_at=datetime.now(),
            completed_at=datetime.now()
        )


@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint."""
    return {
        "service": "FlashNet Weather Forecasting API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    config = get_config()
    gcs_connected = False
    model_loaded = False

    try:
        gcs_client = get_gcs_client()
        files = gcs_client.list_files(max_results=1)
        gcs_connected = True
    except Exception as e:
        logger.error(f"Health check GCP error: {e}")

    try:
        engine = get_inference_engine()
        model_loaded = engine.model is not None
    except Exception as e:
        logger.error(f"Health check model error: {e}")

    status = "healthy" if (gcs_connected and model_loaded) else "degraded"

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        gcp_connected=gcs_connected,
        timestamp=datetime.now()
    )


@app.get("/files", response_model=List[Dict[str, Any]])
async def list_files(pattern: Optional[str] = None, limit: int = 10):
    """List available files in source bucket."""
    try:
        gcs_client = get_gcs_client()
        files = gcs_client.list_files(
            prefix=pattern or "",
            extension=".h5",
            max_results=limit
        )

        return [
            {
                "name": f.name,
                "bucket": f.bucket,
                "size": f.size,
                "updated": f.updated.isoformat(),
                "gcs_path": f.gcs_path
            }
            for f in files
        ]
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files/latest", response_model=Dict[str, Any])
async def get_latest_file():
    """Get the latest file in source bucket."""
    try:
        gcs_client = get_gcs_client()
        latest = gcs_client.get_latest_file()

        if not latest:
            raise HTTPException(status_code=404, detail="No files found")

        return {
            "name": latest.name,
            "bucket": latest.bucket,
            "size": latest.size,
            "updated": latest.updated.isoformat(),
            "gcs_path": latest.gcs_path
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/infer", response_model=InferenceTaskResponse)
async def submit_inference(task: InferenceTask, background_tasks: BackgroundTasks):
    """Submit a new inference task."""
    task_id = str(uuid.uuid4())
    config = get_config()

    try:
        gcs_client = get_gcs_client()

        # Find source file
        if task.file_pattern:
            files = gcs_client.get_file_by_pattern(task.file_pattern)
            if not files:
                raise HTTPException(
                    status_code=404,
                    detail=f"No files found matching pattern: {task.file_pattern}"
                )
            source_file = files[0]
        else:
            source_file = gcs_client.get_latest_file()
            if not source_file:
                raise HTTPException(
                    status_code=404,
                    detail="No files found in source bucket"
                )

        # Create output directory
        output_dir = os.path.join(config.cache.cache_dir, task_id)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Submit background task
        background_tasks.add_task(
            run_inference_task,
            task_id,
            source_file.gcs_path,
            task.forecast_steps,
            task.nb_forecast,
            output_dir
        )

        return InferenceTaskResponse(
            task_id=task_id,
            status="pending",
            message=f"Inference task submitted for {source_file.name}",
            file_info={
                "name": source_file.name,
                "gcs_path": source_file.gcs_path
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get the status of an inference task."""
    if task_id not in _running_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    result = _running_tasks[task_id]

    return TaskStatusResponse(
        task_id=task_id,
        status=result.status.value,
        message=result.error_message,
        metrics=result.metrics,
        output_files=result.metrics.get("output_files") if result.metrics else None,
        created_at=result.created_at,
        completed_at=result.completed_at
    )


@app.get("/tasks", response_model=List[Dict[str, Any]])
async def list_tasks(limit: int = 10):
    """List recent tasks."""
    tasks = list(_running_tasks.items())[-limit:]
    return [
        {
            "task_id": task_id,
            "status": result.status.value,
            "created_at": result.created_at.isoformat() if result.created_at else None,
            "completed_at": result.completed_at.isoformat() if result.completed_at else None
        }
        for task_id, result in tasks
    ]


@app.get("/models/info")
async def get_model_info():
    """Get information about the loaded model."""
    try:
        engine = get_inference_engine()
        config = get_config()

        return {
            "model_path": config.model.model_path,
            "model_type": config.model.model_type,
            "patch_size": config.model.patch_size,
            "denoising_steps": config.model.denoising_steps,
            "batch_size": config.model.batch_size,
            "context_frames": config.model.context_frames,
            "use_residual": config.model.use_residual,
            "device": engine.device,
            "model_loaded": engine.model is not None
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app


if __name__ == "__main__":
    import uvicorn

    config = get_config()

    uvicorn.run(
        "main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload
    )