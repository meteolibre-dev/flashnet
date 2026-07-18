"""
FlashNet CLI for running weather forecast inference pipeline.

Usage:
    python main.py --mode=cli --date="2026-01-12 13:40:00"
    python main.py --mode=web  # FastAPI server for health checks
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Disable HDF5 file locking
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["HDF5_PLUGIN_PATH"] = ""


def setup_gcp_client():
    """Initialize GCP client with credentials from environment."""
    from backend.gcp_client import GCPStorageClient
    from backend.config import get_config

    config = get_config()
    return GCPStorageClient(config.gcp)


def download_latest_h5(gcs_client, pattern: str = None) -> str:
    """Download the latest H5 file from GCP bucket.

    Args:
        gcs_client: GCP storage client
        pattern: Optional pattern to filter files

    Returns:
        Local path to downloaded file
    """
    from backend.config import get_config

    config = get_config()
    cache_dir = Path(config.cache.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if pattern:
        files = gcs_client.get_file_by_pattern(pattern)
        if not files:
            raise FileNotFoundError(f"No files found matching pattern: {pattern}")
        source_file = files[0]
    else:
        source_file = gcs_client.get_latest_file()
        if not source_file:
            raise FileNotFoundError("No files found in source bucket")

    local_path = cache_dir / source_file.name
    gcs_client.download_file(source_file.gcs_path, str(local_path))

    logger.info(f"Downloaded {source_file.name} to {local_path}")
    return str(local_path)


def run_inference(data_path: str, gcs_client=None) -> str:
    """Run inference on downloaded H5 file, uploading each TIFF as it is written.

    Args:
        data_path: Path to input H5 file
        gcs_client: Optional GCPStorageClient used to upload files concurrently
                    with GPU inference.

    Returns:
        Path to output directory
    """
    from backend.inference_engine import InferenceEngine
    from backend.config import get_config

    config = get_config()
    output_dir = Path(config.cache.cache_dir) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("config model" + str(config.model))

    engine = InferenceEngine(
        model_path=config.model.model_path,
        model_type=config.model.model_type,
        patch_size=config.model.patch_size,
        denoising_steps=config.model.denoising_steps,
        batch_size=config.model.batch_size,
        context_frames=config.model.context_frames,
        use_residual=config.model.use_residual,
        interpolation=config.model.interpolation,
        poly_power=config.model.poly_power,
        noise_rho=config.model.noise_rho,
        sampler=config.model.sampler,
        sde_eps=config.model.sde_eps,
        sde_eps_schedule=config.model.sde_eps_schedule,
        bridge_ode_t_eps=config.model.bridge_ode_t_eps,
        bridge_sigma=config.model.bridge_sigma,
        bridge_sigma_min=config.model.bridge_sigma_min,
        inference_seed=config.model.inference_seed,
    )

    upload_fn = None
    if gcs_client is not None:
        # Extract datetime subfolder from H5 filename (e.g. "2026-04-11_08-20_full.h5" → "2026-04-11_08-20")
        h5_name = Path(data_path).name
        h5_datetime_str = h5_name.split("_full.h5")[0]  # e.g. "2026-04-11_08-20"
        # Use the H5 file's own date for the date folder so all forecasts
        # from a single run land in the same bucket, even when predictions
        # cross midnight UTC.
        h5_date_prefix = h5_datetime_str.split("_")[0]  # e.g. "2026-04-11"

        def upload_fn(filepath: str):
            try:
                filepath = Path(filepath)
                # Preserve the debug_endpoints/ subdirectory in the blob path
                # so debug snapshots land in their own folder in the bucket.
                subdir = "debug_endpoints/" if filepath.parent.name == "debug_endpoints" else ""
                dest_blob = f"{config.gcp.dest_prefix}/{h5_date_prefix}/{h5_datetime_str}/{subdir}{filepath.name}"
                gcs_client.upload_file(str(filepath), dest_blob)
            except Exception:
                logger.exception(f"Failed to upload {filepath}")

    # Parse debug endpoint t-values (comma-separated string -> list[float])
    debug_t_values = None
    if config.model.debug_endpoint_t_values:
        debug_t_values = [
            float(x.strip()) for x in config.model.debug_endpoint_t_values.split(",") if x.strip()
        ]
        logger.info(f"Debug endpoint snapshots enabled at t={debug_t_values}")

    result = engine.run_inference(
        data_path=data_path,
        output_dir=str(output_dir),
        forecast_steps=config.model.forecast_steps,
        nb_forecast=config.model.nb_forecast,
        upload_fn=upload_fn,
        debug_endpoint_t_values=debug_t_values,
    )

    if result.status.value != "completed":
        raise RuntimeError(f"Inference failed: {result.error_message}")

    logger.info(f"Inference completed. Output: {output_dir}")
    return str(output_dir)


def upload_results(output_dir: str, gcs_client, h5_path: str = None):
    """Upload inference results to GCP bucket.

    Args:
        output_dir: Directory containing output files
        gcs_client: GCP storage client
        h5_path: Path to the source H5 file; its datetime is used as a subfolder
    """
    from backend.config import get_config

    config = get_config()
    output_path = Path(output_dir)

    # Extract H5 datetime subfolder (e.g. "2026-04-11_08-20_full.h5" → "2026-04-11_08-20")
    h5_datetime_str = None
    if h5_path:
        h5_name = Path(h5_path).name
        h5_datetime_str = h5_name.split("_full.h5")[0]

    tiff_files = list(output_path.glob("*.tiff"))
    if not tiff_files:
        raise FileNotFoundError("No TIFF files found in output directory")

    # Use the H5 file's own date for the date folder so all forecasts
    # from a single run land in the same bucket, even when predictions
    # cross midnight UTC.
    if h5_datetime_str:
        date_prefix = h5_datetime_str.split("_")[0]  # e.g. "2026-04-11"
        dest_prefix = f"{config.gcp.dest_prefix}/{date_prefix}/{h5_datetime_str}"
    else:
        # Fallback: derive date from the first TIFF filename
        first_file = tiff_files[0]
        try:
            date_str = first_file.stem.split("_")[1]
            date_prefix = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            dest_prefix = f"{config.gcp.dest_prefix}/{date_prefix}"
        except (IndexError, ValueError) as e:
            raise ValueError(f"Could not parse date from filename {first_file.name}: {e}")

    uploaded = []
    for filepath in tiff_files:
        dest_blob_name = f"{dest_prefix}/{filepath.name}"
        gcs_client.upload_file(str(filepath), dest_blob_name)
        uploaded.append(filepath.name)
        logger.info(f"Uploaded {filepath.name} to {config.gcp.dest_bucket}/{dest_blob_name}")

    return uploaded


def process_date_pipeline(target_date: datetime) -> dict:
    """Run the full download-inference-upload pipeline.

    Args:
        target_date: Datetime for the forecast

    Returns:
        Dictionary with pipeline results
    """
    start_time = datetime.now()
    logger.info(f"Starting pipeline for {target_date}")

    try:
        # Setup GCP client
        gcs_client = setup_gcp_client()

        # Download latest H5 file (search whole day, get most recent)
        pattern = target_date.strftime("%Y-%m-%d") + "_*.h5"
        try:
            data_path = download_latest_h5(gcs_client, pattern=pattern)
        except FileNotFoundError:
            logger.warning(
                f"No H5 file found for date {target_date.date()}, "
                "falling back to latest available file in bucket"
            )
            data_path = download_latest_h5(gcs_client)

        # Run inference — each TIFF is uploaded to GCS as soon as it is written
        output_dir = run_inference(data_path, gcs_client=gcs_client)
        uploaded = list(Path(output_dir).glob("*.tiff"))
        uploaded = [f.name for f in uploaded]

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        return {
            "status": "success",
            "target_date": target_date.isoformat(),
            "input_file": os.path.basename(data_path),
            "output_files": uploaded,
            "duration_seconds": duration,
            "completed_at": end_time.isoformat()
        }

    except Exception as e:
        logger.exception("Pipeline failed")
        return {
            "status": "failed",
            "target_date": target_date.isoformat(),
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        }


def run_cli(target_date: datetime):
    """Run pipeline in CLI mode."""
    result = process_date_pipeline(target_date)

    if result["status"] == "success":
        print(f"\n✓ Pipeline completed successfully!")
        print(f"  Duration: {result['duration_seconds']:.1f}s")
        print(f"  Output files: {len(result['output_files'])}")
        for f in result["output_files"]:
            print(f"    - {f}")
    else:
        print(f"\n✗ Pipeline failed: {result['error']}")
        sys.exit(1)


# =============================================================================
# FastAPI Web Server (Simple health checks only)
# =============================================================================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn


app = FastAPI(title="FlashNet Backend")


class ProcessRequest(BaseModel):
    date: Optional[str] = None  # Format YYYY-MM-DD HH:MM:SS


@app.get("/")
def health_check():
    return {"status": "ok", "service": "flashnet-backend"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/pipeline/run")
def run_pipeline(request: ProcessRequest):
    """Trigger the pipeline via HTTP."""
    try:
        if request.date:
            target_date = datetime.strptime(request.date, "%Y-%m-%d %H:%M:%S")
        else:
            target_date = datetime.now() - timedelta(hours=1)

        result = process_date_pipeline(target_date)

        if result["status"] == "failed":
            raise HTTPException(status_code=500, detail=result["error"])

        return result

    except Exception as e:
        logger.exception("Pipeline HTTP endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="FlashNet Weather Forecast Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run pipeline for specific date
  python main.py --mode=cli --date="2026-01-12 13:40:00"

  # Run pipeline for latest available data
  python main.py --mode=cli

  # Run web server (for Cloud Run with health checks)
  python main.py --mode=web

  # Run with custom model path
  python main.py --mode=cli --model-path=/path/to/model.safetensors
        """
    )

    parser.add_argument(
        "--mode",
        choices=["web", "cli"],
        default="cli",
        help="Run mode: web (FastAPI server) or cli (single pipeline run)"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Target date in YYYY-MM-DD HH:MM:SS format (default: auto-detect latest)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to model weights (default: from config)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for web server (default: 8080)"
    )

    args = parser.parse_args()

    if args.mode == "cli":
        if args.date:
            target_date = datetime.strptime(args.date, "%Y-%m-%d %H:%M:%S")
        else:
            target_date = datetime.now() - timedelta(hours=1)

        if args.model_path:
            os.environ["MODEL_PATH"] = args.model_path

        run_cli(target_date)

    else:
        # Web server mode for Cloud Run
        port = int(os.environ.get("PORT", args.port))
        host = os.environ.get("HOST", "0.0.0.0")

        logger.info(f"Starting web server on {host}:{port}")
        uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()