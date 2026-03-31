"""
Backend configuration for weather forecasting inference service.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from datetime import timedelta
from dotenv import load_dotenv

_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


class GCPConfig(BaseModel):
    """Google Cloud Platform configuration."""
    source_bucket: str = "eumetsat_mtg_preprocess"
    source_prefix: str = "inference_h5"
    dest_bucket: str = "inference_result"
    dest_prefix: str = "forecasts"
    credentials_path: Optional[str] = None
    project_id: Optional[str] = None


class ModelConfig(BaseModel):
    """Model configuration."""
    model_path: str = "/tmp/flashnet_cache/model.safetensors"
    model_gcs_path: str = "gs://eumetsat_mtg_preprocess/assets/model_v16_mtg_europe_lightning_shortcut_radar_e144.safetensors"
    model_type: str = "jit"
    patch_size: int = 128
    denoising_steps: int = 128
    batch_size: int = 64
    forecast_steps: int = 18
    nb_forecast: int = 3
    context_frames: int = 4
    use_residual: bool = False
    parametrization: str = "standard"
    interpolation: str = "linear"
    quantization: str = "none"  # "none" | "fp8_weight_only" | "fp8_dynamic" | "int8_weight_only" | "int8_dynamic"


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    reload: bool = False


class CacheConfig(BaseModel):
    """Cache configuration."""
    cache_dir: str = "/tmp/flashnet_cache"
    max_cache_size_gb: float = 10.0
    cache_expiry_hours: int = 24


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None


class Config(BaseModel):
    """Main configuration class."""
    gcp: GCPConfig = GCPConfig()
    model: ModelConfig = ModelConfig()
    server: ServerConfig = ServerConfig()
    cache: CacheConfig = CacheConfig()
    logging: LoggingConfig = LoggingConfig()

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        return cls(
            gcp=GCPConfig(
                source_bucket=os.getenv("GCP_SOURCE_BUCKET", "eumetsat_mtg_preprocess"),
                source_prefix=os.getenv("GCP_SOURCE_PREFIX", "inference_h5"),
                dest_bucket=os.getenv("GCP_DEST_BUCKET", "inference_result"),
                dest_prefix=os.getenv("GCP_DEST_PREFIX", "forecasts"),
                credentials_path=os.getenv("GCP_CREDENTIALS_PATH"),
                project_id=os.getenv("GCP_PROJECT_ID"),
            ),
            model=ModelConfig(
                model_path=os.getenv("MODEL_PATH", "/tmp/flashnet_cache/model.safetensors"),
                model_gcs_path=os.getenv("MODEL_GCS_PATH", "gs://eumetsat_mtg_preprocess/assets/model_v16_mtg_europe_lightning_shortcut_radar_e144.safetensors"),
                model_type=os.getenv("MODEL_TYPE", "jit"),
                patch_size=int(os.getenv("PATCH_SIZE", 128)),
                denoising_steps=int(os.getenv("DENOISING_STEPS", 128)),
                batch_size=int(os.getenv("BATCH_SIZE", 64)),
                forecast_steps=int(os.getenv("FORECAST_STEPS", 18)),
                nb_forecast=int(os.getenv("NB_FORECAST", 3)),
                context_frames=int(os.getenv("CONTEXT_FRAMES", 4)),
                use_residual=os.getenv("USE_RESIDUAL", "False").lower() == "true",
                interpolation=os.getenv("INTERPOLATION", "linear"),
                quantization=os.getenv("QUANTIZATION", "none"),
            ),
            server=ServerConfig(
                host=os.getenv("SERVER_HOST", "0.0.0.0"),
                port=int(os.getenv("SERVER_PORT", 8080)),
                debug=os.getenv("SERVER_DEBUG", "False").lower() == "true",
                reload=os.getenv("SERVER_RELOAD", "False").lower() == "true",
            ),
            cache=CacheConfig(
                cache_dir=os.getenv("CACHE_DIR", "/tmp/flashnet_cache"),
                max_cache_size_gb=float(os.getenv("MAX_CACHE_SIZE_GB", 10.0)),
                cache_expiry_hours=int(os.getenv("CACHE_EXPIRY_HOURS", 24)),
            ),
            logging=LoggingConfig(
                level=os.getenv("LOG_LEVEL", "INFO"),
                log_file=os.getenv("LOG_FILE"),
            ),
        )


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def reset_config() -> None:
    """Reset the global configuration instance."""
    global _config
    _config = None