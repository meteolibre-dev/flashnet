"""
Inference engine that encapsulates the tiled inference logic.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
import h5py
import pyproj
from suncalc import get_position
from tqdm.auto import tqdm

# Add project root to sys.path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from meteolibre_model.models.jit3d_dual import DualJiT3D
from meteolibre_model.models.unet3d_film_dual import DualUNet3DFiLM
from meteolibre_model.diffusion.rectified_flow_lightning_shortcut_xpred import (
    normalize,
    denormalize,
    CLIP_MIN,
)
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


class InferenceStatus(Enum):
    """Status of an inference operation."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class InferenceResult:
    """Result of an inference operation."""
    status: InferenceStatus
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class InferenceEngine:
    """Engine for running weather forecast inference."""

    def __init__(
        self,
        model_path: str,
        model_type: str = "jit",
        patch_size: int = 128,
        denoising_steps: int = 128,
        batch_size: int = 64,
        context_frames: int = 4,
        use_residual: bool = True,
        device: Optional[str] = None
    ):
        """Initialize the inference engine.

        Args:
            model_path: Path to the model weights
            model_type: Type of model ("jit" or "unet")
            patch_size: Size of patches for tiled inference
            denoising_steps: Number of denoising steps
            batch_size: Batch size for processing patches
            context_frames: Number of context frames
            use_residual: Whether to use residual connections
            device: Device to run inference on (auto-detected if None)
        """
        self.model_path = model_path
        self.model_type = model_type
        self.patch_size = patch_size
        self.denoising_steps = denoising_steps
        self.batch_size = batch_size
        self.context_frames = context_frames
        self.use_residual = use_residual

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: Optional[torch.nn.Module] = None
        self.params: Dict[str, Any] = {}

        self._load_config()
        self._load_model()

    def _load_config(self) -> None:
        """Load model configuration."""
        config_path = os.path.join(project_root, "meteolibre_model/config/configs.yml")
        import yaml

        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.params = config["model_v15_mtg_world_lightning_shortcut"]

    def _download_model_from_gcs(self, gcs_path: str, local_path: str) -> None:
        """Download model from Google Cloud Storage.

        Args:
            gcs_path: GCS path (e.g., gs://bucket/path/model.safetensors)
            local_path: Local path to save the model
        """
        from google.cloud import storage
        from google.oauth2 import service_account
        import logging

        logger.info(f"Downloading model from {gcs_path} to {local_path}")

        # Parse GCS path
        if not gcs_path.startswith("gs://"):
            raise ValueError(f"Invalid GCS path: {gcs_path}")

        path_parts = gcs_path[5:].split("/")
        bucket_name = path_parts[0]
        blob_name = "/".join(path_parts[1:])

        # Get credentials
        credentials = None
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            credentials = service_account.Credentials.from_service_account_file(
                os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            )

        # Download from GCS
        client = storage.Client(credentials=credentials)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Ensure parent directory exists
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        blob.download_to_filename(local_path)
        logger.info(f"Model downloaded successfully to {local_path}")

    def _load_model(self) -> None:
        """Load the model weights."""
        model_gcs_path = os.environ.get("MODEL_GCS_PATH", "")

        # Check if model needs to be downloaded from GCS
        if model_gcs_path and not os.path.exists(self.model_path):
            self._download_model_from_gcs(model_gcs_path, self.model_path)

        logger.info(f"Loading model from {self.model_path}")

        torch.set_float32_matmul_precision('medium')

        model_params = self.params["model"]

        if self.model_type == "jit":
            self.model = DualJiT3D(**model_params)
            self.model = torch.compile(self.model)
        else:
            self.model = DualUNet3DFiLM(**model_params)

        if os.path.exists(self.model_path):
            loaded_state_dict = load_file(self.model_path)
            self.model.load_state_dict(loaded_state_dict)
            logger.info(f"Loaded model weights from {self.model_path}")
        else:
            logger.warning(f"Model weights not found at {self.model_path}. Using randomly initialized model.")

        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded successfully on {self.device}")

    def _extract_patch(self, image: torch.Tensor, x: int, y: int, patch_size: int) -> torch.Tensor:
        """Extract a patch from an image."""
        return image[..., y : y + patch_size, x : x + patch_size]

    def _get_gaussian_weights(self, patch_size: int, sigma_scale: float = 0.3) -> torch.Tensor:
        """Generate a 2D Gaussian weight mask."""
        x = torch.linspace(-(patch_size - 1) / 2, (patch_size - 1) / 2, patch_size, device=self.device)
        sigma = sigma_scale * patch_size
        w_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        w_2d = w_1d.unsqueeze(1) * w_1d.unsqueeze(0)
        w_2d = w_2d / w_2d.max()
        return w_2d

    @torch.no_grad()
    def tiled_inference(
        self,
        initial_context: torch.Tensor,
        forecast_steps: int = 18,
        nb_forecast: int = 3,
        date: Optional[datetime] = None
    ) -> torch.Tensor:
        """Run tiled inference for weather forecasting.

        Args:
            initial_context: Initial context tensor (B, C, T_ctx, H, W)
            forecast_steps: Number of forecast steps to generate
            nb_forecast: Number of frames to forecast per model call
            date: Date for the first forecast step

        Returns:
            Generated forecast tensor.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        self.model.eval()
        self.model.to(self.device)

        _, C, T_ctx, H_big, W_big = initial_context.shape
        x_t_full_res = torch.randn(1, C, nb_forecast, H_big, W_big, device=self.device)

        patch_weights = self._get_gaussian_weights(self.patch_size)
        patch_weights = patch_weights.view(1, 1, 1, self.patch_size, self.patch_size)

        # Create patch coordinates
        y_starts1 = list(range(0, H_big - self.patch_size + 1, self.patch_size))
        if (H_big - self.patch_size) % self.patch_size != 0:
            y_starts1.append(H_big - self.patch_size)
        x_starts1 = list(range(0, W_big - self.patch_size + 1, self.patch_size))
        if (W_big - self.patch_size) % self.patch_size != 0:
            x_starts1.append(W_big - self.patch_size)
        patch_coords1 = [(x, y) for y in y_starts1 for x in x_starts1]

        shift = self.patch_size // 2
        y_starts2 = list(range(shift, H_big - self.patch_size + 1, self.patch_size))
        if (H_big - self.patch_size - shift) % self.patch_size != 0 and H_big - self.patch_size > shift:
            y_starts2.append(H_big - self.patch_size)
        x_starts2 = list(range(shift, W_big - self.patch_size + 1, self.patch_size))
        if (W_big - self.patch_size - shift) % self.patch_size != 0 and W_big - self.patch_size > shift:
            x_starts2.append(W_big - self.patch_size)
        patch_coords2 = [(x, y) for y in y_starts2 for x in x_starts2]
        patch_coords = patch_coords1 + patch_coords2

        # Get metadata from HDF5 if available
        epsg = getattr(initial_context, 'epsg', 4326)
        transform = getattr(initial_context, 'transform', [1, 0, 0, 0, 1, 0])

        if epsg != 4326:
            transformer = pyproj.Transformer.from_crs(
                f"EPSG:{epsg}", "EPSG:4326", always_xy=True
            )
        else:
            transformer = None

        c_sat = getattr(initial_context, 'c_sat', 16)
        c_lightning = getattr(initial_context, 'c_lightning', 1)

        d_const = 1.0 / self.denoising_steps
        all_forecasts = []

        current_step = 0
        current_high_res_context = initial_context

        while current_step < forecast_steps:
            remaining = forecast_steps - current_step
            this_nb = min(nb_forecast, remaining)

            if date:
                prediction_date = date + timedelta(minutes=10 * (current_step + 1))
            else:
                prediction_date = datetime.now()

            logger.info(f"Generating forecast step {current_step + 1}/{forecast_steps}")

            aggregated_x_pred = torch.zeros(1, C, this_nb, H_big, W_big, device=self.device)
            weights_sum = torch.zeros(1, 1, this_nb, H_big, W_big, device=self.device)

            for i in tqdm(range(self.denoising_steps), desc="Denoising"):
                t_val = 1.0 - i * d_const
                t_batch_val = torch.full((1,), t_val, device=self.device)
                d_batch_val = torch.full((1,), d_const, device=self.device)

                for i_batch in range(0, len(patch_coords), self.batch_size):
                    coords_batch = patch_coords[i_batch : i_batch + self.batch_size]
                    patch_x_t_batch, patch_context_batch, context_global_batch = [], [], []
                    pixel_xs = [x + self.patch_size // 2 for x, y in coords_batch]
                    pixel_ys = [y + self.patch_size // 2 for x, y in coords_batch]
                    lons, lats = [], []

                    for j in range(len(coords_batch)):
                        px = pixel_xs[j]
                        py = pixel_ys[j]
                        x_crs = transform[0] * px + transform[1] * py + transform[2]
                        y_crs = transform[3] * px + transform[4] * py + transform[5]

                        if transformer:
                            lon, lat = transformer.transform(x_crs, y_crs)
                        else:
                            lon, lat = x_crs, y_crs

                        lons.append(lon)
                        lats.append(lat)

                    for j, (x_start, y_start) in enumerate(coords_batch):
                        patch_x_t = self._extract_patch(
                            x_t_full_res, x_start, y_start, self.patch_size
                        )
                        patch_context = self._extract_patch(
                            current_high_res_context, x_start, y_start, self.patch_size
                        )

                        result = get_position(prediction_date, lons[j], lats[j])
                        date_noon = prediction_date.replace(hour=12, minute=0, second=0, microsecond=0)
                        result_noon = get_position(date_noon, lons[j], lats[j])

                        spatial_position = torch.tensor(
                            [result["azimuth"], result["altitude"], result_noon["altitude"], lats[j] / 10.0],
                            device=self.device,
                        )

                        context_global = torch.cat(
                            [
                                spatial_position.unsqueeze(0),
                                t_batch_val.unsqueeze(-1),
                                d_batch_val.unsqueeze(-1),
                            ],
                            dim=1,
                        )

                        patch_x_t_batch.append(patch_x_t)
                        patch_context_batch.append(patch_context)
                        context_global_batch.append(context_global)

                    model_input = torch.cat(
                        [
                            torch.cat(patch_context_batch, dim=0),
                            torch.cat(patch_x_t_batch, dim=0),
                        ],
                        dim=2,
                    )

                    model_input_sat = model_input[:, :c_sat]
                    model_input_lightning = model_input[:, c_sat : (c_sat + c_lightning)]

                    sat_pred_batch, lightning_pred_batch = self.model(
                        model_input_sat.float(),
                        model_input_lightning.float(),
                        torch.cat(context_global_batch, dim=0).float(),
                    )

                    x_pred_batch = torch.cat([sat_pred_batch, lightning_pred_batch], dim=1)[
                        :, :, self.context_frames:, :, :
                    ]

                    for j, (x_start, y_start) in enumerate(coords_batch):
                        aggregated_x_pred[
                            ...,
                            y_start : y_start + self.patch_size,
                            x_start : x_start + self.patch_size,
                        ] += x_pred_batch[j : j + 1] * patch_weights

                        weights_sum[
                            ...,
                            y_start : y_start + self.patch_size,
                            x_start : x_start + self.patch_size,
                        ] += patch_weights

                weights_sum[weights_sum == 0] = 1.0
                averaged_x_pred = aggregated_x_pred / weights_sum
                s_theta = (x_t_full_res - averaged_x_pred) / t_val
                x_t_full_res = x_t_full_res - s_theta * d_const
                x_t_full_res = x_t_full_res.clamp(-7, 7)

            last_context_frame = current_high_res_context[:, :, -1:, :, :]
            if self.use_residual:
                x_t_full_res[:, :, 0:1, :, :] += last_context_frame

            mask = (last_context_frame == CLIP_MIN).expand(-1, -1, this_nb, -1, -1)
            expanded_last = last_context_frame.expand(-1, -1, this_nb, -1, -1)
            x_t_full_res = torch.where(mask, expanded_last, x_t_full_res)

            all_forecasts.append(x_t_full_res.cpu())

            # Update context for next autoregressive step
            T_ctx = self.context_frames
            if this_nb >= T_ctx:
                current_high_res_context = x_t_full_res[:, :, -T_ctx:, :, :].to(self.device)
            else:
                tail = current_high_res_context[:, :, this_nb:, :, :]
                current_high_res_context = torch.cat(
                    [tail, x_t_full_res[:, :, :this_nb, :, :]], dim=2
                ).to(self.device)

            x_t_full_res = torch.randn(1, C, nb_forecast, H_big, W_big, device=self.device)
            current_step += this_nb

        # Concatenate all forecasts
        final_forecast = torch.cat(all_forecasts, dim=2)

        return final_forecast

    def run_inference(
        self,
        data_path: str,
        output_dir: str,
        forecast_steps: int = 18,
        nb_forecast: int = 3
    ) -> InferenceResult:
        """Run full inference pipeline.

        Args:
            data_path: Path to input HDF5 file
            output_dir: Directory to save outputs
            forecast_steps: Number of forecast steps
            nb_forecast: Frames per forecast batch

        Returns:
            InferenceResult with status and output information.
        """
        start_time = datetime.now()
        result = InferenceResult(
            status=InferenceStatus.RUNNING,
            created_at=start_time
        )

        try:
            if not os.path.exists(data_path):
                raise ValueError(f"Data file {data_path} not found")

            os.makedirs(output_dir, exist_ok=True)

            with h5py.File(data_path, "r") as hf:
                sat_data = hf["sat_data"][:]
                lightning_data = hf["lightning_data"][:]
                num_frames = hf.attrs["num_frames"]
                target_H = hf.attrs["target_height"]
                target_W = hf.attrs["target_width"]
                transform = hf.attrs["transform"]
                epsg = hf.attrs["epsg"]
                elevation_data = hf["elevation_data"][:]
                c_sat = hf.attrs["num_sat_channels"] + 1
                c_lightning = hf.attrs["num_lightning_channels"]

            if num_frames < self.context_frames:
                raise ValueError(
                    f"Not enough frames. Need {self.context_frames}, found {num_frames}"
                )

            # Parse date from filename
            filename = os.path.basename(data_path)
            date_str = filename.split("_full.h5")[0]
            date_part, time_part = date_str.split("_")[0], date_str.split("_")[1]
            year, month, day = map(int, date_part.split("-"))
            hour, minute = map(int, time_part.split("-"))
            initial_date = datetime(year, month, day, hour, minute) - timedelta(minutes=18 * 10)

            # Prepare initial context
            initial_frames = []
            for i in range(self.context_frames):
                sat_frame = sat_data[i]
                lightning_frame = lightning_data[i]
                elev_frame = elevation_data[None, :, :]
                elev_frame = np.where(elev_frame < 0, -100, elev_frame)

                sat_elev_frame = np.concatenate([sat_frame, elev_frame], axis=0)
                frame = np.concatenate([sat_elev_frame, lightning_frame], axis=0)[None, ...]
                initial_frames.append(frame)

            current_high_res_context = np.stack(initial_frames, axis=2)
            current_high_res_context = (
                torch.from_numpy(current_high_res_context).float().to(self.device)
            )

            sat_data_tensor = current_high_res_context[:, :c_sat]
            lightning_data_tensor = current_high_res_context[:, c_sat:]
            sat_data_tensor, lightning_data_tensor = normalize(sat_data_tensor, lightning_data_tensor, self.device)
            current_high_res_context = torch.cat([sat_data_tensor, lightning_data_tensor], dim=1)

            # Store metadata for inference
            current_high_res_context.epsg = epsg
            current_high_res_context.transform = transform
            current_high_res_context.c_sat = c_sat
            current_high_res_context.c_lightning = c_lightning

            # Run inference
            generated_forecast = self.tiled_inference(
                initial_context=current_high_res_context,
                forecast_steps=forecast_steps,
                nb_forecast=nb_forecast,
                date=initial_date
            )

            # Save results
            generated_norm = generated_forecast.to(self.device)
            sat_generated = generated_norm[:, :c_sat]
            lightning_generated = generated_norm[:, c_sat:]
            sat_denorm, lightning_denorm = denormalize(sat_generated, lightning_generated, self.device)

            output_files = []
            for k in range(generated_forecast.shape[2]):
                sat_frame = sat_denorm[:, :, k, :, :]
                lightning_frame = lightning_denorm[:, :, k, :, :]
                pred_date = initial_date + timedelta(minutes=10 * (k + 1))
                filename = f"forecast_{pred_date.strftime('%Y%m%d%H%M')}.npz"
                output_filepath = os.path.join(output_dir, filename)

                np.savez_compressed(
                    output_filepath,
                    sat_forecast=sat_frame.squeeze(0).cpu().numpy(),
                    lightning_forecast=lightning_frame.squeeze(0).cpu().numpy(),
                )

                output_files.append(output_filepath)
                logger.info(f"Saved forecast to {output_filepath}")

            result.status = InferenceStatus.COMPLETED
            result.output_path = output_dir
            result.completed_at = datetime.now()
            result.metrics = {
                "num_frames": generated_forecast.shape[2],
                "output_files": len(output_files),
                "duration_seconds": (result.completed_at - start_time).total_seconds()
            }

        except Exception as e:
            logger.exception("Inference failed")
            result.status = InferenceStatus.FAILED
            result.error_message = str(e)
            result.completed_at = datetime.now()

        return result

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.model:
            del self.model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None