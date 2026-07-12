"""
Inference engine that encapsulates the tiled inference logic.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Generator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
import torch
import torch.nn.functional as F
import h5py
import pyproj
import rasterio
from suncalc import get_position
from tqdm.auto import tqdm

# COG conversion
try:
    from rio_cogeo.cogeo import cog_translate
    from rio_cogeo.profiles import cog_profiles
    COG_AVAILABLE = True
except ImportError:
    COG_AVAILABLE = False
    print("Warning: rio-cogeo not available. COG conversion will be skipped.")

# Add project root to sys.path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from meteolibre_model.models.jit3d_dual_v2 import DualJiT3D
from meteolibre_model.models.unet3d_film_dual import DualUNet3DFiLM
from meteolibre_model.diffusion.rectified_flow_lightning_shortcut_xpred_blur_v2 import (
    normalize,
    denormalize,
    CLIP_MIN,
    bridge_coeffs,
)
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


def convert_to_cog(input_path: str, delete_original: bool = True) -> str:
    """
    Convert a TIFF to Cloud Optimized GeoTIFF format in-place.

    Args:
        input_path: Path to input TIFF
        delete_original: Whether to delete the original non-COG file

    Returns:
        Path to the COG file (same as input_path if converted in-place)
    """
    if not COG_AVAILABLE:
        return input_path

    if not os.path.exists(input_path):
        logger.warning(f"Input file not found: {input_path}")
        return input_path

    import tempfile
    temp_dir = tempfile.gettempdir()
    temp_cog = os.path.join(temp_dir, f"temp_cog_{os.path.basename(input_path)}")

    try:
        logger.info(f"Converting to COG: {input_path}")

        # Get image dimensions to choose optimal block size
        with rasterio.open(input_path) as src:
            height = src.height
            width = src.width

        # Choose block size that divides evenly into dimensions
        # Common factors: 256, 512, 1024
        # Find the largest power of 2 that divides both dimensions
        def get_optimal_block_size(dim):
            for size in [512, 256, 128, 64]:
                if dim % size == 0:
                    return size
            return 256  # fallback

        block_size = min(get_optimal_block_size(height), get_optimal_block_size(width), 512)
        
        logger.info(f"Using block size {block_size}x{block_size} for {width}x{height} image")

        # Use the deflate profile (already has dtype, compress, tiled, blockxsize, blockysize)
        output_profile = cog_profiles.get("deflate")
        output_profile.update({
            "blockxsize": block_size,
            "blockysize": block_size,
        })

        # Additional GDAL config options for better compatibility
        config = {
            "GDAL_NUM_THREADS": "ALL_CPUS",
            # Ensure proper block alignment
            "GDAL_TIFF_INTERNAL_MASK": "YES",
            # Use older compression for better compatibility
            "COMPRESS_OVERVIEW": "DEFLATE",
        }

        cog_translate(
            input_path,
            temp_cog,
            output_profile,
            config=config,
            resampling="bilinear",
            quiet=True
        )

        # Verify the COG was created properly
        if os.path.exists(temp_cog):
            with rasterio.open(temp_cog) as cog_src:
                # Check that it's properly tiled
                if cog_src.profile.get('tiled', False):
                    logger.info(f"COG verified: {cog_src.width}x{cog_src.height}, block={cog_src.block_shapes}")
                else:
                    logger.warning(f"COG may not be properly tiled: {cog_src.profile}")

        # Replace original with COG
        if delete_original:
            os.remove(input_path)
            os.rename(temp_cog, input_path)
            logger.info(f"COG created: {input_path}")
        else:
            os.rename(temp_cog, input_path.replace('.tiff', '_cog.tiff'))
            logger.info(f"COG created: {input_path.replace('.tiff', '_cog.tiff')}")

        return input_path

    except Exception as e:
        logger.error(f"Error converting to COG: {e}")
        if os.path.exists(temp_cog):
            os.remove(temp_cog)
        return input_path


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
        use_residual: bool = False,
        interpolation: str = "linear",
        bridge_sigma: float = 0.3,
        bridge_sigma_min: float = 1e-3,
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
            interpolation: "linear" (rectified flow), "polynomial", or "bridge"
                (Brownian-bridge flow matching).
            bridge_sigma: Bridge noise scale (used only if interpolation="bridge").
            bridge_sigma_min: Bridge endpoint variance floor (used only if
                interpolation="bridge").
            device: Device to run inference on (auto-detected if None)
        """
        self.model_path = model_path
        self.model_type = model_type
        self.patch_size = patch_size
        self.denoising_steps = denoising_steps
        self.batch_size = batch_size
        self.context_frames = context_frames
        self.use_residual = use_residual
        self.interpolation = interpolation
        self.bridge_sigma = bridge_sigma
        self.bridge_sigma_min = bridge_sigma_min

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

        self.params = config["model_v26_mtg_europe_lightning_radar_shortcut"]

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
            #self.model = torch.compile(self.model)
        else:
            self.model = DualUNet3DFiLM(**model_params)

        if os.path.exists(self.model_path):
            loaded_state_dict = load_file(self.model_path)
            self.model.load_state_dict(loaded_state_dict)
            logger.info(f"Loaded model weights from {self.model_path}")
        else:
            logger.warning(f"Model weights not found at {self.model_path}. Using randomly initialized model.")

        self.model.to(self.device)

        # if new save setup
        self.model = torch.compile(self.model)

        self.model.eval()

        logger.info(f"Model loaded successfully on {self.device}")

    def _extract_patch(self, image: torch.Tensor, x: int, y: int, patch_size: int) -> torch.Tensor:
        """Extract a patch from an image."""
        return image[..., y : y + patch_size, x : x + patch_size]

    def _fill_bad_pixels(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """Fill pixels <= threshold with average of valid neighbors."""
        # x: (B, C, H, W)
        with torch.no_grad():
            mask = x <= threshold
            if not mask.any():
                return x
            
            # Kernel for sum of neighbors (exclude center)
            # 3x3 kernel with center 0, others 1
            channels = x.shape[1]
            kernel = torch.ones(channels, 1, 3, 3, device=x.device, dtype=x.dtype)
            kernel[..., 1, 1] = 0
            
            # Valid pixels
            valid_x = torch.where(mask, torch.tensor(0.0, device=x.device, dtype=x.dtype), x)
            valid_mask = (~mask).to(x.dtype)
            
            # Sum of valid neighbors
            sum_neighbors = F.conv2d(valid_x, kernel, padding=1, groups=channels)
            
            # Count of valid neighbors
            count_neighbors = F.conv2d(valid_mask, kernel, padding=1, groups=channels)
            
            # Avoid division by zero
            avg_neighbors = sum_neighbors / (count_neighbors + 1e-6)
            
            # Replace bad pixels where we have valid neighbors
            fill_mask = mask & (count_neighbors > 0)
            x = torch.where(fill_mask, avg_neighbors, x)
            
            return x

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
        date: Optional[datetime] = None,
        output_dir: Optional[str] = None,
        c_sat: int = 18,
        c_lightning: int = 2,
        c_radar: int = 1,
        debug_endpoint_t_values: Optional[list[float]] = None,
        debug_upload_fn: Optional[Callable[[str], None]] = None,
        debug_geo_transform=None,
        debug_crs: str = "EPSG:4326",
        debug_crop_range: Optional[slice] = None,
    ) -> Generator[tuple[torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        """Run tiled inference for weather forecasting.

        Args:
            initial_context: Initial context tensor (B, C, T_ctx, H, W)
            forecast_steps: Number of forecast steps to generate
            nb_forecast: Number of frames to forecast per model call
            date: Date for the first forecast step
            output_dir: Directory to save forecast files
            c_sat: Number of satellite channels (including elevation and radar)
            c_lightning: Number of lightning channels
            c_radar: Number of radar channels

        Returns:
            Generator yielding (sat_batch, lightning_batch, radar_batch) CPU tensors
            after each autoregressive step. sat_batch: (1, 2, nb, H, W),
            lightning_batch: (1, 1, nb, H, W), radar_batch: (1, nb, H, W).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        print("interpolation :", self.interpolation)

        self.model.eval()
        self.model.to(self.device)

        _, C, T_ctx, H_big, W_big = initial_context.shape
        x_t_full_res = torch.randn(1, C, nb_forecast, H_big, W_big, device=self.device)
        # Bridge flow matching: keep the fixed noise endpoint (t=1 source). The
        # exact ODE step references it at every denoising step.
        x1_init = x_t_full_res.clone() if self.interpolation == "bridge" else None

        patch_weights = self._get_gaussian_weights(self.patch_size)
        patch_weights = patch_weights.view(1, 1, 1, self.patch_size, self.patch_size)

        # Create patch coordinates
        def get_starts(total, size, start_offset):
            starts = list(range(start_offset, total - size + 1, size))
            if not starts or starts[-1] != total - size:
                if total >= size:
                    starts.append(total - size)
            return starts

        shift = self.patch_size // 2

        y0 = get_starts(H_big, self.patch_size, 0)
        x0 = get_starts(W_big, self.patch_size, 0)
        yS = get_starts(H_big, self.patch_size, shift)
        xS = get_starts(W_big, self.patch_size, shift)

        # Base grids
        coords1 = [(x, y) for y in y0 for x in x0] # Grid 1 (0, 0)
        coords2 = [(x, y) for y in yS for x in xS] # Grid 2 (S, S)

        # Extra bands for top/left/bottom/right coverage (fixes 1-patch overlap at borders)
        extra_top = [(x, y0[0]) for x in xS]
        extra_left = [(x0[0], y) for y in yS]
        extra_bottom = [(x, y0[-1]) for x in xS]
        extra_right = [(x0[-1], y) for y in yS]

        patch_coords = list(set(coords1 + coords2 + extra_top + extra_left + extra_bottom + extra_right))

        # Get metadata from HDF5 if available
        epsg = getattr(initial_context, 'epsg', 4326)
        transform = getattr(initial_context, 'transform', [1, 0, 0, 0, 1, 0])

        if epsg != 4326:
            transformer = pyproj.Transformer.from_crs(
                f"EPSG:{epsg}", "EPSG:4326", always_xy=True
            )
        else:
            transformer = None

        c_sat = getattr(initial_context, 'c_sat', 18)
        c_lightning = getattr(initial_context, 'c_lightning', 1) + getattr(initial_context, 'cg_lightning', 1)

        c_radar = getattr(initial_context, 'c_radar', 1)

        half_steps = self.denoising_steps // 2
        current_step = 0
        current_high_res_context = initial_context

        # Debug: map each target t (0.9, 0.8, ...) to the closest denoising
        # step index so we can snapshot the endpoint prediction there.
        debug_step_map: dict[int, float] = {}
        if debug_endpoint_t_values:
            for tv in debug_endpoint_t_values:
                idx = round((1.0 - tv) * self.denoising_steps)
                idx = min(idx, self.denoising_steps - 1)
                debug_step_map[idx] = tv

        while current_step < forecast_steps:
            remaining = forecast_steps - current_step
            this_nb = min(nb_forecast, remaining)

            if date:
                prediction_date = date + timedelta(minutes=10 * (current_step + 1))
            else:
                prediction_date = datetime.now()

            logger.info(f"Generating forecast step {current_step + 1}/{forecast_steps}")


            for i in tqdm(range(self.denoising_steps), desc="Denoising"):
                torch.cuda.empty_cache()
                # Use bfloat16 for accumulation to save memory
                aggregated_x_pred = torch.zeros(
                    1, C, this_nb, H_big, W_big, device=self.device, dtype=torch.bfloat16
                )
                weights_sum = torch.zeros(
                    1, 1, this_nb, H_big, W_big, device=self.device, dtype=torch.bfloat16
                )

                # Uniform schedule: linearly space t from 1.0 to 0.0
                t_val = 1.0 - i / self.denoising_steps
                dt = 1.0 / self.denoising_steps

                t_batch_val = torch.full((1,), t_val, device=self.device)

                d_batch_val = torch.full((1,), 0, device=self.device)

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
                            [result["azimuth"], result["altitude"], result_noon["altitude"], lats[j] / 20.0],
                            device=self.device,
                        )

                        context_global = torch.cat(
                            [
                                spatial_position.unsqueeze(0),
                                d_batch_val.unsqueeze(-1),
                                t_batch_val.unsqueeze(-1)
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

                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        sat_pred_batch, lightning_pred_batch = self.model(
                            model_input_sat.float(),
                            model_input_lightning.float(),
                            torch.cat(context_global_batch, dim=0).float(),
                        )

                    del model_input, model_input_sat, model_input_lightning, context_global_batch, patch_x_t_batch, patch_context_batch

                    x_pred_batch = torch.cat([sat_pred_batch, lightning_pred_batch], dim=1)[
                        :, :, self.context_frames:, :, :
                    ].to(torch.bfloat16)
                    del sat_pred_batch, lightning_pred_batch

                    # Use bfloat16 for patch weights
                    pw = patch_weights.to(torch.bfloat16)

                    for j, (x_start, y_start) in enumerate(coords_batch):
                        # Aggregate the model prediction x_pred across overlapping
                        # patches. For RF this is equivalent to aggregating velocity
                        # (v is linear in x_pred); for the bridge it lets us use the
                        # exact closed-form ODE step instead of divergent Euler.
                        aggregated_x_pred[
                            ...,
                            y_start : y_start + self.patch_size,
                            x_start : x_start + self.patch_size,
                        ] += x_pred_batch[j : j + 1] * pw

                        weights_sum[
                            ...,
                            y_start : y_start + self.patch_size,
                            x_start : x_start + self.patch_size,
                        ] += pw
                    del x_pred_batch, pw

                weights_sum[weights_sum == 0] = 1.0
                # In-place average in bfloat16 to save memory
                aggregated_x_pred.div_(weights_sum)

                # Convert back to float32 for the update step
                averaged_x_pred = aggregated_x_pred.float()
                del aggregated_x_pred, weights_sum

                # Debug: snapshot the full-Europe endpoint prediction (x0-hat)
                # at the target t values during the first AR step.
                if (debug_endpoint_t_values is not None
                        and current_step == 0
                        and i in debug_step_map):
                    self._save_debug_endpoint(
                        averaged_x_pred,
                        debug_step_map[i],
                        c_sat,
                        output_dir or ".",
                        prediction_date,
                        debug_geo_transform,
                        debug_crs,
                        debug_crop_range or slice(None),
                        debug_upload_fn,
                    )

                if self.interpolation == "bridge":
                    # Exact closed-form bridge ODE step — absorbs the stiff c'/c
                    # mean-reversion into an O(1) c_next/c_t ratio. Naive Euler
                    # with the bridge velocity diverges near the data end
                    # (cp_over_c ~ sigma^2/(2 sigma_min^2) ~ 1e4, lambda*dt >> 2).
                    #   x_{t_next} = mu_next + (x_t - mu_t) * c_next/c_t
                    #   mu_t     = (1-t)      x_pred + t      x1_init
                    #   mu_next  = (1-t_next) x_pred + t_next x1_init
                    t_next = max(t_val - dt, 0.0)
                    c_t, _ = bridge_coeffs(t_val, self.bridge_sigma, self.bridge_sigma_min)
                    c_next, _ = bridge_coeffs(t_next, self.bridge_sigma, self.bridge_sigma_min)
                    ratio = (c_next / c_t).item()
                    mu_t = (1.0 - t_val) * averaged_x_pred + t_val * x1_init
                    mu_next = (1.0 - t_next) * averaged_x_pred + t_next * x1_init
                    del averaged_x_pred
                    # In-place: x_t = mu_next + (x_t - mu_t) * ratio
                    x_t_full_res.sub_(mu_t).mul_(ratio).add_(mu_next)
                    del mu_t, mu_next
                else:
                    # RF Euler step from the aggregated prediction (equivalent to
                    # averaging velocity, since v is linear in x_pred).
                    if self.interpolation == "polynomial":
                        averaged_velocity = (x_t_full_res - averaged_x_pred) / (2 * t_val + 1e-8)
                    else:  # linear
                        averaged_velocity = (x_t_full_res - averaged_x_pred) / t_val
                    del averaged_x_pred
                    x_t_full_res.sub_(averaged_velocity, alpha=dt)
                    del averaged_velocity

                x_t_full_res.clamp_(-7, 7)

            torch.cuda.empty_cache()

            last_context_frame = current_high_res_context[:, :, -1:, :, :]
            if self.use_residual:
                x_t_full_res[:, :, 0:1, :, :] += last_context_frame

            mask = (last_context_frame == CLIP_MIN).expand(-1, -1, this_nb, -1, -1)
            expanded_last = last_context_frame.expand(-1, -1, this_nb, -1, -1)
            x_t_full_res = torch.where(mask, expanded_last, x_t_full_res)

            sat_frame = x_t_full_res[:, :c_sat, :, :, :]
            lightning_frame = x_t_full_res[:, c_sat:, :, :, :]
            sat_denorm, lightning_denorm = denormalize(sat_frame, lightning_frame, self.device)
            # Radar is the last channel (index 17 = -1) of sat_denorm which includes elevation + radar
            radar_denorm = sat_denorm[:, -1, :, :]
            sat_batch_cpu = sat_denorm[:, [0, 11], :, :].cpu()
            lightning_batch_cpu = lightning_denorm.cpu()
            radar_batch_cpu = radar_denorm.cpu()
            del sat_denorm, lightning_denorm

            # Yield this step's results before updating context so the caller
            # can dispatch I/O (write + upload) concurrently with the next GPU step.
            yield sat_batch_cpu, lightning_batch_cpu, radar_batch_cpu

            # Update context for next autoregressive step
            T_ctx = self.context_frames
            if this_nb >= T_ctx:
                new_context = x_t_full_res[:, :, -T_ctx:, :, :].clone()
            else:
                tail = current_high_res_context[:, :, this_nb:, :, :]
                new_context = torch.cat(
                    [tail, x_t_full_res[:, :, :this_nb, :, :]], dim=2
                ).to(self.device)
            del current_high_res_context
            torch.cuda.empty_cache()
            current_high_res_context = new_context

            if x1_init is not None:
                del x1_init
            del x_t_full_res
            torch.cuda.empty_cache()
            x_t_full_res = torch.randn(1, C, nb_forecast, H_big, W_big, device=self.device)
            x1_init = x_t_full_res.clone() if self.interpolation == "bridge" else None
            current_step += this_nb

    def _save_debug_endpoint(
        self,
        x_pred: torch.Tensor,
        t_val: float,
        c_sat: int,
        output_dir: str,
        pred_date: datetime,
        geo_transform,
        crs: str,
        crop_range: slice,
        upload_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Save the model's endpoint prediction (x0-hat) at a given denoising t.

        Denormalizes the full-resolution aggregated x-prediction and writes TIFF
        files for the first forecast frame — useful for inspecting how the
        prediction evolves across the denoising trajectory (t=0.9 -> 0) and
        spotting artifacts from patch averaging. Files are optionally uploaded
        to the bucket via upload_fn.
        """
        debug_dir = os.path.join(output_dir, "debug_endpoints")
        os.makedirs(debug_dir, exist_ok=True)

        t_label = f"t{t_val:.1f}"

        # Denormalize a clone so we never touch the live inference tensor.
        sat_x = x_pred[:, :c_sat].clone()
        lightning_x = x_pred[:, c_sat:].clone()
        sat_denorm, lightning_denorm = denormalize(sat_x, lightning_x, self.device)
        radar_denorm = sat_denorm[:, -1]  # (1, nb, H, W) — radar is last sat channel

        # Extract first forecast frame (k=0) with the same channels as the
        # regular output: sat [0, 11], lightning, radar.
        sat_frame = sat_denorm[:, [0, 11], 0, :, :].cpu()       # (1, 2, H, W)
        lightning_frame = lightning_denorm[:, :, 0, :, :].cpu()  # (1, c_lightning, H, W)
        radar_frame = radar_denorm[:, 0:1, :, :].cpu()           # (1, 1, H, W)

        self._save_timestep_files(
            sat_frame,
            lightning_frame,
            radar_frame,
            debug_dir,
            pred_date,
            geo_transform,
            crs,
            crop_range,
            upload_fn,
            filename_prefix=f"endpoint_{t_label}",
            make_cog=False,  # skip COG for debug speed
        )
        logger.info(f"  [debug] saved endpoint prediction at {t_label}")

    def _save_timestep_files(
        self,
        sat_frame: torch.Tensor,
        lightning_frame: torch.Tensor,
        radar_frame: torch.Tensor,
        output_dir: str,
        pred_date: datetime,
        geo_transform,
        crs: str,
        crop_range: slice,
        upload_fn: Optional[Callable[[str], None]] = None,
        filename_prefix: str = "forecast",
        make_cog: bool = True,
    ) -> list:
        """Write TIFF files for one forecast timestep and optionally upload them.

        This is designed to run in a background thread so GPU inference can
        proceed concurrently.
        """
        base_filename = f"{filename_prefix}_{pred_date.strftime('%Y%m%d%H%M')}"
        saved = []

        sat_np = self._fill_bad_pixels(sat_frame, CLIP_MIN).squeeze(0).numpy().astype(np.float32)
        lightning_np = lightning_frame.squeeze(0).numpy().astype(np.float32)
        radar_np = radar_frame.squeeze(0).squeeze(0).numpy().astype(np.float32)

        sat_np[sat_np <= CLIP_MIN] = np.nan
        lightning_np[lightning_np < 0.1] = -9999
        radar_np[radar_np < 0] = np.nan

        common_kwargs = dict(
            driver='GTiff', crs=crs, transform=geo_transform,
            compress='deflate', tiled=True, blockxsize=512, blockysize=512,
        )

        for ch in range(sat_np.shape[0]):
            ch_path = os.path.join(output_dir, f"{base_filename}_sat_ch{ch}.tiff")
            with rasterio.open(
                ch_path, 'w', height=sat_np[ch, crop_range].shape[0],
                width=sat_np[ch].shape[1], count=1, dtype=sat_np[ch].dtype,
                nodata=np.nan, **common_kwargs,
            ) as dst:
                dst.write(sat_np[ch, crop_range], 1)
            if make_cog:
                convert_to_cog(ch_path)
            if upload_fn:
                upload_fn(ch_path)
            saved.append(ch_path)

        lightning_path = os.path.join(output_dir, f"{base_filename}_lightning.tiff")
        with rasterio.open(
            lightning_path, 'w', height=lightning_np[0, crop_range].shape[0],
            width=lightning_np[0].shape[1], count=1, dtype=lightning_np[0].dtype,
            nodata=-9999, **common_kwargs,
        ) as dst:
            dst.write(lightning_np[0, crop_range], 1)
        if make_cog:
            convert_to_cog(lightning_path)
        if upload_fn:
            upload_fn(lightning_path)
        saved.append(lightning_path)

        radar_path = os.path.join(output_dir, f"{base_filename}_radar.tiff")
        with rasterio.open(
            radar_path, 'w', height=radar_np[crop_range].shape[0],
            width=radar_np.shape[1], count=1, dtype=radar_np.dtype,
            nodata=np.nan, **common_kwargs,
        ) as dst:
            dst.write(radar_np[crop_range], 1)
        if make_cog:
            convert_to_cog(radar_path)
        if upload_fn:
            upload_fn(radar_path)
        saved.append(radar_path)

        logger.info(f"Saved forecast to {base_filename}_*.tiff")
        return saved

    def run_inference(
        self,
        data_path: str,
        output_dir: str,
        forecast_steps: int = 18,
        nb_forecast: int = 3,
        upload_fn: Optional[Callable[[str], None]] = None,
        debug_endpoint_t_values: Optional[list[float]] = None,
    ) -> InferenceResult:
        """Run full inference pipeline.

        Args:
            data_path: Path to input HDF5 file
            output_dir: Directory to save outputs
            forecast_steps: Number of forecast steps
            nb_forecast: Frames per forecast batch
            upload_fn: Optional callback called with each saved file path so that
                       upload can run concurrently with GPU inference.

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

                if "cg_lightning_data" in hf.keys():
                    print("keys finded")
                    cg_lightning_data = hf["cg_lightning_data"][:]
                else:
                    cg_lightning_data = np.zeros_like(lightning_data)

                print("cg_lightning_data shape:", cg_lightning_data.shape)

                radar_data = hf["radar_data"][:]

                # nan correction
                isnan_radar = np.isnan(radar_data)
                radar_data = np.where(isnan_radar, -10000.0, radar_data)

                num_frames = hf.attrs["num_frames"]
                target_H = hf.attrs["target_height"]
                target_W = hf.attrs["target_width"]
                transform = hf.attrs["transform"]
                epsg = hf.attrs["epsg"]
                elevation_data = hf["elevation_data"][:]
                # c_sat = num_sat_channels (16) + 1 (elevation) + num_radar_channels (1) = 18
                c_sat = hf.attrs["num_sat_channels"] + 1 + hf.attrs["num_radar_channels"]
                c_lightning = hf.attrs["num_lightning_channels"]
                c_radar = hf.attrs["num_radar_channels"]

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
            initial_date = datetime(year, month, day, hour, minute)

            # Prepare initial context
            initial_frames = []
            for i in range(self.context_frames):
                sat_frame = sat_data[i]

                lightning_frame = lightning_data[i]
                cg_lightning_frame = cg_lightning_data[i]

                radar_frame = radar_data[i]
                elev_frame = elevation_data[None, :, :]
                elev_frame = np.where(elev_frame < 0, -100, elev_frame)

                # Concatenate sat + elevation + radar (now 18 channels: 16 sat + 1 elev + 1 radar)
                sat_elev_radar_frame = np.concatenate([sat_frame, elev_frame, radar_frame], axis=0)
                frame = np.concatenate([sat_elev_radar_frame, lightning_frame, cg_lightning_frame], axis=0)[None, ...]
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
            current_high_res_context.c_radar = c_radar

            # Geospatial metadata shared across all timesteps
            crop_start = 450
            crop_end = 2600
            crop_range = slice(crop_start, crop_end)
            lat_max_shifted = 65.0 - (crop_start * 0.012)
            geo_transform = rasterio.transform.from_origin(-10.0, lat_max_shifted, 0.012, 0.012)
            crs = "EPSG:4326"

            # Full-image geo transform for debug endpoint snapshots (no crop
            # offset — the uncropped top-left is at lat 65.0).
            debug_geo_transform = rasterio.transform.from_origin(-10.0, 65.0, 0.012, 0.012)

            # Drive the generator and dispatch I/O to a thread pool so that
            # TIFF writing + COG conversion + GCS upload overlap with GPU inference.
            output_files = []
            pending_futures: list[Future] = []
            global_step = 0

            with ThreadPoolExecutor(max_workers=2) as executor:
                for sat_batch, lightning_batch, radar_batch in self.tiled_inference(
                    initial_context=current_high_res_context,
                    forecast_steps=forecast_steps,
                    nb_forecast=nb_forecast,
                    date=initial_date,
                    output_dir=output_dir,
                    debug_endpoint_t_values=debug_endpoint_t_values,
                    debug_upload_fn=upload_fn,
                    debug_geo_transform=debug_geo_transform,
                    debug_crs=crs,
                    debug_crop_range=slice(None),  # full Europe, no crop
                ):
                    batch_nb = sat_batch.shape[2]
                    for k in range(batch_nb):
                        pred_date = initial_date + timedelta(minutes=10 * (global_step + k + 1))
                        sat_frame = sat_batch[:, :, k, :, :]
                        lightning_frame = lightning_batch[:, :, k, :, :]
                        radar_frame = radar_batch[:, k:k+1, :, :]  # (1, 1, H, W)

                        fut = executor.submit(
                            self._save_timestep_files,
                            sat_frame, lightning_frame, radar_frame,
                            output_dir, pred_date, geo_transform, crs,
                            crop_range, upload_fn,
                        )
                        pending_futures.append(fut)

                    global_step += batch_nb

                # Collect results (also re-raises any exception from a thread)
                for fut in pending_futures:
                    output_files.extend(fut.result())

            result.status = InferenceStatus.COMPLETED
            result.output_path = output_dir
            result.completed_at = datetime.now()
            result.metrics = {
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
