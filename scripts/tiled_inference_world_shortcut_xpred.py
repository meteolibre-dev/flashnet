import torch
import numpy as np
import argparse
from tqdm import tqdm
import os
import sys
from datetime import datetime, timedelta
from suncalc import get_position
import pyproj
import h5py
import yaml

# Add project root to sys.path
project_root = os.path.abspath("/workspace/flashnet/")
sys.path.insert(0, project_root)

# Load config
config_path = os.path.join(project_root, "meteolibre_model/config/configs.yml")
with open(config_path) as f:
    config = yaml.safe_load(f)
params = config["model_v11_mtg_world_lightning_shortcut"]

# Constants for coordinates (will be set from HDF5 file)
from meteolibre_model.models.unet3d_film_dual import DualUNet3DFiLM
from meteolibre_model.diffusion.rectified_flow_lightning_shortcut_xpred import (
    normalize,
    denormalize,
    CLIP_MIN,
)

from safetensors.torch import load_file


def _extract_patch(image, x, y, patch_size):
    return image[..., y : y + patch_size, x : x + patch_size]


def _place_patch(full_image, patch, x, y, patch_size):
    full_image[..., y : y + patch_size, x : x + patch_size] = patch
    return full_image


@torch.no_grad()
def tiled_inference(
    model,
    initial_context,  # (B, C, T_ctx, H_big, W_big)
    patch_size,
    steps,
    device,
    date,
    c_sat,
    c_lightning,
    transform,
    epsg,
    transformer,
    batch_size=64,
    use_residual=True,
    nb_forecast=1,
):
    model.eval()
    model.to(device)
    _, C, T_ctx, H_big, W_big = initial_context.shape
    x_t_full_res = torch.randn(1, C, nb_forecast, H_big, W_big, device=device)
    # Create two grids of patches
    # Grid 1: Standard grid
    y_starts1 = list(range(0, H_big - patch_size + 1, patch_size))
    if (H_big - patch_size) % patch_size != 0:
        y_starts1.append(H_big - patch_size)
    x_starts1 = list(range(0, W_big - patch_size + 1, patch_size))
    if (W_big - patch_size) % patch_size != 0:
        x_starts1.append(W_big - patch_size)
    patch_coords1 = [(x, y) for y in y_starts1 for x in x_starts1]
    # Grid 2: Shifted grid
    shift = patch_size // 2
    y_starts2 = list(range(shift, H_big - patch_size + 1, patch_size))
    if (H_big - patch_size - shift) % patch_size != 0 and H_big - patch_size > shift:
        y_starts2.append(H_big - patch_size)
    x_starts2 = list(range(shift, W_big - patch_size + 1, patch_size))
    if (W_big - patch_size - shift) % patch_size != 0 and W_big - patch_size > shift:
        x_starts2.append(W_big - patch_size)
    patch_coords2 = [(x, y) for y in y_starts2 for x in x_starts2]
    patch_coords = patch_coords1 + patch_coords2
    d_const = 1.0 / steps
    for i in tqdm(range(steps), desc="Tiled Denoising"):
        t_val = 1.0 - i * d_const
        t_batch_val = torch.full((1,), t_val, device=device)
        d_batch_val = torch.full((1,), d_const, device=device)
        aggregated_x_pred = torch.zeros(1, C, nb_forecast, H_big, W_big, device=device)
        overlap_counts = torch.zeros(1, 1, nb_forecast, H_big, W_big, device=device)
        with torch.no_grad():
            for i_batch in range(0, len(patch_coords), batch_size):
                coords_batch = patch_coords[i_batch : i_batch + batch_size]
                patch_x_t_batch, patch_context_batch, context_global_batch = [], [], []
                pixel_xs = [x + patch_size // 2 for x, y in coords_batch]
                pixel_ys = [y + patch_size // 2 for x, y in coords_batch]
                lons, lats = [], []
                for j in range(len(coords_batch)):
                    px = pixel_xs[j]
                    py = pixel_ys[j]
                    x_crs = transform[0] * px + transform[1] * py + transform[2]
                    y_crs = transform[3] * px + transform[4] * py + transform[5]
                    if epsg != 4326:
                        lon, lat = transformer.transform(x_crs, y_crs)
                    else:
                        lon, lat = x_crs, y_crs
                    lons.append(lon)
                    lats.append(lat)
                for j, (x_start, y_start) in enumerate(coords_batch):
                    patch_x_t = _extract_patch(
                        x_t_full_res, x_start, y_start, patch_size
                    )
                    patch_context = _extract_patch(
                        initial_context, x_start, y_start, patch_size
                    )
                    result = get_position(date, lons[j], lats[j])
                    date_noon = date.replace(hour=12, minute=0, second=0, microsecond=0)
                    result_noon = get_position(date_noon, lons[j], lats[j])
                    spatial_position = torch.tensor(
                        [result["azimuth"], result["altitude"], result_noon["altitude"], lats[j] / 10.0],
                        device=device,
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
                sat_pred_batch, lightning_pred_batch = model(
                    model_input_sat.float(),
                    model_input_lightning.float(),
                    torch.cat(context_global_batch, dim=0).float(),
                )
                x_pred_batch = torch.cat([sat_pred_batch, lightning_pred_batch], dim=1)[
                    :, :, T_ctx:, :, :
                ]
                for j, (x_start, y_start) in enumerate(coords_batch):
                    aggregated_x_pred[
                        ...,
                        y_start : y_start + patch_size,
                        x_start : x_start + patch_size,
                    ] += x_pred_batch[j : j + 1]
                    overlap_counts[
                        ...,
                        y_start : y_start + patch_size,
                        x_start : x_start + patch_size,
                    ] += 1
        overlap_counts[overlap_counts == 0] = 1
        averaged_x_pred = aggregated_x_pred / overlap_counts
        s_theta = (x_t_full_res - averaged_x_pred) / t_val
        x_t_full_res = x_t_full_res - s_theta * d_const
        x_t_full_res = x_t_full_res.clamp(-7, 7)
    last_context_frame = initial_context[:, :, -1:, :, :]
    if use_residual:
        x_t_full_res[:, :, 0:1, :, :] += last_context_frame
    # Clamp results to be within the expected normalized range
    mask = (last_context_frame == CLIP_MIN).expand(-1, -1, nb_forecast, -1, -1)
    expanded_last = last_context_frame.expand(-1, -1, nb_forecast, -1, -1)
    x_t_full_res = torch.where(mask, expanded_last, x_t_full_res)
    return x_t_full_res.cpu()


def main():
    parser = argparse.ArgumentParser(
        description="Tiled Diffusion for large image forecasting with Shortcut Rectified Flow (x-prediction)."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/20251019_epoch61_mtg_lightning_soap.safetensors",
        help="Path to the pre-trained model .safetensors file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_forecasts_shortcut_xpred",
        help="Directory to save generated forecasts.",
    )
    parser.add_argument(
        "--forecast_steps",
        type=int,
        default=18,
        help="Number of autoregressive forecast steps to generate.",
    )
    parser.add_argument(
        "--nb_forecast",
        type=int,
        default=params.get("nb_forecast", 3),
        help="Number of frames to forecast per model call (from config).",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="../dataset/data_inference_full/2025-10-14_04-00_full.h5",
        help="Path to the HDF5 file containing the initial context.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=128,
        help="Size of patches trained on (e.g., 128 for 128x128).",
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        default=128,
        help="Number of denoising steps for each tiled diffusion process.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=params["batch_size"],
        help="Batch size for processing patches during inference.",
    )
    parser.add_argument(
        "--context_frames",
        type=int,
        default=params["model"]["context_frames"],
        help="Number of context frames expected by the model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda or cpu).",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Initialize model
    model_params = params["model"]
    model = DualUNet3DFiLM(**model_params)
    # Load model weights
    if os.path.exists(args.model_path):
        loaded_state_dict = load_file(args.model_path)
        model.load_state_dict(loaded_state_dict)
        print(f"Loaded model weights from {args.model_path}")
    else:
        print(
            f"Warning: Model weights not found at {args.model_path}. Using randomly initialized model."
        )
    model.to(args.device)
    # Load initial context from HDF5 file
    if not os.path.exists(args.data_file):
        raise ValueError(f"Data file {args.data_file} not found.")
    with h5py.File(args.data_file, "r") as hf:
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
    if epsg != 4326:
        transformer = pyproj.Transformer.from_crs(
            f"EPSG:{epsg}", "EPSG:4326", always_xy=True
        )
    else:
        transformer = None
    nb_channels = c_sat + c_lightning
    # Parse initial_date from filename
    filename = os.path.basename(args.data_file)
    # Assuming filename like "2025-10-14_04-00_full.h5"
    date_str = filename.split("_full.h5")[0]  # "2025-10-14_04-00"
    date_part, time_part = date_str.split("_")[0], date_str.split("_")[1]
    year, month, day = map(int, date_part.split("-"))
    hour, minute = map(int, time_part.split("-"))
    initial_date = datetime(year, month, day, hour, minute) - timedelta(minutes=18 * 10)

    if num_frames < args.context_frames:
        raise ValueError(
            f"Not enough frames in {args.data_file}. Need at least {args.context_frames}, found {num_frames}"
        )
    initial_frames = []
    for i in range(args.context_frames):
        sat_frame = sat_data[i]  # (C, H, W)
        lightning_frame = lightning_data[i]  # (1, H, W)
        
        elev_frame = elevation_data[None, :, :]  # (1, H, W)
        elev_frame = np.where(elev_frame < 0, -100, elev_frame)

        sat_elev_frame = np.concatenate([sat_frame, elev_frame], axis=0)
        frame = np.concatenate([sat_elev_frame, lightning_frame], axis=0)[
            None, ...
        ]  # (1, nb_channels, H, W)
        initial_frames.append(frame)
    # Stack along time dimension: (1, nb_channels, T, H, W)
    current_high_res_context = np.stack(initial_frames, axis=2)
    current_high_res_context = (
        torch.from_numpy(current_high_res_context).float().to(args.device)
    )
    # Split into sat and lightning
    sat_data = current_high_res_context[:, :c_sat]
    lightning_data = current_high_res_context[:, c_sat:]
    # Normalize
    sat_data, lightning_data = normalize(sat_data, lightning_data, args.device)
    # Concat back
    current_high_res_context = torch.cat([sat_data, lightning_data], dim=1)
    # Autoregressive Generation Loop

    use_residual = params.get("residual", True)

    print(use_residual)

    current_step = 0
    while current_step < args.forecast_steps:
        remaining = args.forecast_steps - current_step
        this_nb = min(args.nb_forecast, remaining)
        print(f"Autoregressive batch {current_step // args.nb_forecast + 1}, predicting frames {current_step + 1} to {current_step + this_nb}/{args.forecast_steps}")
        # Compute the date for the first predicted frame in this batch
        prediction_date = initial_date + timedelta(minutes=10 * (current_step + 1))
        # Perform tiled inference for the next this_nb frames

        generated_frame = tiled_inference(
            model=model,
            initial_context=current_high_res_context,
            patch_size=args.patch_size,
            steps=args.denoising_steps,
            device=args.device,
            date=prediction_date,
            c_sat=c_sat,
            c_lightning=c_lightning,
            batch_size=args.batch_size,
            transform=transform,
            epsg=epsg,
            transformer=transformer,
            use_residual=use_residual,
            nb_forecast=this_nb,
        )  # Shape: (1, nb_channels, this_nb, H_big, W_big)
        generated_norm = generated_frame.to(args.device)
        # Split and denormalize for saving
        sat_generated = generated_norm[:, :c_sat]
        lightning_generated = generated_norm[:, c_sat:]
        sat_denorm, lightning_denorm = denormalize(
            sat_generated, lightning_generated, args.device
        )
        # Save each generated frame in the batch
        for k in range(this_nb):
            sat_frame = sat_denorm[:, :, k, :, :]
            lightning_frame = lightning_denorm[:, :, k, :, :]
            pred_date = initial_date + timedelta(minutes=10 * (current_step + k + 1))
            filename = f"forecast_{pred_date.strftime('%Y%m%d%H%M')}.npz"
            output_filepath = os.path.join(args.output_dir, filename)
            np.savez_compressed(
                output_filepath,
                sat_forecast=sat_frame.squeeze(0).cpu().numpy(),
                lightning_forecast=lightning_frame.squeeze(0).cpu().numpy(),
            )
            print(f"Saved forecast to {output_filepath}")
        # Update current_high_res_context for the next autoregressive step
        # Shift the context by this_nb frames: take the last (T_ctx - this_nb) from current + the new this_nb normalized frames
        T_ctx = args.context_frames
        if this_nb >= T_ctx:
            current_high_res_context = generated_norm[:, :, -T_ctx:, :, :]
        else:
            tail = current_high_res_context[:, :, this_nb:, :, :]
            current_high_res_context = torch.cat(
                [tail, generated_norm[:, :, :this_nb, :, :]], dim=2
            )
        current_step += this_nb
    print("Tiled diffusion and autoregressive forecasting complete.")


if __name__ == "__main__":
    main()