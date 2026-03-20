"""
Computes mean and std of the normalized residuals:
    x0 = normalized_batch[:, :, context_frames:] - normalized_batch[:, :, context_frames-1:context_frames]

These values can be used to further normalize the model target when use_residual=True,
which helps convergence by centering and scaling the actual prediction target.
"""
import numpy as np
import sys
import os
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from meteolibre_model.dataset.dataset_mtg_lightning_radar import MeteoLibreMapDataset
from meteolibre_model.diffusion.rectified_flow_lightning_shortcut_xpred_reflow import normalize
from tqdm import tqdm

CONTEXT_FRAMES = 4
NB_TEMPORAL = 7
NUM_SAMPLES = 2000
LOCALREPO = "/workspace/dataset"


def compute_residual_mean_std():
    dataset = MeteoLibreMapDataset(
        localrepo=LOCALREPO,
        cache_size=4,
        seed=44,
        nb_temporal=NB_TEMPORAL,
    )

    print(f"Computing residual mean/std over {NUM_SAMPLES} samples (context_frames={CONTEXT_FRAMES})...")

    sat_sum = None
    sat_sum_sq = None
    sat_count = None
    lightning_sum = None
    lightning_sum_sq = None
    lightning_count = None

    device = "cpu"

    for i, sample in enumerate(tqdm(dataset, total=NUM_SAMPLES)):
        if i >= NUM_SAMPLES:
            break

        # dataset returns (T, C, H, W) — reshape to (1, C, T, H, W) for normalize()
        sat_data = sample["sat_patch_data"].permute(1, 0, 2, 3).unsqueeze(0).float()
        lightning_data = sample["lightning_patch_data"].permute(1, 0, 2, 3).unsqueeze(0).float()

        sat_data, lightning_data = normalize(sat_data, lightning_data, device)

        c_sat = sat_data.shape[1]
        c_lightning = lightning_data.shape[1]

        batch_data = torch.cat([sat_data, lightning_data], dim=1)  # (1, C, T, H, W)

        # Residual: same operation as trainer_step with use_residual=True
        # Shape: (1, C, T_forecast, H, W)
        residual = (
            batch_data[:, :, CONTEXT_FRAMES:]
            - batch_data[:, :, CONTEXT_FRAMES - 1 : CONTEXT_FRAMES]
        )

        res_sat   = residual[0, :c_sat].numpy()    # (C_sat,   T_forecast, H, W)
        res_light = residual[0, c_sat:].numpy()    # (C_light, T_forecast, H, W)

        if sat_sum is None:
            sat_sum      = np.zeros(c_sat,       dtype=np.float64)
            sat_sum_sq   = np.zeros(c_sat,       dtype=np.float64)
            sat_count    = np.zeros(c_sat,       dtype=np.int64)
            lightning_sum    = np.zeros(c_lightning, dtype=np.float64)
            lightning_sum_sq = np.zeros(c_lightning, dtype=np.float64)
            lightning_count  = np.zeros(c_lightning, dtype=np.int64)

        n_spatial = res_sat.shape[1] * res_sat.shape[2] * res_sat.shape[3]
        sat_sum    += res_sat.sum(axis=(1, 2, 3))
        sat_sum_sq += (res_sat ** 2).sum(axis=(1, 2, 3))
        sat_count  += n_spatial

        n_spatial_l = res_light.shape[1] * res_light.shape[2] * res_light.shape[3]
        lightning_sum    += res_light.sum(axis=(1, 2, 3))
        lightning_sum_sq += (res_light ** 2).sum(axis=(1, 2, 3))
        lightning_count  += n_spatial_l

    sat_mean = sat_sum / sat_count
    sat_std  = np.sqrt(np.maximum(sat_sum_sq / sat_count - sat_mean ** 2, 0))

    lightning_mean = lightning_sum / lightning_count
    lightning_std  = np.sqrt(np.maximum(lightning_sum_sq / lightning_count - lightning_mean ** 2, 0))

    print("\n--- Sat residual (normalized space) ---")
    print("Mean:", sat_mean)
    print("Std: ", sat_std)

    print("\n--- Lightning residual (normalized space) ---")
    print("Mean:", lightning_mean)
    print("Std: ", lightning_std)

    print("\n# Paste these into utils.py:")
    print(f"MEAN_SAT_RESIDUAL       = torch.tensor({sat_mean.tolist()})")
    print(f"STD_SAT_RESIDUAL        = torch.tensor({sat_std.tolist()})")
    print(f"MEAN_LIGHTNING_RESIDUAL = torch.tensor({lightning_mean.tolist()})")
    print(f"STD_LIGHTNING_RESIDUAL  = torch.tensor({lightning_std.tolist()})")


if __name__ == "__main__":
    compute_residual_mean_std()
