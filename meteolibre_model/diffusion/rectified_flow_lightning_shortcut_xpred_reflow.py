"""
Shortcut Rectified Flow implementation for weather forecasting diffusion model.
This module provides functions for training and generation using shortcut models.
https://arxiv.org/pdf/2410.12557
This script supports multiple interpolation schedules:
- 'linear': Standard Rectified Flow interpolation.
- 'polynomial': A cubic noise schedule inspired by https://arxiv.org/abs/2301.11093
"""

import torch
import math
import random

try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend to avoid display issues
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # If not available, set to None


from meteolibre_model.diffusion.utils import (
    MEAN_CHANNEL_WORLD_ELEVATION_RADAR,
    STD_CHANNEL_WORLD_ELEVATION_RADAR,
    MEAN_CHANNEL_WORLD_ELEVATION,
    STD_CHANNEL_WORLD_ELEVATION,
    MEAN_CHANNEL_WORLD,
    STD_CHANNEL_WORLD,
    MEAN_LIGHTNING,
    STD_LIGHTNING,
)

# -- Parameters --
CLIP_MIN = -4
SHORTCUT_M = 128  # Number of base steps (M=128 as in the paper)
SHORTCUT_K = 0.25  # Fraction of batch for self-consistency (k=1/4 as in the paper)


def normalize(sat_data, lightning_data, device):
    """
    Normalize the batch data using precomputed mean and std.
    """

    sat_data = (
        sat_data
        - MEAN_CHANNEL_WORLD_ELEVATION_RADAR.unsqueeze(0)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .to(device)
    ) / STD_CHANNEL_WORLD_ELEVATION_RADAR.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(
        device
    )

    # Clamp to prevent extreme values
    sat_data = sat_data.clamp(CLIP_MIN, 4)

    lightning_data = (
        lightning_data
        - MEAN_LIGHTNING.unsqueeze(0)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .to(device)
    ) / STD_LIGHTNING.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)

    # Clamp to prevent extreme values
    lightning_data = lightning_data.clamp(CLIP_MIN, 10)

    return sat_data, lightning_data


def denormalize(sat_data, lightning_data, device):
    """
    Denormalize the batch data using precomputed mean and std.
    """
    sat_data = sat_data.to(device) * STD_CHANNEL_WORLD_ELEVATION_RADAR.unsqueeze(0).unsqueeze(
        -1
    ).unsqueeze(-1).unsqueeze(-1).to(device) + MEAN_CHANNEL_WORLD_ELEVATION_RADAR.unsqueeze(
        0
    ).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)

    lightning_data = lightning_data.to(device) * STD_LIGHTNING.unsqueeze(0).unsqueeze(
        -1
    ).unsqueeze(-1).unsqueeze(-1).to(device) + MEAN_LIGHTNING.unsqueeze(0).unsqueeze(
        -1
    ).unsqueeze(-1).unsqueeze(-1).to(device)

    return sat_data, lightning_data


def get_x_t_rf(x0, x1, t, interpolation="linear"):
    """
    Get the interpolated point x_t based on the chosen schedule.
    - 'linear': x_t = (1 - t) * x0 + t * x1
    - 'polynomial': x_t = (1 - t)^3 * x0 + (1 - (1 - t)^3) * x1
    """
    if interpolation == "linear":
        return (1 - t) * x0 + t * x1
    elif interpolation == "polynomial":
        u = 1 - t
        alpha = u**3
        return alpha * x0 + (1 - alpha) * x1
    else:
        raise ValueError(f"Unknown interpolation schedule: {interpolation}")

def trainer_step(
    model, batch, device, sigma=0.0, parametrization="standard", interpolation="linear", use_residual=True
):
    if parametrization != "standard":
        raise ValueError("Only 'standard' parametrization is supported for x-prediction.")

    # (B, C, T, H, W) after permute
    sat_data = batch["sat_patch_data"].permute(0, 2, 1, 3, 4)
    lightning_data = batch["lightning_patch_data"].permute(0, 2, 1, 3, 4)

    b, c_sat, t_dim, h, w = sat_data.shape
    _, c_lightning, _, _, _ = lightning_data.shape

    mask_data_sat = sat_data != CLIP_MIN

    sat_data, lightning_data = normalize(sat_data, lightning_data, device)
    batch_data = torch.cat([sat_data, lightning_data], dim=1)

    x_context = batch_data[:, :, : model.context_frames]



    if use_residual:
        x0 = batch_data[:, :, model.context_frames:] - batch_data[:, :, model.context_frames-1:model.context_frames]
    else:
        x0 = batch_data[:, :, model.context_frames:]

    context_info = batch["spatial_position"]

    x1 = torch.randn_like(x0)

    loss_sat = loss_lightning = 0.0

    # ====================== EMPIRICAL (flow-matching) PART ======================
    num_emp = b
    x0_emp = x0

    x1_emp = x1
    context_info_emp = context_info
    mask_emp = mask_data_sat[:num_emp, :, model.context_frames:]

    t_emp = torch.rand(num_emp, device=device)

    if sigma > 0:
        # We add noise to the context, including the one used for residual if use_residual is True
        # because x_context is a view of batch_data.
        x_context += torch.randn_like(x_context) * sigma * t_emp.view(num_emp,1,1,1,1)


    xt_emp = get_x_t_rf(x0_emp, x1_emp, t_emp.view(num_emp,1,1,1,1), interpolation)

    # da_dt for correct v-loss weighting (paper's 1/(1-t)² or 1/t² equivalent)
    if interpolation == "linear":
        da_dt = torch.full_like(t_emp, -1.0)
    else:  # polynomial
        u = 1 - t_emp
        da_dt = -3 * u ** 2

    da_dt = da_dt.view(num_emp, 1, 1, 1, 1)

    # model predicts clean target (x-prediction)
    model_input_emp = torch.cat([x_context, xt_emp], dim=2)
    context_global_emp = torch.cat([context_info_emp, t_emp.unsqueeze(1), torch.zeros_like(t_emp).unsqueeze(1)], dim=1)

    sat_x_pred_emp, lightning_x_pred_emp = model(
        model_input_emp[:, :c_sat].float(),
        model_input_emp[:, c_sat:].float(),
        context_global_emp.float(),
    )

    x_sat_pred_emp = sat_x_pred_emp[:, :, model.context_frames:]
    x_light_pred_emp = lightning_x_pred_emp[:, :, model.context_frames:]

    weight = 1.0 / (t_emp.view(b,1,1,1,1) + 1e-2) ** 2
    weight = weight.clamp(0.9, 10.)

    # direct x-loss
    loss_sat     = (weight * (x_sat_pred_emp - x0_emp[:, :c_sat]) ** 2)[mask_emp].mean()
    loss_lightning = (weight * (x_light_pred_emp - x0_emp[:, c_sat:]) ** 2).mean()

    return loss_sat + 5.0 * loss_lightning, loss_sat, loss_lightning

def full_image_generation(
    model,
    batch,
    steps=128,
    device="cuda",
    parametrization="standard",
    nb_element=1,
    normalize_input=True,
    use_residual=True,
):
    model.eval()
    with torch.no_grad():
        sat_data = batch["sat_patch_data"].permute(0, 2, 1, 3, 4)
        lightning_data = batch["lightning_patch_data"].permute(0, 2, 1, 3, 4)

        b, c_sat, t, h, w = sat_data.shape
        b, c_lightning, t, h, w = lightning_data.shape

        nb_forecasted_frame = t - model.context_frames

        if normalize_input:
            sat_data, lightning_data = normalize(sat_data, lightning_data, device=device)

        batch_data = torch.cat([sat_data, lightning_data], dim=1)[0:nb_element]

        x_context = batch_data[:, :, : model.context_frames]
        last_context = x_context[:, :, model.context_frames-1:model.context_frames]

        context_info = batch["spatial_position"].to(device)[0:nb_element]

        batch_size, nb_channel, _, h, w = x_context.shape
        x_t = torch.randn(batch_size, nb_channel, nb_forecasted_frame, h, w, device=device)

        d_const = 1.0 / steps
        t_val = 1.0

        for _ in range(steps):
            t_batch = torch.full((batch_size,), t_val, device=device)
            d_batch = torch.full((batch_size,), 0., device=device)

            model_input = torch.cat([x_context, x_t], dim=2)
            context_global = torch.cat([context_info, t_batch.unsqueeze(1), d_batch.unsqueeze(1)], dim=1)

            sat_x_pred, lightning_x_pred = model(
                model_input[:, :c_sat].float(), model_input[:, c_sat:].float(), context_global.float()
            )

            x_pred = torch.cat([sat_x_pred, lightning_x_pred], dim=1)[:, :, model.context_frames:]

            # constant-velocity approximation used in self-consistency loss
            s_theta = (x_t - x_pred) / t_val
            x_t = x_t - s_theta * d_const
            x_t = x_t.clamp(-7, 7)

            t_val -= d_const

        if use_residual:
            x_t = x_t + last_context.expand_as(x_t)

        # keep clipped values from last context frame
        x_t = torch.where(last_context == CLIP_MIN, last_context, x_t)

        generated = x_t.cpu()
        target    = batch_data[:, :, model.context_frames:].cpu()

    model.train()
    return generated, target