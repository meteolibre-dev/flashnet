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
        - MEAN_CHANNEL_WORLD_ELEVATION.unsqueeze(0)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .to(device)
    ) / STD_CHANNEL_WORLD_ELEVATION.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(
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
    sat_data = sat_data.to(device) * STD_CHANNEL_WORLD_ELEVATION.unsqueeze(0).unsqueeze(
        -1
    ).unsqueeze(-1).unsqueeze(-1).to(device) + MEAN_CHANNEL_WORLD_ELEVATION.unsqueeze(
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

    # Replace context with CLIP_MIN (no previous weather context)
    x_context = torch.full_like(batch_data[:, :, : model.context_frames], CLIP_MIN)
    if sigma > 0:
        x_context += torch.randn_like(x_context) * sigma

    if use_residual:
        # Use CLIP_MIN as the base for residual instead of the true last context frame
        x0 = batch_data[:, :, model.context_frames:] - CLIP_MIN
    else:
        x0 = batch_data[:, :, model.context_frames:]

    context_info = batch["spatial_position"]

    x1 = torch.randn_like(x0)

    num_self = int(b * SHORTCUT_K + 0.5)
    num_emp = b - num_self

    loss_sat = loss_lightning = 0.0

    # ====================== EMPIRICAL (flow-matching) PART ======================
    if num_emp > 0:
        x_context_emp = x_context[:num_emp]
        x0_emp = x0[:num_emp]
        x1_emp = x1[:num_emp]
        context_info_emp = context_info[:num_emp]
        mask_emp = mask_data_sat[:num_emp, :, model.context_frames:]

        t_emp = torch.rand(num_emp, device=device)

        xt_emp = get_x_t_rf(x0_emp, x1_emp, t_emp.view(num_emp,1,1,1,1), interpolation)

        # da_dt for correct v-loss weighting (paper's 1/(1-t)² or 1/t² equivalent)
        if interpolation == "linear":
            da_dt = torch.full_like(t_emp, -1.0)
        else:  # polynomial
            u = 1 - t_emp
            da_dt = -3 * u ** 2

        da_dt = da_dt.view(num_emp, 1, 1, 1, 1)

        # model predicts clean target (x-prediction)
        model_input_emp = torch.cat([x_context_emp, xt_emp], dim=2)
        context_global_emp = torch.cat([context_info_emp, t_emp.unsqueeze(1), torch.zeros_like(t_emp).unsqueeze(1)], dim=1)

        sat_x_pred_emp, lightning_x_pred_emp = model(
            model_input_emp[:, :c_sat].float(),
            model_input_emp[:, c_sat:].float(),
            context_global_emp.float(),
        )

        x_sat_pred_emp = sat_x_pred_emp[:, :, model.context_frames:]
        x_light_pred_emp = lightning_x_pred_emp[:, :, model.context_frames:]

        # v-pred = da_dt * (x_pred - x1)   →  true v-loss
        v_sat_pred = da_dt * (x_sat_pred_emp - x1_emp[:, :c_sat])
        v_light_pred = da_dt * (x_light_pred_emp - x1_emp[:, c_sat:])

        v_sat_target = da_dt * (x0_emp[:, :c_sat] - x1_emp[:, :c_sat])
        v_light_target = da_dt * (x0_emp[:, c_sat:] - x1_emp[:, c_sat:])

        # Apply loss weights
        sq_err_sat = (v_sat_pred - v_sat_target) ** 2
        sq_err_light = (v_light_pred - v_light_target) ** 2
        
        loss_sat += sq_err_sat[mask_emp].mean()
        loss_lightning += sq_err_light.mean()

    # ====================== SELF-CONSISTENCY (shortcut) PART ======================
    if num_self > 0:
        x_context_self = x_context[num_emp:]
        x0_self = x0[num_emp:]
        x1_self = x1[num_emp:]
        context_info_self = context_info[num_emp:]
        mask_self = mask_data_sat[num_emp:, :, model.context_frames:]

        # sample d and t exactly as before
        levels = list(range(7))
        min_d = 1.0 / SHORTCUT_M
        d_self = torch.zeros(num_self, device=device)
        t_self = torch.zeros(num_self, device=device)
        for ii in range(num_self):
            l = random.choice(levels)
            dd = (2 ** l) / SHORTCUT_M
            max_k = math.floor(1.0 / dd)
            k = random.randint(2, max_k)
            tt = k * dd
            d_self[ii] = dd
            t_self[ii] = tt

        xt_self = get_x_t_rf(x0_self, x1_self, t_self.view(num_self,1,1,1,1), interpolation)

        # first call at (t, d)
        context_global_d = torch.cat([context_info_self, t_self.unsqueeze(1), d_self.unsqueeze(1)], dim=1)
        model_input_d = torch.cat([x_context_self, xt_self], dim=2)

        sat_x_d, lightning_x_d = model(model_input_d[:, :c_sat].float(), model_input_d[:, c_sat:].float(), context_global_d.float())
        x_d = torch.cat([sat_x_d, lightning_x_d], dim=1)[:, :, model.context_frames:]
        s_theta = x1_self - x_d

        # step
        x_mid = xt_self - s_theta * d_self.view(num_self,1,1,1,1)

        t_mid = t_self - d_self
        d_mid = d_self.clone()
        d_mid[d_self == min_d] = 0.0

        # second call at (t-d, d)
        context_global_mid = torch.cat([context_info_self, t_mid.unsqueeze(1), d_mid.unsqueeze(1)], dim=1)
        model_input_mid = torch.cat([x_context_self, x_mid], dim=2)
        sat_x_mid, lightning_x_mid = model(model_input_mid[:, :c_sat].float(), model_input_mid[:, c_sat:].float(), context_global_mid.float())
        x_mid_pred = torch.cat([sat_x_mid, lightning_x_mid], dim=1)[:, :, model.context_frames:]
        s_theta_mid = x1_self - x_mid_pred

        starget = ((s_theta + s_theta_mid) / 2).detach()

        # call at (t, 2d)
        context_global_2d = torch.cat([context_info_self, t_self.unsqueeze(1), (2 * d_self).unsqueeze(1)], dim=1)
        sat_x_2d, lightning_x_2d = model(model_input_d[:, :c_sat].float(), model_input_d[:, c_sat:].float(), context_global_2d.float())
        x_2d = torch.cat([sat_x_2d, lightning_x_2d], dim=1)[:, :, model.context_frames:]
        s_2d = x1_self - x_2d

        # Apply loss weights
        sq_err_sat_self = (s_2d[:, :c_sat] - starget[:, :c_sat]) ** 2
        sq_err_light_self = (s_2d[:, c_sat:] - starget[:, c_sat:]) ** 2

        loss_sat += sq_err_sat_self[mask_self].mean()
        loss_lightning += sq_err_light_self.mean()

    return 10. * loss_sat + 20.0 * loss_lightning, loss_sat, loss_lightning

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

        # Replace context with CLIP_MIN (no previous weather context)
        x_context = torch.full_like(batch_data[:, :, : model.context_frames], CLIP_MIN)
        last_context = x_context[:, :, model.context_frames-1:model.context_frames]

        context_info = batch["spatial_position"].to(device)[0:nb_element]

        batch_size, nb_channel, _, h, w = x_context.shape
        x_t = torch.randn(batch_size, nb_channel, nb_forecasted_frame, h, w, device=device)

        d_const = 1.0 / steps
        t_val = 1.0

        for _ in range(steps):
            t_batch = torch.full((batch_size,), t_val, device=device)
            d_batch = torch.full((batch_size,), d_const, device=device)

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
            # Use CLIP_MIN as the base for residual
            x_t = x_t + CLIP_MIN

        # We don't apply the mask from last_context here because last_context is now full CLIP_MIN
        # x_t = torch.where(last_context == CLIP_MIN, last_context, x_t)

        generated = x_t.cpu()
        target    = batch_data[:, :, model.context_frames:].cpu()

    model.train()
    return generated, target