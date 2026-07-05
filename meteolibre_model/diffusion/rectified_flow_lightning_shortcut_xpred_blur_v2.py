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

import torch.nn.functional as F

from meteolibre_model.diffusion.utils import (
    MEAN_CHANNEL_WORLD_ELEVATION_RADAR,
    STD_CHANNEL_WORLD_ELEVATION_RADAR,
    MEAN_CHANNEL_WORLD_ELEVATION,
    STD_CHANNEL_WORLD_ELEVATION,
    MEAN_CHANNEL_WORLD,
    STD_CHANNEL_WORLD,
    MEAN_LIGHTNING_CG,
    STD_LIGHTNING_CG,
    MEAN_SAT_RESIDUAL,
    STD_SAT_RESIDUAL,
    MEAN_LIGHTNING_RESIDUAL,
    STD_LIGHTNING_RESIDUAL,
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
        - MEAN_LIGHTNING_CG.unsqueeze(0)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .to(device)
    ) / STD_LIGHTNING_CG.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)

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

    lightning_data = lightning_data.to(device) * STD_LIGHTNING_CG.unsqueeze(0).unsqueeze(
        -1
    ).unsqueeze(-1).unsqueeze(-1).to(device) + MEAN_LIGHTNING_CG.unsqueeze(0).unsqueeze(
        -1
    ).unsqueeze(-1).unsqueeze(-1).to(device)

    return sat_data, lightning_data


def normalize_residual(x0, c_sat, device):
    """Normalize residual target (sat + lightning channels concatenated)."""
    mean = torch.cat([MEAN_SAT_RESIDUAL, MEAN_LIGHTNING_RESIDUAL]).to(device).view(1, -1, 1, 1, 1)
    std  = torch.cat([STD_SAT_RESIDUAL,  STD_LIGHTNING_RESIDUAL ]).to(device).view(1, -1, 1, 1, 1)
    return (x0 - mean) / std


def denormalize_residual(x0, c_sat, device):
    """Denormalize residual back to normalized-data space."""
    mean = torch.cat([MEAN_SAT_RESIDUAL, MEAN_LIGHTNING_RESIDUAL]).to(device).view(1, -1, 1, 1, 1)
    std  = torch.cat([STD_SAT_RESIDUAL,  STD_LIGHTNING_RESIDUAL ]).to(device).view(1, -1, 1, 1, 1)
    return x0 * std + mean


def get_x_t_rf(x0, x1, t, interpolation="linear"):
    """
    Get the interpolated point x_t based on the chosen schedule.
    - 'linear': x_t = (1 - t) * x0 + t * x1
    - 'polynomial': x_t = (1 - t^(1/2)) * x0 + t^(1/2) * x1
    - 'bridge': handled inline in trainer_step / full_image_generation (needs
      an extra noise draw eps), see `bridge_coeffs`.
    """
    if interpolation == "linear":
        return (1 - t) * x0 + t * x1
    elif interpolation == "polynomial":
        alpha = 1 - t ** 0.5
        return alpha * x0 + (1 - alpha) * x1
    else:
        raise ValueError(f"Unknown interpolation schedule: {interpolation}")


def bridge_coeffs(t, sigma, sigma_min):
    """
    Brownian-bridge probability path coefficients (user's convention:
    t=1 -> noise x1, t=0 -> data x0).

        c_t^2   = sigma^2 * t * (1 - t) + sigma_min^2
        c'_t/c_t = sigma^2 * (1 - 2t) / (2 * c_t^2)

    The interpolant is  x_t = (1-t) x0 + t x1 + c_t * eps,
    and the deterministic flow velocity (used at inference) is
        v_t = (x1 - x0) + (c'_t / c_t) * (x_t - mu_t),
    with mu_t = (1-t) x0 + t x1.

    Variance is minimal (sigma_min^2) at both endpoints and maximal
    (sigma^2/4) in the middle t=0.5, which keeps the vector-field
    variance low for strongly-correlated spatio-temporal data and
    enables few-step sampling. See Lim et al. 2024,
    "Elucidating the Design Choice of Probability Paths in Flow Matching
    for Forecasting" (arXiv:2410.03229).

    Args:
        t: scalar or tensor (any shape).
        sigma: bridge noise scale (max extra std ~ sigma/2 at t=0.5).
        sigma_min: small floor > 0 for numerical stability at endpoints.
    Returns (c_t, cp_over_c) as tensors.
    """
    t = torch.as_tensor(t, dtype=torch.float32)
    var = sigma ** 2 * t * (1.0 - t) + sigma_min ** 2
    c = torch.sqrt(var)
    cp_over_c = sigma ** 2 * (1.0 - 2.0 * t) / (2.0 * var + 1e-12)
    return c, cp_over_c

def apply_blur_with_sigma_batched(x, blur_sigma, n_bins=8, min_kernel=0, sigma_factor=3):
    """
    Vectorisé via binning des sigma.
    blur_sigma: (B,) tensor, sigma en pixels
    n_bins: nombre de niveaux de blur distincts
    """
    b, c, t, h, w = x.shape
    out = torch.zeros_like(x)
    
    # Discrétise blur_sigma en n_bins niveaux
    sigma_max = blur_sigma.max().item()
    bin_edges = torch.linspace(0, sigma_max + 1e-6, n_bins + 1, device=x.device)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_ids = torch.bucketize(blur_sigma, bin_edges[1:])  # (B,)

    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        if not mask.any():
            continue
        
        s = bin_centers[bin_idx].item()
        x_bin = x[mask]  # (B_bin, C, T, H, W)
        b_bin = x_bin.shape[0]

        if s < 0.1:
            out[mask] = x_bin
            continue

        k = max(min_kernel, 2 * int(sigma_factor * s) + 1)

        coords = torch.arange(k, dtype=torch.float32, device=x.device) - k // 2
        kernel_1d = torch.exp(-(coords ) ** 2 / (2 * s ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel = kernel_2d.expand(c * t, 1, k, k)

        x_flat = x_bin.reshape(b_bin, c * t, h, w)
        pad = k // 2
        blurred = F.conv2d(x_flat, kernel, padding=pad, groups=c * t)
        out[mask] = blurred.reshape(b_bin, c, t, h, w)

    return out


def trainer_step(
    model, batch, device, sigma=0.0, parametrization="standard", interpolation="linear", use_residual=True,
    bridge_sigma=0.5, bridge_sigma_min=1e-3,
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
        x0 = normalize_residual(x0, c_sat, device)
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

    # Stratified sampling with 32 bins
    n_bins = 32
    bin_size = 1.0 / n_bins
    bin_indices = torch.randperm(n_bins, device=device).repeat_interleave((num_emp + n_bins - 1) // n_bins)[:num_emp]
    t_emp = (bin_indices.float() + torch.rand(num_emp, device=device)) * bin_size
    t_emp = t_emp[torch.randperm(num_emp, device=device)]

    # # log norm sampling for t
    # eps = torch.randn(num_emp, device=device)
    # t_emp = torch.sigmoid(-0.5 + 1.2 * eps).clamp(1e-4, 1 - 1e-4)

    # On-manifold context augmentation: blur + per-sample amplitude jitter.
    #
    # Rationale (validated in AR_drifting/FINDINGS_MANIFOLD.md):
    #   - The AR-rollout failure mode is exposure bias — the model at inference
    #     feeds its own slightly-degraded outputs back as context. On structured
    #     data the degradation looks like a *wider, lower-amplitude* version of
    #     the true frame, NOT like isotropic per-pixel noise.
    #   - Gaussian blur + multiplicative amplitude jitter is a plausible
    #     on-manifold perturbation (it looks like real data, just slightly
    #     mis-rendered) and teaches the model to sharpen/deblur a degraded
    #     context. Pixel-space additive noise breaks the smooth profile and
    #     causes catastrophic blur (sharpness ~0.22), so it is NOT used.
    #   - Inference uses CLEAN context; the augmentation is train-only.
    if sigma > 0:
        # Per-sample blur strength on a logit-normal schedule (most samples get
        # mild blur, a long tail gets stronger blur).
        eps = torch.randn(num_emp, device=device)
        t_emp_blur = torch.sigmoid(1.4 + 1.8 * eps).clamp(1e-4, 1 - 1e-4)
        blur_sigma = t_emp_blur * sigma  # (B,)
        x_context_t = apply_blur_with_sigma_batched(x_context, blur_sigma)

        # Per-sample multiplicative amplitude jitter (±~20% std in the champion
        # config). Broadcasts as (B, 1, 1, 1, 1) so it scales every element of a
        # sample's context by the same factor — preserves the spatial/temporal
        # structure, only the overall amplitude is perturbed.
        amplitude_jitter_std = 0.20
        scale = 1.0 + amplitude_jitter_std * torch.randn(
            b, 1, 1, 1, 1, device=device
        )
        x_context_t = x_context_t * scale
    else:
        x_context_t = x_context

    xt_emp = get_x_t_rf(x0_emp, x1_emp, t_emp.view(num_emp,1,1,1,1), interpolation) if interpolation != "bridge" else None
    if interpolation == "bridge":
        # x_t = (1-t) x0 + t x1 + c_t * eps,  c_t^2 = sigma^2 t(1-t) + sigma_min^2
        c_t, _ = bridge_coeffs(t_emp, bridge_sigma, bridge_sigma_min)  # (B,)
        tsh = t_emp.view(num_emp, 1, 1, 1, 1)
        csh = c_t.view(num_emp, 1, 1, 1, 1)
        eps_bb = torch.randn_like(x0_emp)
        xt_emp = (1.0 - tsh) * x0_emp + tsh * x1_emp + csh * eps_bb

    # da_dt for correct v-loss weighting
    # alpha(t) = 1 - sqrt(t)  =>  da/dt = -1 / (2 * sqrt(t))
    if interpolation == "linear":
        da_dt = torch.full_like(t_emp, -1.0)
    else:  # polynomial: alpha(t) = 1 - t^(1/2)
        da_dt = -0.5 / (t_emp ** 0.5 + 1e-8)

    da_dt = da_dt.view(num_emp, 1, 1, 1, 1)

    # model predicts clean target (x-prediction)
    model_input_emp = torch.cat([x_context_t, xt_emp], dim=2)
    context_global_emp = torch.cat([context_info_emp, (torch.zeros_like(t_emp) ).unsqueeze(1), t_emp.unsqueeze(1)], dim=1)

    sat_x_pred_emp, lightning_x_pred_emp = model(
        model_input_emp[:, :c_sat].float(),
        model_input_emp[:, c_sat:].float(),
        context_global_emp.float(),
    )

    x_sat_pred_emp = sat_x_pred_emp[:, :, model.context_frames:]
    x_light_pred_emp = lightning_x_pred_emp[:, :, model.context_frames:]

    if interpolation == "polynomial":
        # da/dt = -1/(2*sqrt(t))  =>  (da/dt)^2 ∝ 1/t
        weight = 1.0 / (t_emp.view(b,1,1,1,1) + 1e-2) ** 2
    elif interpolation == "bridge":
        # Bridge VF variance is already low & well-balanced; uniform x-pred
        # MSE weighting works well (cf. Lim et al. 2024). Override by tuning
        # if small-t detail (lightning) is under-fit.
        weight = torch.ones_like(t_emp.view(b, 1, 1, 1, 1))
    else:
        # linear: da/dt = -1  =>  empirical 1/t^2 upweighting of small t
        weight = 1.0 / (t_emp.view(b, 1, 1, 1, 1) + 1e-2) ** 2

    weight = weight.clamp(0.9, 10.)

    # direct x-loss
    loss_sat     = (weight * (x_sat_pred_emp - x0_emp[:, :c_sat]) ** 2)[mask_emp].mean()
    loss_lightning = (weight * (x_light_pred_emp - x0_emp[:, c_sat:]) ** 2).mean()

    return loss_sat + 5.0 * loss_lightning, loss_sat, loss_lightning

def full_image_generation(
    model,
    batch,
    steps=256,
    device="cuda",
    parametrization="standard",
    interpolation="linear",
    nb_element=1,
    normalize_input=True,
    use_residual=True,
    schedule_power=1.0,
    bridge_sigma=0.5,
    bridge_sigma_min=1e-3,
):
    """
    Non-uniform timestep schedule for the Euler solver.

    Visited nodes are t = 1 - u**schedule_power,  u in linspace(0, 1, steps+1):
      - schedule_power = 1.0  -> uniform dt (the original behaviour)
      - schedule_power > 1.0  -> start-heavy (finer steps near t=1, the noise end)
      - schedule_power < 1.0  -> end-heavy  (finer steps near t=0, the data end)

    The per-step dt is taken from the grid, so integration stays consistent and
    the final state still equals the model's x-prediction at the smallest t.
    """
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

        batch_size, nb_channel, nb_context, h, w = x_context.shape
        x_t = torch.randn(batch_size, nb_channel, nb_forecasted_frame, h, w, device=device)
        # Fixed noise endpoint (the t=1 source) for the bridge solver.
        x1_init = x_t.clone()

        # Non-uniform timestep grid: t descends 1 -> 0.
        u_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)
        t_nodes = 1.0 - u_grid ** schedule_power  # [1, ..., 0]

        for i in range(steps):
            t_val = t_nodes[i].item()
            d_const = (t_nodes[i] - t_nodes[i + 1]).item()

            t_batch = torch.full((batch_size,), t_val, device=device)
            d_batch = torch.full((batch_size,), d_const, device=device)

            # to comment if false
            x_context_t = x_context

            model_input = torch.cat([x_context_t, x_t], dim=2)
            context_global = torch.cat([context_info, d_batch.unsqueeze(1), t_batch.unsqueeze(1)], dim=1)

            sat_x_pred, lightning_x_pred = model(
                model_input[:, :c_sat].float(), model_input[:, c_sat:].float(), context_global.float()
            )

            x_pred = torch.cat([sat_x_pred, lightning_x_pred], dim=1)[:, :, model.context_frames:]

            # Euler step: x_{t-dt} = x_t - v(x_t, t) * dt
            # For linear:     alpha(t) = 1 - t      => v = (x_t - x_pred) / t
            # For polynomial: alpha(t) = 1 - sqrt(t) => v = (x_t - x_pred) / (2 * t)
            # For bridge:     v = (x1_init - x_pred) + (c'/c)(x_t - mu_t),
            #                 mu_t = (1-t) x_pred + t x1_init  (x1_init = fixed noise source)
            if interpolation == "bridge":
                _, cp_over_c = bridge_coeffs(t_val, bridge_sigma, bridge_sigma_min)
                mu_t = (1.0 - t_val) * x_pred + t_val * x1_init
                v_t = (x1_init - x_pred) + cp_over_c * (x_t - mu_t)
                x_t = x_t - v_t * d_const
            else:
                if interpolation == "polynomial":
                    s_theta = (x_t - x_pred) / (2 * t_val + 1e-8)
                else:
                    s_theta = (x_t - x_pred) / t_val
                x_t = x_t - s_theta * d_const
            x_t = x_t.clamp(-7, 8)

        if use_residual:
            x_t = denormalize_residual(x_t, c_sat, device)
            x_t = x_t + last_context.expand_as(x_t)

        # keep clipped values from last context frame
        x_t = torch.where(last_context == CLIP_MIN, last_context, x_t)

        generated = x_t.cpu()
        target    = batch_data[:, :, model.context_frames:].cpu()

    model.train()
    return generated, target