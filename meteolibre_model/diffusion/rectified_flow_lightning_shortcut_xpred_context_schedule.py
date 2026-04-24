"""
Shortcut Rectified Flow with independent context noise schedule.

Key idea: instead of adding fixed noise+blur to the conditional context frames,
we corrupt the context with its own independent noise schedule `s`:

    x_context_t = (1 - s) * x_context + s * noise_context

The model receives `s` as conditioning signal (replacing the `d` shortcut slot
in context_global), so it learns to be robust to any level of context corruption.

At inference, s=0 → clean context (no corruption), same as standard conditional
generation.

This is designed to improve autoregressive rollout quality: during training the
model sees corrupted contexts at various levels, so at test time when context
comes from previous (imperfect) model outputs, the model is more robust.

Based on rectified_flow_lightning_shortcut_xpred_blur.py
"""

import torch

from meteolibre_model.diffusion.utils import (
    MEAN_CHANNEL_WORLD_ELEVATION_RADAR,
    STD_CHANNEL_WORLD_ELEVATION_RADAR,
    MEAN_LIGHTNING,
    STD_LIGHTNING,
    MEAN_SAT_RESIDUAL,
    STD_SAT_RESIDUAL,
    MEAN_LIGHTNING_RESIDUAL,
    STD_LIGHTNING_RESIDUAL,
)

CLIP_MIN = -4


# ---------------------------------------------------------------------------
# Normalize / denormalize (same as blur version)
# ---------------------------------------------------------------------------

def normalize(sat_data, lightning_data, device):
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
    sat_data = sat_data.clamp(CLIP_MIN, 4)

    lightning_data = (
        lightning_data
        - MEAN_LIGHTNING.unsqueeze(0)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .to(device)
    ) / STD_LIGHTNING.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
    lightning_data = lightning_data.clamp(CLIP_MIN, 10)

    return sat_data, lightning_data


def denormalize(sat_data, lightning_data, device):
    sat_data = (
        sat_data.to(device)
        * STD_CHANNEL_WORLD_ELEVATION_RADAR.unsqueeze(0)
            .unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
        + MEAN_CHANNEL_WORLD_ELEVATION_RADAR.unsqueeze(0)
            .unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
    )

    lightning_data = (
        lightning_data.to(device)
        * STD_LIGHTNING.unsqueeze(0)
            .unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
        + MEAN_LIGHTNING.unsqueeze(0)
            .unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
    )

    return sat_data, lightning_data


def normalize_residual(x0, c_sat, device):
    mean = torch.cat([MEAN_SAT_RESIDUAL, MEAN_LIGHTNING_RESIDUAL]).to(device).view(1, -1, 1, 1, 1)
    std  = torch.cat([STD_SAT_RESIDUAL,  STD_LIGHTNING_RESIDUAL ]).to(device).view(1, -1, 1, 1, 1)
    return (x0 - mean) / std


def denormalize_residual(x0, c_sat, device):
    mean = torch.cat([MEAN_SAT_RESIDUAL, MEAN_LIGHTNING_RESIDUAL]).to(device).view(1, -1, 1, 1, 1)
    std  = torch.cat([STD_SAT_RESIDUAL,  STD_LIGHTNING_RESIDUAL ]).to(device).view(1, -1, 1, 1, 1)
    return x0 * std + mean


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def get_x_t_rf(x0, x1, t, interpolation="linear"):
    if interpolation == "linear":
        return (1 - t) * x0 + t * x1
    elif interpolation == "polynomial":
        alpha = 1 - t ** 0.5
        return alpha * x0 + (1 - alpha) * x1
    else:
        raise ValueError(f"Unknown interpolation schedule: {interpolation}")


# ---------------------------------------------------------------------------
# Context noise schedule:  x_context_t = (1 - s) * x_context + s * noise
# ---------------------------------------------------------------------------

def corrupt_context(x_context, s):
    """
    Corrupt context frames with independent noise schedule s.

    Args:
        x_context: (B, C, T_ctx, H, W) clean context
        s:         (B,) noise level in [0, 1]

    Returns:
        x_context_t: (B, C, T_ctx, H, W) corrupted context
    """
    # Single noise pattern shared across all channels (correlated corruption)
    noise = torch.randn(x_context.shape[0], 1, x_context.shape[2], x_context.shape[3], x_context.shape[4], device=x_context.device)
    return (1 - s.view(-1, 1, 1, 1, 1)) * x_context + s.view(-1, 1, 1, 1, 1) * noise


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def trainer_step(
    model, batch, device, sigma=0.0, parametrization="standard",
    interpolation="linear", use_residual=True,
):
    """
    Training step with independent context noise schedule.

    sigma here controls the max context noise level. If sigma=0, no context
    corruption is applied (equivalent to the base model).
    """
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
        x0 = batch_data[:, :, model.context_frames:] - batch_data[:, :, model.context_frames - 1:model.context_frames]
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

    # --- Sample diffusion timestep t (stratified, same as blur) ---
    n_bins = 32
    bin_size = 1.0 / n_bins
    bin_indices = torch.randperm(n_bins, device=device).repeat_interleave(
        (num_emp + n_bins - 1) // n_bins
    )[:num_emp]
    t_emp = (bin_indices.float() + torch.rand(num_emp, device=device)) * bin_size
    t_emp = t_emp[torch.randperm(num_emp, device=device)]

    # --- Sample independent context noise schedule s ---
    if sigma > 0:
        # s is drawn uniformly in [0, sigma].
        # At sigma=1.0, s spans the full range → max augmentation.
        # At sigma=0.5, s ∈ [0, 0.5] → mild augmentation.
        s_emp = torch.rand(num_emp, device=device) # * sigma
        x_context_t = corrupt_context(x_context, s_emp)
    else:
        s_emp = torch.zeros(num_emp, device=device)
        x_context_t = x_context

    # --- Interpolate target ---
    xt_emp = get_x_t_rf(x0_emp, x1_emp, t_emp.view(num_emp, 1, 1, 1, 1), interpolation)

    # --- Weighting ---
    if interpolation == "polynomial":
        da_dt = -0.5 / (t_emp ** 0.5 + 1e-8)
    else:
        da_dt = torch.full_like(t_emp, -1.0)

    da_dt = da_dt.view(num_emp, 1, 1, 1, 1)

    # --- Model forward ---
    model_input_emp = torch.cat([x_context_t, xt_emp], dim=2)
    # context_global: [spatial_pos, t, s]  —  s replaces the old `d` shortcut slot
    context_global_emp = torch.cat(
        [context_info_emp, t_emp.unsqueeze(1), s_emp.unsqueeze(1)], dim=1
    )

    sat_x_pred_emp, lightning_x_pred_emp = model(
        model_input_emp[:, :c_sat].float(),
        model_input_emp[:, c_sat:].float(),
        context_global_emp.float(),
    )

    x_sat_pred_emp = sat_x_pred_emp[:, :, model.context_frames:]
    x_light_pred_emp = lightning_x_pred_emp[:, :, model.context_frames:]

    if interpolation == "polynomial":
        weight = 1.0 / (t_emp.view(b, 1, 1, 1, 1) + 1e-2) ** 2
    else:
        weight = 1.0 / (t_emp.view(b, 1, 1, 1, 1) + 1e-2) ** 2
    weight = weight.clamp(0.9, 10.0)

    # direct x-loss
    loss_sat = (weight * (x_sat_pred_emp - x0_emp[:, :c_sat]) ** 2)[mask_emp].mean()
    loss_lightning = (weight * (x_light_pred_emp - x0_emp[:, c_sat:]) ** 2).mean()

    return loss_sat + 5.0 * loss_lightning, loss_sat, loss_lightning


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

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
):
    """
    Full image generation with s=0 (clean context).

    During inference the context is always clean (s=0), which the model has
    seen during training. This is equivalent to the standard conditional
    generation pipeline.
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
        last_context = x_context[:, :, model.context_frames - 1:model.context_frames]

        context_info = batch["spatial_position"].to(device)[0:nb_element]

        batch_size, nb_channel, nb_context, h, w = x_context.shape
        x_t = torch.randn(batch_size, nb_channel, nb_forecasted_frame, h, w, device=device)

        d_const = 1.0 / steps
        t_val = 1.0

        for _ in range(steps):
            t_batch = torch.full((batch_size,), t_val, device=device)
            # s=0 at inference: clean context, no corruption
            s_batch = torch.full((batch_size,), 0.0, device=device)

            model_input = torch.cat([x_context, x_t], dim=2)
            context_global = torch.cat(
                [context_info, t_batch.unsqueeze(1), s_batch.unsqueeze(1)], dim=1
            )

            sat_x_pred, lightning_x_pred = model(
                model_input[:, :c_sat].float(),
                model_input[:, c_sat:].float(),
                context_global.float(),
            )

            x_pred = torch.cat([sat_x_pred, lightning_x_pred], dim=1)[
                :, :, model.context_frames:
            ]

            if interpolation == "polynomial":
                s_theta = (x_t - x_pred) / (2 * t_val + 1e-8)
            else:
                s_theta = (x_t - x_pred) / t_val
            x_t = x_t - s_theta * d_const
            x_t = x_t.clamp(-7, 8)

            t_val -= d_const

        if use_residual:
            x_t = denormalize_residual(x_t, c_sat, device)
            x_t = x_t + last_context.expand_as(x_t)

        # keep clipped values from last context frame
        x_t = torch.where(last_context == CLIP_MIN, last_context, x_t)

        generated = x_t.cpu()
        target = batch_data[:, :, model.context_frames:].cpu()

    model.train()
    return generated, target


# ---------------------------------------------------------------------------
# Autoregressive rollout generation
# ---------------------------------------------------------------------------

def autoregressive_generation(
    model,
    batch,
    steps=256,
    device="cuda",
    interpolation="linear",
    nb_element=1,
    use_residual=True,
    s_override=None,
):
    """
    Autoregressive rollout where each step uses the previous prediction as context.

    Args:
        s_override: float or None. If set, inject a fixed s>0 during inference
                    to simulate the training-time context corruption and potentially
                    improve robustness. Default None → s=0 (clean).

    Returns:
        all_generated: list of (generated, target) tuples per rollout step
    """
    model.eval()
    with torch.no_grad():
        sat_data = batch["sat_patch_data"].permute(0, 2, 1, 3, 4)
        lightning_data = batch["lightning_patch_data"].permute(0, 2, 1, 3, 4)

        b, c_sat, t_total, h, w = sat_data.shape
        _, c_lightning, _, _, _ = lightning_data.shape
        c_total = c_sat + c_lightning

        sat_data, lightning_data = normalize(sat_data, lightning_data, device=device)
        batch_data = torch.cat([sat_data, lightning_data], dim=1)[0:nb_element]

        context_info = batch["spatial_position"].to(device)[0:nb_element]

        n_context = model.context_frames
        n_pred = t_total - n_context  # frames per rollout step
        n_rollouts = n_pred // n_pred  # TODO: adapt for multi-step rollout

        # Initial context
        x_context = batch_data[:, :, :n_context]
        last_context = x_context[:, :, n_context - 1:n_context]

        nb_forecasted_frame = n_pred
        batch_size = nb_element

        x_t = torch.randn(batch_size, c_total, nb_forecasted_frame, h, w, device=device)

        d_const = 1.0 / steps
        t_val = 1.0

        for _ in range(steps):
            t_batch = torch.full((batch_size,), t_val, device=device)
            s_val = s_override if s_override is not None else 0.0
            s_batch = torch.full((batch_size,), s_val, device=device)

            model_input = torch.cat([x_context, x_t], dim=2)
            context_global = torch.cat(
                [context_info, t_batch.unsqueeze(1), s_batch.unsqueeze(1)], dim=1
            )

            sat_x_pred, lightning_x_pred = model(
                model_input[:, :c_sat].float(),
                model_input[:, c_sat:].float(),
                context_global.float(),
            )

            x_pred = torch.cat([sat_x_pred, lightning_x_pred], dim=1)[
                :, :, model.context_frames:
            ]

            if interpolation == "polynomial":
                s_theta = (x_t - x_pred) / (2 * t_val + 1e-8)
            else:
                s_theta = (x_t - x_pred) / t_val
            x_t = x_t - s_theta * d_const
            x_t = x_t.clamp(-7, 8)

            t_val -= d_const

        if use_residual:
            x_t = denormalize_residual(x_t, c_sat, device)
            x_t = x_t + last_context.expand_as(x_t)

        x_t = torch.where(last_context == CLIP_MIN, last_context, x_t)

        generated = x_t.cpu()
        target = batch_data[:, :, n_context:].cpu()

    model.train()
    return generated, target
