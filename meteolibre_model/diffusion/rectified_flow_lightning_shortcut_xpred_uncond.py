"""
Unconditional Shortcut Rectified Flow (x-prediction) for weather generation.
Generates frames directly from noise — no context frames are passed to the model.
https://arxiv.org/pdf/2410.12557
"""

import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from meteolibre_model.diffusion.utils import (
    MEAN_CHANNEL_WORLD_ELEVATION_RADAR,
    STD_CHANNEL_WORLD_ELEVATION_RADAR,
    MEAN_LIGHTNING,
    STD_LIGHTNING,
)

CLIP_MIN = -4
SHORTCUT_M = 128
SHORTCUT_K = 0.25


def normalize(sat_data, lightning_data, device):
    sat_data = (
        sat_data
        - MEAN_CHANNEL_WORLD_ELEVATION_RADAR.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
    ) / STD_CHANNEL_WORLD_ELEVATION_RADAR.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
    sat_data = sat_data.clamp(CLIP_MIN, 4)

    lightning_data = (
        lightning_data
        - MEAN_LIGHTNING.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
    ) / STD_LIGHTNING.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
    lightning_data = lightning_data.clamp(CLIP_MIN, 10)

    return sat_data, lightning_data


def denormalize(sat_data, lightning_data, device):
    sat_data = (
        sat_data.to(device) * STD_CHANNEL_WORLD_ELEVATION_RADAR.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
        + MEAN_CHANNEL_WORLD_ELEVATION_RADAR.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
    )
    lightning_data = (
        lightning_data.to(device) * STD_LIGHTNING.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
        + MEAN_LIGHTNING.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
    )
    return sat_data, lightning_data


def get_x_t_rf(x0, x1, t, interpolation="linear"):
    if interpolation == "linear":
        return (1 - t) * x0 + t * x1
    elif interpolation == "polynomial":
        alpha = 1 - t ** 0.5
        return alpha * x0 + (1 - alpha) * x1
    else:
        raise ValueError(f"Unknown interpolation schedule: {interpolation}")


def trainer_step(
    model, batch, device, sigma=0.0, parametrization="standard", interpolation="linear"
):
    if parametrization != "standard":
        raise ValueError("Only 'standard' parametrization is supported for x-prediction.")

    # (B, C, T, H, W)
    sat_data = batch["sat_patch_data"].permute(0, 2, 1, 3, 4)
    lightning_data = batch["lightning_patch_data"].permute(0, 2, 1, 3, 4)

    b, c_sat, t_dim, h, w = sat_data.shape

    mask_data_sat = sat_data != CLIP_MIN

    sat_data, lightning_data = normalize(sat_data, lightning_data, device)
    batch_data = torch.cat([sat_data, lightning_data], dim=1)

    # Unconditional: target is ALL frames (no split into context / forecast)
    x0 = batch_data[:, :, model.context_frames:]
    mask_emp = mask_data_sat[:, :, model.context_frames:]

    context_info = batch["spatial_position"]
    x1 = torch.randn_like(x0)

    t_emp = torch.rand(b, device=device)

    xt_emp = get_x_t_rf(x0, x1, t_emp.view(b, 1, 1, 1, 1), interpolation)

    if interpolation == "linear":
        da_dt = torch.full_like(t_emp, -1.0)
    else:
        da_dt = -0.5 / (t_emp ** 0.5 + 1e-8)
    da_dt = da_dt.view(b, 1, 1, 1, 1)

    # Unconditional: model receives only the noisy target frames (no context prepended)
    context_global_emp = torch.cat(
        [context_info, t_emp.unsqueeze(1), torch.zeros_like(t_emp).unsqueeze(1)], dim=1
    )

    sat_x_pred_emp, lightning_x_pred_emp = model(
        xt_emp[:, :c_sat].float(),
        xt_emp[:, c_sat:].float(),
        context_global_emp.float(),
    )

    # No slicing needed — output shape matches target shape directly
    x_sat_pred_emp = sat_x_pred_emp
    x_light_pred_emp = lightning_x_pred_emp

    weight = 1.0 / (t_emp.view(b, 1, 1, 1, 1) + 1e-2) ** 2
    weight = weight.clamp(0.9, 10.0)

    loss_sat = (weight * (x_sat_pred_emp - x0[:, :c_sat]) ** 2)[mask_emp].mean()
    loss_lightning = (weight * (x_light_pred_emp - x0[:, c_sat:]) ** 2).mean()

    return loss_sat + 5.0 * loss_lightning, loss_sat, loss_lightning


def full_image_generation(
    model,
    batch,
    steps=128,
    device="cuda",
    parametrization="standard",
    interpolation="linear",
    nb_element=1,
    normalize_input=True,
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

        context_info = batch["spatial_position"].to(device)[0:nb_element]
        batch_size, nb_channel = batch_data.shape[0], batch_data.shape[1]

        x_t = torch.randn(batch_size, nb_channel, nb_forecasted_frame, h, w, device=device)

        d_const = 1.0 / steps
        t_val = 1.0

        for _ in range(steps):
            t_batch = torch.full((batch_size,), t_val, device=device)
            d_batch = torch.full((batch_size,), 0.0, device=device)

            context_global = torch.cat(
                [context_info, t_batch.unsqueeze(1), d_batch.unsqueeze(1)], dim=1
            )

            # Unconditional: model receives only the noisy frames
            sat_x_pred, lightning_x_pred = model(
                x_t[:, :c_sat].float(),
                x_t[:, c_sat:].float(),
                context_global.float(),
            )

            # No slicing needed
            x_pred = torch.cat([sat_x_pred, lightning_x_pred], dim=1)

            if interpolation == "polynomial":
                s_theta = (x_t - x_pred) / (2 * t_val + 1e-8)
            else:
                s_theta = (x_t - x_pred) / t_val

            x_t = x_t - s_theta * d_const
            x_t = x_t.clamp(-7, 7)

            t_val -= d_const

        generated = x_t.cpu()
        target = batch_data[:, :, model.context_frames:].cpu()

    model.train()
    return generated, target
