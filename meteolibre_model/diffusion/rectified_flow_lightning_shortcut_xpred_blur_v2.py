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


def structured_gaussian_noise(shape, device, dtype=torch.float32, rho=0.50, generator=None):
    """Structured Gaussian noise with shared and independent components.

    .. math::
        \\epsilon_{c,t} = \\sqrt{\\rho}\\, \\epsilon_{\\text{shared}}
                       + \\sqrt{1-\\rho}\\, \\epsilon_{c,t}^{\\text{indep}}

    The shared component is a single 2D Gaussian field per batch element with
    shape ``(B, 1, 1, H, W)``, correlated across **all** channels and temporal
    frames.  The independent component is fully i.i.d. per channel and per
    timestep ``(B, C, T, H, W)``.

    This follows the partially-shared-noise construction recommended in the
    multi-view / video diffusion literature (e.g. Theiss et al. 2024, arXiv
    2412.03756) rather than the fully-rank-one (rho=1) extreme.

    Args:
        shape: ``(B, C, T, H, W)``.
        rho: correlation strength in [0, 1].
            * rho=1.0 → fully shared (rank-one, identical across C and T).
            * rho=0.0 → fully independent (standard ``torch.randn``).
            * rho=0.90 (default) → 90 % shared, 10 % independent.
        generator: optional ``torch.Generator`` for reproducibility.
    """
    if len(shape) != 5:
        raise ValueError(f"Expected a 5D (B, C, T, H, W) shape, got {shape}")
    batch, channels, temporal, height, width = shape

    if rho >= 1.0:
        shared = torch.randn(
            batch, 1, 1, height, width,
            device=device, dtype=dtype, generator=generator,
        )
        return shared.expand(batch, channels, temporal, height, width)
    elif rho <= 0.0:
        return torch.randn(
            batch, channels, temporal, height, width,
            device=device, dtype=dtype, generator=generator,
        )
    else:
        sqrt_rho = math.sqrt(rho)
        sqrt_omr = math.sqrt(1.0 - rho)
        shared = torch.randn(
            batch, 1, 1, height, width,
            device=device, dtype=dtype, generator=generator,
        )
        independent = torch.randn(
            batch, channels, temporal, height, width,
            device=device, dtype=dtype, generator=generator,
        )
        return sqrt_rho * shared.expand(
            batch, channels, temporal, height, width
        ) + sqrt_omr * independent


# -- Spectral ("red") context-noise augmentation ----------------------------
# Radial-PSD fit on near-obs IR frames from 3 rollout dates (see
# linear_18july_sde/fit_ir_psd.py):  P(k) = P0 / (1 + (k/k0)^alpha) with
# k0 = 0.0165 cyc/px (~80 km at 0.012 deg) and alpha = 4.65. Only the SHAPE
# is used — synthesized noise is renormalized to unit variance. The IR shape
# is assumed representative for all satellite channels.
SPEC_NOISE_K0 = 0.0165
SPEC_NOISE_ALPHA = 4.65
_spectral_filter_cache: dict = {}


def _spectral_filter(height, width, device):
    """sqrt(PSD) radial filter for spectral synthesis (cached per H/W/device)."""
    key = (height, width, str(device))
    filt = _spectral_filter_cache.get(key)
    if filt is None:
        fy = torch.fft.fftfreq(height, device=device).unsqueeze(1)
        fx = torch.fft.fftfreq(width, device=device).unsqueeze(0)
        k = torch.sqrt(fy * fy + fx * fx)
        filt = torch.sqrt(
            1.0 / (1.0 + (k.clamp_min(1e-8) / SPEC_NOISE_K0) ** SPEC_NOISE_ALPHA)
        )
        filt[0, 0] = 0.0  # kill DC — marginal/mean drift is a separate failure
        _spectral_filter_cache[key] = filt
    return filt


def spectral_gaussian_noise(shape, device, dtype=torch.float32, generator=None):
    """Unit-variance Gaussian field with the fitted IR radial PSD (red noise).

    Synthesized by filtering white noise in Fourier space:
    ``noise = IFFT(FFT(white) * sqrt(P(k)))``, then renormalized to unit
    variance per (B, C, T) slice. FFTs run in float32; result is cast to
    ``dtype``.
    """
    if len(shape) != 5:
        raise ValueError(f"Expected a 5D (B, C, T, H, W) shape, got {shape}")
    *_, height, width = shape
    white = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    filt = _spectral_filter(height, width, device)
    out = torch.fft.ifft2(torch.fft.fft2(white, norm="ortho") * filt, norm="ortho").real
    out = out / out.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return out.to(dtype)


def get_x_t_rf(x0, x1, t, interpolation="linear", poly_power=10.0):
    """
    Get the interpolated point x_t based on the chosen schedule.
    Convention: t=0 -> data x0, t=1 -> noise x1.  alpha(t) is the DATA
    coefficient:  x_t = alpha(t) * x0 + (1 - alpha(t)) * x1.

    - 'linear':      alpha = 1 - t
    - 'polynomial':  alpha = 1 - sqrt(t)   (mild noise bias)
    - 'rev_poly':    alpha = (1 - t)**poly_power   (HIGH-NOISE bias)
        The data coefficient stays near 0 (almost pure noise) for t in
        [~0.1, 1] and rises *rapidly* to 1 (pure data) only as t -> 0. This is
        the schedule to use when you want the model to operate in a high-noise
        regime for most of the trajectory and do all of its denoising in a
        narrow window near the data end (t -> 0).

        poly_power vs alpha(0.1):
            p=10 -> 0.35  (default),
            p=15 -> 0.21,  p=20 -> 0.12,
            p=22 -> 0.10  (matches the 'alpha(0.1) ~= 0.1' target),
            p=25 -> 0.07
        p=1 collapses to 'linear'. Start around p=10-15 and tune.

        IMPORTANT: the existing 'polynomial' branch (alpha = 1 - sqrt(t)) is a
        DIFFERENT family (the exponent sits on t, not on (1-t)). Lowering its
        exponent flattens the curve and can NOT produce this high-noise cliff
        shape -- it just keeps alpha low everywhere, including near t=0.
    """
    if interpolation == "linear":
        return (1 - t) * x0 + t * x1
    elif interpolation == "polynomial":
        alpha = 1 - t ** 0.5
        return alpha * x0 + (1 - alpha) * x1
    elif interpolation == "rev_poly":
        alpha = (1 - t) ** poly_power
        return alpha * x0 + (1 - alpha) * x1
    else:
        raise ValueError(f"Unknown interpolation schedule: {interpolation}")


def apply_blur_with_sigma_batched(x, blur_sigma, n_bins=8, min_kernel=0, sigma_factor=2):
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

        if bin_idx <= 1:
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
    model, batch, device, sigma=15.0, parametrization="standard", interpolation="linear", use_residual=True,
    grad_weight=0.,
    poly_power=10.0,
    noise_rho=0.,
    temporal_weight_scale=1.0,
    context_spec_noise=0.,
    context_spec_noise_prob=0.2,
    context_spec_noise_frame_ramp=0.0,
    d_x0_blur_prob=0.,
    d_x0_blur_scale=5.0,
    temporal_grad_weight=0.,
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

    # Structured Gaussian prior (partially shared across channels AND
    # temporal frames):  eps = sqrt(rho)*eps_shared + sqrt(1-rho)*eps_indep
    # Prevents the model's channel projection from fully suppressing the
    # source by averaging independent channel-noise, while retaining some
    # band-specific stochastic structure.  rho=1 collapses to the previous
    # fully-shared construction; rho=0 recovers standard i.i.d. noise.
    x1 = structured_gaussian_noise(x0.shape, x0.device, x0.dtype, rho=noise_rho)

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
        # Applied to only 50% of samples so the model still frequently sees
        # clean context (prevents over-regularization / loss of calibration).
        amplitude_jitter_std = 0.10
        amplitude_jitter_prob = 0.1
        jitter_mask = (torch.rand(b, device=device) < amplitude_jitter_prob).view(b, 1, 1, 1, 1)
        scale = 1.0 + amplitude_jitter_std * torch.randn(b, 1, 1, 1, 1, device=device)
        x_context_t = x_context_t * (jitter_mask * scale + ~jitter_mask)
    else:
        x_context_t = x_context

    # Spectrally-shaped ("red") additive context noise.
    #
    # Motivation (RAPSD measurements on SDE rollouts, 3 dates, see
    # linear_18july_sde/psd_rollout.py): the AR-rollout degradation is NOT
    # white pixel noise — it is a smooth red-spectrum anomaly field (plus
    # marginal drift). White additive noise is maximally off-manifold for
    # this red-spectrum data and caused the catastrophic blur collapse in
    # earlier XPs. The red noise injects energy only where the data has
    # power (fitted IR PSD: k0 ~ 80 km, alpha=4.65), so the per-band SNR
    # stays balanced and the model is never pushed to discard the
    # high-frequency context channel. IR shape assumed for all channels.
    #
    # Amplitude reuses the blur schedule draw (t_emp_blur) when the blur aug
    # is active, so blur/jitter/red-noise form ONE correlated degradation
    # bundle on the same samples; otherwise a fresh logit-normal draw is
    # used. Optional frame ramp: OLDER context frames (the most degraded in
    # a real rollout queue) receive proportionally more noise.
    #
    # Train-only (inference uses clean context).
    if context_spec_noise > 0:
        if sigma > 0:
            amp_draw = t_emp_blur  # correlated with the blur bundle
        else:
            eps = torch.randn(num_emp, device=device)
            amp_draw = torch.sigmoid(1.4 + 1.8 * eps).clamp(1e-4, 1 - 1e-4)
        spec_amp = amp_draw * context_spec_noise  # (B,)
        spec_mask = (
            torch.rand(num_emp, device=device) < context_spec_noise_prob
        ).view(num_emp, 1, 1, 1, 1)
        spec_noise = spectral_gaussian_noise(
            x_context_t.shape, device, x_context_t.dtype
        )
        if context_spec_noise_frame_ramp > 0:
            # age: 1.0 for the OLDEST context frame -> frame_ramp for the newest
            n_ctx = x_context_t.shape[2]
            age = torch.linspace(1.0, 0.0, n_ctx, device=device)
            frame_scale = (
                context_spec_noise_frame_ramp
                + (1.0 - context_spec_noise_frame_ramp) * age
            )
            spec_noise = spec_noise * frame_scale.view(1, 1, n_ctx, 1, 1)
        x_context_t = x_context_t + spec_mask * spec_amp.view(num_emp, 1, 1, 1, 1) * spec_noise

    # --- Data-component degradation D(x0) on the interpolant input ----------
    # AR-rollout robustification for the RUNNING STATE x_t (the context augs
    # above only cover the context frames). At inference, the data component
    # of x_t is the model's own accumulated output — smooth / under-dispersed
    # — while in training x_t = (1-t)*x0_clean + t*x1 has a perfect clean
    # data component. This block degrades the DATA SIDE of the interpolant:
    #     x_t = (1-t) * D(x0) + t * x1,     regression target = CLEAN x0,
    # so the model learns to sharpen a degraded data component instead of
    # trusting it. Key properties:
    #   - D = Gaussian blur only (the dominant rollout degradation), drawn
    #     INDEPENDENTLY of the context-aug bundle: at inference the two
    #     degradations are not perfectly correlated (AR step 1 = clean
    #     context + degraded x_t; late AR steps = both degraded).
    #   - Target stays clean x0: do NOT re-derive the velocity target from
    #     the degraded path (v = x1 - D(x0) would make the optimal
    #     x-prediction D(x0) — a blur-reproduction model). The 1/t^2-weighted
    #     x-loss w.r.t. the clean endpoints below is unchanged.
    #   - No sampler change needed: for the linear path, Euler integration
    #     still terminates at x_pred regardless of the intermediate
    #     trajectory, so inference is untouched.
    #   - Train-only. d_x0_blur_prob=0 disables (backward compatible).
    if d_x0_blur_prob > 0:
        d_mask = torch.rand(num_emp, device=device) < d_x0_blur_prob
        x0_path = x0_emp
        if d_mask.any():
            # Blur only the selected subset so masked samples are EXACTLY
            # identity (binning in apply_blur_with_sigma_batched could
            # otherwise leak a small blur onto sigma=0 samples).
            eps = torch.randn(int(d_mask.sum()), device=device)
            d_draw = torch.sigmoid(1.4 + 1.8 * eps).clamp(1e-4, 1 - 1e-4)
            d_blur_sigma = d_draw * d_x0_blur_scale  # (B_sel,)
            x0_path = x0_emp.clone()
            x0_path[d_mask] = apply_blur_with_sigma_batched(
                x0_emp[d_mask], d_blur_sigma
            )
    else:
        x0_path = x0_emp

    xt_emp = get_x_t_rf(x0_path, x1_emp, t_emp.view(num_emp,1,1,1,1), interpolation, poly_power=poly_power)

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
        weight = (1.0 / (t_emp.view(b, 1, 1, 1, 1) + 1e-2) ** 2).clamp(0.9, 10.)
    else:
        # linear / rev_poly: empirical 1/t^2 upweighting of small t.
        # For 'rev_poly' the small-t region [0, ~0.1] is exactly where the
        # data/noise transition (and thus the velocity) lives, so this
        # upweights the part that matters; the high-noise tail t>0.1 still
        # receives weight ~1-4 (not zeroed). If you instead want the model to
        # train harder across the whole noise range, switch this branch to a
        # uniform weight (=1.0) or to the velocity-matching weight
        # (1-t)**(2*(poly_power-1)).
        weight = (1.0 / (t_emp.view(b, 1, 1, 1, 1) + 1e-2) ** 2).clamp(0.9, 10.)

    # Temporal weighting: upweight later forecast frames so the model learns
    # harder on frames farther in the future (frame 0 -> w=1, frame 1 -> w=2,
    # frame 2 -> w=3, ...).  The ramp is normalized to mean 1 so the overall
    # loss magnitude is preserved.  temporal_weight_scale=0 disables it
    # (uniform), 1.0 = full linear ramp.
    n_forecast = x0_emp.shape[2]
    if temporal_weight_scale > 0 and n_forecast > 1:
        ramp = torch.arange(1, n_forecast + 1, device=device, dtype=weight.dtype)
        ramp = ramp / ramp.mean()                     # mean == 1
        blend = (1.0 - temporal_weight_scale) + temporal_weight_scale * ramp
        weight = weight * blend.view(1, 1, n_forecast, 1, 1)

    # direct x-loss
    err_sat = weight * (x_sat_pred_emp - x0_emp[:, :c_sat]) ** 2
    loss_sat     = err_sat[mask_emp].mean()
    loss_lightning = (weight * (x_light_pred_emp - x0_emp[:, c_sat:]) ** 2).mean()

    # --- Horizontal-gradient regularization (FastNet-style artifact suppressor) ---
    # Adapted from Dunstan et al. 2026 (FastNet, AIES-D-25-0090.1): penalize
    # mismatches in the spatial derivatives (∂/∂y, ∂/∂x) of the predicted field
    # vs the target. FastNet credits this as the PRIMARY lever for suppressing
    # the nonphysical artifacts that compound during autoregressive rollout.
    # Applied here on the x-prediction (clean denoised field) and weighted by
    # the same per-t factor as the main x-loss, so it focuses on data-end
    # predictions (t→0) where gradients are meaningful and barely penalizes
    # near-noise samples (t→1). grad_weight=0 disables it (backward compatible).
    # Lightning is EXCLUDED: its intermittent on/off structure makes legitimate
    # spatial gradients large and sharp, so a squared-gradient penalty risks
    # over-smoothing flashes instead of suppressing artifacts.
    if grad_weight > 0:
        # sat spatial gradients on (H, W) dims, masked like the main sat loss
        gy_sp, gx_sp = torch.gradient(x_sat_pred_emp, dim=(-2, -1))
        gy_st, gx_st = torch.gradient(x0_emp[:, :c_sat], dim=(-2, -1))
        grad_err_sat = (gy_sp - gy_st) ** 2 + (gx_sp - gx_st) ** 2
        loss_grad_sat = (weight * grad_err_sat)[mask_emp].mean()
    else:
        loss_grad_sat = torch.tensor(0.0, device=device)

    # --- Temporal-gradient regularization (FastNet-style, forecast-time axis) ---
    # Complement to the spatial term above: penalize mismatches in the
    # adjacent-frame finite difference (∂/∂T) of predicted vs target. The
    # spatial term kills per-frame spatial artifacts; this kills frame-to-frame
    # jitter / implausible evolution that compounds during AR rollout. On the
    # residual target x0[f] = data[f] - last_context  =>  Δx0 = Δdata, so this
    # is the true inter-frame evolution in normalized-residual units; the
    # prediction lives in the same space, so the comparison is well-posed.
    # Uses a clean forward difference so only forecast frames are involved
    # (the context→forecast-0 transition is excluded by construction). Both
    # adjacent frames must be valid (masked) for the error to count. Lightning
    # excluded for the same reason as the spatial term. Backward compatible:
    # temporal_grad_weight=0 disables it.
    if temporal_grad_weight > 0:
        # later-frame weight when the temporal ramp is active, else broadcast
        wt = weight if weight.shape[2] == 1 else weight[:, :, 1:]

        dT_pred = x_sat_pred_emp[:, :, 1:] - x_sat_pred_emp[:, :, :-1]
        dT_tgt = x0_emp[:, :c_sat][:, :, 1:] - x0_emp[:, :c_sat][:, :, :-1]
        pair_sat = mask_emp[:, :, 1:] & mask_emp[:, :, :-1]
        loss_tgrad_sat = (wt * (dT_pred - dT_tgt) ** 2)[pair_sat].mean()
    else:
        loss_tgrad_sat = torch.tensor(0.0, device=device)

    total = (loss_sat + 5.0 * loss_lightning
             + grad_weight * loss_grad_sat
             + temporal_grad_weight * loss_tgrad_sat)
    return total, loss_sat, loss_lightning

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
    poly_power=10.0,
    noise_rho=0.90,
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
        # Match training: structured Gaussian prior with partially shared
        # noise across channels and temporal frames.
        x_t = structured_gaussian_noise(
            (batch_size, nb_channel, nb_forecasted_frame, h, w),
            device=device,
            rho=noise_rho,
        ).clone()

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

            # Integration step: Euler  x_{t-dt} = x_t - v*dt
            #   linear:      v = (x_t - x_pred) / t
            #   polynomial:  v = (x_t - x_pred) / (2*t)
            #   rev_poly:    v = p*(1-t)^(p-1)/(1-(1-t)^p) * (x_t - x_pred)
            if interpolation == "polynomial":
                s_theta = (x_t - x_pred) / (2 * t_val + 1e-8)
            elif interpolation == "rev_poly":
                # velocity of alpha=(1-t)^p path, expressed via x_pred:
                #   dx/dt = p*(1-t)^(p-1) / (1-(1-t)^p) * (x_t - x_pred)
                # -> behaves like 1/t (linear) as t->0 and like 0 as t->1.
                omt = 1.0 - t_val
                ratio = poly_power * omt ** (poly_power - 1) / (1.0 - omt ** poly_power + 1e-8)
                s_theta = (x_t - x_pred) * ratio
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
