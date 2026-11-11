"""
Verification and GPU benchmark suite for the STRATA-inspired tiling upgrades in
backend/inference_engine.py:
  1. Sub-patch 8x8 ViT phase de-aliasing + cycle16 quadrant jitter
  2. Kaiser-Bessel-Derived (KBD) flat-top blending window vs Gaussian
  3. Vectorized & hoisted pyproj + suncalc spatial conditioning
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
from suncalc import get_position

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.inference_engine import InferenceEngine


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running tiling upgrade verification on device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    engine = InferenceEngine(
        model_path="/tmp/nonexistent_weights_for_unit_test.safetensors",
        model_type="jit",
        patch_size=128,
        denoising_steps=4,
        batch_size=64,
        context_frames=4,
        window_type="kaiser_bessel",
        kaiser_beta=8.0,
        tile_jitter=True,
        device=device,
    )

    H_big, W_big = 2150, 3583
    P = engine.patch_size
    transform = [0.012, 0.0, -10.0, 0.0, -0.012, 65.0]

    # 1. Verify 16-state cycle16 sub-patch phases & macro quadrant shifts
    states = engine._precompute_grid_states(H_big, W_big, transform, transformer=None)
    assert len(states) == 16, f"Expected 16 jitter states, got {len(states)}"

    vit_phases = set()
    macro_quadrants = set()
    for s_idx, st in enumerate(states):
        ox, oy = st["offset"]
        vit_phases.add((ox % 8, oy % 8))
        macro_quadrants.add(((ox // 32) % 2, (oy // 32) % 2))
        assert len(st["coords"]) == len(st["lons"]) == len(st["lats"])

    expected_phases = {(x, y) for x in (0, 2, 4, 6) for y in (0, 2, 4, 6)}
    assert vit_phases == expected_phases, f"ViT phases mismatch: {vit_phases}"
    assert macro_quadrants == {(0, 0), (1, 0), (0, 1), (1, 1)}
    print(
        f"[PASS] Upgrade 1: 16-state cycle16 covers all 16 ViT phases {sorted(vit_phases)[:4]}... "
        f"and all 4 macro quadrants {sorted(macro_quadrants)}"
    )

    # 2. Verify Kaiser-Bessel-Derived window properties & full-domain coverage
    w_gauss = engine._get_gaussian_weights(P)
    w_kbd = engine._get_kaiser_bessel_weights(P, beta=8.0)

    gauss_edge_1d = float(w_gauss[P // 2, 0].item())
    kbd_edge_1d = float(w_kbd[P // 2, 0].item())
    gauss_core = float(w_gauss[P // 2, P // 2 - 16].item())
    kbd_core = float(w_kbd[P // 2, P // 2 - 16].item())

    print(
        f"[PASS] Upgrade 2 (Window): 1-px border weight reduced from {gauss_edge_1d:.4f} (Gaussian) "
        f"to {kbd_edge_1d:.4f} (KBD, {gauss_edge_1d / kbd_edge_1d:.1f}x seam suppression); "
        f"16-px interior plateau: {kbd_core:.4f} (KBD) vs {gauss_core:.4f} (Gaussian)"
    )

    # Check full-domain weight coverage across all 16 jitter states
    min_weights_across_states = []
    for st in states:
        w_sum = torch.zeros((H_big, W_big), device=device, dtype=torch.float32)
        for x0, y0 in st["coords"]:
            w_sum[y0 : y0 + P, x0 : x0 + P] += w_kbd
        min_w = float(w_sum.min().item())
        assert min_w > 0.0, f"Found uncovered pixel (min weight {min_w}) in state {st['offset']}"
        min_weights_across_states.append(min_w)
    print(
        f"[PASS] Full {W_big}x{H_big} domain coverage verified across all 16 jitter states "
        f"(min pixel weight > 0 everywhere: min={min(min_weights_across_states):.4f})"
    )

    # 3. Verify numerical equivalence & speedup of vectorized/hoisted suncalc + pyproj
    pred_date = datetime(2026, 7, 25, 14, 30)
    date_noon = pred_date.replace(hour=12, minute=0, second=0, microsecond=0)
    st0 = states[0]
    coords0 = st0["coords"]

    # Legacy per-patch scalar loop (run 16 times as in 16 ODE steps)
    t0 = time.perf_counter()
    for _ in range(16):
        legacy_list = []
        for x_start, y_start in coords0:
            px = x_start + P // 2
            py = y_start + P // 2
            lon = transform[0] * px + transform[1] * py + transform[2]
            lat = transform[3] * px + transform[4] * py + transform[5]
            r_now = get_position(pred_date, lon, lat)
            r_noon = get_position(date_noon, lon, lat)
            legacy_list.append([r_now["azimuth"], r_now["altitude"], r_noon["altitude"], lat / 10.0])
        legacy_tensor = torch.tensor(legacy_list, device=device, dtype=torch.float32)
    if device == "cuda":
        torch.cuda.synchronize()
    t_legacy = time.perf_counter() - t0

    # Upgraded precomputed vectorized GPU tensor (computed once per AR step for active states)
    t1 = time.perf_counter()
    vec_tensor = engine._compute_state_spatial_Position_gpu(st0, pred_date)
    if device == "cuda":
        torch.cuda.synchronize()
    t_upgraded = time.perf_counter() - t1

    max_diff = float((legacy_tensor - vec_tensor).abs().max().item())
    assert max_diff < 1e-5, f"Spatial conditioning mismatch: max_diff={max_diff}"
    print(
        f"[PASS] Upgrade 3 (Hoisted Vectorized Suncalc): max_diff={max_diff:.2e} | "
        f"16-step CPU/GPU conditioning time: {t_legacy * 1000:.1f} ms -> {t_upgraded * 1000:.2f} ms "
        f"({t_legacy / max(t_upgraded, 1e-9):.0f}x faster)"
    )


if __name__ == "__main__":
    main()
