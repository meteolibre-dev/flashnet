"""
Spiral RoPE for 3D data (Time × Height × Width).

Implements Spiral RoPE (arXiv:2602.03227) for the spatial (H, W) dimensions
combined with standard 1D axial RoPE for the temporal (T) dimension.

Standard Axial 2D RoPE splits the head dimension into two halves (x, y) and
applies 1D RoPE independently. In the Fourier domain, this places ALL frequencies
on the horizontal and vertical axes only, making the attention mechanism
**insensitive to diagonal/oblique positional relationships**.

Spiral RoPE fixes this by distributing K groups of rotation frequencies across
K uniformly spaced directions (0°, 45°, 90°, 135° for K=4). Each group is
rotated by the patch position *projected* onto its assigned direction:

    t_k(h, w) = h·cos(φ_k) + w·sin(φ_k)

This gives multi-directional frequency coverage with zero extra parameters
and zero extra compute vs. Axial 2D RoPE.

References:
    - Spiral RoPE:  arXiv:2602.03227 (Liu et al., Feb 2026)
    - RoPE-Mixed:   arXiv:2403.13298 (Heo et al., ECCV 2024)
    - Original RoPE: arXiv:2104.09864 (Su et al., 2023)

Usage (drop-in replacement for RoPE3D in JiT3D_Modern):

    # In JiT3D_Modern.__init__, replace:
    #   self.rope = RoPE3D(head_dim, grid_t*2, grid_h*2, grid_w*2)
    # With:
    #   self.rope = RoPE3DSpiral(head_dim, grid_t*2, grid_h*2, grid_w*2,
    #                            n_spiral_directions=4)
    #
    # The forward signature is identical: rope(q, k, T, H, W)
"""

import math
import torch
import torch.nn as nn

from meteolibre_model.models.jit3d import precompute_freqs_cis, apply_rotary_emb


# ==============================================================================
# == Spiral Spatial Frequency Builder
# ==============================================================================

def build_spiral_spatial_freqs(
    d_spatial: int,
    n_directions: int,
    max_h: int,
    max_w: int,
    base: float = 10000.0,
    freq_scale: float = 1.0,
) -> torch.Tensor:
    """
    Precompute the Spiral RoPE frequency buffer for the 2D spatial grid.

    Uses the *grouped interleaved* frequency assignment from the paper (Eq. 13):
    base frequencies are paired and distributed round-robin across perpendicular
    direction pairs so every direction gets a mix of low and high frequencies.

    Args:
        d_spatial:   Total embedding dimension for the spatial part.
                     Must be divisible by ``4 * n_directions``.
        n_directions: Number K of uniformly-spaced spiral directions (must be even).
        max_h:       Maximum height in patch tokens.
        max_w:       Maximum width in patch tokens.
        base:        RoPE base frequency θ (default 10 000).
        freq_scale:  Global frequency multiplier (paper recommends 1.5 for ViT;
                     1.0 matches the standard RoPE frequencies).

    Returns:
        Complex tensor of shape ``(max_h, max_w, d_spatial // 2)``.
    """
    K = n_directions
    n_base = d_spatial // 4                    # total unique base frequencies
    n_per_dir = d_spatial // (2 * K)           # complex rotation blocks per direction

    # --- Base frequency values:  θ_t = freq_scale · base^{-t/n_base}  ------------
    base_freqs = freq_scale / (
        base ** (torch.arange(n_base, dtype=torch.float32) / n_base)
    )  # (n_base,)

    # --- Direction angles:  φ_k = k·π/K  -----------------------------------------
    angles_k = torch.arange(K, dtype=torch.float32) * (math.pi / K)
    cos_a = torch.cos(angles_k)                # (K,)
    sin_a = torch.sin(angles_k)                # (K,)

    # --- Interleaved frequency assignment  (Eq. 13)  -----------------------------
    # For perpendicular pair index j (j = 0 … K/2-1):
    #   direction j   and   direction j+K/2  share the same frequency set
    #   frequencies at indices:  2j, 2j+1,  2j+K, 2j+K+1,  2j+2K, 2j+2K+1, …
    dir_freq_sets: list[torch.Tensor] = []
    for j in range(K // 2):
        indices: list[int] = []
        step = 0
        while len(indices) < n_per_dir:
            idx = 2 * j + step * K
            indices.append(idx)
            indices.append(idx + 1)
            step += 1
        indices = indices[:n_per_dir]           # exact trim (n_per_dir is always even)
        dir_freq_sets.append(base_freqs[indices])  # (n_per_dir,)

    # --- Build the full (H, W, d_spatial/2) complex buffer  ----------------------
    h_coords = torch.arange(max_h, dtype=torch.float32)
    w_coords = torch.arange(max_w, dtype=torch.float32)

    parts: list[torch.Tensor] = []
    for k in range(K):
        # Projected position:  t_k(h, w) = h·cos(φ_k) + w·sin(φ_k)
        proj = (h_coords[:, None] * cos_a[k]) + (w_coords[None, :] * sin_a[k])
        # proj: (H, W)

        freq_set = dir_freq_sets[k % (K // 2)]  # shared with perpendicular partner
        rot_angles = proj[..., None] * freq_set[None, None, :]
        # rot_angles: (H, W, n_per_dir)

        freqs_cis_k = torch.polar(torch.ones_like(rot_angles), rot_angles)
        parts.append(freqs_cis_k)

    return torch.cat(parts, dim=-1)  # (max_h, max_w, d_spatial // 2)


# ==============================================================================
# == RoPE3DSpiral  –  drop-in replacement for RoPE3D
# ==============================================================================

class RoPE3DSpiral(nn.Module):
    """
    3D Rotary Positional Embedding with Spiral RoPE for spatial dims.

    Head dimension layout::

        |<--- d_t --->|<---------- d_spatial (spiral H,W) ------------>|
                       | dir 0 | dir 1 | … | dir K-1 |

    * **Time** (T) uses standard 1D axial RoPE — time is genuinely 1-D.
    * **Space** (H, W) uses Spiral RoPE with K uniformly spaced directions.

    ``n_spiral_directions`` (K) controls the angular resolution:
        K=2  → equivalent to standard Axial 2D RoPE  (0°, 90°)
        K=4  → captures diagonals                     (0°, 45°, 90°, 135°)
        K=8  → fine angular coverage                  (every 22.5°)

    More directions = richer spatial encoding but each direction gets fewer
    frequency slots. K=4 or K=8 is recommended for weather / radar data.
    """

    def __init__(
        self,
        head_dim: int,
        max_t: int,
        max_h: int,
        max_w: int,
        n_spiral_directions: int = 4,
        base: float = 10000.0,
        freq_scale: float = 1.0,
        spatial_ratio: float = 2.0 / 3.0,
    ):
        """
        Args:
            head_dim:            Dimension per attention head.
            max_t:               Maximum temporal grid size (in patch tokens).
            max_h:               Maximum spatial height (in patch tokens).
            max_w:               Maximum spatial width (in patch tokens).
            n_spiral_directions: Number K of spiral directions (even, ≥ 2).
            base:                RoPE base frequency θ (default 10 000).
            freq_scale:          Frequency scaling factor. The paper uses 1.5
                                 for ViT; 1.0 keeps standard RoPE frequencies.
            spatial_ratio:       Target fraction of head_dim for spatial encoding.
                                 Default 2/3 mirrors the original 1/3-per-axis split.
        """
        super().__init__()

        K = n_spiral_directions
        assert K >= 2 and K % 2 == 0, (
            f"n_spiral_directions must be even and >= 2, got {K}"
        )

        # --- Dimension split ---------------------------------------------------
        # d_spatial must be divisible by 4*K (for clean interleaved assignment).
        # Round to the nearest valid value from the ideal spatial_ratio target.
        step = 4 * K
        d_spatial_ideal = int(head_dim * spatial_ratio)
        d_spatial = round(d_spatial_ideal / step) * step

        # Clamp so that d_t >= 2 and d_t is even
        d_spatial = max(step, min(d_spatial, head_dim - 2))
        d_t = head_dim - d_spatial
        if d_t % 2 != 0:
            d_spatial -= 2
            d_t += 2

        assert d_t > 0 and d_t % 2 == 0, (
            f"d_t must be > 0 and even, got {d_t}. "
            f"Try a smaller K or larger head_dim."
        )
        assert d_spatial % step == 0, (
            f"d_spatial must be divisible by 4*K={step}, got {d_spatial}"
        )

        self.d_t = d_t
        self.d_spatial = d_spatial
        self.K = K

        # --- Time: standard 1D RoPE --------------------------------------------
        self.register_buffer(
            "freqs_t",
            precompute_freqs_cis(d_t, max_t, base),
            persistent=False,
        )

        # --- Spatial: Spiral RoPE ----------------------------------------------
        spatial_freqs_cis = build_spiral_spatial_freqs(
            d_spatial, K, max_h, max_w, base=base, freq_scale=freq_scale
        )
        self.register_buffer("spatial_freqs_cis", spatial_freqs_cis, persistent=False)

    def extra_repr(self) -> str:
        return (
            f"d_t={self.d_t}, d_spatial={self.d_spatial}, K={self.K}, "
            f"directions=[{', '.join(f'{k*180/self.K:.0f}°' for k in range(self.K))}]"
        )

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        T: int,
        H: int,
        W: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 3D Spiral RoPE.

        Args:
            xq, xk: ``(B, num_heads, T*H*W, head_dim)``
            T, H, W: current grid dimensions (patch tokens).

        Returns:
            Rotated ``(xq, xk)`` with identical shapes.
        """
        # ---- split into time / spatial sub-vectors ----------------------------
        q_t, q_s = torch.split(xq, [self.d_t, self.d_spatial], dim=-1)
        k_t, k_s = torch.split(xk, [self.d_t, self.d_spatial], dim=-1)

        # ---- Time: 1D axial RoPE (identical to RoPE3D) -----------------------
        f_t = (
            self.freqs_t[:T]                             # (T, d_t/2)
            .view(T, 1, 1, -1)
            .expand(T, H, W, -1)
            .contiguous()
            .view(T * H * W, -1)                         # (T*H*W, d_t/2)
        )
        q_t, k_t = apply_rotary_emb(q_t, k_t, f_t)

        # ---- Spatial: Spiral RoPE ---------------------------------------------
        f_spatial = (
            self.spatial_freqs_cis[:H, :W]               # (H, W, d_spatial/2)
            .unsqueeze(0)
            .expand(T, H, W, -1)
            .contiguous()
            .view(T * H * W, -1)                         # (T*H*W, d_spatial/2)
        )
        q_s, k_s = apply_rotary_emb(q_s, k_s, f_spatial)

        # ---- reassemble -------------------------------------------------------
        xq_out = torch.cat([q_t, q_s], dim=-1)
        xk_out = torch.cat([k_t, k_s], dim=-1)
        return xq_out, xk_out


# ==============================================================================
# == Self-test
# ==============================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing RoPE3DSpiral on {device}\n")

    # --- Test 1: basic forward / backward --------------------------------------
    head_dim = 64
    T, H, W = 7, 8, 8
    B, num_heads = 2, 8

    for K in [2, 4, 8]:
        rope = RoPE3DSpiral(
            head_dim=head_dim,
            max_t=T * 2,
            max_h=H * 2,
            max_w=W * 2,
            n_spiral_directions=K,
            freq_scale=1.0,
        ).to(device)
        print(f"K={K}: {rope}")                              # extra_repr

        q = torch.randn(B, num_heads, T * H * W, head_dim, device=device, requires_grad=True)
        k = torch.randn(B, num_heads, T * H * W, head_dim, device=device, requires_grad=True)

        q_out, k_out = rope(q, k, T, H, W)
        assert q_out.shape == q.shape, f"Shape mismatch: {q_out.shape}"
        assert k_out.shape == k.shape, f"Shape mismatch: {k_out.shape}"

        loss = q_out.sum() + k_out.sum()
        loss.backward()
        assert q.grad is not None, "No gradient for q"
        print(f"  forward + backward OK  (d_t={rope.d_t}, d_spatial={rope.d_spatial})\n")

    # --- Test 2: compare K=2 output with axial RoPE3D (should be equivalent) ----
    from meteolibre_model.models.jit3d import RoPE3D

    print("Comparing K=2 Spiral RoPE vs Axial RoPE3D ...")
    rope_axial = RoPE3D(head_dim, max_t=T * 2, max_h=H * 2, max_w=W * 2).to(device)
    rope_spiral_k2 = RoPE3DSpiral(
        head_dim, max_t=T * 2, max_h=H * 2, max_w=W * 2, n_spiral_directions=2
    ).to(device)

    # K=2 Spiral RoPE has the *same mathematical structure* as Axial 2D RoPE
    # (directions 0° and 90°), but the dimension split may differ slightly
    # (RoPE3D splits into 3 chunks, Spiral puts 2/3 into spatial).
    # So we just verify they both run without error on the same input.
    q_test = torch.randn(1, 4, T * H * W, head_dim, device=device)
    k_test = torch.randn_like(q_test)

    q_a, k_a = rope_axial(q_test, k_test, T, H, W)
    q_s2, k_s2 = rope_spiral_k2(q_test, k_test, T, H, W)

    print(f"  Axial   d_t={rope_axial.d_t}, d_h={rope_axial.d_h}, d_w={rope_axial.d_w}")
    print(f"  Spiral  d_t={rope_spiral_k2.d_t}, d_spatial={rope_spiral_k2.d_spatial}")
    print(f"  Axial  output shapes: q={q_a.shape}, k={k_a.shape}")
    print(f"  Spiral output shapes: q={q_s2.shape}, k={k_s2.shape}")
    print(f"  Both run successfully.\n")

    # --- Test 3: dimension budget table ----------------------------------------
    print("Dimension budget for common configurations (head_dim=64):")
    print(f"  {'K':>3}  {'d_t':>4}  {'d_spatial':>10}  {'freqs/dir':>10}  directions")
    for K in [2, 4, 8]:
        r = RoPE3DSpiral(64, 16, 16, 16, n_spiral_directions=K)
        dirs = [f"{k*180/K:.0f}°" for k in range(K)]
        print(
            f"  {K:>3}  {r.d_t:>4}  {r.d_spatial:>10}  "
            f"{r.d_spatial // (2*K):>10}  {', '.join(dirs)}"
        )

    print("\nAll tests passed!")
