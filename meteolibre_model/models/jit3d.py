import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================================================================
# == 1. Modern Components (RMSNorm, SwiGLU, RoPE)
# ==============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)
        self.w3 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

# ==============================================================================
# == 2. 3D Rotary Positional Embeddings (Axial RoPE)
# ==============================================================================

def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, 1, *freqs_cis.shape)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class RoPE3D(nn.Module):
    def __init__(self, head_dim, max_t, max_h, max_w, base=10000.0):
        super().__init__()
        chunk = (head_dim // 3)
        self.d_t = (chunk // 2) * 2
        self.d_h = (chunk // 2) * 2
        self.d_w = head_dim - (self.d_t + self.d_h)
        assert self.d_w % 2 == 0
        self.register_buffer("freqs_t", precompute_freqs_cis(self.d_t, max_t, base), persistent=False)
        self.register_buffer("freqs_h", precompute_freqs_cis(self.d_h, max_h, base), persistent=False)
        self.register_buffer("freqs_w", precompute_freqs_cis(self.d_w, max_w, base), persistent=False)

    def forward(self, xq, xk, T, H, W):
        q_t, q_h, q_w = torch.split(xq, [self.d_t, self.d_h, self.d_w], dim=-1)
        k_t, k_h, k_w = torch.split(xk, [self.d_t, self.d_h, self.d_w], dim=-1)
        f_t = self.freqs_t[:T].view(T, 1, 1, -1).expand(T, H, W, -1).flatten(0, 2)
        f_h = self.freqs_h[:H].view(1, H, 1, -1).expand(T, H, W, -1).flatten(0, 2)
        f_w = self.freqs_w[:W].view(1, 1, W, -1).expand(T, H, W, -1).flatten(0, 2)
        q_t, k_t = apply_rotary_emb(q_t, k_t, f_t)
        q_h, k_h = apply_rotary_emb(q_h, k_h, f_h)
        q_w, k_w = apply_rotary_emb(q_w, k_w, f_w)
        return torch.cat([q_t, q_h, q_w], dim=-1), torch.cat([k_t, k_h, k_w], dim=-1)

# ==============================================================================
# == 3. Latent Context Corruption
# ==============================================================================

class LatentContextCorruptor(nn.Module):
    """
    Injects noise on context tokens only, at two points:
      - Stage 'embed': right after patch_embed, before any block
      - Stage 'block0': after block 0 output, before block 1

    Normalization strategy: normalize the context slice to zero-mean / unit-std
    per sample BEFORE adding noise, then re-scale back. This prevents the model
    from learning to output large latent magnitudes to drown out the noise.

    Args:
        corruption_prob (float): probability of applying corruption to a sample.
        embed_noise_scale (float): noise std at embed stage (relative to unit-norm latent).
        block0_noise_scale (float): noise std at block0 stage (relative to unit-norm latent).
    """
    def __init__(
        self,
        corruption_prob: float = 0.3,
        embed_noise_scale: float = 0.10,
        block0_noise_scale: float = 0.05,
    ):
        super().__init__()
        self.corruption_prob = corruption_prob
        self.embed_noise_scale = embed_noise_scale
        self.block0_noise_scale = block0_noise_scale

    @torch.compiler.disable
    def _corrupt(self, tokens: torch.Tensor, n_ctx: int, noise_scale: float) -> torch.Tensor:
        """
        tokens  : (B, N_total, D)  — full sequence (ctx + target)
        n_ctx   : number of context tokens (first n_ctx positions)
        returns : tokens with noise added on context slice for selected samples
        """
        B = tokens.shape[0]

        # Per-sample binary mask: which samples get corrupted
        mask = torch.rand(B, device=tokens.device) < self.corruption_prob
        if not mask.any():
            return tokens

        ctx = tokens[mask, :n_ctx, :]          # (B', n_ctx, D)

        # --- Normalize context slice ---
        # Per-sample mean and std over (n_ctx, D) so the noise scale is meaningful
        # regardless of how large the latent values are at this stage
        mean = ctx.mean(dim=(1, 2), keepdim=True)          # (B', 1, 1)
        std  = ctx.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)  # (B', 1, 1)
        ctx_norm = (ctx - mean) / std

        # --- Add noise in normalized space ---
        noise = torch.randn_like(ctx_norm) * noise_scale
        ctx_corrupted_norm = ctx_norm + noise

        # --- Re-scale back to original distribution ---
        ctx_corrupted = ctx_corrupted_norm * std + mean

        # Write back only the context slice of masked samples
        out = tokens.clone()
        out[mask, :n_ctx, :] = ctx_corrupted
        return out

    def corrupt_embed(self, tokens: torch.Tensor, n_ctx: int) -> torch.Tensor:
        """Call after patch_embed, before block 0."""
        return self._corrupt(tokens, n_ctx, self.embed_noise_scale)

    def corrupt_block0(self, tokens: torch.Tensor, n_ctx: int) -> torch.Tensor:
        """Call after block 0 output, before block 1."""
        return self._corrupt(tokens, n_ctx, self.block0_noise_scale)

# ==============================================================================
# == 4. Custom Transformer Block
# ==============================================================================

class JiTAttention(nn.Module):
    def __init__(self, dim, num_heads, qk_norm=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, rope_module, T, H, W):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = rope_module(q, k, T, H, W)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class JiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = JiTAttention(dim, num_heads, qk_norm=True)
        self.norm2 = RMSNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = SwiGLU(dim, hidden_dim, dim)

    def forward(self, x, rope_module, T, H, W):
        x = x + self.attn(self.norm1(x), rope_module, T, H, W)
        x = x + self.mlp(self.norm2(x))
        return x

# ==============================================================================
# == 5. The Full JiT-3D Model
# ==============================================================================

class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(2, 16, 16), in_channels=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)

class FinalLayer(nn.Module):
    def __init__(self, patch_size, out_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.patch_dim = out_channels * patch_size[0] * patch_size[1] * patch_size[2]
        self.linear = nn.Linear(embed_dim, self.patch_dim)

    def forward(self, x, T, H, W):
        pt, ph, pw = self.patch_size
        Tp, Hp, Wp = T // pt, H // ph, W // pw
        x = self.linear(x)
        x = x.view(x.shape[0], Tp, Hp, Wp, self.out_channels, pt, ph, pw)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return x.view(x.shape[0], self.out_channels, T, H, W)

class JiT3D_Modern(nn.Module):
    def __init__(
        self,
        img_size=(6, 128, 128),
        patch_size=(1, 8, 8),       # temporal patch = 1 → each token = 1 frame
        in_channels=3,
        out_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        context_dim=128,
        time_emb_dim=64,
        n_context_frames=4,         # how many frames are "context" at the start of x
        # --- Corruption hyperparams ---
        corruption_prob: float = 0.5,
        embed_noise_scale: float = 0.10,
        block0_noise_scale: float = 0.05,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.n_context_frames = n_context_frames

        # Spatial tokens per frame (with temporal patch_size=1)
        self.tokens_per_frame = (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        # Total context tokens = n_context_frames × tokens_per_frame
        self.n_ctx_tokens = n_context_frames * self.tokens_per_frame

        # Patch Embed
        self.patch_embed = PatchEmbed3D(patch_size, in_channels, embed_dim)

        # RoPE
        self.grid_t = img_size[0] // patch_size[0]
        self.grid_h = img_size[1] // patch_size[1]
        self.grid_w = img_size[2] // patch_size[2]
        self.rope = RoPE3D(embed_dim // num_heads, self.grid_t * 2, self.grid_h * 2, self.grid_w * 2)

        # Context/Time conditioning
        input_context_dim = context_dim - 1 + time_emb_dim
        self.time_freq_emb = nn.Sequential(nn.Linear(1, time_emb_dim), nn.SiLU())
        self.context_mlp = nn.Sequential(
            nn.Linear(input_context_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            JiTBlock(embed_dim, num_heads, mlp_ratio=2.6)
            for _ in range(depth)
        ])

        self.norm_final = RMSNorm(embed_dim)
        self.final_layer = FinalLayer(patch_size, out_channels, embed_dim)

        # ── Latent context corruptor (training only) ──────────────────────────
        self.corruptor = LatentContextCorruptor(
            corruption_prob=corruption_prob,
            embed_noise_scale=embed_noise_scale,
            block0_noise_scale=block0_noise_scale,
        )

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            nn.init.trunc_normal_(m.weight, std=0.02)

    def get_sinusoidal_time(self, t):
        device = t.device
        half_dim = 64 // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

    def forward(self, x, t):
        """
        x : (B, C, T, H, W)   — context frames first, target frames after
        t : (B, context_dim)
        """
        B, C, T, H, W = x.shape

        # 1. Context conditioning
        time_val = t[:, -1]
        t_emb = self.get_sinusoidal_time(time_val)
        combined = torch.cat([t[:, :-1], t_emb], dim=1)
        c_emb = self.context_mlp(combined).unsqueeze(1)  # (B, 1, D)

        # 2. Patchify
        x = self.patch_embed(x)  # (B, N_total, D)
        x = x + c_emb

        # ── Corruption stage 1: embed ─────────────────────────────────────────
        # Only active during training; n_ctx_tokens isolates context frames
        if self.training:
            x = self.corruptor.corrupt_embed(x, self.n_ctx_tokens)

        # 3. Transformer loop
        grid_t = T // self.patch_size[0]
        grid_h = H // self.patch_size[1]
        grid_w = W // self.patch_size[2]

        for i, block in enumerate(self.blocks):
            x = block(x, self.rope, grid_t, grid_h, grid_w)

            # ── Corruption stage 2: after block 0 ────────────────────────────
            if self.training and i == 0:
                x = self.corruptor.corrupt_block0(x, self.n_ctx_tokens)

        x = self.norm_final(x)
        return self.final_layer(x, T, H, W)


# ==============================================================================
# == Test
# ==============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing Modern JiT-3D on {device}")

    T_ctx, T_tgt = 4, 3
    T = T_ctx + T_tgt
    H, W = 128, 128

    model = JiT3D_Modern(
        img_size=(T, H, W),
        patch_size=(1, 8, 8),
        embed_dim=768,
        depth=12,
        num_heads=12,
        n_context_frames=T_ctx,
        corruption_prob=0.3,
        embed_noise_scale=0.10,
        block0_noise_scale=0.05,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")
    print(f"Model Size (approx): {total_params * 4 / 1024**2:.2f} MB (FP32)")

    # --- Training mode: corruption active ---
    model.train()
    x = torch.randn(4, 3, T, H, W).to(device)
    t = torch.randn(4, 128).to(device)
    out = model(x, t)
    print(f"[train] Output shape: {out.shape}")
    out.sum().backward()
    print("[train] Backward pass successful.")

    # --- Eval mode: corruption disabled ---
    model.eval()
    with torch.no_grad():
        out_eval = model(x, t)
    print(f"[eval]  Output shape: {out_eval.shape}")
    print("Done.")