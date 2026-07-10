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

# (LatentContextCorruptor removed — superseded by the hyperspherical KV-noise
#  augmentation in JiTAttention; not used.)

# ==============================================================================
# == 4. Custom Transformer Block
# ==============================================================================

class JiTAttention(nn.Module):
    def __init__(self, dim, num_heads, qk_norm=True,
                 kv_ctx_noise=0.0, block_causal=False, prefix_attn=False,
                 tokens_per_frame=1, n_ctx_tokens=0, seq_len=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        # --- AR-rollout-stability augmentation (OFF by default; backward compatible) ---
        # kv_ctx_noise: training-only HYPERSPHERICAL noise on the K/V of the CONTEXT
        #   token slice only (the forecast token is left untouched). Applied AFTER
        #   qk_norm (so k lives on a hypersphere) and BEFORE RoPE. Isotropic noise
        #   scaled relative to each token's radius, then RENORMALIZED to that exact
        #   radius -> the vector stays on its hypersphere (direction-only jitter,
        #   norm preserved) so attention keeps its well-conditioned geometry.
        # block_causal: replace naive full attention with a block-causal mask
        #   (bidirectional WITHIN a frame, causal ACROSS frames). Implemented as an
        #   additive (N,N) SDPA mask (same pattern as prefix_attn below) so it runs
        #   on SDPA's fused Flash/mem-efficient backend. This is functionally
        #   identical to a flex_attention block_mask, but avoids the Triton
        #   flex-kernel dependency that can silently degrade to the (very slow)
        #   eager path under torch.compile on some builds.
        self.kv_ctx_noise = float(kv_ctx_noise)
        self.block_causal = bool(block_causal)
        self.tokens_per_frame = max(1, int(tokens_per_frame))
        self.n_ctx_tokens = int(n_ctx_tokens)
        self.prefix_attn = bool(prefix_attn)
        # Precompute the (rarely-changing) additive SDPA masks up-front when the
        # sequence length is known, storing them as non-persistent buffers. This
        # keeps them as STABLE tensors during torch.compile: the old lazy
        # `if self._mask is None: build()` pattern flipped a dynamo guard on the
        # 2nd forward and forced a (slow) recompile. When seq_len is unknown at
        # construction (direct JiTAttention usage) they stay None and the getters
        # below build them lazily instead.
        bm = self._build_block_mask(seq_len) if (
            seq_len and self.block_causal and seq_len > self.tokens_per_frame) else None
        self.register_buffer("_block_mask", bm, persistent=False)
        pm = self._build_prefix_mask(seq_len) if (
            seq_len and self.prefix_attn and 0 < self.n_ctx_tokens < seq_len) else None
        self.register_buffer("_prefix_mask", pm, persistent=False)

    def _noise_ctx_kv(self, t, scale):
        # t: (B, H, N, D). Add HYPERSPHERICAL (norm-preserving) noise to the leading
        # context-token slice only (training; forecast slice untouched). `scale` is a
        # per-BATCH-ELEMENT angular scale (radians), shape (B,), shared across all
        # layers/heads/tokens for that element -> simulates a MIX of clean (scale~0)
        # and degraded (scale~max) context within each batch. k lives on a hypersphere
        # after qk_norm: isotropic noise (scaled by the per-element angle / sqrt(d))
        # is added then renormalized to each token's exact radius -> direction-only
        # jitter, ||.|| preserved. scale=None / eval -> no-op.
        if scale is None or not self.training or self.n_ctx_tokens <= 0:
            return t
        nc = self.n_ctx_tokens
        ctx = t[..., :nc, :]
        r = ctx.norm(dim=-1, keepdim=True).clamp(min=1e-6)           # preserve each token's radius
        g = torch.randn_like(ctx)
        sb = scale.clamp(min=0.0).view(-1, 1, 1, 1)                  # (B,1,1,1) per-element angle (rad)
        y = ctx + (sb / math.sqrt(self.head_dim)) * r * g            # isotropic jitter, magnitude ~ sb rad
        y = y * (r / y.norm(dim=-1, keepdim=True).clamp(min=1e-6))   # renormalize -> back on the hypersphere
        return torch.cat([y, t[..., nc:, :]], dim=2)

    @staticmethod
    def _make_additive_mask(allow, device):
        """Boolean (N,N) allow-matrix -> fp32 additive SDPA mask (0 / -inf)."""
        neg = torch.finfo(torch.float32).min
        return torch.where(allow, 0.0, neg).to(device=device, dtype=torch.float32)

    def _build_block_mask(self, N, device="cpu"):
        # Frame-block-causal ADDITIVE mask (N, N) for SDPA: a query attends to
        # every key in its OWN temporal block (frame) and all EARLIER blocks.
        # Bidirectional within a frame, strictly causal across frames. Honours
        # video temporal order instead of naive per-token causality (which would
        # split a frame). Built in fp32; cast to q.dtype at the call site.
        tpf = self.tokens_per_frame
        qi = torch.arange(N).view(N, 1)            # (N, 1) query frame index
        ki = torch.arange(N).view(1, N)            # (1, N) key   frame index
        allow = (ki // tpf) <= (qi // tpf)         # (N, N) boolean block-causal
        return self._make_additive_mask(allow, device)

    def _build_prefix_mask(self, N, device="cpu"):
        # PREFIX causal mask (additive, (N,N)): context tokens (first n_ctx) attend
        # BIDIRECTIONALLY to all context (an encoder over the known past); the
        # future/forecast token attends to EVERYTHING; context CANNOT attend to
        # the (noised) future -> no target leakage into the conditioning.
        n = self.n_ctx_tokens
        qi = torch.arange(N).view(N, 1)
        ki = torch.arange(N).view(1, N)
        allow = (ki < n) | (qi >= n)               # key-in-context OR query-in-future
        return self._make_additive_mask(allow, device)

    def _get_block_mask(self, N, device):
        # Returns the precomputed buffer (compile-safe: a stable tensor). Rebuilds
        # only if seq_len was unknown at construction or the runtime N changed.
        if self._block_mask is None or self._block_mask.shape[0] != N:
            self._block_mask = self._build_block_mask(N, device)
        return self._block_mask

    def _get_prefix_mask(self, N, device):
        if self._prefix_mask is None or self._prefix_mask.shape[0] != N:
            self._prefix_mask = self._build_prefix_mask(N, device)
        return self._prefix_mask

    def forward(self, x, rope_module, T, H, W, ctx_noise_scale=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # context-only K/V noise ON THE HYPERSPHERE (training; future token untouched):
        # applied AFTER qk_norm (k lives on a hypersphere) and BEFORE RoPE so RoPE
        # then rotates the already-jittered directions exactly as usual. The per-batch
        # element scale (shared across layers) simulates a mix of clean/noisy context.
        k = self._noise_ctx_kv(k, ctx_noise_scale)
        v = self._noise_ctx_kv(v, ctx_noise_scale)
        q, k = rope_module(q, k, T, H, W)
        if self.block_causal and N > self.tokens_per_frame:
            m = self._get_block_mask(N, q.device).to(q.dtype)
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        elif self.prefix_attn and 0 < self.n_ctx_tokens < N:
            m = self._get_prefix_mask(N, q.device).to(q.dtype)
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        else:
            x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class JiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0,
                 kv_ctx_noise=0.0, block_causal=False, prefix_attn=False,
                 tokens_per_frame=1, n_ctx_tokens=0, seq_len=None):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = JiTAttention(dim, num_heads, qk_norm=True,
                                 kv_ctx_noise=kv_ctx_noise,
                                 block_causal=block_causal,
                                 prefix_attn=prefix_attn,
                                 tokens_per_frame=tokens_per_frame,
                                 n_ctx_tokens=n_ctx_tokens,
                                 seq_len=seq_len)
        self.norm2 = RMSNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = SwiGLU(dim, hidden_dim, dim)

    def forward(self, x, rope_module, T, H, W, ctx_noise_scale=None):
        x = x + self.attn(self.norm1(x), rope_module, T, H, W, ctx_noise_scale)
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
        patch_size=(1, 8, 8),
        in_channels=3,
        out_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        context_dim=128,
        time_emb_dim=64,
        n_context_frames=4,
        kv_ctx_noise: float = 0.3,
        block_causal: bool = True,
        prefix_attn: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.n_context_frames = n_context_frames
        self.kv_ctx_noise = kv_ctx_noise

        self.tokens_per_frame = (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        self.n_ctx_tokens = n_context_frames * self.tokens_per_frame

        self.patch_embed = PatchEmbed3D(patch_size, in_channels, embed_dim)

        self.grid_t = img_size[0] // patch_size[0]
        self.grid_h = img_size[1] // patch_size[1]
        self.grid_w = img_size[2] // patch_size[2]
        self.rope = RoPE3D(embed_dim // num_heads, self.grid_t * 2, self.grid_h * 2, self.grid_w * 2)

        input_context_dim = context_dim - 1 + time_emb_dim
        self.time_freq_emb = nn.Sequential(nn.Linear(1, time_emb_dim), nn.SiLU())
        self.context_mlp = nn.Sequential(
            nn.Linear(input_context_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        seq_len = self.grid_t * self.tokens_per_frame   # total tokens -> compile-stable masks
        self.blocks = nn.ModuleList([
            JiTBlock(embed_dim, num_heads, mlp_ratio=2.6,
                    kv_ctx_noise=kv_ctx_noise, block_causal=block_causal,
                    prefix_attn=prefix_attn,
                    tokens_per_frame=self.tokens_per_frame,
                    n_ctx_tokens=self.n_ctx_tokens,
                    seq_len=seq_len)
            for _ in range(depth)
        ])

        self.norm_final = RMSNorm(embed_dim)
        self.final_layer = FinalLayer(patch_size, out_channels, embed_dim)

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
        x : (B, C, T, H, W)
        t : (B, context_dim)
        """
        B, C, T, H, W = x.shape

        time_val = t[:, -1]
        t_emb = self.get_sinusoidal_time(time_val)
        combined = torch.cat([t[:, :-1], t_emb], dim=1)
        c_emb = self.context_mlp(combined).unsqueeze(1)

        x = self.patch_embed(x)
        x = x + c_emb

        grid_t = T // self.patch_size[0]
        grid_h = H // self.patch_size[1]
        grid_w = W // self.patch_size[2]

        # Per-batch-element context-noise scale (radians), sampled ONCE per forward
        # and SHARED across all layers/heads/tokens of each element -> a mix of clean
        # (scale~0) and degraded (scale~max) context per batch. kv_ctx_noise is the
        # max; each element draws U(0, max). None in eval / when disabled.
        ctx_scale = None
        if self.training and self.kv_ctx_noise > 0:
            ctx_scale = torch.rand(B, device=x.device) * self.kv_ctx_noise

        for i, block in enumerate(self.blocks):
            x = block(x, self.rope, grid_t, grid_h, grid_w, ctx_scale)

        x = self.norm_final(x)
        return self.final_layer(x, T, H, W)