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

class AdaLNModulation(nn.Module):
    """Low-rank adaLN modulation head (DiT-style conditioning).

    Consumes the SHARED conditioning vector ``c`` (B, dim) produced by
    ``JiT3D_Modern.context_mlp`` (diffusion timestep + global scalars) and
    regresses ``n_params * dim`` modulation values: per-block heads use
    n_params=6 -> (shift, scale, gate) for the attention sub-layer and
    (shift, scale, gate) for the MLP sub-layer; the final-layer head uses
    n_params=2 -> (shift, scale) applied to ``norm_final``'s output.

    Low-rank for parameter efficiency: SiLU -> Linear(dim, rank, no bias) ->
    Linear(rank, n_params*dim). At dim=768, rank=128, a 6-param head costs
    ~0.7M/block (+8.3M over 12 blocks) vs ~3.5M/block (+42M) for the
    full-rank DiT head.

    Init contract: ``JiT3D_Modern._init_adaln_heads`` runs AFTER the global
    ``initialize_weights()`` (whose trunc_normal would otherwise overwrite
    this head) and zeros ``up`` so the head emits only its bias at init:
    shift=0, scale=0, gate=``gate_init``. With gate_init=1.0 the modulated
    block is EXACTLY the un-modulated post-LN block (bit-for-bit resume no-op
    when fine-tuning from a pre-adaLN checkpoint via strict=False); with
    gate_init=0.0 every block is the identity at init (canonical adaLN-Zero,
    for training from scratch).
    """

    def __init__(self, dim, n_params, rank=128, gate_init=0.0, gate_idx=()):
        super().__init__()
        self.dim = dim
        self.n_params = int(n_params)
        self.gate_init = float(gate_init)
        self.gate_idx = tuple(int(i) for i in gate_idx)
        self.silu = nn.SiLU()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, self.n_params * dim)

    def forward(self, c):
        # c: (B, dim) -> (B, n_params, dim)
        return self.up(self.silu(self.down(c))).view(-1, self.n_params, self.dim)

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

    Normalization strategy: **per-token L2 (power) normalization**. For each
    context token we divide by its L2 norm along the feature dimension *before*
    adding noise, then re-scale back. This differs from the previous global
    mean/std normalization in two important ways:

      1. The injected noise energy is identical for every token regardless of
         that token's raw magnitude — so a low-energy token (a quiet region of
         the field) gets the same effective corruption as a high-energy token
         (an active convective cell). The model can no longer "hide" the
         transmitted information in a few very-large-magnitude dimensions / tokens
         and is forced to **spread information uniformly across the whole token
         feature space** to be robust to corruption.
      2. Because the norm is computed per-token (not collapsed over all
         (n_ctx, D)), the per-dimension noise budget is shared across the entire
         feature axis — directly targeting the "some dims are very low scale,
         others very high scale" failure mode.

    Args:
        token_dim (int): feature dimension D of each token (embed_dim). Used to
            convert ``noise_scale`` from a *fraction of the token's L2 norm* to
            the actual per-element std in unit-L2 space (see below).
        corruption_prob (float): probability of applying corruption to a sample.
        embed_noise_scale (float): noise energy at embed stage, expressed as a
            fraction of each token's L2 norm (0.10 = noise vector norm is 10% of
            the token norm). Internally scaled by ``1/sqrt(token_dim)`` to get
            the per-element std.
        block0_noise_scale (float): same, for the block0 stage.
    """
    def __init__(
        self,
        token_dim: int,
        corruption_prob: float = 0.3,
        embed_noise_scale: float = 0.10,
        block0_noise_scale: float = 0.05,
    ):
        super().__init__()
        self.corruption_prob = corruption_prob
        # After per-token L2 normalization each token is a unit vector, so each
        # of its D elements has magnitude ~1/√D. A noise vector of per-element
        # std ``s`` has total norm ~s·√D. To corrupt a fraction ``noise_scale``
        # of the unit-norm token we therefore need s = noise_scale / √D.
        dim_scale = 1.0 / math.sqrt(token_dim)
        self.embed_noise_scale = embed_noise_scale * dim_scale
        self.block0_noise_scale = block0_noise_scale * dim_scale

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

        # --- Per-token power (L2) normalization ---
        # Normalize each token independently to unit L2 norm along the feature
        # axis. This makes the injected noise scale uniform across tokens AND
        # across dimensions (no single dim/token can dominate the energy),
        # forcing the model to spread information over the whole token space.
        #
        # `norm` is kept so we can re-scale back to the original magnitude after
        # injecting noise (the model still sees the right latent distribution;
        # only the *corruption* happens in normalized space).
        norm = ctx.norm(dim=2, keepdim=True, p=2).clamp(min=1e-6)   # (B', n_ctx, 1)
        ctx_norm = ctx / norm                                        # unit L2 per token

        # --- Add noise in normalized space ---
        noise = torch.randn_like(ctx_norm) * noise_scale
        ctx_corrupted_norm = ctx_norm + noise

        # --- Re-scale back to the original per-token magnitudes ---
        ctx_corrupted = ctx_corrupted_norm * norm

        # Write back only the context slice of masked samples
        out = tokens.clone()
        out[mask, :n_ctx, :] = ctx_corrupted.to(tokens.dtype)
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
    def __init__(self, dim, num_heads, mlp_ratio=4.0,
                 use_adaln=False, adaln_rank=128, adaln_gate_init=1.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = JiTAttention(dim, num_heads, qk_norm=True)
        self.norm2 = RMSNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = SwiGLU(dim, hidden_dim, dim)
        # adaLN modulation head: regresses (shift, scale, gate) x2 from the
        # shared conditioning vector. None -> classic post-LN block (the
        # pre-adaLN code path, numerically unchanged). Gate slots are params
        # #2 and #5 of the 6-vector (DiT convention: shift_msa, scale_msa,
        # gate_msa, shift_mlp, scale_mlp, gate_mlp).
        self.adaln = (
            AdaLNModulation(dim, n_params=6, rank=adaln_rank,
                            gate_init=adaln_gate_init, gate_idx=(2, 5))
            if use_adaln else None
        )

    def forward(self, x, rope_module, T, H, W, c=None):
        if self.adaln is not None:
            # c: (B, D) shared conditioning vector (timestep + global scalars).
            # Each modulation param is (B, 1, D) -> broadcast over the N tokens:
            # ALL tokens of a sample share the modulation (global conditioning,
            # as in DiT). At the identity init (scale=shift=0, gate=1) this is
            # bit-for-bit the else-branch below.
            shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = \
                self.adaln(c).unsqueeze(1).unbind(dim=2)
            h = self.norm1(x) * (1.0 + scale_a) + shift_a
            x = x + gate_a * self.attn(h, rope_module, T, H, W)
            h = self.norm2(x) * (1.0 + scale_m) + shift_m
            x = x + gate_m * self.mlp(h)
        else:
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
        corruption_prob: float = 0.0,
        embed_noise_scale: float = 0.10,
        block0_noise_scale: float = 0.05,
        # --- adaLN(-Zero) conditioning (DiT-style per-block modulation) ---
        # use_adaln: give EVERY block a low-rank head regressing (shift, scale,
        #   gate) x2 from the shared conditioning vector (diffusion t + global
        #   scalars), plus a (shift, scale) head on norm_final. This replaces
        #   the single additive input bias as the primary conditioning path:
        #   instead of being injected once and diluted over 12 blocks, the
        #   timestep re-enters at every block and gates control residual
        #   strength (DiT ablations: adaLN-Zero > cross-attn > in-context >
        #   additive).
        # cond_additive: keep the legacy `x = patch_embed(x) + c_emb` input
        #   bias. FINE-TUNE recipe (defaults): cond_additive=True with
        #   adaln_gate_init=1.0 -> resuming a pre-adaLN checkpoint with
        #   strict=False reproduces its outputs BIT-FOR-BIT (the new heads are
        #   exact identities) and adaLN capacity grows on top.
        #   SCRATCH recipe: cond_additive=False with adaln_gate_init=0.0 ->
        #   canonical adaLN-Zero (blocks start as identity; conditioning flows
        #   only through the modulation heads).
        # adaln_rank: low-rank bottleneck of the heads (+~8.6M params at 768/128).
        # adaln_gate_init: initial residual-gate value (see recipes above).
        use_adaln: bool = True,
        cond_additive: bool = True,
        adaln_rank: int = 128,
        adaln_gate_init: float = 1.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.n_context_frames = n_context_frames
        self.use_adaln = bool(use_adaln)
        self.cond_additive = bool(cond_additive)
        if not (self.use_adaln or self.cond_additive):
            raise ValueError(
                "JiT3D_Modern needs at least one conditioning path: set "
                "use_adaln=True and/or cond_additive=True (otherwise the "
                "diffusion timestep is invisible to the model)."
            )

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
            JiTBlock(embed_dim, num_heads, mlp_ratio=2.6,
                     use_adaln=use_adaln, adaln_rank=adaln_rank,
                     adaln_gate_init=adaln_gate_init)
            for _ in range(depth)
        ])

        self.norm_final = RMSNorm(embed_dim)
        # Final-layer adaLN modulation (shift, scale only -- no gate): applied
        # to norm_final's output before the decoder head, DiT-style. Fully
        # zero-initialized (see _init_adaln_heads) so it starts as an exact
        # identity regardless of adaln_gate_init.
        self.adaLN_final = (
            AdaLNModulation(embed_dim, n_params=2, rank=adaln_rank)
            if self.use_adaln else None
        )
        self.final_layer = FinalLayer(patch_size, out_channels, embed_dim)

        # ── Latent context corruptor (training only) ──────────────────────────
        self.corruptor = LatentContextCorruptor(
            token_dim=embed_dim,
            corruption_prob=corruption_prob,
            embed_noise_scale=embed_noise_scale,
            block0_noise_scale=block0_noise_scale,
        )

        self.initialize_weights()
        # Re-initialize the adaLN modulation heads AFTER the global init (which
        # trunc_normal-overwrites every Linear): zero up-projections restore
        # the identity-at-init contract (see AdaLNModulation docstring).
        self._init_adaln_heads()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            nn.init.trunc_normal_(m.weight, std=0.02)

    def _init_adaln_heads(self):
        """Zero-init for ALL adaLN modulation heads (no-op when use_adaln=False).

        ``initialize_weights`` applies trunc_normal(0.02) to every Linear,
        including the new heads; this restores the identity-at-init property:
        ``up.weight = 0``, ``up.bias = 0`` except the gate slots, which are
        set to the head's ``gate_init``. Consequences at init:
          * gate_init=1.0 (fine-tune): scale=shift=0, gate=1 -> each block is
            EXACTLY the pre-adaLN post-LN block; a strict=False resume of an
            old checkpoint reproduces its outputs bit-for-bit.
          * gate_init=0.0 (scratch): gate=0 -> blocks are the identity
            (canonical adaLN-Zero); gradients still flow into up.weight (and
            into down/context_mlp once up.weight moves off zero).
        The final-layer head (n_params=2, no gates) is fully zeroed: identity.
        """
        if not self.use_adaln:
            return
        heads = [b.adaln for b in self.blocks]
        if self.adaLN_final is not None:
            heads.append(self.adaLN_final)
        for head in heads:
            nn.init.zeros_(head.up.weight)
            nn.init.zeros_(head.up.bias)
            if head.gate_init != 0.0:
                with torch.no_grad():
                    for gi in head.gate_idx:
                        head.up.bias[gi * head.dim:(gi + 1) * head.dim] = head.gate_init

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

        # 1. Context conditioning: ONE shared conditioning vector (B, D) from
        # the diffusion timestep + global scalars. Feeds the legacy additive
        # input bias (cond_additive) and/or the per-block adaLN modulation
        # heads (use_adaln) -- same embedding, two paths.
        time_val = t[:, -1]
        t_emb = self.get_sinusoidal_time(time_val)
        combined = torch.cat([t[:, :-1], t_emb], dim=1)
        c_vec = self.context_mlp(combined)  # (B, D)

        # 2. Patchify
        x = self.patch_embed(x)  # (B, N_total, D)
        if self.cond_additive:
            x = x + c_vec.unsqueeze(1)

        # ── Corruption stage 1: embed ─────────────────────────────────────────
        # Only active during training; n_ctx_tokens isolates context frames
        if self.training:
            x = self.corruptor.corrupt_embed(x, self.n_ctx_tokens)

        # 3. Transformer loop
        grid_t = T // self.patch_size[0]
        grid_h = H // self.patch_size[1]
        grid_w = W // self.patch_size[2]

        for i, block in enumerate(self.blocks):
            # c_vec is consumed only by blocks with an adaLN head (use_adaln);
            # ignored otherwise (classic post-LN path, numerics unchanged).
            x = block(x, self.rope, grid_t, grid_h, grid_w, c_vec)

            # ── Corruption stage 2: after block 0 ────────────────────────────
            if self.training and i == 0:
                x = self.corruptor.corrupt_block0(x, self.n_ctx_tokens)

        x = self.norm_final(x)
        if self.adaLN_final is not None:
            # Final-layer modulation (shift, scale), zero-init -> identity at
            # resume; the decoder head sees the modulated tokens.
            shift_f, scale_f = self.adaLN_final(c_vec).unsqueeze(1).unbind(dim=2)
            x = x * (1.0 + scale_f) + shift_f
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

    # --- adaLN conditioning: resume no-op equivalence (fine-tune recipe) ---
    model_adaln = JiT3D_Modern(
        img_size=(T, H, W),
        patch_size=(1, 8, 8),
        embed_dim=768,
        depth=12,
        num_heads=12,
        n_context_frames=T_ctx,
        corruption_prob=0.3,
        embed_noise_scale=0.10,
        block0_noise_scale=0.05,
        use_adaln=True,        # per-block shift/scale/gate modulation
        cond_additive=True,    # keep legacy additive c_emb (fine-tune recipe)
        adaln_gate_init=1.0,   # identity at init -> exact resume no-op
    ).to(device)
    n_adaln_params = sum(p.numel() for p in model_adaln.parameters())
    print(f"adaLN overhead: +{n_adaln_params - total_params:,} params")
    model_adaln.load_state_dict(model.state_dict(), strict=False)  # trunk copy
    model_adaln.eval()
    with torch.no_grad():
        out_adaln = model_adaln(x, t)
    diff = (out_adaln - out_eval).abs().max().item()
    print(f"[adaln] resume no-op max|diff| vs base = {diff:.3e} (expect 0)")
    print("Done.")
