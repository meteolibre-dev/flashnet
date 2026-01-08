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
        # x: (B, N, D)
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # SwiGLU typically uses 2/3rds hidden dim to keep param count similar to GeLU
        # But here we stick to the user's config or standard 4x expansion if desired
        self.w1 = nn.Linear(in_features, hidden_features, bias=False) # Gate
        self.w2 = nn.Linear(in_features, hidden_features, bias=False) # Value
        self.w3 = nn.Linear(hidden_features, out_features, bias=False) # Output

    def forward(self, x):
        # Swish(Gate) * Value
        x1 = F.silu(self.w1(x))
        x2 = self.w2(x)
        return self.w3(x1 * x2)

# ==============================================================================
# == 2. 3D Rotary Positional Embeddings (Axial RoPE)
# ==============================================================================

def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    # xq, xk: (B, H, Seq_Len, Head_Dim)
    # freqs_cis: (Seq_Len, Head_Dim/2) -> reshaped to broadcast
    
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Broadcast freqs to match batch and heads
    freqs_cis = freqs_cis.view(1, 1, *freqs_cis.shape) 
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class RoPE3D(nn.Module):
    """
    Axial RoPE for 3D data (Time, Height, Width).
    Splits head_dim into 3 chunks and applies 1D RoPE to each axis independently.
    """
    def __init__(self, head_dim, max_t, max_h, max_w, base=10000.0):
        super().__init__()
        # Split head dimension roughly into 3
        self.d_t = head_dim // 3
        self.d_h = head_dim // 3
        self.d_w = head_dim - (self.d_t + self.d_h) # Remainder goes to width
        
        self.register_buffer("freqs_t", precompute_freqs_cis(self.d_t, max_t, base), persistent=False)
        self.register_buffer("freqs_h", precompute_freqs_cis(self.d_h, max_h, base), persistent=False)
        self.register_buffer("freqs_w", precompute_freqs_cis(self.d_w, max_w, base), persistent=False)

    def forward(self, xq, xk, T, H, W):
        # xq, xk shape: (B, Num_Heads, Seq_Len, Head_Dim)
        # Seq_Len must equal T*H*W
        
        # Split heads into axial components
        q_t, q_h, q_w = torch.split(xq, [self.d_t, self.d_h, self.d_w], dim=-1)
        k_t, k_h, k_w = torch.split(xk, [self.d_t, self.d_h, self.d_w], dim=-1)

        # Reshape to 3D grid to align with precomputed freqs
        # (B, nH, T*H*W, d) -> (B, nH, T, H, W, d)
        B, nH, _, _ = q_t.shape
        
        # Apply Time RoPE
        # We take the column corresponding to T, broadcast over H, W
        # But standard apply_rotary expects linear sequence. 
        # Strategy: Expand freqs to full grid (T, H, W) then flatten.
        
        # Construct full grid freqs
        # freqs_t: (T, d/2). We need (T*H*W, d/2) where H and W are repeated
        f_t = self.freqs_t[:T].view(T, 1, 1, -1).expand(T, H, W, -1).flatten(0, 2)
        f_h = self.freqs_h[:H].view(1, H, 1, -1).expand(T, H, W, -1).flatten(0, 2)
        f_w = self.freqs_w[:W].view(1, 1, W, -1).expand(T, H, W, -1).flatten(0, 2)

        # Apply
        q_t, k_t = apply_rotary_emb(q_t, k_t, f_t)
        q_h, k_h = apply_rotary_emb(q_h, k_h, f_h)
        q_w, k_w = apply_rotary_emb(q_w, k_w, f_w)

        # Concat back
        xq_out = torch.cat([q_t, q_h, q_w], dim=-1)
        xk_out = torch.cat([k_t, k_h, k_w], dim=-1)
        
        return xq_out, xk_out

# ==============================================================================
# == 3. Custom Transformer Block
# ==============================================================================

class JiTAttention(nn.Module):
    def __init__(self, dim, num_heads, qk_norm=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
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

        # 1. QK Norm (Paper recommendation for stability)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # 2. Apply 3D RoPE
        q, k = rope_module(q, k, T, H, W)

        # 3. Attention
        x = F.scaled_dot_product_attention(q, k, v) # Uses FlashAttn if available

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class JiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = JiTAttention(dim, num_heads, qk_norm=True)
        
        self.norm2 = RMSNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = SwiGLU(dim, hidden_dim, dim)

    def forward(self, x, rope_module, T, H, W):
        # Pre-Norm Architecture
        x = x + self.attn(self.norm1(x), rope_module, T, H, W)
        x = x + self.mlp(self.norm2(x))
        return x

# ==============================================================================
# == 4. The Full JiT-3D Model (Updated)
# ==============================================================================

class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(2, 16, 16), in_channels=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2) # (B, N, D)

class FinalLayer(nn.Module):
    """ Projects tokens back to pixels (Linear) """
    def __init__(self, patch_size, out_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.patch_dim = out_channels * patch_size[0] * patch_size[1] * patch_size[2]
        self.linear = nn.Linear(embed_dim, self.patch_dim)

    def forward(self, x, T, H, W):
        # Flattened tokens -> 3D Video
        pt, ph, pw = self.patch_size
        Tp, Hp, Wp = T // pt, H // ph, W // pw
        x = self.linear(x)
        x = x.view(x.shape[0], Tp, Hp, Wp, self.out_channels, pt, ph, pw)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return x.view(x.shape[0], self.out_channels, T, H, W)

class JiT3D_Modern(nn.Module):
    def __init__(
        self,
        img_size=(6, 128, 128),     # T, H, W
        patch_size=(2, 16, 16),
        in_channels=3,
        out_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        context_dim=128,
        time_emb_dim=64
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # Patch Embed
        self.patch_embed = PatchEmbed3D(patch_size, in_channels, embed_dim)
        
        # Calculate Grid Dimensions for RoPE
        self.grid_t = img_size[0] // patch_size[0]
        self.grid_h = img_size[1] // patch_size[1]
        self.grid_w = img_size[2] // patch_size[2]

        # RoPE (Axial)
        # Note: We pre-allocate enough size (e.g., 2x strict size) to allow flexibility
        self.rope = RoPE3D(embed_dim // num_heads, self.grid_t*2, self.grid_h*2, self.grid_w*2)
        
        # Time/Context Injection
        input_context_dim = context_dim - 1 + time_emb_dim 
        self.time_freq_emb = nn.Sequential(
            nn.Linear(1, time_emb_dim), # Simple projection or Sinusoidal
            nn.SiLU()
        )
        # We replace SinusoidalPosEmb with a simple MLP for the diffusion time scalar
        # to keep things "modern" (RoPE handles spatial/seq pos).
        self.context_mlp = nn.Sequential(
            nn.Linear(input_context_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            JiTBlock(embed_dim, num_heads, mlp_ratio=2.6) # 2.6 * 3 (GLU gates) ~= 8 
            for _ in range(depth)
        ])
        
        self.norm_final = RMSNorm(embed_dim)
        self.final_layer = FinalLayer(patch_size, out_channels, embed_dim)
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize like a modern ViT (trunc_normal)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
             nn.init.trunc_normal_(m.weight, std=0.02)

    def get_sinusoidal_time(self, t):
        device = t.device
        half_dim = 64 // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

    def forward(self, x, t):
        """
        x: (B, C, T, H, W)
        t: (B, context_dim)
        """
        B, C, T, H, W = x.shape
        
        # 1. Context Embedding
        # Extract scalar time and spatial vector
        time_val = t[:, -1] # Scalar diffusion time
        t_emb = self.get_sinusoidal_time(time_val)
        spatial = t[:, :-1]
        
        combined = torch.cat([spatial, t_emb], dim=1)
        c_emb = self.context_mlp(combined).unsqueeze(1) # (B, 1, D)

        # 2. Patchify
        x = self.patch_embed(x) # (B, N, D)
        
        # 3. Add Context (Broadcasting)
        # Note: In pure RoPE, we DON'T add absolute positional embeddings
        # We only add the conditioning context.
        x = x + c_emb

        # 4. Transformer Loop
        # We pass grid dimensions for RoPE logic
        grid_t = T // self.patch_size[0]
        grid_h = H // self.patch_size[1]
        grid_w = W // self.patch_size[2]
        
        for block in self.blocks:
            x = block(x, self.rope, grid_t, grid_h, grid_w)

        x = self.norm_final(x)

        # 5. Unpatchify (Predict Clean X)
        x = self.final_layer(x, T, H, W)
        
        return x

# ==============================================================================
# == Test
# ==============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing Modern JiT-3D on {device}")
    
    T, H, W = 6, 128, 128
    model = JiT3D_Modern(
        img_size=(T, H, W),
        patch_size=(2, 16, 16),
        embed_dim=512,
        depth=4,
        num_heads=8
    ).to(device)

    x = torch.randn(2, 3, T, H, W).to(device)
    t = torch.randn(2, 128).to(device) # Context

    out = model(x, t)
    print(f"Output shape: {out.shape}")
    
    # Check if params have gradients (sanity check)
    loss = out.sum()
    loss.backward()
    print("Backward pass successful.")