import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import math

# ==============================================================================
# == Conditioning Blocks
# ==============================================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        if half_dim == 0:
            # For dim=1, use sin
            return torch.sin(x).unsqueeze(-1)
        elif half_dim == 1:
            # For dim=2, use sin and cos with scale 1
            emb = x[:, None] * 1.0
            return torch.cat((emb.sin(), emb.cos()), dim=-1)
        else:
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
            emb = x[:, None] * emb[None, :]
            emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
            return emb


class FilmLayer(nn.Module):
    def __init__(self, embedding_dim, num_channels):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(embedding_dim, num_channels * 2), nn.ReLU())

    def forward(self, x, context):
        mlp_out = self.mlp(context)
        scale = mlp_out[:, : x.shape[1]]
        bias = mlp_out[:, x.shape[1] :]

        scale = scale.view(x.shape[0], x.shape[1], 1, 1, 1)
        bias = bias.view(x.shape[0], x.shape[1], 1, 1, 1)

        return (1.0 + scale) * x + bias


# ==============================================================================
# == 3D U-Net Components
# ==============================================================================

class CausalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        
        # Time padding: We pad (k-1) on the left (past) and 0 on the right (future)
        # Spatial padding: We pad symmetrically usually (k // 2)
        
        # Assuming kernel_size[0] is time, [1] is height, [2] is width
        time_kernel = kernel_size[0]
        self.time_pad = time_kernel - 1
        
        # Spatial padding (standard "same" padding logic for odd kernels)
        spatial_pad_h = kernel_size[1] // 2
        spatial_pad_w = kernel_size[2] // 2
        
        # We use manual padding for Time, but let Conv3d handle Spatial padding
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=(0, spatial_pad_h, spatial_pad_w), # 0 padding for time in the layer
            bias=bias
        )

    def forward(self, x):
        # F.pad input format: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        # We only want to pad the "front" (past) of the time dimension.
        x = F.pad(x, (0, 0, 0, 0, self.time_pad, 0))
        return self.conv(x)

class ResNetBlock3D(nn.Module):
    """
    A 3D ResNet block with FiLM conditioning.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_dim: int,
        context_frames: int,
        causal=False,
    ):
        super().__init__()
        self.context_frames = context_frames

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            bias=False,
        )
        self.bn1 = nn.Identity()  # nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

        # UPDATED: Use CausalConv3d if necessary
        if causal:
            self.conv2 = CausalConv3d(
                out_channels,
                out_channels,
                kernel_size=(3, 3, 3),
                bias=False,
            )
        else:

            self.conv2 = nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                bias=False,
            )

        self.bn2 = nn.InstanceNorm3d(out_channels, affine=True)

        self.film = FilmLayer(embedding_dim, out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
            )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.bn1(self.conv1(x)))

        # Apply FiLM only to the frames after context_frames
        #h_context = h[:, :, : self.context_frames, :, :]
        #h_noisy = h[:, :, self.context_frames :, :, :]

        #h_noisy_filmed = self.film(h_noisy, context)
        h = self.film(h, context)

        #h = torch.cat([h_context, h_noisy_filmed], dim=2)

        h = self.bn2(self.conv2(h))
        return self.relu(h + self.shortcut(x))


# ==============================================================================
# == Full 3D U-Net Architecture with Temporal Reduction in last 2 downsamples
# ==============================================================================
class UNet3DFILMTimeReduce(nn.Module):
    """
    A 3D U-Net architecture where the last 2 downsamplings also reduce time dim.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: List[int] = [32, 64, 128, 256],
        context_dim: int = 4,
        embedding_dim: int = 128,
        context_frames: int = 4,
        num_additional_resnet_blocks: int = 0,
        time_emb_dim: int = 64,
        causal=False,
    ):
        super().__init__()
        self.features = features
        self.context_dim = context_dim
        self.embedding_dim = embedding_dim
        self.context_frames = context_frames
        self.num_additional_resnet_blocks = num_additional_resnet_blocks
        self.time_emb_dim = time_emb_dim

        # --- Time Embedding ---
        time_mlp_input_dim = context_dim - 1 + self.time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_mlp_input_dim, 128), nn.ReLU(), nn.Linear(128, embedding_dim)
        )
        self.time_emb = SinusoidalPosEmb(dim=self.time_emb_dim)

        self.encoder_convs = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.down_strides = []

        # --- Encoder (Downsampling Path) ---
        current_channels = in_channels
        for i, feature in enumerate(features):
            self.encoder_convs.append(
                ResNetBlock3D(
                    current_channels, feature * 2, self.embedding_dim, self.context_frames, causal
                )
            )
            if i >= len(features) - 2:
                stride = (2, 2, 2)
            else:
                stride = (1, 2, 2)
            self.down_strides.append(stride)
            self.downs.append(nn.MaxPool3d(kernel_size=stride, stride=stride))
            current_channels = feature * 2

        # --- Bottleneck ---
        bottleneck_channels = features[-1] * 2
        self.bottleneck = ResNetBlock3D(
            bottleneck_channels, bottleneck_channels, self.embedding_dim, self.context_frames, causal
        )

        # --- Decoder (Upsampling Path) ---
        for feature in reversed(features):
            self.decoder_convs.append(
                ResNetBlock3D(feature * 4, feature, self.embedding_dim, self.context_frames, causal)
            )

        self.additional_resnet_blocks = nn.ModuleList()
        for feature in reversed(features):
            blocks = nn.ModuleList()
            for _ in range(self.num_additional_resnet_blocks):
                blocks.append(
                    ResNetBlock3D(feature, feature, self.embedding_dim, self.context_frames, causal)
                )
            self.additional_resnet_blocks.append(blocks)

        # --- Final Output Layer ---
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=(1, 1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_val = t[:, -1]
        emb = self.time_emb(time_val)
        spatial = t[:, :-1]
        combined = torch.cat([spatial, emb], dim=1)
        context = self.time_mlp(combined)
        skip_connections = []

        # --- Encoder Path ---
        for i in range(len(self.features)):
            x = self.encoder_convs[i](x, context)
            skip_connections.append(x)
            x = self.downs[i](x)

        # --- Bottleneck ---
        x = self.bottleneck(x, context)

        # --- Decoder Path ---
        skip_connections = skip_connections[::-1]
        down_strides_rev = self.down_strides[::-1]
        for i in range(len(self.decoder_convs)):
            x = F.interpolate(x, scale_factor=down_strides_rev[i], mode="nearest")
            skip_connection = skip_connections[i]

            if x.shape[2:] != skip_connection.shape[2:]:
                x = F.interpolate(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder_convs[i](concat_skip, context)

            for block in self.additional_resnet_blocks[i]:
                x = block(x, context)

        return self.final_conv(x)


# --- Example Usage ---
if __name__ == "__main__":
    print(
        "--- Testing Full 3D U-Net FiLM with Temporal Downsampling in last 2 layers ---"
    )

    # Define model parameters
    CONTEXT_FRAMES = 4
    IMG_DEPTH = CONTEXT_FRAMES + 2
    IMG_HEIGHT, IMG_WIDTH = 128, 128
    IN_CHANNELS = 3
    OUT_CHANNELS = 3
    BATCH_SIZE = 2
    CONTEXT_DIM = 128

    # Create a random input tensor (N, C, D, H, W)
    input_tensor = torch.randn(
        BATCH_SIZE, IN_CHANNELS, IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH
    )
    t = torch.rand(BATCH_SIZE, CONTEXT_DIM)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Time shape: {t.shape}")

    # Initialize the model
    model = UNet3DFILMTimeReduce(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        features=[64, 128, 256],
        context_dim=CONTEXT_DIM,
        embedding_dim=128,
        context_frames=CONTEXT_FRAMES,
        num_additional_resnet_blocks=3,
    )

    # Perform a forward pass
    output_tensor = model(input_tensor, t)

    print(f"Output shape: {output_tensor.shape}")

    # Verify the output shape is as expected
    expected_shape = (BATCH_SIZE, OUT_CHANNELS, IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH)
    assert output_tensor.shape == expected_shape, (
        f"Shape mismatch! Expected {expected_shape}, got {output_tensor.shape}"
    )

    print("✅ 3D U-Net FiLM Time Reduce model shape test PASSED.")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")