import torch
import torch.nn as nn

from meteolibre_model.models.unet3d_film_time_reduce import UNet3DFILMTimeReduce


class DualUNet3DFILMTimeReduce(nn.Module):
    """
    A single UNet3DFILMTimeReduce with a simple MLP encoder for KPI input, and modified final layer for dual outputs.
    """

    def __init__(
        self,
        sat_in_channels: int,
        kpi_in_channels: int,
        sat_out_channels: int,
        kpi_out_channels: int,
        additional_channels: int = 32,
        features: list = [32, 64, 128, 256],
        context_dim: int = 4,
        embedding_dim: int = 128,
        context_frames: int = 4,
        num_additional_resnet_blocks: int = 0,
        time_emb_dim: int = 64,
        causal=False,
    ):
        super().__init__()
        self.sat_out_channels = sat_out_channels
        self.context_frames = context_frames

        # KPI encoder: 1x1 conv to encode channels
        self.kpi_encoder = nn.Conv3d(kpi_in_channels, additional_channels, kernel_size=1)

        print("causal definition is: ", causal)

        # Single UNet with combined inputs
        self.unet = UNet3DFILMTimeReduce(
            in_channels=sat_in_channels + additional_channels,
            out_channels=sat_out_channels + kpi_out_channels,
            features=features,
            context_dim=context_dim,
            embedding_dim=embedding_dim,
            context_frames=context_frames,
            num_additional_resnet_blocks=num_additional_resnet_blocks,
            time_emb_dim=time_emb_dim,
            causal=causal
        )

    def forward(self, sat_input: torch.Tensor, kpi_input: torch.Tensor, context: torch.Tensor):
        # Encode KPI channels
        encoded_kpi = self.kpi_encoder(kpi_input)

        # Concatenate with sat_input
        combined_input = torch.cat([sat_input, encoded_kpi], dim=1)

        # Forward through UNet
        pred = self.unet(combined_input, context)

        # Split outputs
        sat_pred = pred[:, :self.sat_out_channels]
        kpi_pred = pred[:, self.sat_out_channels:]

        return sat_pred, kpi_pred