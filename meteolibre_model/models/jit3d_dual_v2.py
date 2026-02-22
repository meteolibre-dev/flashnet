import torch
import torch.nn as nn

from meteolibre_model.models.jit3d import JiT3D_Modern


class DualJiT3D(nn.Module):
    """
    A wrapper for JiT3D_Modern that handles dual inputs (satellite and KPI)
    and produces dual outputs. The KPI input is encoded via 1x1 conv and
    concatenated with the satellite input before being passed to the transformer.
    """

    def __init__(
        self,
        sat_in_channels: int,
        kpi_in_channels: int,
        sat_out_channels: int,
        kpi_out_channels: int,
        img_size=(7, 128, 128),
        patch_size=(1, 16, 16),
        embed_dim=768,
        depth=12,
        num_heads=12,
        context_dim=128,
        time_emb_dim=64,
        context_frames = 4,
    ):
        super().__init__()
        self.sat_out_channels = sat_out_channels
        self.context_frames = context_frames

        self.jit = JiT3D_Modern(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=sat_in_channels + kpi_in_channels,
            out_channels=sat_out_channels + kpi_out_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            context_dim=context_dim,
            time_emb_dim=time_emb_dim,
        )

    def forward(self, sat_input: torch.Tensor, kpi_input: torch.Tensor, context: torch.Tensor):

        combined_input = torch.cat([kpi_input, sat_input], dim=1)
        pred = self.jit(combined_input, context)
        sat_pred = pred[:, :self.sat_out_channels]
        kpi_pred = pred[:, self.sat_out_channels:]
        return sat_pred, kpi_pred


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing DualJiT3D on {device}")

    T, H, W = 7, 128, 128
    B = 2

    model = DualJiT3D(
        sat_in_channels=3,
        kpi_in_channels=4,
        sat_out_channels=3,
        kpi_out_channels=2,
        img_size=(T, H, W),
        patch_size=(1, 16, 16),
        embed_dim=512,
        depth=4,
        num_heads=8,
        context_dim=128,
        intermediate_dim=4,
    ).to(device)

    sat_input = torch.randn(B, 3, T, H, W).to(device)
    kpi_input = torch.randn(B, 4, T, H, W).to(device)
    context = torch.randn(B, 128).to(device)

    sat_out, kpi_out = model(sat_input, kpi_input, context)
    print(f"Sat output shape: {sat_out.shape}")
    print(f"KPI output shape: {kpi_out.shape}")

    loss = sat_out.sum() + kpi_out.sum()
    loss.backward()
    print("Backward pass successful.")
