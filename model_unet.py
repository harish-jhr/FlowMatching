import torch.nn as nn
from diffusers import UNet2DModel


class UNetFlow(nn.Module):
    def __init__(self, img_size=64):
        super().__init__()
        # 4 resolution levels for 64px: 64->32->16->8->4
        # attention at 16x16 and 8x8 — flow matching needs global context
        # earlier trail had attention only at 2x2 which is too compressed to capture enough spatial semantics
        self.unet = UNet2DModel(
            sample_size=img_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
            attention_head_dim=8,
        )

    def forward(self, x, t):
        return self.unet(x, (t * 999).long()).sample

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
