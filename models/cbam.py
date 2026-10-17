"""Standard CBAM (Woo et al., 2018): sequential channel attention then spatial attention.

Used as the E2 ablation baseline. Unlike DyDA, the two attentions are applied in a
fixed order with no input-dependent mixing.
"""
from __future__ import annotations

import torch
from torch import nn


class ChannelAttention(nn.Module):
    """``A_c = sigmoid(MLP(GAP(F)) + MLP(GMP(F)))``; output shape ``(B, C, 1, 1)``."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        mx = torch.amax(x, dim=(2, 3), keepdim=True)
        return torch.sigmoid(self.mlp(avg) + self.mlp(mx))


class SpatialAttention(nn.Module):
    """``A_s = sigmoid(conv_k([mean_c(F); max_c(F)]))``; output shape ``(B, 1, H, W)``."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        desc = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return torch.sigmoid(self.conv(desc))


class CBAM(nn.Module):
    """``F' = F * A_c(F)``; ``F'' = F' * A_s(F')``. Shape preserved: ``(B,C,H,W) -> (B,C,H,W)``."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel(x)
        return x * self.spatial(x)
