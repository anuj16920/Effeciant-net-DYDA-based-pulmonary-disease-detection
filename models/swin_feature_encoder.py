"""Feature-level Swin encoder operating on the DyDA feature map (not on raw RGB).

Pipeline (default config, ``feature_stage=final``, 224x224 input)::

    F_DyDA (B,1536,7,7) -> 1x1 conv adapter (B,256,7,7) -> bilinear upsample (B,256,14,14)
      -> NHWC tokens (B,14,14,256) -> [W-MSA block (shift 0), SW-MSA block (shift 3)] x depth/2
      -> LayerNorm -> mean over tokens -> v_swin (B,256)

The blocks are ``timm.models.swin_transformer.SwinTransformerBlock`` (real windowed
multi-head self-attention with relative position bias; shifted blocks use a cyclic
shift plus an attention mask).

7x7 pitfall: with a 7x7 grid and ``window_size=7`` there is a single window; timm then
silently sets ``shift_size=0`` and drops the mask, so SW-MSA degenerates to W-MSA. This
encoder refuses such configurations (``grid // window_size < 2``) and verifies after
construction that every odd block really has a non-zero shift and an attention mask.
Note that when the grid is obtained by *upsampling* a 7x7 map, the windows/shift are
genuinely exercised but the underlying information resolution is still 7x7.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F_

import utils.common  # noqa: F401
from timm.models.swin_transformer import SwinTransformerBlock


class SwinFeatureEncoder(nn.Module):
    """Swin-style encoder over a CNN feature map.

    Args:
        in_channels: channels of the incoming feature map.
        in_size: expected ``(H, W)`` of the incoming feature map.
        embed_dim: token dimension after the 1x1 adapter.
        depth: number of Swin blocks (even blocks W-MSA, odd blocks SW-MSA). Must be >= 2.
        num_heads: attention heads (``embed_dim % num_heads == 0``).
        window_size: square window side.
        shift_size: shift used by SW-MSA blocks (``0 < shift_size < window_size``).
        upsample_size: if set, the adapted map is bilinearly resized to
            ``(upsample_size, upsample_size)`` before tokenisation.

    Shapes:
        ``(B, in_channels, H, W) -> (B, embed_dim)``.
    """

    def __init__(self, in_channels: int, in_size: Tuple[int, int], embed_dim: int = 256, depth: int = 2,
                 num_heads: int = 8, window_size: int = 7, shift_size: int = 3,
                 upsample_size: Optional[int] = 14, mlp_ratio: float = 4.0, drop_path: float = 0.1,
                 attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2 so that both W-MSA and SW-MSA blocks exist")
        if not 0 < shift_size < window_size:
            raise ValueError(f"need 0 < shift_size < window_size, got shift={shift_size}, window={window_size}")
        self.in_channels = in_channels
        self.in_size = tuple(in_size)
        self.grid = (upsample_size, upsample_size) if upsample_size else self.in_size
        gh, gw = self.grid
        if gh % window_size or gw % window_size:
            raise ValueError(f"token grid {self.grid} must be divisible by window_size {window_size}")
        if gh // window_size < 2 or gw // window_size < 2:
            raise ValueError(
                f"token grid {self.grid} with window_size {window_size} gives fewer than 2x2 windows, so the "
                "shifted-window (SW-MSA) blocks would be a no-op. Set model.swin.upsample_size (e.g. 14) or use "
                "model.feature_stage=stride16.")
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.adapter = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=embed_dim, input_resolution=self.grid, num_heads=num_heads, window_size=window_size,
                shift_size=0 if i % 2 == 0 else shift_size, mlp_ratio=mlp_ratio,
                proj_drop=proj_drop, attn_drop=attn_drop, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self._verify_shifted_windows()

    def _verify_shifted_windows(self) -> None:
        for i, blk in enumerate(self.blocks):
            shift = tuple(blk.shift_size)
            if i % 2 == 1:
                if shift != (self.shift_size, self.shift_size) or blk.attn_mask is None:
                    raise RuntimeError(
                        f"Swin block {i} is not a real SW-MSA block (shift={shift}, mask="
                        f"{'None' if blk.attn_mask is None else tuple(blk.attn_mask.shape)}); timm disabled shifting.")
            elif shift != (0, 0):
                raise RuntimeError(f"Swin block {i} should be W-MSA (shift 0), got shift={shift}")

    @property
    def num_windows(self) -> int:
        return (self.grid[0] // self.window_size) * (self.grid[1] // self.window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4 and x.shape[1] == self.in_channels, \
            f"SwinFeatureEncoder expected (B,{self.in_channels},H,W), got {tuple(x.shape)}"
        x = self.adapter(x)                                    # (B, D, h, w)
        if tuple(x.shape[-2:]) != self.grid:
            if self.grid == self.in_size:
                raise ValueError(f"feature map {tuple(x.shape[-2:])} does not match expected {self.in_size}; "
                                 "check preprocessing.img_size")
            x = F_.interpolate(x, size=self.grid, mode="bilinear", align_corners=False)
        x = x.permute(0, 2, 3, 1).contiguous()                 # (B, gh, gw, D) NHWC tokens
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x.mean(dim=(1, 2))                              # (B, D)
