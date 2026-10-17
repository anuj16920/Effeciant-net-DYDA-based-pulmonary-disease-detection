r"""DyDA: Dynamic Dual Attention.

Given a feature map :math:`F \in \mathbb{R}^{B\times C\times H\times W}`:

.. math::

    A_c &= \sigma\big(\mathrm{MLP}(\mathrm{GAP}(F)) + \mathrm{MLP}(\mathrm{GMP}(F))\big) \in \mathbb{R}^{B\times C\times 1\times 1}\\
    A_s &= \sigma\big(\mathrm{conv}_{k}([\mathrm{mean}_c(F);\ \max_c(F)])\big) \in \mathbb{R}^{B\times 1\times H\times W}\\
    F_{ch} &= F\odot A_c,\qquad F_{sp} = F\odot A_s\\
    [\alpha, \beta] &= \mathrm{softmax}\big(g(\mathrm{GAP}(F))\big) \in \mathbb{R}^{B\times 2}\quad(\alpha+\beta=1,\ \text{per sample})\\
    F_{DyDA} &= \alpha\,F_{ch} + \beta\,F_{sp}

Both attentions are computed from the *same* input ``F`` (parallel branches), and the
mixing weights are produced per sample by the gate network ``g``. This differs from
CBAM (sequential, fixed order, no mixing) and from DANet-style fusion (a global,
input-independent sum/weight).
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F_

from .cbam import ChannelAttention, SpatialAttention

logger = logging.getLogger(__name__)


class DyDA(nn.Module):
    """Dynamic Dual Attention.

    Args:
        channels: number of input channels ``C``.
        reduction: channel-MLP reduction ratio.
        spatial_kernel: odd kernel size of the spatial-attention conv (default 7).
        gate_hidden: hidden width of the gate MLP ``g``.

    Shapes:
        ``forward(F)``: ``(B, C, H, W) -> (B, C, H, W)``; with ``return_gates=True`` also
        returns gates of shape ``(B, 2)`` with columns ``[alpha, beta]``.

    Diagnostics:
        :attr:`last_gates` holds the detached gates from the most recent forward pass, and
        callables registered with :meth:`add_gate_hook` receive ``(gates,)`` on each pass.
    """

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7, gate_hidden: int = 64) -> None:
        super().__init__()
        self.channels = channels
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)
        self.gate = nn.Sequential(
            nn.Linear(channels, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, 2),
        )
        self.last_gates: Optional[torch.Tensor] = None
        self._gate_hooks: List[Callable[[torch.Tensor], None]] = []

    def add_gate_hook(self, fn: Callable[[torch.Tensor], None]) -> Callable[[], None]:
        """Register ``fn(gates)``; returns a function that removes the hook."""
        self._gate_hooks.append(fn)
        return lambda: self._gate_hooks.remove(fn)

    def compute_gates(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample gates ``(B, 2)`` = softmax(g(GAP(F)))."""
        return F_.softmax(self.gate(torch.mean(x, dim=(2, 3))), dim=1)

    def forward(self, x: torch.Tensor, return_gates: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert x.dim() == 4 and x.shape[1] == self.channels, \
            f"DyDA expected (B,{self.channels},H,W), got {tuple(x.shape)}"
        a_c = self.channel_attention(x)            # (B, C, 1, 1)
        a_s = self.spatial_attention(x)            # (B, 1, H, W)
        gates = self.compute_gates(x)              # (B, 2)
        alpha = gates[:, 0].view(-1, 1, 1, 1)
        beta = gates[:, 1].view(-1, 1, 1, 1)
        out = alpha * (x * a_c) + beta * (x * a_s)
        assert out.shape == x.shape
        self.last_gates = gates.detach()
        for fn in self._gate_hooks:
            fn(self.last_gates)
        return (out, gates) if return_gates else out
