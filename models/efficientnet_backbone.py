"""EfficientNet-B3 feature extractor (timm).

Real feature-map shapes for a 224x224 input (timm 1.0.24, verified in Phase 0):

* ``feature_stage="final"``    -> conv_head + bn2 output, ``(B, 1536, 7, 7)``  (stride 32)
* ``feature_stage="stride16"`` -> output of ``blocks[4]``, ``(B, 136, 14, 14)`` (stride 16)
"""
from __future__ import annotations

import logging
from typing import Tuple

import torch
from torch import nn

import utils.common  # noqa: F401  (sets TRANSFORMERS_VERBOSITY before timm import)
import timm

logger = logging.getLogger(__name__)

_STAGE_SPEC = {
    # name: (number of timm `blocks` stages kept, use conv_head/bn2, channels, reduction)
    "final": (7, True, 1536, 32),
    "stride16": (5, False, 136, 16),
}


class EfficientNetB3Backbone(nn.Module):
    """EfficientNet-B3 truncated at a configurable stage.

    Args:
        pretrained: load ImageNet weights (``efficientnet_b3.ra2_in1k``). On failure a
            warning is logged and the network is randomly initialised; check
            :attr:`pretrained_loaded`.
        feature_stage: ``"final"`` or ``"stride16"`` (see module docstring).
        freeze: if True, backbone parameters get ``requires_grad=False`` and its
            BatchNorm layers stay in eval mode.

    Shapes:
        input ``(B, 3, H, W)`` -> output ``(B, C, H/r, W/r)`` with ``C = out_channels``,
        ``r = reduction``.
    """

    def __init__(self, pretrained: bool = True, feature_stage: str = "final", freeze: bool = False) -> None:
        super().__init__()
        if feature_stage not in _STAGE_SPEC:
            raise ValueError(f"feature_stage must be one of {list(_STAGE_SPEC)}, got {feature_stage!r}")
        n_stages, use_head, channels, reduction = _STAGE_SPEC[feature_stage]
        self.pretrained_loaded = False
        net = None
        if pretrained:
            try:
                net = timm.create_model("efficientnet_b3", pretrained=True, num_classes=0, global_pool="")
                self.pretrained_loaded = True
            except Exception as exc:  # network / cache / hub errors
                logger.warning(
                    "Could not load pretrained EfficientNet-B3 weights (%s: %s). Falling back to RANDOM "
                    "initialisation. Set model.pretrained=false to silence this, or pre-download the weights "
                    "(timm/efficientnet_b3.ra2_in1k) into the HuggingFace cache.", type(exc).__name__, exc)
        if net is None:
            net = timm.create_model("efficientnet_b3", pretrained=False, num_classes=0, global_pool="")

        self.feature_stage = feature_stage
        self.out_channels = channels
        self.reduction = reduction
        self.stem = nn.Sequential(net.conv_stem, net.bn1)
        self.blocks = nn.Sequential(*list(net.blocks.children())[:n_stages])
        self.head = nn.Sequential(net.conv_head, net.bn2) if use_head else nn.Identity()
        self.frozen = freeze
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)

    @property
    def target_layer(self) -> nn.Module:
        """Module whose output is the returned feature map (default Grad-CAM target)."""
        return self.head if isinstance(self.head, nn.Sequential) else self.blocks[-1]

    def train(self, mode: bool = True) -> "EfficientNetB3Backbone":
        super().train(mode)
        if self.frozen:  # keep BN running stats fixed when frozen
            for m in self.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    def set_frozen(self, frozen: bool) -> None:
        """Freeze or unfreeze all backbone parameters."""
        self.frozen = frozen
        for p in self.parameters():
            p.requires_grad_(not frozen)
        self.train(self.training)

    def output_shape(self, img_size: int) -> Tuple[int, int, int]:
        """``(C, H, W)`` of the feature map for a square ``img_size`` input."""
        side = -(-img_size // self.reduction)  # ceil, matches timm 'same'-style padding at stride 2
        return self.out_channels, side, side

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4 and x.shape[1] == 3, f"expected (B,3,H,W), got {tuple(x.shape)}"
        return self.head(self.blocks(self.stem(x)))
