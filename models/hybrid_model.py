"""Model factory for the four ablation variants (all share backbone + head code).

=====  ========  ==========================================================
ID     model     pipeline
=====  ========  ==========================================================
E1     backbone  EfficientNet-B3 -> GAP -> head
E2     cbam      EfficientNet-B3 -> CBAM -> GAP -> head
E3     dyda      EfficientNet-B3 -> DyDA -> GAP -> head
E4     hybrid    EfficientNet-B3 -> DyDA -> [GAP ; Swin encoder] -> concat -> head
=====  ========  ==========================================================
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from utils.config import ABLATION_IDS, MODEL_IDS

from .cbam import CBAM
from .classifier import ClassifierHead
from .dyda_attention import DyDA
from .efficientnet_backbone import EfficientNetB3Backbone
from .swin_feature_encoder import SwinFeatureEncoder

logger = logging.getLogger(__name__)


class HybridClassifier(nn.Module):
    """Configurable EfficientNet-B3 classifier covering E1-E4.

    Shapes (224 input, ``feature_stage=final``):
        x ``(B,3,224,224)`` -> F ``(B,1536,7,7)`` -> attention -> ``(B,1536,7,7)``
        -> v_cnn ``(B,1536)`` [+ v_swin ``(B,embed_dim)``] -> logits ``(B,num_classes)``.
    """

    def __init__(self, variant: str, backbone: EfficientNetB3Backbone, attention: Optional[nn.Module],
                 swin: Optional[SwinFeatureEncoder], head: ClassifierHead) -> None:
        super().__init__()
        if variant not in MODEL_IDS:
            raise ValueError(f"variant must be one of {MODEL_IDS}")
        self.variant = variant
        self.backbone = backbone
        self.attention = attention if attention is not None else nn.Identity()
        self.swin = swin
        self.head = head

    @property
    def ablation_id(self) -> str:
        return ABLATION_IDS[self.variant]

    @property
    def dyda(self) -> Optional[DyDA]:
        return self.attention if isinstance(self.attention, DyDA) else None

    @property
    def gradcam_target_layer(self) -> nn.Module:
        return self.backbone.target_layer

    def forward(self, x: torch.Tensor, return_gates: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        feat = self.backbone(x)
        gates = None
        if self.dyda is not None:
            feat, gates = self.dyda(feat, return_gates=True)
        else:
            feat = self.attention(feat)
        vec = torch.mean(feat, dim=(2, 3))                   # GAP -> (B, C)
        if self.swin is not None:
            vec = torch.cat([vec, self.swin(feat)], dim=1)   # (B, C + embed_dim)
        logits = self.head(vec)
        assert logits.shape == (x.shape[0], self.head.fc.out_features)
        return (logits, gates) if return_gates else logits


def build_model(cfg: Dict[str, Any], variant: Optional[str] = None) -> HybridClassifier:
    """Build E1-E4 from config. ``variant`` defaults to ``cfg['experiment']['model']``."""
    variant = variant or cfg["experiment"]["model"]
    if variant not in MODEL_IDS:
        raise ValueError(f"Unknown model variant {variant!r}; choose from {MODEL_IDS}")
    m = cfg["model"]
    num_classes = len(cfg["classes"])
    backbone = EfficientNetB3Backbone(pretrained=m["pretrained"], feature_stage=m["feature_stage"],
                                      freeze=m["freeze_backbone"])
    c, h, w = backbone.output_shape(cfg["preprocessing"]["img_size"])
    attention: Optional[nn.Module] = None
    swin = None
    if variant == "cbam":
        attention = CBAM(c, m["cbam"]["reduction"], m["cbam"]["spatial_kernel"])
    elif variant in ("dyda", "hybrid"):
        attention = DyDA(c, m["dyda"]["reduction"], m["dyda"]["spatial_kernel"], m["dyda"]["gate_hidden"])
    if variant == "hybrid":
        s = m["swin"]
        swin = SwinFeatureEncoder(
            in_channels=c, in_size=(h, w), embed_dim=s["embed_dim"], depth=s["depth"], num_heads=s["num_heads"],
            window_size=s["window_size"], shift_size=s["shift_size"], upsample_size=s["upsample_size"],
            mlp_ratio=s["mlp_ratio"], drop_path=s["drop_path"], attn_drop=s["attn_drop"],
            proj_drop=s["proj_drop"])
    head_in = c + (swin.embed_dim if swin is not None else 0)
    model = HybridClassifier(variant, backbone, attention, swin, ClassifierHead(head_in, num_classes, m["dropout"]))
    logger.info("Built %s (%s): %s params, backbone pretrained=%s", variant, ABLATION_IDS[variant],
                f"{count_parameters(model):,}", backbone.pretrained_loaded)
    return model


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def parameter_breakdown(model: HybridClassifier) -> Dict[str, int]:
    """Parameter counts per component."""
    out = {
        "backbone": count_parameters(model.backbone),
        "attention": count_parameters(model.attention),
        "swin": count_parameters(model.swin) if model.swin is not None else 0,
        "head": count_parameters(model.head),
    }
    out["total"] = sum(out.values())
    return out
