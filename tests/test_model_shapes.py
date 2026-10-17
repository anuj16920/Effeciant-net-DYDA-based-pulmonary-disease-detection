"""End-to-end model shape/gradient tests for E1-E4 (synthetic tensors, random init)."""
from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from models.hybrid_model import build_model, parameter_breakdown
from models.swin_feature_encoder import SwinFeatureEncoder

VARIANTS = ["backbone", "cbam", "dyda", "hybrid"]


@pytest.fixture(scope="module")
def fast_cfg(base_cfg):
    c = copy.deepcopy(base_cfg)
    c["model"]["pretrained"] = False
    c["model"]["swin"].update(embed_dim=64, num_heads=4)
    return c


@pytest.fixture(scope="module")
def models(fast_cfg):
    torch.manual_seed(0)
    return {v: build_model(fast_cfg, v) for v in VARIANTS}


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("batch", [1, 4])
def test_forward_backward(models, variant, batch):
    model = models[variant]
    model.train()
    model.zero_grad(set_to_none=True)
    x = torch.randn(batch, 3, 224, 224)
    y = torch.randint(0, 4, (batch,))
    logits, gates = model(x, return_gates=True)
    assert logits.shape == (batch, 4)
    assert torch.isfinite(logits).all()
    loss = nn.functional.cross_entropy(logits, y)
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is None or torch.isfinite(g).all() for g in grads)
    assert model.backbone.stem[0].weight.grad.abs().sum() > 0
    if variant in ("dyda", "hybrid"):
        assert gates.shape == (batch, 2)
        assert torch.allclose(gates.sum(1), torch.ones(batch), atol=1e-6)
        for name, p in model.attention.named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, f"no grad reached DyDA param {name}"
    else:
        assert gates is None
    if variant == "hybrid":
        for name, p in model.swin.named_parameters():
            assert p.grad is not None, f"no grad reached Swin param {name}"
        assert model.swin.blocks[1].attn.qkv.weight.grad.abs().sum() > 0


def test_eval_mode_batch_one_is_deterministic(models):
    m = models["hybrid"].eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(m(x), m(x))


def test_head_input_dims(models, fast_cfg):
    assert models["backbone"].head.in_features == 1536
    assert models["hybrid"].head.in_features == 1536 + fast_cfg["model"]["swin"]["embed_dim"]
    bd = parameter_breakdown(models["hybrid"])
    assert bd["swin"] > 0 and bd["attention"] > 0
    assert parameter_breakdown(models["backbone"])["attention"] == 0


def test_shifted_window_attention_is_really_active(models):
    swin: SwinFeatureEncoder = models["hybrid"].swin
    assert swin.grid == (14, 14) and swin.num_windows == 4
    w_msa, sw_msa = swin.blocks[0], swin.blocks[1]
    assert tuple(w_msa.shift_size) == (0, 0) and w_msa.attn_mask is None
    assert tuple(sw_msa.shift_size) == (3, 3)
    assert sw_msa.attn_mask is not None and tuple(sw_msa.attn_mask.shape) == (4, 49, 49)
    assert (sw_msa.attn_mask < 0).any(), "mask must block attention across shifted-region borders"
    # The mask changes the result: zeroing it alters the SW-MSA output.
    sw_msa.eval()
    x = torch.randn(2, 14, 14, swin.embed_dim)
    with torch.no_grad():
        masked = sw_msa(x)
        saved = sw_msa.attn_mask.clone()
        sw_msa.attn_mask.zero_()
        unmasked = sw_msa(x)
        sw_msa.attn_mask.copy_(saved)
    assert not torch.allclose(masked, unmasked)


def test_single_window_grid_is_rejected():
    with pytest.raises(ValueError, match="no-op"):
        SwinFeatureEncoder(in_channels=32, in_size=(7, 7), embed_dim=32, num_heads=4, window_size=7,
                           shift_size=3, upsample_size=None)


def test_stride16_variant(fast_cfg):
    c = copy.deepcopy(fast_cfg)
    c["model"]["feature_stage"] = "stride16"
    c["model"]["swin"]["upsample_size"] = None
    m = build_model(c, "hybrid").eval()
    with torch.no_grad():
        feat = m.backbone(torch.randn(1, 3, 224, 224))
        assert feat.shape == (1, 136, 14, 14)
        assert m(torch.randn(2, 3, 224, 224)).shape == (2, 4)
    assert m.swin.grid == (14, 14) and m.swin.blocks[1].attn_mask is not None


def test_backbone_feature_shapes(models):
    with torch.no_grad():
        f = models["backbone"].eval().backbone(torch.randn(2, 3, 224, 224))
    assert f.shape == (2, 1536, 7, 7)
    assert models["backbone"].backbone.output_shape(224) == (1536, 7, 7)


def test_freeze_backbone(fast_cfg):
    c = copy.deepcopy(fast_cfg)
    c["model"]["freeze_backbone"] = True
    m = build_model(c, "dyda").train()
    assert not any(p.requires_grad for p in m.backbone.parameters())
    assert all(not bn.training for bn in m.backbone.modules() if isinstance(bn, nn.BatchNorm2d))
    assert any(p.requires_grad for p in m.attention.parameters())
