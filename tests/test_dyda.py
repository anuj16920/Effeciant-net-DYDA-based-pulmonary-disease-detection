"""DyDA unit tests (synthetic tensors only)."""
from __future__ import annotations

import pytest
import torch

from models.cbam import CBAM
from models.dyda_attention import DyDA


@pytest.fixture()
def dyda():
    torch.manual_seed(0)
    return DyDA(channels=32, reduction=4, spatial_kernel=7, gate_hidden=16)


def test_gates_sum_to_one_and_are_per_sample(dyda):
    x = torch.randn(6, 32, 7, 7) * torch.linspace(0.1, 5, 6).view(-1, 1, 1, 1)
    out, gates = dyda(x, return_gates=True)
    assert gates.shape == (6, 2)
    assert torch.allclose(gates.sum(dim=1), torch.ones(6), atol=1e-6)
    assert torch.all(gates > 0) and torch.all(gates < 1)
    # Different inputs -> different gate logits -> different gates (not a global constant).
    logits = dyda.gate(x.mean(dim=(2, 3)))
    assert not torch.allclose(logits[0], logits[-1])
    assert gates[:, 0].std() > 1e-4


def test_gates_are_input_dependent_not_parameters(dyda):
    names = [n for n, _ in dyda.named_parameters()]
    assert not any(n in ("alpha", "beta") for n in names)
    a = dyda(torch.randn(1, 32, 7, 7), return_gates=True)[1]
    b = dyda(torch.randn(1, 32, 7, 7) * 10 + 3, return_gates=True)[1]
    assert not torch.allclose(a, b)


def test_shape_preserved_and_batch_size_one(dyda):
    for shape in [(1, 32, 7, 7), (4, 32, 14, 14), (2, 32, 5, 9)]:
        x = torch.randn(*shape)
        assert dyda(x).shape == x.shape
    dyda.train()
    out = dyda(torch.randn(1, 32, 7, 7))
    out.sum().backward()
    assert dyda.gate[0].weight.grad is not None


def test_output_matches_equation(dyda):
    x = torch.randn(3, 32, 7, 7)
    out, gates = dyda(x, return_gates=True)
    a_c = dyda.channel_attention(x)
    a_s = dyda.spatial_attention(x)
    assert a_c.shape == (3, 32, 1, 1) and a_s.shape == (3, 1, 7, 7)
    expected = gates[:, 0].view(-1, 1, 1, 1) * x * a_c + gates[:, 1].view(-1, 1, 1, 1) * x * a_s
    assert torch.allclose(out, expected, atol=1e-6)


def test_extreme_gates_select_one_branch(dyda):
    x = torch.randn(2, 32, 7, 7)
    with torch.no_grad():
        dyda.gate[2].weight.zero_()
        dyda.gate[2].bias.copy_(torch.tensor([50.0, -50.0]))  # alpha ~ 1
    out = dyda(x)
    assert torch.allclose(out, x * dyda.channel_attention(x), atol=1e-5)


def test_gate_logging_hook_and_last_gates(dyda):
    seen = []
    remove = dyda.add_gate_hook(lambda g: seen.append(g.clone()))
    dyda(torch.randn(3, 32, 7, 7))
    remove()
    dyda(torch.randn(3, 32, 7, 7))
    assert len(seen) == 1 and seen[0].shape == (3, 2)
    assert dyda.last_gates is not None and not dyda.last_gates.requires_grad


def test_wrong_channels_raise(dyda):
    with pytest.raises(AssertionError):
        dyda(torch.randn(1, 16, 7, 7))


def test_cbam_is_sequential_channel_then_spatial():
    cbam = CBAM(32, 4, 7)
    x = torch.randn(2, 32, 7, 7)
    x_c = x * cbam.channel(x)
    assert torch.allclose(cbam(x), x_c * cbam.spatial(x_c), atol=1e-6)
    assert cbam(x).shape == x.shape
