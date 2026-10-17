"""Classification head: Dropout -> Linear."""
from __future__ import annotations

import torch
from torch import nn


class ClassifierHead(nn.Module):
    """``(B, in_features) -> (B, num_classes)`` raw logits (no softmax)."""

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.in_features = in_features
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.shape[1] == self.in_features, \
            f"head expected (B,{self.in_features}), got {tuple(x.shape)}"
        return self.fc(self.dropout(x))
