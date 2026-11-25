"""A small CNN whose penultimate 128-d features feed a linear head (the part the Laplace baseline treats as Bayesian)."""

from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int = 8, p_drop: float = 0.1, p_fc: float = 0.3, width: int = 32):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(o, o, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(p_drop),
            )

        self.body = nn.Sequential(block(3, width), block(width, 2 * width), nn.Flatten())
        self.fc = nn.Linear(2 * width * 7 * 7, 128)
        self.drop = nn.Dropout(p_fc)
        self.head = nn.Linear(128, n_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(self.body((x - 0.5) / 0.25)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.features(x)))


def set_mc_dropout(model: nn.Module, on: bool) -> None:
    """Keep dropout layers stochastic at test time (MC-dropout) while everything else stays in eval mode."""
    model.eval()
    if on:
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
