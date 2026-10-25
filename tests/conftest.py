"""Shared fixtures: a tiny CNN and a random input, so that tests run on CPU in seconds."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    """Small 4-class CNN with the same input size (224 x 224) as the thesis models."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


@pytest.fixture(scope="session")
def tiny_model() -> nn.Module:
    torch.manual_seed(0)
    return TinyCNN().eval()


@pytest.fixture()
def image() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(1, 3, 224, 224)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(7)
