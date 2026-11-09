"""Model definitions and loading for the three architectures used in the thesis."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 4


class SimpleCNN(nn.Module):
    """Three convolution blocks + two fully connected layers (input 224 x 224)."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def build_model(name: str) -> nn.Module:
    """Return an untrained model with a 4-class head."""
    if name == "simplecnn":
        return SimpleCNN(NUM_CLASSES)
    if name == "mobilenet":
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, NUM_CLASSES)
        return m
    if name == "squeezenet":
        m = models.squeezenet1_1(weights=None)
        m.classifier[1] = nn.Conv2d(512, NUM_CLASSES, kernel_size=(1, 1), stride=(1, 1))
        m.num_classes = NUM_CLASSES
        return m
    raise ValueError(f"Unknown model '{name}'")


def gradcam_layer(model: nn.Module, name: str) -> nn.Module:
    """Layer used by Grad-CAM (last convolutional stage of each network).

    SimpleCNN: conv3 (56 x 56), MobileNet-V3-Small: features[-1] (7 x 7),
    SqueezeNet 1.1: last Fire module features[-1] (13 x 13).
    """
    if name == "simplecnn":
        return cast(nn.Module, model.conv3)
    features = cast(nn.Sequential, model.features)
    return cast(nn.Module, features[-1])


def load_model(name: str, path: str, device: torch.device) -> nn.Module:
    model = build_model(name)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()


MODEL_NAMES = ("simplecnn", "mobilenet", "squeezenet")
DISPLAY_NAME = {"simplecnn": "SimpleCNN", "mobilenet": "MobileNet-V3-Small", "squeezenet": "SqueezeNet 1.1"}
