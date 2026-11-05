"""
Sign Language MNIST Benchmark CNN.

A compact multi-layer Conv2D CNN for classifying 24 static ASL hand sign classes
from 1x28x28 grayscale images.

Architecture:
  Block 1: Conv2d(1,32,3,pad=1) -> BN -> ReLU -> MaxPool(2)   => 32x14x14
  Block 2: Conv2d(32,64,3,pad=1) -> BN -> ReLU -> MaxPool(2)  => 64x7x7
  Block 3: Conv2d(64,128,3,pad=1) -> BN -> ReLU -> MaxPool(2) => 128x3x3
  Flatten -> Linear(128*3*3, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 24)

Output: 24 raw logits (no softmax — designed for CrossEntropyLoss).
"""

import torch
import torch.nn as nn


class SignLanguageCNN(nn.Module):
    """Compact CNN for Sign Language MNIST classification (24 classes)."""

    NUM_CLASSES = 24

    def __init__(self):
        super().__init__()

        # --- Convolutional blocks ---
        self.features = nn.Sequential(
            # Block 1: 1x28x28 -> 32x14x14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 2: 32x14x14 -> 64x7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 3: 64x7x7 -> 128x3x3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # Calculate flattened size dynamically
        self._flat_size = self._get_flat_size()

        # --- Classifier head ---
        self.classifier = nn.Sequential(
            nn.Linear(self._flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, self.NUM_CLASSES),
        )

    def _get_flat_size(self) -> int:
        """Run a dummy tensor through the conv blocks to determine the flattened size."""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)
            out = self.features(dummy)
            return out.view(1, -1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x  # raw logits — no softmax


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
