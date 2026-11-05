import torch
import torch.nn as nn

class ASLLandmarkMLP(nn.Module):
    """
    Multi-Layer Perceptron for 28-Class ASL Gesture Recognition based on 63-dimensional
    hand landmark features.

    Architecture:
    Input: 63
        |
    Linear(63, 128)
        |
    BatchNorm1d(128)
        |
    ReLU
        |
    Dropout(0.20)
        |
    Linear(128, 64)
        |
    ReLU
        |
    Dropout(0.20)
        |
    Linear(64, 28)
        |
    Raw logits
    """

    def __init__(self):
        super(ASLLandmarkMLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(63, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.20),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.20),

            nn.Linear(64, 28)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 63)

        Returns:
            Output tensor of shape (batch_size, 28) containing raw logits.
        """
        return self.net(x)
