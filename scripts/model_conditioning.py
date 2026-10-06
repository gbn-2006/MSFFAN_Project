
import torch
import torch.nn as nn


class ResolutionPositionConditioning(nn.Module):
    """
    Resolution/position-aware conditioning module.

    Input:
        [B, 3]
        3 values:
          1. normalized scale factor
          2. normalized x position
          3. normalized y position

    Output:
        [B, conditioning_dim]
    """

    def __init__(self, conditioning_dim=64, hidden_dim=32):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, conditioning_dim)
        )

    def forward(self, condition):
        if condition.ndim != 2:
            raise ValueError(
                f"Expected condition shape [B, 3], got {condition.shape}"
            )

        if condition.shape[1] != 3:
            raise ValueError(
                f"Expected 3 conditioning values, got {condition.shape[1]}"
            )

        return self.mlp(condition)
