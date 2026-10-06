import sys
from pathlib import Path

import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


from model_multiscale_features import (
    PixelUnshuffleBlock,
    RRDB
)


class HFResidualBranch(nn.Module):
    """
    Learned high-frequency-sensitive residual feature branch.

    Input:
        [B, 64, 128, 128]

    Output:
        [B, 64, 128, 128]

    This branch learns a residual feature representation
    intended to complement the existing multi-scale pathway.
    """

    def __init__(self, channels=64):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.activation = nn.LeakyReLU(
            0.2,
            inplace=True
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        # Start with zero HF contribution while keeping the branch trainable.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):

        residual = self.conv2(
            self.activation(
                self.conv1(x)
            )
        )

        return residual


class MSFFANHighFrequencyFeatures(nn.Module):
    """
    MSFFAN feature extraction with an additional
    learned high-frequency residual pathway.

    Main pathway:
        Pixel-Unshuffle
        -> Feature Projection
        -> RRDB
        -> 3x3 / 5x5 Multi-Scale Fusion

    Additional pathway:
        RRDB output
        -> HF Residual Branch

    Fusion:
        Main Features + HF Residual Features
    """

    def __init__(
        self,
        in_channels=3,
        feature_channels=64,
        growth_channels=32
    ):
        super().__init__()

        self.pixel_unshuffle = PixelUnshuffleBlock(
            in_channels=in_channels,
            scale=2
        )

        self.feature_projection = nn.Conv2d(
            in_channels * 4,
            feature_channels,
            kernel_size=3,
            padding=1
        )

        self.activation = nn.LeakyReLU(
            0.2,
            inplace=True
        )

        self.rrdb = RRDB(
            channels=feature_channels,
            growth_channels=growth_channels
        )

        self.branch_3x3 = nn.Conv2d(
            feature_channels,
            feature_channels,
            kernel_size=3,
            padding=1
        )

        self.branch_5x5 = nn.Conv2d(
            feature_channels,
            feature_channels,
            kernel_size=5,
            padding=2
        )

        self.fusion = nn.Conv2d(
            feature_channels * 2,
            feature_channels,
            kernel_size=1
        )

        self.hf_branch = HFResidualBranch(
            channels=feature_channels
        )

    def forward(self, x):

        x = self.pixel_unshuffle(x)

        x = self.activation(
            self.feature_projection(x)
        )

        x = self.rrdb(x)

        feature_3x3 = self.activation(
            self.branch_3x3(x)
        )

        feature_5x5 = self.activation(
            self.branch_5x5(x)
        )

        fused = torch.cat(
            [feature_3x3, feature_5x5],
            dim=1
        )

        main_features = self.fusion(fused)

        hf_features = self.hf_branch(x)

        features = main_features + hf_features

        return features
