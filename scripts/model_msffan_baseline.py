import sys
from pathlib import Path

import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model_multiscale_features_baseline import MultiScaleFeatureExtraction
from model_channel_attention import ChannelAttention
from model_pixel_attention import PixelAttention
from model_reconstruction import ReconstructionHead


class MSFFANBaselineFeatureAttention(nn.Module):

    def __init__(
        self,
        in_channels=3,
        feature_channels=64,
        pixel_unshuffle_scale=2,
        attention_reduction=16
    ):
        super().__init__()

        self.feature_extractor = MultiScaleFeatureExtraction(
            in_channels=in_channels,
            feature_channels=feature_channels
        )

        self.channel_attention = ChannelAttention(
            channels=feature_channels,
            reduction=attention_reduction
        )

        self.pixel_attention = PixelAttention(
            channels=feature_channels
        )

    def forward(self, x):

        if x.ndim != 4:
            raise ValueError(
                f"Expected [B,C,H,W], got {tuple(x.shape)}"
            )

        features = self.feature_extractor(x)
        features = self.channel_attention(features)
        features = self.pixel_attention(features)

        return features


class MSFFANBaselineModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.feature_attention = MSFFANBaselineFeatureAttention(
            in_channels=3,
            feature_channels=64,
            pixel_unshuffle_scale=2,
            attention_reduction=16
        )

        self.reconstruction = ReconstructionHead(
            feature_channels=64,
            out_channels=3,
            upscale_factor=2
        )

    def forward(self, x):

        if x.ndim != 4:
            raise ValueError(
                f"Expected [B,C,H,W], got {tuple(x.shape)}"
            )

        features = self.feature_attention(x)
        output = self.reconstruction(features)

        return output


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
