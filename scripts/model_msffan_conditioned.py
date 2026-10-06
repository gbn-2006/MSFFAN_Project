
import torch
import torch.nn as nn

from model_feature_attention import MSFFANFeatureAttention
from model_reconstruction import ReconstructionHead
from model_conditioning import ResolutionPositionConditioning


class ConditionedFeatureFusion(nn.Module):
    """
    Resolution/position-aware FiLM-style conditioning.

    Features:
        [B, 64, H, W]

    Condition:
        [B, 3]
        [scale_norm, x_norm, y_norm]
    """

    def __init__(self, feature_channels=64):
        super().__init__()

        self.feature_channels = feature_channels

        self.conditioning = ResolutionPositionConditioning(
            conditioning_dim=feature_channels,
            hidden_dim=32
        )

        self.gamma_layer = nn.Linear(
            feature_channels,
            feature_channels
        )

        self.beta_layer = nn.Linear(
            feature_channels,
            feature_channels
        )

    def forward(self, features, condition):

        if features.ndim != 4:
            raise ValueError(
                f"Expected features [B,C,H,W], got {features.shape}"
            )

        if features.shape[1] != self.feature_channels:
            raise ValueError(
                f"Expected {self.feature_channels} feature channels, "
                f"got {features.shape[1]}"
            )

        if condition.ndim != 2 or condition.shape[1] != 3:
            raise ValueError(
                f"Expected condition [B,3], got {condition.shape}"
            )

        if features.shape[0] != condition.shape[0]:
            raise ValueError(
                "Feature and condition batch sizes do not match."
            )

        embedding = self.conditioning(condition)

        gamma = self.gamma_layer(embedding)
        beta = self.beta_layer(embedding)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        conditioned = (
            features * (1.0 + gamma) + beta
        )

        return conditioned


class MSFFANConditionedModel(nn.Module):
    """
    Proposed MSFFAN model.

    Baseline:
        MSFFAN Feature Attention
        +
        Reconstruction Head

    Proposed extension:
        Resolution/Position-Aware Conditioning
        between feature extraction and reconstruction.
    """

    def __init__(self):
        super().__init__()

        # Original MSFFAN feature extraction
        self.feature_attention = MSFFANFeatureAttention()

        # Proposed conditioning branch
        self.conditioning_fusion = ConditionedFeatureFusion(
            feature_channels=64
        )

        # Original reconstruction head
        self.reconstruction = ReconstructionHead()

    def forward(self, x, condition):

        features = self.feature_attention(x)

        conditioned_features = self.conditioning_fusion(
            features,
            condition
        )

        output = self.reconstruction(
            conditioned_features
        )

        return output
