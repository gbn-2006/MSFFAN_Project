
import sys
from pathlib import Path

import torch
import torch.nn as nn


# -------------------------------------------------------------
# Make the project scripts directory importable
# -------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


from model_multiscale_features import MultiScaleFeatureExtraction
from model_channel_attention import ChannelAttention
from model_pixel_attention import PixelAttention


class MSFFANFeatureAttention(nn.Module):
    """
    Integrated feature-extraction and dual-attention pipeline.

    Pipeline:

        Input RGB image
            ↓
        Pixel-Unshuffle ×2
            ↓
        Feature projection
            ↓
        RRDB
            ↓
        Multi-scale feature extraction
            ↓
        Channel attention
            ↓
        Pixel attention
            ↓
        Feature representation

    Input:
        [B, 3, 256, 256]

    Output:
        [B, 64, 128, 128]

    Note:
        This is an implementation-level integration module.
        Exact architectural hyperparameters are not specified
        in the project PPT.
    """

    def __init__(
        self,
        in_channels=3,
        feature_channels=64,
        pixel_unshuffle_scale=2,
        attention_reduction=16
    ):
        super().__init__()

        if in_channels != 3:
            raise ValueError(
                "This implementation expects RGB input with 3 channels."
            )

        if pixel_unshuffle_scale != 2:
            raise ValueError(
                "This implementation uses a fixed Pixel-Unshuffle scale of 2."
            )

        # Multi-scale feature extractor already contains:
        # Pixel-Unshuffle + feature projection + RRDB +
        # multi-scale branches + fusion.
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


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def main():

    print("=" * 65)
    print("STEP 7 - INTEGRATED FEATURE + DUAL ATTENTION")
    print("=" * 65)

    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    # ---------------------------------------------------------
    # Test input
    # ---------------------------------------------------------

    x = torch.randn(
        4,
        3,
        256,
        256,
        dtype=torch.float32,
        device=device,
        requires_grad=True
    )

    print(f"Input shape: {tuple(x.shape)}")

    # ---------------------------------------------------------
    # Create model
    # ---------------------------------------------------------

    model = MSFFANFeatureAttention().to(device)

    print(
        f"Total trainable parameters: "
        f"{count_parameters(model):,}"
    )

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------

    y = model(x)

    print(f"Output shape: {tuple(y.shape)}")
    print(f"Output dtype: {y.dtype}")

    # ---------------------------------------------------------
    # Shape verification
    # ---------------------------------------------------------

    expected_shape = (
        4,
        64,
        128,
        128
    )

    assert y.shape == expected_shape

    print("Integrated shape check: PASSED")

    # ---------------------------------------------------------
    # Numerical stability
    # ---------------------------------------------------------

    assert torch.isfinite(y).all()

    print("Integrated finite-value check: PASSED")

    # ---------------------------------------------------------
    # Output statistics
    # ---------------------------------------------------------

    print(
        f"Output min: {y.min().item():.6f}"
    )

    print(
        f"Output max: {y.max().item():.6f}"
    )

    print(
        f"Output mean: {y.mean().item():.6f}"
    )

    print(
        f"Output std: {y.std().item():.6f}"
    )

    # ---------------------------------------------------------
    # Backward test
    # ---------------------------------------------------------

    loss = y.mean()

    print(
        f"Test loss: {loss.item():.6f}"
    )

    loss.backward()

    assert x.grad is not None

    assert torch.isfinite(x.grad).all()

    print("Integrated backward check: PASSED")

    # ---------------------------------------------------------
    # Parameter-gradient verification
    # ---------------------------------------------------------

    parameters_with_grad = 0
    parameters_without_grad = 0

    for parameter in model.parameters():

        if not parameter.requires_grad:
            continue

        if parameter.grad is None:
            parameters_without_grad += 1
        else:
            parameters_with_grad += 1

    print(
        f"Parameters receiving gradients: "
        f"{parameters_with_grad}"
    )

    print(
        f"Parameters without gradients: "
        f"{parameters_without_grad}"
    )

    assert parameters_without_grad == 0

    print("Parameter-gradient check: PASSED")

    print("=" * 65)
    print("INTEGRATED FEATURE + DUAL ATTENTION: PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()
