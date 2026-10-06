
import sys
from pathlib import Path

import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


from model_feature_attention_hf import MSFFANFeatureAttention
from model_reconstruction import ReconstructionHead


class MSFFANModel(nn.Module):
    """
    Complete MSFFAN forward model.

    Input:
        [B, 3, 256, 256]

    Output:
        [B, 3, 256, 256]

    Main pipeline:

        RGB input
            ↓
        Feature extraction
            ↓
        RRDB
            ↓
        Multi-scale features
            ↓
        Channel attention
            ↓
        Pixel attention
            ↓
        Reconstruction
            ↓
        RGB output

    Note:
        Architectural hyperparameters not explicitly specified
        in the project PPT are implementation choices.
    """

    def __init__(self):

        super().__init__()

        self.feature_attention = MSFFANFeatureAttention(
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


def main():

    print("=" * 70)
    print("STEP 9 - COMPLETE MSFFAN FORWARD MODEL")
    print("=" * 70)

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
    # Create complete model
    # ---------------------------------------------------------

    model = MSFFANModel().to(device)

    total_parameters = count_parameters(model)

    print(
        f"Total trainable parameters: "
        f"{total_parameters:,}"
    )

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------

    y = model(x)

    print(f"Output shape: {tuple(y.shape)}")
    print(f"Output dtype: {y.dtype}")

    # ---------------------------------------------------------
    # Shape check
    # ---------------------------------------------------------

    expected_shape = (
        4,
        3,
        256,
        256
    )

    assert y.shape == expected_shape

    print("Complete-model shape check: PASSED")

    # ---------------------------------------------------------
    # Numerical stability
    # ---------------------------------------------------------

    assert torch.isfinite(y).all()

    print("Complete-model finite-value check: PASSED")

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
        f"Integration test loss: "
        f"{loss.item():.6f}"
    )

    loss.backward()

    assert x.grad is not None

    assert torch.isfinite(
        x.grad
    ).all()

    print("Complete-model backward check: PASSED")

    # ---------------------------------------------------------
    # Parameter gradient verification
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

    print("Complete-model gradient check: PASSED")

    # ---------------------------------------------------------
    # Final status
    # ---------------------------------------------------------

    print("=" * 70)
    print("COMPLETE MSFFAN FORWARD MODEL: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
