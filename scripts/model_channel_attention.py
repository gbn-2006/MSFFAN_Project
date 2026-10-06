
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel Attention module.

    Input:
        [B, C, H, W]

    Output:
        [B, C, H, W]

    Implementation choice:
        Global Average Pooling followed by a small bottleneck
        gating network and sigmoid channel weights.
    """

    def __init__(self, channels=64, reduction=16):
        super().__init__()

        if channels <= 0:
            raise ValueError("channels must be positive")

        if reduction <= 0:
            raise ValueError("reduction must be positive")

        hidden_channels = max(channels // reduction, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.gate = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=1,
                bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                channels,
                kernel_size=1,
                bias=True
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D input [B,C,H,W], got {tuple(x.shape)}"
            )

        channel_descriptor = self.pool(x)
        attention_weights = self.gate(channel_descriptor)

        return x * attention_weights


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def main():

    print("=" * 60)
    print("STEP 5 - CHANNEL ATTENTION")
    print("=" * 60)

    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # ---------------------------------------------------------
    # Test input from Step 4
    # ---------------------------------------------------------

    x = torch.randn(
        4, 64, 128, 128,
        dtype=torch.float32,
        device=device,
        requires_grad=True
    )

    print(f"Input shape: {tuple(x.shape)}")

    # ---------------------------------------------------------
    # Create module
    # ---------------------------------------------------------

    model = ChannelAttention(
        channels=64,
        reduction=16
    ).to(device)

    print(
        f"Trainable parameters: "
        f"{count_parameters(model):,}"
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

    assert y.shape == x.shape
    print("Shape check: PASSED")

    # ---------------------------------------------------------
    # Finite-value check
    # ---------------------------------------------------------

    assert torch.isfinite(y).all()
    print("Finite-value check: PASSED")

    # ---------------------------------------------------------
    # Reconstruct attention weights for verification
    # ---------------------------------------------------------

    with torch.no_grad():

        descriptor = model.pool(x.detach())
        weights = model.gate(descriptor)

        print(
            f"Attention weight shape: "
            f"{tuple(weights.shape)}"
        )

        print(
            f"Attention weight min: "
            f"{weights.min().item():.6f}"
        )

        print(
            f"Attention weight max: "
            f"{weights.max().item():.6f}"
        )

        print(
            f"Attention weight mean: "
            f"{weights.mean().item():.6f}"
        )

        assert weights.shape == (4, 64, 1, 1)
        assert torch.isfinite(weights).all()
        assert weights.min() >= 0
        assert weights.max() <= 1

    print("Attention-weight check: PASSED")

    # ---------------------------------------------------------
    # Gradient / backward check
    # ---------------------------------------------------------

    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    print("Backward/input-gradient check: PASSED")

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

    print("=" * 60)
    print("CHANNEL ATTENTION: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
