
import torch
import torch.nn as nn


class PixelAttention(nn.Module):
    """
    Pixel / spatial attention module.

    Input:
        [B, C, H, W]

    Output:
        [B, C, H, W]

    Implementation choice:
        A lightweight 3x3 convolution produces one spatial
        attention map, followed by sigmoid normalization.
    """

    def __init__(self, channels=64):
        super().__init__()

        if channels <= 0:
            raise ValueError("channels must be positive")

        self.attention = nn.Sequential(
            nn.Conv2d(
                channels,
                1,
                kernel_size=3,
                padding=1,
                bias=True
            ),
            nn.Sigmoid()
        )

    def forward(self, x):

        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D input [B,C,H,W], "
                f"got {tuple(x.shape)}"
            )

        attention_map = self.attention(x)

        return x * attention_map


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def main():

    print("=" * 60)
    print("STEP 6 - PIXEL ATTENTION")
    print("=" * 60)

    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # ---------------------------------------------------------
    # Test input from Step 5
    # ---------------------------------------------------------

    x = torch.randn(
        4,
        64,
        128,
        128,
        dtype=torch.float32,
        device=device,
        requires_grad=True
    )

    print(f"Input shape: {tuple(x.shape)}")

    # ---------------------------------------------------------
    # Create module
    # ---------------------------------------------------------

    model = PixelAttention(
        channels=64
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
    # Inspect attention map
    # ---------------------------------------------------------

    with torch.no_grad():

        attention_map = model.attention(x.detach())

        print(
            f"Attention map shape: "
            f"{tuple(attention_map.shape)}"
        )

        print(
            f"Attention map min: "
            f"{attention_map.min().item():.6f}"
        )

        print(
            f"Attention map max: "
            f"{attention_map.max().item():.6f}"
        )

        print(
            f"Attention map mean: "
            f"{attention_map.mean().item():.6f}"
        )

        assert attention_map.shape == (
            4, 1, 128, 128
        )

        assert torch.isfinite(
            attention_map
        ).all()

        assert attention_map.min() >= 0
        assert attention_map.max() <= 1

    print("Attention-map check: PASSED")

    # ---------------------------------------------------------
    # Backward / gradient check
    # ---------------------------------------------------------

    loss = y.mean()

    loss.backward()

    assert x.grad is not None

    assert torch.isfinite(
        x.grad
    ).all()

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
    print("PIXEL ATTENTION: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
