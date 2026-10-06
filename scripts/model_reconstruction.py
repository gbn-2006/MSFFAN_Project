
import torch
import torch.nn as nn


class ReconstructionHead(nn.Module):
    """
    Reconstruction / output head.

    Input:
        [B, 64, 128, 128]

    Output:
        [B, 3, 256, 256]

    Implementation choice:
        Feature refinement followed by Pixel Shuffle x2
        and RGB reconstruction.
    """

    def __init__(
        self,
        feature_channels=64,
        out_channels=3,
        upscale_factor=2
    ):
        super().__init__()

        if feature_channels <= 0:
            raise ValueError(
                "feature_channels must be positive"
            )

        if out_channels != 3:
            raise ValueError(
                "This implementation expects RGB output."
            )

        if upscale_factor != 2:
            raise ValueError(
                "This implementation uses a fixed x2 "
                "Pixel Shuffle."
            )

        self.refine = nn.Sequential(
            nn.Conv2d(
                feature_channels,
                feature_channels,
                kernel_size=3,
                padding=1
            ),
            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True
            ),
            nn.Conv2d(
                feature_channels,
                out_channels * upscale_factor ** 2,
                kernel_size=3,
                padding=1
            )
        )

        self.pixel_shuffle = nn.PixelShuffle(
            upscale_factor
        )

    def forward(self, x):

        if x.ndim != 4:
            raise ValueError(
                f"Expected [B,C,H,W], "
                f"got {tuple(x.shape)}"
            )

        features = self.refine(x)

        output = self.pixel_shuffle(features)

        return output


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def main():

    print("=" * 65)
    print("STEP 8 - RECONSTRUCTION / OUTPUT HEAD")
    print("=" * 65)

    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    # ---------------------------------------------------------
    # Feature representation from Step 7
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

    print(f"Input feature shape: {tuple(x.shape)}")

    # ---------------------------------------------------------
    # Create reconstruction head
    # ---------------------------------------------------------

    model = ReconstructionHead().to(device)

    print(
        f"Trainable parameters: "
        f"{count_parameters(model):,}"
    )

    # ---------------------------------------------------------
    # Forward
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

    print("Output shape check: PASSED")

    # ---------------------------------------------------------
    # Numerical stability
    # ---------------------------------------------------------

    assert torch.isfinite(y).all()

    print("Finite-value check: PASSED")

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

    assert torch.isfinite(
        x.grad
    ).all()

    print("Backward/input-gradient check: PASSED")

    # ---------------------------------------------------------
    # Parameter-gradient check
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
    print("RECONSTRUCTION HEAD: PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()
