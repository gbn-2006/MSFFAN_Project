
import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelUnshuffleBlock(nn.Module):
    """
    Fixed Pixel-Unshuffle preprocessing block.

    The MSFFAN implementation uses a fixed network factor of 2.
    Dataset x2/x3/x4 values describe degradation conditions and
    are not used as the Pixel-Unshuffle factor.
    """

    def __init__(self, in_channels=3, scale=2):
        super().__init__()

        if scale != 2:
            raise ValueError(
                "This MSFFAN implementation uses a fixed "
                "Pixel-Unshuffle scale of 2."
            )

        self.in_channels = in_channels
        self.scale = scale

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(
                f"Expected [B,C,H,W], got {tuple(x.shape)}"
            )

        b, c, h, w = x.shape

        if c != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {c}"
            )

        if h % self.scale != 0 or w % self.scale != 0:
            raise ValueError(
                f"Input dimensions must be divisible by {self.scale}. "
                f"Got H={h}, W={w}"
            )

        return F.pixel_unshuffle(x, self.scale)


def test_pixel_unshuffle():

    print("=" * 70)
    print("MSFFAN PROJECT - STEP 2")
    print("PIXEL-UNSHUFFLE VERIFICATION")
    print("=" * 70)

    torch.manual_seed(42)

    # ------------------------------------------------------------
    # 1. Dataset-shaped input
    # ------------------------------------------------------------

    print("\n[1] DATASET INPUT TEST")

    x = torch.rand(
        4, 3, 256, 256,
        dtype=torch.float32
    )

    print(f"Input shape : {tuple(x.shape)}")
    print(f"Input dtype : {x.dtype}")

    block = PixelUnshuffleBlock(
        in_channels=3,
        scale=2
    )

    y = block(x)

    expected = (4, 12, 128, 128)

    print(f"Output shape: {tuple(y.shape)}")
    print(f"Expected    : {expected}")

    assert tuple(y.shape) == expected
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all()

    print("Shape check : PASSED")
    print("Dtype check : PASSED")
    print("Finite check: PASSED")

    # ------------------------------------------------------------
    # 2. Channel calculation
    # ------------------------------------------------------------

    print("\n[2] CHANNEL CALCULATION")

    expected_channels = 3 * (2 ** 2)

    print(f"Input channels : 3")
    print(f"Unshuffle factor: 2")
    print(f"Expected output : 3 × 2² = {expected_channels}")
    print(f"Actual output   : {y.shape[1]}")

    assert y.shape[1] == expected_channels

    print("Channel calculation: PASSED")

    # ------------------------------------------------------------
    # 3. Spatial calculation
    # ------------------------------------------------------------

    print("\n[3] SPATIAL DIMENSION CALCULATION")

    print("256 / 2 = 128")

    assert y.shape[2] == 128
    assert y.shape[3] == 128

    print("Spatial calculation: PASSED")

    # ------------------------------------------------------------
    # 4. Information-preservation test
    # ------------------------------------------------------------

    print("\n[4] INFORMATION-PRESERVATION TEST")

    reconstructed = F.pixel_shuffle(y, 2)

    max_error = torch.max(
        torch.abs(x - reconstructed)
    ).item()

    mean_error = torch.mean(
        torch.abs(x - reconstructed)
    ).item()

    print(f"Maximum reconstruction error: {max_error:.10f}")
    print(f"Mean reconstruction error   : {mean_error:.10f}")

    assert max_error < 1e-6

    print("Pixel-Unshuffle → Pixel-Shuffle: PASSED")

    # ------------------------------------------------------------
    # 5. Model block test
    # ------------------------------------------------------------

    print("\n[5] MODEL BLOCK TEST")

    assert isinstance(block, nn.Module)

    y2 = block(x)

    assert torch.equal(y, y2)

    print("PyTorch module test: PASSED")

    # ------------------------------------------------------------
    # 6. Invalid scale protection
    # ------------------------------------------------------------

    print("\n[6] SCALE PROTECTION TEST")

    try:
        PixelUnshuffleBlock(
            in_channels=3,
            scale=3
        )
        raise AssertionError(
            "Invalid scale=3 was accepted."
        )

    except ValueError:
        print("Invalid scale rejection: PASSED")

    # ------------------------------------------------------------
    # Final
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 2 PIXEL-UNSHUFFLE: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_pixel_unshuffle()
