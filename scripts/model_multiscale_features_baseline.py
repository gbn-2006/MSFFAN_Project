
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# Pixel-Unshuffle
# ================================================================

class PixelUnshuffleBlock(nn.Module):

    def __init__(self, in_channels=3, scale=2):
        super().__init__()

        if scale != 2:
            raise ValueError(
                "This implementation uses Pixel-Unshuffle scale 2."
            )

        self.in_channels = in_channels
        self.scale = scale

    def forward(self, x):

        if x.ndim != 4:
            raise ValueError(
                f"Expected [B,C,H,W], got {tuple(x.shape)}"
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, "
                f"got {x.shape[1]}"
            )

        return F.pixel_unshuffle(x, self.scale)


# ================================================================
# Residual Dense Block
# ================================================================

class DenseResidualBlock(nn.Module):

    def __init__(
        self,
        channels,
        growth_channels=32,
        residual_scale=0.2
    ):
        super().__init__()

        self.residual_scale = residual_scale

        self.conv1 = nn.Conv2d(
            channels,
            growth_channels,
            3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            channels + growth_channels,
            growth_channels,
            3,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            channels + 2 * growth_channels,
            growth_channels,
            3,
            padding=1
        )

        self.conv4 = nn.Conv2d(
            channels + 3 * growth_channels,
            channels,
            3,
            padding=1
        )

        self.activation = nn.LeakyReLU(
            0.2,
            inplace=True
        )

    def forward(self, x):

        x1 = self.activation(
            self.conv1(x)
        )

        x2 = self.activation(
            self.conv2(
                torch.cat([x, x1], dim=1)
            )
        )

        x3 = self.activation(
            self.conv3(
                torch.cat([x, x1, x2], dim=1)
            )
        )

        x4 = self.conv4(
            torch.cat([x, x1, x2, x3], dim=1)
        )

        return x + self.residual_scale * x4


# ================================================================
# RRDB
# ================================================================

class RRDB(nn.Module):

    def __init__(
        self,
        channels,
        growth_channels=32,
        residual_scale=0.2
    ):
        super().__init__()

        self.rdb1 = DenseResidualBlock(
            channels,
            growth_channels,
            residual_scale
        )

        self.rdb2 = DenseResidualBlock(
            channels,
            growth_channels,
            residual_scale
        )

        self.rdb3 = DenseResidualBlock(
            channels,
            growth_channels,
            residual_scale
        )

        self.residual_scale = residual_scale

    def forward(self, x):

        identity = x

        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)

        return identity + self.residual_scale * (out - identity)


# ================================================================
# Multi-Scale Feature Extraction
# ================================================================

class MultiScaleFeatureExtraction(nn.Module):

    def __init__(
        self,
        in_channels=3,
        feature_channels=64,
        growth_channels=32
    ):
        super().__init__()

        # Pixel-Unshuffle:
        # [B,3,256,256] -> [B,12,128,128]

        self.pixel_unshuffle = PixelUnshuffleBlock(
            in_channels=in_channels,
            scale=2
        )

        # Project 12 channels -> 64 feature channels

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

        # RRDB feature extraction

        self.rrdb = RRDB(
            channels=feature_channels,
            growth_channels=growth_channels
        )

        # Multi-scale branch 1
        # Local 3x3 receptive field

        self.branch_3x3 = nn.Conv2d(
            feature_channels,
            feature_channels,
            kernel_size=3,
            padding=1
        )

        # Multi-scale branch 2
        # Larger 5x5 receptive field

        self.branch_5x5 = nn.Conv2d(
            feature_channels,
            feature_channels,
            kernel_size=5,
            padding=2
        )

        # Fuse both scales

        self.fusion = nn.Conv2d(
            feature_channels * 2,
            feature_channels,
            kernel_size=1
        )

    def forward(self, x):

        # Pixel-Unshuffle

        x = self.pixel_unshuffle(x)

        # Feature projection

        x = self.activation(
            self.feature_projection(x)
        )

        # RRDB

        x = self.rrdb(x)

        # Multi-scale branches

        feature_3x3 = self.activation(
            self.branch_3x3(x)
        )

        feature_5x5 = self.activation(
            self.branch_5x5(x)
        )

        # Concatenate multi-scale features

        fused = torch.cat(
            [feature_3x3, feature_5x5],
            dim=1
        )

        # Fuse back to 64 channels

        fused = self.fusion(fused)

        return fused


# ================================================================
# Verification
# ================================================================

def test_multiscale():

    print("=" * 70)
    print("MSFFAN PROJECT - STEP 4")
    print("MULTI-SCALE FEATURE EXTRACTION VERIFICATION")
    print("=" * 70)

    torch.manual_seed(42)

    # ------------------------------------------------------------
    # 1. Input
    # ------------------------------------------------------------

    print("\n[1] INPUT")

    x = torch.rand(
        4,
        3,
        256,
        256,
        dtype=torch.float32
    )

    print(f"Input shape : {tuple(x.shape)}")
    print(f"Input dtype : {x.dtype}")

    # ------------------------------------------------------------
    # 2. Model
    # ------------------------------------------------------------

    print("\n[2] MODEL CONFIGURATION")

    model = MultiScaleFeatureExtraction(
        in_channels=3,
        feature_channels=64,
        growth_channels=32
    )

    print("Pixel-Unshuffle : ×2")
    print("Feature channels: 64")
    print("RRDB            : enabled")
    print("Branch 1        : 3×3")
    print("Branch 2        : 5×5")
    print("Fusion          : 1×1 convolution")

    # ------------------------------------------------------------
    # 3. Forward pass
    # ------------------------------------------------------------

    print("\n[3] FORWARD PASS")

    output = model(x)

    print(f"Output shape: {tuple(output.shape)}")

    expected_shape = (
        4,
        64,
        128,
        128
    )

    print(f"Expected    : {expected_shape}")

    assert tuple(output.shape) == expected_shape

    print("Output shape: PASSED")

    # ------------------------------------------------------------
    # 4. Numerical validity
    # ------------------------------------------------------------

    print("\n[4] NUMERICAL VALIDITY")

    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()

    print("Dtype  : PASSED")
    print("Finite : PASSED")

    print(
        f"Minimum : {output.min().item():.6f}"
    )

    print(
        f"Maximum : {output.max().item():.6f}"
    )

    print(
        f"Mean    : {output.mean().item():.6f}"
    )

    print(
        f"Std     : {output.std().item():.6f}"
    )

    # ------------------------------------------------------------
    # 5. Gradient test
    # ------------------------------------------------------------

    print("\n[5] GRADIENT FLOW")

    x_grad = torch.rand(
        2,
        3,
        256,
        256,
        requires_grad=True
    )

    model_grad = MultiScaleFeatureExtraction()

    y = model_grad(x_grad)

    loss = y.mean()

    loss.backward()

    assert x_grad.grad is not None
    assert torch.isfinite(x_grad.grad).all()

    print("Forward pass : PASSED")
    print("Backward pass: PASSED")
    print("Input gradient: PASSED")

    # ------------------------------------------------------------
    # 6. Parameter count
    # ------------------------------------------------------------

    print("\n[6] PARAMETER COUNT")

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Trainable parameters: {parameters:,}")

    assert parameters > 0

    print("Parameter check: PASSED")

    # ------------------------------------------------------------
    # Final
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 4 MULTI-SCALE FEATURES: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_multiscale()
