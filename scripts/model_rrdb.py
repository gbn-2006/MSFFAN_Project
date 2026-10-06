
import torch
import torch.nn as nn


class DenseResidualBlock(nn.Module):
    """
    Compact Residual Dense Block.

    Each convolution receives the outputs of all previous
    convolutions through concatenation.
    """

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
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            channels + growth_channels,
            growth_channels,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            channels + 2 * growth_channels,
            growth_channels,
            kernel_size=3,
            padding=1
        )

        self.conv4 = nn.Conv2d(
            channels + 3 * growth_channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.activation = nn.LeakyReLU(
            negative_slope=0.2,
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


class RRDB(nn.Module):
    """
    Residual-in-Residual Dense Block.

    Three Residual Dense Blocks are nested inside
    an outer residual connection.
    """

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

        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)

        return x + self.residual_scale * (out - x)


def test_rrdb():

    print("=" * 70)
    print("MSFFAN PROJECT - STEP 3")
    print("RRDB VERIFICATION")
    print("=" * 70)

    torch.manual_seed(42)

    # ------------------------------------------------------------
    # 1. Configuration
    # ------------------------------------------------------------

    batch_size = 4
    channels = 64
    height = 128
    width = 128
    growth_channels = 32

    print("\n[1] CONFIGURATION")
    print(f"Batch size       : {batch_size}")
    print(f"Feature channels : {channels}")
    print(f"Height           : {height}")
    print(f"Width            : {width}")
    print(f"Growth channels  : {growth_channels}")

    # ------------------------------------------------------------
    # 2. Input
    # ------------------------------------------------------------

    x = torch.randn(
        batch_size,
        channels,
        height,
        width,
        dtype=torch.float32
    )

    print("\n[2] INPUT")
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Input dtype: {x.dtype}")

    # ------------------------------------------------------------
    # 3. RDB test
    # ------------------------------------------------------------

    print("\n[3] RESIDUAL DENSE BLOCK TEST")

    rdb = DenseResidualBlock(
        channels=channels,
        growth_channels=growth_channels
    )

    rdb_output = rdb(x)

    print(f"RDB output shape: {tuple(rdb_output.shape)}")

    assert rdb_output.shape == x.shape
    assert rdb_output.dtype == x.dtype
    assert torch.isfinite(rdb_output).all()

    print("Shape check : PASSED")
    print("Dtype check : PASSED")
    print("Finite check: PASSED")

    # ------------------------------------------------------------
    # 4. RRDB test
    # ------------------------------------------------------------

    print("\n[4] RRDB TEST")

    rrdb = RRDB(
        channels=channels,
        growth_channels=growth_channels
    )

    output = rrdb(x)

    print(f"RRDB output shape: {tuple(output.shape)}")

    assert output.shape == x.shape
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()

    print("Shape preservation: PASSED")
    print("Dtype check       : PASSED")
    print("Finite check      : PASSED")

    # ------------------------------------------------------------
    # 5. Gradient test
    # ------------------------------------------------------------

    print("\n[5] GRADIENT FLOW TEST")

    x_grad = torch.randn(
        2,
        channels,
        height,
        width,
        requires_grad=True
    )

    rrdb_grad = RRDB(
        channels=channels,
        growth_channels=growth_channels
    )

    y = rrdb_grad(x_grad)

    loss = y.mean()
    loss.backward()

    assert x_grad.grad is not None
    assert torch.isfinite(x_grad.grad).all()

    print("Backward pass : PASSED")
    print("Input gradient: PASSED")

    # ------------------------------------------------------------
    # 6. Parameter test
    # ------------------------------------------------------------

    print("\n[6] PARAMETER TEST")

    parameter_count = sum(
        p.numel()
        for p in rrdb.parameters()
        if p.requires_grad
    )

    print(f"Trainable parameters: {parameter_count:,}")

    assert parameter_count > 0

    print("Parameter check: PASSED")

    # ------------------------------------------------------------
    # 7. Batch independence / finite output
    # ------------------------------------------------------------

    print("\n[7] OUTPUT STATISTICS")

    print(f"Output minimum : {output.min().item():.6f}")
    print(f"Output maximum : {output.max().item():.6f}")
    print(f"Output mean    : {output.mean().item():.6f}")
    print(f"Output std     : {output.std().item():.6f}")

    assert torch.isfinite(output).all()

    print("Output statistics: PASSED")

    # ------------------------------------------------------------
    # Final
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 3 RRDB: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_rrdb()
