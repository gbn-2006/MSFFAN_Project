import torch
import torch.nn as nn


class AdaptiveScaleGatedFusion(nn.Module):
    """
    Adaptive Scale-Gated Fusion (ASGF)

    Learns spatially varying weights for two multi-scale
    feature branches.

    Input:
        feature_3x3 : [B, C, H, W]
        feature_5x5 : [B, C, H, W]

    Output:
        fused       : [B, C, H, W]

    The gate predicts two spatial weights:
        w_3x3(x,y)
        w_5x5(x,y)

    Softmax ensures:
        w_3x3 + w_5x5 = 1
    """

    def __init__(self, channels=64):

        super().__init__()

        self.gate = nn.Conv2d(
            channels * 2,
            2,
            kernel_size=1,
            padding=0
        )

    def forward(self, feature_3x3, feature_5x5):

        if feature_3x3.shape != feature_5x5.shape:
            raise ValueError(
                "ASGF requires both feature branches "
                "to have identical shapes. "
                f"Got {tuple(feature_3x3.shape)} and "
                f"{tuple(feature_5x5.shape)}"
            )

        combined = torch.cat(
            [feature_3x3, feature_5x5],
            dim=1
        )

        weights = torch.softmax(
            self.gate(combined),
            dim=1
        )

        weight_3x3 = weights[:, 0:1, :, :]
        weight_5x5 = weights[:, 1:2, :, :]

        fused = (
            weight_3x3 * feature_3x3
            + weight_5x5 * feature_5x5
        )

        return fused


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def main():

    print("=" * 70)
    print("ASGF - ADAPTIVE SCALE-GATED FUSION TEST")
    print("=" * 70)

    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    feature_3x3 = torch.randn(
        4,
        64,
        128,
        128,
        device=device,
        requires_grad=True
    )

    feature_5x5 = torch.randn(
        4,
        64,
        128,
        128,
        device=device,
        requires_grad=True
    )

    model = AdaptiveScaleGatedFusion(
        channels=64
    ).to(device)

    print(
        f"ASGF trainable parameters: "
        f"{count_parameters(model):,}"
    )

    output = model(
        feature_3x3,
        feature_5x5
    )

    print(
        f"Input 3x3 shape: {tuple(feature_3x3.shape)}"
    )

    print(
        f"Input 5x5 shape: {tuple(feature_5x5.shape)}"
    )

    print(
        f"Output shape: {tuple(output.shape)}"
    )

    assert output.shape == feature_3x3.shape

    assert torch.isfinite(output).all()

    print("Shape check: PASSED")
    print("Finite-value check: PASSED")

    loss = output.mean()

    loss.backward()

    assert feature_3x3.grad is not None
    assert feature_5x5.grad is not None

    print("Backward check: PASSED")

    print("=" * 70)
    print("ASGF TEST: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
