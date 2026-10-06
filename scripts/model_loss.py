
import torch
import torch.nn as nn
import torch.nn.functional as F


class LaplacianEdgeLoss(nn.Module):
    """
    Laplacian edge-aware loss.

    Computes the absolute difference between the
    Laplacian responses of prediction and target.

    Input:
        prediction: [B, C, H, W]
        target:     [B, C, H, W]

    Output:
        scalar edge loss
    """

    def __init__(self):

        super().__init__()

        kernel = torch.tensor(
            [
                [0.0,  1.0, 0.0],
                [1.0, -4.0, 1.0],
                [0.0,  1.0, 0.0]
            ],
            dtype=torch.float32
        )

        self.register_buffer(
            "kernel",
            kernel.view(1, 1, 3, 3)
        )

    def forward(self, prediction, target):

        if prediction.ndim != 4:
            raise ValueError(
                "Prediction must have shape [B,C,H,W]"
            )

        if target.ndim != 4:
            raise ValueError(
                "Target must have shape [B,C,H,W]"
            )

        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: "
                f"{prediction.shape} vs {target.shape}"
            )

        channels = prediction.shape[1]

        kernel = self.kernel.repeat(
            channels, 1, 1, 1
        )

        pred_edges = F.conv2d(
            prediction,
            kernel,
            padding=1,
            groups=channels
        )

        target_edges = F.conv2d(
            target,
            kernel,
            padding=1,
            groups=channels
        )

        return F.l1_loss(
            pred_edges,
            target_edges
        )


class MSFFANLoss(nn.Module):
    """
    Combined reconstruction + Laplacian edge loss.

    Total loss:

        L_total = L_pixel + lambda_edge * L_edge

    The weighting is an implementation choice.
    """

    def __init__(self, lambda_edge=0.1):

        super().__init__()

        self.pixel_loss = nn.L1Loss()

        self.edge_loss = LaplacianEdgeLoss()

        self.lambda_edge = lambda_edge

    def forward(self, prediction, target):

        pixel = self.pixel_loss(
            prediction,
            target
        )

        edge = self.edge_loss(
            prediction,
            target
        )

        total = (
            pixel
            + self.lambda_edge * edge
        )

        return total, pixel, edge


def main():

    print("=" * 70)
    print("STEP 10 - LAPLACIAN EDGE-AWARE LOSS")
    print("=" * 70)

    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    # ---------------------------------------------------------
    # Test tensors
    # ---------------------------------------------------------

    prediction = torch.rand(
        4,
        3,
        256,
        256,
        device=device,
        requires_grad=True
    )

    target = torch.rand(
        4,
        3,
        256,
        256,
        device=device
    )

    print(
        f"Prediction shape: "
        f"{tuple(prediction.shape)}"
    )

    print(
        f"Target shape: "
        f"{tuple(target.shape)}"
    )

    # ---------------------------------------------------------
    # Edge loss
    # ---------------------------------------------------------

    edge_loss_fn = LaplacianEdgeLoss().to(device)

    edge_loss = edge_loss_fn(
        prediction,
        target
    )

    print(
        f"Laplacian edge loss: "
        f"{edge_loss.item():.6f}"
    )

    assert edge_loss.ndim == 0
    assert torch.isfinite(edge_loss)

    print("Edge-loss scalar/finite check: PASSED")

    # ---------------------------------------------------------
    # Combined loss
    # ---------------------------------------------------------

    loss_fn = MSFFANLoss(
        lambda_edge=0.1
    ).to(device)

    total, pixel, edge = loss_fn(
        prediction,
        target
    )

    print(
        f"Pixel L1 loss: "
        f"{pixel.item():.6f}"
    )

    print(
        f"Edge loss: "
        f"{edge.item():.6f}"
    )

    print(
        f"Total loss: "
        f"{total.item():.6f}"
    )

    assert torch.isfinite(total)
    assert torch.isfinite(pixel)
    assert torch.isfinite(edge)

    print("Combined-loss finite check: PASSED")

    # ---------------------------------------------------------
    # Backward
    # ---------------------------------------------------------

    total.backward()

    assert prediction.grad is not None

    assert torch.isfinite(
        prediction.grad
    ).all()

    print("Loss backward check: PASSED")

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print("=" * 70)
    print("LAPLACIAN EDGE-AWARE LOSS: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
