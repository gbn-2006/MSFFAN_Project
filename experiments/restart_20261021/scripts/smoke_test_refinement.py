import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path("/mnt/c/MSFFAN_Project")
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "restart_20261021"
SCRIPT_DIR = PROJECT_ROOT / "scripts"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(EXPERIMENT_ROOT / "models"))

from model_msffan_baseline import MSFFANBaselineModel
from model_refinement import InputConditionedResidualRefinement


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    baseline = MSFFANBaselineModel().to(device).eval()
    refinement = InputConditionedResidualRefinement().to(device).eval()

    lr = torch.rand(1, 3, 256, 256, device=device)

    with torch.no_grad():
        sr = baseline(lr)
        refined = refinement(lr, sr)

    print("LR shape:", tuple(lr.shape))
    print("Baseline output shape:", tuple(sr.shape))
    print("Refined output shape:", tuple(refined.shape))
    print("Baseline parameters:", f"{count_parameters(baseline):,}")
    print("Refinement parameters:", f"{count_parameters(refinement):,}")
    print("Max initial output difference:", (refined - sr).abs().max().item())

    assert sr.shape == (1, 3, 256, 256)
    assert refined.shape == sr.shape
    assert torch.isfinite(refined).all()
    assert torch.allclose(refined, sr, atol=0, rtol=0)

    # Check trainable gradients through the refinement module.
    refinement.train()
    test_sr = baseline(lr).detach()
    output = refinement(lr, test_sr)
    loss = output.mean()
    loss.backward()

    assert all(p.grad is not None for p in refinement.parameters())
    assert all(torch.isfinite(p.grad).all() for p in refinement.parameters())

    print("Shape check: PASSED")
    print("Finite-value check: PASSED")
    print("Identity initialization check: PASSED")
    print("Gradient check: PASSED")


if __name__ == "__main__":
    main()
