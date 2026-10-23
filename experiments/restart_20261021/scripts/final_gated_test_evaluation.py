import sys
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity

ROOT = Path("/mnt/c/MSFFAN_Project")
EXP = ROOT / "experiments" / "restart_20261021"
RUN = EXP / "full_run_100_epochs"
OUT = RUN / "results" / "final_gated_test_alpha_025"
OUT.mkdir(parents=True, exist_ok=False)

BASELINE_PATH = ROOT / "models" / "baseline_best.pth"
REFINEMENT_PATH = RUN / "checkpoints" / "refinement_best.pth"
EXPECTED_HASH = (
    "4f1843da5ad95cc9f4759e84cfc71d824b7f05227c9b448f7c004205faaca21c"
)
ALPHA = 0.25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(EXP / "models"))

from dataset_loader import MSFFANDataset
from model_msffan_baseline import MSFFANBaselineModel
from model_refinement import InputConditionedResidualRefinement


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def calculate_metrics(prediction, target):
    prediction = prediction.detach().clamp(0, 1).cpu().numpy()
    target = target.detach().clamp(0, 1).cpu().numpy()
    psnr_values, ssim_values = [], []

    for pred, truth in zip(prediction, target):
        mse = float(np.mean((pred - truth) ** 2))
        psnr = float("inf") if mse == 0 else float(10 * np.log10(1 / mse))
        ssim = float(
            structural_similarity(
                truth, pred, channel_axis=0, data_range=1.0
            )
        )
        psnr_values.append(psnr)
        ssim_values.append(ssim)

    return psnr_values, ssim_values


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {OUT}")

    baseline_hash = sha256(BASELINE_PATH)
    if baseline_hash != EXPECTED_HASH:
        raise RuntimeError("Original baseline checkpoint hash mismatch.")

    baseline_ckpt = torch.load(
        BASELINE_PATH, map_location=DEVICE, weights_only=False
    )
    refinement_ckpt = torch.load(
        REFINEMENT_PATH, map_location=DEVICE, weights_only=False
    )

    if refinement_ckpt.get("baseline_sha256") != baseline_hash:
        raise RuntimeError("Refinement checkpoint references a different baseline.")

    baseline = MSFFANBaselineModel().to(DEVICE)
    baseline.load_state_dict(
        baseline_ckpt["model_state_dict"], strict=True
    )
    baseline.eval()
    for parameter in baseline.parameters():
        parameter.requires_grad_(False)

    refinement = InputConditionedResidualRefinement().to(DEVICE)
    refinement.load_state_dict(
        refinement_ckpt["model_state_dict"], strict=True
    )
    refinement.eval()

    dataset = MSFFANDataset(ROOT, split="test")
    if len(dataset) != 500:
        raise RuntimeError(
            f"Frozen test split must contain 500 samples; found {len(dataset)}."
        )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    rows = []
    with torch.inference_mode():
        for batch in loader:
            lr = batch["lr"].to(DEVICE)
            hr = batch["hr"].to(DEVICE)
            base = baseline(lr)
            full_refined = refinement(lr, base)

            # Locked inference configuration: 25% of the learned residual.
            prediction = (base + ALPHA * (full_refined - base)).clamp(0, 1)

            base_psnr, base_ssim = calculate_metrics(base, hr)
            gate_psnr, gate_ssim = calculate_metrics(prediction, hr)

            metadata = batch.get("metadata", {})
            sample_ids = metadata.get("sample_id")

            for i in range(len(base_psnr)):
                rows.append({
                    "sample_id": str(sample_ids[i]) if sample_ids is not None else "",
                    "baseline_psnr": base_psnr[i],
                    "gated_psnr": gate_psnr[i],
                    "delta_psnr": gate_psnr[i] - base_psnr[i],
                    "baseline_ssim": base_ssim[i],
                    "gated_ssim": gate_ssim[i],
                    "delta_ssim": gate_ssim[i] - base_ssim[i],
                })

            if len(rows) % 100 == 0:
                print(f"Evaluated {len(rows)}/500 test images")

    if len(rows) != 500:
        raise RuntimeError(f"Expected 500 test results, got {len(rows)}.")

    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    dp = np.asarray([r["delta_psnr"] for r in rows], dtype=float)
    ds = np.asarray([r["delta_ssim"] for r in rows], dtype=float)

    summary = {
        "evaluation": "final_locked_test_comparison",
        "split": "test",
        "num_images": len(rows),
        "alpha": ALPHA,
        "refinement_checkpoint_epoch": int(refinement_ckpt["epoch"]),
        "baseline_sha256": baseline_hash,
        "baseline_checkpoint": str(BASELINE_PATH),
        "refinement_checkpoint": str(REFINEMENT_PATH),
        "metrics": "Mean of per-image PSNR and skimage SSIM; output clamped to [0,1].",
        "baseline_mean_psnr": mean("baseline_psnr"),
        "gated_mean_psnr": mean("gated_psnr"),
        "mean_delta_psnr": mean("delta_psnr"),
        "baseline_mean_ssim": mean("baseline_ssim"),
        "gated_mean_ssim": mean("gated_ssim"),
        "mean_delta_ssim": mean("delta_ssim"),
        "psnr_improved_images": int((dp > 1e-12).sum()),
        "psnr_unchanged_images": int((np.abs(dp) <= 1e-12).sum()),
        "psnr_regressed_images": int((dp < -1e-12).sum()),
        "ssim_improved_images": int((ds > 1e-12).sum()),
        "ssim_regressed_images": int((ds < -1e-12).sum()),
    }

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (OUT / "per_image.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    if sha256(BASELINE_PATH) != EXPECTED_HASH:
        raise RuntimeError("Baseline hash changed during evaluation.")

    print("\n========== FINAL FROZEN TEST RESULTS ==========")
    print(json.dumps(summary, indent=2))
    print("\nSaved results to:", OUT)
    print("Baseline hash unchanged: True")


if __name__ == "__main__":
    main()
