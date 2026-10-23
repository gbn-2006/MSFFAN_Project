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
OUT = RUN / "results" / "residual_gate_experiment"
OUT.mkdir(parents=True, exist_ok=True)

BASELINE_PATH = ROOT / "models" / "baseline_best.pth"
REFINEMENT_PATH = RUN / "checkpoints" / "refinement_best.pth"
EXPECTED_HASH = (
    "4f1843da5ad95cc9f4759e84cfc71d824b7f05227c9b448f7c004205faaca21c"
)
GATES = [0.0, 0.25, 0.5, 0.75, 1.0]
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


def metrics(pred, target):
    pred = pred.detach().clamp(0, 1).cpu().numpy()
    target = target.detach().clamp(0, 1).cpu().numpy()
    psnr_values, ssim_values = [], []

    for p, t in zip(pred, target):
        mse = float(np.mean((p - t) ** 2))
        psnr_values.append(
            float("inf") if mse == 0 else float(10 * np.log10(1 / mse))
        )
        ssim_values.append(float(structural_similarity(
            t, p, channel_axis=0, data_range=1.0
        )))
    return psnr_values, ssim_values


def main():
    actual_hash = sha256(BASELINE_PATH)
    if actual_hash != EXPECTED_HASH:
        raise RuntimeError("Baseline checkpoint hash mismatch.")
    if not REFINEMENT_PATH.is_file():
        raise FileNotFoundError(REFINEMENT_PATH)

    checkpoint = torch.load(
        REFINEMENT_PATH, map_location=DEVICE, weights_only=False
    )
    if checkpoint.get("baseline_sha256") != actual_hash:
        raise RuntimeError("Refinement checkpoint baseline hash mismatch.")

    base_ckpt = torch.load(
        BASELINE_PATH, map_location=DEVICE, weights_only=False
    )
    baseline = MSFFANBaselineModel().to(DEVICE)
    baseline.load_state_dict(base_ckpt["model_state_dict"], strict=True)
    baseline.eval()
    for p in baseline.parameters():
        p.requires_grad_(False)

    refinement = InputConditionedResidualRefinement().to(DEVICE)
    refinement.load_state_dict(checkpoint["model_state_dict"], strict=True)
    refinement.eval()

    dataset = MSFFANDataset(ROOT, split="val")
    if len(dataset) != 500:
        raise RuntimeError(f"Expected 500 validation images; got {len(dataset)}")
    loader = DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    rows = []
    with torch.inference_mode():
        for batch in loader:
            lr = batch["lr"].to(DEVICE)
            hr = batch["hr"].to(DEVICE)
            base = baseline(lr)
            full_refined = refinement(lr, base)
            residual = full_refined - base

            base_psnr, base_ssim = metrics(base, hr)
            row_metrics = {
                "baseline_psnr": base_psnr,
                "baseline_ssim": base_ssim,
            }

            for alpha in GATES:
                output = (base + alpha * residual).clamp(0, 1)
                psnr, ssim = metrics(output, hr)
                row_metrics[f"gate_{alpha:g}_psnr"] = psnr
                row_metrics[f"gate_{alpha:g}_ssim"] = ssim

            meta = batch.get("metadata", {})
            sample_ids = meta.get("sample_id")
            for i in range(len(base_psnr)):
                row = {
                    "sample_id": str(sample_ids[i]) if sample_ids is not None else "",
                    "baseline_psnr": base_psnr[i],
                    "baseline_ssim": base_ssim[i],
                }
                for alpha in GATES:
                    tag = f"gate_{alpha:g}"
                    row[f"{tag}_psnr"] = row_metrics[f"{tag}_psnr"][i]
                    row[f"{tag}_ssim"] = row_metrics[f"{tag}_ssim"][i]
                    row[f"{tag}_delta_psnr"] = row[f"{tag}_psnr"] - base_psnr[i]
                    row[f"{tag}_delta_ssim"] = row[f"{tag}_ssim"] - base_ssim[i]
                rows.append(row)

            if len(rows) % 100 == 0:
                print(f"Evaluated {len(rows)}/500 images")

    if len(rows) != 500:
        raise RuntimeError(f"Expected 500 rows; got {len(rows)}")

    summary = {
        "split": "val",
        "num_images": len(rows),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "baseline_sha256": actual_hash,
        "gates": {},
    }

    for alpha in GATES:
        tag = f"gate_{alpha:g}"
        dp = np.array([r[f"{tag}_delta_psnr"] for r in rows], dtype=float)
        ds = np.array([r[f"{tag}_delta_ssim"] for r in rows], dtype=float)
        ps = np.array([r[f"{tag}_psnr"] for r in rows], dtype=float)
        ss = np.array([r[f"{tag}_ssim"] for r in rows], dtype=float)

        summary["gates"][str(alpha)] = {
            "mean_psnr": float(ps.mean()),
            "mean_ssim": float(ss.mean()),
            "mean_delta_psnr": float(dp.mean()),
            "median_delta_psnr": float(np.median(dp)),
            "mean_delta_ssim": float(ds.mean()),
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

    print("\n========== RESIDUAL GATE SUMMARY ==========")
    print(json.dumps(summary, indent=2))
    print("\nSaved to:", OUT)

    if sha256(BASELINE_PATH) != EXPECTED_HASH:
        raise RuntimeError("Baseline hash changed during evaluation.")
    print("Baseline hash unchanged: True")


if __name__ == "__main__":
    main()
