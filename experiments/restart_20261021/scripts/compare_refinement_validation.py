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
RESULTS = RUN / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

BASELINE_PATH = ROOT / "models" / "baseline_best.pth"
REFINEMENT_PATH = RUN / "checkpoints" / "refinement_best.pth"
EXPECTED_BASELINE_HASH = (
    "4f1843da5ad95cc9f4759e84cfc71d824b7f05227c9b448f7c004205faaca21c"
)

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


def image_metrics(prediction, target):
    prediction = prediction.detach().clamp(0, 1).cpu().numpy()
    target = target.detach().clamp(0, 1).cpu().numpy()

    psnr_values, ssim_values = [], []

    for pred, truth in zip(prediction, target):
        mse = float(np.mean((pred - truth) ** 2))
        psnr = float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)
        ssim = structural_similarity(
            truth, pred, channel_axis=0, data_range=1.0
        )
        psnr_values.append(psnr)
        ssim_values.append(float(ssim))

    return psnr_values, ssim_values


def main():
    if not BASELINE_PATH.is_file():
        raise FileNotFoundError(f"Baseline checkpoint missing: {BASELINE_PATH}")
    if not REFINEMENT_PATH.is_file():
        raise FileNotFoundError(f"Refinement checkpoint missing: {REFINEMENT_PATH}")

    baseline_hash = sha256(BASELINE_PATH)
    if baseline_hash != EXPECTED_BASELINE_HASH:
        raise RuntimeError(
            "Baseline hash mismatch; refusing to evaluate an unexpected baseline."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Baseline SHA-256 verified:", baseline_hash)

    baseline_checkpoint = torch.load(
        BASELINE_PATH, map_location=device, weights_only=False
    )
    refinement_checkpoint = torch.load(
        REFINEMENT_PATH, map_location=device, weights_only=False
    )

    if refinement_checkpoint.get("baseline_sha256") != baseline_hash:
        raise RuntimeError("Refinement checkpoint references a different baseline.")

    baseline = MSFFANBaselineModel().to(device)
    baseline.load_state_dict(baseline_checkpoint["model_state_dict"], strict=True)
    baseline.eval()
    for parameter in baseline.parameters():
        parameter.requires_grad_(False)

    refinement = InputConditionedResidualRefinement().to(device)
    refinement.load_state_dict(
        refinement_checkpoint["model_state_dict"], strict=True
    )
    refinement.eval()

    dataset = MSFFANDataset(project_root=ROOT, split="val")
    if len(dataset) != 500:
        raise RuntimeError(
            f"Expected exactly 500 validation images; found {len(dataset)}."
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
        for batch_index, batch in enumerate(loader, start=1):
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            metadata = batch.get("metadata", {})
            base_output = baseline(lr)
            refined_output = refinement(lr, base_output)

            base_psnr, base_ssim = image_metrics(base_output, hr)
            ref_psnr, ref_ssim = image_metrics(refined_output, hr)

            sample_ids = metadata.get("sample_id")
            for i in range(len(base_psnr)):
                rows.append({
                    "index": len(rows),
                    "sample_id": str(sample_ids[i]) if sample_ids is not None else "",
                    "baseline_psnr": base_psnr[i],
                    "refinement_psnr": ref_psnr[i],
                    "delta_psnr": ref_psnr[i] - base_psnr[i],
                    "baseline_ssim": base_ssim[i],
                    "refinement_ssim": ref_ssim[i],
                    "delta_ssim": ref_ssim[i] - base_ssim[i],
                })

            if batch_index % 25 == 0 or len(rows) == len(dataset):
                print(f"Evaluated {len(rows)}/{len(dataset)} validation images")

    if len(rows) != 500:
        raise RuntimeError(f"Expected 500 evaluated images; got {len(rows)}.")

    def mean(key):
        return float(np.mean([row[key] for row in rows]))

    summary = {
        "split": "val",
        "num_images": len(rows),
        "checkpoint_epoch": int(refinement_checkpoint["epoch"]),
        "baseline_sha256": baseline_hash,
        "baseline_checkpoint": str(BASELINE_PATH),
        "refinement_checkpoint": str(REFINEMENT_PATH),
        "metric_protocol": (
            "Per-image PSNR and skimage SSIM; predictions and targets clamped "
            "to [0,1]; same images and preprocessing for both models."
        ),
        "baseline_mean_psnr": mean("baseline_psnr"),
        "refinement_mean_psnr": mean("refinement_psnr"),
        "mean_delta_psnr": mean("delta_psnr"),
        "baseline_mean_ssim": mean("baseline_ssim"),
        "refinement_mean_ssim": mean("refinement_ssim"),
        "mean_delta_ssim": mean("delta_ssim"),
        "images_psnr_improved": int(sum(r["delta_psnr"] > 1e-12 for r in rows)),
        "images_psnr_unchanged": int(sum(abs(r["delta_psnr"]) <= 1e-12 for r in rows)),
        "images_psnr_regressed": int(sum(r["delta_psnr"] < -1e-12 for r in rows)),
        "images_ssim_improved": int(sum(r["delta_ssim"] > 1e-12 for r in rows)),
        "images_ssim_regressed": int(sum(r["delta_ssim"] < -1e-12 for r in rows)),
    }

    summary_path = RESULTS / "validation_comparison.json"
    rows_path = RESULTS / "validation_per_image.json"

    summary_path.write_text(json.dumps(summary, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value)), encoding="utf-8")
    rows_path.write_text(json.dumps(rows, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value)), encoding="utf-8")

    print("\n========== VALIDATION COMPARISON ==========")
    print(json.dumps(summary, indent=2, default=lambda v: v.item() if hasattr(v, 'item') else str(v)))
    print("\nSaved summary:", summary_path)
    print("Saved per-image results:", rows_path)

    if sha256(BASELINE_PATH) != EXPECTED_BASELINE_HASH:
        raise RuntimeError("Baseline checkpoint hash changed during evaluation.")
    print("Baseline hash unchanged after evaluation: True")


if __name__ == "__main__":
    main()
