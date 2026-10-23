import sys
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("/mnt/c/MSFFAN_Project")
SCRIPT_DIR = PROJECT_ROOT / "scripts"
FILM_DIR = PROJECT_ROOT / "experiments/architecture_comparison_20261023/film"
RESULT_DIR = FILM_DIR / "evaluation"

BASELINE_CHECKPOINT = PROJECT_ROOT / "models/baseline_best.pth"
FILM_CHECKPOINT = FILM_DIR / "models/conditioned_best.pth"

EXPECTED_BASELINE_SHA256 = (
    "4f1843da5ad95cc9f4759e84cfc71d824b7f05227c9b448f7c004205faaca21c"
)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset_loader import MSFFANDataset
from model_msffan_baseline import MSFFANBaselineModel
from model_msffan_conditioned import MSFFANConditionedModel

SOURCE_DIMS = {
    "san_francisco": (1345, 5574),
    "baltimore": (5567, 7086),
    "denver": (5566, 7135),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_condition(metadata):
    conditions = []

    for i in range(len(metadata["sample_id"])):
        location = metadata["location"][i]
        width, height = SOURCE_DIMS[location]

        scale = float(metadata["scale_factor"][i])
        x = float(metadata["x"][i])
        y = float(metadata["y"][i])

        conditions.append([
            (scale - 2.0) / 2.0,
            x / width,
            y / height,
        ])

    return torch.tensor(
        conditions,
        dtype=torch.float32,
        device=DEVICE,
    )


def image_metrics(prediction, target):
    prediction = prediction.detach().float().clamp(0, 1).cpu().numpy()
    target = target.detach().float().clamp(0, 1).cpu().numpy()

    mse = float(np.mean(
        (prediction.astype(np.float64) - target.astype(np.float64)) ** 2
    ))

    psnr = float("inf") if mse == 0 else float(
        10.0 * np.log10(1.0 / mse)
    )

    ssim = float(structural_similarity(
        target,
        prediction,
        channel_axis=0,
        data_range=1.0,
    ))

    return psnr, ssim


def load_model(model_class, checkpoint_path):
    model = model_class().to(DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    return model, checkpoint


@torch.inference_mode()
def main():
    print("=" * 70)
    print("ISOLATED FILM VS BASELINE TEST EVALUATION")
    print("=" * 70)

    assert BASELINE_CHECKPOINT.is_file()
    assert FILM_CHECKPOINT.is_file()

    baseline_hash = sha256_file(BASELINE_CHECKPOINT)
    assert baseline_hash == EXPECTED_BASELINE_SHA256, (
        "Original baseline checkpoint hash mismatch. Evaluation stopped."
    )

    dataset = MSFFANDataset(
        project_root=PROJECT_ROOT,
        split="test",
    )

    assert len(dataset) == 500, (
        f"Expected 500 frozen test samples; found {len(dataset)}"
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    baseline, baseline_ckpt = load_model(
        MSFFANBaselineModel,
        BASELINE_CHECKPOINT,
    )

    film, film_ckpt = load_model(
        MSFFANConditionedModel,
        FILM_CHECKPOINT,
    )

    assert film_ckpt["epoch"] == 93, (
        "Unexpected FiLM checkpoint epoch"
    )

    print("Device:", DEVICE)
    print("Test samples:", len(dataset))
    print("Baseline checkpoint SHA256:", baseline_hash)
    print("FiLM best epoch:", film_ckpt["epoch"])

    records = []

    for batch_index, batch in enumerate(loader, 1):
        lr = batch["lr"].to(DEVICE, non_blocking=True)
        hr = batch["hr"].to(DEVICE, non_blocking=True)
        metadata = batch["metadata"]

        baseline_prediction = baseline(lr)
        film_prediction = film(lr, build_condition(metadata))

        for i in range(lr.size(0)):
            baseline_psnr, baseline_ssim = image_metrics(
                baseline_prediction[i],
                hr[i],
            )
            film_psnr, film_ssim = image_metrics(
                film_prediction[i],
                hr[i],
            )

            records.append({
                "sample_id": str(metadata["sample_id"][i]),
                "location": str(metadata["location"][i]),
                "scale_factor": int(metadata["scale_factor"][i]),
                "baseline_psnr": baseline_psnr,
                "baseline_ssim": baseline_ssim,
                "film_psnr": film_psnr,
                "film_ssim": film_ssim,
                "delta_psnr": film_psnr - baseline_psnr,
                "delta_ssim": film_ssim - baseline_ssim,
            })

        if batch_index == 1 or batch_index % 25 == 0:
            print(f"Processed {len(records)}/500 images")

    assert len(records) == 500

    def mean(key):
        return float(np.mean([record[key] for record in records]))

    summary = {
        "test_samples": len(records),
        "metric_protocol": (
            "Per-image PSNR and skimage SSIM, predictions and "
            "targets clamped to [0,1], arithmetic mean over 500 images"
        ),
        "baseline": {
            "checkpoint": str(BASELINE_CHECKPOINT),
            "sha256": baseline_hash,
            "psnr": mean("baseline_psnr"),
            "ssim": mean("baseline_ssim"),
        },
        "film": {
            "checkpoint": str(FILM_CHECKPOINT),
            "sha256": sha256_file(FILM_CHECKPOINT),
            "best_epoch": film_ckpt["epoch"],
            "psnr": mean("film_psnr"),
            "ssim": mean("film_ssim"),
        },
        "difference": {
            "psnr": mean("delta_psnr"),
            "ssim": mean("delta_ssim"),
        },
        "psnr_improved_images": sum(
            record["delta_psnr"] > 0 for record in records
        ),
        "psnr_regressed_images": sum(
            record["delta_psnr"] < 0 for record in records
        ),
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULT_DIR / "film_test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(RESULT_DIR / "film_per_sample.json", "w") as f:
        json.dump(records, f, indent=2)

    print("\nFINAL TEST SUMMARY")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
