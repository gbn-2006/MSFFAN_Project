import sys
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("/mnt/c/MSFFAN_Project")
SCRIPT_DIR = PROJECT_ROOT / "scripts"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "results" / "test_evaluation"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset_loader import MSFFANDataset
from model_msffan_hf_full import MSFFANModel


# ================================================================
# CONFIGURATION
# ================================================================

BATCH_SIZE = 4
NUM_WORKERS = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

SOURCE_DIMS = {
    "san_francisco": (1345, 5574),
    "baltimore":     (5567, 7086),
    "denver":        (5566, 7135),
}


# ================================================================
# CONDITIONING
# ================================================================

def build_condition(metadata, device):

    conditions = []

    batch_size = len(metadata["sample_id"])

    for i in range(batch_size):

        location = metadata["location"][i]

        scale_factor = float(
            metadata["scale_factor"][i]
        )

        x = float(metadata["x"][i])
        y = float(metadata["y"][i])

        width, height = SOURCE_DIMS[location]

        scale_norm = (
            scale_factor - 2.0
        ) / 2.0

        x_norm = x / width
        y_norm = y / height

        conditions.append([
            scale_norm,
            x_norm,
            y_norm
        ])

    return torch.tensor(
        conditions,
        dtype=torch.float32,
        device=device
    )


# ================================================================
# METRICS
# ================================================================

def calculate_psnr(prediction, target):

    mse = torch.mean(
        (prediction - target) ** 2
    )

    if mse.item() == 0:
        return float("inf")

    return (
        10.0 *
        torch.log10(
            1.0 / mse
        )
    ).item()


def calculate_ssim(prediction, target):

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = prediction.mean(
        dim=(2, 3),
        keepdim=True
    )

    mu_y = target.mean(
        dim=(2, 3),
        keepdim=True
    )

    sigma_x = (
        (prediction - mu_x) ** 2
    ).mean(
        dim=(2, 3),
        keepdim=True
    )

    sigma_y = (
        (target - mu_y) ** 2
    ).mean(
        dim=(2, 3),
        keepdim=True
    )

    sigma_xy = (
        (prediction - mu_x) *
        (target - mu_y)
    ).mean(
        dim=(2, 3),
        keepdim=True
    )

    numerator = (
        (2 * mu_x * mu_y + C1) *
        (2 * sigma_xy + C2)
    )

    denominator = (
        (mu_x ** 2 + mu_y ** 2 + C1) *
        (sigma_x + sigma_y + C2)
    )

    ssim = numerator / denominator

    return ssim.mean().item()


# ================================================================
# MODEL LOADING
# ================================================================

def load_hf():

    model = MSFFANModel().to(DEVICE)

    checkpoint_path = (
        MODEL_DIR / "hf_best.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


# ================================================================
# EVALUATION
# ================================================================

@torch.no_grad()
def evaluate():

    print("=" * 70)
    print("FINAL HF TEST EVALUATION")
    print("=" * 70)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Device       : {DEVICE}")

    if torch.cuda.is_available():
        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ------------------------------------------------------------
    # DATASET
    # ------------------------------------------------------------

    print("\nLoading frozen test dataset...")

    test_dataset = MSFFANDataset(
        project_root=PROJECT_ROOT,
        split="test"
    )

    print(
        f"Test samples: {len(test_dataset)}"
    )

    if len(test_dataset) != 500:
        raise RuntimeError(
            f"Expected 500 test samples, "
            f"found {len(test_dataset)}"
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    # ------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------

    print("\nLoading HF-MSFFAN checkpoint...")
    hf_model = load_hf()

    print("HF-MSFFAN checkpoint loaded successfully.")

    # ------------------------------------------------------------
    # ACCUMULATORS
    # ------------------------------------------------------------

    hf_psnr = 0.0
    hf_ssim = 0.0

    sample_count = 0

    per_sample_results = []

    # ------------------------------------------------------------
    # EVALUATION LOOP
    # ------------------------------------------------------------

    for batch_idx, batch in enumerate(
        test_loader,
        start=1
    ):

        lr = batch["lr"].to(
            DEVICE,
            non_blocking=True
        )

        hr = batch["hr"].to(
            DEVICE,
            non_blocking=True
        )

        metadata = batch["metadata"]

        # --------------------------------------------------------
        # HF-MSFFAN
        # --------------------------------------------------------

        hf_prediction = hf_model(lr)

        batch_size = lr.size(0)

        # --------------------------------------------------------
        # BATCH METRICS
        # --------------------------------------------------------

        batch_hf_psnr = calculate_psnr(
            hf_prediction,
            hr
        )

        batch_hf_ssim = calculate_ssim(
            hf_prediction,
            hr
        )

        hf_psnr += (
            batch_hf_psnr * batch_size
        )

        hf_ssim += (
            batch_hf_ssim * batch_size
        )

        sample_count += batch_size

        # --------------------------------------------------------
        # PER-SAMPLE RECORDS
        # --------------------------------------------------------

        for i in range(batch_size):

            sample_hf_psnr = calculate_psnr(
                hf_prediction[i:i+1],
                hr[i:i+1]
            )

            sample_hf_ssim = calculate_ssim(
                hf_prediction[i:i+1],
                hr[i:i+1]
            )

            per_sample_results.append({
                "sample_id":
                    metadata["sample_id"][i],
                "location":
                    metadata["location"][i],
                "scale_factor":
                    int(metadata["scale_factor"][i]),
                "x":
                    int(metadata["x"][i]),
                "y":
                    int(metadata["y"][i]),
                "blur_type":
                    metadata["blur_type"][i],
                "jpeg_quality":
                    int(metadata["jpeg_quality"][i]),
                "contrast_factor":
                    float(metadata["contrast_factor"][i]),
                "hf_psnr":
                    sample_hf_psnr,
                "hf_ssim":
                    sample_hf_ssim,
            })

        if (
            batch_idx == 1 or
            batch_idx % 25 == 0 or
            batch_idx == len(test_loader)
        ):
            print(
                f"Batch {batch_idx}/{len(test_loader)} "
                f"| Samples {sample_count}/500"
            )

    # ------------------------------------------------------------
    # FINAL RESULTS
    # ------------------------------------------------------------

    results = {
        "test_samples": sample_count,
        "device": str(DEVICE),
        "hf_msffan": {
            "checkpoint":
                str(MODEL_DIR / "hf_best.pth"),
            "parameters": 799281,
            "psnr":
                hf_psnr / sample_count,
            "ssim":
                hf_ssim / sample_count,
        },
    }

    # ------------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------------

    summary_path = (
        RESULT_DIR / "test_summary.json"
    )

    samples_path = (
        RESULT_DIR / "per_sample_results.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            indent=4
        )

    with open(
        samples_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            per_sample_results,
            f,
            indent=4
        )

    # ------------------------------------------------------------
    # PRINT RESULTS
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL HF-MSFFAN TEST RESULTS")
    print("=" * 70)

    print(
        f"Test samples : {sample_count}"
    )

    print("\nHF-MSFFAN")
    print(
        f"Parameters : "
        f"{results['hf_msffan']['parameters']:,}"
    )
    print(
        f"PSNR       : "
        f"{results['hf_msffan']['psnr']:.4f} dB"
    )
    print(
        f"SSIM       : "
        f"{results['hf_msffan']['ssim']:.6f}"
    )

    print("\nResults saved:")
    print(summary_path)
    print(samples_path)


if __name__ == "__main__":
    evaluate()
