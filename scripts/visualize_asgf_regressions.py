from pathlib import Path
import json
import sys

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dataset_loader import MSFFANDataset
from model_msffan_baseline import MSFFANBaselineModel
from model_msffan import MSFFANModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "results" / "asgf_visual_diagnostic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_IDS = [
    "test_00267",
    "test_00070",
    "test_00439",
    "test_00100",
]


def load_model(model_class, checkpoint_name):
    model = model_class().to(DEVICE)
    checkpoint = torch.load(
        MODEL_DIR / checkpoint_name,
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def to_image(tensor):
    # Match the visualization convention: display model output clipped to [0, 1].
    arr = tensor.detach().float().cpu().squeeze(0).permute(1, 2, 0).numpy()
    return np.clip(arr, 0.0, 1.0)


def psnr(pred, target):
    mse = torch.mean((pred.float() - target.float()) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def main():
    print("Device:", DEVICE)
    dataset = MSFFANDataset(PROJECT_ROOT, split="test")

    rows = json.loads(
        (PROJECT_ROOT / "results/test_evaluation/per_sample_results.json")
        .read_text()
    )
    result_by_id = {row["sample_id"]: row for row in rows}

    print("Loading baseline and ASGF checkpoints...")
    baseline = load_model(MSFFANBaselineModel, "baseline_best.pth")
    asgf = load_model(MSFFANModel, "asgf_best.pth")

    output_metrics = []

    with torch.no_grad():
        for sample_id in TARGET_IDS:
            matches = [
                i for i, row in dataset.metadata.iterrows()
                if row["sample_id"] == sample_id
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected exactly one test row for {sample_id}; found {len(matches)}"
                )

            sample = dataset[matches[0]]
            lr = sample["lr"].unsqueeze(0).to(DEVICE)
            hr = sample["hr"].unsqueeze(0).to(DEVICE)

            base_pred = baseline(lr)
            asgf_pred = asgf(lr)

            base_psnr = psnr(base_pred, hr)
            asgf_psnr = psnr(asgf_pred, hr)

            metadata = sample["metadata"]
            original = result_by_id.get(sample_id, {})

            print(
                f"{sample_id}: {metadata['location']} scale={metadata['scale_factor']}x | "
                f"baseline PSNR={base_psnr:.4f} | ASGF PSNR={asgf_psnr:.4f} | "
                f"delta={asgf_psnr - base_psnr:+.4f} dB | "
                f"saved-evaluation delta="
                f"{original.get('proposed_psnr', float('nan')) - original.get('baseline_psnr', float('nan')):+.4f} dB"
            )

            panels = [
                ("LR input", to_image(lr)),
                ("HR target", to_image(hr)),
                (f"Baseline\nPSNR {base_psnr:.2f} dB", to_image(base_pred)),
                (f"ASGF\nPSNR {asgf_psnr:.2f} dB", to_image(asgf_pred)),
                ("Baseline absolute error", np.abs(to_image(base_pred) - to_image(hr))),
                ("ASGF absolute error", np.abs(to_image(asgf_pred) - to_image(hr))),
            ]

            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            for ax, (title, image) in zip(axes.flat, panels):
                ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
                ax.set_title(title)
                ax.axis("off")
            fig.suptitle(
                f"{sample_id} | {metadata['location']} | "
                f"{metadata['scale_factor']}x | "
                f"ASGF ΔPSNR {asgf_psnr - base_psnr:+.3f} dB",
                fontsize=13,
            )
            fig.tight_layout()
            fig.savefig(OUT_DIR / f"{sample_id}_comparison.png", dpi=140)
            plt.close(fig)

            output_metrics.append({
                "sample_id": sample_id,
                "location": metadata["location"],
                "scale_factor": metadata["scale_factor"],
                "baseline_psnr_recomputed": base_psnr,
                "asgf_psnr_recomputed": asgf_psnr,
                "delta_psnr_recomputed": asgf_psnr - base_psnr,
                "baseline_psnr_saved": original.get("baseline_psnr"),
                "asgf_psnr_saved": original.get("proposed_psnr"),
            })

    (OUT_DIR / "diagnostic_metrics.json").write_text(
        json.dumps(output_metrics, indent=2)
    )
    print("\nSaved diagnostic files to:", OUT_DIR)
    print("Original dataset, checkpoints, and evaluation results were not modified.")


if __name__ == "__main__":
    main()
