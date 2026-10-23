from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from dataset_loader import MSFFANDataset
from model_msffan_baseline import MSFFANBaselineModel
from model_msffan import MSFFANModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IDS = ["test_00267", "test_00070", "test_00439", "test_00100"]

def load_model(cls, filename):
    model = cls().to(DEVICE)
    ckpt = torch.load(ROOT / "models" / filename,
                      map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def stats(name, tensor):
    a = tensor.detach().float().cpu().numpy()
    print(
        f"  {name:10s} min={a.min():.6f} max={a.max():.6f} "
        f"mean={a.mean():.6f} std={a.std():.6f}"
    )

def main():
    ds = MSFFANDataset(ROOT, split="test")
    baseline = load_model(MSFFANBaselineModel, "baseline_best.pth")
    asgf = load_model(MSFFANModel, "asgf_best.pth")

    for sample_id in IDS:
        indexes = ds.metadata.index[
            ds.metadata["sample_id"] == sample_id
        ].tolist()
        if len(indexes) != 1:
            raise RuntimeError(f"{sample_id}: expected one dataset row")

        sample = ds[indexes[0]]
        lr = sample["lr"].unsqueeze(0).to(DEVICE)
        hr = sample["hr"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            bp = baseline(lr)
            ap = asgf(lr)

        print(f"\n{sample_id} | {sample['metadata']['location']} "
              f"| scale={sample['metadata']['scale_factor']}x")
        print("  metadata:",
              "blur=", sample["metadata"]["blur_type"],
              "jpeg_quality=", sample["metadata"]["jpeg_quality"],
              "contrast=", sample["metadata"]["contrast_factor"])
        for name, tensor in [
            ("LR input", lr), ("HR target", hr),
            ("Baseline", bp), ("ASGF", ap),
            ("Baseline err", bp - hr), ("ASGF err", ap - hr)
        ]:
            stats(name, tensor)

        print(f"  MAE baseline={torch.mean(torch.abs(bp-hr)).item():.8f}"
              f" | ASGF={torch.mean(torch.abs(ap-hr)).item():.8f}")
        print(f"  MSE baseline={torch.mean((bp-hr)**2).item():.10f}"
              f" | ASGF={torch.mean((ap-hr)**2).item():.10f}")

if __name__ == "__main__":
    main()

