
import sys
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path("/mnt/c/MSFFAN_Project")
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATASET_ROOT = PROJECT_ROOT / "data" / "dataset"
LOG_DIR = PROJECT_ROOT / "logs"
MODEL_DIR = PROJECT_ROOT / "models"

BATCH_SIZE = 4
NUM_WORKERS = 2

LEARNING_RATE = 1e-4
EPOCHS = 5

# False = complete epoch
# True  = limited smoke test
SMOKE_TEST = False

SMOKE_TRAIN_BATCHES = 10
SMOKE_VAL_BATCHES = 5

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DIRECT MODULE LOADER
# ============================================================

def load_module(module_name, filename):

    path = SCRIPTS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required module not found: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        str(path)
    )

    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module


# ============================================================
# LOAD PROJECT MODULES
# ============================================================

dataset_module = load_module(
    "dataset_loader",
    "dataset_loader.py"
)

model_module = load_module(
    "model_msffan_hf_full",
    "model_msffan_hf_full.py"
)

loss_module = load_module(
    "model_loss",
    "model_loss.py"
)

MSFFANDataset = dataset_module.MSFFANDataset
MSFFANModel = model_module.MSFFANModel
MSFFANLoss = loss_module.MSFFANLoss


# ============================================================
# METRICS
# ============================================================

def calculate_psnr(prediction, target):

    prediction = torch.clamp(prediction, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)

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

    prediction = torch.clamp(
        prediction, 0.0, 1.0
    ).detach().cpu().numpy()

    target = torch.clamp(
        target, 0.0, 1.0
    ).detach().cpu().numpy()

    values = []

    for i in range(prediction.shape[0]):

        score = structural_similarity(
            target[i],
            prediction[i],
            channel_axis=0,
            data_range=1.0
        )

        values.append(score)

    return float(np.mean(values))


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    epoch
):

    model.train()

    total_loss = 0.0
    total_pixel = 0.0
    total_edge = 0.0

    batches = 0

    for batch_index, batch in enumerate(loader):

        if SMOKE_TEST and batch_index >= SMOKE_TRAIN_BATCHES:
            break

        lr = batch["lr"].to(
            DEVICE,
            non_blocking=True
        )

        hr = batch["hr"].to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = model(lr)

        total, pixel, edge = criterion(
            prediction,
            hr
        )

        total.backward()

        optimizer.step()

        total_loss += total.item()
        total_pixel += pixel.item()
        total_edge += edge.item()

        batches += 1

        if (
            batch_index < 3
            or (batch_index + 1) % 100 == 0
        ):

            print(
                f"Epoch {epoch} | "
                f"Batch {batch_index + 1} | "
                f"Loss {total.item():.6f}"
            )

    return {
        "loss": total_loss / batches,
        "pixel": total_pixel / batches,
        "edge": total_edge / batches
    }


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0
    total_pixel = 0.0
    total_edge = 0.0

    psnr_values = []
    ssim_values = []

    batches = 0

    for batch_index, batch in enumerate(loader):

        if SMOKE_TEST and batch_index >= SMOKE_VAL_BATCHES:
            break

        lr = batch["lr"].to(
            DEVICE,
            non_blocking=True
        )

        hr = batch["hr"].to(
            DEVICE,
            non_blocking=True
        )

        prediction = model(lr)

        total, pixel, edge = criterion(
            prediction,
            hr
        )

        total_loss += total.item()
        total_pixel += pixel.item()
        total_edge += edge.item()

        psnr_values.append(
            calculate_psnr(
                prediction,
                hr
            )
        )

        ssim_values.append(
            calculate_ssim(
                prediction,
                hr
            )
        )

        batches += 1

    return {
        "loss": total_loss / batches,
        "pixel": total_pixel / batches,
        "edge": total_edge / batches,
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values))
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MSFFAN HIGH-FREQUENCY TRAINING")
    print("=" * 70)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset root : {DATASET_ROOT}")
    print(f"Device       : {DEVICE}")

    if torch.cuda.is_available():

        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(f"Batch size   : {BATCH_SIZE}")
    print(f"Workers      : {NUM_WORKERS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs       : {EPOCHS}")
    print(f"Smoke test   : {SMOKE_TEST}")

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print("\nCreating datasets...")

    train_dataset = MSFFANDataset(
        project_root=PROJECT_ROOT,
        split="train"
    )

    val_dataset = MSFFANDataset(
        project_root=PROJECT_ROOT,
        split="val"
    )

    print(
        f"Training samples  : {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(val_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    print("DataLoaders created.")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print("\nCreating HF-MSFFAN...")

    model = MSFFANModel().to(DEVICE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = MSFFANLoss().to(DEVICE)

    print(
        f"Loss kernel device: "
        f"{criterion.edge_loss.kernel.device}"
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # CHECKPOINT TRACKING
    # --------------------------------------------------------

    best_psnr = -float("inf")

    best_checkpoint = (
        MODEL_DIR / "hf_best.pth"
    )

    last_checkpoint = (
        MODEL_DIR / "hf_last.pth"
    )

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        print("\n" + "=" * 70)
        print(f"EPOCH {epoch}/{EPOCHS}")
        print("=" * 70)

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            epoch
        )

        print("\nTRAINING RESULTS")
        print("-" * 70)

        print(
            f"Train loss : "
            f"{train_metrics['loss']:.6f}"
        )

        print(
            f"Train pixel: "
            f"{train_metrics['pixel']:.6f}"
        )

        print(
            f"Train edge : "
            f"{train_metrics['edge']:.6f}"
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_metrics = validate(
            model,
            val_loader,
            criterion
        )

        print("\nVALIDATION RESULTS")
        print("-" * 70)

        print(
            f"Val loss : "
            f"{val_metrics['loss']:.6f}"
        )

        print(
            f"Val pixel: "
            f"{val_metrics['pixel']:.6f}"
        )

        print(
            f"Val edge : "
            f"{val_metrics['edge']:.6f}"
        )

        print(
            f"Val PSNR : "
            f"{val_metrics['psnr']:.4f} dB"
        )

        print(
            f"Val SSIM : "
            f"{val_metrics['ssim']:.6f}"
        )

        # ----------------------------------------------------
        # LAST CHECKPOINT
        # ----------------------------------------------------

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "best_psnr": best_psnr
            },
            last_checkpoint
        )

        # ----------------------------------------------------
        # BEST CHECKPOINT
        # ----------------------------------------------------

        if val_metrics["psnr"] > best_psnr:

            best_psnr = val_metrics["psnr"]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "best_psnr": best_psnr
                },
                best_checkpoint
            )

            print(
                f"\nNew best model saved."
            )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best validation PSNR: "
        f"{best_psnr:.4f} dB"
    )

    print(
        f"Best checkpoint: "
        f"{best_checkpoint}"
    )

    print(
        f"Last checkpoint: "
        f"{last_checkpoint}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
