
import os
import sys
import random
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = Path("/mnt/c/MSFFAN_Project")
SCRIPT_DIR = PROJECT_ROOT / "scripts"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# ================================================================
# REPRODUCIBILITY
# ================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ================================================================
# CONFIGURATION
# ================================================================

BATCH_SIZE = 4
NUM_WORKERS = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ================================================================
# IMPORT PROJECT MODULES
# ================================================================

from dataset_loader import MSFFANDataset
from model_msffan_conditioned import MSFFANConditionedModel
from model_loss import MSFFANLoss

# ================================================================
# SOURCE RASTER DIMENSIONS
# ================================================================

SOURCE_DIMS = {
    "san_francisco": (1345, 5574),
    "baltimore":     (5567, 7086),
    "denver":        (5566, 7135),
}

# ================================================================
# CONDITIONING FUNCTION
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

        # Resolution normalization
        # 2x -> 0
        # 3x -> 0.5
        # 4x -> 1

        scale_norm = (scale_factor - 2.0) / 2.0

        # Position normalization

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
# PSNR
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


# ================================================================
# SSIM
# ================================================================

def calculate_ssim(prediction, target):

    # Simple global SSIM implementation.
    # Kept consistent for validation reporting.

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
# VALIDATION
# ================================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    for batch in loader:

        lr = batch["lr"].to(
            device,
            non_blocking=True
        )

        hr = batch["hr"].to(
            device,
            non_blocking=True
        )

        condition = build_condition(
            batch["metadata"],
            device
        )

        prediction = model(
            lr,
            condition
        )

        loss_output = criterion(
            prediction,
            hr
        )

        if isinstance(loss_output, tuple):
            loss = loss_output[0]
        else:
            loss = loss_output

        batch_size = lr.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_psnr += (
            calculate_psnr(
                prediction,
                hr
            ) * batch_size
        )

        total_ssim += (
            calculate_ssim(
                prediction,
                hr
            ) * batch_size
        )

        count += batch_size

    return {
        "loss": total_loss / count,
        "psnr": total_psnr / count,
        "ssim": total_ssim / count
    }


# ================================================================
# MAIN TRAINING
# ================================================================

def main():

    print("=" * 70)
    print("STEP 13G / 13I — PROPOSED MSFFAN TRAINING")
    print("=" * 70)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Device       : {DEVICE}")

    if torch.cuda.is_available():

        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(f"Seed         : {SEED}")
    print(f"Batch size   : {BATCH_SIZE}")
    print(f"Workers      : {NUM_WORKERS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs       : {NUM_EPOCHS}")

    # ------------------------------------------------------------
    # DATASETS
    # ------------------------------------------------------------

    print("\nLoading datasets...")

    train_dataset = MSFFANDataset(
        project_root=PROJECT_ROOT,
        split="train"
    )

    val_dataset = MSFFANDataset(
        project_root=PROJECT_ROOT,
        split="val"
    )

    print(
        f"Train samples: {len(train_dataset)}"
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

    # ------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------

    model = MSFFANConditionedModel().to(
        DEVICE
    )

    num_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nProposed model parameters: "
        f"{num_params:,}"
    )

    # ------------------------------------------------------------
    # LOSS
    # ------------------------------------------------------------

    criterion = MSFFANLoss().to(
        DEVICE
    )

    # ------------------------------------------------------------
    # OPTIMIZER
    # ------------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ------------------------------------------------------------
    # RESUME FROM EXISTING CHECKPOINT
    # ------------------------------------------------------------

    resume_checkpoint = MODEL_DIR / "conditioned_last.pth"
    start_epoch = 1
    resumed_history = []
    resumed_best_psnr = -float("inf")
    resumed_best_epoch = 0

    if resume_checkpoint.exists():

        checkpoint = torch.load(
            resume_checkpoint,
            map_location=DEVICE
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_epoch = checkpoint["epoch"] + 1

        resumed_best_psnr = checkpoint.get(
            "best_psnr",
            -float("inf")
        )

        resumed_best_epoch = checkpoint["epoch"]

        print(
            f"\nResuming from checkpoint: "
            f"{resume_checkpoint}"
        )

        print(
            f"Previous epoch : "
            f"{checkpoint['epoch']}"
        )

        print(
            f"Starting epoch : "
            f"{start_epoch}"
        )

        print(
            f"Previous best PSNR : "
            f"{resumed_best_psnr:.4f} dB"
        )

    # ------------------------------------------------------------
    # TRAINING HISTORY
    # ------------------------------------------------------------

    history = resumed_history

    best_psnr = resumed_best_psnr
    best_epoch = resumed_best_epoch

    best_checkpoint = (
        MODEL_DIR /
        "conditioned_best.pth"
    )

    last_checkpoint = (
        MODEL_DIR /
        "conditioned_last.pth"
    )

    # ------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------

    training_start = time.time()

    for epoch in range(
        start_epoch,
        NUM_EPOCHS + 1
    ):

        epoch_start = time.time()

        model.train()

        running_loss = 0.0
        sample_count = 0

        for batch_idx, batch in enumerate(
            train_loader,
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

            condition = build_condition(
                batch["metadata"],
                DEVICE
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            prediction = model(
                lr,
                condition
            )

            loss_output = criterion(
                prediction,
                hr
            )

            if isinstance(
                loss_output,
                tuple
            ):

                loss = loss_output[0]

            else:

                loss = loss_output

            if not torch.isfinite(loss):

                raise RuntimeError(
                    f"Non-finite loss detected "
                    f"at epoch {epoch}, "
                    f"batch {batch_idx}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            batch_size = lr.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            sample_count += batch_size

            if (
                batch_idx == 1 or
                batch_idx % 100 == 0
            ):

                print(
                    f"Epoch {epoch}/{NUM_EPOCHS} | "
                    f"Batch {batch_idx}/"
                    f"{len(train_loader)} | "
                    f"Loss {loss.item():.6f}"
                )

        train_loss = (
            running_loss /
            sample_count
        )

        # --------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------

        val_metrics = validate(
            model,
            val_loader,
            criterion,
            DEVICE
        )

        epoch_time = (
            time.time() -
            epoch_start
        )

        epoch_record = {

            "epoch": epoch,

            "train_loss":
                train_loss,

            "val_loss":
                val_metrics["loss"],

            "val_psnr":
                val_metrics["psnr"],

            "val_ssim":
                val_metrics["ssim"],

            "epoch_time_seconds":
                epoch_time
        }

        history.append(
            epoch_record
        )

        print("\n" + "-" * 70)

        print(
            f"Epoch {epoch}/{NUM_EPOCHS} COMPLETE"
        )

        print(
            f"Train Loss : "
            f"{train_loss:.6f}"
        )

        print(
            f"Val Loss   : "
            f"{val_metrics['loss']:.6f}"
        )

        print(
            f"Val PSNR   : "
            f"{val_metrics['psnr']:.4f} dB"
        )

        print(
            f"Val SSIM   : "
            f"{val_metrics['ssim']:.6f}"
        )

        print(
            f"Time       : "
            f"{epoch_time:.2f} sec"
        )

        print("-" * 70)

        # --------------------------------------------------------
        # BEST CHECKPOINT
        # --------------------------------------------------------

        if val_metrics["psnr"] > best_psnr:

            best_psnr = (
                val_metrics["psnr"]
            )

            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "train_metrics":
                        {
                            "loss":
                                train_loss
                        },

                    "val_metrics":
                        val_metrics,

                    "best_psnr":
                        best_psnr,

                    "num_parameters":
                        num_params,

                    "seed":
                        SEED,

                    "config":
                        {
                            "batch_size":
                                BATCH_SIZE,

                            "learning_rate":
                                LEARNING_RATE,

                            "epochs":
                                NUM_EPOCHS
                        }
                },
                best_checkpoint
            )

            print(
                f"NEW BEST CHECKPOINT "
                f"→ {best_checkpoint}"
            )

        # --------------------------------------------------------
        # LAST CHECKPOINT
        # --------------------------------------------------------

        torch.save(
            {
                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "train_metrics":
                    {
                        "loss":
                            train_loss
                    },

                "val_metrics":
                    val_metrics,

                "best_psnr":
                    best_psnr,

                "num_parameters":
                    num_params,

                "seed":
                    SEED
            },
            last_checkpoint
        )

        # --------------------------------------------------------
        # SAVE HISTORY
        # --------------------------------------------------------

        history_path = (
            LOG_DIR /
            "conditioned_training_history.json"
        )

        with open(
            history_path,
            "w"
        ) as f:

            json.dump(
                history,
                f,
                indent=2
            )

    total_time = (
        time.time() -
        training_start
    )

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    print("\n")
    print("=" * 70)
    print("PROPOSED TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best epoch       : {best_epoch}"
    )

    print(
        f"Best Val PSNR    : "
        f"{best_psnr:.4f} dB"
    )

    print(
        f"Total time       : "
        f"{total_time:.2f} sec"
    )

    print(
        f"Best checkpoint  : "
        f"{best_checkpoint}"
    )

    print(
        f"Last checkpoint  : "
        f"{last_checkpoint}"
    )

    print(
        f"History          : "
        f"{history_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
