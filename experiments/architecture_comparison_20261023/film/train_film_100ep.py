
import os
import sys
import random
import json
import time
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity
import torch.nn as nn
from torch.utils.data import DataLoader

# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = Path("/mnt/c/MSFFAN_Project")
SCRIPT_DIR = PROJECT_ROOT / "scripts"
MODEL_DIR = Path(__file__).resolve().parent / "models"
LOG_DIR = Path(__file__).resolve().parent / "logs"

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
NUM_EPOCHS = 100
EARLY_STOP_PATIENCE = 15

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
# PSNR, SSIM, AND VALIDATION
# Match the residual trainer's per-image metric protocol.

def calculate_psnr(prediction, target):
    prediction = prediction.detach().clamp(0, 1)
    target = target.detach().clamp(0, 1)
    mse = torch.mean((prediction - target) ** 2)
    if mse.item() == 0:
        return float("inf")
    return (10.0 * torch.log10(1.0 / mse)).item()


def calculate_ssim(prediction, target):
    prediction = prediction.detach().clamp(0, 1).cpu().numpy()
    target = target.detach().clamp(0, 1).cpu().numpy()
    values = []
    for pred, truth in zip(prediction, target):
        values.append(
            structural_similarity(
                truth, pred, channel_axis=0, data_range=1.0
            )
        )
    return float(np.mean(values))


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    psnr_values = []
    ssim_values = []
    count = 0

    for batch in loader:
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)
        condition = build_condition(batch["metadata"], device)
        prediction = model(lr, condition)

        loss_output = criterion(prediction, hr)
        loss = loss_output[0] if isinstance(loss_output, tuple) else loss_output

        batch_size = lr.size(0)
        total_loss += loss.item() * batch_size

        pred_clamped = prediction.detach().clamp(0, 1)
        hr_clamped = hr.detach().clamp(0, 1)

        for pred, truth in zip(pred_clamped, hr_clamped):
            mse = torch.mean((pred - truth) ** 2).item()
            psnr_values.append(
                float("inf") if mse == 0
                else float(10.0 * np.log10(1.0 / mse))
            )

        ssim_values.extend(
            # calculate_ssim returns one score per image for this batch.
            [float(x) for x in _ssim_per_image(pred_clamped, hr_clamped)]
        )
        count += batch_size

    if count == 0:
        raise RuntimeError("Validation loader returned zero samples.")

    return {
        "loss": total_loss / count,
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)),
    }


def _ssim_per_image(prediction, target):
    prediction = prediction.detach().clamp(0, 1).cpu().numpy()
    target = target.detach().clamp(0, 1).cpu().numpy()
    return [
        float(structural_similarity(
            truth, pred, channel_axis=0, data_range=1.0
        ))
        for pred, truth in zip(prediction, target)
    ]


# MAIN TRAINING
# ================================================================

def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


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
    history_path = LOG_DIR / "conditioned_training_history.json"
    resumed_history = []
    if history_path.exists():
        with open(history_path) as f:
            resumed_history = json.load(f)

    resumed_best_psnr = -float("inf")
    resumed_best_epoch = 0
    resumed_epochs_without_improvement = 0

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

        # Restore RNG states when available for safer recovery.
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if "cuda_rng_state_all" in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])

        start_epoch = checkpoint["epoch"] + 1

        resumed_best_psnr = checkpoint.get(
            "best_psnr",
            -float("inf")
        )

        resumed_best_epoch = checkpoint.get("best_epoch", checkpoint["epoch"])
        resumed_epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )

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
    epochs_without_improvement = resumed_epochs_without_improvement

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

            epochs_without_improvement = 0
            best_psnr = (
                val_metrics["psnr"]
            )

            best_epoch = epoch

            atomic_save(
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

                    "best_epoch":
                        best_epoch,

                    "epochs_without_improvement":
                        epochs_without_improvement,

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
        else:
            epochs_without_improvement += 1

        # --------------------------------------------------------
        # LAST CHECKPOINT
        # --------------------------------------------------------

        atomic_save(
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

                "best_epoch":
                    best_epoch,

                "epochs_without_improvement":
                    epochs_without_improvement,

                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else []
                ),

                "num_parameters":
                    num_params,

                "seed": SEED,
                "config": {
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "epochs": NUM_EPOCHS,
                    "early_stop_patience": EARLY_STOP_PATIENCE,
                },
            },
            last_checkpoint
        )

        # --------------------------------------------------------
        # SAVE HISTORY
        # --------------------------------------------------------

        history_path = LOG_DIR / "conditioned_training_history.json"
        temporary_history = history_path.with_suffix(".json.tmp")
        with open(temporary_history, "w") as f:
            json.dump(history, f, indent=2)
        temporary_history.replace(history_path)

        print(
            f"Epochs without improvement: "
            f"{epochs_without_improvement}/{EARLY_STOP_PATIENCE}"
        )
        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(
                f"Early stopping at epoch {epoch}: "
                f"validation PSNR did not improve for "
                f"{EARLY_STOP_PATIENCE} consecutive epochs."
            )
            break

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
