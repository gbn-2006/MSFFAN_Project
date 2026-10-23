import sys
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity

ROOT = Path("/mnt/c/MSFFAN_Project")
SCRIPTS = ROOT / "scripts"
EXP = ROOT / "experiments" / "restart_20261021"
RUN = EXP / "full_run_100_epochs"
CHECKPOINTS = RUN / "checkpoints"
LOGS = RUN / "logs"
RESULTS = RUN / "results"
MODELS = EXP / "models"

BASELINE_PATH = ROOT / "models" / "baseline_best.pth"
BEST_PATH = CHECKPOINTS / "refinement_best.pth"
LAST_PATH = CHECKPOINTS / "refinement_last.pth"
HISTORY_PATH = LOGS / "training_history.jsonl"
SNAPSHOT_EVERY = 5
EARLY_STOP_PATIENCE = 15
RESUME = True

BATCH_SIZE = 4
NUM_WORKERS = 2
LEARNING_RATE = 1e-4
EPOCHS = 100
SEED = 42
SMOKE_TEST = False
SMOKE_TRAIN_BATCHES = 2
SMOKE_VAL_BATCHES = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for directory in (CHECKPOINTS, LOGS, RESULTS):
    directory.mkdir(parents=True, exist_ok=True)

# Resume only when a recovery checkpoint exists. Never overwrite an
# existing history without the corresponding recovery checkpoint.
if HISTORY_PATH.exists() and not LAST_PATH.exists():
    raise FileExistsError(
        "History exists without a latest checkpoint. Preserve this run "
        "and investigate before starting training."
    )
if BEST_PATH.exists() and not LAST_PATH.exists():
    raise FileExistsError(
        "Best checkpoint exists without a latest checkpoint. "
        "Preserve this run and investigate first."
    )
if LAST_PATH.exists() and not RESUME:
    raise FileExistsError(
        "Latest checkpoint exists but RESUME=False. Choose a new run "
        "directory or explicitly enable resumption."
    )

sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(MODELS))

from dataset_loader import MSFFANDataset
from model_msffan_baseline import MSFFANBaselineModel
from model_loss import MSFFANLoss
from model_refinement import InputConditionedResidualRefinement


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def atomic_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def per_image_metrics(prediction, target):
    """Return PSNR and SSIM lists, one value per image."""
    prediction = prediction.detach().clamp(0, 1).cpu().numpy()
    target = target.detach().clamp(0, 1).cpu().numpy()

    psnr_values = []
    ssim_values = []

    for pred, truth in zip(prediction, target):
        mse = float(np.mean((pred - truth) ** 2))
        psnr = float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)

        ssim = structural_similarity(
            truth,
            pred,
            channel_axis=0,
            data_range=1.0,
        )

        psnr_values.append(float(psnr))
        ssim_values.append(float(ssim))

    return psnr_values, ssim_values


def make_loader(split, shuffle, generator=None):
    dataset = MSFFANDataset(project_root=ROOT, split=split)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    return dataset, loader


def run_epoch(baseline, refinement, loader, criterion, optimizer=None):
    training = optimizer is not None
    refinement.train(training)
    baseline.eval()

    loss_sum = 0.0
    pixel_sum = 0.0
    edge_sum = 0.0
    batch_count = 0
    psnr_values = []
    ssim_values = []

    for batch_index, batch in enumerate(loader, start=1):
        if SMOKE_TEST and batch_index > (SMOKE_TRAIN_BATCHES if training else SMOKE_VAL_BATCHES):
            break
        lr = batch["lr"].to(DEVICE, non_blocking=True)
        hr = batch["hr"].to(DEVICE, non_blocking=True)

        # Baseline is frozen: only its predictions are needed.
        with torch.no_grad():
            baseline_prediction = baseline(lr)

        if training:
            optimizer.zero_grad(set_to_none=True)

        prediction = refinement(lr, baseline_prediction)
        total, pixel, edge = criterion(prediction, hr)

        if training:
            total.backward()
            optimizer.step()

        if not torch.isfinite(total):
            raise FloatingPointError(
                f"Non-finite loss at batch {batch_index}"
            )

        loss_sum += total.item()
        pixel_sum += pixel.item()
        edge_sum += edge.item()
        batch_count += 1

        batch_psnr, batch_ssim = per_image_metrics(prediction, hr)
        psnr_values.extend(batch_psnr)
        ssim_values.extend(batch_ssim)

        if batch_index == 1 or batch_index % 100 == 0:
            mode = "TRAIN" if training else "VAL"
            print(
                f"{mode} batch {batch_index}/{len(loader)} "
                f"| loss={total.item():.6f}",
                flush=True,
            )

    if batch_count == 0:
        raise RuntimeError("No batches were processed.")

    return {
        "loss": loss_sum / batch_count,
        "pixel": pixel_sum / batch_count,
        "edge": edge_sum / batch_count,
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)),
        "samples": len(psnr_values),
    }


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    original_hash = sha256(BASELINE_PATH)

    # Verify the expected protected baseline checkpoint.
    expected_hash = (
        "4f1843da5ad95cc9f4759e84cfc71d824b7f05227c9b448f7c004205faaca21c"
    )
    if original_hash != expected_hash:
        raise RuntimeError(
            "Baseline checkpoint hash differs from the recorded value. "
            "Stop and investigate; no training was started."
        )

    print("=" * 72)
    print("ISOLATED MSFFAN REFINEMENT TRAINING")
    print("=" * 72)
    print("Device:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Baseline SHA-256:", original_hash)
    print(f"Batch size={BATCH_SIZE}, epochs={EPOCHS}, LR={LEARNING_RATE}")

    train_generator = torch.Generator()
    train_generator.manual_seed(SEED)
    val_generator = torch.Generator()
    val_generator.manual_seed(SEED + 1)

    train_dataset, train_loader = make_loader("train", True, train_generator)
    val_dataset, val_loader = make_loader("val", False, val_generator)

    if len(train_dataset) != 4096 or len(val_dataset) != 500:
        raise RuntimeError(
            f"Unexpected split sizes: train={len(train_dataset)}, "
            f"val={len(val_dataset)}. Expected 4096 and 500."
        )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    baseline = MSFFANBaselineModel().to(DEVICE)
    checkpoint = torch.load(
        BASELINE_PATH, map_location=DEVICE, weights_only=False
    )
    baseline.load_state_dict(checkpoint["model_state_dict"], strict=True)
    baseline.eval()

    for parameter in baseline.parameters():
        parameter.requires_grad_(False)

    refinement = InputConditionedResidualRefinement().to(DEVICE)
    criterion = MSFFANLoss().to(DEVICE)

    optimizer = torch.optim.Adam(
        refinement.parameters(),
        lr=LEARNING_RATE,
    )

    best_psnr = -float("inf")
    start_epoch = 1
    epochs_without_improvement = 0

    if LAST_PATH.exists() and RESUME:
        saved = torch.load(LAST_PATH, map_location=DEVICE, weights_only=False)

        # Refuse to resume from a different baseline or incompatible run.
        if saved.get("baseline_sha256") != original_hash:
            raise RuntimeError("Resume checkpoint baseline hash mismatch.")
        config = saved.get("config", {})
        expected_config = {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "seed": SEED,
            "baseline_frozen": True,
        }
        for key, expected in expected_config.items():
            if config.get(key) != expected:
                raise RuntimeError(
                    f"Resume configuration mismatch for {key}: "
                    f"checkpoint={config.get(key)!r}, expected={expected!r}"
                )
        if "optimizer_state_dict" not in saved:
            raise RuntimeError("Resume checkpoint lacks optimizer state.")
        if "model_state_dict" not in saved:
            raise RuntimeError("Resume checkpoint lacks model state.")

        refinement.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        start_epoch = int(saved["epoch"]) + 1
        best_psnr = float(saved.get("best_val_psnr", -float("inf")))
        epochs_without_improvement = int(saved.get("epochs_without_improvement", 0))
        if "torch_rng_state" in saved:
            torch.set_rng_state(saved["torch_rng_state"].cpu())
        if torch.cuda.is_available() and "cuda_rng_state_all" in saved:
            torch.cuda.set_rng_state_all(saved["cuda_rng_state_all"])
        if "numpy_rng_state" in saved:
            np.random.set_state(saved["numpy_rng_state"])
        if "train_generator_state" in saved:
            train_generator.set_state(saved["train_generator_state"].cpu())
        if "val_generator_state" in saved:
            val_generator.set_state(saved["val_generator_state"].cpu())
        print(f"Resuming after epoch {saved['epoch']}; best PSNR={best_psnr:.4f} dB")
    elif HISTORY_PATH.exists():
        raise FileExistsError("History already exists; refusing to restart from epoch 1.")

    if start_epoch > EPOCHS:
        print(f"Training already reached the configured maximum of {EPOCHS} epochs.")
        return

    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\n{'=' * 24} EPOCH {epoch}/{EPOCHS} {'=' * 24}")

        train_metrics = run_epoch(
            baseline, refinement, train_loader, criterion, optimizer
        )
        val_metrics = run_epoch(
            baseline, refinement, val_loader, criterion
        )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
            "learning_rate": LEARNING_RATE,
        }

        with open(HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        print(
            f"Train loss={train_metrics['loss']:.6f} "
            f"| Val loss={val_metrics['loss']:.6f}"
        )
        print(
            f"Val mean per-image PSNR={val_metrics['psnr']:.4f} dB "
            f"| SSIM={val_metrics['ssim']:.6f}"
        )

        payload = {
            "epoch": epoch,
            "model_state_dict": refinement.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "best_val_psnr": max(best_psnr, val_metrics["psnr"]),
            "epochs_without_improvement": epochs_without_improvement,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy_rng_state": np.random.get_state(),
            "train_generator_state": train_generator.get_state(),
            "val_generator_state": val_generator.get_state(),
            "baseline_checkpoint": str(BASELINE_PATH),
            "baseline_sha256": original_hash,
            "config": {
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
                "seed": SEED,
                "baseline_frozen": True,
            },
        }

        improved = val_metrics["psnr"] > best_psnr
        if improved:
            best_psnr = val_metrics["psnr"]
            epochs_without_improvement = 0
            payload["best_val_psnr"] = best_psnr
            payload["epochs_without_improvement"] = 0
            atomic_save(payload, BEST_PATH)
            print("Saved new best validation checkpoint.")
        else:
            epochs_without_improvement += 1

        payload["best_val_psnr"] = best_psnr
        payload["epochs_without_improvement"] = epochs_without_improvement
        atomic_save(payload, LAST_PATH)

        if epoch % SNAPSHOT_EVERY == 0:
            snapshot = CHECKPOINTS / f"epoch_{epoch:03d}.pth"
            atomic_save(payload, snapshot)
            print(f"Saved periodic checkpoint: {snapshot.name}")

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"Early stopping: validation PSNR did not improve for "
                  f"{EARLY_STOP_PATIENCE} consecutive epochs.")
            break

    final_hash = sha256(BASELINE_PATH)
    if final_hash != original_hash:
        raise RuntimeError("Baseline checkpoint changed during training!")

    print("\nTraining complete.")
    print(f"Best validation PSNR: {best_psnr:.4f} dB")
    print("Best checkpoint:", BEST_PATH)
    print("Last checkpoint:", LAST_PATH)
    print("History:", HISTORY_PATH)
    print("Baseline checkpoint hash unchanged:", final_hash == original_hash)
    print("SMOKE_TEST:", SMOKE_TEST)


if __name__ == "__main__":
    main()
