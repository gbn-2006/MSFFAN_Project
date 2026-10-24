# FiLM-Conditioned MSFFAN Experiment

## Objective

Evaluate an MSFFAN model conditioned on image-scale and spatial-location metadata. This is an additional experiment; it does not replace the original baseline implementation.

## Model conditioning

The recorded conditioning vector contains three normalized values:

- `scale_norm = (scale_factor - 2) / 2`
- normalized horizontal position: `x / source_width`
- normalized vertical position: `y / source_height`

The source-dimension mapping used by the experiment is:

| Location | Source dimensions (height, width) |
|---|---:|
| San Francisco | `(1345, 5574)` |
| Baltimore | `(5567, 7086)` |
| Denver | `(5566, 7135)` |

Check the training and evaluation scripts for the exact metadata fields and coordinate conventions before reproducing the run.

## Recorded configuration

- Maximum epochs: 100
- Batch size: 4
- Learning rate: `1e-4`
- Random seed: 42
- Early-stopping patience: 15
- Recorded best epoch: 93
- Trainable parameters: 727,987
- Training split: 4,096 pairs
- Validation split: 500 pairs
- Frozen test split: 500 pairs

## Entry points

Run commands from the repository root.

Training:

```bash
python experiments/architecture_comparison_20261023/film/train_film_100ep.py
```

Evaluation:

```bash
python experiments/architecture_comparison_20261023/film/evaluate_film.py
```

These commands assume the project's Python environment, dataset, and dependencies are already configured. Inspect each script's configuration and paths before rerunning it. Do not rerun evaluation if doing so would overwrite the retained results.

## Artifact inventory

| Path | Description |
|---|---|
| `train_film_100ep.py` | Training entry point |
| `evaluate_film.py` | Frozen-test evaluation entry point |
| `models/conditioned_best.pth` | Best saved checkpoint |
| `models/conditioned_last.pth` | Last saved checkpoint |
| `logs/conditioned_training_history.json` | Epoch-level training history |
| `logs/train_film_console.log` | Training console log |
| `evaluation/film_test_summary.json` | Aggregate test metrics |
| `evaluation/film_per_sample.json` | Per-image results for the 500 test samples |
| `evaluation/evaluation_console.log` | Evaluation console log |

## Recorded results

| Metric | Baseline | FiLM | Difference |
|---|---:|---:|---:|
| Mean PSNR | 26.2406 dB | 26.3998 dB | +0.1592 dB |
| Mean SSIM | 0.6163 | 0.6227 | +0.0064 |

PSNR improved on 407 test images and regressed on 93. These are descriptive results for this recorded evaluation; they do not establish statistical significance or generalization to other datasets.

Best validation metrics recorded during training:

- PSNR: 24.3503 dB
- SSIM: 0.6114

Read [`docs/film_experiment_report.md`](../../../docs/film_experiment_report.md) for the detailed report and artifact provenance.

## Reproducibility notes

- Preserve the original baseline checkpoint and source.
- Use validation data for checkpoint selection.
- Do not use frozen-test results to change the model or hyperparameters.
- Keep the saved per-image results alongside aggregate metrics.
- Verify that evaluation preprocessing and metric definitions match the intended comparison protocol before comparing against other experiment tracks.
