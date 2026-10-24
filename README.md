# MSFFAN: Multi-Scale Feature Fusion Attention Network

**Research project:** Enhancing edge super-resolution of variable-resolution optical satellite images.

## 1. Project overview

This project implements and evaluates an MSFFAN baseline for optical satellite-image super-resolution and investigates additional architectural approaches. The original baseline is retained as the reference model; proposed changes are evaluated as separate experiments.

The project includes satellite-data preparation, image alignment, model training, validation, evaluation, and experimental diagnostics.

## 2. Research objectives

- Implement and preserve the original MSFFAN baseline.
- Investigate additional feature-fusion, conditioning, and reconstruction-refinement approaches.
- Compare models using consistent preprocessing and evaluation protocols.
- Preserve experiment configurations, checkpoints, logs, and per-image results for reproducibility.
- Report limitations and avoid claiming an improvement without supporting evidence.

## 3. Dataset and evaluation protocol

The frozen dataset split is:

| Split | Images/pairs |
|---|---:|
| Training | 4,096 |
| Validation | 500 |
| Test | 500 |
| **Total** | **5,096** |

The data covers San Francisco, Baltimore, and Denver. The frozen split must not be regenerated during model comparison.

Use training data for fitting and validation data for model selection. Reserve the frozen test set for final evaluation; do not use test results to tune architecture or hyperparameters.

## 4. Repository guide

| Path | Purpose |
|---|---|
| `scripts/` | Dataset preparation, model definitions, training, evaluation, and diagnostics |
| `models/` | Existing model checkpoints, including the original baseline |
| `data/` | Local dataset and source data; not intended for wholesale Git tracking |
| `results/` | Existing evaluation outputs |
| `logs/` | Existing run logs |
| `experiments/` | Isolated architecture and residual-refinement experiments |
| `docs/` | Detailed experiment reports |

See [`experiments/README.md`](experiments/README.md) for the experiment index.

## 5. Current experiment records

### FiLM-conditioned MSFFAN

The FiLM experiment conditions the model on normalized scale and spatial-location metadata. Its training script, evaluation script, checkpoints, logs, and test-evaluation artifacts are stored in [`experiments/architecture_comparison_20261023/film/`](experiments/architecture_comparison_20261023/film/).

- Best validation PSNR: 24.3503 dB (epoch 93).
- Recorded frozen-test mean PSNR: 26.3998 dB.
- Recorded frozen-test mean SSIM: 0.6227.
- Baseline comparison in the same FiLM evaluation: PSNR 26.2406 dB; SSIM 0.6163.

These values are experiment-specific. Confirm that metrics, preprocessing, and evaluation implementations are comparable before drawing conclusions across separate experiment tracks.

See the [FiLM experiment report](docs/film_experiment_report.md) and [FiLM README](experiments/architecture_comparison_20261023/film/README.md).

### Residual-refinement experiment

The isolated residual-refinement experiment and its evaluation artifacts are stored in [`experiments/restart_20261021/`](experiments/restart_20261021/). Its hypothesis and experimental controls are documented in that directory. Treat its results as a separate experiment, not as proof of improvement.

### Other architecture tracks

ASGF and HF-MSFFAN scripts and related diagnostic utilities are retained in the repository. Consult the experiment documentation and result files before making comparative claims.

## 6. Environment and reproducibility

The project was developed in a Linux environment under WSL with Python and PyTorch. Exact package versions and commands should be recorded with each experiment.

Before running a training or evaluation script:

1. Inspect its configuration and output paths.
2. Confirm that it uses the intended dataset split.
3. Confirm that it will not overwrite the original baseline or existing evaluation artifacts.
4. Run syntax checks and a small smoke test.
5. Record the command, configuration, checkpoint, and output location.

Training scripts may depend on modules in `scripts/` and on local dataset files that are not stored in Git. Follow the experiment-specific README before running them.

## 7. Scientific reporting principles

- Preserve the original baseline and frozen dataset.
- Keep proposed architectures and their artifacts separate.
- Select models using validation results, not test results.
- Retain per-image metrics and experiment logs.
- Report regressions and limitations as well as improvements.
- Do not claim statistical significance without an appropriate analysis.

## 8. Version control

Git tracks source code, documentation, selected checkpoints, and experimental evidence. Large local datasets and virtual environments should remain outside version control.

See `.gitignore` for the repository's tracking rules.
