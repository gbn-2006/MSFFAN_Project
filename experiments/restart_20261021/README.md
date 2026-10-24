# Residual-Refinement Experiment — 2026-10-21

## Objective

Investigate whether a lightweight, input-conditioned reconstruction-refinement module can improve MSFFAN reconstruction under variable image scales and image content.

This is an experimental hypothesis, not a proven improvement or a validated claim of novelty.

## Immutable reference assets

- Original baseline checkpoint: `models/baseline_best.pth`
- Original baseline source: `scripts/model_msffan_baseline.py`
- Frozen dataset: `data/dataset/`
- Dataset split: 4,096 training, 500 validation, 500 frozen-test pairs
- Existing baseline, ASGF, and original test-evaluation artifacts must remain unchanged.

## Experimental controls

- Initialize compatible model parameters from the baseline checkpoint.
- Record missing or unexpected checkpoint keys and newly initialized layers.
- Select model configurations and checkpoints using validation data only.
- Evaluate the frozen test set only after model selection.
- Preserve per-image PSNR and SSIM results.
- Keep all new results within this experiment directory.
- Do not claim superiority without compatible comparisons and supporting evidence.

## Directory map

| Path | Purpose |
|---|---|
| `models/model_refinement.py` | Residual-refinement model definition |
| `scripts/train_refinement.py` | Training script for the initial run |
| `scripts/train_refinement_full.py` | Full training entry point |
| `scripts/train_refinement_smoke_backup.py` | Retained smoke-test/training script backup |
| `scripts/smoke_test_refinement.py` | Small integration/smoke test |
| `scripts/compare_refinement_validation.py` | Validation comparison |
| `scripts/evaluate_residual_gates.py` | Residual-gate evaluation |
| `scripts/final_gated_test_evaluation.py` | Final gated test evaluation |
| `checkpoints/` | Checkpoints from the initial run |
| `full_run_100_epochs/checkpoints/` | Checkpoints saved during the longer run |
| `full_run_100_epochs/results/` | Retained validation and test result files |
| `logs/training_history.jsonl` | Training history |
| `notes/experiment_plan.md` | Hypothesis, controls, and experiment plan |

## Important result files

- `full_run_100_epochs/results/validation_comparison.json`
- `full_run_100_epochs/results/validation_per_image.json`
- `full_run_100_epochs/results/residual_gate_experiment/summary.json`
- `full_run_100_epochs/results/residual_gate_experiment/per_image.json`
- `full_run_100_epochs/results/final_gated_test_alpha_025/summary.json`
- `full_run_100_epochs/results/final_gated_test_alpha_025/per_image.json`

The result filenames identify the experiment and split, but their scientific interpretation depends on the recorded configuration, checkpoint, and evaluation protocol. Inspect the JSON files and training logs before reporting numerical conclusions.

## Reproduction workflow

Run commands from the repository root. The scripts may require local dataset files and modules from `scripts/`.

1. Inspect `notes/experiment_plan.md` and the target script configuration.
2. Run `python experiments/restart_20261021/scripts/smoke_test_refinement.py`.
3. Check the dataset path, checkpoint initialization, and output directories.
4. Train using the intended training script only after confirming the configuration.
5. Compare candidate checkpoints on validation data.
6. Run the frozen-test evaluation only after model selection, and preserve the resulting artifacts.

Do not run a script blindly: verify its output paths first to avoid overwriting retained results.

## Interpretation and limitations

The recorded experiment plan lists baseline mean per-image PSNR of approximately 26.2406 dB and ASGF mean per-image PSNR of approximately 26.0871 dB, with corresponding SSIM values of approximately 0.838523 and 0.838064. Treat these as historical reference values from that plan; verify metric implementation and aggregation before comparing them with results from other experiment tracks.

Keep validation and test results distinct. Test results must not be used for model selection or hyperparameter tuning.
