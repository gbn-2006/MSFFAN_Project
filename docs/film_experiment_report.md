# FiLM-Conditioned MSFFAN Experiment Report

**Date:** 2026-10-23
**Status:** Training and frozen-test evaluation completed

## Objective

Evaluate whether FiLM conditioning using scale factor and spatial
position improves the original MSFFAN baseline for satellite-image
super-resolution.

## Training Configuration

- Training samples: 4096
- Validation samples: 500
- Frozen test samples: 500
- Batch size: 4
- Learning rate: 1e-4
- Random seed: 42
- Maximum epochs: 100
- Early-stopping patience: 15
- Best checkpoint epoch: 93
- Trainable parameters: 727,987
- Device: CUDA, NVIDIA GeForce RTX 3050 Laptop GPU

## Conditioning

The conditioning vector contains:

- Scale normalization: (scale_factor - 2) / 2
- Horizontal position: x / source_width
- Vertical position: y / source_height

The vector is used by the FiLM-conditioned architecture to modulate
feature representations.

## Best Validation Results

- Best validation PSNR: 24.350347 dB
- SSIM at the best epoch: 0.611383
- Best checkpoint: models/conditioned_best.pth
- Final checkpoint: models/conditioned_last.pth

## Frozen Test Evaluation

Both the original baseline and FiLM were evaluated on the same
500-image test split. Predictions and targets were clamped to [0, 1].
PSNR and scikit-image SSIM were calculated per image and averaged
over the test set.

| Metric | Original baseline | FiLM | Difference |
|---|---:|---:|---:|
| Mean PSNR (dB) | 26.240586 | 26.399753 | +0.159167 |
| Mean SSIM | 0.616336 | 0.622694 | +0.006358 |

PSNR improved on 407 images (81.4%) and regressed on 93 images (18.6%).

## Integrity Checks

- Training history contains 100 records.
- Best checkpoint corresponds to epoch 93.
- Final checkpoint corresponds to epoch 100.
- Both models passed a single-batch smoke test.
- The baseline checkpoint SHA-256 was verified before evaluation.
- All 500 test samples were processed.
- Saved summary and per-image results were verified.

### Baseline checkpoint SHA-256

`4f1843da5ad95cc9f4759e84cfc71d824b7f05227c9b448f7c004205faaca21c`

### FiLM checkpoint SHA-256

`3664376c0315fbf372a8cf4b59e09469315176d96c620b7667a6b43cb3c12c65`

## Conclusion

FiLM improved mean PSNR by 0.159167 dB and mean SSIM by 0.006358
over the original baseline under the recorded evaluation protocol.

This is a positive experimental result, not yet proof of statistical
significance or evidence that FiLM is the best architecture among all
planned variants. ASGF and HF-MSFFAN comparisons remain pending.
The frozen test set must not be used for further model tuning.

## Artifacts

- Training script: experiments/architecture_comparison_20261023/film/train_film_100ep.py
- Evaluator: experiments/architecture_comparison_20261023/film/evaluate_film.py
- Training history: experiments/architecture_comparison_20261023/film/logs/conditioned_training_history.json
- Training console log: experiments/architecture_comparison_20261023/film/logs/train_film_console.log
- Test summary: experiments/architecture_comparison_20261023/film/evaluation/film_test_summary.json
- Per-image results: experiments/architecture_comparison_20261023/film/evaluation/film_per_sample.json
- Evaluation console log: experiments/architecture_comparison_20261023/film/evaluation/evaluation_console.log
