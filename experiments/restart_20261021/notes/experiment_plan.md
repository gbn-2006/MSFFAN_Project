# MSFFAN Isolated Experiment — 2026-10-21

## Objective
Test one lightweight reconstruction-refinement hypothesis while
preserving the existing MSFFAN baseline and ASGF experiment.

## Immutable assets
- Existing dataset and split assignments
- Existing baseline checkpoint
- Existing ASGF checkpoint
- Existing original test-evaluation results

## Dataset
- Train: 4096
- Validation: 500
- Frozen test: 500
- Do not regenerate or repartition the dataset.

## Existing evidence
- Baseline mean per-image PSNR: approximately 26.2406 dB
- Baseline mean per-image SSIM: approximately 0.838523
- ASGF mean per-image PSNR: approximately 26.0871 dB
- ASGF mean per-image SSIM: approximately 0.838064

These values are reference results; preserve their original files.

## Proposed hypothesis
A lightweight input-conditioned residual-refinement module may improve
reconstruction under variable image scales and content.

This is an unvalidated hypothesis, not a proven novelty or improvement.

## Experimental controls
- Initialize compatible model parameters from the baseline checkpoint.
- Record missing/unexpected checkpoint keys and newly initialized layers.
- Select checkpoints using validation data only.
- Evaluate the frozen test set only after model selection.
- Calculate PSNR and SSIM per image.
- Save all new results outside the original results directory.
- Report aggregate results and location/scale breakdowns.
- Do not claim superiority unless supported by results.
