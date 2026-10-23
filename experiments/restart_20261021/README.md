# MSFFAN Clean Experimental Workflow

Original baseline and frozen dataset must remain unchanged.

- Baseline checkpoint: models/baseline_best.pth
- Dataset: data/dataset/
- Dataset splits: train=4096, val=500, test=500
- Proposed model checkpoints: experiments/restart_20261021/checkpoints/
- Experiment results and logs: experiments/restart_20261021/
- Select model configurations using validation data only.
- Use the frozen test set for final comparison only.
- Never overwrite original baseline or test evaluation results.
