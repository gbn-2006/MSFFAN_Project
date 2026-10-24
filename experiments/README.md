# Experiment Index

This directory contains isolated experiments that extend or investigate the original MSFFAN implementation.

## Experiment tracks

| Directory | Purpose | Documentation |
|---|---|---|
| `architecture_comparison_20261023/` | Architecture-comparison work, including FiLM and archived training scripts | [`README.md`](architecture_comparison_20261023/README.md) |
| `restart_20261021/` | Residual-refinement hypothesis, training runs, checkpoints, and evaluation artifacts | [`README.md`](restart_20261021/README.md) |

## Shared experimental rules

- Keep the original baseline and frozen dataset unchanged.
- Keep each experiment's code, checkpoints, logs, and outputs identifiable.
- Use training data for fitting and validation data for checkpoint selection.
- Reserve the frozen test set for final evaluation.
- Do not tune a model after examining its frozen-test results.
- Record metric definitions, preprocessing, checkpoint identity, and commands.
- Compare models only when the evaluation protocol is compatible.

## Status interpretation

The presence of an experiment directory means its artifacts were retained; it does not by itself establish that the model is reproducible, scientifically validated, or better than the baseline. Read the experiment's report and inspect its logs and result files before making claims.
