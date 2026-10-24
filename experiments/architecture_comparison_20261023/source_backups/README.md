# Archived Training Scripts

This directory preserves copies of earlier training scripts for reference and experimental provenance.

| File | Historical track |
|---|---|
| `train_asgf.py` | ASGF adaptive-scale-gated fusion |
| `train_conditioned.py` | Conditioned MSFFAN training |
| `train_hf.py` | HF-MSFFAN training |

These files are source backups, not necessarily the current canonical entry points. Before executing one, inspect its imports, configuration, checkpoint-loading behavior, and output paths. In particular, verify that it cannot overwrite the original baseline or retained evaluation results.

The active FiLM experiment is documented separately in [`../film/README.md`](../film/README.md).
