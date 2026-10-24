# Architecture Comparison — 2026-10-23

## Purpose

This directory isolates the FiLM-conditioned MSFFAN experiment from the existing model scripts and checkpoints. Archived training scripts are retained under `source_backups/` for reference.

## Directory map

| Path | Contents |
|---|---|
| `film/` | FiLM training/evaluation scripts, checkpoints, logs, and recorded test results |
| `source_backups/` | Archived ASGF, conditioned-model, and HF-MSFFAN training scripts |

## FiLM experiment

The FiLM track uses normalized scale and spatial-location metadata as conditioning inputs. Its training and evaluation details, recorded metrics, and artifact paths are documented in [`film/README.md`](film/README.md) and [`docs/film_experiment_report.md`](../../docs/film_experiment_report.md).

## Comparison requirements

Before ranking architectures, verify that they use:

1. The same frozen dataset split.
2. The same preprocessing and image alignment.
3. The same test samples.
4. The same metric implementations and aggregation method.
5. Clearly identified checkpoints and model configurations.

A metric reported by one experiment should not automatically be compared with a metric from another experiment if these conditions have not been verified.

## Source backups

Files in `source_backups/` are historical script copies. They are not automatically the canonical training entry points. See [`source_backups/README.md`](source_backups/README.md).
