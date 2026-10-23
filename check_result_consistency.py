
import json
from pathlib import Path
from statistics import mean

root = Path("results/test_evaluation")

with (root / "test_summary.json").open() as f:
    summary = json.load(f)

with (root / "per_sample_results.json").open() as f:
    rows = json.load(f)

print("Records:", len(rows))

for metric in ("psnr", "ssim"):
    baseline_key = f"baseline_{metric}"
    proposed_key = f"proposed_{metric}"

    baseline = mean(r[baseline_key] for r in rows)
    proposed = mean(r[proposed_key] for r in rows)
    delta = mean(
        r[proposed_key] - r[baseline_key]
        for r in rows
    )

    print(f"\n{metric.upper()}")
    print(f"Per-image baseline mean: {baseline:.9f}")
    print(f"Per-image proposed mean: {proposed:.9f}")
    print(f"Mean paired difference:  {delta:+.9f}")
    print(f"Summary baseline:        {summary['baseline'][metric]:.9f}")
    print(f"Summary proposed:        {summary['proposed'][metric]:.9f}")
    print(f"Summary difference:      {summary['difference'][metric]:+.9f}")
