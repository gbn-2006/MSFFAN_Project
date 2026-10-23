
import json
from pathlib import Path

path = Path("results/test_evaluation/per_sample_results.json")

with path.open() as f:
    rows = json.load(f)

assert len(rows) == 500, f"Expected 500 rows, found {len(rows)}"

print("Total samples:", len(rows))

print("\nSamples by location and scale:")
groups = {}
for r in rows:
    key = (r["location"], r["scale_factor"])
    groups[key] = groups.get(key, 0) + 1

for key, count in sorted(groups.items()):
    print(f"{key[0]:16s} scale={key[1]}x: {count}")

print("\n10 largest PSNR regressions:")
worst = sorted(
    rows,
    key=lambda r: r["proposed_psnr"] - r["baseline_psnr"]
)[:10]

for r in worst:
    delta = r["proposed_psnr"] - r["baseline_psnr"]
    print(
        f'{r["sample_id"]:15s} '
        f'{r["location"]:16s} '
        f'scale={r["scale_factor"]}x '
        f'delta={delta:+.4f} dB '
        f'baseline={r["baseline_psnr"]:.4f} '
        f'ASGF={r["proposed_psnr"]:.4f}'
    )

print("\n10 largest PSNR improvements:")
best = sorted(
    rows,
    key=lambda r: r["proposed_psnr"] - r["baseline_psnr"],
    reverse=True
)[:10]

for r in best:
    delta = r["proposed_psnr"] - r["baseline_psnr"]
    print(
        f'{r["sample_id"]:15s} '
        f'{r["location"]:16s} '
        f'scale={r["scale_factor"]}x '
        f'delta={delta:+.4f} dB '
        f'baseline={r["baseline_psnr"]:.4f} '
        f'ASGF={r["proposed_psnr"]:.4f}'
    )
