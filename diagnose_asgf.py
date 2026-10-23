
import json
import math
import random
from collections import defaultdict
from pathlib import Path

path = Path("results/test_evaluation/per_sample_results.json")
with path.open() as f:
    records = json.load(f)

assert len(records) == 500, f"Expected 500 samples, got {len(records)}"

ids = [r["sample_id"] for r in records]
assert len(set(ids)) == len(ids), "Duplicate sample IDs found"

def summarize(rows, label):
    dp = [r["proposed_psnr"] - r["baseline_psnr"] for r in rows]
    ds = [r["proposed_ssim"] - r["baseline_ssim"] for r in rows]

    print(f"\n{label} (n={len(rows)})")
    print(f"  Mean PSNR change: {sum(dp)/len(dp):+.5f} dB")
    print(f"  Mean SSIM change: {sum(ds)/len(ds):+.6f}")
    print(f"  PSNR wins / ties / losses: "
          f"{sum(x > 1e-9 for x in dp)} / "
          f"{sum(abs(x) <= 1e-9 for x in dp)} / "
          f"{sum(x < -1e-9 for x in dp)}")
    print(f"  SSIM wins / ties / losses: "
          f"{sum(x > 1e-9 for x in ds)} / "
          f"{sum(abs(x) <= 1e-9 for x in ds)} / "
          f"{sum(x < -1e-9 for x in ds)}")

    return dp

# Validate numeric metrics.
for r in records:
    for key in ("baseline_psnr", "proposed_psnr",
                "baseline_ssim", "proposed_ssim"):
        assert math.isfinite(r[key]), (
            f"Invalid {key} in {r['sample_id']}"
        )

print("===== OVERALL PAIRED COMPARISON =====")
dp = summarize(records, "All test samples")

print("\n===== BY GEOGRAPHIC LOCATION =====")
locations = defaultdict(list)
for r in records:
    locations[r["location"]].append(r)

for loc, rows in sorted(locations.items()):
    summarize(rows, loc)

print("\n===== BY SCALE FACTOR =====")
scales = defaultdict(list)
for r in records:
    scales[r["scale_factor"]].append(r)

for scale, rows in sorted(scales.items()):
    summarize(rows, f"{scale}x super-resolution")

print("\n===== LOCATION x SCALE =====")
groups = defaultdict(list)
for r in records:
    groups[(r["location"], r["scale_factor"])].append(r)

for (loc, scale), rows in sorted(groups.items()):
    summarize(rows, f"{loc}, {scale}x")

# Exploratory paired bootstrap confidence interval for mean PSNR change.
random.seed(42)
n = len(dp)
boot_means = []
for _ in range(10000):
    sample = [dp[random.randrange(n)] for _ in range(n)]
    boot_means.append(sum(sample) / n)

boot_means.sort()
print("\n===== EXPLORATORY UNCERTAINTY =====")
print(f"Mean PSNR change: {sum(dp)/n:+.5f} dB")
print("Approx. 95% paired-bootstrap CI: "
      f"[{boot_means[249]:+.5f}, {boot_means[9749]:+.5f}] dB")
print("Note: this interval assumes test samples can be resampled "
      "independently; spatially overlapping patches may violate that assumption.")

import json
import math
import random
from collections import defaultdict
from pathlib import Path

path = Path("results/test_evaluation/per_sample_results.json")
with path.open() as f:
    records = json.load(f)

assert len(records) == 500, f"Expected 500 samples, got {len(records)}"

ids = [r["sample_id"] for r in records]
assert len(set(ids)) == len(ids), "Duplicate sample IDs found"

def summarize(rows, label):
    dp = [r["proposed_psnr"] - r["baseline_psnr"] for r in rows]
    ds = [r["proposed_ssim"] - r["baseline_ssim"] for r in rows]

    print(f"\n{label} (n={len(rows)})")
    print(f"  Mean PSNR change: {sum(dp)/len(dp):+.5f} dB")
    print(f"  Mean SSIM change: {sum(ds)/len(ds):+.6f}")
    print(f"  PSNR wins / ties / losses: "
          f"{sum(x > 1e-9 for x in dp)} / "
          f"{sum(abs(x) <= 1e-9 for x in dp)} / "
          f"{sum(x < -1e-9 for x in dp)}")
    print(f"  SSIM wins / ties / losses: "
          f"{sum(x > 1e-9 for x in ds)} / "
          f"{sum(abs(x) <= 1e-9 for x in ds)} / "
          f"{sum(x < -1e-9 for x in ds)}")

    return dp

# Validate numeric metrics.
for r in records:
    for key in ("baseline_psnr", "proposed_psnr",
                "baseline_ssim", "proposed_ssim"):
        assert math.isfinite(r[key]), (
            f"Invalid {key} in {r['sample_id']}"
        )

print("===== OVERALL PAIRED COMPARISON =====")
dp = summarize(records, "All test samples")

print("\n===== BY GEOGRAPHIC LOCATION =====")
locations = defaultdict(list)
for r in records:
    locations[r["location"]].append(r)

for loc, rows in sorted(locations.items()):
    summarize(rows, loc)

print("\n===== BY SCALE FACTOR =====")
scales = defaultdict(list)
for r in records:
    scales[r["scale_factor"]].append(r)

for scale, rows in sorted(scales.items()):
    summarize(rows, f"{scale}x super-resolution")

print("\n===== LOCATION x SCALE =====")
groups = defaultdict(list)
for r in records:
    groups[(r["location"], r["scale_factor"])].append(r)

for (loc, scale), rows in sorted(groups.items()):
    summarize(rows, f"{loc}, {scale}x")

# Exploratory paired bootstrap confidence interval for mean PSNR change.
random.seed(42)
n = len(dp)
boot_means = []
for _ in range(10000):
    sample = [dp[random.randrange(n)] for _ in range(n)]
    boot_means.append(sum(sample) / n)

boot_means.sort()
print("\n===== EXPLORATORY UNCERTAINTY =====")
print(f"Mean PSNR change: {sum(dp)/n:+.5f} dB")
print("Approx. 95% paired-bootstrap CI: "
      f"[{boot_means[249]:+.5f}, {boot_means[9749]:+.5f}] dB")
print("Note: this interval assumes test samples can be resampled "
      "independently; spatially overlapping patches may violate that assumption.")
