"""
MSFFAN Dataset - Constant Image Diagnostic
============================================

Purpose:
    Diagnose the 1,428 constant images detected during IMAGE INTEGRITY QC.

Checks:
    1. Constant HR images
    2. Constant LR images
    3. Constant images by split
    4. Constant images by location
    5. Constant images by scale factor
    6. HR/LR pair relationship
    7. Pixel statistics of problematic images
    8. Metadata lookup for problematic samples

IMPORTANT:
    This script DOES NOT modify or delete the dataset.
"""

from pathlib import Path
from collections import Counter, defaultdict
import re

import numpy as np
from PIL import Image
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "data" / "dataset"
METADATA_FILE = DATASET_ROOT / "dataset_metadata.csv"

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = LOG_DIR / "constant_image_diagnostic.txt"


# ============================================================
# CONFIGURATION
# ============================================================

SPLITS = ["train", "val", "test"]
IMAGE_TYPES = ["HR", "LR"]

# A truly constant image has only one unique pixel value.
# We also calculate standard deviation so we can see how severe
# the problem is.
CONSTANT_STD_THRESHOLD = 0.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_filename(filename):
    """
    Parse filenames such as:

        train_00009_baltimore_x1180_y2441_x3.png

    Returns:
        split
        location
        x
        y
        scale
    """

    pattern = (
        r"^(train|val|test)_"
        r"(\d+)_"
        r"(.+?)_"
        r"x(\d+)_"
        r"y(\d+)_"
        r"x(\d+)\.png$"
    )

    match = re.match(pattern, filename)

    if not match:
        return {
            "split": None,
            "location": None,
            "x": None,
            "y": None,
            "scale": None,
        }

    split, sample_id, location, x, y, scale = match.groups()

    return {
        "split": split,
        "sample_id": int(sample_id),
        "location": location,
        "x": int(x),
        "y": int(y),
        "scale": int(scale),
    }


def analyze_image(path):
    """
    Read an image and calculate statistics.
    """

    try:
        with Image.open(path) as img:
            arr = np.asarray(img)

        # Convert grayscale to 3D for consistency
        if arr.ndim == 2:
            arr = arr[:, :, None]

        finite = np.isfinite(arr).all()

        min_value = float(arr.min())
        max_value = float(arr.max())
        mean_value = float(arr.mean())
        std_value = float(arr.std())

        unique_values = np.unique(arr)

        return {
            "ok": True,
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "min": min_value,
            "max": max_value,
            "mean": mean_value,
            "std": std_value,
            "unique_values": len(unique_values),
            "finite": finite,
            "constant": (
                std_value <= CONSTANT_STD_THRESHOLD
                or len(unique_values) == 1
            ),
        }

    except Exception as e:

        return {
            "ok": False,
            "error": str(e),
        }


def find_image(split, image_type, filename):
    """
    Find an image inside:

        data/dataset/<split>/<HR|LR>/<filename>
    """

    path = DATASET_ROOT / split / image_type / filename

    if path.exists():
        return path

    return None


# ============================================================
# LOAD METADATA
# ============================================================

print("=" * 70)
print("MSFFAN CONSTANT IMAGE DIAGNOSTIC")
print("=" * 70)

print(f"Project root : {PROJECT_ROOT}")
print(f"Dataset root : {DATASET_ROOT}")
print(f"Metadata     : {METADATA_FILE}")
print()

if not METADATA_FILE.exists():

    print("ERROR: dataset_metadata.csv not found.")
    print(f"Expected: {METADATA_FILE}")
    raise SystemExit(1)


metadata = pd.read_csv(METADATA_FILE)

print("Metadata loaded successfully.")
print(f"Metadata rows: {len(metadata)}")
print()

print("Metadata columns:")
for column in metadata.columns:
    print(f"  - {column}")

print()


# ============================================================
# STORAGE
# ============================================================

constant_records = []
all_records = []

summary_split = defaultdict(lambda: Counter())
summary_location = defaultdict(lambda: Counter())
summary_scale = defaultdict(lambda: Counter())
summary_type = Counter()

pair_records = []


# ============================================================
# SCAN DATASET
# ============================================================

print("=" * 70)
print("SCANNING IMAGES")
print("=" * 70)

for split in SPLITS:

    for image_type in IMAGE_TYPES:

        folder = DATASET_ROOT / split / image_type

        if not folder.exists():

            print(f"WARNING: Missing folder: {folder}")
            continue

        files = sorted(folder.glob("*.png"))

        print(
            f"{split.upper():5s} | "
            f"{image_type:2s} | "
            f"{len(files):5d} images"
        )

        for image_path in files:

            filename = image_path.name

            parsed = parse_filename(filename)

            stats = analyze_image(image_path)

            record = {
                "split": split,
                "image_type": image_type,
                "filename": filename,
                "path": str(image_path),
            }

            record.update(parsed)
            record.update(stats)

            all_records.append(record)

            # ------------------------------------------------
            # IMAGE READ ERROR
            # ------------------------------------------------

            if not stats["ok"]:

                summary_type["read_error"] += 1

                constant_records.append(record)

                continue

            # ------------------------------------------------
            # CONSTANT IMAGE
            # ------------------------------------------------

            if stats["constant"]:

                summary_type["constant"] += 1

                summary_split[split]["constant"] += 1

                if parsed["location"] is not None:
                    summary_location[
                        parsed["location"]
                    ]["constant"] += 1

                if parsed["scale"] is not None:
                    summary_scale[
                        parsed["scale"]
                    ]["constant"] += 1

                constant_records.append(record)

            else:

                summary_split[split]["valid"] += 1

                if parsed["location"] is not None:
                    summary_location[
                        parsed["location"]
                    ]["valid"] += 1

                if parsed["scale"] is not None:
                    summary_scale[
                        parsed["scale"]
                    ]["valid"] += 1


print()


# ============================================================
# BASIC SUMMARY
# ============================================================

print("=" * 70)
print("1. OVERALL SUMMARY")
print("=" * 70)

total_images = len(all_records)
constant_count = len(
    [
        r for r in all_records
        if r.get("constant", False)
    ]
)

read_errors = len(
    [
        r for r in all_records
        if not r.get("ok", False)
    ]
)

print(f"Total images scanned : {total_images}")
print(f"Constant images      : {constant_count}")
print(f"Read errors          : {read_errors}")

if total_images > 0:

    percentage = 100.0 * constant_count / total_images

    print(
        f"Constant percentage  : {percentage:.2f}%"
    )

print()


# ============================================================
# SPLIT SUMMARY
# ============================================================

print("=" * 70)
print("2. CONSTANT IMAGES BY SPLIT")
print("=" * 70)

for split in SPLITS:

    records = [
        r for r in all_records
        if r["split"] == split
    ]

    total = len(records)

    constants = sum(
        r.get("constant", False)
        for r in records
    )

    print(
        f"{split.upper():5s} | "
        f"Total: {total:5d} | "
        f"Constant: {constants:5d} | "
        f"Percentage: "
        f"{(100 * constants / total if total else 0):6.2f}%"
    )

print()


# ============================================================
# HR VS LR
# ============================================================

print("=" * 70)
print("3. HR VS LR CONSTANT IMAGES")
print("=" * 70)

for image_type in IMAGE_TYPES:

    records = [
        r for r in all_records
        if r["image_type"] == image_type
    ]

    total = len(records)

    constants = sum(
        r.get("constant", False)
        for r in records
    )

    print(
        f"{image_type:2s} | "
        f"Total: {total:5d} | "
        f"Constant: {constants:5d} | "
        f"Percentage: "
        f"{(100 * constants / total if total else 0):6.2f}%"
    )

print()


# ============================================================
# LOCATION ANALYSIS
# ============================================================

print("=" * 70)
print("4. CONSTANT IMAGES BY LOCATION")
print("=" * 70)

locations = sorted(
    set(
        r["location"]
        for r in all_records
        if r.get("location") is not None
    )
)

for location in locations:

    records = [
        r for r in all_records
        if r.get("location") == location
    ]

    total = len(records)

    constants = sum(
        r.get("constant", False)
        for r in records
    )

    print(
        f"{location:20s} | "
        f"Total: {total:5d} | "
        f"Constant: {constants:5d} | "
        f"Percentage: "
        f"{(100 * constants / total if total else 0):6.2f}%"
    )

print()


# ============================================================
# SCALE ANALYSIS
# ============================================================

print("=" * 70)
print("5. CONSTANT IMAGES BY SCALE")
print("=" * 70)

for scale in [2, 3, 4]:

    records = [
        r for r in all_records
        if r.get("scale") == scale
    ]

    total = len(records)

    constants = sum(
        r.get("constant", False)
        for r in records
    )

    print(
        f"x{scale} | "
        f"Total: {total:5d} | "
        f"Constant: {constants:5d} | "
        f"Percentage: "
        f"{(100 * constants / total if total else 0):6.2f}%"
    )

print()


# ============================================================
# SPLIT + LOCATION
# ============================================================

print("=" * 70)
print("6. SPLIT + LOCATION BREAKDOWN")
print("=" * 70)

for split in SPLITS:

    print(f"\n[{split.upper()}]")

    for location in locations:

        records = [
            r for r in all_records
            if r["split"] == split
            and r.get("location") == location
        ]

        if not records:
            continue

        constants = sum(
            r.get("constant", False)
            for r in records
        )

        print(
            f"  {location:20s} "
            f"total={len(records):4d} "
            f"constant={constants:4d}"
        )

print()


# ============================================================
# HR/LR PAIR ANALYSIS
# ============================================================

print("=" * 70)
print("7. HR/LR PAIR ANALYSIS")
print("=" * 70)

# Create lookup by filename/sample ID information

hr_lookup = {}
lr_lookup = {}

for record in all_records:

    if not record.get("ok", False):
        continue

    key = (
        record["split"],
        record.get("sample_id"),
        record.get("location"),
        record.get("x"),
        record.get("y"),
        record.get("scale"),
    )

    if record["image_type"] == "HR":
        hr_lookup[key] = record

    elif record["image_type"] == "LR":
        lr_lookup[key] = record


both_constant = 0
hr_only_constant = 0
lr_only_constant = 0
neither_constant = 0

pair_count = 0

for key in sorted(
    set(hr_lookup.keys()) | set(lr_lookup.keys())
):

    hr = hr_lookup.get(key)
    lr = lr_lookup.get(key)

    if hr is None or lr is None:
        continue

    pair_count += 1

    hr_constant = hr["constant"]
    lr_constant = lr["constant"]

    if hr_constant and lr_constant:

        both_constant += 1

    elif hr_constant:

        hr_only_constant += 1

    elif lr_constant:

        lr_only_constant += 1

    else:

        neither_constant += 1


print(f"Complete HR/LR pairs analyzed : {pair_count}")
print(f"Both HR + LR constant         : {both_constant}")
print(f"HR only constant              : {hr_only_constant}")
print(f"LR only constant              : {lr_only_constant}")
print(f"Neither constant              : {neither_constant}")

print()


# ============================================================
# SAMPLE CONSTANT IMAGE DETAILS
# ============================================================

print("=" * 70)
print("8. FIRST 50 CONSTANT IMAGE DETAILS")
print("=" * 70)

constant_valid = [
    r for r in constant_records
    if r.get("ok", False)
    and r.get("constant", False)
]

for record in constant_valid[:50]:

    print(
        f"{record['split']:5s} | "
        f"{record['image_type']:2s} | "
        f"{record['filename']} | "
        f"shape={record['shape']} | "
        f"min={record['min']:.2f} | "
        f"max={record['max']:.2f} | "
        f"mean={record['mean']:.2f} | "
        f"std={record['std']:.6f} | "
        f"unique={record['unique_values']}"
    )

print()


# ============================================================
# METADATA CROSS-CHECK
# ============================================================

print("=" * 70)
print("9. METADATA CROSS-CHECK")
print("=" * 70)

metadata_columns = set(metadata.columns)

required_columns = [
    "split",
    "location",
    "scale_factor",
    "blur_type",
    "jpeg_quality",
    "contrast_factor",
    "x",
    "y",
]

print("Expected metadata columns:")

for column in required_columns:

    if column in metadata_columns:
        print(f"  [OK] {column}")

    else:
        print(f"  [MISSING] {column}")

print()


# ============================================================
# MATCH CONSTANT IMAGES TO METADATA
# ============================================================

print("=" * 70)
print("10. CONSTANT IMAGE METADATA")
print("=" * 70)

metadata_by_key = {}

for _, row in metadata.iterrows():

    try:

        key = (
            str(row["split"]),
            str(row["location"]),
            int(row["x"]),
            int(row["y"]),
            int(row["scale_factor"]),
        )

        metadata_by_key[key] = row

    except Exception:
        continue


matched = 0
unmatched = 0

for record in constant_valid[:100]:

    key = (
        str(record["split"]),
        str(record["location"]),
        int(record["x"]),
        int(record["y"]),
        int(record["scale"]),
    )

    row = metadata_by_key.get(key)

    if row is None:

        unmatched += 1

        print(
            f"UNMATCHED: "
            f"{record['filename']}"
        )

        continue

    matched += 1

    print(
        f"{record['filename']} | "
        f"blur={row.get('blur_type', 'N/A')} | "
        f"JPEG={row.get('jpeg_quality', 'N/A')} | "
        f"contrast={row.get('contrast_factor', 'N/A')}"
    )

print()

print(f"Metadata matches   : {matched}")
print(f"Metadata unmatched : {unmatched}")

print()


# ============================================================
# SAVE DETAILED CSV
# ============================================================

diagnostic_df = pd.DataFrame(all_records)

csv_output = LOG_DIR / "constant_image_diagnostic.csv"

diagnostic_df.to_csv(
    csv_output,
    index=False
)

print("=" * 70)
print("11. OUTPUT FILES")
print("=" * 70)

print(f"Detailed CSV:")
print(csv_output)

print()


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print("=" * 70)
print("12. INTERPRETATION")
print("=" * 70)

if constant_count == 0:

    print("NO CONSTANT IMAGES DETECTED.")
    print("Dataset image-content QC looks clean.")

else:

    print(
        f"{constant_count} constant images were detected."
    )

    print(
        "DO NOT delete them yet."
    )

    print(
        "The next step is to determine whether "
        "the problem originates in source-patch "
        "selection or the degradation pipeline."
    )

print()

print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)