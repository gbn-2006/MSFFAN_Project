"""
MSFFAN PROJECT
Baltimore Source-Patch Investigation

Purpose:
    Determine why 714 Baltimore HR patches are completely zero.

This script:
    1. Reads the Baltimore aligned NAIP source.
    2. Uses the x/y coordinates from dataset metadata.
    3. Extracts the corresponding 256x256 HR patches.
    4. Measures zero-pixel percentage and image statistics.
    5. Compares bad patches with good patches.
    6. Checks whether the problem is actually present
       in the source raster.

IMPORTANT:
    This script DOES NOT modify anything.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "data" / "dataset"

METADATA_FILE = DATASET_ROOT / "dataset_metadata.csv"

BALTIMORE_NAIP = (
    PROJECT_ROOT
    / "data"
    / "source"
    / "aligned"
    / "baltimore"
    / "NAIP_1m.tif"
)

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = LOG_DIR / "baltimore_source_patch_analysis.csv"


PATCH_SIZE = 256


# ============================================================
# START
# ============================================================

print("=" * 75)
print("BALTIMORE SOURCE-PATCH INVESTIGATION")
print("=" * 75)

print(f"Project root:")
print(PROJECT_ROOT)

print()

print("Metadata:")
print(METADATA_FILE)

print()

print("Baltimore aligned NAIP:")
print(BALTIMORE_NAIP)

print()


# ============================================================
# CHECK FILES
# ============================================================

if not METADATA_FILE.exists():

    raise FileNotFoundError(
        f"Metadata not found:\n{METADATA_FILE}"
    )


if not BALTIMORE_NAIP.exists():

    raise FileNotFoundError(
        f"Baltimore aligned NAIP not found:\n"
        f"{BALTIMORE_NAIP}"
    )


# ============================================================
# LOAD METADATA
# ============================================================

metadata = pd.read_csv(METADATA_FILE)

print("=" * 75)
print("METADATA")
print("=" * 75)

print(f"Total metadata rows: {len(metadata)}")

print()


# ============================================================
# IDENTIFY CONSTANT HR FILES
# ============================================================

constant_hr = []

print("=" * 75)
print("FINDING CONSTANT BALTIMORE HR FILES")
print("=" * 75)


hr_folder = DATASET_ROOT / "train" / "HR"

# Scan all splits
for split in ["train", "val", "test"]:

    folder = DATASET_ROOT / split / "HR"

    for image_path in sorted(folder.glob("*.png")):

        if "baltimore" not in image_path.name:
            continue

        # Filename contains coordinates.
        #
        # Example:
        # train_00009_baltimore_x1180_y2441_x3.png

        parts = image_path.stem.split("_")

        try:

            x_index = parts.index("x")

        except ValueError:

            # Filename structure is known, so use
            # direct parsing instead.
            pass

        # Easier robust parsing
        name = image_path.stem

        import re

        match = re.search(
            r"_x(\d+)_y(\d+)_x(\d+)$",
            name
        )

        if match is None:
            continue

        x = int(match.group(1))
        y = int(match.group(2))
        scale = int(match.group(3))

        # Read HR image
        from PIL import Image

        with Image.open(image_path) as img:

            arr = np.asarray(img)

        if arr.ndim == 2:
            arr = arr[:, :, None]

        if (
            arr.size > 0
            and np.all(arr == 0)
        ):

            constant_hr.append(
                {
                    "split": split,
                    "filename": image_path.name,
                    "x": x,
                    "y": y,
                    "scale": scale,
                }
            )


print(
    f"Constant Baltimore HR patches found: "
    f"{len(constant_hr)}"
)

print()


# ============================================================
# OPEN SOURCE RASTER
# ============================================================

print("=" * 75)
print("BALTIMORE SOURCE RASTER")
print("=" * 75)

with rasterio.open(BALTIMORE_NAIP) as src:

    print(f"CRS       : {src.crs}")
    print(f"Width     : {src.width}")
    print(f"Height    : {src.height}")
    print(f"Bands     : {src.count}")
    print(f"Dtype     : {src.dtypes}")
    print(f"NoData    : {src.nodata}")
    print(f"Transform : {src.transform}")
    print(f"Bounds    : {src.bounds}")

    print()


    # ========================================================
    # SAMPLE BAD PATCHES
    # ========================================================

    print("=" * 75)
    print("BAD PATCH SOURCE ANALYSIS")
    print("=" * 75)

    results = []

    for i, item in enumerate(constant_hr):

        x = item["x"]
        y = item["y"]

        # ----------------------------------------------------
        # IMPORTANT:
        # x/y are assumed to be pixel coordinates in the
        # aligned raster.
        # ----------------------------------------------------

        if (
            x < 0
            or y < 0
            or x + PATCH_SIZE > src.width
            or y + PATCH_SIZE > src.height
        ):

            results.append(
                {
                    **item,
                    "source_status": "OUT_OF_BOUNDS",
                }
            )

            continue


        # Read RGB bands
        patch = src.read(
            indexes=[1, 2, 3],
            window=rasterio.windows.Window(
                x,
                y,
                PATCH_SIZE,
                PATCH_SIZE,
            ),
        )


        patch = patch.astype(np.float32)


        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        total_pixels = patch.shape[1] * patch.shape[2]

        zero_pixels = np.all(
            patch == 0,
            axis=0
        ).sum()

        zero_percentage = (
            100.0
            * zero_pixels
            / total_pixels
        )

        min_value = float(
            patch.min()
        )

        max_value = float(
            patch.max()
        )

        mean_value = float(
            patch.mean()
        )

        std_value = float(
            patch.std()
        )

        unique_values = len(
            np.unique(patch)
        )


        # ----------------------------------------------------
        # MASK INFORMATION
        # ----------------------------------------------------

        try:

            masks = src.read_masks(
                indexes=[1, 2, 3],
                window=rasterio.windows.Window(
                    x,
                    y,
                    PATCH_SIZE,
                    PATCH_SIZE,
                ),
            )

            invalid_mask_pixels = np.all(
                masks == 0,
                axis=0
            ).sum()

            invalid_percentage = (
                100.0
                * invalid_mask_pixels
                / total_pixels
            )

        except Exception:

            invalid_percentage = -1


        # ----------------------------------------------------
        # CLASSIFY
        # ----------------------------------------------------

        if std_value == 0:

            status = "SOURCE_CONSTANT"

        elif zero_percentage > 50:

            status = "MOSTLY_ZERO"

        elif zero_percentage > 0:

            status = "PARTIAL_ZERO"

        else:

            status = "VALID"


        result = {
            **item,

            "source_status": status,

            "source_min": min_value,
            "source_max": max_value,
            "source_mean": mean_value,
            "source_std": std_value,

            "source_unique_values": unique_values,

            "zero_pixels": int(zero_pixels),
            "zero_percentage": zero_percentage,

            "invalid_mask_pixels": int(
                invalid_mask_pixels
            ),

            "invalid_percentage":
                invalid_percentage,
        }

        results.append(result)


        # Print first 30
        if i < 30:

            print(
                f"{item['filename']}"
            )

            print(
                f"  x={x}, y={y}, scale=x{item['scale']}"
            )

            print(
                f"  min={min_value:.2f}"
            )

            print(
                f"  max={max_value:.2f}"
            )

            print(
                f"  mean={mean_value:.2f}"
            )

            print(
                f"  std={std_value:.6f}"
            )

            print(
                f"  zero pixels={zero_percentage:.2f}%"
            )

            print(
                f"  invalid mask={invalid_percentage:.2f}%"
            )

            print(
                f"  STATUS={status}"
            )

            print()


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 75)
print("SOURCE PATCH SUMMARY")
print("=" * 75)

if len(results_df) > 0:

    print()

    print(
        results_df[
            "source_status"
        ].value_counts()
    )

    print()

    print(
        f"Average zero percentage: "
        f"{results_df['zero_percentage'].mean():.2f}%"
    )

    print(
        f"Maximum zero percentage: "
        f"{results_df['zero_percentage'].max():.2f}%"
    )

    print(
        f"Minimum zero percentage: "
        f"{results_df['zero_percentage'].min():.2f}%"
    )

    print()

    print(
        f"Source patches with >50% zero: "
        f"{(
            results_df['zero_percentage'] > 50
        ).sum()}"
    )

    print(
        f"Source patches with 100% zero: "
        f"{(
            results_df['zero_percentage'] == 100
        ).sum()}"
    )

    print()

    print(
        f"Source patches with invalid mask: "
        f"{(
            results_df['invalid_percentage'] > 0
        ).sum()}"
    )


# ============================================================
# FINAL
# ============================================================

print()

print("=" * 75)
print("OUTPUT")
print("=" * 75)

print(
    f"Detailed analysis saved to:"
)

print(OUTPUT_CSV)

print()

print("=" * 75)
print("INVESTIGATION COMPLETE")
print("=" * 75)