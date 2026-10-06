"""
MSFFAN PROJECT
============================================================
FIX BALTIMORE NAIP + DEM ALIGNMENT
============================================================

Problem discovered:
    The previous Baltimore aligned raster was created using
    an oversized target extent. Approximately 47% of the
    aligned raster was zero-filled outside the actual NAIP
    footprint.

Correct approach:
    1. Read original Baltimore NAIP.
    2. Read existing aligned Baltimore DEM.
    3. Calculate geographic intersection.
    4. Restrict target grid to that intersection.
    5. Snap the grid inward to whole 1-meter pixels.
    6. Reproject NAIP RGB to the target grid.
    7. Reproject DEM to the exact same target grid.
    8. Verify CRS, dimensions, resolution, transform and bounds.
    9. Verify that NAIP contains no large zero-filled region.
   10. Replace the old aligned files only after successful QC.

IMPORTANT:
    This script does NOT touch the generated dataset.
    It only rebuilds the Baltimore aligned source files.
"""

from pathlib import Path
import math
import shutil
import tempfile

import numpy as np
import rasterio

from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ORIGINAL_NAIP = (
    PROJECT_ROOT
    / "data"
    / "source"
    / "baltimore"
    / "NAIP"
    / "md_m_3907644_sw_18_030_20230525_20231018.tif"
)

ALIGNED_DIR = (
    PROJECT_ROOT
    / "data"
    / "source"
    / "aligned"
    / "baltimore"
)

OLD_NAIP = ALIGNED_DIR / "NAIP_1m.tif"
OLD_DEM = ALIGNED_DIR / "DEM_1m.tif"

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


# ============================================================
# TEMPORARY OUTPUTS
# ============================================================

TEMP_NAIP = ALIGNED_DIR / "NAIP_1m_fixed_tmp.tif"
TEMP_DEM = ALIGNED_DIR / "DEM_1m_fixed_tmp.tif"

BACKUP_NAIP = ALIGNED_DIR / "NAIP_1m_old_backup.tif"
BACKUP_DEM = ALIGNED_DIR / "DEM_1m_old_backup.tif"


# ============================================================
# TARGET SETTINGS
# ============================================================

TARGET_RESOLUTION = 1.0

# Use RGB only.
# Baltimore original has 4 bands, but the project dataset
# is standardized to RGB because San Francisco has 3 bands.
RGB_BANDS = [1, 2, 3]


# ============================================================
# HELPERS
# ============================================================

def print_bounds(label, bounds):
    print(f"{label}:")
    print(f"  left   = {bounds.left:.6f}")
    print(f"  bottom = {bounds.bottom:.6f}")
    print(f"  right  = {bounds.right:.6f}")
    print(f"  top    = {bounds.top:.6f}")


def raster_stats(path):
    """
    Calculate basic RGB statistics for a raster.
    """

    with rasterio.open(path) as src:

        total_pixels = 0
        zero_pixels = 0

        minimum = float("inf")
        maximum = float("-inf")

        sum_value = 0.0
        sum_square = 0.0

        for _, window in src.block_windows(1):

            data = src.read(
                indexes=RGB_BANDS,
                window=window
            ).astype(np.float64)

            total = data.shape[1] * data.shape[2]

            zero = np.all(
                data == 0,
                axis=0
            ).sum()

            total_pixels += total
            zero_pixels += int(zero)

            minimum = min(
                minimum,
                float(data.min())
            )

            maximum = max(
                maximum,
                float(data.max())
            )

            sum_value += float(data.sum())
            sum_square += float(
                np.square(data).sum()
            )

        mean = sum_value / (
            total_pixels * len(RGB_BANDS)
        )

        variance = (
            sum_square /
            (total_pixels * len(RGB_BANDS))
            - mean ** 2
        )

        std = math.sqrt(
            max(variance, 0.0)
        )

        zero_percentage = (
            100.0
            * zero_pixels
            / total_pixels
        )

        return {
            "width": src.width,
            "height": src.height,
            "bounds": src.bounds,
            "resolution": src.res,
            "zero_pixels": zero_pixels,
            "zero_percentage": zero_percentage,
            "min": minimum,
            "max": maximum,
            "mean": mean,
            "std": std,
        }


# ============================================================
# START
# ============================================================

print("=" * 80)
print("FIXING BALTIMORE NAIP + DEM ALIGNMENT")
print("=" * 80)

print()
print(f"Project root:")
print(PROJECT_ROOT)

print()
print(f"Original NAIP:")
print(ORIGINAL_NAIP)

print()
print(f"Existing aligned DEM:")
print(OLD_DEM)

print()


# ============================================================
# CHECK INPUTS
# ============================================================

if not ORIGINAL_NAIP.exists():
    raise FileNotFoundError(
        f"Original Baltimore NAIP not found:\n"
        f"{ORIGINAL_NAIP}"
    )

if not OLD_DEM.exists():
    raise FileNotFoundError(
        f"Existing aligned Baltimore DEM not found:\n"
        f"{OLD_DEM}"
    )


ALIGNED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# READ INPUT METADATA
# ============================================================

print("=" * 80)
print("1. READING INPUT DATA")
print("=" * 80)

with rasterio.open(ORIGINAL_NAIP) as naip:

    naip_crs = naip.crs
    naip_bounds = naip.bounds
    naip_width = naip.width
    naip_height = naip.height
    naip_resolution = naip.res

    print("ORIGINAL NAIP")
    print(f"  CRS        : {naip_crs}")
    print(f"  Size       : {naip_width} x {naip_height}")
    print(f"  Resolution : {naip_resolution}")
    print(f"  Bands      : {naip.count}")

    print_bounds(
        "  Bounds",
        naip_bounds
    )

print()


with rasterio.open(OLD_DEM) as dem:

    dem_crs = dem.crs
    dem_bounds = dem.bounds
    dem_width = dem.width
    dem_height = dem.height

    print("ALIGNED DEM")
    print(f"  CRS        : {dem_crs}")
    print(f"  Size       : {dem_width} x {dem_height}")
    print(f"  Resolution : {dem.res}")

    print_bounds(
        "  Bounds",
        dem_bounds
    )

print()


# ============================================================
# VERIFY CRS
# ============================================================

print("=" * 80)
print("2. CRS CHECK")
print("=" * 80)

if naip_crs != dem_crs:

    raise RuntimeError(
        "NAIP and DEM CRS do not match."
    )

print("CRS match: TRUE")
print()


# ============================================================
# CALCULATE INTERSECTION
# ============================================================

print("=" * 80)
print("3. CALCULATING ACTUAL NAIP/DEM INTERSECTION")
print("=" * 80)

intersection_left = max(
    naip_bounds.left,
    dem_bounds.left
)

intersection_bottom = max(
    naip_bounds.bottom,
    dem_bounds.bottom
)

intersection_right = min(
    naip_bounds.right,
    dem_bounds.right
)

intersection_top = min(
    naip_bounds.top,
    dem_bounds.top
)


if (
    intersection_right <= intersection_left
    or
    intersection_top <= intersection_bottom
):

    raise RuntimeError(
        "NAIP and DEM do not overlap."
    )


print(
    f"Intersection left   : "
    f"{intersection_left:.6f}"
)

print(
    f"Intersection bottom : "
    f"{intersection_bottom:.6f}"
)

print(
    f"Intersection right  : "
    f"{intersection_right:.6f}"
)

print(
    f"Intersection top    : "
    f"{intersection_top:.6f}"
)

print()


# ============================================================
# SNAP INWARD TO 1-METER GRID
# ============================================================

print("=" * 80)
print("4. CREATING SAFE 1-METER TARGET GRID")
print("=" * 80)

"""
We deliberately snap inward.

This prevents the target grid from extending even slightly
outside the real source footprint.

For:
    left   -> ceil
    bottom -> ceil
    right  -> floor
    top    -> floor
"""

target_left = math.ceil(
    intersection_left
)

target_bottom = math.ceil(
    intersection_bottom
)

target_right = math.floor(
    intersection_right
)

target_top = math.floor(
    intersection_top
)


target_width = int(
    target_right - target_left
)

target_height = int(
    target_top - target_bottom
)


if target_width <= 0 or target_height <= 0:

    raise RuntimeError(
        "Invalid target dimensions."
    )


target_transform = from_origin(
    target_left,
    target_top,
    TARGET_RESOLUTION,
    TARGET_RESOLUTION
)


print(
    f"Target left   : {target_left}"
)

print(
    f"Target bottom : {target_bottom}"
)

print(
    f"Target right  : "
    f"{target_left + target_width}"
)

print(
    f"Target top    : "
    f"{target_top}"
)

print(
    f"Target size   : "
    f"{target_width} x {target_height}"
)

print(
    f"Resolution    : "
    f"{TARGET_RESOLUTION} x "
    f"{TARGET_RESOLUTION}"
)

print()


# ============================================================
# VERIFY TARGET IS INSIDE BOTH INPUTS
# ============================================================

print("=" * 80)
print("5. TARGET COVERAGE CHECK")
print("=" * 80)

target_right_actual = (
    target_left + target_width
)

target_bottom_actual = (
    target_top - target_height
)


inside_naip = (
    target_left >= naip_bounds.left
    and
    target_right_actual <= naip_bounds.right
    and
    target_bottom_actual >= naip_bounds.bottom
    and
    target_top <= naip_bounds.top
)


inside_dem = (
    target_left >= dem_bounds.left
    and
    target_right_actual <= dem_bounds.right
    and
    target_bottom_actual >= dem_bounds.bottom
    and
    target_top <= dem_bounds.top
)


print(
    f"Target completely inside NAIP : "
    f"{inside_naip}"
)

print(
    f"Target completely inside DEM  : "
    f"{inside_dem}"
)

if not inside_naip:
    raise RuntimeError(
        "Target grid is outside NAIP coverage."
    )

if not inside_dem:
    raise RuntimeError(
        "Target grid is outside DEM coverage."
    )

print()
print("TARGET COVERAGE CHECK PASSED")
print()


# ============================================================
# REMOVE TEMP FILES
# ============================================================

for path in [
    TEMP_NAIP,
    TEMP_DEM,
]:

    if path.exists():
        path.unlink()


# ============================================================
# CREATE FIXED NAIP
# ============================================================

print("=" * 80)
print("6. REPROJECTING NAIP TO SAFE COMMON GRID")
print("=" * 80)

with rasterio.open(ORIGINAL_NAIP) as src:

    profile = src.profile.copy()

    profile.update(
        driver="GTiff",
        width=target_width,
        height=target_height,
        count=3,
        dtype="uint8",
        crs=naip_crs,
        transform=target_transform,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
        nodata=None,
    )

    with rasterio.open(
        TEMP_NAIP,
        "w",
        **profile
    ) as dst:

        for output_band, source_band in enumerate(
            RGB_BANDS,
            start=1
        ):

            reproject(
                source=rasterio.band(
                    src,
                    source_band
                ),

                destination=rasterio.band(
                    dst,
                    output_band
                ),

                src_transform=src.transform,
                src_crs=src.crs,

                dst_transform=target_transform,
                dst_crs=naip_crs,

                src_nodata=None,
                dst_nodata=0,

                resampling=Resampling.bilinear,
            )


print(
    f"Saved temporary NAIP:"
)

print(TEMP_NAIP)

print()


# ============================================================
# CREATE FIXED DEM
# ============================================================

print("=" * 80)
print("7. REPROJECTING DEM TO SAME GRID")
print("=" * 80)

with rasterio.open(OLD_DEM) as src:

    profile = src.profile.copy()

    profile.update(
        driver="GTiff",
        width=target_width,
        height=target_height,
        count=1,
        dtype="float32",
        crs=naip_crs,
        transform=target_transform,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
        nodata=-999999.0,
    )

    with rasterio.open(
        TEMP_DEM,
        "w",
        **profile
    ) as dst:

        reproject(
            source=rasterio.band(
                src,
                1
            ),

            destination=rasterio.band(
                dst,
                1
            ),

            src_transform=src.transform,
            src_crs=src.crs,

            dst_transform=target_transform,
            dst_crs=naip_crs,

            src_nodata=src.nodata,
            dst_nodata=-999999.0,

            resampling=Resampling.bilinear,
        )


print(
    f"Saved temporary DEM:"
)

print(TEMP_DEM)

print()


# ============================================================
# VERIFY TEMPORARY NAIP
# ============================================================

print("=" * 80)
print("8. FIXED NAIP QUALITY CHECK")
print("=" * 80)

stats = raster_stats(
    TEMP_NAIP
)

print(
    f"Width           : {stats['width']}"
)

print(
    f"Height          : {stats['height']}"
)

print(
    f"Resolution      : {stats['resolution']}"
)

print(
    f"Zero pixels     : {stats['zero_pixels']:,}"
)

print(
    f"Zero percentage : "
    f"{stats['zero_percentage']:.4f}%"
)

print(
    f"Min             : {stats['min']:.2f}"
)

print(
    f"Max             : {stats['max']:.2f}"
)

print(
    f"Mean            : {stats['mean']:.2f}"
)

print(
    f"Std             : {stats['std']:.4f}"
)

print()


# ============================================================
# FINAL ZERO CHECK
# ============================================================

if stats["zero_percentage"] > 5.0:

    raise RuntimeError(
        "Fixed NAIP still contains more than 5% "
        "zero pixels. Stop before replacing files."
    )

print(
    "NAIP ZERO-PIXEL CHECK PASSED"
)

print()


# ============================================================
# VERIFY EXACT ALIGNMENT
# ============================================================

print("=" * 80)
print("9. FINAL NAIP/DEM ALIGNMENT CHECK")
print("=" * 80)

with rasterio.open(TEMP_NAIP) as naip:

    with rasterio.open(TEMP_DEM) as dem:

        checks = {

            "CRS":
                naip.crs == dem.crs,

            "Width":
                naip.width == dem.width,

            "Height":
                naip.height == dem.height,

            "Resolution":
                np.allclose(
                    naip.res,
                    dem.res
                ),

            "Transform":
                np.allclose(
                    tuple(naip.transform),
                    tuple(dem.transform)
                ),

            "Bounds":
                np.allclose(
                    tuple(naip.bounds),
                    tuple(dem.bounds)
                ),
        }


for name, passed in checks.items():

    print(
        f"{name:12s}: "
        f"{passed}"
    )


if not all(checks.values()):

    raise RuntimeError(
        "FINAL ALIGNMENT CHECK FAILED."
    )


print()
print("FINAL ALIGNMENT CHECK PASSED")
print()


# ============================================================
# BACKUP OLD FILES
# ============================================================

print("=" * 80)
print("10. BACKING UP OLD ALIGNMENT")
print("=" * 80)

if BACKUP_NAIP.exists():
    BACKUP_NAIP.unlink()

if BACKUP_DEM.exists():
    BACKUP_DEM.unlink()


if OLD_NAIP.exists():

    shutil.copy2(
        OLD_NAIP,
        BACKUP_NAIP
    )

    print(
        f"Old NAIP backed up to:"
    )

    print(BACKUP_NAIP)


if OLD_DEM.exists():

    shutil.copy2(
        OLD_DEM,
        BACKUP_DEM
    )

    print(
        f"Old DEM backed up to:"
    )

    print(BACKUP_DEM)


print()


# ============================================================
# REPLACE OLD ALIGNMENT
# ============================================================

print("=" * 80)
print("11. INSTALLING FIXED ALIGNMENT")
print("=" * 80)

shutil.move(
    TEMP_NAIP,
    OLD_NAIP
)

shutil.move(
    TEMP_DEM,
    OLD_DEM
)

print(
    "Fixed NAIP installed."
)

print(
    "Fixed DEM installed."
)

print()


# ============================================================
# FINAL FILE VERIFICATION
# ============================================================

print("=" * 80)
print("12. FINAL INSTALLED FILE VERIFICATION")
print("=" * 80)

with rasterio.open(OLD_NAIP) as naip:

    print("NAIP:")
    print(
        f"  Size       : "
        f"{naip.width} x {naip.height}"
    )

    print(
        f"  Resolution : "
        f"{naip.res}"
    )

    print(
        f"  CRS        : "
        f"{naip.crs}"
    )

    print_bounds(
        "  Bounds",
        naip.bounds
    )

print()


with rasterio.open(OLD_DEM) as dem:

    print("DEM:")
    print(
        f"  Size       : "
        f"{dem.width} x {dem.height}"
    )

    print(
        f"  Resolution : "
        f"{dem.res}"
    )

    print(
        f"  CRS        : "
        f"{dem.crs}"
    )

    print_bounds(
        "  Bounds",
        dem.bounds
    )

print()


# ============================================================
# COMPLETE
# ============================================================

print("=" * 80)
print("BALTIMORE ALIGNMENT FIX COMPLETE")
print("=" * 80)

print()
print("IMPORTANT:")
print("The generated dataset has NOT been modified.")
print()
print("Next step:")
print("Regenerate the 5,096-pair dataset after verifying")
print("the corrected source raster.")
print()
print("Old alignment backups were preserved.")
print()
print("=" * 80)