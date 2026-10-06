"""
MSFFAN PROJECT
Baltimore Alignment Coverage Verification

Purpose:
    Verify whether the aligned Baltimore NAIP raster extends
    outside the actual original NAIP coverage.

This script DOES NOT modify any files.
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


# ============================================================
# PATHS
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

ALIGNED_NAIP = (
    PROJECT_ROOT
    / "data"
    / "source"
    / "aligned"
    / "baltimore"
    / "NAIP_1m.tif"
)

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


# ============================================================
# START
# ============================================================

print("=" * 75)
print("BALTIMORE ALIGNMENT COVERAGE VERIFICATION")
print("=" * 75)

print()


# ============================================================
# ORIGINAL
# ============================================================

print("=" * 75)
print("1. ORIGINAL NAIP")
print("=" * 75)

with rasterio.open(ORIGINAL_NAIP) as src:

    original_crs = src.crs
    original_bounds = src.bounds
    original_width = src.width
    original_height = src.height
    original_resolution = src.res
    original_count = src.count

    print(f"CRS         : {original_crs}")
    print(f"Width       : {original_width}")
    print(f"Height      : {original_height}")
    print(f"Bands       : {original_count}")
    print(f"Resolution  : {original_resolution}")
    print(f"Bounds      : {original_bounds}")

print()


# ============================================================
# ALIGNED
# ============================================================

print("=" * 75)
print("2. ALIGNED NAIP")
print("=" * 75)

with rasterio.open(ALIGNED_NAIP) as src:

    aligned_crs = src.crs
    aligned_bounds = src.bounds
    aligned_width = src.width
    aligned_height = src.height
    aligned_resolution = src.res
    aligned_count = src.count

    print(f"CRS         : {aligned_crs}")
    print(f"Width       : {aligned_width}")
    print(f"Height      : {aligned_height}")
    print(f"Bands       : {aligned_count}")
    print(f"Resolution  : {aligned_resolution}")
    print(f"Bounds      : {aligned_bounds}")

print()


# ============================================================
# BOUNDS COMPARISON
# ============================================================

print("=" * 75)
print("3. BOUNDS COMPARISON")
print("=" * 75)

print()

print("Original:")
print(f"  left   = {original_bounds.left}")
print(f"  bottom = {original_bounds.bottom}")
print(f"  right  = {original_bounds.right}")
print(f"  top    = {original_bounds.top}")

print()

print("Aligned:")
print(f"  left   = {aligned_bounds.left}")
print(f"  bottom = {aligned_bounds.bottom}")
print(f"  right  = {aligned_bounds.right}")
print(f"  top    = {aligned_bounds.top}")

print()


# ============================================================
# EXTENSION OUTSIDE ORIGINAL
# ============================================================

left_extension = (
    original_bounds.left
    - aligned_bounds.left
)

right_extension = (
    aligned_bounds.right
    - original_bounds.right
)

bottom_extension = (
    original_bounds.bottom
    - aligned_bounds.bottom
)

top_extension = (
    aligned_bounds.top
    - original_bounds.top
)

print("=" * 75)
print("4. ALIGNED AREA OUTSIDE ORIGINAL NAIP")
print("=" * 75)

print(
    f"Left extension   : {left_extension:.3f} m"
)

print(
    f"Right extension  : {right_extension:.3f} m"
)

print(
    f"Bottom extension : {bottom_extension:.3f} m"
)

print(
    f"Top extension    : {top_extension:.3f} m"
)

print()


# ============================================================
# GEOMETRIC AREA
# ============================================================

original_area = (
    original_bounds.right
    - original_bounds.left
) * (
    original_bounds.top
    - original_bounds.bottom
)

aligned_area = (
    aligned_bounds.right
    - aligned_bounds.left
) * (
    aligned_bounds.top
    - aligned_bounds.bottom
)

print("=" * 75)
print("5. AREA")
print("=" * 75)

print(
    f"Original NAIP area : "
    f"{original_area:,.2f} m²"
)

print(
    f"Aligned area       : "
    f"{aligned_area:,.2f} m²"
)

print(
    f"Aligned / Original : "
    f"{100 * aligned_area / original_area:.2f}%"
)

print()


# ============================================================
# ACTUAL GEOMETRIC INTERSECTION
# ============================================================

intersection_left = max(
    original_bounds.left,
    aligned_bounds.left
)

intersection_bottom = max(
    original_bounds.bottom,
    aligned_bounds.bottom
)

intersection_right = min(
    original_bounds.right,
    aligned_bounds.right
)

intersection_top = min(
    original_bounds.top,
    aligned_bounds.top
)

if (
    intersection_right > intersection_left
    and intersection_top > intersection_bottom
):

    intersection_area = (
        intersection_right
        - intersection_left
    ) * (
        intersection_top
        - intersection_bottom
    )

else:

    intersection_area = 0


print("=" * 75)
print("6. ACTUAL ORIGINAL/ALIGNED INTERSECTION")
print("=" * 75)

print(
    f"Intersection left   : {intersection_left}"
)

print(
    f"Intersection bottom : {intersection_bottom}"
)

print(
    f"Intersection right  : {intersection_right}"
)

print(
    f"Intersection top    : {intersection_top}"
)

print(
    f"Intersection area   : "
    f"{intersection_area:,.2f} m²"
)

print(
    f"Original covered    : "
    f"{100 * intersection_area / original_area:.2f}%"
)

print()


# ============================================================
# ZERO PIXEL CHECK
# ============================================================

print("=" * 75)
print("7. ALIGNED RASTER ZERO-PIXEL CHECK")
print("=" * 75)

with rasterio.open(ALIGNED_NAIP) as src:

    total_pixels = 0
    zero_pixels = 0
    nonzero_pixels = 0

    # Process in 512x512 blocks
    block_size = 512

    for y in range(0, src.height, block_size):

        height = min(
            block_size,
            src.height - y
        )

        for x in range(0, src.width, block_size):

            width = min(
                block_size,
                src.width - x
            )

            data = src.read(
                indexes=[1, 2, 3],
                window=Window(
                    x,
                    y,
                    width,
                    height
                )
            )

            data = data.astype(
                np.uint8,
                copy=False
            )

            pixels = (
                data.shape[1]
                * data.shape[2]
            )

            zeros = np.all(
                data == 0,
                axis=0
            ).sum()

            total_pixels += pixels
            zero_pixels += int(zeros)

            nonzero_pixels += (
                pixels - int(zeros)
            )


print(
    f"Total RGB pixels : {total_pixels:,}"
)

print(
    f"Zero RGB pixels  : {zero_pixels:,}"
)

print(
    f"Non-zero pixels  : {nonzero_pixels:,}"
)

print(
    f"Zero percentage  : "
    f"{100 * zero_pixels / total_pixels:.2f}%"
)

print()


# ============================================================
# FINAL
# ============================================================

print("=" * 75)
print("FINAL INTERPRETATION")
print("=" * 75)

if (
    aligned_bounds.left < original_bounds.left
    or aligned_bounds.right > original_bounds.right
    or aligned_bounds.bottom < original_bounds.bottom
    or aligned_bounds.top > original_bounds.top
):

    print(
        "WARNING: Aligned raster extends outside "
        "original NAIP coverage."
    )

    print()

    print(
        "This can create zero-filled areas and "
        "is consistent with the constant patches."
    )

else:

    print(
        "Aligned raster is fully inside original "
        "NAIP coverage."
    )

print()

print(
    "NO FILES WERE MODIFIED."
)

print("=" * 75)
print("VERIFICATION COMPLETE")
print("=" * 75)