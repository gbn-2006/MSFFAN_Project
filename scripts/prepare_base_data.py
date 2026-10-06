import os

import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling


# ============================================================
# INPUT FILES
# ============================================================

NAIP_FILE = (
    "data/source/NAIP/NAIP_base.tif"
)

DEM_FILE = (
    "data/source/DEM/"
    "USGS_1M_10_x54y418_CA_SanFrancisco_B23.tif"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    "data/source/aligned"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


NAIP_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "NAIP_1m.tif"
)

DEM_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "DEM_1m.tif"
)


# ============================================================
# STEP 1
# READ NAIP INFORMATION
# ============================================================

print()
print("======================================")
print("STEP 1: READING NAIP")
print("======================================")


with rasterio.open(
    NAIP_FILE
) as naip:

    print(
        "CRS:",
        naip.crs
    )

    print(
        "Original resolution:",
        naip.res
    )

    print(
        "Original size:",
        naip.width,
        "x",
        naip.height
    )

    print(
        "Bounds:",
        naip.bounds
    )

    # Store the NAIP bounds.
    #
    # We use the NAIP footprint as our
    # study area.

    study_bounds = naip.bounds

    target_crs = naip.crs


# ============================================================
# STEP 2
# PREPARE 1-METER NAIP GRID
# ============================================================

print()
print("======================================")
print("STEP 2: CREATING 1-METER NAIP")
print("======================================")


with rasterio.open(
    NAIP_FILE
) as src:

    # Desired output resolution

    target_resolution = 1.0

    # Calculate output dimensions

    target_width = int(
        (study_bounds.right -
         study_bounds.left)
        / target_resolution
    )

    target_height = int(
        (study_bounds.top -
         study_bounds.bottom)
        / target_resolution
    )


    # Create output transform

    target_transform = rasterio.transform.from_bounds(

        study_bounds.left,

        study_bounds.bottom,

        study_bounds.right,

        study_bounds.top,

        target_width,

        target_height

    )


    profile = src.profile.copy()

    profile.update({

        "driver": "GTiff",

        "height": target_height,

        "width": target_width,

        "transform": target_transform,

        "crs": target_crs,

        "count": 3,

        "dtype": "uint8"

    })


    with rasterio.open(
        NAIP_OUTPUT,
        "w",
        **profile
    ) as dst:

        for band in range(
            1,
            4
        ):

            reproject(

                source=rasterio.band(
                    src,
                    band
                ),

                destination=rasterio.band(
                    dst,
                    band
                ),

                src_transform=src.transform,

                src_crs=src.crs,

                dst_transform=
                    target_transform,

                dst_crs=target_crs,

                resampling=
                    Resampling.bilinear

            )


print(
    "Saved:",
    NAIP_OUTPUT
)


# ============================================================
# STEP 3
# PREPARE DEM ON SAME GRID
# ============================================================

print()
print("======================================")
print("STEP 3: ALIGNING DEM")
print("======================================")


with rasterio.open(
    DEM_FILE
) as src:

    print(
        "DEM CRS:",
        src.crs
    )

    print(
        "DEM resolution:",
        src.res
    )

    profile = src.profile.copy()

    profile.update({

        "driver": "GTiff",

        "height": target_height,

        "width": target_width,

        "transform":
            target_transform,

        "crs":
            target_crs,

        "dtype":
            "float32",

        "count":
            1

    })


    with rasterio.open(
        DEM_OUTPUT,
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

            dst_transform=
                target_transform,

            dst_crs=target_crs,

            resampling=
                Resampling.bilinear

        )


print(
    "Saved:",
    DEM_OUTPUT
)


# ============================================================
# FINISHED
# ============================================================

print()
print("======================================")
print("PREPROCESSING COMPLETE")
print("======================================")

print()
print(
    "NAIP:",
    NAIP_OUTPUT
)

print(
    "DEM:",
    DEM_OUTPUT
)

print()
print(
    "Both datasets now use the same"
)

print(
    "CRS, geographic extent, and 1 m grid."
)

print()