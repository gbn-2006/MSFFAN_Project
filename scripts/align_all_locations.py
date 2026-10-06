import os
import math
import rasterio

from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.transform import from_origin
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    transform_bounds,
)
from rasterio.windows import from_bounds


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SOURCE_ROOT = os.path.join(PROJECT_ROOT, "data", "source")
ALIGNED_ROOT = os.path.join(SOURCE_ROOT, "aligned")


LOCATIONS = {
    "san_francisco": {
        "naip": os.path.join(
            SOURCE_ROOT,
            "NAIP",
            "NAIP_base.tif",
        ),
        "dem": os.path.join(
            SOURCE_ROOT,
            "DEM",
            "USGS_1M_10_x54y418_CA_SanFrancisco_B23.tif",
        ),
    },

    "baltimore": {
        "naip": os.path.join(
            SOURCE_ROOT,
            "baltimore",
            "NAIP",
            "md_m_3907644_sw_18_030_20230525_20231018.tif",
        ),
        "dem": os.path.join(
            SOURCE_ROOT,
            "baltimore",
            "DEM",
            "S1M_n1980e1640_20260807.tif",
        ),
    },

    "denver": {
        "naip": os.path.join(
            SOURCE_ROOT,
            "denver",
            "NAIP",
            "co_m_3910417_nw_13_060_20210728.tif",
        ),
        "dem": os.path.join(
            SOURCE_ROOT,
            "denver",
            "DEM",
            "USGS_1M_13_x50y440_CO_DRCOG_2020_B20.tif",
        ),
    },
}


TARGET_RESOLUTION = 1.0

# We use RGB consistently for the MSFFAN optical input.
# Original source files remain untouched.
USE_RGB_ONLY = True


def separator():
    print("=" * 75)


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)


def get_intersection_in_crs(
    naip_path,
    dem_path,
):
    """
    Find the true spatial intersection between NAIP and DEM.

    The intersection is calculated in the DEM CRS so that
    the area calculation is meaningful.
    """

    with rasterio.open(naip_path) as naip:
        with rasterio.open(dem_path) as dem:

            naip_bounds_in_dem = transform_bounds(
                naip.crs,
                dem.crs,
                *naip.bounds,
                densify_pts=21,
            )

            dem_bounds = dem.bounds

            left = max(
                naip_bounds_in_dem[0],
                dem_bounds.left,
            )

            bottom = max(
                naip_bounds_in_dem[1],
                dem_bounds.bottom,
            )

            right = min(
                naip_bounds_in_dem[2],
                dem_bounds.right,
            )

            top = min(
                naip_bounds_in_dem[3],
                dem_bounds.top,
            )

            if left >= right or bottom >= top:
                raise RuntimeError(
                    "NAIP and DEM do not have a valid spatial intersection."
                )

            return (
                left,
                bottom,
                right,
                top,
                dem.crs,
            )


def transform_intersection_to_naip_crs(
    intersection,
    dem_crs,
    naip_crs,
):
    left, bottom, right, top, _ = intersection

    transformed = transform_bounds(
        dem_crs,
        naip_crs,
        left,
        bottom,
        right,
        top,
        densify_pts=21,
    )

    return transformed


def create_target_grid(
    bounds,
    resolution,
):
    """
    Create a deterministic north-up grid.

    Pixel dimensions are derived from the spatial bounds
    and target resolution.
    """

    left, bottom, right, top = bounds

    width = math.floor(
        (right - left) / resolution
    )

    height = math.floor(
        (top - bottom) / resolution
    )

    if width <= 0 or height <= 0:
        raise RuntimeError(
            "Invalid target grid dimensions."
        )

    # Recalculate exact extent so transform,
    # dimensions and resolution agree.
    exact_right = left + width * resolution
    exact_bottom = top - height * resolution

    transform = from_origin(
        left,
        top,
        resolution,
        resolution,
    )

    return (
        transform,
        width,
        height,
        (
            left,
            exact_bottom,
            exact_right,
            top,
        ),
    )


def align_location(
    location_name,
    naip_path,
    dem_path,
):
    separator()
    print(
        f"ALIGNING LOCATION: "
        f"{location_name.upper()}"
    )
    separator()

    if not os.path.exists(naip_path):
        raise FileNotFoundError(
            f"NAIP not found: {naip_path}"
        )

    if not os.path.exists(dem_path):
        raise FileNotFoundError(
            f"DEM not found: {dem_path}"
        )

    output_dir = os.path.join(
        ALIGNED_ROOT,
        location_name,
    )

    ensure_directory(output_dir)

    naip_output = os.path.join(
        output_dir,
        "NAIP_1m.tif",
    )

    dem_output = os.path.join(
        output_dir,
        "DEM_1m.tif",
    )

    print("\nSTEP 1: READING SOURCE DATA")

    with rasterio.open(naip_path) as naip:
        print("\nNAIP")
        print(f"  CRS: {naip.crs}")
        print(
            f"  Size: "
            f"{naip.width} x {naip.height}"
        )
        print(f"  Resolution: {naip.res}")
        print(f"  Bands: {naip.count}")
        print(f"  Bounds: {naip.bounds}")

        naip_crs = naip.crs
        naip_dtype = naip.dtypes[0]

    with rasterio.open(dem_path) as dem:
        print("\nDEM")
        print(f"  CRS: {dem.crs}")
        print(
            f"  Size: "
            f"{dem.width} x {dem.height}"
        )
        print(f"  Resolution: {dem.res}")
        print(f"  Bands: {dem.count}")
        print(f"  Bounds: {dem.bounds}")

        dem_crs = dem.crs

    print("\nSTEP 2: CALCULATING TRUE OVERLAP")

    intersection = get_intersection_in_crs(
        naip_path,
        dem_path,
    )

    (
        int_left,
        int_bottom,
        int_right,
        int_top,
        intersection_crs,
    ) = intersection

    print(
        f"  Intersection CRS: "
        f"{intersection_crs}"
    )

    print(
        f"  Intersection bounds:\n"
        f"    left   = {int_left:.3f}\n"
        f"    bottom = {int_bottom:.3f}\n"
        f"    right  = {int_right:.3f}\n"
        f"    top    = {int_top:.3f}"
    )

    intersection_area = (
        (int_right - int_left)
        * (int_top - int_bottom)
    )

    print(
        f"  Approx. overlap area: "
        f"{intersection_area:,.2f}"
    )

    print(
        "\nSTEP 3: TRANSFORMING OVERLAP "
        "TO NAIP CRS"
    )

    target_bounds = (
        transform_intersection_to_naip_crs(
            intersection,
            intersection_crs,
            naip_crs,
        )
    )

    print(
        f"  Target bounds in NAIP CRS:\n"
        f"    left   = {target_bounds[0]:.3f}\n"
        f"    bottom = {target_bounds[1]:.3f}\n"
        f"    right  = {target_bounds[2]:.3f}\n"
        f"    top    = {target_bounds[3]:.3f}"
    )

    print(
        "\nSTEP 4: CREATING COMMON 1-METER GRID"
    )

    (
        target_transform,
        target_width,
        target_height,
        exact_bounds,
    ) = create_target_grid(
        target_bounds,
        TARGET_RESOLUTION,
    )

    print(
        f"  Resolution: "
        f"{TARGET_RESOLUTION} m"
    )

    print(
        f"  Output size: "
        f"{target_width} x {target_height}"
    )

    print(
        f"  Output bounds: "
        f"{exact_bounds}"
    )

    print(
        "\nSTEP 5: RESAMPLING NAIP TO 1 METER"
    )

    with rasterio.open(naip_path) as src:

        if USE_RGB_ONLY:
            band_indexes = [1, 2, 3]

            if src.count < 3:
                raise RuntimeError(
                    "NAIP source has fewer than "
                    "3 bands."
                )
        else:
            band_indexes = list(
                range(1, src.count + 1)
            )

        output_count = len(band_indexes)

        profile = src.profile.copy()

        profile.update(
            driver="GTiff",
            height=target_height,
            width=target_width,
            count=output_count,
            dtype=naip_dtype,
            crs=naip_crs,
            transform=target_transform,
            compress="deflate",
            predictor=2,
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        # We deliberately do not propagate the original
        # source NoData because the RGB imagery may not
        # have one.
        profile.pop("nodata", None)

        with rasterio.open(
            naip_output,
            "w",
            **profile,
        ) as dst:

            for out_index, src_index in enumerate(
                band_indexes,
                start=1,
            ):

                reproject(
                    source=rasterio.band(
                        src,
                        src_index,
                    ),
                    destination=rasterio.band(
                        dst,
                        out_index,
                    ),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=target_transform,
                    dst_crs=naip_crs,
                    resampling=Resampling.bilinear,
                )

    print(f"  Saved: {naip_output}")

    print(
        "\nSTEP 6: REPROJECTING DEM TO "
        "COMMON NAIP GRID"
    )

    with rasterio.open(dem_path) as src:

        dem_profile = src.profile.copy()

        dem_profile.update(
            driver="GTiff",
            height=target_height,
            width=target_width,
            count=1,
            dtype="float32",
            crs=naip_crs,
            transform=target_transform,
            compress="deflate",
            predictor=3,
            tiled=True,
            BIGTIFF="IF_SAFER",
            nodata=-999999.0,
        )

        with rasterio.open(
            dem_output,
            "w",
            **dem_profile,
        ) as dst:

            reproject(
                source=rasterio.band(
                    src,
                    1,
                ),
                destination=rasterio.band(
                    dst,
                    1,
                ),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=naip_crs,
                src_nodata=src.nodata,
                dst_nodata=-999999.0,
                resampling=Resampling.bilinear,
            )

    print(f"  Saved: {dem_output}")

    print("\nSTEP 7: FINAL ALIGNMENT CHECK")

    with rasterio.open(naip_output) as naip:
        with rasterio.open(dem_output) as dem:

            same_crs = (
                naip.crs == dem.crs
            )

            same_width = (
                naip.width == dem.width
            )

            same_height = (
                naip.height == dem.height
            )

            same_resolution = (
                naip.res == dem.res
            )

            same_transform = (
                naip.transform
                == dem.transform
            )

            same_bounds = (
                naip.bounds
                == dem.bounds
            )

            print(
                f"  Same CRS: "
                f"{same_crs}"
            )

            print(
                f"  Same width: "
                f"{same_width}"
            )

            print(
                f"  Same height: "
                f"{same_height}"
            )

            print(
                f"  Same resolution: "
                f"{same_resolution}"
            )

            print(
                f"  Same transform: "
                f"{same_transform}"
            )

            print(
                f"  Same bounds: "
                f"{same_bounds}"
            )

            passed = all(
                [
                    same_crs,
                    same_width,
                    same_height,
                    same_resolution,
                    same_transform,
                    same_bounds,
                ]
            )

    if not passed:
        raise RuntimeError(
            f"Alignment verification FAILED "
            f"for {location_name}."
        )

    print(
        f"\nALIGNMENT PASSED: "
        f"{location_name.upper()}"
    )

    return {
        "naip": naip_output,
        "dem": dem_output,
        "width": target_width,
        "height": target_height,
    }


def main():

    separator()
    print(
        "MSFFAN PROJECT - "
        "MULTI-LOCATION ALIGNMENT"
    )
    separator()

    print(
        "\nTarget working resolution: "
        f"{TARGET_RESOLUTION} meter"
    )

    print(
        "Optical bands used: RGB"
    )

    results = {}

    for location_name, paths in LOCATIONS.items():

        try:
            results[location_name] = align_location(
                location_name,
                paths["naip"],
                paths["dem"],
            )

        except Exception as e:

            print("\n")
            separator()
            print(
                f"ALIGNMENT FAILED: "
                f"{location_name.upper()}"
            )
            separator()

            print(
                f"Error: "
                f"{type(e).__name__}: {e}"
            )

            raise

    print("\n")
    separator()
    print("ALL LOCATIONS ALIGNED SUCCESSFULLY")
    separator()

    for location, result in results.items():

        print(f"\n{location.upper()}")

        print(
            f"  NAIP: {result['naip']}"
        )

        print(
            f"  DEM:  {result['dem']}"
        )

        print(
            f"  Grid: "
            f"{result['width']} x "
            f"{result['height']}"
        )

    print("\n")
    separator()
    print("MULTI-LOCATION ALIGNMENT COMPLETE")
    separator()


if __name__ == "__main__":
    main()