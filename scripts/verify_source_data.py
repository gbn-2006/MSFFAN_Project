import os
import rasterio
from rasterio.warp import transform_bounds
from rasterio.coords import BoundingBox


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


SOURCES = {
    "SAN FRANCISCO": {
        "NAIP": os.path.join(
            PROJECT_ROOT,
            "data", "source", "NAIP", "NAIP_base.tif"
        ),
        "DEM": os.path.join(
            PROJECT_ROOT,
            "data", "source", "DEM",
            "USGS_1M_10_x54y418_CA_SanFrancisco_B23.tif"
        ),
    },

    "BALTIMORE": {
        "NAIP": os.path.join(
            PROJECT_ROOT,
            "data", "source", "baltimore", "NAIP",
            "md_m_3907644_sw_18_030_20230525_20231018.tif"
        ),
        "DEM": os.path.join(
            PROJECT_ROOT,
            "data", "source", "baltimore", "DEM",
            "S1M_n1980e1640_20260807.tif"
        ),
    },

    "DENVER": {
        "NAIP": os.path.join(
            PROJECT_ROOT,
            "data", "source", "denver", "NAIP",
            "co_m_3910417_nw_13_060_20210728.tif"
        ),
        "DEM": os.path.join(
            PROJECT_ROOT,
            "data", "source", "denver", "DEM",
            "USGS_1M_13_x50y440_CO_DRCOG_2020_B20.tif"
        ),
    },
}


def print_separator():
    print("=" * 70)


def inspect_raster(label, filepath):
    print(f"\n--- {label} ---")
    print(f"File: {filepath}")

    if not os.path.exists(filepath):
        print("READABLE: NO")
        print("STATUS: FILE DOES NOT EXIST")
        return None

    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")

    try:
        with rasterio.open(filepath) as src:

            # Force reading metadata and one small pixel window.
            # This helps verify that the GeoTIFF is actually readable.
            sample_width = min(10, src.width)
            sample_height = min(10, src.height)
            src.read(
                1,
                window=rasterio.windows.Window(
                    0, 0, sample_width, sample_height
                )
            )

            print("READABLE: YES")
            print(f"Driver: {src.driver}")
            print(f"CRS: {src.crs}")
            print(f"Width: {src.width}")
            print(f"Height: {src.height}")
            print(f"Bands: {src.count}")
            print(f"Resolution: {src.res}")
            print(f"Data type: {src.dtypes}")
            print(f"NoData: {src.nodata}")
            print(f"Bounds: {src.bounds}")
            print(f"Transform: {src.transform}")

            return {
                "crs": src.crs,
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "resolution": src.res,
                "bounds": src.bounds,
                "transform": src.transform,
                "nodata": src.nodata,
            }

    except Exception as e:
        print("READABLE: NO")
        print(f"ERROR: {type(e).__name__}: {e}")
        return None


def calculate_overlap(naip_info, dem_info):
    if naip_info is None or dem_info is None:
        return

    print("\n--- SPATIAL OVERLAP CHECK ---")

    try:
        naip_bounds_in_dem = transform_bounds(
            naip_info["crs"],
            dem_info["crs"],
            *naip_info["bounds"],
            densify_pts=21,
        )

        dem_bounds = dem_info["bounds"]

        left = max(naip_bounds_in_dem[0], dem_bounds.left)
        bottom = max(naip_bounds_in_dem[1], dem_bounds.bottom)
        right = min(naip_bounds_in_dem[2], dem_bounds.right)
        top = min(naip_bounds_in_dem[3], dem_bounds.top)

        if left < right and bottom < top:
            overlap_width = right - left
            overlap_height = top - bottom
            overlap_area = overlap_width * overlap_height

            print("OVERLAP: YES")
            print(f"Overlap width: {overlap_width:.2f}")
            print(f"Overlap height: {overlap_height:.2f}")
            print(f"Approx. overlap area: {overlap_area:,.2f} square CRS units")

        else:
            print("OVERLAP: NO")

    except Exception as e:
        print(f"OVERLAP CHECK ERROR: {type(e).__name__}: {e}")


def main():
    print_separator()
    print("MSFFAN PROJECT - SOURCE DATA VERIFICATION")
    print_separator()

    all_passed = True

    for location, files in SOURCES.items():

        print("\n")
        print_separator()
        print(f"LOCATION: {location}")
        print_separator()

        naip_info = inspect_raster("NAIP / OPTICAL IMAGE", files["NAIP"])
        dem_info = inspect_raster("DEM / ELEVATION DATA", files["DEM"])

        if naip_info is None or dem_info is None:
            all_passed = False

        calculate_overlap(naip_info, dem_info)

    print("\n")
    print_separator()
    print("VERIFICATION SUMMARY")
    print_separator()

    if all_passed:
        print("All required source files were found and readable.")
        print("Source verification stage: PASSED")
    else:
        print("One or more source files failed verification.")
        print("Source verification stage: FAILED")

    print_separator()


if __name__ == "__main__":
    main()