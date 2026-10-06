import pystac_client
import planetary_computer
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import box


# ============================================================
# CONFIGURATION
# ============================================================

LOCATIONS = {

    "baltimore_md": {
        "bbox": [-76.68, 39.23, -76.62, 39.29],

        "dem_urls": [
            (
                "S1M_n1980e1640_20260807",
                "https://prd-tnm.s3.amazonaws.com/"
                "StagedProducts/Elevation/S1M/n19e16/"
                "n1980e1640/S1M_n1980e1640_20260807.tif"
            ),

            (
                "MD_4County_D24",
                "https://prd-tnm.s3.amazonaws.com/"
                "StagedProducts/Elevation/1m/Projects/"
                "MD_4County_D24/TIFF/"
                "USGS_1M_18_x35y435_MD_4County_D24.tif"
            ),

            (
                "MD_Central_Processing_D24",
                "https://prd-tnm.s3.amazonaws.com/"
                "StagedProducts/Elevation/1m/Projects/"
                "MD_Central_Processing_D24/TIFF/"
                "USGS_1M_18_x35y435_MD_Central_Processing_D24.tif"
            ),

            (
                "VA_UpperMiddleNeck_2018_D18",
                "https://prd-tnm.s3.amazonaws.com/"
                "StagedProducts/Elevation/1m/Projects/"
                "VA_UpperMiddleNeck_2018_D18/TIFF/"
                "USGS_1M_18_x35y435_VA_UpperMiddleNeck_2018_D18.tif"
            ),
        ],
    },

    "denver_co": {
        "bbox": [-105.02, 39.72, -105.00, 39.74],

        "dem_urls": [
            (
                "S1M_n1900w0770_20250521",
                "https://prd-tnm.s3.amazonaws.com/"
                "StagedProducts/Elevation/S1M/n19w07/"
                "n1900w0770/S1M_n1900w0770_20250521.tif"
            ),

            (
                "CO_DRCOG_2020_x49y440",
                "https://prd-tnm.s3.amazonaws.com/"
                "StagedProducts/Elevation/1m/Projects/"
                "CO_DRCOG_2020_B20/TIFF/"
                "USGS_1M_13_x49y440_CO_DRCOG_2020_B20.tif"
            ),

            (
                "CO_DRCOG_2020_x50y440",
                "https://prd-tnm.s3.amazonaws.com/"
                "StagedProducts/Elevation/1m/Projects/"
                "CO_DRCOG_2020_B20/TIFF/"
                "USGS_1M_13_x50y440_CO_DRCOG_2020_B20.tif"
            ),
        ],
    },
}


# ============================================================
# OVERLAP FUNCTION
# ============================================================

def calculate_overlap(naip_bounds, naip_crs, dem_bounds, dem_crs):

    transformed_bounds = transform_bounds(
        naip_crs,
        dem_crs,
        *naip_bounds
    )

    naip_box = box(*transformed_bounds)
    dem_box = box(*dem_bounds)

    intersection = naip_box.intersection(dem_box)

    if intersection.is_empty:
        return 0.0, 0.0

    naip_area = naip_box.area
    intersection_area = intersection.area

    percentage = (
        intersection_area / naip_area
    ) * 100.0

    return percentage, intersection_area


# ============================================================
# CONNECT
# ============================================================

print("=" * 70)
print("BEST NAIP + DEM PAIR SEARCH")
print("=" * 70)

print("\nConnecting to Planetary Computer...")

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

print("Planetary Computer connected.")


# ============================================================
# PROCESS EACH LOCATION
# ============================================================

for location_name, config in LOCATIONS.items():

    print("\n" + "=" * 70)
    print(f"LOCATION: {location_name}")
    print("=" * 70)

    print(f"BBOX: {config['bbox']}")

    # --------------------------------------------------------
    # SEARCH NAIP
    # --------------------------------------------------------

    search = catalog.search(
        collections=["naip"],
        bbox=config["bbox"],
    )

    items = list(search.item_collection())

    print(f"\nNAIP scenes found: {len(items)}")

    if not items:
        print("No NAIP scenes found.")
        continue

    # --------------------------------------------------------
    # OPEN DEM PRODUCTS
    # --------------------------------------------------------

    dem_info = []

    print("\nOpening DEM products...")

    for dem_name, dem_url in config["dem_urls"]:

        try:

            with rasterio.open(dem_url) as dem:

                dem_info.append({
                    "name": dem_name,
                    "url": dem_url,
                    "bounds": dem.bounds,
                    "crs": dem.crs,
                })

                print(
                    f"  OK: {dem_name} | "
                    f"CRS={dem.crs} | "
                    f"Bounds={dem.bounds}"
                )

        except Exception as e:

            print(f"  FAILED: {dem_name}")
            print(f"    {e}")

    if not dem_info:
        print("No DEM products could be opened.")
        continue

    # --------------------------------------------------------
    # TEST EVERY NAIP × DEM COMBINATION
    # --------------------------------------------------------

    results = []

    print("\nCalculating all NAIP × DEM overlaps...")

    for item in items:

        image_asset = None

        for key, asset in item.assets.items():

            media_type = asset.media_type or ""

            if (
                "image/tiff" in media_type.lower()
                or key.lower() in ["image", "visual", "analytic"]
            ):
                image_asset = asset
                break

        if image_asset is None:
            continue

        try:

            signed_url = planetary_computer.sign(
                image_asset.href
            )

            with rasterio.open(signed_url) as src:

                naip_bounds = src.bounds
                naip_crs = src.crs

                for dem in dem_info:

                    percentage, area = calculate_overlap(
                        naip_bounds,
                        naip_crs,
                        dem["bounds"],
                        dem["crs"],
                    )

                    results.append({
                        "naip_id": item.id,
                        "date": item.datetime,
                        "dem": dem["name"],
                        "percentage": percentage,
                        "area": area,
                    })

        except Exception as e:

            print(
                f"WARNING: Could not inspect {item.id}: {e}"
            )

    # --------------------------------------------------------
    # SORT RESULTS
    # --------------------------------------------------------

    results.sort(
        key=lambda x: x["percentage"],
        reverse=True
    )

    print("\n" + "-" * 70)
    print("TOP OVERLAP RESULTS")
    print("-" * 70)

    for i, result in enumerate(results[:10], start=1):

        print(
            f"\n#{i}"
            f"\n  NAIP: {result['naip_id']}"
            f"\n  Date: {result['date']}"
            f"\n  DEM: {result['dem']}"
            f"\n  Coverage: {result['percentage']:.2f}%"
            f"\n  Intersection: {result['area']:.2f} m²"
        )

    # --------------------------------------------------------
    # FINAL RECOMMENDATION
    # --------------------------------------------------------

    if results:

        best = results[0]

        print("\n" + "-" * 70)
        print("BEST AVAILABLE PAIR")
        print("-" * 70)

        print(f"NAIP: {best['naip_id']}")
        print(f"DEM: {best['dem']}")
        print(f"Coverage: {best['percentage']:.2f}%")

        if best["percentage"] >= 90:

            print("\nSTATUS: EXCELLENT")
            print("Use this pair.")

        elif best["percentage"] >= 75:

            print("\nSTATUS: GOOD")
            print(
                "Potentially usable, but we should inspect "
                "the actual overlapping region."
            )

        elif best["percentage"] >= 50:

            print("\nSTATUS: PARTIAL")
            print(
                "Do not lock yet. Search for another "
                "location/DEM pair if possible."
            )

        else:

            print("\nSTATUS: INSUFFICIENT")
            print(
                "This location should not be locked "
                "with the currently available products."
            )

    else:

        print("No valid NAIP/DEM combinations found.")


print("\n" + "=" * 70)
print("BEST PAIR SEARCH COMPLETE")
print("=" * 70)