import pystac_client
import planetary_computer
import requests
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import box
from shapely.ops import transform
from pyproj import Transformer


# ============================================================
# CONFIGURATION
# ============================================================

LOCATIONS = {

    "baltimore_md": {
        "bbox": [-76.66, 39.25, -76.64, 39.27],
        "naip_collection": "naip",
        "dem_url": (
            "https://prd-tnm.s3.amazonaws.com/"
            "StagedProducts/Elevation/S1M/n19e16/"
            "n1980e1640/S1M_n1980e1640_20260807.tif"
        ),
    },

    "denver_co": {
        "bbox": [-105.02, 39.72, -105.00, 39.74],
        "naip_collection": "naip",
        "dem_url": (
            "https://prd-tnm.s3.amazonaws.com/"
            "StagedProducts/Elevation/S1M/n19w07/"
            "n1900w0770/S1M_n1900w0770_20250521.tif"
        ),
    },
}


# ============================================================
# HELPERS
# ============================================================

def print_bounds(label, bounds):
    print(f"{label}:")
    print(f"  left   = {bounds.left}")
    print(f"  bottom = {bounds.bottom}")
    print(f"  right  = {bounds.right}")
    print(f"  top    = {bounds.top}")


def calculate_overlap(naip_bounds, naip_crs, dem_bounds, dem_crs):

    # Transform NAIP bounds into DEM CRS
    transformed = transform_bounds(
        naip_crs,
        dem_crs,
        *naip_bounds
    )

    naip_in_dem_crs = box(*transformed)
    dem_box = box(*dem_bounds)

    intersection = naip_in_dem_crs.intersection(dem_box)

    if intersection.is_empty:
        return 0.0, 0.0, intersection

    naip_area = naip_in_dem_crs.area
    intersection_area = intersection.area

    percentage = (intersection_area / naip_area) * 100.0

    return percentage, intersection_area, intersection


# ============================================================
# CONNECT TO PLANETARY COMPUTER
# ============================================================

print("=" * 60)
print("EXACT NAIP + DEM OVERLAP VERIFICATION")
print("=" * 60)

print("\nConnecting to Planetary Computer...")

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

print("Planetary Computer connected.")


# ============================================================
# PROCESS LOCATIONS
# ============================================================

for location_name, config in LOCATIONS.items():

    print("\n" + "=" * 60)
    print(f"LOCATION: {location_name}")
    print("=" * 60)

    bbox = config["bbox"]

    print(f"BBOX: {bbox}")

    # --------------------------------------------------------
    # NAIP SEARCH
    # --------------------------------------------------------

    print("\n----------------------------------------------")
    print("SEARCHING NAIP")
    print("----------------------------------------------")

    search = catalog.search(
        collections=[config["naip_collection"]],
        bbox=bbox,
    )

    items = list(search.item_collection())

    print(f"NAIP scenes found: {len(items)}")

    if not items:
        print("ERROR: No NAIP scenes found.")
        continue

    # Use first scene for exact footprint inspection
    naip_item = items[0]

    print(f"\nSelected NAIP scene:")
    print(f"  ID: {naip_item.id}")
    print(f"  Date: {naip_item.datetime}")

    # --------------------------------------------------------
    # FIND NAIP RASTER ASSET
    # --------------------------------------------------------

    image_asset = None

    for key, asset in naip_item.assets.items():

        media_type = asset.media_type or ""

        if (
            "image/tiff" in media_type.lower()
            or key.lower() in ["image", "visual", "analytic"]
        ):
            image_asset = asset
            print(f"\nNAIP raster asset: {key}")
            break

    if image_asset is None:
        print("ERROR: Could not identify NAIP raster asset.")
        continue

    # --------------------------------------------------------
    # READ NAIP METADATA
    # --------------------------------------------------------

    print("\nOpening NAIP raster...")

    try:
        signed_url = planetary_computer.sign(image_asset.href)

        with rasterio.open(signed_url) as src:

            naip_bounds = src.bounds
            naip_crs = src.crs

            print("\nNAIP metadata:")
            print(f"  CRS: {naip_crs}")
            print(f"  Width: {src.width}")
            print(f"  Height: {src.height}")
            print(f"  Resolution: {src.res}")

            print_bounds("  Bounds", naip_bounds)

    except Exception as e:

        print(f"ERROR opening NAIP raster: {e}")
        continue

    # --------------------------------------------------------
    # OPEN DEM
    # --------------------------------------------------------

    dem_url = config["dem_url"]

    print("\n----------------------------------------------")
    print("OPENING DEM")
    print("----------------------------------------------")

    print(f"DEM URL:")
    print(f"  {dem_url}")

    try:

        with rasterio.open(dem_url) as dem:

            dem_bounds = dem.bounds
            dem_crs = dem.crs

            print("\nDEM metadata:")
            print(f"  CRS: {dem_crs}")
            print(f"  Width: {dem.width}")
            print(f"  Height: {dem.height}")
            print(f"  Resolution: {dem.res}")
            print(f"  NoData: {dem.nodata}")

            print_bounds("  Bounds", dem_bounds)

    except Exception as e:

        print(f"ERROR opening DEM: {e}")
        continue

    # --------------------------------------------------------
    # OVERLAP CALCULATION
    # --------------------------------------------------------

    print("\n----------------------------------------------")
    print("CALCULATING EXACT OVERLAP")
    print("----------------------------------------------")

    try:

        percentage, area, intersection = calculate_overlap(
            naip_bounds,
            naip_crs,
            dem_bounds,
            dem_crs,
        )

        print(f"\nNAIP footprint covered by DEM: {percentage:.2f}%")
        print(f"Intersection area: {area:.2f} square meters")

        if percentage >= 90:

            print("\nRESULT: EXCELLENT OVERLAP")
            print("This NAIP + DEM pair is suitable.")

        elif percentage >= 50:

            print("\nRESULT: ACCEPTABLE OVERLAP")
            print("We can use this pair, but coverage is not complete.")

        elif percentage > 0:

            print("\nRESULT: POOR OVERLAP")
            print("Choose another NAIP scene or DEM tile.")

        else:

            print("\nRESULT: NO OVERLAP")
            print("This pair must NOT be used.")

    except Exception as e:

        print(f"ERROR calculating overlap: {e}")


print("\n" + "=" * 60)
print("OVERLAP VERIFICATION COMPLETE")
print("=" * 60)