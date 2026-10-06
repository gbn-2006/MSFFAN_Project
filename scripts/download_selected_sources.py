from pathlib import Path
import requests
import pystac_client
import planetary_computer

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "source"

LOCATIONS = {
    "baltimore": {
        "naip_id": "md_m_3907644_sw_18_030_20230525_20231018",
        "dem_url": (
            "https://prd-tnm.s3.amazonaws.com/"
            "StagedProducts/Elevation/S1M/n19e16/"
            "n1980e1640/S1M_n1980e1640_20260807.tif"
        ),
        "dem_name": "S1M_n1980e1640_20260807.tif",
    },

    "denver": {
        "naip_id": "co_m_3910417_nw_13_060_20210728",
        "dem_url": (
            "https://prd-tnm.s3.amazonaws.com/"
            "StagedProducts/Elevation/1m/Projects/"
            "CO_DRCOG_2020_B20/TIFF/"
            "USGS_1M_13_x50y440_CO_DRCOG_2020_B20.tif"
        ),
        "dem_name":
            "USGS_1M_13_x50y440_CO_DRCOG_2020_B20.tif",
    },
}


# ============================================================
# DOWNLOAD FUNCTION
# ============================================================

def download_file(url, destination):

    if destination.exists():
        print(f"SKIP: already exists")
        print(f"  {destination}")
        return

    print(f"Downloading:")
    print(f"  {url}")
    print(f"To:")
    print(f"  {destination}")

    with requests.get(url, stream=True, timeout=120) as response:

        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(destination, "wb") as f:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if not chunk:
                    continue

                f.write(chunk)
                downloaded += len(chunk)

                if total:
                    percent = downloaded / total * 100
                    print(
                        f"\r  Progress: {percent:6.2f}%",
                        end="",
                        flush=True
                    )

    print("\n  DONE")


# ============================================================
# CONNECT TO PLANETARY COMPUTER
# ============================================================

print("=" * 70)
print("SELECTED SOURCE DATA DOWNLOAD")
print("=" * 70)

print("\nConnecting to Planetary Computer...")

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

print("Connected.")


# ============================================================
# PROCESS LOCATIONS
# ============================================================

for name, config in LOCATIONS.items():

    print("\n" + "=" * 70)
    print(f"LOCATION: {name.upper()}")
    print("=" * 70)

    naip_dir = SOURCE / name / "NAIP"
    dem_dir = SOURCE / name / "DEM"

    naip_dir.mkdir(parents=True, exist_ok=True)
    dem_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # GET EXACT NAIP ITEM
    # --------------------------------------------------------

    print("\nFinding exact NAIP scene...")

    search = catalog.search(
        collections=["naip"],
        ids=[config["naip_id"]],
    )

    items = list(search.item_collection())

    if not items:
        print("ERROR: exact NAIP scene not found.")
        continue

    item = items[0]

    print(f"Found:")
    print(f"  {item.id}")
    print(f"  Date: {item.datetime}")

    image_asset = item.assets.get("image")

    if image_asset is None:
        print("ERROR: NAIP 'image' asset not found.")
        continue

    naip_url = planetary_computer.sign(
        image_asset.href
    )

    naip_path = (
        naip_dir /
        f"{item.id}.tif"
    )

    # --------------------------------------------------------
    # DOWNLOAD NAIP
    # --------------------------------------------------------

    print("\n--- NAIP ---")

    download_file(
        naip_url,
        naip_path
    )

    # --------------------------------------------------------
    # DOWNLOAD DEM
    # --------------------------------------------------------

    print("\n--- DEM ---")

    dem_path = (
        dem_dir /
        config["dem_name"]
    )

    download_file(
        config["dem_url"],
        dem_path
    )


print("\n" + "=" * 70)
print("DOWNLOAD STAGE COMPLETE")
print("=" * 70)