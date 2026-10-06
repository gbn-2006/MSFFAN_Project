import os
import requests
from tqdm import tqdm


# ============================================================
# SETTINGS
# ============================================================

OUTPUT_DIR = "data/source/DEM"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# USGS 1-METER SAN FRANCISCO DEM
# ============================================================

DEM_URLS = [

    "https://prd-tnm.s3.amazonaws.com/"
    "StagedProducts/Elevation/1m/Projects/"
    "CA_SanFrancisco_B23/TIFF/"
    "USGS_1M_10_x54y418_CA_SanFrancisco_B23.tif",

    "https://prd-tnm.s3.amazonaws.com/"
    "StagedProducts/Elevation/1m/Projects/"
    "CA_SanFrancisco_B23/TIFF/"
    "USGS_1M_10_x55y418_CA_SanFrancisco_B23.tif"

]


# ============================================================
# DOWNLOAD
# ============================================================

def download_file(url):

    filename = os.path.basename(url)

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    print()
    print("======================================")
    print("Downloading:")
    print(filename)
    print("======================================")

    if os.path.exists(output_path):

        print("File already exists. Skipping.")

        return

    try:

        response = requests.get(
            url,
            stream=True,
            timeout=(10, 300)
        )

        print(
            "HTTP status:",
            response.status_code
        )

        response.raise_for_status()

        total_size = int(
            response.headers.get(
                "content-length",
                0
            )
        )

        print(
            "File size:",
            round(
                total_size / (1024 ** 2),
                2
            ),
            "MB"
        )

        with open(
            output_path,
            "wb"
        ) as file:

            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=filename
            ) as progress:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if chunk:

                        file.write(chunk)

                        progress.update(
                            len(chunk)
                        )

        print("Download completed!")

        print(
            "Saved to:",
            output_path
        )

    except Exception as e:

        print()
        print("DOWNLOAD FAILED:")

        print(
            type(e).__name__,
            ":",
            e
        )


# ============================================================
# MAIN
# ============================================================

print()
print("USGS 3DEP 1-METER DEM DOWNLOAD")
print("San Francisco, California")
print()

for url in DEM_URLS:

    download_file(url)

print()
print("======================================")
print("ALL DEM DOWNLOADS FINISHED")
print("======================================")