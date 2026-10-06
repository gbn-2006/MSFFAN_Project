import os
import rasterio


NAIP_FILE = (
    "data/source/NAIP/NAIP_base.tif"
)


print()
print("======================================")
print("CHECKING NAIP")
print("======================================")


if not os.path.exists(NAIP_FILE):

    print(
        "ERROR:"
    )

    print(
        "NAIP_base.tif was not found."
    )

    print(
        "Expected location:"
    )

    print(
        NAIP_FILE
    )

    raise SystemExit


with rasterio.open(
    NAIP_FILE
) as src:

    print(
        "CRS:",
        src.crs
    )

    print(
        "Width:",
        src.width
    )

    print(
        "Height:",
        src.height
    )

    print(
        "Resolution:",
        src.res
    )

    print(
        "Bands:",
        src.count
    )

    print(
        "Bounds:",
        src.bounds
    )

    print(
        "Data type:",
        src.dtypes[0]
    )


print()
print("======================================")
print("NAIP CHECK COMPLETE")
print("======================================")