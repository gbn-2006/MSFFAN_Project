import os
import glob
import rasterio


DEM_DIR = "data/source/DEM"


print()
print("======================================")
print("CHECKING DOWNLOADED DEM")
print("======================================")


files = glob.glob(
    os.path.join(
        DEM_DIR,
        "*.tif"
    )
)


print(
    "DEM files found:",
    len(files)
)


for file in files:

    print()
    print("--------------------------------------")

    print(
        "File:",
        os.path.basename(file)
    )

    with rasterio.open(file) as src:

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
            "Bounds:",
            src.bounds
        )

        print(
            "Data type:",
            src.dtypes[0]
        )

        print(
            "NoData:",
            src.nodata
        )


print()
print("======================================")
print("DEM CHECK COMPLETE")
print("======================================")