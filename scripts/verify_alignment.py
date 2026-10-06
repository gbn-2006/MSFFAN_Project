import rasterio


NAIP_FILE = "data/source/aligned/NAIP_1m.tif"
DEM_FILE = "data/source/aligned/DEM_1m.tif"


print()
print("======================================")
print("VERIFYING ALIGNMENT")
print("======================================")


# ------------------------------------------------------------
# Read NAIP metadata
# ------------------------------------------------------------

with rasterio.open(NAIP_FILE) as naip:

    naip_crs = naip.crs
    naip_width = naip.width
    naip_height = naip.height
    naip_res = naip.res
    naip_bounds = naip.bounds
    naip_transform = naip.transform

    print()
    print("NAIP")
    print("--------------------------------------")

    print("CRS:", naip_crs)
    print("Size:", naip_width, "x", naip_height)
    print("Resolution:", naip_res)
    print("Bounds:", naip_bounds)


# ------------------------------------------------------------
# Read DEM metadata
# ------------------------------------------------------------

with rasterio.open(DEM_FILE) as dem:

    dem_crs = dem.crs
    dem_width = dem.width
    dem_height = dem.height
    dem_res = dem.res
    dem_bounds = dem.bounds
    dem_transform = dem.transform

    print()
    print("DEM")
    print("--------------------------------------")

    print("CRS:", dem_crs)
    print("Size:", dem_width, "x", dem_height)
    print("Resolution:", dem_res)
    print("Bounds:", dem_bounds)


# ------------------------------------------------------------
# Compare
# ------------------------------------------------------------

print()
print("======================================")
print("COMPARISON")
print("======================================")


print(
    "Same CRS:",
    naip_crs == dem_crs
)

print(
    "Same width:",
    naip_width == dem_width
)

print(
    "Same height:",
    naip_height == dem_height
)

print(
    "Same resolution:",
    naip_res == dem_res
)

print(
    "Same bounds:",
    naip_bounds == dem_bounds
)

print(
    "Same transform:",
    naip_transform == dem_transform
)


# ------------------------------------------------------------
# Final result
# ------------------------------------------------------------

aligned = (
    naip_crs == dem_crs
    and
    naip_width == dem_width
    and
    naip_height == dem_height
    and
    naip_res == dem_res
    and
    naip_bounds == dem_bounds
    and
    naip_transform == dem_transform
)


print()

if aligned:

    print("======================================")
    print("RESULT: DATASETS ARE ALIGNED")
    print("======================================")

else:

    print("======================================")
    print("RESULT: DATASETS ARE NOT FULLY ALIGNED")
    print("======================================")