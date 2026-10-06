import requests
import pystac_client


# ============================================================
# CANDIDATE LOCATIONS
# ============================================================

LOCATIONS = {

    "san_francisco_ca": {
        "bbox": [
            -122.45,
            37.70,
            -122.40,
            37.75
        ]
    },

    "baltimore_md": {
        "bbox": [
            -76.66,
            39.25,
            -76.64,
            39.27
        ]
    },

    "austin_tx": {
        "bbox": [
            -97.76,
            30.25,
            -97.74,
            30.27
        ]
    },

    "denver_co": {
        "bbox": [
            -105.02,
            39.72,
            -105.00,
            39.74
        ]
    }

}


# ============================================================
# PLANETARY COMPUTER STAC
# ============================================================

STAC_URL = (
    "https://planetarycomputer.microsoft.com/"
    "api/stac/v1/"
)


# ============================================================
# USGS TNM API
# ============================================================

USGS_URL = (
    "https://tnmaccess.nationalmap.gov/"
    "api/v1/products"
)


# ============================================================
# CONNECT TO PLANETARY COMPUTER
# ============================================================

print()
print("==============================================")
print("DOM + DEM LOCATION AVAILABILITY CHECK")
print("==============================================")

print()
print("Connecting to Planetary Computer...")

try:

    catalog = pystac_client.Client.open(
        STAC_URL
    )

    print(
        "Planetary Computer connected."
    )

except Exception as e:

    print()
    print("ERROR connecting to Planetary Computer:")

    print(
        type(e).__name__,
        ":",
        e
    )

    raise SystemExit


# ============================================================
# CHECK LOCATIONS
# ============================================================

for name, info in LOCATIONS.items():

    bbox = info["bbox"]


    print()
    print()
    print("==============================================")
    print("LOCATION:", name)
    print("==============================================")

    print(
        "BBOX:",
        bbox
    )


    # ========================================================
    # NAIP CHECK
    # ========================================================

    print()
    print("----------------------------------------------")
    print("CHECKING NAIP")
    print("----------------------------------------------")

    naip_available = False
    naip_items = []


    try:

        search = catalog.search(
            collections=["naip"],
            bbox=bbox
        )

        naip_items = list(
            search.items()
        )


        print(
            "NAIP scenes found:",
            len(naip_items)
        )


        if len(naip_items) > 0:

            naip_available = True

            print()
            print(
                "Example NAIP scenes:"
            )


            for item in naip_items[:3]:

                print(
                    "  ID:",
                    item.id
                )

                print(
                    "  Date:",
                    item.datetime
                )

                print()


        else:

            print(
                "NO NAIP FOUND"
            )


    except Exception as e:

        print(
            "NAIP ERROR:",
            type(e).__name__,
            ":",
            e
        )


    # ========================================================
    # USGS DEM CHECK
    # ========================================================

    print()
    print("----------------------------------------------")
    print("CHECKING USGS DEM PRODUCTS")
    print("----------------------------------------------")


    s1m_products = []
    project_1m_products = []


    params = {

        "bbox": (
            f"{bbox[0]},"
            f"{bbox[1]},"
            f"{bbox[2]},"
            f"{bbox[3]}"
        ),

        "max": 50

    }


    try:

        response = requests.get(
            USGS_URL,
            params=params,
            timeout=(10, 120)
        )


        print(
            "USGS HTTP status:",
            response.status_code
        )


        response.raise_for_status()


        data = response.json()


        items = data.get(
            "items",
            []
        )


        print(
            "Total USGS products:",
            len(items)
        )


        # ====================================================
        # IDENTIFY 1-METER PRODUCTS
        # ====================================================

        for item in items:

            title = str(
                item.get(
                    "title",
                    ""
                )
            )


            dataset = str(
                item.get(
                    "dataset",
                    ""
                )
            )


            name_field = str(
                item.get(
                    "name",
                    ""
                )
            )


            text = (
                title
                + " "
                + dataset
                + " "
                + name_field
            ).lower()


            # ------------------------------------------------
            # S1M
            # ------------------------------------------------

            if "s1m" in text:

                s1m_products.append(
                    item
                )


            # ------------------------------------------------
            # Traditional/project 1 m DEM
            # ------------------------------------------------

            elif (
                "1 meter" in text
                or
                "1m" in text
            ):

                project_1m_products.append(
                    item
                )


        # ====================================================
        # DISPLAY S1M PRODUCTS
        # ====================================================

        print()
        print(
            "S1M products:",
            len(s1m_products)
        )


        if len(s1m_products) > 0:

            print()
            print(
                "S1M PRODUCTS:"
            )


            for item in s1m_products[:10]:

                print(
                    "  Title:",
                    item.get("title")
                )

                print(
                    "  Format:",
                    item.get("format")
                )

                print(
                    "  Download URL:",
                    item.get("downloadURL")
                )

                print()


        # ====================================================
        # DISPLAY PROJECT-BASED 1 M PRODUCTS
        # ====================================================

        print(
            "Project-based 1 m products:",
            len(project_1m_products)
        )


        if len(project_1m_products) > 0:

            print()
            print(
                "PROJECT-BASED 1 M PRODUCTS:"
            )


            for item in project_1m_products[:10]:

                print(
                    "  Title:",
                    item.get("title")
                )

                print(
                    "  Format:",
                    item.get("format")
                )

                print(
                    "  Download URL:",
                    item.get("downloadURL")
                )

                print()


    except requests.exceptions.Timeout:

        print()
        print(
            "USGS ERROR: Request timed out."
        )


    except requests.exceptions.HTTPError as e:

        print()
        print(
            "USGS HTTP ERROR:"
        )

        print(e)


    except Exception as e:

        print()
        print(
            "USGS ERROR:",
            type(e).__name__,
            ":",
            e
        )


    # ========================================================
    # DETERMINE DEM AVAILABILITY
    # ========================================================

    dem_available = (

        len(s1m_products) > 0

        or

        len(project_1m_products) > 0

    )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print("----------------------------------------------")


    if (
        naip_available
        and
        dem_available
    ):

        print(
            "RESULT: GOOD CANDIDATE"
        )

        print(
            "Both NAIP and 1 m DEM are available."
        )


    elif naip_available:

        print(
            "RESULT: NAIP ONLY"
        )

        print(
            "NAIP is available, but no suitable"
        )

        print(
            "1 m DEM was found."
        )


    elif dem_available:

        print(
            "RESULT: DEM ONLY"
        )

        print(
            "1 m DEM is available, but NAIP"
        )

        print(
            "was not found."
        )


    else:

        print(
            "RESULT: NOT SUITABLE"
        )

        print(
            "Neither suitable NAIP nor 1 m DEM"
        )

        print(
            "was found."
        )


# ============================================================
# FINISHED
# ============================================================

print()
print()
print("==============================================")
print("LOCATION SEARCH COMPLETE")
print("==============================================")