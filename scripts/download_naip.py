import pystac_client

print("Starting NAIP test...")

# Planetary Computer STAC API
URL = "https://planetarycomputer.microsoft.com/api/stac/v1/"

# Same test area we used earlier
BBOX = [
    -122.45,  # west
    37.70,    # south
    -122.40,  # east
    37.75     # north
]

try:

    print("Connecting to Planetary Computer...")

    catalog = pystac_client.Client.open(
        URL
    )

    print("Connected successfully.")

    print("Searching for NAIP imagery...")

    search = catalog.search(
        collections=["naip"],
        bbox=BBOX
    )

    print("Search completed.")

    items = list(
        search.items()
    )

    print()
    print("==============================")
    print("NAIP SEARCH RESULT")
    print("==============================")

    print(
        "Number of NAIP scenes found:",
        len(items)
    )

    if len(items) == 0:

        print()
        print(
            "No NAIP imagery was found "
            "for this area."
        )

    else:

        print()
        print("Available scenes:")

        for i, item in enumerate(
            items[:10]
        ):

            print()
            print(
                "Scene:",
                i
            )

            print(
                "ID:",
                item.id
            )

            print(
                "Date:",
                item.datetime
            )

            print(
                "Assets:",
                list(item.assets.keys())
            )

    print()
    print("NAIP test finished.")

except Exception as e:

    print()
    print("==============================")
    print("ERROR")
    print("==============================")

    print(
        type(e).__name__,
        ":",
        e
    )