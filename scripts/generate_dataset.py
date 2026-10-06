"""
MSFFAN PROJECT
FINAL CONTROLLED DATASET GENERATOR

Dataset:
    Training   : 4096
    Validation : 500
    Testing    : 500
    Total      : 5096

Source locations:
    San Francisco
    Baltimore
    Denver

HR/LR generation:
    Aligned 1 m optical imagery
        ↓
    Spatially controlled HR patch extraction
        ↓
    Downsample ×2 / ×3 / ×4
        ↓
    Contrast adjustment
        ↓
    Gaussian / windowed-Sinc blur
        ↓
    JPEG compression
        ↓
    Upsample to HR dimensions
        ↓
    LR image

Important:
    This is a reproducible implementation of the documented
    degradation pipeline. The exact proprietary/undocumented
    DOM+DEM orbital simulation details from the paper are not
    assumed or invented.
"""

from pathlib import Path
import csv
import io
import random

import numpy as np
import rasterio
from PIL import Image, ImageEnhance
from scipy import ndimage


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "source"
    / "aligned"
)

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "dataset"
)

METADATA_PATH = (
    DATASET_ROOT
    / "dataset_metadata.csv"
)


# ============================================================
# SOURCE DATA
# ============================================================

LOCATIONS = {
    "san_francisco": {
        "optical": (
            SOURCE_ROOT
            / "san_francisco"
            / "NAIP_1m.tif"
        ),
        "dem": (
            SOURCE_ROOT
            / "san_francisco"
            / "DEM_1m.tif"
        ),
    },

    "baltimore": {
        "optical": (
            SOURCE_ROOT
            / "baltimore"
            / "NAIP_1m.tif"
        ),
        "dem": (
            SOURCE_ROOT
            / "baltimore"
            / "DEM_1m.tif"
        ),
    },

    "denver": {
        "optical": (
            SOURCE_ROOT
            / "denver"
            / "NAIP_1m.tif"
        ),
        "dem": (
            SOURCE_ROOT
            / "denver"
            / "DEM_1m.tif"
        ),
    },
}


# ============================================================
# DATASET CONFIGURATION
# ============================================================

TRAIN_SAMPLES = 4096
VAL_SAMPLES = 500
TEST_SAMPLES = 500

TOTAL_SAMPLES = (
    TRAIN_SAMPLES
    + VAL_SAMPLES
    + TEST_SAMPLES
)

HR_PATCH_SIZE = 256

SCALE_FACTORS = [2, 3, 4]

JPEG_MIN = 10
JPEG_MAX = 80

BLUR_TYPES = [
    "gaussian",
    "sinc",
]

RANDOM_SEED = 42


# ============================================================
# SPATIAL SPLIT PARAMETERS
# ============================================================
#
# Each split occupies a separate Y-region.
#
# The unused regions are guard bands to reduce spatial leakage.
#
# TRAIN: 0%   - 60%
# GUARD: 60%  - 65%
# VAL:   65%  - 80%
# GUARD: 80%  - 85%
# TEST:  85%  - 100%
#
# The HR patch is 256 pixels, so these gaps are intentionally
# larger than the patch size for all three source rasters.
# ============================================================

SPLIT_RANGES = {
    "train": (0.00, 0.60),
    "val":   (0.65, 0.80),
    "test":  (0.85, 1.00),
}


# ============================================================
# RANDOM STATE
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# DIRECTORY CREATION
# ============================================================

for split in ["train", "val", "test"]:

    (
        DATASET_ROOT
        / split
        / "HR"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        DATASET_ROOT
        / split
        / "LR"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# SAMPLE COUNT DISTRIBUTION
# ============================================================

def distribute_count(total, n):
    """
    Distribute 'total' samples as evenly as possible among n
    categories.

    Example:
        10 / 3 -> [4, 3, 3]
    """

    base = total // n
    remainder = total % n

    counts = [base] * n

    for i in range(remainder):
        counts[i] += 1

    return counts


# ============================================================
# CREATE BALANCED SAMPLE PLAN
# ============================================================

def create_sample_plan():

    locations = list(LOCATIONS.keys())

    plan = []

    split_counts = {
        "train": TRAIN_SAMPLES,
        "val": VAL_SAMPLES,
        "test": TEST_SAMPLES,
    }

    for split in ["train", "val", "test"]:

        total = split_counts[split]

        combinations = []

        for location in locations:

            for scale in SCALE_FACTORS:

                combinations.append(
                    (
                        location,
                        scale,
                    )
                )

        counts = distribute_count(
            total,
            len(combinations),
        )

        for combination, count in zip(
            combinations,
            counts,
        ):

            location, scale = combination

            for _ in range(count):

                plan.append(
                    {
                        "split": split,
                        "location": location,
                        "scale": scale,
                    }
                )

    # Deterministically shuffle the plan while preserving
    # the exact balance.
    random.shuffle(plan)

    return plan


# ============================================================
# READ RGB PATCH
# ============================================================

def read_rgb_patch(
    dataset,
    x,
    y,
    size,
):

    window = rasterio.windows.Window(
        col_off=x,
        row_off=y,
        width=size,
        height=size,
    )

    data = dataset.read(
        [1, 2, 3],
        window=window,
    )

    image = np.transpose(
        data,
        (1, 2, 0),
    )

    image = np.clip(
        image,
        0,
        255,
    ).astype(
        np.uint8
    )

    return image


# ============================================================
# WINDOWED SINC KERNEL
# ============================================================

def create_sinc_kernel(
    size=9,
    cutoff=0.45,
):

    if size % 2 == 0:
        raise ValueError(
            "Sinc kernel size must be odd."
        )

    radius = size // 2

    coords = np.arange(
        -radius,
        radius + 1,
        dtype=np.float64,
    )

    # Ideal low-pass sinc response.
    #
    # np.sinc(x) = sin(pi*x)/(pi*x)
    #
    # cutoff is expressed relative to Nyquist.

    kernel_1d = (
        2
        * cutoff
        * np.sinc(
            2
            * cutoff
            * coords
        )
    )

    # Hann window reduces ringing.
    window = (
        0.5
        + 0.5
        * np.cos(
            np.pi
            * coords
            / radius
        )
    )

    kernel_1d *= window

    kernel_1d /= kernel_1d.sum()

    kernel_2d = np.outer(
        kernel_1d,
        kernel_1d,
    )

    kernel_2d /= kernel_2d.sum()

    return kernel_2d


SINC_KERNEL = create_sinc_kernel(
    size=9,
    cutoff=0.45,
)


# ============================================================
# APPLY SINC BLUR
# ============================================================

def apply_sinc_blur(image):

    array = np.asarray(
        image,
        dtype=np.float32,
    )

    output = np.empty_like(
        array
    )

    for channel in range(3):

        output[:, :, channel] = (
            ndimage.convolve(
                array[:, :, channel],
                SINC_KERNEL,
                mode="reflect",
            )
        )

    output = np.clip(
        output,
        0,
        255,
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        output,
        mode="RGB",
    )


# ============================================================
# GAUSSIAN BLUR
# ============================================================

def apply_gaussian_blur(
    image,
):

    radius = random.uniform(
        0.5,
        1.5,
    )

    array = np.asarray(
        image,
        dtype=np.float32,
    )

    output = np.empty_like(
        array
    )

    for channel in range(3):

        output[:, :, channel] = (
            ndimage.gaussian_filter(
                array[:, :, channel],
                sigma=radius,
            )
        )

    output = np.clip(
        output,
        0,
        255,
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        output,
        mode="RGB",
    )


# ============================================================
# CONTRAST
# ============================================================

def apply_contrast(
    image,
):

    factor = random.uniform(
        0.8,
        1.2,
    )

    enhancer = (
        ImageEnhance.Contrast(
            image
        )
    )

    return (
        enhancer.enhance(
            factor
        ),
        factor,
    )


# ============================================================
# DOWNSAMPLE
# ============================================================

def downsample(
    image,
    scale,
):

    width, height = image.size

    new_width = width // scale
    new_height = height // scale

    return image.resize(
        (
            new_width,
            new_height,
        ),
        Image.Resampling.BICUBIC,
    )


# ============================================================
# JPEG COMPRESSION
# ============================================================

def jpeg_compress(
    image,
    quality,
):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=quality,
    )

    buffer.seek(0)

    compressed = (
        Image.open(
            buffer
        ).convert("RGB")
    )

    return compressed.copy()


# ============================================================
# UPSAMPLE
# ============================================================

def upsample(
    image,
    target_size,
):

    return image.resize(
        target_size,
        Image.Resampling.BICUBIC,
    )


# ============================================================
# DEGRADATION PIPELINE
# ============================================================

def degrade_image(
    hr_image,
    scale,
):

    jpeg_quality = random.randint(
        JPEG_MIN,
        JPEG_MAX,
    )

    blur_type = random.choice(
        BLUR_TYPES
    )

    # --------------------------------------------------------
    # 1. Downsample
    # --------------------------------------------------------

    image = downsample(
        hr_image,
        scale,
    )

    # --------------------------------------------------------
    # 2. Contrast
    # --------------------------------------------------------

    image, contrast_factor = (
        apply_contrast(
            image
        )
    )

    # --------------------------------------------------------
    # 3. Blur
    # --------------------------------------------------------

    if blur_type == "gaussian":

        image = apply_gaussian_blur(
            image
        )

    elif blur_type == "sinc":

        image = apply_sinc_blur(
            image
        )

    else:

        raise ValueError(
            f"Unsupported blur type: "
            f"{blur_type}"
        )

    # --------------------------------------------------------
    # 4. JPEG
    # --------------------------------------------------------

    image = jpeg_compress(
        image,
        jpeg_quality,
    )

    # --------------------------------------------------------
    # 5. Upsample
    # --------------------------------------------------------

    image = upsample(
        image,
        hr_image.size,
    )

    return (
        image,
        blur_type,
        jpeg_quality,
        contrast_factor,
    )


# ============================================================
# VALID PATCH RANGE
# ============================================================

def get_patch_limits(
    dataset,
    split,
):

    width = dataset.width
    height = dataset.height

    start_fraction, end_fraction = (
        SPLIT_RANGES[split]
    )

    y_start = int(
        height
        * start_fraction
    )

    y_end = int(
        height
        * end_fraction
    )

    # Ensure the patch remains entirely inside the region.
    max_y = (
        y_end
        - HR_PATCH_SIZE
    )

    min_y = y_start

    max_x = (
        width
        - HR_PATCH_SIZE
    )

    if max_y < min_y:
        raise ValueError(
            f"{split} region is too small "
            f"for a {HR_PATCH_SIZE}px patch."
        )

    if max_x < 0:
        raise ValueError(
            f"Raster width is too small "
            f"for a {HR_PATCH_SIZE}px patch."
        )

    return (
        0,
        max_x,
        min_y,
        max_y,
    )


# ============================================================
# CHOOSE PATCH
# ============================================================

def choose_patch(
    dataset,
    split,
):

    (
        min_x,
        max_x,
        min_y,
        max_y,
    ) = get_patch_limits(
        dataset,
        split,
    )

    x = random.randint(
        min_x,
        max_x,
    )

    y = random.randint(
        min_y,
        max_y,
    )

    return x, y


# ============================================================
# SAVE PNG
# ============================================================

def save_png(
    image,
    path,
):

    image.save(
        path,
        format="PNG",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "MSFFAN PROJECT - FINAL DATASET GENERATION"
    )
    print("=" * 75)

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Dataset:")
    print(
        f"  Training   : {TRAIN_SAMPLES}"
    )
    print(
        f"  Validation : {VAL_SAMPLES}"
    )
    print(
        f"  Testing    : {TEST_SAMPLES}"
    )
    print(
        f"  TOTAL      : {TOTAL_SAMPLES}"
    )

    print()
    print("HR patch size:", HR_PATCH_SIZE)

    print()
    print(
        "Scale factors:",
        SCALE_FACTORS,
    )

    print()
    print(
        "JPEG quality:",
        f"{JPEG_MIN}-{JPEG_MAX}",
    )

    print()
    print(
        "Blur types:",
        BLUR_TYPES,
    )

    print()
    print(
        "Random seed:",
        RANDOM_SEED,
    )

    # --------------------------------------------------------
    # Verify sources
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("VERIFYING SOURCE FILES")
    print("=" * 75)

    for location, paths in LOCATIONS.items():

        print()
        print(location.upper())

        for source_type, path in paths.items():

            print(
                f"  {source_type.upper()}:"
            )
            print(
                f"    {path}"
            )

            if not path.exists():

                raise FileNotFoundError(
                    f"Missing source file: {path}"
                )

    # --------------------------------------------------------
    # Open optical sources
    # --------------------------------------------------------

    datasets = {}

    print()
    print("=" * 75)
    print("OPENING ALIGNED OPTICAL SOURCES")
    print("=" * 75)

    for location, paths in LOCATIONS.items():

        optical_path = paths[
            "optical"
        ]

        src = rasterio.open(
            optical_path
        )

        datasets[location] = src

        print()
        print(
            location.upper()
        )

        print(
            "  Size:",
            src.width,
            "x",
            src.height,
        )

        print(
            "  CRS:",
            src.crs,
        )

        print(
            "  Resolution:",
            src.res,
        )

        print(
            "  Bands:",
            src.count,
        )

        if src.count < 3:

            raise ValueError(
                f"{location} has fewer than "
                f"3 RGB bands."
            )

    # --------------------------------------------------------
    # Create balanced sample plan
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("CREATING CONTROLLED SAMPLE PLAN")
    print("=" * 75)

    sample_plan = create_sample_plan()

    print()
    print(
        "Total planned samples:",
        len(sample_plan),
    )

    # --------------------------------------------------------
    # Metadata CSV
    # --------------------------------------------------------

    metadata_fields = [
        "sample_id",
        "split",
        "location",
        "x",
        "y",
        "patch_size",
        "scale_factor",
        "blur_type",
        "jpeg_quality",
        "contrast_factor",
        "source_file",
        "seed",
    ]

    metadata_file = open(
        METADATA_PATH,
        "w",
        newline="",
        encoding="utf-8",
    )

    writer = csv.DictWriter(
        metadata_file,
        fieldnames=metadata_fields,
    )

    writer.writeheader()

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("GENERATING DATASET")
    print("=" * 75)

    counters = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    for index, specification in enumerate(
        sample_plan,
        start=1,
    ):

        split = specification[
            "split"
        ]

        location = specification[
            "location"
        ]

        scale = specification[
            "scale"
        ]

        dataset = datasets[
            location
        ]

        # ----------------------------------------------------
        # Patch
        # ----------------------------------------------------

        x, y = choose_patch(
            dataset,
            split,
        )

        hr_array = read_rgb_patch(
            dataset,
            x,
            y,
            HR_PATCH_SIZE,
        )

        hr_image = Image.fromarray(
            hr_array,
            mode="RGB",
        )

        # ----------------------------------------------------
        # Degradation
        # ----------------------------------------------------

        (
            lr_image,
            blur_type,
            jpeg_quality,
            contrast_factor,
        ) = degrade_image(
            hr_image,
            scale,
        )

        # ----------------------------------------------------
        # IDs
        # ----------------------------------------------------

        counters[split] += 1

        sample_id = (
            f"{split}_{counters[split]:05d}"
        )

        filename = (
            f"{sample_id}_"
            f"{location}_"
            f"x{x}_y{y}_"
            f"x{scale}.png"
        )

        hr_path = (
            DATASET_ROOT
            / split
            / "HR"
            / filename
        )

        lr_path = (
            DATASET_ROOT
            / split
            / "LR"
            / filename
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_png(
            hr_image,
            hr_path,
        )

        save_png(
            lr_image,
            lr_path,
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        writer.writerow(
            {
                "sample_id": sample_id,
                "split": split,
                "location": location,
                "x": x,
                "y": y,
                "patch_size": HR_PATCH_SIZE,
                "scale_factor": scale,
                "blur_type": blur_type,
                "jpeg_quality": jpeg_quality,
                "contrast_factor": (
                    f"{contrast_factor:.6f}"
                ),
                "source_file": str(
                    LOCATIONS[
                        location
                    ]["optical"]
                ),
                "seed": RANDOM_SEED,
            }
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            index <= 20
            or index % 100 == 0
            or index == len(sample_plan)
        ):

            print(
                f"[{index}/{len(sample_plan)}]"
                f" {sample_id}"
                f" | {location}"
                f" | {split}"
                f" | x{scale}"
                f" | {blur_type}"
                f" | JPEG {jpeg_quality}"
            )

    # --------------------------------------------------------
    # Close metadata
    # --------------------------------------------------------

    metadata_file.close()

    # --------------------------------------------------------
    # Close rasters
    # --------------------------------------------------------

    for dataset in datasets.values():
        dataset.close()

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print(
        "FINAL DATASET GENERATION COMPLETE"
    )
    print("=" * 75)

    print()
    print(
        "TRAIN:",
        counters["train"],
    )

    print(
        "VAL:",
        counters["val"],
    )

    print(
        "TEST:",
        counters["test"],
    )

    print(
        "TOTAL:",
        sum(counters.values()),
    )

    print()
    print(
        "Metadata:"
    )
    print(
        METADATA_PATH
    )

    print()
    print(
        "Dataset root:"
    )
    print(
        DATASET_ROOT
    )

    print()
    print("=" * 75)


if __name__ == "__main__":
    main()