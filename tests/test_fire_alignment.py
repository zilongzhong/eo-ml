import os
from pathlib import Path

import pytest
import rasterio

from src.config import load_config
from src.data.file_utils import resolve_unique_file
from src.data.raster import rasters_are_aligned
from src.data.landsat_validation import validate_landsat_raster


CONFIG = load_config(
    os.environ.get("EO_FIRE_CONFIG", "configs/fire_segmentation.yaml")
)

DATA_CONFIG = CONFIG["data"]


@pytest.mark.parametrize(
    "region",
    DATA_CONFIG["train_regions"]
    + DATA_CONFIG["val_regions"]
    + DATA_CONFIG["test_regions"],
)
def test_fire_image_mask_alignment(region):

    image_path = resolve_unique_file(
        DATA_CONFIG["root"],
        DATA_CONFIG["image_pattern"],
        region,
    )

    mask_path = resolve_unique_file(
        DATA_CONFIG["root"],
        DATA_CONFIG["target_pattern"],
        region,
    )

    assert rasters_are_aligned(
        image_path,
        mask_path,
    )
    with rasterio.open(image_path) as src:
        validate_landsat_raster(src, DATA_CONFIG["bands"])
        assert src.dtypes == ("uint16",) * 8
