from types import SimpleNamespace

import pytest

from src.data.landsat_validation import validate_landsat_raster
from src.data.sensors.landsat import LANDSAT_BANDS


@pytest.mark.parametrize("descriptions", [
    tuple(LANDSAT_BANDS),
    ("Coastal", "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "ST_B10"),
])
def test_native_order_is_accepted(descriptions):
    validate_landsat_raster(SimpleNamespace(count=8, descriptions=descriptions, name="native.tif"))


@pytest.mark.parametrize("descriptions", [
    ("Coastal", "Red", "Green", "Blue", "NIR", "SWIR1", "SWIR2", "ST_B10"),
    tuple(LANDSAT_BANDS[:-1]),
    (None,) * 8,
])
def test_wrong_or_unknown_order_is_rejected(descriptions):
    src = SimpleNamespace(count=len(descriptions), descriptions=descriptions, name="wrong.tif")
    with pytest.raises(ValueError, match="Unexpected Landsat band order"):
        validate_landsat_raster(src)
