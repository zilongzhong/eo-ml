import numpy as np
import pytest

from src.data.sensors.landsat import (
    scale_landsat_c2_l2,
)


def test_landsat_output_shape():

    x = np.ones(
        (8, 32, 32),
        dtype=np.uint16,
    )

    y = scale_landsat_c2_l2(x)

    assert y.shape == x.shape
    assert y.dtype == np.float32


def test_landsat_rejects_wrong_channels():

    x = np.ones(
        (7, 32, 32),
        dtype=np.uint16,
    )

    with pytest.raises(ValueError):
        scale_landsat_c2_l2(x)


def test_landsat_scaling_known_value():

    x = np.full(
        (8, 1, 1),
        10000,
        dtype=np.uint16,
    )

    y = scale_landsat_c2_l2(x)

    expected_sr = (
        10000 * 0.0000275
        - 0.2
    )

    expected_st = (
        10000 * 0.00341802
        + 149.0
    )

    assert np.isclose(
        y[0, 0, 0],
        expected_sr,
    )

    assert np.isclose(
        y[7, 0, 0],
        expected_st,
    )
