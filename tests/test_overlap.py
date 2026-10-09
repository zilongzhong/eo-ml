
import numpy as np

from src.data.overlap import ProbabilityMosaic


def test_overlapping_probability_average():
    mosaic = ProbabilityMosaic(
        height=2,
        width=3,
    )

    valid = np.ones(
        (2, 2),
        dtype=bool,
    )

    mosaic.add(
        np.full((2, 2), 0.2),
        valid,
        row_off=0,
        col_off=0,
    )

    mosaic.add(
        np.full((2, 2), 0.8),
        valid,
        row_off=0,
        col_off=1,
    )

    probability, coverage = mosaic.finalize()

    np.testing.assert_allclose(
        probability,
        [
            [0.2, 0.5, 0.8],
            [0.2, 0.5, 0.8],
        ],
        atol=1e-6,
    )

    np.testing.assert_array_equal(
        coverage,
        [
            [1, 2, 1],
            [1, 2, 1],
        ],
    )


def test_invalid_pixels_excluded():
    mosaic = ProbabilityMosaic(
        height=2,
        width=2,
    )

    probability = np.full(
        (2, 2),
        0.8,
        dtype=np.float32,
    )

    valid = np.array([
        [False, True],
        [True, True],
    ])

    mosaic.add(
        probability,
        valid,
        row_off=0,
        col_off=0,
    )

    output, coverage = mosaic.finalize()

    assert output[0, 0] == -9999.0
    assert coverage[0, 0] == 0
    assert np.isclose(output[1, 1], 0.8)
    assert coverage[1, 1] == 1
