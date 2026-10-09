
import numpy as np
import pytest

from src.inference import OverlapAccumulator


def test_overlap_averaging():
    accumulator = OverlapAccumulator(
        height=2,
        width=3,
    )

    accumulator.add(
        np.ones((2, 2), dtype=np.float32),
        row_off=0,
        col_off=0,
    )

    accumulator.add(
        np.zeros((2, 2), dtype=np.float32),
        row_off=0,
        col_off=1,
    )

    result = accumulator.finalize()

    expected = np.array(
        [
            [1.0, 0.5, 0.0],
            [1.0, 0.5, 0.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        result, expected
    )


def test_invalid_pixels_become_nan():
    accumulator = OverlapAccumulator(2, 2)

    accumulator.add(
        np.ones((2, 2)),
        row_off=0,
        col_off=0,
    )

    valid = np.array(
        [
            [True, False],
            [True, True],
        ]
    )

    result = accumulator.finalize(valid)

    assert np.isnan(result[0, 1])
    assert np.isfinite(result[valid]).all()


def test_uncovered_valid_pixels_raise():
    accumulator = OverlapAccumulator(4, 4)

    accumulator.add(
        np.ones((2, 2)),
        row_off=0,
        col_off=0,
    )

    with pytest.raises(ValueError):
        accumulator.finalize()
