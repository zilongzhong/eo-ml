import pytest

from src.data.patches import (
    window_starts,
    iter_patch_windows,
)


def test_exact_patch_size():
    assert window_starts(
        128, 128, 64
    ) == [0]


def test_patch_coverage():
    starts = window_starts(
        300, 128, 64
    )

    assert starts == [
        0, 64, 128, 172
    ]


def test_invalid_patch_size():
    with pytest.raises(ValueError):
        window_starts(
            100, 128, 64
        )


def test_patch_window_shape():

    windows = list(
        iter_patch_windows(
            height=256,
            width=256,
            patch_size=128,
            stride=128,
        )
    )

    assert len(windows) == 4

    for row, col, height, width in windows:

        assert height == 128
        assert width == 128

        assert 0 <= row <= 128
        assert 0 <= col <= 128
