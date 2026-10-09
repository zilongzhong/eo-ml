
import os
import numpy as np
import pytest
import torch

from src.data.normalization import normalize_channels
from src.data.fire_dataset import FireSegmentationDataset


def test_normalization_channels():
    # Generic: also works for Sentinel-2's 12 bands.
    for channels in [8, 12]:
        image = np.ones(
            (channels, 16, 16),
            dtype=np.float32,
        )

        mean = np.zeros(channels)
        std = np.ones(channels)

        result = normalize_channels(
            image, mean, std
        )

        assert result.shape == image.shape
        assert result.dtype == np.float32
        assert np.isfinite(result).all()


def test_missing_values_replaced():
    image = np.ones(
        (8, 4, 4),
        dtype=np.float32,
    )

    image[7, 0, 0] = np.nan

    result = normalize_channels(
        image,
        mean=np.zeros(8),
        std=np.ones(8),
    )

    assert result[7, 0, 0] == 0
    assert np.isfinite(result).all()


@pytest.mark.parametrize(
    "split,expected_count",
    [
        ("train", 253),
        ("val", 276),
        ("test", 198),
    ],
)
def test_fire_dataset(split, expected_count):

    dataset = FireSegmentationDataset(
        config_path=os.environ.get("EO_FIRE_CONFIG", "configs/fire_segmentation.yaml"),
        split=split
    )

    assert len(dataset) == expected_count

    sample = dataset[0]

    assert sample["image"].shape == (
        8, 128, 128
    )

    assert sample["mask"].shape == (
        1, 128, 128
    )

    assert sample["valid_mask"].shape == (
        1, 128, 128
    )

    assert sample["image"].dtype == torch.float32
    assert sample["mask"].dtype == torch.float32

    assert sample["valid_mask"].dtype == torch.bool

    assert torch.isfinite(
        sample["image"]
    ).all()

    assert torch.all(
        (sample["mask"] == 0)
        | (sample["mask"] == 1)
    )

    # Verify the patch's valid-pixel count
    # matches the Step 5 manifest.
    expected_valid = int(
        dataset.records.iloc[0]["valid_optical_pixels"]
    )

    assert (
        sample["valid_mask"].sum().item()
        == expected_valid
    )
