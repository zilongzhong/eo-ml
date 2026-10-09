"""Regression checks for exported band order and diagnostic displays."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import rasterio

from src.data.check_fire_dataloader import make_rgb
from src.data.fire_dataset import FireSegmentationDataset
from src.data.sensors.landsat import LANDSAT_BANDS, scale_landsat_c2_l2
from src.data.visualization import (
    landsat_dn_rgb, landsat_rgb_indices, plot_validity_mask, reflectance_rgb,
    rgb_display_settings,
)


@pytest.mark.parametrize("descriptions,expected", [
    (("Coastal", "Red", "Green", "Blue", "NIR", "SWIR1", "SWIR2", "ST_B10"), [1, 2, 3]),
    (("SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10"), [3, 2, 1]),
    (("Coastal", "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "ST_B10"), [3, 2, 1]),
    ((None,) * 8, [3, 2, 1]),
])
def test_rgb_follows_band_metadata(descriptions, expected):
    assert landsat_rgb_indices(descriptions) == expected


@pytest.mark.parametrize("descriptions", [
    ("Red", "Green", "NIR"),
    ("Red", "Green", "Blue", "Red"),
    ("band1", "band2", "band3"),
])
def test_ambiguous_metadata_is_not_silently_guessed(descriptions):
    with pytest.raises(ValueError):
        landsat_rgb_indices(descriptions)


def test_uniform_color_uses_fixed_reflectance_limits():
    reflectance = np.broadcast_to(
        np.array([0.3, 0.1, 0.0])[:, None, None], (3, 4, 4)
    )
    rgb = reflectance_rgb(
        reflectance, np.ones((4, 4), dtype=bool),
        vmin=0, vmax=0.3, gamma=1, shadow_floor=0,
    )
    assert np.allclose(rgb[0, 0], [1, 1 / 3, 0])
    assert rgb.dtype == np.float32


def test_missing_pixels_do_not_produce_nan():
    image = np.array([
        [[0.02, 0.04], [0.06, 999]],
        [[0.03, 0.05], [0.07, np.nan]],
        [[0.01, 0.03], [0.05, -999]],
    ])
    valid = np.array([[True, True], [True, False]])
    rgb = reflectance_rgb(image, valid)
    image[:, 1, 1] = -10000
    assert np.array_equal(rgb, reflectance_rgb(image, valid))
    assert np.all(rgb[1, 1] == 0)
    assert np.isfinite(rgb).all()
    assert np.all((rgb >= 0) & (rgb <= 1))


def test_all_missing_and_constant_gray_have_defined_displays():
    image = np.full((3, 2, 2), 0.025, dtype=np.float32)
    assert np.all(reflectance_rgb(image, np.zeros((2, 2), dtype=bool)) == 0)
    assert np.allclose(
        reflectance_rgb(image, np.ones((2, 2), dtype=bool), gamma=2, shadow_floor=0), 0.5
    )


def test_gamma_brightens_positive_reflectance_without_lifting_negative_values():
    image = np.broadcast_to(
        np.array([-0.1, 0, 0.025, 0.1, 0.2])[None, None, :], (3, 1, 5)
    )
    valid = np.ones((1, 5), dtype=bool)
    rgb = reflectance_rgb(image, valid, vmin=0, vmax=0.1, gamma=2, shadow_floor=0)
    assert np.allclose(rgb[0, :, 0], [0, 0, 0.5, 1, 1])
    linear = reflectance_rgb(image, valid, vmin=0, vmax=0.1, gamma=1, shadow_floor=0)
    assert rgb[0, 2, 0] > linear[0, 2, 0]


def test_same_reflectance_is_stable_across_scene_content():
    image = np.full((3, 2, 2), 0.025)
    valid = np.ones((2, 2), dtype=bool)
    original = reflectance_rgb(image, valid)
    image[:, 1, 1] = 500
    image[:, 1, 0] = -500
    changed = reflectance_rgb(image, valid)
    assert np.array_equal(original[0], changed[0])


def test_dn_uses_official_scale_offset_exactly_once():
    raw = np.full((3, 2, 2), 10000, dtype=np.uint16)
    rgb = landsat_dn_rgb(
        raw, np.ones((2, 2), dtype=bool), vmin=0, vmax=0.3, gamma=2, shadow_floor=0,
    )
    # USGS: DN 10000 -> 0.075 reflectance -> sqrt(0.075 / 0.3) = 0.5.
    assert np.allclose(rgb, 0.5)


@pytest.mark.parametrize("settings", [
    {"vmin": 0.1, "vmax": 0.1}, {"vmin": 0.2, "vmax": 0.1},
    {"gamma": 0}, {"gamma": -1}, {"vmax": np.nan}, {"gamma": np.inf},
    {"shadow_floor": -0.01}, {"shadow_floor": 1}, {"shadow_floor": np.nan},
])
def test_invalid_reflectance_display_settings_raise(settings):
    with pytest.raises(ValueError):
        reflectance_rgb(np.ones((3, 2, 2)), np.ones((2, 2), dtype=bool), **settings)


def test_shadow_lift_preserves_valid_dark_pixels_and_keeps_missing_pixels_black():
    image = np.broadcast_to(
        np.array([-0.05, 0, 0.01, 0.1, -0.1])[None, None, :], (3, 1, 5)
    ).copy()
    original = image.copy()
    valid = np.array([[True, True, True, True, False]])
    rgb = reflectance_rgb(image, valid, gamma=2, shadow_floor=0.08)
    assert np.allclose(rgb[0, :2], 0.08)
    assert np.all(rgb[0, 2] > 0.08)
    assert np.allclose(rgb[0, 3], 1)
    assert np.all(rgb[0, 4] == 0)
    assert np.array_equal(image, original)


def test_all_valid_mask_is_visible_and_labelled():
    fig, ax = plt.subplots()
    try:
        plot_validity_mask(ax, np.ones((4, 4), dtype=bool), "Optical valid")
        artist = ax.images[0]
        assert artist.cmap(artist.norm(1)) != (1, 1, 1, 1)
        assert "100.00% valid" in ax.get_title()
        assert [text.get_text() for text in ax.get_legend().get_texts()] == [
            "Missing (0)", "Valid (1)",
        ]
    finally:
        plt.close(fig)


@pytest.fixture(params=["sensor_names", "color_names", "unnamed"])
def native_rgb_sample(tmp_path, request):
    """A native-order TIFF with distinguishable blue/green/red values."""
    from types import SimpleNamespace
    import pandas as pd
    import torch

    raw = np.full((8, 4, 4), 10000, dtype=np.uint16)
    pattern = np.arange(16, dtype=np.uint16).reshape(4, 4)
    raw[1] = 8000 + pattern * 20   # SR_B2: blue
    raw[2] = 9000 + pattern * 20   # SR_B3: green
    raw[3] = 10000 + pattern * 20  # SR_B4: red
    raw[7] = 45000
    path = tmp_path / "native_landsat.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=4, width=4, count=8,
        dtype="uint16", crs="EPSG:32610",
        transform=rasterio.transform.from_origin(1000, 2000, 30, 30),
    ) as dst:
        dst.write(raw)
        if request.param == "sensor_names":
            dst.descriptions = tuple(LANDSAT_BANDS)
        elif request.param == "color_names":
            dst.descriptions = (
                "Coastal", "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "ST_B10",
            )
    mean = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 300], dtype=np.float32)
    std = np.array([0.03] * 7 + [6], dtype=np.float32)
    dataset = SimpleNamespace(
        config={"data": {"bands": LANDSAT_BANDS}},
        mean=mean, std=std,
        records=pd.DataFrame([{"patch_id": "native_patch", "image_path": str(path)}]),
    )
    sample = {
        "patch_id": "native_patch", "region": 2,
        "image": torch.from_numpy(
            (scale_landsat_c2_l2(raw) - mean[:, None, None]) / std[:, None, None]
        ),
        "mask": torch.zeros((1, 4, 4), dtype=torch.float32),
        "valid_mask": torch.ones((1, 4, 4), dtype=torch.bool),
    }
    return dataset, sample, raw


def test_native_tiff_rgb_matches_dataset_and_evaluation(native_rgb_sample):
    from src.evaluate_fire import make_rgb as evaluation_rgb

    dataset, sample, raw = native_rgb_sample
    with rasterio.open(dataset.records.iloc[0].image_path) as src:
        indices = landsat_rgb_indices(src.descriptions, dataset.config["data"]["bands"])
        assert indices == [3, 2, 1]
        raw_rgb = src.read(indexes=[i + 1 for i in indices])
    valid = sample["valid_mask"][0].numpy()
    expected = landsat_dn_rgb(raw[[3, 2, 1]], valid)
    assert np.allclose(make_rgb(sample, dataset), expected, atol=1e-6)
    assert np.allclose(evaluation_rgb(raw_rgb, valid), expected, atol=1e-6)
    # Make the regression sensitive to accidentally swapping red and blue.
    assert expected[0, 0, 0] > expected[0, 0, 2]


def test_tensorboard_rgb_matches_dataset_rendering(native_rgb_sample):
    from types import SimpleNamespace
    from src.train_fire import log_validation_images

    dataset, sample, _ = native_rgb_sample
    captured = {}
    writer = SimpleNamespace(
        add_images=lambda tag, image, epoch: captured.update({tag: image})
    )
    example = {
        "images": sample["image"].unsqueeze(0),
        "targets": sample["mask"].unsqueeze(0),
        "valid": sample["valid_mask"].unsqueeze(0),
        "probabilities": sample["mask"].unsqueeze(0),
        "regions": [2],
    }
    log_validation_images(
        writer, example, dataset.mean, dataset.std,
        epoch=1, threshold=0.5, rgb_indices_by_region={2: [3, 2, 1]},
        rgb_settings=rgb_display_settings(dataset.config),
    )
    rgb = captured["val/RGB"][0].permute(1, 2, 0).numpy()
    assert np.allclose(rgb, make_rgb(sample, dataset), atol=1e-6)


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_dataset_rgb_matches_original_dn_without_double_scaling(split):
    dataset = FireSegmentationDataset(
        config_path=os.environ.get("EO_FIRE_CONFIG", "configs/fire_segmentation.yaml"),
        split=split,
    )
    index = int(np.argmin(np.abs(dataset.records["burned_fraction"].to_numpy() - 0.5)))
    sample = dataset[index]
    record = dataset.records.iloc[index]
    window = rasterio.windows.Window(
        int(record.col_off), int(record.row_off), int(record.width), int(record.height)
    )
    with rasterio.open(record.image_path) as src:
        # Updated exports follow native SR_B1 ... SR_B7, ST_B10 order.
        assert landsat_rgb_indices(src.descriptions, dataset.config["data"]["bands"]) == [3, 2, 1]
        raw_rgb = src.read(indexes=[4, 3, 2], window=window)
    expected = landsat_dn_rgb(
        raw_rgb, sample["valid_mask"][0].numpy(), **rgb_display_settings(dataset.config)
    )
    assert np.allclose(make_rgb(sample, dataset), expected, atol=1e-6)
