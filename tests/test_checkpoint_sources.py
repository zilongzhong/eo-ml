import pytest

from src.evaluate_fire import validate_checkpoint_sources


@pytest.fixture
def current_export(tmp_path):
    path = tmp_path / "Landsat89_2ndMIN_SR_MAX_ST_DN_2023_region1.tif"
    path.touch()
    config = {
        "root": str(tmp_path),
        "image_pattern": "Landsat89_2ndMIN_SR_MAX_ST_DN_2023_region{region}.tif",
        "train_regions": [1],
    }
    return path, config


def test_checkpoint_accepts_current_export(current_export):
    path, config = current_export
    validate_checkpoint_sources({"training_stats": {"source_files": [str(path)]}}, config)


def test_checkpoint_rejects_previous_min_export(current_export):
    path, config = current_export
    old = path.with_name("Landsat89_MIN_SR_MAX_ST_DN_2023_region1.tif")
    with pytest.raises(ValueError, match="different or unidentified Landsat exports"):
        validate_checkpoint_sources({"training_stats": {"source_files": [str(old)]}}, config)


def test_checkpoint_rejects_missing_source_metadata(current_export):
    _, config = current_export
    with pytest.raises(ValueError, match="different or unidentified Landsat exports"):
        validate_checkpoint_sources({}, config)
