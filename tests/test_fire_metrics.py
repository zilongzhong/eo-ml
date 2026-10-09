
import torch

from src.metrics import (
    binary_segmentation_counts,
    metrics_from_counts,
)


def test_segmentation_metrics_known_values():
    # Prediction: [1, 1, 0, 0]
    # Target:     [1, 0, 1, 0]
    #
    # TP=1, FP=1, FN=1, TN=1

    logits = torch.tensor(
        [[[[5.0, 5.0, -5.0, -5.0]]]]
    )

    targets = torch.tensor(
        [[[[1.0, 0.0, 1.0, 0.0]]]]
    )

    valid = torch.ones_like(
        targets,
        dtype=torch.bool,
    )

    counts = binary_segmentation_counts(
        logits,
        targets,
        valid,
    )

    assert counts == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "tn": 1,
    }

    metrics = metrics_from_counts(counts)

    assert abs(metrics["precision"] - 0.5) < 1e-6
    assert abs(metrics["recall"] - 0.5) < 1e-6
    assert abs(metrics["f1"] - 0.5) < 1e-6
    assert abs(metrics["iou"] - 1/3) < 1e-6
    assert abs(metrics["accuracy"] - 0.5) < 1e-6


def test_invalid_pixel_excluded():
    logits = torch.tensor(
        [[[[5.0, 5.0]]]]
    )

    targets = torch.tensor(
        [[[[1.0, 0.0]]]]
    )

    valid = torch.tensor(
        [[[[True, False]]]]
    )

    counts = binary_segmentation_counts(
        logits,
        targets,
        valid,
    )

    assert counts["tp"] == 1
    assert counts["fp"] == 0
    assert counts["fn"] == 0
    assert counts["tn"] == 0

    metrics = metrics_from_counts(counts)

    assert metrics["iou"] == 1.0
    assert metrics["f1"] == 1.0
