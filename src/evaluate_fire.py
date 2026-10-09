
import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.fire_dataset import FireSegmentationDataset
from src.data.file_utils import resolve_unique_file
from src.data.raster import rasters_are_aligned
from src.data.visualization import (
    landsat_dn_rgb, landsat_rgb_indices, rgb_display_label, rgb_display_settings,
)
from src.inference import OverlapAccumulator
from src.metrics import metrics_from_counts
from src.models.unet import build_model


CONFIG_PATH = "configs/fire_segmentation.yaml"
PROB_NODATA = -9999.0
MASK_NODATA = 255


def find_checkpoint(checkpoint_dir):
    paths = list(
        Path(checkpoint_dir).glob("fire_*/best.pt")
    )

    if not paths:
        raise FileNotFoundError(
            "No best.pt checkpoint found. "
            "Complete Step 8 training first."
        )

    return max(paths, key=lambda p: p.stat().st_mtime)


def load_best_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model = build_model(
        checkpoint["config"]["model"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, checkpoint


def validate_checkpoint_sources(checkpoint, data_config):
    """Reject predictions from a checkpoint trained on different exports."""
    sources = [
        str(resolve_unique_file(
            data_config["root"], data_config["image_pattern"], region
        ))
        for region in data_config["train_regions"]
    ]
    checkpoint_sources = checkpoint.get("training_stats", {}).get("source_files")
    if checkpoint_sources is None or sorted(checkpoint_sources) != sorted(sources):
        raise ValueError(
            "Checkpoint was trained on different or unidentified Landsat exports. "
            "Train a new checkpoint for the current exports."
        )


def write_raster(path, array, profile, nodata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    output_profile = profile.copy()
    output_profile.update(
        driver="GTiff",
        count=1,
        dtype=str(array.dtype),
        nodata=nodata,
        compress="deflate",
    )

    with rasterio.open(
        path, "w", **output_profile
    ) as dst:
        dst.write(array, 1)


def make_rgb(raw_rgb, valid_mask, display_settings=None):
    """Render DN bands already selected in metadata R/G/B order."""
    return landsat_dn_rgb(raw_rgb, valid_mask, **(display_settings or {}))


def save_comparison_figure(
    path,
    rgb,
    target,
    probabilities,
    prediction,
    valid,
    region,
    metrics,
    rgb_settings=None,
):
    """
    Save a six-panel comparison figure.
    """

    fig, axes = plt.subplots(
        2, 3, figsize=(18, 11)
    )

    ax = axes.ravel()

    ax[0].imshow(rgb, interpolation="nearest")
    ax[0].set_title(f"Landsat RGB composite\n{rgb_display_label(rgb_settings)}")

    target_display = np.where(
        valid, target, np.nan
    )

    ax[1].imshow(
        target_display,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    ax[1].set_title("NBAC ground truth")

    im = ax[2].imshow(
        probabilities,
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    ax[2].set_title("Burned-area probability")
    fig.colorbar(
        im, ax=ax[2], fraction=0.046, pad=0.04
    )

    prediction_display = np.where(
        valid, prediction, np.nan
    )

    ax[3].imshow(
        prediction_display,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    ax[3].set_title("Binary prediction")

    # Error codes:
    # 0 = TN, 1 = TP, 2 = FP, 3 = FN, 4 = invalid
    errors = np.full(
        target.shape, 4, dtype=np.uint8
    )

    errors[valid] = 0
    errors[valid & (target == 1) & (prediction == 1)] = 1
    errors[valid & (target == 0) & (prediction == 1)] = 2
    errors[valid & (target == 1) & (prediction == 0)] = 3

    colors = [
        "#cbd5e1",
        "#16a34a",
        "#dc2626",
        "#f59e0b",
        "#111827",
    ]

    ax[4].imshow(
        errors,
        cmap=ListedColormap(colors),
        vmin=0,
        vmax=4,
        interpolation="nearest",
    )

    ax[4].set_title("Prediction outcomes")

    labels = [
        "True negative",
        "True positive",
        "False positive",
        "False negative",
        "Invalid",
    ]

    handles = [
        Patch(facecolor=color, label=label)
        for color, label in zip(colors, labels)
    ]

    ax[4].legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
        fontsize=8,
    )

    # Probability distributions by reference class.
    bins = np.linspace(0, 1, 41)

    for label, name, color in [
        (0, "Unburned", "#2563eb"),
        (1, "Burned", "#dc2626"),
    ]:
        values = probabilities[
            valid & (target == label)
        ]

        if values.size:
            ax[5].hist(
                values,
                bins=bins,
                density=True,
                alpha=0.55,
                color=color,
                label=name,
            )

    ax[5].set_title("Probability by reference class")
    ax[5].set_xlabel("Predicted burn probability")
    ax[5].set_ylabel("Density")
    ax[5].legend()

    for a in ax[:5]:
        a.axis("off")

    fig.suptitle(
        f"Region {region} — "
        f"IoU={metrics['iou']:.3f}, "
        f"F1={metrics['f1']:.3f}, "
        f"Precision={metrics['precision']:.3f}, "
        f"Recall={metrics['recall']:.3f}",
        fontsize=15,
    )

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def evaluate_region(
    region,
    accumulator,
    data_config,
    output_dir,
    threshold,
    inference_seconds,
    num_patches,
    checkpoint_path,
    rgb_settings=None,
):
    image_path = resolve_unique_file(
        data_config["root"],
        data_config["image_pattern"],
        region,
    )

    target_path = resolve_unique_file(
        data_config["root"],
        data_config["target_pattern"],
        region,
    )

    if not rasters_are_aligned(
        image_path, target_path
    ):
        raise ValueError(
            f"Unaligned rasters for region {region}"
        )

    with rasterio.open(image_path) as src:
        optical_dn = src.read(
            indexes=[1, 2, 3, 4, 5, 6, 7]
        )

        rgb_indices = landsat_rgb_indices(
            src.descriptions, data_config["bands"]
        )
        raw_rgb = src.read(indexes=[i + 1 for i in rgb_indices])
        profile = src.profile.copy()

    with rasterio.open(target_path) as src:
        target = src.read(1)

    valid = np.all(
        optical_dn != 0, axis=0
    )

    if not np.isin(target, [0, 1]).all():
        raise ValueError(
            f"Unexpected NBAC labels in region {region}"
        )

    probabilities = accumulator.finalize(valid)

    prediction = (
        probabilities >= threshold
    ).astype(np.uint8)

    # Each valid geographic pixel is counted once.
    tp = int(np.sum(
        valid & (prediction == 1) & (target == 1)
    ))
    fp = int(np.sum(
        valid & (prediction == 1) & (target == 0)
    ))
    fn = int(np.sum(
        valid & (prediction == 0) & (target == 1)
    ))
    tn = int(np.sum(
        valid & (prediction == 0) & (target == 0)
    ))

    counts = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }

    metrics = metrics_from_counts(counts)

    valid_pixels = int(valid.sum())

    result = {
        "region": int(region),
        "checkpoint": str(checkpoint_path),
        "threshold": float(threshold),
        "patches": int(num_patches),
        "valid_pixels": valid_pixels,
        "total_pixels": int(valid.size),
        "coverage_fraction": float(valid.mean()),
        "inference_seconds": float(inference_seconds),
        **counts,
        **metrics,
    }

    region_dir = output_dir / f"region{region}"
    region_dir.mkdir(
        parents=True, exist_ok=True
    )

    probability_raster = np.where(
        valid,
        probabilities,
        PROB_NODATA,
    ).astype(np.float32)

    prediction_raster = np.where(
        valid,
        prediction,
        MASK_NODATA,
    ).astype(np.uint8)

    write_raster(
        region_dir / f"region{region}_probability.tif",
        probability_raster,
        profile,
        PROB_NODATA,
    )

    write_raster(
        region_dir / f"region{region}_prediction.tif",
        prediction_raster,
        profile,
        MASK_NODATA,
    )

    rgb = make_rgb(raw_rgb, valid, rgb_settings)

    save_comparison_figure(
        region_dir / f"region{region}_comparison.png",
        rgb,
        target,
        probabilities,
        prediction,
        valid,
        region,
        metrics,
        rgb_settings=rgb_settings,
    )

    return result


def redraw_evaluation_figures(config, checkpoint_path):
    """Redraw figures from saved predictions without model inference."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_checkpoint_sources(checkpoint, config["data"])
    output_dir = (
        Path(config["output"]["output_dir"])
        / "evaluation" / checkpoint_path.parent.name
    )
    with open(output_dir / "metrics.json") as f:
        results = json.load(f)
    data_config = config["data"]
    rgb_settings = rgb_display_settings(config)
    for region in data_config["test_regions"]:
        result = next(row for row in results if row["region"] == region)
        image_path = resolve_unique_file(
            data_config["root"], data_config["image_pattern"], region
        )
        target_path = resolve_unique_file(
            data_config["root"], data_config["target_pattern"], region
        )
        region_dir = output_dir / f"region{region}"
        probability_path = region_dir / f"region{region}_probability.tif"
        prediction_path = region_dir / f"region{region}_prediction.tif"
        for path in (target_path, probability_path, prediction_path):
            if not rasters_are_aligned(image_path, path):
                raise ValueError(f"Unaligned evaluation raster: {path}")
        with rasterio.open(image_path) as src:
            indices = landsat_rgb_indices(src.descriptions, data_config["bands"])
            raw_rgb = src.read(indexes=[i + 1 for i in indices])
            valid = np.all(src.read(indexes=list(range(1, 8))) != 0, axis=0)
        with rasterio.open(target_path) as src:
            target = src.read(1)
        with rasterio.open(probability_path) as src:
            probabilities = src.read(1, masked=True).filled(np.nan)
        with rasterio.open(prediction_path) as src:
            prediction = src.read(1, masked=True)
            if np.any(valid & np.ma.getmaskarray(prediction)):
                raise ValueError("Missing predictions for valid optical pixels")
            prediction = prediction.filled(0)
        if not np.isfinite(probabilities[valid]).all():
            raise ValueError("Missing probabilities for valid optical pixels")
        figure_path = region_dir / f"region{region}_comparison.png"
        save_comparison_figure(
            figure_path, make_rgb(raw_rgb, valid, rgb_settings), target,
            probabilities, prediction, valid, region, result,
            rgb_settings=rgb_settings,
        )
        print(f"Redrew figure: {figure_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=CONFIG_PATH,
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
    )
    parser.add_argument(
        "--plots-only", action="store_true",
        help="Redraw figures using existing evaluation rasters and metrics",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    if args.checkpoint is None:
        checkpoint_path = find_checkpoint(
            config["output"]["checkpoint_dir"]
        )
    else:
        checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    if args.plots_only:
        redraw_evaluation_figures(config, checkpoint_path)
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable. Start Docker with --gpus all."
        )
    device = torch.device("cuda")

    print("=" * 75)
    print("FIRE SEGMENTATION — HELD-OUT EVALUATION")
    print("=" * 75)

    print("Checkpoint:", checkpoint_path)
    print("GPU:", torch.cuda.get_device_name(0))

    model, checkpoint = load_best_model(
        checkpoint_path, device
    )

    checkpoint_stats = checkpoint.get(
        "training_stats", {}
    )

    dataset = FireSegmentationDataset(
        config_path=args.config,
        split="test",
    )

    # Ensure the checkpoint and Dataset use
    # the same training normalization statistics.
    for key in ["mean", "std"]:
        if key not in checkpoint_stats:
            raise ValueError(
                f"Checkpoint missing training statistic: {key}"
            )

        if not np.allclose(
            checkpoint_stats[key],
            dataset.stats[key],
            rtol=0,
            atol=1e-6,
        ):
            raise ValueError(
                "Checkpoint and Dataset normalization differ. "
                "Use the original training configuration."
            )

    validate_checkpoint_sources(checkpoint, config["data"])

    if (
        checkpoint["config"]["data"]["train_regions"]
        != config["data"]["train_regions"]
    ):
        raise ValueError("Training-region mismatch")

    if (
        checkpoint["config"]["data"]["test_regions"]
        != config["data"]["test_regions"]
    ):
        raise ValueError("Test-region mismatch")

    train_cfg = checkpoint["config"]["training"]

    threshold = float(
        train_cfg.get("threshold", 0.5)
    )

    use_amp = bool(
        train_cfg.get("amp", True)
    )

    batch_size = int(
        config["training"]["batch_size"]
    )

    workers = int(
        config["training"].get("num_workers", 2)
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )

    # Patch lookup from the Step 5 index.
    patch_lookup = {
        row.patch_id: (
            int(row.region),
            int(row.row_off),
            int(row.col_off),
        )
        for row in dataset.records.itertuples(index=False)
    }

    if len(patch_lookup) != len(dataset):
        raise ValueError("Duplicate patch IDs found")

    accumulators = {}

    data_config = config["data"]

    test_regions = data_config["test_regions"]

    for region in test_regions:
        image_path = resolve_unique_file(
            data_config["root"],
            data_config["image_pattern"],
            region,
        )

        with rasterio.open(image_path) as src:
            accumulators[region] = OverlapAccumulator(
                height=src.height,
                width=src.width,
            )

    patch_counts = {
        region: 0 for region in test_regions
    }

    # -------------------------------------------------
    # Batched GPU inference
    # -------------------------------------------------

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(
                device, non_blocking=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)

            probabilities = torch.sigmoid(
                logits.float()
            ).cpu().numpy()[:, 0]

            for patch_id, probability in zip(
                batch["patch_id"],
                probabilities,
            ):
                region, row_off, col_off = (
                    patch_lookup[patch_id]
                )

                accumulators[region].add(
                    probability,
                    row_off,
                    col_off,
                )

                patch_counts[region] += 1

    torch.cuda.synchronize()
    inference_seconds = (
        time.perf_counter() - start_time
    )

    # -------------------------------------------------
    # Full-resolution evaluation
    # -------------------------------------------------

    output_dir = (
        Path(config["output"]["output_dir"])
        / "evaluation"
        / checkpoint_path.parent.name
    )

    output_dir.mkdir(
        parents=True, exist_ok=True
    )

    results = []

    total_patches = sum(
        patch_counts.values()
    )

    for region in test_regions:
        region_time = (
            inference_seconds
            * patch_counts[region]
            / total_patches
        )

        result = evaluate_region(
            region=region,
            accumulator=accumulators[region],
            data_config=data_config,
            output_dir=output_dir,
            threshold=threshold,
            inference_seconds=region_time,
            num_patches=patch_counts[region],
            checkpoint_path=checkpoint_path,
            rgb_settings=rgb_display_settings(config),
        )

        results.append(result)

        print(
            f"\nRegion {region}: "
            f"IoU={result['iou']:.4f}, "
            f"F1={result['f1']:.4f}, "
            f"Precision={result['precision']:.4f}, "
            f"Recall={result['recall']:.4f}"
        )

    # Aggregate confusion counts across test regions.
    if len(results) > 1:
        counts = {
            key: sum(row[key] for row in results)
            for key in ["tp", "fp", "fn", "tn"]
        }

        aggregate_metrics = metrics_from_counts(counts)

        results.append({
            "region": "ALL_TEST",
            "checkpoint": str(checkpoint_path),
            "threshold": threshold,
            "patches": total_patches,
            "valid_pixels": sum(
                r["valid_pixels"] for r in results
            ),
            "total_pixels": sum(
                r["total_pixels"] for r in results
            ),
            "coverage_fraction": (
                sum(r["valid_pixels"] for r in results)
                / sum(r["total_pixels"] for r in results)
            ),
            "inference_seconds": inference_seconds,
            **counts,
            **aggregate_metrics,
        })

    csv_path = output_dir / "metrics.csv"
    json_path = output_dir / "metrics.json"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 75)
    print("EVALUATION COMPLETED")
    print("=" * 75)
    print(f"Total inference time: {inference_seconds:.2f}s")
    print(f"Processed patches: {total_patches}")
    print(f"Results: {output_dir}")
    print(f"Metrics: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
