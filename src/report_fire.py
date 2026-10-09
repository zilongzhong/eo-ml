
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import numpy as np
import pandas as pd
import rasterio

from src.config import load_config
from src.data.patches import iter_patch_windows
from src.data.sensors.landsat import preprocess_landsat_c2_l2
from src.metrics import metrics_from_counts


CONFIG_PATH = "configs/fire_segmentation.yaml"

ERROR_NAMES = {
    0: "True negative",
    1: "True positive",
    2: "False positive",
    3: "False negative",
}

ERROR_COLORS = [
    "#888888",
    "#2ca25f",
    "#fdae35",
    "#de2d26",
]

ERROR_CMAP = ListedColormap(ERROR_COLORS)


def read_band(path):
    with rasterio.open(path) as src:
        return src.read(1)


def check_grid(path, reference_path):
    """Verify identical raster dimensions and pixel grid."""
    with rasterio.open(path) as a, \
         rasterio.open(reference_path) as b:

        aligned = (
            a.width == b.width
            and a.height == b.height
            and a.crs == b.crs
            and np.allclose(
                tuple(a.transform),
                tuple(b.transform),
                rtol=0,
                atol=1e-10,
            )
        )

    if not aligned:
        raise ValueError(
            f"Raster grid mismatch:\n{path}\n{reference_path}"
        )


def make_rgb(image_path):
    """
    Create display-only RGB composite from Landsat bands.

    Returns float RGB [H,W,3], stretched to [0,1].
    """

    with rasterio.open(image_path) as src:
        raw = src.read()

    scaled, optical_valid, _ = (
        preprocess_landsat_c2_l2(raw)
    )

    # SR_B4, SR_B3, SR_B2
    rgb = scaled[[3, 2, 1]].transpose(1, 2, 0)

    output = np.zeros_like(rgb, dtype=np.float32)

    for channel in range(3):
        values = rgb[..., channel][optical_valid]
        values = values[np.isfinite(values)]

        if len(values) == 0:
            continue

        low, high = np.percentile(values, [2, 98])

        if high > low:
            output[..., channel] = np.clip(
                (rgb[..., channel] - low) / (high - low),
                0,
                1,
            )

    output[~optical_valid] = 0
    return output


def verify_results(data, report):
    """
    Verify Step 10 metrics against the saved error raster.

    Each covered geographic pixel is counted once.
    """

    error = data["error"]
    coverage = data["coverage"]
    prediction = data["prediction"]
    reference = data["reference"]

    covered = coverage > 0
    valid = error != 255

    if not np.array_equal(covered, valid):
        raise ValueError(
            "Coverage and error-map validity disagree"
        )

    if not np.isin(prediction[valid], [0, 1]).all():
        raise ValueError("Unexpected prediction classes")

    if not np.isin(reference[valid], [0, 1]).all():
        raise ValueError("Unexpected reference classes")

    counts = {
        "tn": int(np.count_nonzero(error == 0)),
        "tp": int(np.count_nonzero(error == 1)),
        "fp": int(np.count_nonzero(error == 2)),
        "fn": int(np.count_nonzero(error == 3)),
    }

    if counts != report["confusion_counts"]:
        raise ValueError(
            "Error-map counts do not match metrics.json"
        )

    metrics = metrics_from_counts(counts)

    for key, value in metrics.items():
        stored = report["metrics"][key]
        if not np.isclose(value, stored, atol=1e-8):
            raise ValueError(
                f"Metric mismatch for {key}: "
                f"{value} != {stored}"
            )

    return counts, metrics


def save_region_overview(data, output_path):
    rgb = data["rgb"]
    reference = data["reference"]
    prediction = data["prediction"]
    probability = data["probability"]
    coverage = data["coverage"]
    error = data["error"]

    valid = coverage > 0

    fig, axes = plt.subplots(
        2, 3, figsize=(18, 11)
    )

    panels = [
        "Landsat RGB",
        "NBAC ground truth",
        "U-Net prediction",
        "Burn probability",
        "Prediction errors",
        "Patch coverage",
    ]

    for ax, title in zip(axes.flat, panels):
        ax.set_title(title, fontsize=13)
        ax.axis("off")

    axes[0, 0].imshow(rgb)

    axes[0, 1].imshow(
        reference,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[0, 2].imshow(
        np.ma.masked_where(~valid, prediction),
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    p = axes[1, 0].imshow(
        np.ma.masked_where(~valid, probability),
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    fig.colorbar(p, ax=axes[1, 0], fraction=0.046)

    axes[1, 1].imshow(
        np.ma.masked_where(~valid, error),
        cmap=ERROR_CMAP,
        vmin=-0.5,
        vmax=3.5,
        interpolation="nearest",
    )

    handles = [
        Patch(color=color, label=ERROR_NAMES[i])
        for i, color in enumerate(ERROR_COLORS)
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=10,
    )

    c = axes[1, 2].imshow(
        coverage,
        cmap="viridis",
        vmin=0,
    )
    fig.colorbar(c, ax=axes[1, 2], fraction=0.046)

    fig.suptitle(
        "Burned-area segmentation: full-region results",
        fontsize=16,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_error_examples(
    data,
    output_path,
    patch_size=128,
    stride=64,
):
    """
    Select patches with the highest count of each
    error/true-positive category.

    These are illustrative cases, not random samples.
    """

    error = data["error"]
    height, width = error.shape

    examples = []

    for code in [1, 2, 3]:
        best_count = 0
        best_window = None

        for row, col, h, w in iter_patch_windows(
            height,
            width,
            patch_size,
            stride,
        ):
            window = error[row:row+h, col:col+w]
            count = int(np.count_nonzero(window == code))

            if count > best_count:
                best_count = count
                best_window = (row, col, h, w)

        if best_window is not None:
            examples.append(
                (code, best_window, best_count)
            )

    if not examples:
        print("No positive/error examples available.")
        return False

    fig, axes = plt.subplots(
        len(examples),
        4,
        figsize=(16, 4.2 * len(examples)),
        squeeze=False,
    )

    for row_idx, (code, window, count) in enumerate(examples):
        row, col, h, w = window
        sl = np.s_[row:row+h, col:col+w]

        rgb = data["rgb"][sl]
        reference = data["reference"][sl]
        prediction = data["prediction"][sl]
        errors = data["error"][sl]

        panels = [
            (rgb, None),
            (reference, "gray"),
            (prediction, "gray"),
            (errors, ERROR_CMAP),
        ]

        titles = [
            f"{ERROR_NAMES[code]}: {count:,} pixels",
            "NBAC",
            "Prediction",
            "Error categories",
        ]

        for j, ((image, cmap), title) in enumerate(
            zip(panels, titles)
        ):
            ax = axes[row_idx, j]

            if j == 0:
                ax.imshow(image)
            elif j in (1, 2):
                ax.imshow(image, cmap=cmap, vmin=0, vmax=1)
            else:
                ax.imshow(
                    np.ma.masked_where(image == 255, image),
                    cmap=cmap,
                    vmin=-0.5,
                    vmax=3.5,
                    interpolation="nearest",
                )

            ax.set_title(title, fontsize=11)
            ax.axis("off")

    fig.suptitle(
        "Selected true-positive and failure examples",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return True


def save_training_curves(history_path, output_path):
    if not history_path.exists():
        print(f"No training history found: {history_path}")
        return False

    df = pd.read_csv(history_path)

    required = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_iou",
        "val_f1",
        "val_precision",
        "val_recall",
    ]

    if not set(required).issubset(df.columns):
        raise ValueError("Missing training-history columns")

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5)
    )

    axes[0].plot(
        df["epoch"], df["train_loss"],
        label="Training loss",
    )
    axes[0].plot(
        df["epoch"], df["val_loss"],
        label="Validation loss",
    )
    axes[0].set_title("Training and validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Masked BCE loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    for column, label in [
        ("val_iou", "IoU"),
        ("val_f1", "F1 / Dice"),
        ("val_precision", "Precision"),
        ("val_recall", "Recall"),
    ]:
        axes[1].plot(
            df["epoch"], df[column], label=label
        )

    axes[1].set_title("Validation segmentation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return True


def save_confusion_matrix(counts, output_path):
    matrix = np.array([
        [counts["tn"], counts["fp"]],
        [counts["fn"], counts["tp"]],
    ])

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks([0, 1], ["Unburned", "Burned"])
    ax.set_yticks([0, 1], ["Unburned", "Burned"])

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference")
    ax.set_title("Full-region confusion matrix")

    for i in range(2):
        row_sum = matrix[i].sum()

        for j in range(2):
            count = matrix[i, j]
            percent = (
                100 * count / row_sum
                if row_sum > 0 else 0
            )

            ax.text(
                j, i,
                f"{count:,}\n({percent:.1f}%)",
                ha="center",
                va="center",
                color="black",
                bbox=dict(
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="none",
                ),
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_summary_csv(path, report, counts, metrics):
    row = {
        "run_name": report["run_name"],
        "split": report["split"],
        "region": report["region"],
        "total_pixels": report["total_pixels"],
        "covered_pixels": report["covered_pixels"],
        "coverage_fraction": report["coverage_fraction"],
        "evaluated_pixels": report["evaluated_pixels"],
        **counts,
        **metrics,
        "inference_seconds": report["inference_seconds"],
    }

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(row.keys())
        )
        writer.writeheader()
        writer.writerow(row)


def save_markdown_report(
    path,
    report,
    counts,
    metrics,
    has_history,
    has_examples,
):
    lines = [
        "# Wildfire Burned-Area Segmentation Report",
        "",
        f"**Experiment:** {report['run_name']}",
        f"**Region:** {report['region']}",
        f"**Split:** {report['split']}",
        "",
        "## 1. Dataset and methodology",
        "",
        "- Landsat 8/9, 8-channel input "
        "(SR_B1–SR_B7 and ST_B10)",
        "- NBAC binary burned-area reference",
        "- U-Net with ResNet34 encoder",
        "- Geographic training/validation/test separation",
        "- Training-only channel normalization",
        "- Invalid optical pixels excluded",
        "- Overlapping predictions averaged spatially",
        "",
        "## 2. Full-region results",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]

    for name in [
        "iou", "f1", "precision", "recall", "accuracy"
    ]:
        lines.append(
            f"| {name.upper()} | {metrics[name]:.4f} |"
        )

    lines += [
        f"| Coverage | {report['coverage_fraction']:.2%} |",
        f"| Evaluated pixels | {report['evaluated_pixels']:,} |",
        f"| Inference time (seconds) | "
        f"{report['inference_seconds']:.2f} |",
        "",
        "### Confusion counts",
        "",
        "| Category | Pixels |",
        "|---|---:|",
    ]

    for key in ["tp", "fp", "fn", "tn"]:
        lines.append(
            f"| {key.upper()} | {counts[key]:,} |"
        )

    lines += [
        "",
        "## 3. Full-region prediction maps",
        "",
        "![Region overview](figures/region_overview.png)",
        "",
        "## 4. Confusion matrix",
        "",
        "![Confusion matrix](figures/confusion_matrix.png)",
        "",
    ]

    if has_history:
        lines += [
            "## 5. Training curves",
            "",
            "![Training curves](figures/training_curves.png)",
            "",
        ]

    if has_examples:
        lines += [
            "## 6. Selected prediction examples",
            "",
            "![Prediction examples](figures/error_examples.png)",
            "",
            "Examples were selected by the highest count "
            "of each error/true-positive category. They are "
            "illustrative, not randomly sampled.",
            "",
        ]

    lines += [
        "## 7. Limitations",
        "",
        "- Metrics are calculated on the held-out region "
        "using each evaluated geographic pixel once.",
        "- Results depend on the quality of the NBAC reference.",
        "- Temporal min/max composites may contain bands "
        "selected from different acquisition dates.",
        "- RGB images use percentile contrast stretching "
        "for visualization only.",
        "- Generalization beyond these geographic regions "
        "has not been established.",
        "- Pixel counts alone must not be interpreted as "
        "physical burned area without area calculations.",
        "",
        "## 8. Future work",
        "",
        "- Compare against a stronger spectral baseline.",
        "- Test alternative loss functions and thresholds.",
        "- Investigate false positives and missed burn scars.",
        "- Evaluate on additional geographic regions and years.",
        "",
    ]

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--map-dir",
        required=True,
        help="Directory containing Step 10 GeoTIFF outputs",
    )
    parser.add_argument(
        "--history",
        default=None,
        help="Optional training-history CSV",
    )

    args = parser.parse_args()

    config = load_config(CONFIG_PATH)
    map_dir = Path(args.map_dir)

    metrics_path = map_dir / "metrics.json"

    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    with open(metrics_path) as f:
        report = json.load(f)

    run_name = report["run_name"]
    region = report["region"]

    output_dir = (
        Path(config["output"]["output_dir"])
        / "reports"
        / run_name
        / f"region{region}"
    )

    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"

    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    # Confirm all spatial outputs share the reference grid.
    reference_path = report["reference_path"]

    for filename in [
        "burn_probability.tif",
        "burn_prediction.tif",
        "patch_coverage.tif",
        "error_map.tif",
    ]:
        check_grid(map_dir / filename, reference_path)

    check_grid(report["image_path"], reference_path)

    data = {
        "rgb": make_rgb(report["image_path"]),
        "reference": read_band(reference_path),
        "prediction": read_band(
            map_dir / "burn_prediction.tif"
        ),
        "probability": read_band(
            map_dir / "burn_probability.tif"
        ),
        "coverage": read_band(
            map_dir / "patch_coverage.tif"
        ),
        "error": read_band(
            map_dir / "error_map.tif"
        ),
    }

    counts, metrics = verify_results(data, report)

    print("Verified metrics:", metrics)

    save_region_overview(
        data,
        figure_dir / "region_overview.png",
    )

    has_examples = save_error_examples(
        data,
        figure_dir / "error_examples.png",
        patch_size=int(config["patches"]["size"]),
        stride=int(config["patches"]["stride"]),
    )

    save_confusion_matrix(
        counts,
        figure_dir / "confusion_matrix.png",
    )

    if args.history:
        history_path = Path(args.history)
    else:
        history_path = (
            Path(config["output"]["output_dir"])
            / "tables"
            / f"{run_name}_training_history.csv"
        )

    has_history = save_training_curves(
        history_path,
        figure_dir / "training_curves.png",
    )

    save_summary_csv(
        table_dir / "performance_summary.csv",
        report,
        counts,
        metrics,
    )

    save_markdown_report(
        output_dir / "report.md",
        report,
        counts,
        metrics,
        has_history,
        has_examples,
    )

    print("\nSTEP 11 COMPLETED")
    print("Report:", output_dir / "report.md")
    print("Figures:", figure_dir)
    print(
        "Performance table:",
        table_dir / "performance_summary.csv",
    )


if __name__ == "__main__":
    main()
