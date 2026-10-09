
import argparse
import json
import time
from pathlib import Path

import numpy as np
import rasterio
import torch

from torch.utils.data import DataLoader, Subset

from src.config import load_config
from src.data.fire_dataset import FireSegmentationDataset
from src.data.overlap import ProbabilityMosaic
from src.models.unet import build_model
from src.metrics import metrics_from_counts


CONFIG_PATH = "configs/fire_segmentation.yaml"
NODATA_FLOAT = -9999.0
NODATA_CLASS = 255


def write_geotiff(path, array, geo, nodata, description):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": geo["height"],
        "width": geo["width"],
        "count": 1,
        "dtype": array.dtype.name,
        "crs": geo["crs"],
        "transform": geo["transform"],
        "nodata": nodata,
        "compress": "deflate",
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)
        dst.set_band_description(1, description)

    print(f"Saved: {path}")


def evaluate_map(prediction, reference, valid):
    if not np.isin(reference[valid], [0, 1]).all():
        raise ValueError("Unexpected reference labels")

    positive = prediction == 1
    negative = prediction == 0
    truth = reference == 1

    counts = {
        "tp": int(np.count_nonzero(
            positive & truth & valid
        )),
        "fp": int(np.count_nonzero(
            positive & ~truth & valid
        )),
        "fn": int(np.count_nonzero(
            negative & truth & valid
        )),
        "tn": int(np.count_nonzero(
            negative & ~truth & valid
        )),
    }

    return counts, metrics_from_counts(counts)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the best model checkpoint",
    )

    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    config = load_config(CONFIG_PATH)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Only load checkpoints that you created or trust.
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    model = build_model(
        checkpoint["config"]["model"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    use_amp = (
        device.type == "cuda"
        and bool(config["training"].get("amp", True))
    )

    threshold = float(
        config["training"].get("threshold", 0.5)
    )

    dataset = FireSegmentationDataset(
        config_path=CONFIG_PATH,
        split=args.split,
    )

    # Ensure the normalization used for inference matches
    # the normalization recorded in the checkpoint.
    checkpoint_stats = checkpoint["training_stats"]

    if not np.allclose(
        dataset.mean,
        checkpoint_stats["mean"],
        rtol=1e-6,
        atol=1e-7,
    ):
        raise ValueError(
            "Normalization statistics differ from checkpoint"
        )

    if not np.allclose(
        dataset.std,
        checkpoint_stats["std"],
        rtol=1e-6,
        atol=1e-7,
    ):
        raise ValueError(
            "Normalization std differs from checkpoint"
        )

    run_name = Path(args.checkpoint).parent.name

    print("=" * 75)
    print("FULL-REGION FIRE INFERENCE")
    print("=" * 75)
    print("Checkpoint:", args.checkpoint)
    print("Device:", device)
    print("Split:", args.split)
    print("Threshold:", threshold)

    all_regions = sorted(
        dataset.records["region"].unique()
    )

    for region in all_regions:
        region_indices = dataset.records.index[
            dataset.records["region"] == region
        ].tolist()

        region_records = dataset.records.loc[
            region_indices
        ]

        image_paths = region_records["image_path"].unique()
        target_paths = region_records["target_path"].unique()

        if len(image_paths) != 1 or len(target_paths) != 1:
            raise ValueError(
                f"Region {region} has inconsistent files"
            )

        image_path = str(image_paths[0])
        target_path = str(target_paths[0])

        with rasterio.open(image_path) as src:
            geo = {
                "height": src.height,
                "width": src.width,
                "crs": src.crs,
                "transform": src.transform,
            }

        mosaic = ProbabilityMosaic(
            geo["height"],
            geo["width"],
        )

        subset = Subset(dataset, region_indices)

        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
        )

        print(f"\nRegion {region}")
        print("Image:", image_path)
        print("Patches:", len(subset))

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.inference_mode():

            for batch in loader:
                images = batch["image"].to(
                    device,
                    non_blocking=True,
                )

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    logits = model(images)

                probabilities = (
                    torch.sigmoid(logits.float())
                    .cpu()
                    .numpy()[:, 0]
                )

                validity = (
                    batch["valid_mask"]
                    .numpy()[:, 0]
                )

                for i in range(probabilities.shape[0]):
                    mosaic.add(
                        probability=probabilities[i],
                        valid_mask=validity[i],
                        row_off=int(batch["row_off"][i]),
                        col_off=int(batch["col_off"][i]),
                    )

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        probability, coverage = mosaic.finalize(
            nodata=NODATA_FLOAT
        )

        covered = coverage > 0

        prediction = np.full(
            covered.shape,
            NODATA_CLASS,
            dtype=np.uint8,
        )

        prediction[covered] = (
            probability[covered] >= threshold
        ).astype(np.uint8)

        with rasterio.open(target_path) as src:
            reference = src.read(1)

            if reference.shape != covered.shape:
                raise ValueError(
                    "Reference shape does not match predictions"
                )

        valid_evaluation = (
            covered
            & np.isin(reference, [0, 1])
        )

        counts, metrics = evaluate_map(
            prediction,
            reference,
            valid_evaluation,
        )

        # Error-map coding:
        # 0 = true negative
        # 1 = true positive
        # 2 = false positive
        # 3 = false negative
        # 255 = NoData
        error_map = np.full(
            covered.shape,
            NODATA_CLASS,
            dtype=np.uint8,
        )

        error_map[
            valid_evaluation
            & (prediction == 0)
            & (reference == 0)
        ] = 0

        error_map[
            valid_evaluation
            & (prediction == 1)
            & (reference == 1)
        ] = 1

        error_map[
            valid_evaluation
            & (prediction == 1)
            & (reference == 0)
        ] = 2

        error_map[
            valid_evaluation
            & (prediction == 0)
            & (reference == 1)
        ] = 3

        output_dir = (
            Path(config["output"]["output_dir"])
            / "maps"
            / run_name
            / f"region{region}"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        write_geotiff(
            output_dir / "burn_probability.tif",
            probability,
            geo,
            NODATA_FLOAT,
            "Burned-area probability",
        )

        write_geotiff(
            output_dir / "burn_prediction.tif",
            prediction,
            geo,
            NODATA_CLASS,
            "Binary burned-area prediction",
        )

        write_geotiff(
            output_dir / "patch_coverage.tif",
            coverage,
            geo,
            0,
            "Number of valid overlapping predictions",
        )

        write_geotiff(
            output_dir / "error_map.tif",
            error_map,
            geo,
            NODATA_CLASS,
            "0=TN, 1=TP, 2=FP, 3=FN",
        )

        report = {
            "run_name": run_name,
            "split": args.split,
            "region": int(region),
            "checkpoint": str(args.checkpoint),
            "image_path": image_path,
            "reference_path": target_path,
            "threshold": threshold,
            "device": str(device),
            "inference_seconds": elapsed,
            "total_pixels": int(covered.size),
            "covered_pixels": int(covered.sum()),
            "coverage_fraction": float(covered.mean()),
            "evaluated_pixels": int(
                valid_evaluation.sum()
            ),
            "confusion_counts": counts,
            "metrics": metrics,
        }

        metrics_path = output_dir / "metrics.json"

        with open(metrics_path, "w") as f:
            json.dump(report, f, indent=2)

        print("\nFULL-REGION RESULTS")
        print("Coverage:", f"{covered.mean():.4%}")
        print("IoU:", f"{metrics['iou']:.4f}")
        print("F1:", f"{metrics['f1']:.4f}")
        print("Precision:", f"{metrics['precision']:.4f}")
        print("Recall:", f"{metrics['recall']:.4f}")
        print("Inference time:", f"{elapsed:.2f}s")
        print("Output:", output_dir)


if __name__ == "__main__":
    main()
