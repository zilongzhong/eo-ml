
import argparse
import csv
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.config import load_config
from src.data.fire_dataset import FireSegmentationDataset
from src.data.visualization import landsat_rgb_indices, reflectance_rgb, rgb_display_settings
from src.models.unet import build_model
from src.losses import masked_bce_loss
from src.metrics import (
    binary_segmentation_counts,
    metrics_from_counts,
)


CONFIG_PATH = "configs/fire_segmentation.yaml"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_loader(dataset, batch_size, shuffle, workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    use_amp,
    writer,
    global_step,
):
    model.train()

    loss_sum = 0.0
    pixel_count = 0

    for batch in loader:
        images = batch["image"].to(
            device, non_blocking=True
        )
        targets = batch["mask"].to(
            device, non_blocking=True
        )
        valid = batch["valid_mask"].to(
            device, non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)

            loss = masked_bce_loss(
                logits,
                targets,
                valid,
            )

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        scaler.step(optimizer)
        scaler.update()

        n_valid = int(valid.sum().item())

        loss_sum += loss.item() * n_valid
        pixel_count += n_valid

        global_step += 1

        writer.add_scalar(
            "train/step_loss",
            loss.item(),
            global_step,
        )

    return loss_sum / pixel_count, global_step


@torch.no_grad()
def validate(
    model,
    loader,
    device,
    use_amp,
    threshold,
):
    model.eval()

    loss_sum = 0.0
    pixel_count = 0

    counts = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
    }

    example = None

    for batch in loader:
        images = batch["image"].to(
            device, non_blocking=True
        )
        targets = batch["mask"].to(
            device, non_blocking=True
        )
        valid = batch["valid_mask"].to(
            device, non_blocking=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)

            loss = masked_bce_loss(
                logits,
                targets,
                valid,
            )

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite validation loss")

        n_valid = int(valid.sum().item())

        loss_sum += loss.item() * n_valid
        pixel_count += n_valid

        batch_counts = binary_segmentation_counts(
            logits,
            targets,
            valid,
            threshold=threshold,
        )

        for key in counts:
            counts[key] += batch_counts[key]

        # Save a small fixed validation batch for
        # TensorBoard visualization.
        if example is None:
            example = {
                "images": images[:3].detach().cpu(),
                "targets": targets[:3].detach().cpu(),
                "valid": valid[:3].detach().cpu(),
                "regions": batch["region"][:3].tolist(),
                "probabilities": (
                    torch.sigmoid(logits[:3].float())
                    .detach()
                    .cpu()
                ),
            }

    metrics = metrics_from_counts(counts)

    return (
        loss_sum / pixel_count,
        metrics,
        counts,
        example,
    )


def log_validation_images(
    writer,
    example,
    mean,
    std,
    epoch,
    threshold,
    rgb_indices_by_region,
    rgb_settings=None,
):
    """
    Log RGB, ground truth, predicted probability,
    and binary prediction images to TensorBoard.
    """

    images = example["images"].float()
    targets = example["targets"].float()
    valid = example["valid"].bool()
    probabilities = example["probabilities"]

    mean = torch.as_tensor(
        mean, dtype=torch.float32
    )
    std = torch.as_tensor(
        std, dtype=torch.float32
    )

    rendered = []
    for n, region in enumerate(example["regions"]):
        indices = rgb_indices_by_region[region]
        reflectance = (
            images[n, indices] * std[indices, None, None]
            + mean[indices, None, None]
        )
        rgb = reflectance_rgb(
            reflectance.numpy(), valid[n, 0].numpy(), **(rgb_settings or {})
        )
        rendered.append(torch.from_numpy(rgb).permute(2, 0, 1))
    rgb = torch.stack(rendered)

    binary_prediction = (
        probabilities >= threshold
    ).float()

    writer.add_images(
        "val/RGB",
        rgb,
        epoch,
    )

    writer.add_images(
        "val/ground_truth",
        targets.repeat(1, 3, 1, 1),
        epoch,
    )

    writer.add_images(
        "val/probability",
        probabilities.repeat(1, 3, 1, 1),
        epoch,
    )

    writer.add_images(
        "val/binary_prediction",
        binary_prediction.repeat(1, 3, 1, 1),
        epoch,
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    config,
    stats,
    metrics,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": config,
            "training_stats": stats,
            "validation_metrics": metrics,
        },
        path,
    )


def save_history(path, history):
    if not history:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(history[0].keys()),
        )
        writer.writeheader()
        writer.writerows(history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    train_cfg = config["training"]

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable. Start Docker with --gpus all."
        )

    device = torch.device("cuda")

    batch_size = int(train_cfg["batch_size"])
    epochs = int(train_cfg["epochs"])
    workers = int(train_cfg.get("num_workers", 2))

    use_amp = bool(train_cfg.get("amp", True))
    threshold = float(train_cfg.get("threshold", 0.5))
    patience = int(
        train_cfg.get("early_stopping_patience", 8)
    )

    run_name = args.run_name or datetime.now().strftime("fire_%Y%m%d_%H%M%S_%f")
    if not run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_name):
        raise ValueError("Run name may contain only letters, numbers, '_' and '-'")

    output_cfg = config["output"]

    checkpoint_dir = (
        Path(output_cfg["checkpoint_dir"]) / run_name
    )

    log_dir = (
        Path(output_cfg["tensorboard_dir"]) / run_name
    )

    history_path = (
        Path(output_cfg["output_dir"])
        / "tables"
        / f"{run_name}_training_history.csv"
    )

    checkpoint_dir.mkdir(
        parents=True, exist_ok=False
    )
    log_dir.mkdir(
        parents=True, exist_ok=True
    )

    writer = SummaryWriter(log_dir=str(log_dir))

    print("=" * 75)
    print("FIRE SEGMENTATION TRAINING")
    print("=" * 75)
    print("Run:", run_name)
    print("Device:", torch.cuda.get_device_name(0))
    print("TensorBoard:", log_dir)
    print("Checkpoints:", checkpoint_dir)

    # --------------------------------------------------
    # Datasets
    # --------------------------------------------------

    train_dataset = FireSegmentationDataset(
        args.config, split="train"
    )

    val_dataset = FireSegmentationDataset(
        args.config, split="val"
    )
    rgb_indices_by_region = {}
    for region, records in val_dataset.records.groupby("region"):
        with rasterio.open(records.iloc[0]["image_path"]) as src:
            rgb_indices_by_region[int(region)] = landsat_rgb_indices(
                src.descriptions, config["data"]["bands"]
            )

    train_loader = create_loader(
        train_dataset,
        batch_size,
        shuffle=True,
        workers=workers,
    )

    val_loader = create_loader(
        val_dataset,
        batch_size,
        shuffle=False,
        workers=workers,
    )

    print("Training patches:", len(train_dataset))
    print("Validation patches:", len(val_dataset))

    # --------------------------------------------------
    # Model and optimizer
    # --------------------------------------------------

    model = build_model(
        config["model"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(
            train_cfg.get("weight_decay", 1e-4)
        ),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    writer.add_text(
        "experiment/model",
        str(config["model"]),
        0,
    )

    writer.add_text(
        "experiment/splits",
        "Train: Region 1; Validation: Region 2; "
        "Test: Region 3 (not used during training)",
        0,
    )

    best_iou = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    global_step = 0
    history = []

    torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()

        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            writer=writer,
            global_step=global_step,
        )

        (
            val_loss,
            val_metrics,
            counts,
            example,
        ) = validate(
            model=model,
            loader=val_loader,
            device=device,
            use_amp=use_amp,
            threshold=threshold,
        )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        # TensorBoard scalars
        writer.add_scalar(
            "train/loss", train_loss, epoch
        )
        writer.add_scalar(
            "val/loss", val_loss, epoch
        )
        writer.add_scalar(
            "val/iou", val_metrics["iou"], epoch
        )
        writer.add_scalar(
            "val/f1", val_metrics["f1"], epoch
        )
        writer.add_scalar(
            "val/precision",
            val_metrics["precision"],
            epoch,
        )
        writer.add_scalar(
            "val/recall",
            val_metrics["recall"],
            epoch,
        )
        writer.add_scalar(
            "val/accuracy",
            val_metrics["accuracy"],
            epoch,
        )
        writer.add_scalar(
            "train/learning_rate",
            current_lr,
            epoch,
        )
        writer.add_scalar(
            "performance/epoch_seconds",
            elapsed,
            epoch,
        )

        ## Image logging: first epoch and every 5 epochs.
        #if epoch == 1 or epoch % 5 == 0:
        #    log_validation_images(
        #        writer,
        #        example,
        #        mean=val_dataset.mean,
        #        std=val_dataset.std,
        #        epoch=epoch,
        #        threshold=threshold,
        #        rgb_indices_by_region=rgb_indices_by_region,
        #        rgb_settings=rgb_display_settings(config),
        #    )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_iou": val_metrics["iou"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_accuracy": val_metrics["accuracy"],
            "learning_rate": current_lr,
            "epoch_seconds": elapsed,
        })

        scheduler.step()

        # Save best model by validation IoU.
        improved = val_metrics["iou"] > best_iou + 1e-8

        if improved:
            best_iou = val_metrics["iou"]
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                config,
                train_dataset.stats,
                val_metrics,
            )
        else:
            epochs_without_improvement += 1

        # Save current model every epoch.
        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            config,
            train_dataset.stats,
            val_metrics,
        )

        save_history(history_path, history)
        writer.flush()

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"IoU: {val_metrics['iou']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"Precision: {val_metrics['precision']:.4f} | "
            f"Recall: {val_metrics['recall']:.4f} | "
            f"Time: {elapsed:.1f}s"
            + (" | BEST" if improved else "")
        )

        if epochs_without_improvement >= patience:
            print(
                f"\nEarly stopping: no IoU improvement "
                f"for {patience} epochs."
            )
            break

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    peak_memory_gb = (
        torch.cuda.max_memory_allocated() / 1024**3
    )

    print("\n" + "=" * 75)
    print("TRAINING COMPLETED")
    print("=" * 75)
    print("Best epoch:", best_epoch)
    print("Best validation IoU:", f"{best_iou:.4f}")
    print("Peak allocated GPU memory:",
          f"{peak_memory_gb:.3f} GB")
    print("Best checkpoint:", checkpoint_dir / "best.pt")
    print("Last checkpoint:", checkpoint_dir / "last.pt")
    print("History CSV:", history_path)
    print("TensorBoard:", log_dir)

    writer.close()


if __name__ == "__main__":
    main()
