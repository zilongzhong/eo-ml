
import time

import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.fire_dataset import FireSegmentationDataset
from src.models.unet import build_model
from src.losses import masked_bce_loss


CONFIG_PATH = "configs/fire_segmentation.yaml"


def main():

    # --------------------------------------------------
    # 1. Configuration
    # --------------------------------------------------

    config = load_config(CONFIG_PATH)

    torch.manual_seed(42)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU unavailable. "
            "Start Docker with --gpus all."
        )

    device = torch.device("cuda")

    print("=" * 70)
    print("FIRE SEGMENTATION GPU SMOKE TEST")
    print("=" * 70)

    print("PyTorch version:", torch.__version__)
    print("GPU:", torch.cuda.get_device_name(0))

    # --------------------------------------------------
    # 2. Dataset / DataLoader
    # --------------------------------------------------

    dataset = FireSegmentationDataset(
        config_path=CONFIG_PATH,
        split="train",
    )

    batch_size = min(
        4,
        int(config["training"]["batch_size"]),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    batch = next(iter(loader))

    images = batch["image"].to(
        device,
        non_blocking=True,
    )

    targets = batch["mask"].to(
        device,
        non_blocking=True,
    )

    valid_mask = batch["valid_mask"].to(
        device,
        non_blocking=True,
    )

    print("\nBATCH")
    print("Images:", tuple(images.shape))
    print("Targets:", tuple(targets.shape))
    print("Valid mask:", tuple(valid_mask.shape))
    print("Image dtype:", images.dtype)

    print(
        "Valid pixels:",
        f"{valid_mask.float().mean().item():.2%}",
    )

    assert images.shape[1] == 8
    assert torch.isfinite(images).all()
    assert torch.any(valid_mask)

    # --------------------------------------------------
    # 3. Build model
    # --------------------------------------------------

    model = build_model(
        config["model"]
    ).to(device)

    model.train()

    n_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nMODEL")
    print("Architecture:", config["model"]["name"])
    print("Encoder:", config["model"]["encoder"])
    print("Trainable parameters:", f"{n_parameters:,}")

    # --------------------------------------------------
    # 4. Optimizer and AMP
    # --------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(
            config["training"]["learning_rate"]
        ),
        weight_decay=1e-4,
    )

    use_amp = bool(
        config["training"].get("amp", True)
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    first_parameter = next(model.parameters())

    before_update = (
        first_parameter.detach().clone()
    )

    # --------------------------------------------------
    # 5. One forward/backward/update iteration
    # --------------------------------------------------

    optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=use_amp,
    ):

        logits = model(images)

        if logits.shape != targets.shape:
            raise RuntimeError(
                f"Unexpected model output: "
                f"{tuple(logits.shape)}"
            )

        loss = masked_bce_loss(
            logits=logits,
            targets=targets,
            valid_mask=valid_mask,
        )

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Non-finite loss: {loss.item()}"
        )

    scaler.scale(loss).backward()

    # Unscale gradients before inspecting/clipping.
    scaler.unscale_(optimizer)

    # Verify finite gradients.
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            if not torch.isfinite(
                parameter.grad
            ).all():
                raise RuntimeError(
                    f"Non-finite gradient: {name}"
                )

    gradient_norm = (
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )
    )

    scaler.step(optimizer)
    scaler.update()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    parameters_changed = not torch.equal(
        before_update,
        first_parameter.detach(),
    )

    if not parameters_changed:
        raise RuntimeError(
            "Optimizer did not update the first "
            "model parameter."
        )

    # --------------------------------------------------
    # 6. Verify results
    # --------------------------------------------------

    print("\nTRAINING RESULT")
    print("Output shape:", tuple(logits.shape))
    print("Loss:", f"{loss.item():.6f}")
    print(
        "Gradient norm before clipping:",
        f"{gradient_norm.item():.6f}",
    )
    print("AMP enabled:", use_amp)
    print("Parameter updated:", parameters_changed)
    print("Iteration time:", f"{elapsed:.3f} seconds")

    with torch.no_grad():
        probabilities = torch.sigmoid(
            logits.float()
        )

    print(
        "Probability range:",
        f"[{probabilities.min().item():.4f}, "
        f"{probabilities.max().item():.4f}]",
    )

    print("\nGPU MEMORY")
    print(
        "Peak allocated:",
        f"{torch.cuda.max_memory_allocated() / 1024**3:.3f} GB",
    )

    print(
        "Peak reserved:",
        f"{torch.cuda.max_memory_reserved() / 1024**3:.3f} GB",
    )

    print("\nSUCCESS: One GPU training iteration completed.")


if __name__ == "__main__":
    main()
