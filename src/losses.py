
import torch
import torch.nn.functional as F


def masked_bce_loss(
    logits,
    targets,
    valid_mask,
):
    """
    Binary cross-entropy loss for segmentation,
    excluding invalid pixels.

    Parameters
    ----------
    logits : torch.Tensor
        [B,1,H,W], raw model outputs.

    targets : torch.Tensor
        [B,1,H,W], binary ground truth.

    valid_mask : torch.Tensor
        [B,1,H,W], boolean validity mask.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """

    if logits.shape != targets.shape:
        raise ValueError(
            "Logits and targets must have the same shape"
        )

    if logits.shape != valid_mask.shape:
        raise ValueError(
            "Validity mask shape mismatch"
        )

    if not torch.any(valid_mask):
        raise ValueError(
            "No valid pixels in batch"
        )

    loss_per_pixel = (
        F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            reduction="none",
        )
    )

    weights = valid_mask.to(
        dtype=loss_per_pixel.dtype
    )

    loss = (
        (loss_per_pixel * weights).sum()
        / weights.sum()
    )

    return loss
