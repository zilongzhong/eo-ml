
import torch


@torch.no_grad()
def binary_segmentation_counts(
    logits,
    targets,
    valid_mask,
    threshold=0.5,
):
    """
    Calculate binary segmentation confusion counts.

    Invalid pixels are excluded.

    Returns:
        tp, fp, fn, tn
    """

    if logits.shape != targets.shape:
        raise ValueError("Logits/targets shape mismatch")

    if logits.shape != valid_mask.shape:
        raise ValueError("Validity mask shape mismatch")

    if not 0.0 < threshold < 1.0:
        raise ValueError("Threshold must be between 0 and 1")

    prediction = (
        torch.sigmoid(logits.float()) >= threshold
    )

    target = targets >= 0.5
    valid = valid_mask.bool()

    tp = (prediction & target & valid).sum().item()
    fp = (prediction & ~target & valid).sum().item()
    fn = (~prediction & target & valid).sum().item()
    tn = (~prediction & ~target & valid).sum().item()

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def metrics_from_counts(counts):
    """
    Compute metrics from accumulated confusion counts.
    """

    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]

    precision = (
        tp / (tp + fp)
        if tp + fp > 0 else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0 else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn > 0 else 1.0
    )

    accuracy = (
        (tp + tn) / (tp + fp + fn + tn)
        if tp + fp + fn + tn > 0 else 0.0
    )

    return {
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }
