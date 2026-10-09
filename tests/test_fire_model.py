
import torch

from src.models.unet import build_unet
from src.losses import masked_bce_loss


def test_unet_output_shape():

    model = build_unet(
        in_channels=8,
        num_classes=1,
        encoder="resnet34",
    )

    model.eval()

    x = torch.randn(
        1, 8, 64, 64
    )

    with torch.no_grad():
        output = model(x)

    assert output.shape == (
        1, 1, 64, 64
    )

    assert torch.isfinite(output).all()


def test_masked_bce_finite():

    logits = torch.randn(
        2, 1, 16, 16,
        requires_grad=True,
    )

    targets = torch.randint(
        0, 2,
        (2, 1, 16, 16),
    ).float()

    valid_mask = torch.ones(
        2, 1, 16, 16,
        dtype=torch.bool,
    )

    loss = masked_bce_loss(
        logits,
        targets,
        valid_mask,
    )

    assert torch.isfinite(loss)

    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(
        logits.grad
    ).all()


def test_masked_bce_ignores_invalid_pixels():

    logits = torch.tensor(
        [[[[0.0, 5.0]]]],
        requires_grad=True,
    )

    targets = torch.tensor(
        [[[[1.0, 0.0]]]]
    )

    valid_mask = torch.tensor(
        [[[[True, False]]]]
    )

    loss = masked_bce_loss(
        logits,
        targets,
        valid_mask,
    )

    expected = (
        torch.nn.functional
        .binary_cross_entropy_with_logits(
            logits[:, :, :, :1],
            targets[:, :, :, :1],
        )
    )

    assert torch.allclose(
        loss,
        expected,
    )

    loss.backward()

    assert logits.grad[
        0, 0, 0, 1
    ].item() == 0.0
