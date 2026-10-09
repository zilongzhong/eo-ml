
import segmentation_models_pytorch as smp


def build_unet(
    in_channels=8,
    num_classes=1,
    encoder="resnet34",
):
    """
    U-Net for multispectral binary segmentation.

    Input:
        [B, C, H, W]

    Output:
        [B, 1, H, W] logits

    The output is not passed through sigmoid.
    """

    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )

    return model


def build_model(model_config):
    """
    Build a model from the YAML configuration.
    """

    model_name = model_config["name"].lower()

    if model_name == "unet":
        return build_unet(
            in_channels=int(
                model_config["in_channels"]
            ),
            num_classes=int(
                model_config["num_classes"]
            ),
            encoder=model_config["encoder"],
        )

    raise ValueError(
        f"Unsupported model: {model_name}"
    )
