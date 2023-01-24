"""Model building Utilites"""
import segmentation_models_pytorch as smp
import torch


def build_model(
    device: torch.device = torch.device("cuda:0"),
    arch: str = "Unet",
    backbone: str = "efficientnet-b7",
    num_classes: int = 3,
) -> torch.nn.Module:
    """build_model Returns Model given the configuration

    :param device: Which device to use, defaults to torch.device("cuda:0")
    :type device: torch.device, optional
    :param arch: Which architecture from segmentation_models_pytorch to use,
                defaults to "Unet"
    :type arch: str, optional
    :param backbone: Backbone architecture for the model, defaults to "efficientnet-b7"
    :type backbone: str, optional
    :param num_classes: Number of Classes for the model, defaults to 3
    :type num_classes: int, optional
    :return: A Segmentation Model
    :rtype: torch.nn.Module
    """
    if arch == "Unet":
        model = smp.Unet(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None,
        )
    elif arch == "UnetPlusPlus":
        model = smp.UnetPlusPlus(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None,
        )
    elif arch == "PAN":
        model = smp.PAN(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None,
        )
    elif arch == "MAnet":
        model = smp.MAnet(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None,
        )
    elif arch == "DeepLabV3":
        model = smp.DeepLabV3(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None,
        )
    elif arch == "DeepLabV3Plus":
        model = smp.DeepLabV3Plus(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None,
        )

    # Move to Device
    model.to(device)
    return model


def load_model(path: str) -> torch.nn.Module:
    """
    Loads a model given weights

    :param path: Path to the weights
    :type path: str
    :return: Model with loaded weights
    :rtype: torch.nn.Module
    """
    model = build_model()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
