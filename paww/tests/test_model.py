import pytest
import torch.nn as nn

from paww.nn import Model

from .utils import get_dataloader

# -------- Test a EfficientNet instantiation -------- #


@pytest.mark.actions
def test_efficient() -> None:

    cfg = dict(
        model_name="tf_efficientnet_b0_ns", model_type="EfficientNet", target_size=1
    )
    net = Model(cfg, pretrained=False)
    assert isinstance(net, nn.Module)


# -------- Test a EfficientNet Forward Pass -------- #


def test_forwardpass_efficientnet() -> None:

    cfg = dict(
        model_name="tf_efficientnet_b0_ns",
        model_type="EfficientNet",
        target_size=1,
        target_col="Pawpularity",
        folds=5,
        seed=42,
        size=256,
        batch_size=32,
        num_workers=4,
    )

    net = Model(cfg, pretrained=False)

    loader = get_dataloader(cfg)

    for step, (images, labels) in enumerate(loader):
        y_preds = net(images)  # noqa: F841
        break


# -------- Test a ResNet instantiation -------- #


@pytest.mark.actions
def test_resnet() -> None:

    cfg = dict(model_name="resnet18", model_type="ResNet", target_size=1)
    net = Model(cfg, pretrained=False)
    assert isinstance(net, nn.Module)


# -------- Test a ResNet Forward Pass -------- #


@pytest.mark.skip(reason="still figuring out how to make ResNets work")
def test_forwardpass_resnet() -> None:

    cfg = dict(
        model_name="resnet18",
        model_type="ResNet",
        target_size=1,
        target_col="Pawpularity",
        folds=5,
        seed=42,
        size=256,
        batch_size=32,
        num_workers=4,
    )

    net = Model(cfg, pretrained=False)

    loader = get_dataloader(cfg)

    for step, (images, labels) in enumerate(loader):
        y_preds = net(images)  # noqa: F841
        break
