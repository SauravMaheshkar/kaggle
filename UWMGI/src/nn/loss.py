"""Loss Function Utilities"""
from typing import Tuple

import segmentation_models_pytorch as smp
import torch

JaccardLoss = smp.losses.JaccardLoss(mode="multilabel")
DiceLoss = smp.losses.DiceLoss(mode="multilabel")
BCELoss = smp.losses.SoftBCEWithLogitsLoss()
LovaszLoss = smp.losses.LovaszLoss(mode="multilabel", per_image=False)
TverskyLoss = smp.losses.TverskyLoss(mode="multilabel", log_loss=False)


def dice_coef(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    thr: float = 0.5,
    dim: Tuple[int, int] = (2, 3),
    epsilon: float = 0.001,
) -> torch.Tensor:
    """
    Get Dice Coefficients given predicted and true images

    :param y_true: Ground Truth
    :type y_true: torch.Tensor
    :param y_pred: Predicted Output
    :type y_pred: torch.Tensor
    :param thr: Threshold, defaults to 0.5
    :type thr: float, optional
    :param dim: Dimensionality for Calculation, defaults to (2, 3)
    :type dim: Tuple[int, int], optional
    :param epsilon: Epsilon Value for Computation, defaults to 0.001
    :type epsilon: float, optional
    :return: The Dice Coefficient
    :rtype: torch.Tensor
    """
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=(1, 0))
    return dice


def iou_coef(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    thr: float = 0.5,
    dim: Tuple[int, int] = (2, 3),
    epsilon: float = 0.001,
) -> torch.Tensor:
    """
    Compute the IoU Coefficient

    :param y_true: Ground Truth
    :type y_true: torch.Tensor
    :param y_pred: Predicted Output
    :type y_pred: torch.Tensor
    :param thr: Threshold, defaults to 0.5
    :type thr: float, optional
    :param dim: Dimensionality for Computation, defaults to (2, 3)
    :type dim: Tuple[int, int], optional
    :param epsilon: Epsilon for Computation, defaults to 0.001
    :type epsilon: float, optional
    :return: IoU Coefficient
    :rtype: torch.Tensor
    """
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)
    iou = ((inter + epsilon) / (union + epsilon)).mean(dim=(1, 0))
    return iou


def criterion(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    lovasz_weight: float = 0.0,
    dice_weight: float = 0.6,
    jaccard_weight: float = 0.0,
    bce_weight: float = 0.4,
    tversky_weight: float = 0.0,
) -> torch.Tensor:
    """
    Loss Computation

    :param y_pred: Predicted Output
    :type y_pred: torch.Tensor
    :param y_true: Ground Truth
    :type y_true: torch.Tensor
    :param lovasz_weight: Weight for the Lovasz Loss, defaults to 0.0
    :type lovasz_weight: float, optional
    :param dice_weight: Weight for the Dice Loss, defaults to 0.6
    :type dice_weight: float, optional
    :param jaccard_weight: Weight for the Jaccard Loss, defaults to 0.0
    :type jaccard_weight: float, optional
    :param bce_weight: Weight for the BCE Loss, defaults to 0.4
    :type bce_weight: float, optional
    :param tversky_weight: Weights for the Tversky Loss, defaults to 0.0
    :type tversky_weight: float, optional
    :return: Computed Loss
    :rtype: torch.Tensor
    """
    return (
        bce_weight * BCELoss(y_pred, y_true)
        + tversky_weight * TverskyLoss(y_pred, y_true)
        + lovasz_weight * LovaszLoss(y_pred, y_true)
        + dice_weight * DiceLoss(y_pred, y_true)
        + jaccard_weight * JaccardLoss(y_pred, y_true)
    )
