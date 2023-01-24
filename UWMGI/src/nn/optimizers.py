"""Optimizers related Utilities"""
import typing

import torch
from torch import optim


@typing.no_type_check
def fetch_optimizer(
    params, learning_rate: float, weight_decay: float, optimizer: str = "AdamW"
) -> torch.optim.Optimizer:
    """
    Returns the Optimizer based on provided parameters

    :param params: Model parameters
    :param learning_rate: Learning Rate to be used
    :type learning_rate: float
    :param weight_decay: Weight Decay to be used
    :type weight_decay: float
    :param optimizer: Which Optimizer to use, defaults to "AdamW"
    :type optimizer: str, optional
    :return: Optimizer based on the provided input
    :rtype: torch.optim.Optimizer
    """
    if optimizer == "AdamW":
        optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "Adam":
        optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "NAdam":
        optimizer = optim.NAdam(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer is None:
        return None

    return optimizer
