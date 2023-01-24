from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import (CosineAnnealingLR, CosineAnnealingWarmRestarts,
                                      ReduceLROnPlateau)

__all__ = ["get_scheduler", "get_optimizer"]


def get_optimizer(cfg: dict, model):

    # AdamW
    if cfg["optimizer"] == "AdamW":
        optimizer = AdamW(  # type: ignore
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            amsgrad=cfg["amsgrad"],
        )

    # Adam
    if cfg["optimizer"] == "Adam":
        optimizer = Adam(  # type: ignore
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            amsgrad=cfg["amsgrad"],
        )

    # SGD
    if cfg["optimizer"] == "SGD":
        optimizer = SGD(  # type: ignore
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )

    return optimizer


def get_scheduler(cfg: dict, optimizer):

    # ReduceLROnPlateau
    if cfg["scheduler"] == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(  # type: ignore
            optimizer,
            mode="min",
            factor=cfg["factor"],
            patience=cfg["patience"],
            verbose=True,
            eps=cfg["eps"],
        )

    # CosineAnnealingLR
    elif cfg["scheduler"] == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(  # type: ignore
            optimizer, T_max=cfg["T_max"], eta_min=cfg["min_lr"], last_epoch=-1
        )

    # CosineAnnealingWarmRestarts
    elif cfg["scheduler"] == "CosineAnnealingWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(  # type: ignore
            optimizer, T_0=cfg["T_0"], T_mult=1, eta_min=cfg["min_lr"], last_epoch=-1
        )

    return scheduler
