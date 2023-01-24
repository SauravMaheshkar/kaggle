"""Custom Model Classes and Related Utilites"""
from typing import Dict

import monai
import pytorch_lightning as pl
import torch
from torch import nn

__all__ = ["LitModule"]


class LitModule(pl.LightningModule):
    """Custom PyTorch Lightning Module"""

    def __init__(
        self,
        learning_rate: float,  # pylint: disable=W0613
        weight_decay: float,  # pylint: disable=W0613
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.model = self._init_model()
        self.loss_fn = self._init_loss_fn()

    def _init_model(self) -> nn.Module:
        """Initializes Model"""
        return monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

    def _init_loss_fn(self):
        """Initializes Loss Function"""
        return monai.losses.DiceLoss(sigmoid=True)

    def configure_optimizers(self):
        """Configures Optimizers"""
        return torch.optim.AdamW(
            params=self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Computes a Forward Pass"""
        return self.model(images)

    def training_step(  # type: ignore
        self, batch: Dict, batch_idx: int  # pylint: disable=W0613
    ) -> torch.Tensor:
        """Perform a Training Step"""
        images, masks = batch["image"], batch["mask"]
        outputs = self(images)

        loss = self.loss_fn(outputs, masks)

        self.log("train_loss", loss, batch_size=images.shape[0])

        return loss

    def validation_step(  # type: ignore
        self, batch: Dict, batch_idx: int  # pylint: disable=W0613
    ) -> None:
        """Perform a Validation Step"""
        images, masks = batch["image"], batch["mask"]
        outputs = self(images)

        loss = self.loss_fn(outputs, masks)

        self.log("val_loss", loss, prog_bar=True, batch_size=images.shape[0])

    @classmethod
    def load_eval_checkpoint(cls, checkpoint_path: str, device: str) -> nn.Module:
        """Loads a Model from Evaluation Checkpoint"""
        module = cls.load_from_checkpoint(checkpoint_path=checkpoint_path).to(device)
        module.eval()

        return module
