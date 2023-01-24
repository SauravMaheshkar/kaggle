"""Custom Data Classes and Related Utilites"""
from typing import Any, Callable, Dict, Tuple

import monai
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import tifffile
from monai.data import CSVDataset, DataLoader, ImageReader

__all__ = ["LitDataModule"]


class TIFFImageReader(ImageReader):
    """Custom TIFF Image Reader"""

    def read(self, data: str) -> np.ndarray:  # type: ignore
        """Reads a TIFF Image File"""
        return tifffile.imread(data)

    def get_data(self, img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Read Image Data in the desired format

        :param img: Image as a numpy array
        :type img: np.ndarray
        :return: Image and metadata dictionary
        :rtype: Tuple[np.ndarray, Dict[str, Any]]
        """
        return img, {"spatial_shape": np.asarray(img.shape), "original_channel_dim": -1}

    def verify_suffix(self, filename: str) -> bool:  # type: ignore
        """Verifies that the file is TIFF"""
        return ".tiff" in filename


class LitDataModule(pl.LightningDataModule):  # pylint: disable=R0902
    """Custom PyTorch Lightning Data Module"""

    def __init__(
        self,
        train_csv_path: str,  # pylint: disable=W0613
        test_csv_path: str,  # pylint: disable=W0613
        spatial_size: int,  # pylint: disable=W0613
        val_fold: int,  # pylint: disable=W0613
        batch_size: int,  # pylint: disable=W0613
        num_workers: int,  # pylint: disable=W0613
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.train_df = pd.read_csv(train_csv_path)
        self.test_df = pd.read_csv(test_csv_path)

        (
            self.train_transform,
            self.val_transform,
            self.test_transform,
        ) = self._init_transforms()

    def _init_transforms(self) -> Tuple[Callable, Callable, Callable]:
        """Initialize Image Augmentations"""
        spatial_size = (self.hparams.spatial_size, self.hparams.spatial_size)  # type: ignore  # pylint: disable=C0301
        train_transform = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=["image"], reader=TIFFImageReader),  # type: ignore  # pylint: disable=C0301
                monai.transforms.EnsureChannelFirstd(keys=["image"]),
                monai.transforms.ScaleIntensityd(keys=["image"]),
                monai.transforms.LoadImaged(keys=["mask"]),
                monai.transforms.AddChanneld(keys=["mask"]),
                monai.transforms.CropForegroundd(
                    keys=["image", "mask"],
                    source_key="mask",
                    select_fn=lambda x: x > 0.0,
                    margin=0,
                    mode="constant",
                ),
                monai.transforms.Resized(
                    keys=["image", "mask"], spatial_size=spatial_size, mode="nearest"
                ),
                monai.transforms.RandAxisFlipd(keys=["image", "mask"], prob=0.5),
                monai.transforms.RandRotate90d(keys=["image", "mask"], prob=0.5),
                monai.transforms.RandGridDistortiond(
                    keys=["image", "mask"], prob=0.5, distort_limit=0.2
                ),
                monai.transforms.OneOf(
                    [
                        monai.transforms.RandShiftIntensityd(
                            keys=["image"], prob=0.5, offsets=(0.1, 0.2)
                        ),
                        monai.transforms.RandAdjustContrastd(
                            keys=["image"], prob=0.5, gamma=(1.5, 2.5)
                        ),
                        monai.transforms.RandHistogramShiftd(keys=["image"], prob=0.5),
                    ]
                ),
            ]
        )

        val_transform = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=["image"], reader=TIFFImageReader),  # type: ignore  # pylint: disable=C0301
                monai.transforms.EnsureChannelFirstd(keys=["image"]),
                monai.transforms.ScaleIntensityd(keys=["image"]),
                monai.transforms.LoadImaged(keys=["mask"]),
                monai.transforms.AddChanneld(keys=["mask"]),
                monai.transforms.CropForegroundd(
                    keys=["image", "mask"],
                    source_key="mask",
                    select_fn=lambda x: x > 0.0,
                    margin=0,
                    mode="constant",
                ),
                monai.transforms.Resized(
                    keys=["image", "mask"], spatial_size=spatial_size, mode="nearest"
                ),
            ]
        )

        test_transform = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=["image"], reader=TIFFImageReader),  # type: ignore  # pylint: disable=C0301
                monai.transforms.EnsureChannelFirstd(keys=["image"]),
                monai.transforms.ScaleIntensityd(keys=["image"]),
                monai.transforms.Resized(
                    keys=["image"], spatial_size=spatial_size, mode="nearest"
                ),
            ]
        )

        return train_transform, val_transform, test_transform

    def setup(self, stage: str = None) -> None:
        """Pre-Processing"""
        if stage == "fit" or stage is None:
            train_df = self.train_df[
                self.train_df.fold != self.hparams.val_fold  # type: ignore
            ].reset_index(drop=True)
            val_df = self.train_df[
                self.train_df.fold == self.hparams.val_fold  # type: ignore
            ].reset_index(drop=True)

            self.train_dataset = self._dataset(train_df, transform=self.train_transform)
            self.val_dataset = self._dataset(val_df, transform=self.val_transform)

        if stage == "test" or stage is None:
            self.test_dataset = self._dataset(
                self.test_df, transform=self.test_transform
            )

    def _dataset(self, dataframe: pd.DataFrame, transform: Callable) -> CSVDataset:
        return CSVDataset(src=dataframe, transform=transform)

    def train_dataloader(self) -> DataLoader:
        """Initialize Train DataLoader"""
        return self._dataloader(self.train_dataset, train=True)

    def val_dataloader(self) -> DataLoader:
        """Initialize Validation DataLoader"""
        return self._dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        """Initialize Test DataLoader"""
        return self._dataloader(self.test_dataset)

    def _dataloader(self, dataset: CSVDataset, train: bool = False) -> DataLoader:
        """Returns Generic DataLoader"""
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,  # type: ignore
            shuffle=train,
            num_workers=self.hparams.num_workers,  # type: ignore
            pin_memory=True,
        )
