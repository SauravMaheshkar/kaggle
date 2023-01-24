"""Dataset Definitions"""
from typing import List

import numpy as np
import pandas as pd
import torch

from src.io.augmentations import get_data_transforms
from src.io.img_utils import load_img, load_msk


class UWMGIDataset(torch.utils.data.Dataset):
    """
    Dataset for handling competition data
    """

    def __init__(self, dataframe: pd.DataFrame, label: bool = True, transforms=None):
        self.dataframe = dataframe
        self.label = label
        self.img_paths = dataframe["image_path"].tolist()
        self.msk_paths = dataframe["mask_path"].tolist()
        self.transforms = transforms

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = []
        img = load_img(img_path)

        if self.label:
            msk_path = self.msk_paths[index]
            msk = load_msk(msk_path)
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img = data["image"]
                msk = data["mask"]
            img = np.transpose(img, (2, 0, 1))
            msk = np.transpose(msk, (2, 0, 1))
            return torch.tensor(img), torch.tensor(msk)
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img = data["image"]
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img)


def prepare_loaders(
    fold: int,
    dataframe: pd.DataFrame,
    train_batch_size: int = 128,
    valid_batch_size: int = 128 * 2,
    image_size: List[int] = [224, 224],
):
    """
    Returns dataloaders for training and validation

    :param fold: fold corresponding to which dataloaders are to be generated
    :type fold: int
    :param dataframe: Dataframe with fold information
    :type dataframe: pd.DataFrame
    :param train_batch_size: Batch Size for the training dataloader, defaults to 128
    :type train_batch_size: int, optional
    :param valid_batch_size: Batch Size for the validation dataloader, defaults to 128*2
    :type valid_batch_size: int, optional
    :param image_size: Image Size to be used, defaults to [224, 224]
    :type image_size: List[int], optional
    :return: Training and Validation Dataloaders
    """
    transforms = get_data_transforms(image_size=image_size)
    train_df = dataframe.query(f"fold!={fold}").reset_index(drop=True)
    valid_df = dataframe.query(f"fold=={fold}").reset_index(drop=True)
    train_dataset = UWMGIDataset(train_df, transforms=transforms["train"])
    valid_dataset = UWMGIDataset(valid_df, transforms=transforms["valid"])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, valid_loader
