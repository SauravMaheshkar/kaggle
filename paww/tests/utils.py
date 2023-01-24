import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from paww.augmentation import get_transforms
from paww.dataloader import PawwDataset


def get_dataloader(cfg):

    train = pd.read_csv("data/petfinder-pawpularity-score/train.csv")

    def get_train_file_path(image_id):
        return "data/petfinder-pawpularity-score/train/{}.jpg".format(image_id)

    train["file_path"] = train["Id"].apply(get_train_file_path)

    num_bins = int(np.floor(1 + np.log2(len(train))))
    train["bins"] = pd.cut(train[cfg["target_col"]], bins=num_bins, labels=False)

    Fold = StratifiedKFold(
        n_splits=cfg["folds"], shuffle=True, random_state=cfg["seed"]
    )
    for n, (train_index, val_index) in enumerate(Fold.split(train, train["bins"])):
        train.loc[val_index, "fold"] = int(n)

    train["fold"] = train["fold"].astype(int)
    trn_idx = train[train["fold"] != 0].index
    train_folds = train.loc[trn_idx].reset_index(drop=True)

    train_dataset = PawwDataset(
        cfg, train_folds, transform=get_transforms(cfg, data="train")
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    return dataloader
