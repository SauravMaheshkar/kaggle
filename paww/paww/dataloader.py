from typing import Any, Sequence

import cv2
import pandas as pd
import torch

__all__ = ["PawwDataset"]


class PawwDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: dict, df: pd.DataFrame, transform: Any = None) -> None:
        self.df = df
        self.file_names = df["file_path"].values
        self.labels = df[cfg["target_col"]].values
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> Sequence[torch.Tensor]:
        file_path = self.file_names[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        label = torch.tensor(self.labels[idx]).float()
        return image, label
