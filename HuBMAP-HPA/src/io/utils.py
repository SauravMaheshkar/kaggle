"""Custom Data Processing Utilities"""
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from rich.progress import track
from sklearn.model_selection import StratifiedKFold


def add_path_to_df(
    dataframe: pd.DataFrame, data_dir: Path, type_: str, stage: str
) -> pd.DataFrame:
    """
    Appends Path to the DataFrame

    :param dataframe: DataFrame
    :type dataframe: pd.DataFrame
    :param data_dir: Path to the data directory
    :type data_dir: Path
    :param stage: Stage
    :type stage: str
    :return: DataFrame with added paths
    :rtype: pd.DataFrame
    """
    ending = ".tiff" if type_ == "image" else ".npy"

    dir_ = (
        str(data_dir + f"/{stage}_{type_}s")  # type: ignore
        if type_ == "image"
        else (data_dir + f"/{stage}_{type_}s")  # type: ignore
    )
    dataframe[type_] = dir_ + "/" + dataframe["id"].astype(str) + ending
    return dataframe


def add_paths_to_df(
    dataframe: pd.DataFrame, data_dir: Path, stage: str
) -> pd.DataFrame:
    """
    Appends Paths to the DataFrame

    :param dataframe: DataFrame
    :type dataframe: pd.DataFrame
    :param data_dir: Path to the data directory
    :type data_dir: Path
    :param stage: Stage
    :type stage: str
    :return: DataFrame with added paths
    :rtype: pd.DataFrame
    """
    dataframe = add_path_to_df(dataframe, data_dir, "image", stage)
    dataframe = add_path_to_df(dataframe, data_dir, "mask", stage)
    return dataframe


def create_folds(
    dataframe: pd.DataFrame, n_splits: int, random_seed: int
) -> pd.DataFrame:
    """
    Splits the given DataFrame into Folds

    :param dataframe: DataFrame
    :type dataframe: pd.DataFrame
    :param n_splits: number of splits
    :type n_splits: int
    :param random_seed: Random Seed
    :type random_seed: int
    :return: DataFrame with added paths
    :rtype: pd.DataFrame
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    for fold, (_, val_idx) in enumerate(skf.split(X=dataframe, y=dataframe["organ"])):
        dataframe.loc[val_idx, "fold"] = fold

    return dataframe


def prepare_data(
    data_dir: Path, stage: str, n_splits: int, random_seed: int
) -> pd.DataFrame:
    """prepare_data Pre-Process data

    :param data_dir: Path to data directory
    :type data_dir: Path
    :param stage: Stage
    :type stage: str
    :param n_splits: number of splits
    :type n_splits: int
    :param random_seed: Random Seed
    :type random_seed: int
    :return: Processed DataFrame
    :rtype: pd.DataFrame
    """
    dataframe = pd.read_csv(data_dir + f"/{stage}.csv")  # type: ignore
    dataframe = add_paths_to_df(dataframe, data_dir, stage)

    if stage == "train":
        dataframe = create_folds(dataframe, n_splits, random_seed)

    filename = f"data/{stage}_prepared.csv"
    dataframe.to_csv(filename, index=False)

    print(f"Created {filename} with shape {dataframe.shape}")

    return dataframe


def rle2mask(mask_rle: str, shape: Tuple[int, int]) -> np.ndarray:
    """
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background
    Source: https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
    """
    split_masks = mask_rle.split()
    starts, lengths = [
        np.asarray(x, dtype=int) for x in (split_masks[0:][::2], split_masks[1:][::2])
    ]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for low, high in zip(starts, ends):
        img[low:high] = 1
    return img.reshape(shape).T


def save_array(file_path: str, array: np.ndarray) -> None:
    """
    Serializes array

    :param file_path: Path to save array
    :type file_path: str
    :param array: Array in question
    :type array: np.ndarray
    """
    file_path = Path(file_path)  # type: ignore
    file_path.parent.mkdir(parents=True, exist_ok=True)  # type: ignore
    np.save(file_path, array)


def save_masks(dataframe: pd.DataFrame) -> None:
    """
    Serializes masks

    :param dataframe: DataFrame with mask information
    :type dataframe: pd.DataFrame
    """
    for row in track(dataframe.itertuples(), total=len(dataframe)):
        mask = rle2mask(row.rle, shape=(row.img_width, row.img_height))
        save_array(row.mask, mask)
