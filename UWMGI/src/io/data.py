"""Data Processing Functions"""
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


def get_dataframe(
    strategy: str = "StratifiedGroupKFold", num_splits: int = 5, seed: int = 42
) -> pd.DataFrame:
    """
    Pre-process dataset with the provided Cross Validation Strategy

    :param strategy: Which Cross Validation strategy to use,
                    defaults to "StratifiedGroupKFold"
    :type strategy: str, optional
    :param num_splits: Number of folds to split, defaults to 5
    :type num_splits: int, optional
    :param seed: Random Seed for the process, defaults to 42
    :type seed: int, optional
    :raises ValueError: raises an error if an unimplemented strategy is provided
    :return: DataFrame split by folds
    :rtype: pd.DataFrame
    """

    training_dataframe = pd.read_csv("data/mask/train.csv")
    training_dataframe["empty"] = training_dataframe.segmentation.map(
        lambda x: int(pd.isna(x))
    )
    training_dataframe["mask_path"] = training_dataframe.mask_path.str.replace(
        "/png/", "/np"
    ).str.replace(".png", ".npy")

    augmented_dataframe = (
        training_dataframe.groupby(["id"])["class"].agg(list).to_frame().reset_index()
    )
    augmented_dataframe = augmented_dataframe.merge(
        training_dataframe.groupby(["id"])["segmentation"].agg(list), on=["id"]
    )

    training_dataframe = training_dataframe.drop(columns=["segmentation", "class"])
    training_dataframe = (
        training_dataframe.groupby(["id"]).head(1).reset_index(drop=True)
    )
    training_dataframe = training_dataframe.merge(augmented_dataframe, on=["id"])

    # CV Split
    if strategy == "StratifiedGroupKFold":
        skf = StratifiedGroupKFold(n_splits=num_splits, shuffle=True, random_state=seed)
        for fold, (_, val_idx) in enumerate(
            skf.split(
                training_dataframe,
                training_dataframe["empty"],
                groups=training_dataframe["case"],
            )
        ):
            training_dataframe.loc[val_idx, "fold"] = fold

    elif strategy == "GroupKFold":
        group_fold = GroupKFold(n_splits=num_splits)
        for fold, (_, val_idx) in enumerate(
            group_fold.split(
                training_dataframe,
                training_dataframe["empty"],
                groups=training_dataframe["case"],
            )
        ):
            training_dataframe.loc[val_idx, "fold"] = fold

    else:
        raise ValueError(
            f"""{strategy}, is not a recognized strategy,
            please use StratifiedGroupKFold or GroupKFold"""
        )

    return training_dataframe
