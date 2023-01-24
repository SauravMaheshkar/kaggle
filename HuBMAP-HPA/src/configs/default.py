"""Default Configuration"""
import ml_collections

__all__ = ["get_config"]


def get_config() -> ml_collections.ConfigDict:
    """Get the default configuration dictionary"""
    config = ml_collections.ConfigDict()

    config.num_splits = 4
    config.random_seed = 42

    config.train_csv_path = "data/train_prepared.csv"
    config.val_pred_prepared_csv_path = "data/val_pred_prepared.csv"
    config.test_csv_path = "data/test_prepared.csv"

    config.spatial_size = 1024
    config.val_fold = 0
    config.batch_size = 16
    config.num_workers = 8

    config.learning_rate = 2e-3
    config.weight_decay = 0.0

    config.fast_dev_run = False
    config.gpus = 1
    config.max_epochs = 10
    config.precision = 16

    return config
