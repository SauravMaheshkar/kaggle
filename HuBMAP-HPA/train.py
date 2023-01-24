"""Training Code"""
from pathlib import Path

import pytorch_lightning as pl

from src.configs.default import get_config
from src.io.data import LitDataModule
from src.io.utils import prepare_data, save_masks
from src.nn.model import LitModule

DEBUG = False
COMPETITION_DATA_DIR = Path("data")

if __name__ == "__main__":

    config = get_config()

    train_df = prepare_data(
        COMPETITION_DATA_DIR, "train", config.num_splits, config.random_seed
    )
    test_df = prepare_data(
        COMPETITION_DATA_DIR, "test", config.num_splits, config.random_seed
    )

    save_masks(train_df)

    pl.seed_everything(config.random_seed)
    data_module = LitDataModule(
        train_csv_path=config.train_csv_path,
        test_csv_path=config.test_csv_path,
        spatial_size=config.spatial_size,
        val_fold=config.val_fold,
        batch_size=2 if DEBUG else config.batch_size,
        num_workers=config.num_workers,
    )

    module = LitModule(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    trainer = pl.Trainer(
        fast_dev_run=config.fast_dev_run,
        gpus=config.gpus,
        limit_train_batches=0.1 if DEBUG else 1.0,
        limit_val_batches=0.1 if DEBUG else 1.0,
        log_every_n_steps=5,
        logger=pl.loggers.CSVLogger(save_dir="logs/"),
        max_epochs=2 if DEBUG else config.max_epochs,
        precision=config.precision,
    )

    trainer.fit(module, datamodule=data_module)
