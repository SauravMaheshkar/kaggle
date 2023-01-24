"""Training Code"""
import torch
from rich import print

import wandb
from src.io.data import get_dataframe
from src.io.dataset import prepare_loaders
from src.nn.engine import run_training
from src.nn.loss import criterion
from src.nn.model import build_model
from src.nn.optimizers import fetch_optimizer
from src.nn.schedulers import fetch_scheduler
from src.utils import set_seed


class CFG:
    """Configuration"""

    seed = 42
    arch = "Unet"
    backbone = "efficientnet-b1"
    strategy = "StratifiedGroupKFold"
    num_splits = 5
    train_batch_size = 128
    valid_batch_size = train_batch_size * 2
    img_size = [224, 224]
    epochs = 20
    lr = 1e-3
    optimizer = "AdamW"
    scheduler = "CosineAnnealingWarmRestarts"
    min_lr = 1e-6
    T_max = int(30000 / train_batch_size * epochs) + 50
    T_0 = 25
    weight_decay = 1e-6
    accumulation_steps = max(1, 32 // train_batch_size)
    n_fold = 5
    lovasz_weight = 0.0
    dice_weight = 0.6
    jaccard_weight = 0.0
    bce_weight = 0.4
    tversky_weight = 0.0


if __name__ == "__main__":
    # Miscellaneous
    set_seed(seed=CFG.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize a Weights & Biases Run
    wandb.init(
        project="uw-maddison-gi-tract",
        config={k: v for k, v in dict(vars(CFG)).items() if "__" not in k},
    )

    # Get Dataframe
    df = get_dataframe(strategy=CFG.strategy, num_splits=CFG.num_splits, seed=CFG.seed)

    # Training Pipeline
    for fold in range(CFG.n_fold):
        print("#" * 15)
        print(f"### Fold: {fold}")
        print("#" * 15)

        # Get Dataloaders
        train_loader, valid_loader = prepare_loaders(
            fold=fold,
            dataframe=df,
            train_batch_size=CFG.train_batch_size,
            valid_batch_size=CFG.valid_batch_size,
            image_size=CFG.img_size,
        )

        # Build the Model
        model = build_model(device=device, arch=CFG.arch, backbone=CFG.backbone)

        # Get Number of Parameters
        trainable_model_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(f"Model Parameters:{trainable_model_params}")
        wandb.run.summary["Model Parameters"] = trainable_model_params

        # Get Optimizer and Scheduler
        optimizer = fetch_optimizer(
            optimizer=CFG.optimizer,
            params=model.parameters(),
            learning_rate=CFG.lr,
            weight_decay=CFG.weight_decay,
        )
        scheduler = fetch_scheduler(
            scheduler=CFG.scheduler,
            optimizer=optimizer,
            min_lr=CFG.min_lr,
            t_max=CFG.T_max,
            t_0=CFG.T_0,
        )

        # Training !!!
        model, history = run_training(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            num_epochs=CFG.epochs,
            accumulation_steps=CFG.accumulation_steps,
            fold=fold,
        )

    wandb.finish()
