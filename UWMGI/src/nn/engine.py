"""Training Pipeline"""
import copy
import gc
from collections import defaultdict

import numpy as np
import torch
from rich import print
from rich.progress import track
from torch import nn
from torch.cuda import amp

import wandb
from src.nn.loss import dice_coef, iou_coef


# pylint: disable=R0914
def train_one_epoch(
    model,
    optimizer,
    scheduler,
    criterion,
    dataloader,
    device,
    accumulation_steps,
    epoch,
):
    """
    Trains the model for one epoch

    :param model: Model
    :type model: torch.nn.Module
    :param optimizer: Optimizer
    :type optimizer: torch.optim.Optimizer
    :param scheduler: Scheduler
    :param criterion: Loss Function
    :param dataloader: Dataloader for training
    :param device: PyTorch device
    :param accumulation_steps: Gradient Accumulation Steps
    :type accumulation_steps: int
    :param epoch: Which epoch is being trained
    :type epoch: int
    """
    model.train()
    scaler = amp.GradScaler()

    dataset_size = 0
    running_loss = 0.0

    for step, (images, masks) in track(
        enumerate(dataloader),
        total=len(dataloader),
        description=f"Training | Epoch: {epoch}",
    ):
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        batch_size = images.size(0)

        with amp.autocast(enabled=True):
            y_pred = model(images)
            loss = criterion(y_pred, masks)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

    torch.cuda.empty_cache()
    _ = gc.collect()

    return epoch_loss


# pylint: disable=R0914
@torch.no_grad()
def valid_one_epoch(model, dataloader, criterion, device, epoch):
    """
    Performs Validation for one epoch

    :param model: Model
    :type model: torch.nn.Module
    :param dataloader: Dataloader
    :param criterion: Loss Function
    :param device: PyTorch Device
    :param epoch: Epoch under consideration
    :type epoch: int
    """
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    val_scores = []

    for _, (images, masks) in track(
        enumerate(dataloader),
        total=len(dataloader),
        description=f"Validation | Epoch: {epoch}",
    ):
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        batch_size = images.size(0)

        y_pred = model(images)
        loss = criterion(y_pred, masks)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        y_pred = nn.Sigmoid()(y_pred)
        val_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
        val_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()
        val_scores.append([val_dice, val_jaccard])

    val_scores = np.mean(val_scores, axis=0)
    torch.cuda.empty_cache()
    _ = gc.collect()

    return epoch_loss, val_scores


# pylint: disable=R0914
def run_training(
    model,
    train_loader,
    valid_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    num_epochs,
    accumulation_steps,
    fold,
):
    """
    Run Training Pipeline

    :param model: Model
    :type model: torch.nn.Module
    :param train_loader: Training Dataloader
    :param valid_loader: Validation Dataloader
    :param optimizer: Optimizer
    :param scheduler: Scheduler
    :param criterion: Loss Function
    :param device: PyTorch Device
    :param num_epochs: Number of epochs to train
    :type num_epochs: int
    :param accumulation_steps: Number of accumulation steps
    :type accumulation_steps: int
    :param fold: fold being trained
    :type fold: int
    """
    if torch.cuda.is_available():
        print(f"cuda: {torch.cuda.get_device_name()}\n")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_dice = -np.inf
    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        _ = gc.collect()
        print(f"Epoch {epoch}/{num_epochs}", end="")
        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            dataloader=train_loader,
            device=device,
            accumulation_steps=accumulation_steps,
            epoch=epoch,
        )

        val_loss, val_scores = valid_one_epoch(
            model=model,
            dataloader=valid_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )
        val_dice, val_jaccard = val_scores

        history[f"Fold-{fold}/Train Loss"].append(train_loss)
        history[f"Fold-{fold}/Valid Loss"].append(val_loss)
        history[f"Fold-{fold}/Valid Dice"].append(val_dice)
        history[f"Fold-{fold}/Valid Jaccard"].append(val_jaccard)

        # Log the metrics
        wandb.run.log(
            {
                f"Fold-{fold}/Train Loss": train_loss,
                f"Fold-{fold}/Valid Loss": val_loss,
                f"Fold-{fold}/Valid Dice": val_dice,
                f"Fold-{fold}/Valid Jaccard": val_jaccard,
                f"Fold-{fold}/LR": scheduler.get_last_lr()[0],
            }
        )

        print(f"Valid Dice: {val_dice:0.4f} | Valid Jaccard: {val_jaccard:0.4f}")

        # deep copy the model
        if val_dice >= best_dice:
            print(f"Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
            best_dice = val_dice
            best_jaccard = val_jaccard
            wandb.run.summary[f"Fold-{fold}/Best Dice"] = best_dice
            wandb.run.summary[f"Fold-{fold}/Best Jaccard"] = best_jaccard
            best_model_wts = copy.deepcopy(model.state_dict())
            # Save Model
            path = f"models/best_epoch-{fold:02d}.bin"
            torch.save(model.state_dict(), path)
            print(f"Model Saved to :{path}")

        path = f"models/last_epoch-{fold:02d}.bin"
        torch.save(model.state_dict(), path)

    print(f"Best Score: {best_jaccard}")

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history
