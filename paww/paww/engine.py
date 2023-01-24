# Mostly Ported from https://www.kaggle.com/yasufuminakama 's work
import time
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import wandb

from .utils import AverageMeter, set_seed, timeSince

__all__ = ["Trainer", "Evaluator", "RMSELoss"]


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y) -> torch.Tensor:
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

    def train(
        self,
        fold: int,
        config: dict,
        train_loader,
        epoch: int,
        device: Union[str, torch.device],
    ):

        # Time / Step variables
        global_step = 0
        start = end = time.time()

        # Initialize AverageMeter
        losses = AverageMeter()

        # Basic Model Initialization
        self.model.zero_grad()
        self.model.train()

        # Set Random Seed
        set_seed(config["seed"])

        # Begin Train Loop
        for step, (images, labels) in enumerate(train_loader):

            # Move Items to Device
            images = images.to(device)
            labels = labels.to(device)

            # Batch Size
            batch_size = labels.size(0)

            # Forward Pass & Calculate Loss
            y_preds = self.model(images)
            loss = self.criterion(y_preds.view(-1), labels)
            losses.update(loss.item(), batch_size)

            # Perform Gradient Accumulation
            if config["gradient_accumulation_steps"] > 1:
                loss = loss / config["gradient_accumulation_steps"]

            # Backward Pass
            loss.backward()

            # Gradient Normalization
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), config["max_grad_norm"]
            )

            # Update Optimizer
            if (step + 1) % config["gradient_accumulation_steps"] == 0:

                self.optimizer.step()
                self.optimizer.zero_grad()
                global_step += 1

            end = time.time()  # noqa: F841

            # Logging Information
            if step % config["print_freq"] == 0 or step == (len(train_loader) - 1):
                print(
                    "Epoch: [{0}][{1}/{2}] "
                    "Elapsed {remain:s} "
                    "Loss: {loss.val:.4f}({loss.avg:.4f}) "
                    "Grad: {grad_norm:.4f}  "
                    "LR: {lr:.6f}  ".format(
                        epoch + 1,
                        step,
                        len(train_loader),
                        remain=timeSince(start, float(step + 1) / len(train_loader)),
                        loss=losses,
                        grad_norm=grad_norm,
                        lr=self.scheduler.get_last_lr()[0],
                    )
                )

            # Upload Stats to W&B 🔥
            wandb.log(
                {
                    f"[fold{fold}] loss": losses.val,
                    f"[fold{fold}] lr": self.scheduler.get_last_lr()[0],
                }
            )
        return losses.avg


class Evaluator:
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion

    def evaluate(
        self, config: dict, valid_dataloader, device: Union[str, torch.device]
    ):

        # Set model to Evaluation Mode
        self.model.eval()

        # Time / Step / logging variables
        losses = AverageMeter()
        preds = []
        start = end = time.time()

        # Begin Evaluation Loop
        for step, (images, labels) in enumerate(valid_dataloader):

            # Move Items to Device
            images = images.to(device)
            labels = labels.to(device)

            # Batch Size
            batch_size = labels.size(0)

            # Calculate and Update Loss
            with torch.no_grad():
                y_preds = self.model(images)
            loss = self.criterion(y_preds.view(-1), labels)
            losses.update(loss.item(), batch_size)

            # Calculate and Update Accuracy
            preds.append(y_preds.to("cpu").numpy())

            # Perform Gradient Accumulation
            if config["gradient_accumulation_steps"] > 1:
                loss = loss / config["gradient_accumulation_steps"]

            end = time.time()  # noqa: F841

            # Logging Information
            if step % config["print_freq"] == 0 or step == (len(valid_dataloader) - 1):
                print(
                    "EVAL: [{0}/{1}] "
                    "Elapsed {remain:s} "
                    "Loss: {loss.val:.4f}({loss.avg:.4f}) ".format(
                        step,
                        len(valid_dataloader),
                        loss=losses,
                        remain=timeSince(
                            start, float(step + 1) / len(valid_dataloader)
                        ),
                    )
                )
        predictions = np.concatenate(preds)
        return losses.avg, predictions
