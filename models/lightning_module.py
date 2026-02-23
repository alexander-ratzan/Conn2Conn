"""
PyTorch Lightning module for cross-modal Conn2Conn models.
Creates loss via create_loss_fn and owns training/validation steps and optimizer.
"""
import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch as pl

from models.loss import (
    get_target_train_mean,
    create_loss_fn,
    compute_pearson_r,
    compute_demeaned_pearson_r,
)

class CrossModalLightningModule(pl.LightningModule):
    """
    Lightning module that wraps any cross-modal nn.Module.
    Creates loss and optimizer; implements training_step, validation_step, configure_optimizers.
    """
    def __init__(
        self,
        model: nn.Module,
        base,
        lr: float = 1e-4,
        loss_type: str = "mse",
        loss_alpha: float = 0.5,
        loss_beta: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "base"])
        self.model = model
        self.base = base
        self.lr = lr
        self.loss_type = loss_type
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta
        self._target_train_mean = None
        self.loss_fn = None

    def setup(self, stage=None):
        self._target_train_mean = get_target_train_mean(self.base)
        self.loss_fn = create_loss_fn(
            self.loss_type,
            base=self.base,
            alpha=self.loss_alpha,
            beta=self.loss_beta,
        )
        if self.loss_fn is not None:
            self.loss_fn = self.loss_fn.to(self.device)

    def forward(self, x):
        return self.model(x)

    def _unpack_out(self, out):
        if isinstance(out, tuple) and len(out) == 3:
            return out[0], out[1], out[2]
        return out, None, None

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        out = self.model(x)
        y_pred, mu, logvar = self._unpack_out(out)
        loss = self.loss_fn(y_pred, y, mu=mu, logvar=logvar)
        if hasattr(self.model, "get_reg_loss"):
            loss = loss + self.model.get_reg_loss()
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        if self.loss_fn is not None:
            self.loss_fn = self.loss_fn.to(x.device)
        out = self.model(x)
        y_pred, mu, logvar = self._unpack_out(out)
        loss = self.loss_fn(y_pred, y, mu=mu, logvar=logvar)
        target_mean = self._target_train_mean
        if isinstance(target_mean, np.ndarray):
            target_mean = torch.tensor(
                target_mean, dtype=torch.float32, device=y.device
            )
        pr = compute_pearson_r(y_pred, y)
        dr = compute_demeaned_pearson_r(y_pred, y, target_mean)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_pearson_r", pr, on_step=False, on_epoch=True)
        self.log("val_demeaned_r", dr, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
