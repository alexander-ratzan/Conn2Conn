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
from models.models import get_model_input

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
        loss_corr_target: float = 0.4,
        loss_corr_weight: float = 1e-3,
    ):
        super().__init__()
        # Persist only loss hyperparameters relevant to the selected loss_type.
        # This avoids logging unrelated defaults for other model/loss families.
        hparams_to_save = {
            "lr": lr,
            "loss_type": loss_type,
        }
        if loss_type == "weighted_mse":
            hparams_to_save["loss_alpha"] = loss_alpha
        elif loss_type == "vae":
            hparams_to_save["loss_beta"] = loss_beta
        elif loss_type == "sarwar_mse_corr":
            hparams_to_save["loss_corr_target"] = loss_corr_target
            hparams_to_save["loss_corr_weight"] = loss_corr_weight
        self.save_hyperparameters(hparams_to_save)
        self.model = model
        self.base = base
        self.lr = lr
        self.loss_type = loss_type
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta
        self.loss_corr_target = loss_corr_target
        self.loss_corr_weight = loss_corr_weight
        self._target_train_mean = None
        self.loss_fn = None

    def setup(self, stage=None):
        self._target_train_mean = get_target_train_mean(self.base)
        self.loss_fn = create_loss_fn(
            self.loss_type,
            base=self.base,
            alpha=self.loss_alpha,
            beta=self.loss_beta,
            corr_target=self.loss_corr_target,
            corr_weight=self.loss_corr_weight,
        )
        if self.loss_fn is not None:
            self.loss_fn = self.loss_fn.to(self.device)

    def forward(self, x):
        return self.model(x)

    def _forward_model(self, batch):
        x = get_model_input(batch)
        if getattr(self.model, "uses_cov", False) and "cov" in batch:
            kwargs = {"cov": batch["cov"]}
            if getattr(self.model, "use_target_scores_in_projector", False) and "y" in batch:
                kwargs["y"] = batch["y"]
            return self.model(x, **kwargs)
        return self.model(x)

    def _unpack_out(self, out):
        if isinstance(out, tuple) and len(out) == 3:
            return out[0], out[1], out[2]
        return out, None, None

    def training_step(self, batch, batch_idx):
        y = batch["y"]
        out = self._forward_model(batch)
        y_pred, mu, logvar = self._unpack_out(out)
        loss = self.loss_fn(y_pred, y, mu=mu, logvar=logvar)
        if hasattr(self.model, "get_reg_loss"):
            loss = loss + self.model.get_reg_loss()
        
        target_mean = self._target_train_mean
        if isinstance(target_mean, np.ndarray):
            target_mean = torch.tensor(
                target_mean, dtype=torch.float32, device=y_pred.device
            )
        
        pr = compute_pearson_r(y_pred, y)
        dr = compute_demeaned_pearson_r(y_pred, y, target_mean)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_pearson_r", pr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_demeaned_r", dr, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch["y"]
        if self.loss_fn is not None:
            self.loss_fn = self.loss_fn.to(y.device)
        out = self._forward_model(batch)
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
