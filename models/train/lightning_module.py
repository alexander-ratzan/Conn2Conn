"""Lightning training wrapper.

Defines batch-level train/validation behavior, optimizer setup, and metric logging.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from models.train.loss import (
    get_target_train_mean,
    create_loss_fn,
    CompositeLoss,
    compute_latent_reconstruction_loss,
    compute_var_match_loss,
)
from models.eval.metrics import (
    compute_pearson_r,
    compute_demeaned_pearson_r,
    compute_prediction_variance_ratio,
    compute_prediction_norm_ratio,
)
from models.architectures.utils import get_model_input

LATENT_LOSS_TYPES = {"latent_mse", "latent_weighted_mse"}
COMPOSITE_LOSS_TYPES = {"composite"}


def _display_loss_terms(loss_terms):
    if loss_terms is None:
        return []
    if isinstance(loss_terms, str):
        return [loss_terms]
    return list(loss_terms)

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
        loss_terms=None,
        loss_normalize: str = "ema",
        loss_scale_ema_decay: float = 0.95,
        loss_scale_warmup_steps: int = 20,
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
        elif loss_type in COMPOSITE_LOSS_TYPES:
            hparams_to_save["loss_terms"] = _display_loss_terms(loss_terms)
            hparams_to_save["loss_normalize"] = loss_normalize
            hparams_to_save["loss_scale_ema_decay"] = loss_scale_ema_decay
            hparams_to_save["loss_scale_warmup_steps"] = loss_scale_warmup_steps
        self.save_hyperparameters(hparams_to_save)
        self.model = model
        self.base = base
        self.lr = lr
        self.loss_type = loss_type
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta
        self.loss_corr_target = loss_corr_target
        self.loss_corr_weight = loss_corr_weight
        self.loss_terms = loss_terms
        self.loss_normalize = str(loss_normalize or "ema")
        self.loss_scale_ema_decay = float(loss_scale_ema_decay)
        self.loss_scale_warmup_steps = int(loss_scale_warmup_steps)
        self._target_train_mean = None
        self.loss_fn = None

    def setup(self, stage=None):
        self._target_train_mean = get_target_train_mean(self.base)
        if self.loss_type in LATENT_LOSS_TYPES:
            self.loss_fn = None
        else:
            self.loss_fn = create_loss_fn(
                self.loss_type,
                base=self.base,
                alpha=self.loss_alpha,
                beta=self.loss_beta,
                corr_target=self.loss_corr_target,
                corr_weight=self.loss_corr_weight,
                loss_terms=self.loss_terms,
                loss_normalize=self.loss_normalize,
                loss_scale_ema_decay=self.loss_scale_ema_decay,
                loss_scale_warmup_steps=self.loss_scale_warmup_steps,
            )
            if self.loss_fn is not None:
                self.loss_fn = self.loss_fn.to(self.device)

    def forward(self, x):
        return self.model(x)

    def _forward_model(self, batch):
        x = get_model_input(batch)
        kwargs = {}
        if getattr(self.model, "uses_cov", False) and "cov" in batch:
            kwargs["cov"] = batch["cov"]
            if getattr(self.model, "use_target_scores_in_projector", False) and "y" in batch:
                kwargs["y"] = batch["y"]
        if getattr(self.model, "uses_node_features", False) and "node_features" in batch:
            kwargs["node_features"] = batch["node_features"]
        if kwargs:
            return self.model(x, **kwargs)
        return self.model(x)

    def _unpack_out(self, out):
        if isinstance(out, tuple) and len(out) == 3:
            return out[0], out[1], out[2]
        return out, None, None

    def _compute_latent_loss(self, batch, y_pred):
        if hasattr(self.model, "compute_latent_loss"):
            return self.model.compute_latent_loss(batch, self.loss_type)
        if not hasattr(self.model, "predict_target_latents") or not hasattr(self.model, "encode_target_latents"):
            raise ValueError(
                f"loss_type='{self.loss_type}' requires a model with "
                "predict_target_latents(...) and encode_target_latents(...)."
            )
        x = get_model_input(batch)
        y = batch["y"]
        kwargs = {}
        if getattr(self.model, "uses_cov", False) and "cov" in batch:
            kwargs["cov"] = batch["cov"]
            if getattr(self.model, "use_target_scores_in_projector", False) and "y" in batch:
                kwargs["y"] = batch["y"]
        if getattr(self.model, "uses_node_features", False) and "node_features" in batch:
            kwargs["node_features"] = batch["node_features"]
        c_hat = self.model.predict_target_latents(x, **kwargs) if kwargs else self.model.predict_target_latents(x)
        c_true = self.model.encode_target_latents(y)
        weights = getattr(self.model, "latent_loss_weights", None)
        return compute_latent_reconstruction_loss(c_hat, c_true, self.loss_type, weights=weights)

    def _compute_latent_loss_for_type(self, batch, latent_loss_type):
        if hasattr(self.model, "compute_latent_loss"):
            return self.model.compute_latent_loss(batch, latent_loss_type)
        if not hasattr(self.model, "predict_target_latents") or not hasattr(self.model, "encode_target_latents"):
            return None
        x = get_model_input(batch)
        y = batch["y"]
        kwargs = {}
        if getattr(self.model, "uses_cov", False) and "cov" in batch:
            kwargs["cov"] = batch["cov"]
            if getattr(self.model, "use_target_scores_in_projector", False) and "y" in batch:
                kwargs["y"] = batch["y"]
        if getattr(self.model, "uses_node_features", False) and "node_features" in batch:
            kwargs["node_features"] = batch["node_features"]
        c_hat = self.model.predict_target_latents(x, **kwargs) if kwargs else self.model.predict_target_latents(x)
        c_true = self.model.encode_target_latents(y)
        weights = getattr(self.model, "latent_loss_weights", None)
        return compute_latent_reconstruction_loss(c_hat, c_true, latent_loss_type, weights=weights)

    def _compute_aux_losses(self, batch, y_pred, y_true):
        with torch.no_grad():
            aux = {
                "edge_mse": F.mse_loss(y_pred, y_true).detach(),
                "var_match_loss": compute_var_match_loss(y_pred, y_true).detach(),
            }
            latent_mse = self._compute_latent_loss_for_type(batch, "latent_mse")
            if latent_mse is not None:
                aux["latent_mse"] = latent_mse.detach()
            latent_weighted_mse = self._compute_latent_loss_for_type(batch, "latent_weighted_mse")
            if latent_weighted_mse is not None:
                aux["latent_weighted_mse"] = latent_weighted_mse.detach()
        return aux

    def _log_structured_loss_terms(self, phase):
        if not isinstance(self.loss_fn, CompositeLoss):
            return
        for name, value in self.loss_fn.last_raw_terms.items():
            self.log(f"{phase}_loss_raw_{name}", value, on_step=False, on_epoch=True)
        for name, value in self.loss_fn.last_norm_terms.items():
            self.log(f"{phase}_loss_term_{name}", value, on_step=False, on_epoch=True)
        for name, value in self.loss_fn.last_weighted_terms.items():
            self.log(f"{phase}_loss_weighted_{name}", value, on_step=False, on_epoch=True)
        for name, value in self.loss_fn.get_scale_dict().items():
            self.log(f"{phase}_loss_ref_{name}", value, on_step=False, on_epoch=True)

    def _compute_reg_loss(self, device):
        if not hasattr(self.model, "get_reg_loss"):
            return torch.tensor(0.0, device=device)
        reg_loss = self.model.get_reg_loss()
        if reg_loss is None:
            return torch.tensor(0.0, device=device)
        if not torch.is_tensor(reg_loss):
            reg_loss = torch.tensor(float(reg_loss), device=device, dtype=torch.float32)
        else:
            reg_loss = reg_loss.to(device=device)
        return reg_loss

    def training_step(self, batch, batch_idx):
        y = batch["y"]
        out = self._forward_model(batch)
        y_pred, mu, logvar = self._unpack_out(out)
        if self.loss_type in LATENT_LOSS_TYPES:
            loss = self._compute_latent_loss(batch, y_pred)
        else:
            loss = self.loss_fn(y_pred, y, mu=mu, logvar=logvar)
        reg_loss = self._compute_reg_loss(y_pred.device)
        loss = loss + reg_loss
        
        target_mean = self._target_train_mean
        if isinstance(target_mean, np.ndarray):
            target_mean = torch.tensor(
                target_mean, dtype=torch.float32, device=y_pred.device
            )
        
        pr = compute_pearson_r(y_pred, y)
        dr = compute_demeaned_pearson_r(y_pred, y, target_mean)
        vr = compute_prediction_variance_ratio(y_pred, y)
        nr = compute_prediction_norm_ratio(y_pred, y)
        aux_losses = self._compute_aux_losses(batch, y_pred, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_pearson_r", pr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_demeaned_r", dr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_variance_ratio", vr, on_step=False, on_epoch=True)
        self.log("train_norm_ratio", nr, on_step=False, on_epoch=True)
        self.log("train_reg_loss", reg_loss.detach(), on_step=False, on_epoch=True)
        self.log("train_edge_mse", aux_losses["edge_mse"], on_step=False, on_epoch=True)
        self.log("train_var_match_loss", aux_losses["var_match_loss"], on_step=False, on_epoch=True)
        self._log_structured_loss_terms("train")
        if "latent_mse" in aux_losses:
            self.log("train_latent_mse", aux_losses["latent_mse"], on_step=False, on_epoch=True)
        if "latent_weighted_mse" in aux_losses:
            self.log("train_latent_weighted_mse", aux_losses["latent_weighted_mse"], on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y = batch["y"]
        if self.loss_fn is not None:
            self.loss_fn = self.loss_fn.to(y.device)
        out = self._forward_model(batch)
        y_pred, mu, logvar = self._unpack_out(out)
        if self.loss_type in LATENT_LOSS_TYPES:
            loss = self._compute_latent_loss(batch, y_pred)
        else:
            loss = self.loss_fn(y_pred, y, mu=mu, logvar=logvar)
        reg_loss = self._compute_reg_loss(y_pred.device)
        loss = loss + reg_loss
        target_mean = self._target_train_mean
        if isinstance(target_mean, np.ndarray):
            target_mean = torch.tensor(
                target_mean, dtype=torch.float32, device=y.device
            )
        pr = compute_pearson_r(y_pred, y)
        dr = compute_demeaned_pearson_r(y_pred, y, target_mean)
        vr = compute_prediction_variance_ratio(y_pred, y)
        nr = compute_prediction_norm_ratio(y_pred, y)
        aux_losses = self._compute_aux_losses(batch, y_pred, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_pearson_r", pr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_demeaned_r", dr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_variance_ratio", vr, on_step=False, on_epoch=True)
        self.log("val_norm_ratio", nr, on_step=False, on_epoch=True)
        self.log("val_reg_loss", reg_loss.detach(), on_step=False, on_epoch=True)
        self.log("val_edge_mse", aux_losses["edge_mse"], on_step=False, on_epoch=True)
        self.log("val_var_match_loss", aux_losses["var_match_loss"], on_step=False, on_epoch=True)
        self._log_structured_loss_terms("val")
        if "latent_mse" in aux_losses:
            self.log("val_latent_mse", aux_losses["latent_mse"], on_step=False, on_epoch=True)
        if "latent_weighted_mse" in aux_losses:
            self.log("val_latent_weighted_mse", aux_losses["latent_weighted_mse"], on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
