"""Training run orchestration.

Builds the Lightning trainer, callbacks, checkpoints, and training result object.
"""
import pandas as pd
import matplotlib.pyplot as plt
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from collections import OrderedDict

from models.eval.evaluator import evaluate_model
from models.train.loss import get_target_train_mean
from models.train.training_viz import plot_training_history


def get_gpu_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        return allocated, reserved
    return 0.0, 0.0


class ValidationEvalCallback(pl.Callback):
    """
    Records one row per epoch (train_loss, val_loss from Lightning).
    Every log_every epochs runs full evaluate_model and adds val_mse, val_pearson_r,
    val_demeaned_r (and optionally train_*). History is stored as a list of dicts;
    TrainResult builds a DataFrame from it for plotting.
    """
    def __init__(
        self,
        val_loader,
        target_train_mean,
        train_loader=None,
        log_every=1,
        log_train=False,
        log_gpu=False,
    ):
        self.target_train_mean = target_train_mean
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log_every = log_every
        self.log_train = log_train
        self.log_gpu = log_gpu
        self.history = []

    def on_validation_epoch_end(self, trainer, pl_module):
        cm = trainer.callback_metrics
        model = pl_module.model
        device = pl_module.device
        
        # Always run full val evaluate_model so plotted val metrics are from one consistent path (avoids sawtooth from mixing Lightning step vs full pass).
        val_metrics = evaluate_model(
            model, self.val_loader, self.target_train_mean, device
        )
        record = {
            "epoch": trainer.current_epoch,
            "train_loss": cm.get("train_loss"),
            "val_loss": cm.get("val_loss"),
            "train_edge_mse": cm.get("train_edge_mse"),
            "val_edge_mse": cm.get("val_edge_mse"),
            "train_var_match_loss": cm.get("train_var_match_loss"),
            "val_var_match_loss": cm.get("val_var_match_loss"),
            "train_latent_mse": cm.get("train_latent_mse"),
            "val_latent_mse": cm.get("val_latent_mse"),
            "train_latent_weighted_mse": cm.get("train_latent_weighted_mse"),
            "val_latent_weighted_mse": cm.get("val_latent_weighted_mse"),
            "train_reg_loss": cm.get("train_reg_loss"),
            "val_reg_loss": cm.get("val_reg_loss"),
            "val_mse": val_metrics["mse"],
            "val_pearson_r": val_metrics["pearson_r"],
            "val_demeaned_r": val_metrics["demeaned_r"],
            "val_variance_ratio": val_metrics["variance_ratio"],
            "val_norm_ratio": val_metrics["norm_ratio"],
        }
        for key, value in cm.items():
            if key.startswith(("train_loss_term_", "val_loss_term_", "train_loss_raw_", "val_loss_raw_", "train_loss_ref_", "val_loss_ref_")):
                record[key] = value
        pl_module.log("val_mse", val_metrics["mse"], on_epoch=True)
        pl_module.log("val_pearson_r", val_metrics["pearson_r"], on_epoch=True)
        pl_module.log("val_demeaned_r", val_metrics["demeaned_r"], on_epoch=True, prog_bar=True)
        pl_module.log("val_variance_ratio", val_metrics["variance_ratio"], on_epoch=True)
        pl_module.log("val_norm_ratio", val_metrics["norm_ratio"], on_epoch=True)
        if trainer.current_epoch % self.log_every == 0 or trainer.current_epoch == 0:
            if self.log_train and self.train_loader is not None:
                train_metrics = evaluate_model(
                    model, self.train_loader, self.target_train_mean, device
                )
                record["train_mse"] = train_metrics["mse"]
                record["train_pearson_r"] = train_metrics["pearson_r"]
                record["train_demeaned_r"] = train_metrics["demeaned_r"]
                record["train_variance_ratio"] = train_metrics["variance_ratio"]
                record["train_norm_ratio"] = train_metrics["norm_ratio"]
                pl_module.log("train_mse", train_metrics["mse"], on_epoch=True)
                pl_module.log("train_pearson_r", train_metrics["pearson_r"], on_epoch=True)
                pl_module.log("train_demeaned_r", train_metrics["demeaned_r"], on_epoch=True)
                pl_module.log("train_variance_ratio", train_metrics["variance_ratio"], on_epoch=True)
                pl_module.log("train_norm_ratio", train_metrics["norm_ratio"], on_epoch=True)
            if self.log_gpu:
                gpu_alloc, gpu_reserved = get_gpu_memory_usage()
                record["gpu_allocated_mb"] = gpu_alloc
                record["gpu_reserved_mb"] = gpu_reserved
                pl_module.log("gpu_allocated_mb", float(gpu_alloc), on_epoch=True)
                pl_module.log("gpu_reserved_mb", float(gpu_reserved), on_epoch=True)
        self.history.append(record)


class OrderedMetricsProgressBar(TQDMProgressBar):
    """
    Progress bar with stable metric ordering and concise display names.
    """

    DISPLAY_ORDER = [
        ("train_loss", "train_loss"),
        ("val_loss", "val_loss"),
        ("train_demeaned_r", "train_demeaned"),
        ("val_demeaned_r", "val_demeaned"),
        ("train_pearson_r", "train_r"),
        ("val_pearson_r", "val_r"),
    ]

    def get_metrics(self, trainer, pl_module):
        base_metrics = super().get_metrics(trainer, pl_module)
        ordered = OrderedDict()

        if "v_num" in base_metrics:
            ordered["v_num"] = base_metrics["v_num"]

        for raw_key, display_key in self.DISPLAY_ORDER:
            if raw_key in base_metrics:
                ordered[display_key] = base_metrics[raw_key]

        display_source_keys = {raw_key for raw_key, _ in self.DISPLAY_ORDER}
        for key, value in base_metrics.items():
            if key == "v_num" or key in display_source_keys:
                continue
            ordered[key] = value
        return ordered


def train_model(
    model,
    train_loader,
    val_loader,
    base,
    log_every=5,
    lr=1e-4,
    loss_type="mse",
    loss_alpha=0.5,
    loss_beta=1.0,
    loss_corr_target=0.4,
    loss_corr_weight=1e-3,
    loss_terms=None,
    loss_normalize="ema",
    loss_scale_ema_decay=0.95,
    loss_scale_warmup_steps=20,
    max_epochs=100,
    logger=True,
    pl_logger=None,
    enable_progress_bar=False,
):
    """
    Train a cross-modal model using PyTorch Lightning Trainer.

    Args:
        model: nn.Module to train (e.g. CrossModal_PCA_PLS_learnable, CrossModalVAE).
        train_loader: DataLoader for training data (batches with 'x', 'y' keys).
        val_loader: DataLoader for validation data.
        base: Dataset base object (e.g., HCP_Base) with target modality info.
        log_every: run full evaluate_model and log refined metrics every N epochs.
        lr: learning rate (passed to Lightning module).
        loss_type: one of 'mse', 'weighted_mse', 'vae', 'sarwar_mse_corr'.
        loss_alpha: weight for weighted_mse (passed to Lightning module).
        loss_beta: KLD weight for VAE loss (passed to Lightning module).
        loss_corr_target: target inter-subject correlation for sarwar_mse_corr.
        loss_corr_weight: penalty strength for sarwar_mse_corr.
        loss_terms: component loss specs used by composite.
        loss_normalize: normalization mode for composite loss terms ('ema' or 'none').
        loss_scale_ema_decay: EMA decay for composite loss-term scaling.
        loss_scale_warmup_steps: number of training steps used to calibrate EMA loss scales.
        max_epochs: number of epochs (Trainer max_epochs).
        logger: If True, use CSVLogger (writes to disk). If False, no logging to disk (dev mode).
        pl_logger: Optional Lightning logger (e.g. WandbLogger). If set, overrides logger/CSVLogger.
        enable_progress_bar: If True, show training progress bar (e.g. in dev/notebook).

    Returns:
        TrainResult with pl_module, trainer, callback, history_df and .plot().
    """
    from models.train.lightning_module import CrossModalLightningModule

    pl_module = CrossModalLightningModule(
        model=model,
        base=base,
        lr=lr,
        loss_type=loss_type,
        loss_alpha=loss_alpha,
        loss_beta=loss_beta,
        loss_corr_target=loss_corr_target,
        loss_corr_weight=loss_corr_weight,
        loss_terms=loss_terms,
        loss_normalize=loss_normalize,
        loss_scale_ema_decay=loss_scale_ema_decay,
        loss_scale_warmup_steps=loss_scale_warmup_steps,
    )
    target_train_mean = get_target_train_mean(base)
    callback = ValidationEvalCallback(
        val_loader=val_loader,
        target_train_mean=target_train_mean,
        train_loader=train_loader,
        log_every=log_every,
        log_train=True,
        log_gpu=torch.cuda.is_available(),
    )
    if pl_logger is not None:
        logger_pl = pl_logger
    else:
        logger_pl = pl.loggers.CSVLogger(save_dir="results/lightning_logs", name="conn2conn") if logger else False
    # Single device: use "auto" to avoid subprocess spawn and "CUDA in forked subprocess" errors.
    # Multi-GPU: use "ddp_notebook" in notebooks, "ddp" in scripts.
    devices = 1
    try:
        _in_notebook = get_ipython() is not None
    except NameError:
        _in_notebook = False
    if devices == 1:
        strategy = "auto"
    elif _in_notebook:
        strategy = "ddp_notebook"
    else:
        strategy = "ddp"
    callbacks = [callback]
    if enable_progress_bar:
        callbacks.append(OrderedMetricsProgressBar())

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        max_epochs=max_epochs,
        logger=logger_pl,
        callbacks=callbacks,
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=True,
    )
    trainer.fit(
        pl_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)

    # Build DataFrame from callback history (tensors -> float for CSV-friendly storage)
    def _history_to_df(history):
        rows = []
        for r in history:
            row = {}
            for k, v in r.items():
                if hasattr(v, "item"):
                    row[k] = v.item()
                else:
                    row[k] = v
            rows.append(row)
        return pd.DataFrame(rows)

    history_df = _history_to_df(callback.history)

    class TrainResult:
        def __init__(self, pl_module, trainer, callback, history_df):
            self.pl_module = pl_module
            self.trainer = trainer
            self.callback = callback
            self.history_df = history_df

        def plot(self, style="default", **kwargs):
            """Plot training history; use style='default' or style='latent'."""
            if self.history_df.empty:
                return
            fig, _ = plot_training_history(self.history_df, style=style, **kwargs)
            plt.show()

    return TrainResult(pl_module, trainer, callback, history_df)
