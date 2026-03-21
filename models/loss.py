"""
Model training loop and loss functions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_model_input(batch):
    """Return the preferred model input from a batch."""
    return batch["x"] if "x" in batch else batch["x_modalities"]


def get_batch_cov(batch):
    """Return covariate features from a batch when present."""
    return batch.get("cov")


def get_target_train_mean(base):
    """Extract target modality training mean from base dataset."""
    target_modality = getattr(base, "target_modality", None) or getattr(base, "target", None)
    if target_modality == "SC":
        return base.sc_train_avg
    elif target_modality == "FC":
        return base.fc_train_avg
    elif target_modality == "SC_r2t":
        return base.sc_r2t_corr_train_avg
    else:
        raise ValueError(f"Unknown target modality: {target_modality}")

# =============================================================================
# Loss Functions
# =============================================================================
class MSELoss(nn.Module):
    """Standard MSE loss."""    
    def __init__(self):
        super().__init__()
        self.name = "mse"
    
    def forward(self, y_pred, y_true, **kwargs):
        return F.mse_loss(y_pred, y_true)


class DemeanedMSELoss(nn.Module):
    """
    MSE loss on demeaned predictions and targets.
    Measures how well the model captures deviations from the training mean.
    
    Loss = MSE(y_pred - μ_train, y_true - μ_train)
    
    Args:
        target_train_mean: (d,) numpy array or tensor, training set mean for target modality
    """
    def __init__(self, target_train_mean):
        super().__init__()
        self.name = "demeaned_mse"
        if isinstance(target_train_mean, np.ndarray):
            target_train_mean = torch.tensor(target_train_mean, dtype=torch.float32)
        self.register_buffer('target_mean', target_train_mean)
    
    def forward(self, y_pred, y_true, **kwargs):
        y_pred_demeaned = y_pred - self.target_mean
        y_true_demeaned = y_true - self.target_mean
        return F.mse_loss(y_pred_demeaned, y_true_demeaned)


class WeightedMSELoss(nn.Module):
    """
    Weighted combination of standard MSE and demeaned MSE.
    
    Loss = α * MSE(y_pred, y_true) + (1-α) * MSE(y_pred - μ, y_true - μ)
    
    Both terms are normalized by their initial scale (computed from first batch)
    so they contribute equally when α=0.5.
    
    Args:
        target_train_mean: (d,) numpy array or tensor, training set mean for target modality
        alpha: weight for standard MSE (default 0.5). Higher = more weight on absolute accuracy.
    """
    
    def __init__(self, target_train_mean, alpha=0.5):
        super().__init__()
        self.name = "weighted_mse"
        self.alpha = alpha
        if isinstance(target_train_mean, np.ndarray):
            target_train_mean = torch.tensor(target_train_mean, dtype=torch.float32)
        self.register_buffer('target_mean', target_train_mean)
        
        # Normalization factors (estimated from first batch, then frozen)
        self.register_buffer('mse_scale', torch.tensor(1.0))
        self.register_buffer('demeaned_scale', torch.tensor(1.0))
        self._initialized = False
    
    def forward(self, y_pred, y_true, **kwargs):
        # Compute both losses
        mse_loss = F.mse_loss(y_pred, y_true)
        
        y_pred_demeaned = y_pred - self.target_mean
        y_true_demeaned = y_true - self.target_mean
        demeaned_loss = F.mse_loss(y_pred_demeaned, y_true_demeaned)
        
        # Initialize normalization scales from first batch
        if not self._initialized and self.training:
            with torch.no_grad(): # this dynamically standardizes to the first loss computation giving a rough estimate of the error magnitude of each
                self.mse_scale = mse_loss.clone()
                self.demeaned_scale = demeaned_loss.clone()
                self._initialized = True
        
        # Normalize both losses to similar scale
        mse_normalized = mse_loss / (self.mse_scale + 1e-8)
        demeaned_normalized = demeaned_loss / (self.demeaned_scale + 1e-8)
        
        # Weighted combination
        return self.alpha * mse_normalized + (1 - self.alpha) * demeaned_normalized


class SarwarMSECorrLoss(nn.Module):
    """
    MSE plus inter-subject correlation matching penalty.

    Loss = MSE(y_pred, y_true) + corr_weight * |mean_pair_corr(y_pred) - corr_target|
    """
    def __init__(self, corr_target=0.4, corr_weight=1e-3, eps=1e-8):
        super().__init__()
        self.name = "sarwar_mse_corr"
        self.corr_target = float(corr_target)
        self.corr_weight = float(corr_weight)
        self.eps = float(eps)

    def _mean_pairwise_corr(self, y_pred):
        bsz = y_pred.shape[0]
        if bsz < 2:
            return y_pred.new_tensor(0.0)

        centered = y_pred - y_pred.mean(dim=1, keepdim=True)
        norms = torch.sqrt(torch.sum(centered * centered, dim=1, keepdim=True) + self.eps)
        normalized = centered / norms
        corr_mat = normalized @ normalized.t()
        off_diag_sum = corr_mat.sum() - torch.diagonal(corr_mat).sum()
        return off_diag_sum / (bsz * (bsz - 1))

    def forward(self, y_pred, y_true, **kwargs):
        mse = F.mse_loss(y_pred, y_true)
        mean_corr = self._mean_pairwise_corr(y_pred)
        corr_penalty = torch.abs(mean_corr - self.corr_target)
        return mse + self.corr_weight * corr_penalty


class VAELoss(nn.Module):
    """
    VAE loss: reconstruction (MSE) + beta * KLD to standard Gaussian prior.
    KLD = -0.5 * mean(sum(1 + logvar - mu^2 - exp(logvar), dim=1)).
    
    Args:
        beta: weight for KLD term (default 1.0).
    """
    def __init__(self, beta=1.0):
        super().__init__()
        self.name = "vae"
        self.beta = beta
    
    def forward(self, y_pred, y_true, mu=None, logvar=None, **kwargs):
        recon = F.mse_loss(y_pred, y_true)
        if mu is not None and logvar is not None:
            # measures how far each sample’s Gaussian latent embedding deviates from a unit Gaussian
            kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            return recon + self.beta * kld
        return recon


def create_loss_fn(loss_type, base=None, alpha=0.5, beta=1.0, corr_target=0.4, corr_weight=1e-3):
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: one of 'mse', 'demeaned_mse', 'weighted_mse', 'vae'
        base: Dataset base object (required for demeaned losses)
        alpha: weight for weighted_mse (default 0.5)
        beta: weight for VAE KLD term (default 1.0)
        corr_target: target mean inter-subject correlation for sarwar_mse_corr.
        corr_weight: penalty strength for sarwar_mse_corr.

    Returns:
        Loss function module
    """
    if loss_type == "mse":
        return MSELoss()
    elif loss_type == "demeaned_mse":
        if base is None:
            raise ValueError("base is required for demeaned_mse loss")
        target_mean = get_target_train_mean(base)
        return DemeanedMSELoss(target_mean)
    elif loss_type == "weighted_mse":
        if base is None:
            raise ValueError("base is required for weighted_mse loss")
        target_mean = get_target_train_mean(base)
        return WeightedMSELoss(target_mean, alpha=alpha)
    elif loss_type == "vae":
        return VAELoss(beta=beta)
    elif loss_type == "sarwar_mse_corr":
        return SarwarMSECorrLoss(corr_target=corr_target, corr_weight=corr_weight)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from 'mse', 'demeaned_mse', 'weighted_mse', 'vae', 'sarwar_mse_corr'")


def compute_pearson_r(y_pred, y_true):
    """
    Compute mean Pearson correlation across samples.
    Each sample's prediction is correlated with its target across features.
    
    Args:
        y_pred: (batch, d) predicted values
        y_true: (batch, d) true values
    Returns:
        mean_r: scalar, mean Pearson r across batch
    """
    # Center each sample
    y_pred_centered = y_pred - y_pred.mean(dim=1, keepdim=True)
    y_true_centered = y_true - y_true.mean(dim=1, keepdim=True)
    
    # Compute correlation per sample
    numerator = (y_pred_centered * y_true_centered).sum(dim=1)
    denom_pred = torch.sqrt((y_pred_centered ** 2).sum(dim=1))
    denom_true = torch.sqrt((y_true_centered ** 2).sum(dim=1))
    
    r = numerator / (denom_pred * denom_true + 1e-10)
    return r.mean().item()


def compute_demeaned_pearson_r(y_pred, y_true, target_train_mean):
    """
    Compute demeaned Pearson correlation across samples.
    Correlation is computed between (y_pred - μ_train) and (y_true - μ_train).
    This measures how well we capture individual deviations from the population mean.
    
    Args:
        y_pred: (batch, d) predicted values
        y_true: (batch, d) true values
        target_train_mean: (d,) training set mean for target modality
    Returns:
        mean_r: scalar, mean demeaned Pearson r across batch
    """
    # Demean by training set mean
    y_pred_demeaned = y_pred - target_train_mean
    y_true_demeaned = y_true - target_train_mean
    
    # Compute correlation per sample
    numerator = (y_pred_demeaned * y_true_demeaned).sum(dim=1)
    denom_pred = torch.sqrt((y_pred_demeaned ** 2).sum(dim=1))
    denom_true = torch.sqrt((y_true_demeaned ** 2).sum(dim=1))
    
    r = numerator / (denom_pred * denom_true + 1e-10)
    return r.mean().item()


def evaluate_model(model, data_loader, target_train_mean, device):
    """
    Evaluate model on a data loader.
    
    Args:
        model: the model to evaluate
        data_loader: DataLoader yielding batches with model inputs and 'y'
        target_train_mean: (d,) training mean for target modality (for demeaned corr)
        device: torch device
        
    Returns:
        dict with 'mse', 'pearson_r', 'demeaned_r'
    """
    model.eval()
    total_mse = 0.0
    total_pearson_r = 0.0
    total_demeaned_r = 0.0
    n_batches = 0
    
    target_mean_tensor = torch.tensor(target_train_mean, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        for batch in data_loader:
            x = get_model_input(batch)
            y = batch["y"].to(device)

            kwargs = {}
            if getattr(model, "uses_cov", False):
                kwargs["cov"] = get_batch_cov(batch)
                # For the target-leakage sanity test: let the projector see the true targets here too
                if getattr(model, "use_target_scores_in_projector", False) and "y" in batch:
                    kwargs["y"] = batch["y"]
            if getattr(model, "uses_node_features", False) and "node_features" in batch:
                kwargs["node_features"] = batch["node_features"]
            out = model(x, **kwargs) if kwargs else model(x)
            y_pred = out[0] if isinstance(out, tuple) else out

            mse = F.mse_loss(y_pred, y).item()
            total_mse += mse

            pearson_r = compute_pearson_r(y_pred, y)
            total_pearson_r += pearson_r

            demeaned_r = compute_demeaned_pearson_r(y_pred, y, target_mean_tensor)
            total_demeaned_r += demeaned_r

            n_batches += 1
    
    return {
        'mse': total_mse / n_batches,
        'pearson_r': total_pearson_r / n_batches,
        'demeaned_r': total_demeaned_r / n_batches,
    }


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
            "val_loss": val_metrics["mse"],  # use same val source as pearson/demeaned for consistent curve
            "val_mse": val_metrics["mse"],
            "val_pearson_r": val_metrics["pearson_r"],
            "val_demeaned_r": val_metrics["demeaned_r"],
        }
        pl_module.log("val_mse", val_metrics["mse"], on_epoch=True)
        pl_module.log("val_pearson_r", val_metrics["pearson_r"], on_epoch=True)
        pl_module.log("val_demeaned_r", val_metrics["demeaned_r"], on_epoch=True)
        if trainer.current_epoch % self.log_every == 0 or trainer.current_epoch == 0:
            if self.log_train and self.train_loader is not None:
                train_metrics = evaluate_model(
                    model, self.train_loader, self.target_train_mean, device
                )
                record["train_mse"] = train_metrics["mse"]
                record["train_pearson_r"] = train_metrics["pearson_r"]
                record["train_demeaned_r"] = train_metrics["demeaned_r"]
                pl_module.log("train_mse", train_metrics["mse"], on_epoch=True)
                pl_module.log("train_pearson_r", train_metrics["pearson_r"], on_epoch=True)
                pl_module.log("train_demeaned_r", train_metrics["demeaned_r"], on_epoch=True)
            if self.log_gpu:
                gpu_alloc, gpu_reserved = get_gpu_memory_usage()
                record["gpu_allocated_mb"] = gpu_alloc
                record["gpu_reserved_mb"] = gpu_reserved
                pl_module.log("gpu_allocated_mb", float(gpu_alloc), on_epoch=True)
                pl_module.log("gpu_reserved_mb", float(gpu_reserved), on_epoch=True)
        self.history.append(record)


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
        loss_type: one of 'mse', 'demeaned_mse', 'weighted_mse', 'vae'.
        loss_alpha: weight for weighted_mse (passed to Lightning module).
        loss_beta: KLD weight for VAE loss (passed to Lightning module).
        loss_corr_target: target inter-subject correlation for sarwar_mse_corr.
        loss_corr_weight: penalty strength for sarwar_mse_corr.
        max_epochs: number of epochs (Trainer max_epochs).
        logger: If True, use CSVLogger (writes to disk). If False, no logging to disk (dev mode).
        pl_logger: Optional Lightning logger (e.g. WandbLogger). If set, overrides logger/CSVLogger.
        enable_progress_bar: If True, show training progress bar (e.g. in dev/notebook).

    Returns:
        TrainResult with pl_module, trainer, callback, history_df and .plot().
    """
    from models.lightning_module import CrossModalLightningModule

    pl_module = CrossModalLightningModule(
        model=model,
        base=base,
        lr=lr,
        loss_type=loss_type,
        loss_alpha=loss_alpha,
        loss_beta=loss_beta,
        loss_corr_target=loss_corr_target,
        loss_corr_weight=loss_corr_weight,
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
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        max_epochs=max_epochs,
        logger=logger_pl,
        callbacks=[callback],
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

        def plot(self):
            """Plot training history: loss, Pearson r, demeaned r (train and val)."""
            df = self.history_df
            if df.empty:
                return
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            epochs = df["epoch"]
            if "train_loss" in df.columns and df["train_loss"].notna().any():
                axes[0].plot(epochs, df["train_loss"], "b-o", label="Train", markersize=3)
            if "val_loss" in df.columns and df["val_loss"].notna().any():
                axes[0].plot(epochs, df["val_loss"], "r-o", label="Val", markersize=3)
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Loss")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            if "train_pearson_r" in df.columns and df["train_pearson_r"].notna().any():
                axes[1].plot(epochs, df["train_pearson_r"], "b-o", label="Train", markersize=3)
            if "val_pearson_r" in df.columns and df["val_pearson_r"].notna().any():
                axes[1].plot(epochs, df["val_pearson_r"], "r-o", label="Val", markersize=3)
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Pearson r")
            axes[1].set_title("Pearson r")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            if "train_demeaned_r" in df.columns and df["train_demeaned_r"].notna().any():
                axes[2].plot(epochs, df["train_demeaned_r"], "b-o", label="Train", markersize=3)
            if "val_demeaned_r" in df.columns and df["val_demeaned_r"].notna().any():
                axes[2].plot(epochs, df["val_demeaned_r"], "r-o", label="Val", markersize=3)
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("Demeaned Pearson r")
            axes[2].set_title("Demeaned r")
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            fig.tight_layout()
            plt.show()

    return TrainResult(pl_module, trainer, callback, history_df)
