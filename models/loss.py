"""
Model training loop and loss functions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks.progress import TQDMProgressBar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict


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


def compute_var_match_loss(y_pred, y_true, axis=0, relative_to_true=True):
    """
    Match prediction variance to target variance.

    By default this uses the relative formulation from Krakencoder:
    ((var_true - var_pred) / var_true)^2
    """
    true_var = torch.mean((y_true - y_true.mean(dim=axis, keepdim=True)) ** 2)
    pred_var = torch.mean((y_pred - y_pred.mean(dim=axis, keepdim=True)) ** 2)
    if relative_to_true:
        return ((true_var - pred_var) / (true_var + 1e-10)) ** 2
    return (true_var - pred_var) ** 2


class MSEVarMatchLoss(nn.Module):
    """
    Joint objective: edge-space MSE plus a small variance-matching penalty.

    This is aimed at combating prediction shrinkage / under-dispersion.
    """

    def __init__(self, var_weight=0.01, relative_to_true=True):
        super().__init__()
        self.name = "joint_mse_varmatch"
        self.var_weight = float(var_weight)
        self.relative_to_true = bool(relative_to_true)

    def forward(self, y_pred, y_true, **kwargs):
        mse = F.mse_loss(y_pred, y_true)
        var_loss = compute_var_match_loss(
            y_pred,
            y_true,
            axis=0,
            relative_to_true=self.relative_to_true,
        )
        return mse + self.var_weight * var_loss


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


def create_loss_fn(
    loss_type,
    base=None,
    alpha=0.5,
    beta=1.0,
    corr_target=0.4,
    corr_weight=1e-3,
    var_weight=0.01,
):
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: one of 'mse', 'weighted_mse', 'vae', 'sarwar_mse_corr', 'joint_mse_varmatch'
        base: Dataset base object (required for weighted_mse)
        alpha: weight for weighted_mse (default 0.5)
        beta: weight for VAE KLD term (default 1.0)
        corr_target: target mean inter-subject correlation for sarwar_mse_corr.
        corr_weight: penalty strength for sarwar_mse_corr.

    Returns:
        Loss function module
    """
    if loss_type == "mse":
        return MSELoss()
    elif loss_type == "weighted_mse":
        if base is None:
            raise ValueError("base is required for weighted_mse loss")
        target_mean = get_target_train_mean(base)
        return WeightedMSELoss(target_mean, alpha=alpha)
    elif loss_type == "vae":
        return VAELoss(beta=beta)
    elif loss_type == "sarwar_mse_corr":
        return SarwarMSECorrLoss(corr_target=corr_target, corr_weight=corr_weight)
    elif loss_type == "joint_mse_varmatch":
        return MSEVarMatchLoss(var_weight=var_weight, relative_to_true=True)
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. Choose from "
            "'mse', 'weighted_mse', 'vae', 'sarwar_mse_corr', 'joint_mse_varmatch'"
        )


def compute_latent_reconstruction_loss(c_pred, c_true, loss_type, weights=None, mask=None):
    """
    Latent-space reconstruction loss helpers for models that explicitly predict
    target PCA coefficients.

    Args:
        c_pred: predicted latent coefficients, shape (B, k)
        c_true: target latent coefficients, shape (B, k)
        loss_type: 'latent_mse' or 'latent_weighted_mse'
        weights: optional per-component weights, shape (k,)
    """
    diff_sq = (c_pred - c_true) ** 2
    if mask is not None:
        mask = mask.to(c_pred.device).to(c_pred.dtype)
        if mask.ndim == 1:
            mask = mask.view(1, -1)
        diff_sq = diff_sq * mask
        denom = mask.sum().clamp_min(1.0)
    else:
        denom = torch.tensor(diff_sq.numel(), device=c_pred.device, dtype=c_pred.dtype)
    if loss_type == "latent_mse":
        return diff_sq.sum() / denom
    if loss_type == "latent_weighted_mse":
        if weights is None:
            raise ValueError("latent_weighted_mse requires per-component weights.")
        weights = weights.view(1, -1).to(c_pred.device).to(c_pred.dtype)
        diff_sq = diff_sq * weights
        return diff_sq.sum() / denom
    raise ValueError(f"Unknown latent loss type: {loss_type}. Choose from 'latent_mse', 'latent_weighted_mse'.")


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


def compute_prediction_variance_ratio(y_pred, y_true):
    """
    Compare mean across-feature subject variance in predictions vs targets.

    Ratio < 1 suggests under-dispersed predictions across subjects.
    Ratio > 1 suggests over-dispersed predictions.
    """
    pred_var = torch.var(y_pred, dim=0, unbiased=False).mean()
    true_var = torch.var(y_true, dim=0, unbiased=False).mean()
    return (pred_var / (true_var + 1e-10)).item()


def compute_prediction_norm_ratio(y_pred, y_true):
    """
    Compare average per-subject prediction norm to target norm.

    Ratio < 1 suggests globally shrunken predictions.
    Ratio > 1 suggests globally amplified predictions.
    """
    pred_norm = torch.norm(y_pred, dim=1).mean()
    true_norm = torch.norm(y_true, dim=1).mean()
    return (pred_norm / (true_norm + 1e-10)).item()


DEFAULT_HISTORY_LOSS_PAIRS = [
    ("train_loss", "val_loss", "Loss"),
]

LATENT_HISTORY_LOSS_PAIRS = [
    ("train_loss", "val_loss", "Active Loss"),
    ("train_edge_mse", "val_edge_mse", "Edge MSE"),
    ("train_var_match_loss", "val_var_match_loss", "Var Match"),
    ("train_latent_mse", "val_latent_mse", "Latent MSE"),
    ("train_latent_weighted_mse", "val_latent_weighted_mse", "Latent Weighted MSE"),
    ("train_joint_edge_term", "val_joint_edge_term", "Joint Edge Term"),
    ("train_joint_latent_term", "val_joint_latent_term", "Joint Latent Term"),
]

DEFAULT_HISTORY_CORR_PAIRS = [
    ("train_pearson_r", "val_pearson_r", "Pearson r"),
    ("train_demeaned_r", "val_demeaned_r", "Demeaned r"),
    ("train_variance_ratio", "val_variance_ratio", "Pred/Target Variance"),
    ("train_norm_ratio", "val_norm_ratio", "Pred/Target Norm"),
]


def _available_history_pairs(df, pairs):
    out = []
    for train_key, val_key, label in pairs:
        if train_key in df.columns and df[train_key].notna().any():
            out.append((train_key, val_key if val_key in df.columns else None, label))
        elif val_key in df.columns and df[val_key].notna().any():
            out.append((train_key if train_key in df.columns else None, val_key, label))
    return out


def summarize_history_columns(history_df):
    df = history_df if isinstance(history_df, pd.DataFrame) else pd.DataFrame(history_df)
    cols = []
    for col in df.columns:
        if col == "epoch":
            continue
        if df[col].notna().any():
            cols.append(col)
    return cols


def plot_training_history(history_df, style="default", figsize=None, marker_size=3, grid_alpha=0.3):
    """
    Plot training history from TrainResult.history_df.

    style:
    - 'default': classic view with loss, Pearson r, and demeaned r
    - 'latent': losses on the top row and correlation metrics below
    """
    if history_df is None:
        raise ValueError("history_df is None")
    df = history_df if isinstance(history_df, pd.DataFrame) else pd.DataFrame(history_df)
    if df.empty:
        raise ValueError("history_df is empty")
    if "epoch" not in df.columns:
        raise ValueError("history_df must contain an 'epoch' column")

    epochs = df["epoch"]

    if style == "default":
        pairs = _available_history_pairs(df, DEFAULT_HISTORY_LOSS_PAIRS + DEFAULT_HISTORY_CORR_PAIRS)
        if figsize is None:
            figsize = (12, 4)
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        axes = np.atleast_1d(axes)
        plot_pairs = [
            _available_history_pairs(df, DEFAULT_HISTORY_LOSS_PAIRS),
            _available_history_pairs(df, [DEFAULT_HISTORY_CORR_PAIRS[0]]),
            _available_history_pairs(df, [DEFAULT_HISTORY_CORR_PAIRS[1]]),
        ]
        for ax, pair_group in zip(axes, plot_pairs):
            if not pair_group:
                ax.axis("off")
                continue
            train_key, val_key, title = pair_group[0]
            if train_key is not None and train_key in df.columns and df[train_key].notna().any():
                ax.plot(epochs, df[train_key], "b-o", label="Train", markersize=marker_size)
            if val_key is not None and val_key in df.columns and df[val_key].notna().any():
                ax.plot(epochs, df[val_key], "r-o", label="Val", markersize=marker_size)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=grid_alpha)
            ax.legend()
        fig.tight_layout()
        return fig, axes

    if style == "latent":
        loss_pairs = _available_history_pairs(df, LATENT_HISTORY_LOSS_PAIRS)
        corr_pairs = _available_history_pairs(df, DEFAULT_HISTORY_CORR_PAIRS)
        ncols = max(len(loss_pairs), len(corr_pairs), 1)
        if figsize is None:
            figsize = (5 * ncols, 7)
        fig, axes = plt.subplots(2, ncols, figsize=figsize, squeeze=False)
        for ax_idx in range(ncols):
            if ax_idx < len(loss_pairs):
                train_key, val_key, title = loss_pairs[ax_idx]
                ax = axes[0, ax_idx]
                if train_key is not None and train_key in df.columns and df[train_key].notna().any():
                    ax.plot(epochs, df[train_key], "b-o", label="Train", markersize=marker_size)
                if val_key is not None and val_key in df.columns and df[val_key].notna().any():
                    ax.plot(epochs, df[val_key], "r-o", label="Val", markersize=marker_size)
                ax.set_title(title)
                ax.set_xlabel("Epoch")
                ax.grid(True, alpha=grid_alpha)
                ax.legend()
            else:
                axes[0, ax_idx].axis("off")

            if ax_idx < len(corr_pairs):
                train_key, val_key, title = corr_pairs[ax_idx]
                ax = axes[1, ax_idx]
                if train_key is not None and train_key in df.columns and df[train_key].notna().any():
                    ax.plot(epochs, df[train_key], "b-o", label="Train", markersize=marker_size)
                if val_key is not None and val_key in df.columns and df[val_key].notna().any():
                    ax.plot(epochs, df[val_key], "r-o", label="Val", markersize=marker_size)
                ax.set_title(title)
                ax.set_xlabel("Epoch")
                ax.grid(True, alpha=grid_alpha)
                ax.legend()
            else:
                axes[1, ax_idx].axis("off")
        fig.tight_layout()
        return fig, axes

    raise ValueError("Unknown plot style. Choose from {'default', 'latent'}.")


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
    total_variance_ratio = 0.0
    total_norm_ratio = 0.0
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

            variance_ratio = compute_prediction_variance_ratio(y_pred, y)
            total_variance_ratio += variance_ratio

            norm_ratio = compute_prediction_norm_ratio(y_pred, y)
            total_norm_ratio += norm_ratio
            
            n_batches += 1
    
    return {
        'mse': total_mse / n_batches,
        'pearson_r': total_pearson_r / n_batches,
        'demeaned_r': total_demeaned_r / n_batches,
        'variance_ratio': total_variance_ratio / n_batches,
        'norm_ratio': total_norm_ratio / n_batches,
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
            "val_loss": cm.get("val_loss"),
            "train_edge_mse": cm.get("train_edge_mse"),
            "val_edge_mse": cm.get("val_edge_mse"),
            "train_var_match_loss": cm.get("train_var_match_loss"),
            "val_var_match_loss": cm.get("val_var_match_loss"),
            "train_latent_mse": cm.get("train_latent_mse"),
            "val_latent_mse": cm.get("val_latent_mse"),
            "train_latent_weighted_mse": cm.get("train_latent_weighted_mse"),
            "val_latent_weighted_mse": cm.get("val_latent_weighted_mse"),
            "train_joint_edge_term": cm.get("train_joint_edge_term"),
            "val_joint_edge_term": cm.get("val_joint_edge_term"),
            "train_joint_latent_term": cm.get("train_joint_latent_term"),
            "val_joint_latent_term": cm.get("val_joint_latent_term"),
            "train_edge_loss_ref": cm.get("train_edge_loss_ref"),
            "val_edge_loss_ref": cm.get("val_edge_loss_ref"),
            "train_latent_loss_ref": cm.get("train_latent_loss_ref"),
            "val_latent_loss_ref": cm.get("val_latent_loss_ref"),
            "val_mse": val_metrics["mse"],
            "val_pearson_r": val_metrics["pearson_r"],
            "val_demeaned_r": val_metrics["demeaned_r"],
            "val_variance_ratio": val_metrics["variance_ratio"],
            "val_norm_ratio": val_metrics["norm_ratio"],
        }
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
    loss_var_weight=0.01,
    loss_latent_weight=0.25,
    loss_scale_ema_decay=0.95,
    loss_scale_warmup_steps=100,
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
        loss_var_weight: weight on the variance-matching term for joint_mse_varmatch.
        loss_latent_weight: weight on latent reconstruction term for joint edge+latent losses.
        loss_scale_ema_decay: EMA decay for automatic loss scaling in joint_edge_latent_mse_scaled.
        loss_scale_warmup_steps: number of training steps used to calibrate EMA loss scales.
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
        loss_var_weight=loss_var_weight,
        loss_latent_weight=loss_latent_weight,
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
