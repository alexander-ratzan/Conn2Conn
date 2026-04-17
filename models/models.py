import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import pearsonr, spearmanr
from scipy.io import loadmat
from sklearn.decomposition import PCA
from scipy import stats
from scipy.sparse.linalg import LinearOperator, svds
import torch
import torch.nn as nn
from models.loss import create_loss_fn

# QUICK HELPERS
def _build_mlp(in_dim, hidden_dims, out_dim, dropout_p=0.1, use_layer_norm=True):
    """
    Build MLP: in_dim -> hidden_dims -> out_dim.

    Each hidden transition is:  Linear -> ReLU -> [LayerNorm] -> [Dropout]
    The final Linear has no activation, norm, or dropout.

    Args:
        use_layer_norm: If True, insert LayerNorm after the activation of every
            hidden layer.  Normalises intermediate representations, which reduces
            internal covariate shift and acts as a mild regulariser — especially
            useful for auxiliary branches trained on small/structured datasets.
    """
    layers = []
    dims = [in_dim] + list(hidden_dims) + [out_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            if use_layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))
    return nn.Sequential(*layers)


def compute_reg_loss(parameters, l1_l2_tuple=(0.0, 0.0)):
    """
    Compute L1/L2 regularization loss for given parameters.
    
    Args:
        parameters: Iterable of torch.nn.Parameter or torch.Tensor to regularize.
                    Can be a list/tuple of specific parameters or model.parameters().
        l1_l2_tuple: Tuple of (l1_reg, l2_reg) weights. Default (0.0, 0.0).
    
    Returns:
        Regularization loss (scalar tensor or 0.0 if both weights are 0).
    """
    l1_reg, l2_reg = l1_l2_tuple
    
    if l1_reg == 0 and l2_reg == 0:
        return 0.0
    
    reg = 0.0
    for p in parameters:
        if p.requires_grad:
            if l1_reg > 0:
                reg = reg + l1_reg * p.abs().sum()
            if l2_reg > 0:
                reg = reg + l2_reg * (p ** 2).sum()
    return reg


def get_model_input(batch):
    """Return the preferred model input from a batch."""
    return batch["x"] if "x" in batch else batch["x_modalities"]


def get_batch_cov(batch):
    """Return covariate features from a batch when present."""
    return batch.get("cov")


def get_single_modality_data(base, modality, include_scores=True, include_raw_data=False):
    """Extract train-split summaries and optional raw data for one modality."""
    mapping = {
        "SC": (
            base.sc_train_avg,
            base.sc_train_loadings,
            base.sc_train_scores if include_scores else None,
            base.sc_upper_triangles if include_raw_data else None,
        ),
        "FC": (
            base.fc_train_avg,
            base.fc_train_loadings,
            base.fc_train_scores if include_scores else None,
            base.fc_upper_triangles if include_raw_data else None,
        ),
        "SC_r2t": (
            base.sc_r2t_corr_train_avg,
            base.sc_r2t_corr_train_loadings,
            base.sc_r2t_corr_train_scores if include_scores else None,
            base.sc_r2t_corr_upper_triangles if include_raw_data else None,
        ),
    }
    if modality not in mapping:
        raise ValueError(f"Unknown modality: {modality}")

    mean, loadings, scores, raw = mapping[modality]
    if mean is None or loadings is None:
        raise ValueError(f"Modality '{modality}' is unavailable in this dataset instance.")

    result = {
        "mean": mean,
        "loadings": loadings,
    }
    if include_scores:
        result["scores"] = scores
    if include_raw_data:
        result["upper_triangles"] = raw
        result["train_indices"] = base.trainvaltest_partition_indices["train"]
    return result


def get_modality_data(base, device=None, include_scores=True, include_raw_data=False):
    """
    Extract source and target modality data from base dataset object.
    
    Args:
        base: Dataset base object (e.g., HCP_Base) with source/target modalities.
        device: torch device for tensors (default: CPU).
        include_scores: If True, include PCA scores. Default True.
        include_raw_data: If True, include raw upper_triangles arrays. Default False.
    
    Returns:
        dict with keys:
            - source_mean: (d_source,) numpy array
            - source_loadings: (d_source, n_components) numpy array
            - source_scores: (n_train, n_components) numpy array (if include_scores=True)
            - target_mean: (d_target,) numpy array
            - target_loadings: (d_target, n_components) numpy array
            - target_scores: (n_train, n_components) numpy array (if include_scores=True)
            - source_upper_triangles: (n_total, d_source) numpy array (if include_raw_data=True)
            - target_upper_triangles: (n_total, d_target) numpy array (if include_raw_data=True)
            - train_indices: array of training indices (if include_raw_data=True)
    """
    if device is None:
        device = torch.device("cpu")
    
    source_modalities = getattr(base, "source_modalities", [base.source])
    target_modality = getattr(base, "target", None)
    if target_modality is None:
        target_modality = getattr(base, "target_modalities", [base.target])[0]
    source_data = {
        modality: get_single_modality_data(
            base,
            modality,
            include_scores=include_scores,
            include_raw_data=include_raw_data,
        )
        for modality in source_modalities
    }
    target_data = get_single_modality_data(
        base,
        target_modality,
        include_scores=include_scores,
        include_raw_data=include_raw_data,
    )

    result = {
        "sources": source_data,
        "target": target_data,
    }

    # Preserve the legacy flat keys for single-source models.
    if len(source_modalities) == 1:
        source_only = source_data[source_modalities[0]]
        result.update(
            {
                "source_mean": source_only["mean"],
                "source_loadings": source_only["loadings"],
                "target_mean": target_data["mean"],
                "target_loadings": target_data["loadings"],
            }
        )
        if include_scores:
            result["source_scores"] = source_only["scores"]
            result["target_scores"] = target_data["scores"]
        if include_raw_data:
            result["source_upper_triangles"] = source_only["upper_triangles"]
            result["target_upper_triangles"] = target_data["upper_triangles"]
            result["train_indices"] = target_data["train_indices"]

    return result


def predict_from_loader(model, data_loader, device=None):
    """Generate predictions from a model using a data loader."""
    model.eval()
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            # Model has no parameters (e.g. only buffers, like CrossModal_PLS_SVD)
            try:
                device = next(model.buffers()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in data_loader:
            x = get_model_input(batch)
            y = batch["y"].to(device)
            kwargs = {}
            if getattr(model, "uses_cov", False):
                kwargs["cov"] = get_batch_cov(batch)
                # For the target-leakage sanity test: also feed true targets into the projector
                if getattr(model, "use_target_scores_in_projector", False) and "y" in batch:
                    kwargs["y"] = batch["y"]
            if getattr(model, "uses_node_features", False) and "node_features" in batch:
                kwargs["node_features"] = batch["node_features"]
            out = model(x, **kwargs) if kwargs else model(x)
            preds = out[0] if isinstance(out, tuple) else out
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return all_preds, all_targets

from models.architectures.crossmodal_pca_pls import (
    CrossModalPCA,
    CrossModal_PLS_SVD,
    CrossModal_PCA_PLS,
    CrossModal_PCA_PLS_learnable,
    CrossModal_PCA_PLS_CovProjector,
)
from models.architectures.crossmodal_vae import CrossModalVAE
from models.architectures.krakencoder_precomputed import KrakencoderPrecomputed
