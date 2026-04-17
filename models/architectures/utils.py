"""Shared helpers for architecture modules.

Contains small model-construction, regularization, batch, and modality-data helpers
used across multiple architecture implementations.
"""

import torch
import torch.nn as nn


def _build_mlp(in_dim, hidden_dims, out_dim, dropout_p=0.1, use_layer_norm=True):
    """
    Build MLP: in_dim -> hidden_dims -> out_dim.

    Each hidden transition is: Linear -> ReLU -> [LayerNorm] -> [Dropout].
    The final Linear has no activation, norm, or dropout.
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
    """Compute L1/L2 regularization loss for an iterable of parameters."""
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
    Extract source and target modality data from a dataset base object.

    Preserves legacy flat keys for single-source architectures while also returning
    source data keyed by modality.
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
