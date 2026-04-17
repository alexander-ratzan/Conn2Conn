"""Training losses and scalar metrics.

Contains loss factories, composite losses, and batch-level metric helpers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


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


def compute_pairwise_row_correlation(x, y, eps=1e-10):
    """
    Compute pairwise row correlations between x and y.

    Returns a matrix where entry (i, j) is the correlation between row i of x
    and row j of y.
    """
    x_centered = x - x.mean(dim=1, keepdim=True)
    y_centered = y - y.mean(dim=1, keepdim=True)
    x_norm = torch.sqrt(torch.sum(x_centered ** 2, dim=1, keepdim=True) + eps)
    y_norm = torch.sqrt(torch.sum(y_centered ** 2, dim=1, keepdim=True) + eps)
    x_unit = x_centered / x_norm
    y_unit = y_centered / y_norm
    return torch.matmul(x_unit, y_unit.t())


def compute_correye_loss(y_pred, y_true):
    """
    Krakencoder-style identity matching loss on the subject-by-subject
    correlation matrix. Encourages own-subject matches to dominate.
    """
    if y_pred.shape[0] < 2:
        return y_pred.new_tensor(0.0)
    cc = compute_pairwise_row_correlation(y_true, y_pred)
    eye = torch.eye(cc.shape[0], device=cc.device, dtype=cc.dtype)
    return torch.norm(cc - eye)


def compute_neidist_loss(y_pred, y_true, margin=None):
    """
    Krakencoder-style nearest-neighbor distance loss.

    Encourages each prediction to be closer to its own target than to nearby
    competing targets from other subjects.
    """
    if y_pred.shape[0] < 2:
        return y_pred.new_tensor(0.0)
    d = torch.cdist(y_true, y_pred)
    dtrace = torch.trace(d)
    dself = dtrace / d.shape[0]
    dnei = d + torch.eye(d.shape[0], device=d.device, dtype=d.dtype) * d.max()
    dother = torch.mean((dnei.min(dim=0).values + dnei.min(dim=1).values) / 2.0)
    if margin is not None:
        dother = -torch.relu(torch.as_tensor(margin, device=d.device, dtype=d.dtype) - dother)
    return dself - dother


class CompositeLoss(nn.Module):
    """
    Sum weighted edge-space loss terms with optional normalization.

    Supported loss_terms formats:
    - ["mse", "neidist"]
    - "mse+neidist"
    - [{"name": "mse", "weight": 1.0}, {"name": "neidist", "weight": 0.25}]
    """

    VALID_TERMS = ("mse", "varmatch", "correye", "neidist")
    VALID_NORMALIZE = ("ema", "none")

    def __init__(self, loss_terms, normalize="ema", ema_decay=0.95, warmup_steps=100):
        super().__init__()
        term_specs = self._parse_loss_terms(loss_terms)
        if not term_specs:
            raise ValueError("CompositeLoss requires a non-empty loss_terms list.")

        invalid = [spec["name"] for spec in term_specs if spec["name"] not in self.VALID_TERMS]
        if invalid:
            raise ValueError(
                f"Unknown composite loss terms {invalid}. Valid options: {list(self.VALID_TERMS)}"
            )

        normalize = str(normalize or "ema").strip().lower()
        if normalize not in self.VALID_NORMALIZE:
            raise ValueError(f"Unknown loss_normalize='{normalize}'. Valid options: {list(self.VALID_NORMALIZE)}")

        deduped = OrderedDict()
        for spec in term_specs:
            deduped.setdefault(spec["name"], spec)

        self.term_specs = list(deduped.values())
        self.loss_terms = [spec["name"] for spec in self.term_specs]
        self.term_weights = OrderedDict((spec["name"], float(spec["weight"])) for spec in self.term_specs)
        self.normalize = normalize
        self.ema_decay = float(ema_decay)
        self.warmup_steps = int(warmup_steps)
        self.register_buffer("_loss_scales", torch.ones(len(self.loss_terms), dtype=torch.float32))
        self.register_buffer("_loss_scale_initialized", torch.zeros(len(self.loss_terms), dtype=torch.bool))
        self.register_buffer("_loss_scale_updates", torch.tensor(0, dtype=torch.long))
        self.last_raw_terms = OrderedDict()
        self.last_norm_terms = OrderedDict()
        self.last_weighted_terms = OrderedDict()

    @staticmethod
    def _parse_loss_terms(loss_terms):
        if isinstance(loss_terms, str):
            return [{"name": t.strip(), "weight": 1.0, "kwargs": {}} for t in loss_terms.split("+") if t.strip()]

        parsed = []
        for term in loss_terms or []:
            if isinstance(term, dict):
                if "name" not in term:
                    raise ValueError(f"Composite loss term dict is missing required 'name': {term}")
                parsed.append(
                    {
                        "name": str(term["name"]).strip(),
                        "weight": float(term.get("weight", 1.0)),
                        "kwargs": dict(term.get("kwargs") or {}),
                    }
                )
            else:
                name = str(term).strip()
                if name:
                    parsed.append({"name": name, "weight": 1.0, "kwargs": {}})
        return parsed

    def _compute_raw_term(self, spec, y_pred, y_true):
        name = spec["name"]
        kwargs = spec.get("kwargs") or {}
        if name == "mse":
            return F.mse_loss(y_pred, y_true)
        if name == "varmatch":
            return compute_var_match_loss(y_pred, y_true, axis=0, relative_to_true=True)
        if name == "correye":
            return compute_correye_loss(y_pred, y_true)
        if name == "neidist":
            return compute_neidist_loss(y_pred, y_true, margin=kwargs.get("margin"))
        raise ValueError(f"Unsupported composite loss term: {name}")

    def _maybe_update_scales(self, raw_terms):
        if self.normalize != "ema":
            return
        if not self.training:
            return
        if self._loss_scale_updates.item() >= self.warmup_steps:
            return
        for idx, raw_val in enumerate(raw_terms):
            raw_val = torch.clamp(raw_val.detach(), min=1e-8).to(self._loss_scales.device)
            if not bool(self._loss_scale_initialized[idx].item()):
                self._loss_scales[idx] = raw_val
                self._loss_scale_initialized[idx] = True
            else:
                decay = self.ema_decay
                self._loss_scales[idx].mul_(decay).add_(raw_val * (1.0 - decay))
        self._loss_scale_updates.add_(1)

    def get_scale_dict(self):
        return OrderedDict(
            (name, self._loss_scales[idx].detach())
            for idx, name in enumerate(self.loss_terms)
        )

    def forward(self, y_pred, y_true, **kwargs):
        raw_terms = [self._compute_raw_term(spec, y_pred, y_true) for spec in self.term_specs]
        self._maybe_update_scales(raw_terms)
        eps = 1e-8
        norm_terms = []
        weighted_terms = []
        for idx, raw_val in enumerate(raw_terms):
            if self.normalize == "ema":
                ref = torch.clamp(self._loss_scales[idx].detach().to(raw_val.device), min=eps)
                norm_val = raw_val / ref
            else:
                norm_val = raw_val
            norm_terms.append(norm_val)
            weighted_terms.append(raw_val.new_tensor(self.term_specs[idx]["weight"]) * norm_val)

        self.last_raw_terms = OrderedDict(
            (name, value.detach()) for name, value in zip(self.loss_terms, raw_terms)
        )
        self.last_norm_terms = OrderedDict(
            (name, value.detach()) for name, value in zip(self.loss_terms, norm_terms)
        )
        self.last_weighted_terms = OrderedDict(
            (name, value.detach()) for name, value in zip(self.loss_terms, weighted_terms)
        )
        return torch.stack(weighted_terms).sum()


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
    loss_terms=None,
    loss_normalize="ema",
    loss_scale_ema_decay=0.95,
    loss_scale_warmup_steps=20,
):
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: one of 'mse', 'weighted_mse', 'vae', 'sarwar_mse_corr',
            'joint_mse_varmatch', 'composite'
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
    elif loss_type == "composite":
        return CompositeLoss(
            loss_terms=loss_terms,
            normalize=loss_normalize,
            ema_decay=loss_scale_ema_decay,
            warmup_steps=loss_scale_warmup_steps,
        )
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. Choose from "
            "'mse', 'weighted_mse', 'vae', 'sarwar_mse_corr', 'joint_mse_varmatch', 'composite'"
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
