"""Training losses and scalar metrics.

Contains loss factories, composite losses, and batch-level metric helpers.
"""
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

from models.registry import LOSS_WEIGHT_PREFIX, LOSS_KWARG_PREFIX


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


def compute_demeaned_mse_loss(y_pred, y_true, target_mean):
    """MSE after subtracting the training-set target mean from both sides."""
    return F.mse_loss(y_pred - target_mean, y_true - target_mean)


def compute_pairwise_corr_loss(y_pred, corr_target=0.4, eps=1e-8):
    """
    Sarwar et al. inter-subject correlation penalty:
    |mean off-diagonal correlation between predicted subjects - corr_target|.
    """
    bsz = y_pred.shape[0]
    if bsz < 2:
        return y_pred.new_tensor(0.0)
    centered = y_pred - y_pred.mean(dim=1, keepdim=True)
    norms = torch.sqrt(torch.sum(centered * centered, dim=1, keepdim=True) + eps)
    normalized = centered / norms
    corr_mat = normalized @ normalized.t()
    off_diag_sum = corr_mat.sum() - torch.diagonal(corr_mat).sum()
    return torch.abs(off_diag_sum / (bsz * (bsz - 1)) - float(corr_target))


def compute_kld_loss(mu, logvar):
    """KL divergence of the per-sample Gaussian latent to a standard normal (batch mean)."""
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


class CompositeLoss(nn.Module):
    """
    Sum weighted edge-space loss terms with optional normalization.

    Supported loss_terms formats:
    - ["mse", "neidist"]
    - "mse+neidist"
    - [{"name": "mse", "weight": 1.0}, {"name": "neidist", "weight": 0.25}]

    Terms: mse, varmatch, correye, neidist (kwarg margin), demeaned_mse (needs target_mean),
    pairwise_corr (kwarg corr_target; Sarwar et al.), kld (needs the model's mu/logvar).

    Any term accepts kwarg `scale` (> 0, default 1): a fixed reference scale, so the term contributes
    weight * raw / scale (spec v2 E1.1). Fixed scales replace EMA normalization; the two cannot be combined.
    `monitor_terms` are computed each step under no_grad and logged only (last_monitor_terms); they never
    enter the loss or its gradients.
    """

    MONITOR_TERMS = ("mse", "varmatch", "correye", "neidist", "demeaned_mse", "pairwise_corr")

    VALID_TERMS = ("mse", "varmatch", "correye", "neidist", "demeaned_mse", "pairwise_corr", "kld")
    VALID_NORMALIZE = ("ema", "none")

    def __init__(self, loss_terms, normalize="ema", ema_decay=0.95, warmup_steps=100, target_mean=None,
                 monitor_terms=None):
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
        self.term_scales = []
        for spec in self.term_specs:
            scale = float(spec["kwargs"].get("scale", 1.0))
            if not scale > 0:
                raise ValueError(f"Composite term '{spec['name']}' has scale={scale}; scale must be > 0.")
            self.term_scales.append(scale)
        if normalize == "ema" and any(scale != 1.0 for scale in self.term_scales):
            raise ValueError("Fixed term scales replace EMA normalization; use loss_normalize 'none' (or 'auto').")
        active = {spec["name"] for spec in self.term_specs}
        self.monitor_terms = [name for name in dict.fromkeys(monitor_terms or []) if name not in active]
        bad = [name for name in self.monitor_terms if name not in self.MONITOR_TERMS]
        if bad:
            raise ValueError(f"Unsupported monitor terms {bad}. Valid options: {list(self.MONITOR_TERMS)}")
        if target_mean is not None:
            if isinstance(target_mean, np.ndarray):
                target_mean = torch.tensor(target_mean, dtype=torch.float32)
            self.register_buffer("target_mean", torch.as_tensor(target_mean, dtype=torch.float32))
        else:
            self.target_mean = None
        if "demeaned_mse" in active | set(self.monitor_terms) and self.target_mean is None:
            raise ValueError("Composite term 'demeaned_mse' requires target_mean (pass base to create_loss_fn).")
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
        self.last_monitor_terms = OrderedDict()

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

    def _compute_raw_term(self, spec, y_pred, y_true, mu=None, logvar=None):
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
        if name == "demeaned_mse":
            return compute_demeaned_mse_loss(y_pred, y_true, self.target_mean)
        if name == "pairwise_corr":
            return compute_pairwise_corr_loss(y_pred, corr_target=kwargs.get("corr_target", 0.4))
        if name == "kld":
            if mu is None or logvar is None:
                raise ValueError("Composite term 'kld' requires the model to return (y_pred, mu, logvar).")
            return compute_kld_loss(mu, logvar)
        raise ValueError(f"Unsupported composite loss term: {name}")

    def _maybe_update_scales(self, raw_terms):
        if self.normalize != "ema":
            return
        if not self.training:
            return
        if self._loss_scale_updates.item() >= self.warmup_steps:
            return
        for idx, raw_val in enumerate(raw_terms):
            # Scale by magnitude: signed terms (neidist = d_self - d_other) go negative once
            # predictions are identifiable, and clamping the signed value would pin the scale
            # at the floor and inflate the term by ~1e8.
            raw_val = torch.clamp(raw_val.detach().abs(), min=1e-8).to(self._loss_scales.device)
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

    def forward(self, y_pred, y_true, mu=None, logvar=None, **kwargs):
        raw_terms = [self._compute_raw_term(spec, y_pred, y_true, mu=mu, logvar=logvar) for spec in self.term_specs]
        self._maybe_update_scales(raw_terms)
        eps = 1e-8
        norm_terms = []
        weighted_terms = []
        for idx, raw_val in enumerate(raw_terms):
            if self.normalize == "ema":
                ref = torch.clamp(self._loss_scales[idx].detach().to(raw_val.device), min=eps)
                norm_val = raw_val / ref
            elif self.term_scales[idx] != 1.0:
                norm_val = raw_val / self.term_scales[idx]
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
        total = torch.stack(weighted_terms).sum()
        if self.monitor_terms:
            with torch.no_grad():
                self.last_monitor_terms = OrderedDict(
                    (name, self._compute_raw_term({"name": name, "kwargs": {}}, y_pred.detach(), y_true).detach())
                    for name in self.monitor_terms
                )
        return total


EDGE_LOSS_TYPES = ("composite",)
LATENT_LOSS_TYPES = ("latent_mse", "latent_weighted_mse")

# Every trainer-config key that shapes the training loss, with its default.
LOSS_CONFIG_DEFAULTS = {
    "loss_type": "composite",
    "loss_terms": None,   # composite default: ["mse"] (plain MSE)
    "loss_normalize": "auto",
    "loss_scale_ema_decay": 0.95,
    "loss_scale_warmup_steps": 20,
    "loss_monitor_terms": None,   # composite only: extra terms computed and logged, never optimized
}


def loss_signature(loss_cfg):
    """
    Short string describing what a resolved loss config optimizes, e.g. "mse" or
    "mse+0.5*neidist". Used to group runs by objective in W&B.
    """
    loss_type = loss_cfg["loss_type"]
    if loss_type == "composite":
        deduped = OrderedDict()
        for spec in CompositeLoss._parse_loss_terms(loss_cfg["loss_terms"]):
            deduped.setdefault(spec["name"], spec)
        parts = []
        for name, spec in deduped.items():
            weight = float(spec["weight"])
            parts.append(name if weight == 1.0 else f"{weight:g}*{name}")
        return "+".join(parts)
    return loss_type


LOSS_NORMALIZE_MODES = ("auto",) + CompositeLoss.VALID_NORMALIZE


def _apply_loss_term_overrides(specs, trainer_cfg):
    """
    Apply flat `loss_weight_<term>` / `loss_kwarg_<term>__<name>` keys to parsed composite
    specs. These are the Tune-searchable form of loss_terms weights and term kwargs.
    Returns (specs, changed).
    """
    by_name = OrderedDict()
    for spec in specs:
        by_name.setdefault(spec["name"], spec)
    changed = False
    for key in sorted(trainer_cfg):
        value = trainer_cfg[key]
        if key.startswith(LOSS_WEIGHT_PREFIX):
            term = key[len(LOSS_WEIGHT_PREFIX):]
            if term not in by_name:
                raise ValueError(
                    f"{key} sets a weight for '{term}', which is not in loss_terms {list(by_name)}."
                )
            if term == "mse":
                warnings.warn(
                    f"{key} is set; the mse weight is the anchor (1.0) and is not meant to be searched.",
                    stacklevel=3,
                )
            by_name[term] = {**by_name[term], "weight": float(value)}
            changed = True
        elif key.startswith(LOSS_KWARG_PREFIX):
            term, sep, kwarg = key[len(LOSS_KWARG_PREFIX):].partition("__")
            if not sep or not kwarg:
                raise ValueError(f"{key} must look like {LOSS_KWARG_PREFIX}<term>__<kwarg>.")
            if term not in by_name:
                raise ValueError(
                    f"{key} sets a kwarg for '{term}', which is not in loss_terms {list(by_name)}."
                )
            by_name[term] = {**by_name[term], "kwargs": {**by_name[term]["kwargs"], kwarg: value}}
            changed = True
    return list(by_name.values()), changed


def resolve_loss_config(trainer_cfg=None):
    """
    Collect and validate the loss keys of a trainer config.

    Missing keys take LOSS_CONFIG_DEFAULTS; unrelated trainer keys are ignored, so a full
    trainer section (nested) or a flat Tune config can be passed directly. For composite
    losses, flat `loss_weight_<term>` / `loss_kwarg_<term>__<name>` keys override loss_terms,
    a weight of 0 drops the term, and loss_normalize='auto' resolves to 'none' for a single
    active term (exact plain loss) and 'ema' otherwise. Returns a new dict with every
    LOSS_CONFIG_DEFAULTS key (normalize resolved) plus `loss_signature`. Idempotent.
    """
    trainer_cfg = trainer_cfg or {}
    cfg = {key: trainer_cfg.get(key, default) for key, default in LOSS_CONFIG_DEFAULTS.items()}
    cfg["loss_normalize"] = str(cfg["loss_normalize"] or "ema").strip().lower()
    cfg["loss_scale_ema_decay"] = float(cfg["loss_scale_ema_decay"])
    cfg["loss_scale_warmup_steps"] = int(cfg["loss_scale_warmup_steps"])

    loss_type = cfg["loss_type"]
    if loss_type == "mse":
        raise ValueError(
            "loss_type 'mse' was retired: use loss_type 'composite' "
            "(its default loss_terms ['mse'] is exactly plain MSE)."
        )
    if loss_type not in EDGE_LOSS_TYPES + LATENT_LOSS_TYPES:
        raise ValueError(
            f"Unknown loss type: {loss_type}. Choose from {list(EDGE_LOSS_TYPES + LATENT_LOSS_TYPES)}"
        )
    if cfg["loss_normalize"] not in LOSS_NORMALIZE_MODES:
        raise ValueError(
            f"Unknown loss_normalize='{cfg['loss_normalize']}'. Valid options: {list(LOSS_NORMALIZE_MODES)}"
        )
    if loss_type == "composite":
        if cfg["loss_terms"] is None:
            cfg["loss_terms"] = ["mse"]
        specs = CompositeLoss._parse_loss_terms(cfg["loss_terms"])
        if not specs:
            raise ValueError("loss_type='composite' requires a non-empty loss_terms list.")
        invalid = [spec["name"] for spec in specs if spec["name"] not in CompositeLoss.VALID_TERMS]
        if invalid:
            raise ValueError(
                f"Unknown composite loss terms {invalid}. Valid options: {list(CompositeLoss.VALID_TERMS)}"
            )
        specs, changed = _apply_loss_term_overrides(specs, trainer_cfg)
        active = [spec for spec in specs if float(spec["weight"]) != 0.0]
        if not active:
            raise ValueError("Every composite loss term has weight 0; at least one term must be active.")
        if changed or len(active) < len(specs):
            # Rewrite only when something changed, so logged loss_terms keep their original form.
            cfg["loss_terms"] = [
                {"name": s["name"], "weight": float(s["weight"]), **({"kwargs": s["kwargs"]} if s["kwargs"] else {})}
                for s in active
            ]
        scaled = [s["name"] for s in active if float((s.get("kwargs") or {}).get("scale", 1.0)) != 1.0]
        if any(not float((s.get("kwargs") or {}).get("scale", 1.0)) > 0 for s in active):
            raise ValueError("Composite term scales must be > 0.")
        if cfg["loss_normalize"] == "auto":
            # Fixed reference scales are the normalization; otherwise ema balances multiple terms.
            cfg["loss_normalize"] = "none" if (len(active) == 1 or scaled) else "ema"
        elif cfg["loss_normalize"] == "ema" and scaled:
            raise ValueError(f"Terms {scaled} have fixed scales; they replace EMA. Use loss_normalize 'none' or 'auto'.")
        monitors = cfg["loss_monitor_terms"]
        if isinstance(monitors, str):
            monitors = [m.strip() for m in monitors.split("+") if m.strip()]
        active_names = {s["name"] for s in active}
        monitors = [m for m in dict.fromkeys(monitors or []) if m not in active_names]
        bad = [m for m in monitors if m not in CompositeLoss.MONITOR_TERMS]
        if bad:
            raise ValueError(f"Unsupported loss_monitor_terms {bad}. Valid options: {list(CompositeLoss.MONITOR_TERMS)}")
        cfg["loss_monitor_terms"] = monitors or None
    else:
        cfg["loss_monitor_terms"] = None
        if cfg["loss_normalize"] == "auto":
            cfg["loss_normalize"] = "none"
    cfg["loss_signature"] = loss_signature(cfg)
    return cfg


def create_loss_fn(loss_cfg, base=None):
    """
    Build the edge-space loss module for a resolved loss config (see resolve_loss_config).

    Args:
        loss_cfg: resolved loss config dict.
        base: Dataset base object (required for the demeaned_mse composite term).

    Returns:
        Loss function module. Latent loss types have no edge-space module and raise here;
        the Lightning module computes them from model latents instead.
    """
    loss_cfg = resolve_loss_config(loss_cfg)
    loss_type = loss_cfg["loss_type"]
    if loss_type == "composite":
        specs = CompositeLoss._parse_loss_terms(loss_cfg["loss_terms"])
        needs_mean = any(spec["name"] == "demeaned_mse" for spec in specs) or "demeaned_mse" in (loss_cfg["loss_monitor_terms"] or [])
        if needs_mean and base is None:
            raise ValueError("base is required for the demeaned_mse composite term")
        return CompositeLoss(
            loss_terms=loss_cfg["loss_terms"],
            normalize=loss_cfg["loss_normalize"],
            ema_decay=loss_cfg["loss_scale_ema_decay"],
            warmup_steps=loss_cfg["loss_scale_warmup_steps"],
            target_mean=get_target_train_mean(base) if needs_mean else None,
            monitor_terms=loss_cfg["loss_monitor_terms"],
        )
    raise ValueError(f"loss_type='{loss_type}' has no edge-space loss module.")


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
