"""Minimal eval helpers for MaskedLatentPretrainer.

Core question: can component j of modality M be reconstructed from all other
context (the remaining k-1 components of M plus the full other modality)? For
each j we do one deterministic forward with mask = one-hot(j) on M and zeros
everywhere else, then collect squared error at position j across the loader.

With zscore_pca_scores=True, target variance per component is ~1, so
R²_j = 1 - MSE_j.
"""

from __future__ import annotations

import time

import numpy as np
import matplotlib.pyplot as plt
import torch


@torch.no_grad()
def _concat_loader(loader, device, use_cov: bool):
    xs, ys, covs = [], [], []
    for batch in loader:
        xs.append(batch["x"] if "x" in batch else batch["x_modalities"])
        ys.append(batch["y"])
        if use_cov:
            c = batch.get("cov")
            if c is not None:
                covs.append(c)
    x = torch.cat(xs, dim=0).to(device)
    y = torch.cat(ys, dim=0).to(device)
    cov = torch.cat(covs, dim=0).to(device) if covs else None
    return x, y, cov


@torch.no_grad()
def per_component_target_var(model, loader, device, modality: str = "fc") -> np.ndarray:
    """Empirical variance of z-scored target per component on the given loader.
    Needed for honest R² — train ≈ 1.0 by construction, val may be well below 1.0."""
    assert modality in {"sc", "fc"}
    model.eval()
    device = next(model.parameters()).device
    use_cov = getattr(model, "use_covariates_cls", False)
    x, y, _ = _concat_loader(loader, device, use_cov)
    c = model.encode_target_latents(y) if modality == "fc" else model.encode_source_latents(x)
    return c.var(dim=0, unbiased=False).cpu().numpy()


@torch.no_grad()
def per_component_recon_sweep(model, loader, device, modality: str = "fc", verbose: bool = True) -> np.ndarray:
    """Mask component j alone on `modality`, everything else visible. Return MSE per j.

    Concatenates the whole loader into one tensor, then runs exactly k forwards
    (one per component index) on the full population. Avoids per-batch CUDA
    launch overhead, which dominates for tiny models.
    """
    assert modality in {"sc", "fc"}
    model.eval()
    k = model.n_components_pca
    use_cov = getattr(model, "use_covariates_cls", False)

    t0 = time.time()
    x, y, cov = _concat_loader(loader, device, use_cov)
    n = y.shape[0] if torch.is_tensor(y) else x.shape[0]
    t_concat = time.time() - t0
    if verbose:
        print(f"[sweep {modality}] device={device} n={n} k={k} forwards={k} "
              f"concat={t_concat:.2f}s", flush=True)

    sse = torch.zeros(k, device=device)
    zero = torch.zeros(n, k, dtype=torch.bool, device=device)
    m = torch.zeros(n, k, dtype=torch.bool, device=device)

    t_fwd0 = time.time()
    log_every = max(1, k // 8)
    for j in range(k):
        m.zero_()
        m[:, j] = True
        joint_mask = (m, zero) if modality == "sc" else (zero, m)
        c_s_hat, c_t_hat, c_s, c_t, _, _ = model.predict_reconstructions(
            x, y, cov=cov, joint_mask=joint_mask
        )
        hat, true = (c_s_hat, c_s) if modality == "sc" else (c_t_hat, c_t)
        sse[j] = (hat[:, j] - true[:, j]).pow(2).sum()
        if verbose and ((j + 1) % log_every == 0 or j == k - 1):
            print(f"[sweep {modality}] j={j + 1}/{k} elapsed={time.time() - t_fwd0:.2f}s", flush=True)
    total_fwd = time.time() - t_fwd0
    total = time.time() - t0
    if verbose:
        print(f"[sweep {modality}] done: n={n} total={total:.2f}s "
              f"fwd={total_fwd:.2f}s ({1000.0 * total_fwd / k:.2f} ms/fwd) "
              f"concat={t_concat:.2f}s", flush=True)
    return (sse / max(n, 1)).cpu().numpy()


@torch.no_grad()
def diagnose_split(model, loader, device, modality: str = "fc", split_name: str = "split"):
    """Print per-split sanity stats to resolve train/val pattern mysteries.

    Reports, per component:
      - target variance (should be ~1.0 on train if zscore_pca_scores=True; ~1.0 on val)
      - prediction variance (collapse to 0 if model is degenerate)
      - Pearson r between pred and target at the masked position
      - honest R² = 1 - MSE / Var(target)
    """
    assert modality in {"sc", "fc"}
    model.eval()
    device = next(model.parameters()).device
    k = model.n_components_pca
    use_cov = getattr(model, "use_covariates_cls", False)
    x, y, cov = _concat_loader(loader, device, use_cov)
    n = y.shape[0]

    c_all = model.encode_target_latents(y) if modality == "fc" else model.encode_source_latents(x)
    target_var = c_all.var(dim=0, unbiased=False).to(device)
    target_mean = c_all.mean(dim=0).to(device)

    zero = torch.zeros(n, k, dtype=torch.bool, device=device)
    m = torch.zeros(n, k, dtype=torch.bool, device=device)
    hat_cols = torch.zeros(n, k, device=device)
    true_cols = torch.zeros(n, k, device=device)
    for j in range(k):
        m.zero_(); m[:, j] = True
        joint_mask = (m, zero) if modality == "sc" else (zero, m)
        c_s_hat, c_t_hat, c_s, c_t, _, _ = model.predict_reconstructions(
            x, y, cov=cov, joint_mask=joint_mask
        )
        hat, true = (c_s_hat[:, j], c_s[:, j]) if modality == "sc" else (c_t_hat[:, j], c_t[:, j])
        hat_cols[:, j] = hat
        true_cols[:, j] = true
    diff = hat_cols - true_cols
    sse = diff.pow(2).sum(dim=0)
    pred_mean = hat_cols.mean(dim=0)
    pred_var = hat_cols.var(dim=0, unbiased=False)
    cov_pt = ((hat_cols - pred_mean) * (true_cols - true_cols.mean(dim=0))).mean(dim=0)

    mse = (sse / n).cpu().numpy()
    t_var = target_var.cpu().numpy()
    t_mean = target_mean.cpu().numpy()
    p_var = pred_var.cpu().numpy()
    p_mean = pred_mean.cpu().numpy()
    denom = (pred_var.sqrt() * target_var.sqrt()).clamp_min(1e-12)
    pearson = (cov_pt / denom).cpu().numpy()
    r2_naive = 1.0 - mse
    r2_honest = 1.0 - mse / np.maximum(t_var, 1e-12)

    def stats(a):
        return f"mean={a.mean():.4f} med={np.median(a):.4f} min={a.min():.4f} max={a.max():.4f}"

    print(f"[diagnose {split_name} {modality}] n={n} k={k}")
    print(f"  target: var {stats(t_var)} | mean {stats(t_mean)}")
    print(f"  pred  : var {stats(p_var)} | mean {stats(p_mean)}")
    print(f"  MSE   : {stats(mse)}")
    print(f"  Pearson r per comp: {stats(pearson)}")
    print(f"  R² naive (1-MSE)      : {stats(r2_naive)}")
    print(f"  R² honest (1-MSE/Var) : {stats(r2_honest)}")
    return {
        "mse": mse, "target_var": t_var, "target_mean": t_mean,
        "pred_var": p_var, "pred_mean": p_mean, "pearson": pearson,
        "r2_naive": r2_naive, "r2_honest": r2_honest,
    }


def plot_per_component_recon(results: dict, target_vars: dict | None = None, title: str | None = None):
    """results: {label: mse_array}. If target_vars is provided with matching labels,
    R² is computed honestly as 1 - MSE / Var(target_label). Otherwise assumes Var=1
    (only valid on train with zscore_pca_scores=True)."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for label, mse in results.items():
        var = target_vars[label] if target_vars and label in target_vars else np.ones_like(mse)
        r2 = 1.0 - mse / np.maximum(var, 1e-12)
        axes[0].plot(mse, "-o", ms=3, label=label, alpha=0.85)
        axes[1].plot(r2, "-o", ms=3, label=f"{label} (Var̄={var.mean():.2f})", alpha=0.85)
    axes[0].set_ylabel("masked MSE")
    axes[0].set_title(title or "Per-component reconstruction (mask 1, all else visible)")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
    axes[1].axhline(0.0, ls="--", c="k", lw=0.8)
    axes[1].set_ylabel("R² = 1 - MSE / Var(target)")
    axes[1].set_xlabel("component index")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig, axes
