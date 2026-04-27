"""Minimal eval helpers for masked-latent pretrainers.

Core question: can component j of modality M be reconstructed from all other
context (the remaining k-1 components of M plus the full other modality)? For
each j we do one deterministic forward with mask = one-hot(j) on M and zeros
everywhere else, then collect squared error at position j across the loader.
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
    use_cov = bool(getattr(model, "use_covariates_cls", False) or getattr(model, "use_covariates", False))
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
    use_cov = bool(getattr(model, "use_covariates_cls", False) or getattr(model, "use_covariates", False))

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
      - target: mean ± std (std ~1.0 on train if zscore_pca_scores=True)
      - pred  : mean ± std (collapse to ~0 / low std if model is degenerate)
      - masked MSE
      - Pearson r between pred and target at the masked position
    """
    assert modality in {"sc", "fc"}
    model.eval()
    device = next(model.parameters()).device
    k = model.n_components_pca
    use_cov = bool(getattr(model, "use_covariates_cls", False) or getattr(model, "use_covariates", False))
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
    t_std = target_var.clamp_min(0).sqrt().cpu().numpy()
    t_mean = target_mean.cpu().numpy()
    p_std = pred_var.clamp_min(0).sqrt().cpu().numpy()
    p_mean = pred_mean.cpu().numpy()
    denom = (pred_var.sqrt() * target_var.sqrt()).clamp_min(1e-12)
    pearson = (cov_pt / denom).cpu().numpy()

    def agg(a):
        return f"{a.mean():+.3f} ± {a.std():.3f} (min {a.min():+.3f}, max {a.max():+.3f})"

    print(f"[diagnose {split_name} {modality}] n={n} k={k}")
    print(f"  target mean  : {agg(t_mean)}")
    print(f"  target std   : {agg(t_std)}")
    print(f"  pred mean    : {agg(p_mean)}")
    print(f"  pred std     : {agg(p_std)}")
    print(f"  masked MSE   : {agg(mse)}")
    print(f"  Pearson r    : {agg(pearson)}")
    return {
        "mse": mse, "target_std": t_std, "target_mean": t_mean,
        "pred_std": p_std, "pred_mean": p_mean, "pearson": pearson,
    }


def plot_per_component_recon(results: dict, title: str | None = None):
    """Plot per-component masked MSE. results: {label: mse_array}."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for label, mse in results.items():
        ax.plot(mse, "-o", ms=3, label=label, alpha=0.85)
    ax.set_ylabel("masked MSE")
    ax.set_xlabel("component index")
    ax.set_title(title or "Per-component reconstruction (mask 1, all else visible)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


@torch.no_grad()
def component_recon_stats(model, loader, device, modality: str = "fc", verbose: bool = True):
    """Return per-component masked reconstruction stats for one split.

    For each component j, masks j alone on the selected modality and computes:
      - mse: mean squared error on the masked component
      - pearson: Pearson r between predictions and targets on that component
      - pred / true arrays for optional scatter inspection
    """
    assert modality in {"sc", "fc"}
    model.eval()
    k = model.n_components_pca
    use_cov = bool(getattr(model, "use_covariates_cls", False) or getattr(model, "use_covariates", False))

    t0 = time.time()
    x, y, cov = _concat_loader(loader, device, use_cov)
    n = y.shape[0] if torch.is_tensor(y) else x.shape[0]
    if verbose:
        print(f"[component-stats {modality}] device={device} n={n} k={k}", flush=True)

    zero = torch.zeros(n, k, dtype=torch.bool, device=device)
    m = torch.zeros(n, k, dtype=torch.bool, device=device)
    pred_cols = torch.zeros(n, k, device=device)
    true_cols = torch.zeros(n, k, device=device)
    mse = torch.zeros(k, device=device)
    pearson = torch.zeros(k, device=device)
    log_every = max(1, k // 8)

    for j in range(k):
        m.zero_()
        m[:, j] = True
        joint_mask = (m, zero) if modality == "sc" else (zero, m)
        c_s_hat, c_t_hat, c_s, c_t, _, _ = model.predict_reconstructions(
            x, y, cov=cov, joint_mask=joint_mask
        )
        hat = c_s_hat[:, j] if modality == "sc" else c_t_hat[:, j]
        true = c_s[:, j] if modality == "sc" else c_t[:, j]
        pred_cols[:, j] = hat
        true_cols[:, j] = true

        diff = hat - true
        mse[j] = diff.pow(2).mean()

        hat_centered = hat - hat.mean()
        true_centered = true - true.mean()
        denom = hat_centered.pow(2).mean().sqrt() * true_centered.pow(2).mean().sqrt()
        pearson[j] = (hat_centered * true_centered).mean() / denom.clamp_min(1e-12)
        if verbose and ((j + 1) % log_every == 0 or j == k - 1):
            print(f"[component-stats {modality}] j={j + 1}/{k} elapsed={time.time() - t0:.2f}s", flush=True)

    if verbose:
        print(f"[component-stats {modality}] done total={time.time() - t0:.2f}s", flush=True)
    return {
        "mse": mse.cpu().numpy(),
        "pearson": pearson.cpu().numpy(),
        "pred": pred_cols.cpu().numpy(),
        "true": true_cols.cpu().numpy(),
    }


@torch.no_grad()
def collect_component_stats(model, loaders: dict, device, modalities=("sc", "fc"), verbose: bool = True):
    """Collect per-component stats for multiple splits and modalities.

    loaders: {"train": train_loader, "val": val_loader, ...}
    Returns stats[split][modality] = component_recon_stats(...)
    """
    stats = {}
    for split_name, loader in loaders.items():
        stats[split_name] = {}
        for modality in modalities:
            stats[split_name][modality] = component_recon_stats(
                model, loader, device, modality=modality, verbose=verbose
            )
    return stats


def plot_component_metric_grid(stats: dict, metric: str = "mse", title: str | None = None):
    """Plot one metric across splits with SC/FC on separate subplots."""
    assert metric in {"mse", "pearson"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 4), sharex=True)
    for ax, modality in zip(axes, ("sc", "fc")):
        for split_name, split_stats in stats.items():
            if modality not in split_stats:
                continue
            ax.plot(split_stats[modality][metric], marker="o", ms=3, alpha=0.85, label=split_name)
        ax.set_title(modality.upper())
        ax.set_xlabel("component index")
        ax.set_ylabel("masked MSE" if metric == "mse" else "masked Pearson r")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()
    return fig, axes


def _select_component_index(stat_block: dict, metric: str, select: str):
    values = np.asarray(stat_block[metric])
    if select == "best":
        return int(np.nanargmax(values)) if metric == "pearson" else int(np.nanargmin(values))
    if select == "worst":
        return int(np.nanargmin(values)) if metric == "pearson" else int(np.nanargmax(values))
    raise ValueError(f"Unknown select='{select}'. Choose from 'best', 'worst'.")


def plot_best_worst_component_scatter(
    stats: dict,
    split: str = "val",
    rank_by: str = "mse",
    selects=("best", "worst"),
    modalities=("fc", "sc"),
    figsize=(11, 9),
    equal_aspect: bool = True,
):
    """Plot best/worst component prediction-vs-target scatters for selected split.

    Columns correspond to modalities in `modalities`; rows correspond to
    selections in `selects`, each ranked according to `rank_by`.
    """
    assert rank_by in {"mse", "pearson"}
    if split not in stats:
        raise KeyError(f"split='{split}' not found in stats.")

    fig, axes = plt.subplots(len(selects), len(modalities), figsize=figsize, sharex=False, sharey=False)
    axes = np.asarray(axes, dtype=object)
    if axes.ndim == 1:
        axes = axes.reshape(len(selects), len(modalities))

    for row, select in enumerate(selects):
        for col, modality in enumerate(modalities):
            ax = axes[row, col]
            stat_block = stats[split][modality]
            comp_idx = _select_component_index(stat_block, rank_by, select)
            x = np.asarray(stat_block["true"])[:, comp_idx]
            y = np.asarray(stat_block["pred"])[:, comp_idx]
            pearson = float(stat_block["pearson"][comp_idx])
            mse = float(stat_block["mse"][comp_idx])
            ax.scatter(x, y, s=18, alpha=0.65)
            lo = float(min(x.min(), y.min()))
            hi = float(max(x.max(), y.max()))
            ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1, alpha=0.7)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            if equal_aspect:
                ax.set_aspect("equal", adjustable="box")
            ax.set_title(
                f"{select.upper()} {modality.upper()} comp {comp_idx} | "
                f"MSE={mse:.3f} r={pearson:.3f}"
            )
            ax.set_xlabel("true latent value")
            ax.set_ylabel("pred latent value")
            ax.grid(alpha=0.25)
    select_label = "/".join(s.upper() for s in selects)
    fig.suptitle(f"{split} split {select_label} masked-component scatter ranked by {rank_by}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, axes


def _lighten_color(color, amount: float):
    rgb = np.array(plt.matplotlib.colors.to_rgb(color))
    white = np.ones_like(rgb)
    return tuple(rgb + (white - rgb) * amount)


def plot_cross_model_component_metric(
    model_stats: dict,
    metric: str = "mse",
    splits=("train", "test"),
    modalities=("sc", "fc"),
    title: str | None = None,
):
    """Overlay multiple models on the same component curves.

    model_stats: {model_label: stats_dict returned by collect_component_stats}
    Uses one base color per model and light/dark variants for different splits.
    """
    assert metric in {"mse", "pearson"}
    fig, axes = plt.subplots(1, len(modalities), figsize=(16, 4), sharex=True)
    if len(modalities) == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")
    base_colors = [cmap(i % 10) for i in range(len(model_stats))]
    if len(splits) <= 1:
        split_lighten = np.array([0.0], dtype=float)
    else:
        split_lighten = np.linspace(0.45, 0.0, len(splits))

    for model_idx, (model_label, stats) in enumerate(model_stats.items()):
        base = base_colors[model_idx]
        for split_idx, split in enumerate(splits):
            if split not in stats:
                continue
            color = _lighten_color(base, float(split_lighten[split_idx]))
            for ax, modality in zip(axes, modalities):
                if modality not in stats[split]:
                    continue
                values = stats[split][modality][metric]
                ax.plot(
                    values,
                    marker="o",
                    ms=3,
                    linewidth=1.8,
                    alpha=0.95,
                    color=color,
                    label=f"{model_label} ({split})",
                )

    for ax, modality in zip(axes, modalities):
        ax.set_title(modality.upper())
        ax.set_xlabel("component index")
        ax.set_ylabel("masked MSE" if metric == "mse" else "masked Pearson r")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()
    return fig, axes
