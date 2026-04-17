"""Training visualization helpers.

Plots loss, correlation, variance, norm, and composite metric histories.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOSS_PAIRS = [
    ("train_loss", "val_loss", "Active Loss"),
    ("train_edge_mse", "val_edge_mse", "Edge MSE"),
    ("train_latent_mse", "val_latent_mse", "Latent MSE"),
    ("train_latent_weighted_mse", "val_latent_weighted_mse", "Latent Weighted MSE"),
]

CORR_PAIRS = [
    ("train_pearson_r", "val_pearson_r", "Pearson r"),
    ("train_demeaned_r", "val_demeaned_r", "Demeaned r"),
]


def _available_pairs(df, pairs):
    out = []
    for train_key, val_key, label in pairs:
        if train_key in df.columns and df[train_key].notna().any():
            out.append((train_key, val_key if val_key in df.columns else None, label))
        elif val_key in df.columns and df[val_key].notna().any():
            out.append((train_key if train_key in df.columns else None, val_key, label))
    return out


def plot_training_history_panels(history_df, figsize=None, marker_size=3, grid_alpha=0.3):
    """
    Plot training history with losses on the top row and correlation metrics below.

    Args:
        history_df: pandas DataFrame from run_out["train_result"].history_df
        figsize: optional (w, h); computed automatically when omitted
    Returns:
        fig, axes
    """
    if history_df is None:
        raise ValueError("history_df is None")
    df = history_df if isinstance(history_df, pd.DataFrame) else pd.DataFrame(history_df)
    if df.empty:
        raise ValueError("history_df is empty")
    if "epoch" not in df.columns:
        raise ValueError("history_df must contain an 'epoch' column")

    loss_pairs = _available_pairs(df, LOSS_PAIRS)
    corr_pairs = _available_pairs(df, CORR_PAIRS)
    ncols = max(len(loss_pairs), len(corr_pairs), 1)
    if figsize is None:
        figsize = (5 * ncols, 7)

    fig, axes = plt.subplots(2, ncols, figsize=figsize, squeeze=False)
    epochs = df["epoch"]

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


def summarize_history_columns(history_df):
    """Return the non-empty metric columns available for plotting/debugging."""
    df = history_df if isinstance(history_df, pd.DataFrame) else pd.DataFrame(history_df)
    cols = []
    for col in df.columns:
        if col == "epoch":
            continue
        if df[col].notna().any():
            cols.append(col)
    return cols

DEFAULT_HISTORY_LOSS_PAIRS = [
    ("train_loss", "val_loss", "Loss"),
]

REG_HISTORY_PAIR = ("train_reg_loss", "val_reg_loss", "Reg Loss")

LATENT_HISTORY_LOSS_PAIRS = [
    ("train_loss", "val_loss", "Active Loss"),
    ("train_edge_mse", "val_edge_mse", "Edge MSE"),
    ("train_var_match_loss", "val_var_match_loss", "MSE of Across-Subject Edge-Variance Profiles (Pred vs Target)"),
    ("train_latent_mse", "val_latent_mse", "Latent MSE"),
    ("train_latent_weighted_mse", "val_latent_weighted_mse", "Latent Weighted MSE"),
    ("train_joint_edge_term", "val_joint_edge_term", "Joint Edge Term"),
    ("train_joint_latent_term", "val_joint_latent_term", "Joint Latent Term"),
]

DEFAULT_HISTORY_CORR_PAIRS = [
    ("train_pearson_r", "val_pearson_r", "Pearson r"),
    ("train_demeaned_r", "val_demeaned_r", "Demeaned r"),
    ("train_variance_ratio", "val_variance_ratio", "Ratio of Mean Across-Subject Edge Variance (Pred/Target)"),
    ("train_norm_ratio", "val_norm_ratio", "Ratio of Mean Per-Subject Connectome L2 Norm (Pred/Target)"),
]


def _label_from_loss_term(term_name):
    mapping = {
        "mse": "MSE Term",
        "varmatch": "Var Match Term",
        "correye": "CorrEye Term",
        "neidist": "NeiDist Term",
    }
    return mapping.get(term_name, f"{term_name} Term")


def _composite_loss_term_pairs(df):
    term_names = []
    for col in df.columns:
        if col.startswith("train_loss_term_"):
            term_names.append(col.removeprefix("train_loss_term_"))
        elif col.startswith("val_loss_term_"):
            term_names.append(col.removeprefix("val_loss_term_"))
    seen = []
    for name in term_names:
        if name not in seen:
            seen.append(name)
    return [
        (f"train_loss_term_{name}" if f"train_loss_term_{name}" in df.columns else None,
         f"val_loss_term_{name}" if f"val_loss_term_{name}" in df.columns else None,
         _label_from_loss_term(name))
        for name in seen
    ]


def _available_history_pairs(df, pairs):
    out = []
    for train_key, val_key, label in pairs:
        if train_key in df.columns and df[train_key].notna().any():
            out.append((train_key, val_key if val_key in df.columns else None, label))
        elif val_key in df.columns and df[val_key].notna().any():
            out.append((train_key if train_key in df.columns else None, val_key, label))
    return out


def _set_robust_axis_limits(ax, series_list):
    """
    Set y-limits using a robust central range so one warmup/outlier epoch does not
    flatten the rest of the trajectory. If there are too few valid points, fall
    back to matplotlib autoscaling.
    """
    values = []
    for series in series_list:
        if series is None:
            continue
        arr = np.asarray(series, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            values.append(arr)
    if not values:
        return
    vals = np.concatenate(values)
    if vals.size < 4:
        return

    lo = np.quantile(vals, 0.05)
    hi = np.quantile(vals, 0.95)
    vmin = vals.min()
    vmax = vals.max()

    if not np.isfinite(lo) or not np.isfinite(hi):
        return
    if hi <= lo:
        span = max(abs(lo), 1e-6) * 0.05 + 1e-6
        ax.set_ylim(lo - span, hi + span)
        return

    robust_span = hi - lo
    full_span = max(vmax - vmin, 1e-12)
    # Only intervene when there is a meaningful outlier stretching the axis.
    if full_span <= 3.0 * robust_span:
        return

    pad = 0.08 * robust_span
    ax.set_ylim(lo - pad, hi + pad)


def summarize_history_columns(history_df):
    df = history_df if isinstance(history_df, pd.DataFrame) else pd.DataFrame(history_df)
    cols = []
    for col in df.columns:
        if col == "epoch":
            continue
        if df[col].notna().any():
            cols.append(col)
    return cols


def plot_training_history(
    history_df,
    style="default",
    figsize=None,
    marker_size=3,
    grid_alpha=0.3,
    skip_first_n_epochs=0,
):
    """
    Plot training history from TrainResult.history_df.

    style:
    - 'default': classic view with loss, Pearson r, and demeaned r
    - 'latent': losses on the top row and correlation metrics below

    skip_first_n_epochs:
    - if > 0, drop all rows with epoch < skip_first_n_epochs before plotting
    """
    if history_df is None:
        raise ValueError("history_df is None")
    df = history_df if isinstance(history_df, pd.DataFrame) else pd.DataFrame(history_df)
    if df.empty:
        raise ValueError("history_df is empty")
    if "epoch" not in df.columns:
        raise ValueError("history_df must contain an 'epoch' column")
    skip_first_n_epochs = int(skip_first_n_epochs)
    if skip_first_n_epochs < 0:
        raise ValueError(f"skip_first_n_epochs must be >= 0, got {skip_first_n_epochs}")
    if skip_first_n_epochs > 0:
        df = df[df["epoch"] >= skip_first_n_epochs].copy()
    if df.empty:
        raise ValueError(
            "No history remains after applying skip_first_n_epochs="
            f"{skip_first_n_epochs}."
        )

    epochs = df["epoch"]

    if style == "default":
        composite_pairs = _composite_loss_term_pairs(df)
        if composite_pairs:
            top_pairs = (
                _available_history_pairs(df, DEFAULT_HISTORY_LOSS_PAIRS)
                + _available_history_pairs(df, [REG_HISTORY_PAIR])
                + _available_history_pairs(df, composite_pairs)
            )
            corr_pairs = _available_history_pairs(df, DEFAULT_HISTORY_CORR_PAIRS)
            ncols = max(len(top_pairs), len(corr_pairs), 1)
            if figsize is None:
                figsize = (5 * ncols, 7)
            fig, axes = plt.subplots(2, ncols, figsize=figsize, squeeze=False)
            for ax_idx in range(ncols):
                if ax_idx < len(top_pairs):
                    train_key, val_key, title = top_pairs[ax_idx]
                    ax = axes[0, ax_idx]
                    plotted = []
                    if train_key is not None and train_key in df.columns and df[train_key].notna().any():
                        ax.plot(epochs, df[train_key], "b-o", label="Train", markersize=marker_size)
                        plotted.append(df[train_key])
                    if val_key is not None and val_key in df.columns and df[val_key].notna().any():
                        ax.plot(epochs, df[val_key], "r-o", label="Val", markersize=marker_size)
                        plotted.append(df[val_key])
                    _set_robust_axis_limits(ax, plotted)
                    ax.set_title(title)
                    ax.set_xlabel("Epoch")
                    ax.grid(True, alpha=grid_alpha)
                    ax.legend()
                else:
                    axes[0, ax_idx].axis("off")

                if ax_idx < len(corr_pairs):
                    train_key, val_key, title = corr_pairs[ax_idx]
                    ax = axes[1, ax_idx]
                    plotted = []
                    if train_key is not None and train_key in df.columns and df[train_key].notna().any():
                        ax.plot(epochs, df[train_key], "b-o", label="Train", markersize=marker_size)
                        plotted.append(df[train_key])
                    if val_key is not None and val_key in df.columns and df[val_key].notna().any():
                        ax.plot(epochs, df[val_key], "r-o", label="Val", markersize=marker_size)
                        plotted.append(df[val_key])
                    _set_robust_axis_limits(ax, plotted)
                    ax.set_title(title)
                    ax.set_xlabel("Epoch")
                    ax.grid(True, alpha=grid_alpha)
                    ax.legend()
                else:
                    axes[1, ax_idx].axis("off")
            fig.tight_layout()
            return fig, axes

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
            plotted = []
            if train_key is not None and train_key in df.columns and df[train_key].notna().any():
                ax.plot(epochs, df[train_key], "b-o", label="Train", markersize=marker_size)
                plotted.append(df[train_key])
            if val_key is not None and val_key in df.columns and df[val_key].notna().any():
                ax.plot(epochs, df[val_key], "r-o", label="Val", markersize=marker_size)
                plotted.append(df[val_key])
            _set_robust_axis_limits(ax, plotted)
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
                plotted = []
                if train_key is not None and train_key in df.columns and df[train_key].notna().any():
                    ax.plot(epochs, df[train_key], "b-o", label="Train", markersize=marker_size)
                    plotted.append(df[train_key])
                if val_key is not None and val_key in df.columns and df[val_key].notna().any():
                    ax.plot(epochs, df[val_key], "r-o", label="Val", markersize=marker_size)
                    plotted.append(df[val_key])
                _set_robust_axis_limits(ax, plotted)
                ax.set_title(title)
                ax.set_xlabel("Epoch")
                ax.grid(True, alpha=grid_alpha)
                ax.legend()
            else:
                axes[0, ax_idx].axis("off")

            if ax_idx < len(corr_pairs):
                train_key, val_key, title = corr_pairs[ax_idx]
                ax = axes[1, ax_idx]
                plotted = []
                if train_key is not None and train_key in df.columns and df[train_key].notna().any():
                    ax.plot(epochs, df[train_key], "b-o", label="Train", markersize=marker_size)
                    plotted.append(df[train_key])
                if val_key is not None and val_key in df.columns and df[val_key].notna().any():
                    ax.plot(epochs, df[val_key], "r-o", label="Val", markersize=marker_size)
                    plotted.append(df[val_key])
                _set_robust_axis_limits(ax, plotted)
                ax.set_title(title)
                ax.set_xlabel("Epoch")
                ax.grid(True, alpha=grid_alpha)
                ax.legend()
            else:
                axes[1, ax_idx].axis("off")
        fig.tight_layout()
        return fig, axes

    raise ValueError("Unknown plot style. Choose from {'default', 'latent'}.")
