import matplotlib.pyplot as plt
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
