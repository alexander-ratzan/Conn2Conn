"""
plots.py
========
Matplotlib figures for Conn2Conn experiment results.

Generic record-level plots (plot_source_metric_bars, plot_model_metric_scatter)
and cov_dl seed-DataFrame plots (plot_cov_dl_metric_bars,
plot_cov_dl_global_metric_panels). matplotlib is imported lazily inside each
function.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .records import (
    _DISPLAY_METRIC_LABELS,
    _METRIC_FEASIBLE_BOUNDS,
    _METRIC_OPT_DIRECTION,
    _display_model_label,
    _model_plot_color,
    _normalize_cov_type_value,
)
from .tables import _cov_type_mask


def plot_source_metric_bars(
    records: list,
    models: list,
    metric: str = "demeaned_pearson",
    sources: Optional[list] = None,
    seeds: Optional[list] = None,
    metric_label: Optional[str] = None,
    title: Optional[str] = None,
    include_reference_lines: bool = False,
    reference_lines: Optional[list] = None,
    include_reference_bars: Optional[bool] = None,
    reference_conditions: Optional[list] = None,
    figsize: tuple = (13, 6),
    bar_alpha: float = 0.95,
    show_title: bool = False,
    show_model_names: bool = True,
    title_fontsize: int = 24,
    label_fontsize: int = 20,
    tick_fontsize: int = 18,
    legend_fontsize: int = 18,
    legend_title_fontsize: int = 19,
    y_buffer_scale: float = 0.35,
    annotate_best: bool = False,
    annotation_fontsize: int = 16,
):
    """
    Model-colored bar chart with optional source-condition marker overlays.

    Every model occupies the same horizontal space. When multiple sources are
    provided, the plot keeps one bar per model and overlays source-specific
    markers at that model position.

    Parameters
    ----------
    records : list[RunRecord]
        Seed-level scraped records.
    models : list[str]
        Ordered model names to show on the x-axis.
    metric : str
        Key in RunRecord.test_metrics.
    sources : list[str] | None
        Ordered sources to show as grouped bars. Defaults to
        ["SC", "SC_r2t", "SC+SC_r2t"].
    seeds : list[int] | None
        Restrict aggregation to specific seeds.
    metric_label : str | None
        Pretty y-axis / title label. Defaults to `metric`.
    title : str | None
        Optional explicit title.
    show_title : bool
        When True, render a plot title. Defaults to False for cleaner
        notebook iteration.
    show_model_names : bool
        When True, render model names on the x-axis. Defaults to True.
    include_reference_lines : bool
        When True, render dashed horizontal reference lines (e.g.
        oracle/null) using `reference_lines`.
    reference_lines : list[dict] | None
        Optional reference-line specs. Each dict should include:
            {
              "label": "Oracle",
              "model": "CrossModalPCA",
              "source": "FC",
              "color": "#2ca25f",
              "linestyle": "--",
            }
        Each reference condition is aggregated once and drawn as a horizontal
        dashed line across the full plot.
    include_reference_bars : bool | None
        Deprecated alias retained for backward compatibility.
    reference_conditions : list[dict] | None
        Deprecated alias retained for backward compatibility.
    y_buffer_scale : float
        Extra y-axis padding as a fraction of the largest plotted std-dev.
    annotate_best : bool
        When True, annotate the best displayed model/source bar for the chosen
        metric.
    annotation_fontsize : int
        Font size for the best-bar annotation.

    Returns
    -------
    (fig, ax, plot_df)
        Matplotlib figure/axes plus a long-form DataFrame of plotted values.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if sources is None:
        sources = ["SC", "SC_r2t", "SC+SC_r2t"]
    metric_label = metric_label or _DISPLAY_METRIC_LABELS.get(metric, metric.replace("_", " ").title())
    source_markers = {
        "SC": "o",
        "SC_r2t": "*",
        "SC+SC_r2t": "^",
        "FC": "s",
    }

    default_reference_lines = [
        {
            "label": "Oracle",
            "model": "CrossModalPCA",
            "source": "FC",
            "color": "#2CA25F",
            "linestyle": "--",
        },
        {
            "label": "Null",
            "model": "CrossModalPCA",
            "source": "SC",
            "color": "#CB181D",
            "linestyle": "--",
        },
    ]
    if include_reference_bars is not None:
        include_reference_lines = include_reference_bars
    if reference_conditions is not None and reference_lines is None:
        reference_lines = reference_conditions
    reference_lines = reference_lines or default_reference_lines

    def _vals_for(model_name: str, source_name: str) -> list:
        vals = []
        for r in records:
            if r.model_name != model_name or r.source != source_name or r.status != "complete":
                continue
            if seeds is not None and r.seed not in seeds:
                continue
            val = r.test_metrics.get(metric)
            if val is None:
                continue
            vals.append(float(val))
        return vals

    rows = []
    for model_name in models:
        for source_name in sources:
            vals = _vals_for(model_name, source_name)
            rows.append(
                {
                    "display_model": model_name,
                    "series": source_name,
                    "kind": "source",
                    "mean": np.mean(vals) if vals else np.nan,
                    "std": np.std(vals) if vals else np.nan,
                    "n": len(vals),
                    "color": _model_plot_color(model_name),
                }
            )

    plot_df = pd.DataFrame(rows)

    display_models = [_display_model_label(m) for m in models]
    x = np.arange(len(models))
    bar_width = 0.58

    fig, ax = plt.subplots(figsize=figsize)
    plotted_y = []
    best_candidates = []
    baseline_source = "SC"
    for model_idx, model_name in enumerate(models):
        model_rows = plot_df[plot_df["display_model"] == model_name]
        baseline_row = model_rows[model_rows["series"] == baseline_source]
        if not baseline_row.empty and not pd.isna(baseline_row["mean"].iloc[0]):
            bar_height = float(baseline_row["mean"].iloc[0])
            bar_std = float(baseline_row["std"].iloc[0]) if not pd.isna(baseline_row["std"].iloc[0]) else 0.0
        else:
            bar_height = np.nan
            bar_std = 0.0
        if not pd.isna(bar_height):
            plotted_y.extend([bar_height - bar_std, bar_height + bar_std])
        ax.bar(
            x[model_idx],
            0.0 if pd.isna(bar_height) else bar_height,
            bar_width,
            yerr=bar_std if not pd.isna(bar_height) and bar_std > 0 else None,
            capsize=4,
            color=_model_plot_color(model_name),
            alpha=bar_alpha,
            edgecolor="white",
            linewidth=0.8,
            zorder=1,
        )
        for source_name in sources:
            sub = model_rows[model_rows["series"] == source_name]
            if sub.empty or pd.isna(sub["mean"].iloc[0]):
                continue
            mean_val = float(sub["mean"].iloc[0])
            std_val = float(sub["std"].iloc[0]) if not pd.isna(sub["std"].iloc[0]) else 0.0
            x_pos = float(x[model_idx])
            plotted_y.extend([mean_val - std_val, mean_val + std_val])
            best_candidates.append(
                {
                    "model": _display_model_label(model_name),
                    "series": source_name,
                    "mean": mean_val,
                    "std": std_val,
                    "x": x_pos,
                }
            )
            ax.scatter(
                x_pos,
                mean_val,
                s=180 if source_name == "SC_r2t" else 120,
                marker=source_markers.get(source_name, "o"),
                color=_model_plot_color(model_name),
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )

    if include_reference_lines:
        for ref in reference_lines:
            ref_vals = _vals_for(ref["model"], ref["source"])
            if not ref_vals:
                continue
            ref_mean = float(np.mean(ref_vals))
            ref_std = float(np.std(ref_vals)) if ref_vals else 0.0
            plotted_y.extend([ref_mean - ref_std, ref_mean + ref_std])
            ax.axhline(
                ref_mean,
                color=ref.get("color", "#2CA25F"),
                linestyle=ref.get("linestyle", "--"),
                linewidth=2.5,
                alpha=0.95,
                label=ref["label"],
            )

    ax.set_xticks(x)
    if show_model_names:
        ax.set_xticklabels(display_models, rotation=20, ha="right", fontsize=tick_fontsize)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.set_ylabel(metric_label, fontsize=label_fontsize)
    default_title = (
        f"{metric_label} by model ({sources[0]} input)"
        if len(sources) == 1
        else f"{metric_label} by model and input source"
    )
    if show_title:
        ax.set_title(title or default_title, fontsize=title_fontsize)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=source_markers.get(src, "o"),
            linestyle="None",
            markerfacecolor="#4C4C4C",
            markeredgecolor="white",
            markersize=12 if src == "SC_r2t" else 10,
            label=src,
        )
        for src in sources
    ]
    if include_reference_lines:
        for ref in reference_lines:
            legend_handles.append(
                Line2D([0], [0], color=ref.get("color", "#2CA25F"), linestyle=ref.get("linestyle", "--"), linewidth=2.5, label=ref["label"])
            )
    ax.legend(
        handles=legend_handles,
        title="Condition",
        frameon=False,
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
    )
    if plotted_y:
        y_min = float(np.nanmin(plotted_y))
        y_max = float(np.nanmax(plotted_y))
        max_std = float(plot_df["std"].dropna().max()) if not plot_df["std"].dropna().empty else 0.0
        buffer = max(max_std * y_buffer_scale, 0.02 * max(abs(y_max - y_min), 1.0))
        y_lo = y_min - buffer
        y_hi = y_max + buffer

        feasible_lo, feasible_hi = _METRIC_FEASIBLE_BOUNDS.get(metric, (None, None))
        if feasible_lo is not None:
            y_lo = max(y_lo, feasible_lo)
        if feasible_hi is not None:
            y_hi = min(y_hi, feasible_hi)

        if y_hi <= y_lo:
            if feasible_lo is not None and feasible_hi is not None:
                y_lo, y_hi = feasible_lo, feasible_hi
            else:
                midpoint = 0.5 * (y_lo + y_hi)
                y_lo, y_hi = midpoint - 0.5, midpoint + 0.5
        ax.set_ylim(y_lo, y_hi)
    if annotate_best and best_candidates:
        direction = _METRIC_OPT_DIRECTION.get(metric, "max")
        if direction == "min":
            best = min(best_candidates, key=lambda d: d["mean"])
        else:
            best = max(best_candidates, key=lambda d: d["mean"])
        y_lo, y_hi = ax.get_ylim()
        y_pad = 0.02 * (y_hi - y_lo)
        ax.text(
            best["x"],
            best["mean"] + best["std"] + y_pad,
            f'{best["mean"]:.3f}',
            ha="center",
            va="bottom",
            fontsize=annotation_fontsize,
            fontweight="semibold",
        )
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    plt.tight_layout()
    return fig, ax, plot_df


def plot_model_metric_scatter(
    records: list,
    models: list,
    source: str = "SC",
    x_metric: str = "avg_rank",
    y_metric: str = "demeaned_pearson",
    seeds: Optional[list] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    include_reference_points: bool = False,
    reference_points: Optional[list] = None,
    figsize: tuple = (7, 7),
    point_size: int = 190,
    title_fontsize: int = 24,
    label_fontsize: int = 20,
    tick_fontsize: int = 18,
    legend_fontsize: int = 14,
    legend_title_fontsize: int = 15,
    annotation_fontsize: int = 14,
    show_errorbars: bool = False,
    axis_buffer_scale: float = 0.35,
    show_legend: bool = True,
    annotate_points: bool = False,
):
    """
    Model-level point plot over two metrics.

    Defaults to the SC-input slice, matching the schematic for comparing model
    families on a common source. Oracle/null reference points can optionally be
    overlaid using CrossModalPCA with FC (oracle) and SC (null).
    """
    import matplotlib.pyplot as plt

    x_label = x_label or _DISPLAY_METRIC_LABELS.get(x_metric, x_metric.replace("_", " ").title())
    y_label = y_label or _DISPLAY_METRIC_LABELS.get(y_metric, y_metric.replace("_", " ").title())

    default_reference_points = [
        {
            "label": "Oracle",
            "model": "CrossModalPCA",
            "source": "FC",
            "color": "#2CA25F",
            "marker": "D",
        },
        {
            "label": "Null",
            "model": "CrossModalPCA",
            "source": "SC",
            "color": "#CB181D",
            "marker": "D",
        },
    ]
    reference_points = reference_points or default_reference_points

    def _vals_for(model_name: str, source_name: str, metric_name: str) -> list:
        vals = []
        for r in records:
            if r.model_name != model_name or r.source != source_name or r.status != "complete":
                continue
            if seeds is not None and r.seed not in seeds:
                continue
            val = r.test_metrics.get(metric_name)
            if val is None:
                continue
            vals.append(float(val))
        return vals

    rows = []
    for model_name in models:
        point_color = _model_plot_color(model_name)
        x_vals = _vals_for(model_name, source, x_metric)
        y_vals = _vals_for(model_name, source, y_metric)
        if not x_vals or not y_vals:
            rows.append(
                {
                    "label": model_name,
                    "model": model_name,
                    "source": source,
                    "kind": "model",
                    "x_mean": np.nan,
                    "y_mean": np.nan,
                    "x_std": np.nan,
                    "y_std": np.nan,
                    "n": 0,
                    "color": point_color,
                    "marker": "o",
                }
            )
            continue
        rows.append(
            {
                "label": _display_model_label(model_name),
                "model": model_name,
                "source": source,
                "kind": "model",
                "x_mean": float(np.mean(x_vals)),
                "y_mean": float(np.mean(y_vals)),
                "x_std": float(np.std(x_vals)),
                "y_std": float(np.std(y_vals)),
                "n": min(len(x_vals), len(y_vals)),
                "color": point_color,
                "marker": "o",
            }
        )

    if include_reference_points:
        for ref in reference_points:
            x_vals = _vals_for(ref["model"], ref["source"], x_metric)
            y_vals = _vals_for(ref["model"], ref["source"], y_metric)
            if not x_vals or not y_vals:
                continue
            rows.append(
                {
                    "label": ref["label"],
                    "model": ref["model"],
                    "source": ref["source"],
                    "kind": "reference",
                    "x_mean": float(np.mean(x_vals)),
                    "y_mean": float(np.mean(y_vals)),
                    "x_std": float(np.std(x_vals)),
                    "y_std": float(np.std(y_vals)),
                    "n": min(len(x_vals), len(y_vals)),
                    "color": ref.get("color", "#2CA25F"),
                    "marker": ref.get("marker", "D"),
                }
            )

    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=figsize)
    for _, row in plot_df.iterrows():
        if pd.isna(row["x_mean"]) or pd.isna(row["y_mean"]):
            continue
        if show_errorbars:
            ax.errorbar(
                row["x_mean"],
                row["y_mean"],
                xerr=row["x_std"],
                yerr=row["y_std"],
                fmt="none",
                ecolor=row["color"],
                elinewidth=1.8,
                alpha=0.7,
                capsize=3,
                zorder=1,
            )
        ax.scatter(
            row["x_mean"],
            row["y_mean"],
            s=point_size,
            color=row["color"],
            marker=row["marker"],
            edgecolor="white",
            linewidth=0.9,
            label=row["label"],
            zorder=2,
        )
        if annotate_points:
            ax.annotate(
                row["label"],
                (row["x_mean"], row["y_mean"]),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=annotation_fontsize,
                weight="semibold" if row["kind"] == "reference" else None,
            )

    x_vals_all = []
    y_vals_all = []
    for _, row in plot_df.iterrows():
        if pd.isna(row["x_mean"]) or pd.isna(row["y_mean"]):
            continue
        x_vals_all.extend([row["x_mean"] - row["x_std"], row["x_mean"] + row["x_std"]])
        y_vals_all.extend([row["y_mean"] - row["y_std"], row["y_mean"] + row["y_std"]])

    if x_vals_all:
        x_min = float(np.nanmin(x_vals_all))
        x_max = float(np.nanmax(x_vals_all))
        x_buffer = max(float(plot_df["x_std"].dropna().max()) * axis_buffer_scale if not plot_df["x_std"].dropna().empty else 0.0,
                       0.02 * max(abs(x_max - x_min), 1.0))
        x_lo = x_min - x_buffer
        x_hi = x_max + x_buffer
        feasible_lo, feasible_hi = _METRIC_FEASIBLE_BOUNDS.get(x_metric, (None, None))
        if feasible_lo is not None:
            x_lo = max(x_lo, feasible_lo)
        if feasible_hi is not None:
            x_hi = min(x_hi, feasible_hi)
        if x_hi <= x_lo:
            if feasible_lo is not None and feasible_hi is not None:
                x_lo, x_hi = feasible_lo, feasible_hi
            else:
                midpoint = 0.5 * (x_lo + x_hi)
                x_lo, x_hi = midpoint - 0.5, midpoint + 0.5
        ax.set_xlim(x_lo, x_hi)

    if y_vals_all:
        y_min = float(np.nanmin(y_vals_all))
        y_max = float(np.nanmax(y_vals_all))
        y_buffer = max(float(plot_df["y_std"].dropna().max()) * axis_buffer_scale if not plot_df["y_std"].dropna().empty else 0.0,
                       0.02 * max(abs(y_max - y_min), 1.0))
        y_lo = y_min - y_buffer
        y_hi = y_max + y_buffer
        feasible_lo, feasible_hi = _METRIC_FEASIBLE_BOUNDS.get(y_metric, (None, None))
        if feasible_lo is not None:
            y_lo = max(y_lo, feasible_lo)
        if feasible_hi is not None:
            y_hi = min(y_hi, feasible_hi)
        if y_hi <= y_lo:
            if feasible_lo is not None and feasible_hi is not None:
                y_lo, y_hi = feasible_lo, feasible_hi
            else:
                midpoint = 0.5 * (y_lo + y_hi)
                y_lo, y_hi = midpoint - 0.5, midpoint + 0.5
        ax.set_ylim(y_lo, y_hi)

    ax.set_xlabel(x_label, fontsize=label_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    ax.set_title(title or f"{y_label} vs. {x_label} ({source})", fontsize=title_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        uniq_handles, uniq_labels = [], []
        for h, l in zip(handles, labels):
            if l in seen:
                continue
            seen.add(l)
            uniq_handles.append(h)
            uniq_labels.append(l)
        ax.legend(
            uniq_handles,
            uniq_labels,
            title="Model",
            frameon=True,
            fontsize=legend_fontsize,
            title_fontsize=legend_title_fontsize,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            borderaxespad=0.0,
        )
    ax.grid(alpha=0.2, linestyle="--")
    plt.tight_layout()
    return fig, ax, plot_df


def plot_cov_dl_metric_bars(
    seed_df: pd.DataFrame,
    row_specs: list,
    metric: str = "pearson",
    title: Optional[str] = None,
    figsize: tuple = (12, 6),
    show_title: bool = False,
    show_axis_labels: bool = True,
    annotate_best: bool = True,
    title_fontsize: int = 24,
    label_fontsize: int = 20,
    tick_fontsize: int = 17,
    annotation_fontsize: int = 15,
    y_buffer_scale: float = 0.35,
):
    """Plot one bar per experiment-2 condition using model-consistent colors."""
    import matplotlib.pyplot as plt

    metric_label = _DISPLAY_METRIC_LABELS.get(metric, metric.replace("_", " ").title())

    rows = []
    for spec in row_specs:
        spec_cov = _normalize_cov_type_value(spec.get("cov_type"))
        subset = seed_df[
            (seed_df["model"] == spec["model"])
            & (seed_df["source"] == spec.get("source", ""))
            & _cov_type_mask(seed_df["cov_type"], spec_cov)
            & (seed_df["status"] == "complete")
        ].copy()
        col = f"test_{metric}"
        vals = subset[col].dropna().astype(float).values if col in subset.columns else np.array([])
        rows.append(
            {
                "plot_label": spec.get("plot_label", spec.get("display_model", _display_model_label(spec["model"]))),
                "model": spec["model"],
                "mean": float(vals.mean()) if len(vals) else np.nan,
                "std": float(vals.std()) if len(vals) else np.nan,
                "n": len(vals),
                "color": _model_plot_color(spec["model"]),
            }
        )

    plot_df = pd.DataFrame(rows)
    if not plot_df.empty:
        direction = _METRIC_OPT_DIRECTION.get(metric, "max")
        baseline_df = plot_df[plot_df["model"] == "CrossModal_PCA_PLS_learnable"].copy()
        remainder_df = plot_df[plot_df["model"] != "CrossModal_PCA_PLS_learnable"].copy()
        remainder_df["_sort_mean"] = remainder_df["mean"].fillna(
            -np.inf if direction == "max" else np.inf
        )
        remainder_df = remainder_df.sort_values(
            "_sort_mean",
            ascending=(direction == "min"),
            kind="mergesort",
        ).drop(columns="_sort_mean")
        plot_df = pd.concat([baseline_df, remainder_df], ignore_index=True)
    x = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=figsize)
    means = plot_df["mean"].fillna(0.0).astype(float).values
    stds = plot_df["std"].fillna(0.0).astype(float).values
    colors = plot_df["color"].tolist()
    ax.bar(
        x,
        means,
        yerr=stds,
        color=colors,
        capsize=4,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.95,
    )

    plotted_y = []
    for mean_val, std_val in zip(plot_df["mean"], plot_df["std"]):
        if not pd.isna(mean_val):
            plotted_y.extend([float(mean_val - (0.0 if pd.isna(std_val) else std_val)), float(mean_val + (0.0 if pd.isna(std_val) else std_val))])

    if plotted_y:
        y_min = float(np.nanmin(plotted_y))
        y_max = float(np.nanmax(plotted_y))
        max_std = float(plot_df["std"].dropna().max()) if not plot_df["std"].dropna().empty else 0.0
        buffer = max(max_std * y_buffer_scale, 0.02 * max(abs(y_max - y_min), 1.0))
        y_lo = y_min - buffer
        y_hi = y_max + buffer
        feasible_lo, feasible_hi = _METRIC_FEASIBLE_BOUNDS.get(metric, (None, None))
        if feasible_lo is not None:
            y_lo = max(y_lo, feasible_lo)
        if feasible_hi is not None:
            y_hi = min(y_hi, feasible_hi)
        if y_hi <= y_lo:
            if feasible_lo is not None and feasible_hi is not None:
                y_lo, y_hi = feasible_lo, feasible_hi
            else:
                midpoint = 0.5 * (y_lo + y_hi)
                y_lo, y_hi = midpoint - 0.5, midpoint + 0.5
        ax.set_ylim(y_lo, y_hi)

    if annotate_best and not plot_df["mean"].dropna().empty:
        direction = _METRIC_OPT_DIRECTION.get(metric, "max")
        best_idx = plot_df["mean"].idxmin() if direction == "min" else plot_df["mean"].idxmax()
        best_row = plot_df.loc[best_idx]
        y_lo, y_hi = ax.get_ylim()
        y_pad = 0.02 * (y_hi - y_lo)
        ax.text(
            x[best_idx],
            float(best_row["mean"]) + (0.0 if pd.isna(best_row["std"]) else float(best_row["std"])) + y_pad,
            f'{float(best_row["mean"]):.3f}',
            ha="center",
            va="bottom",
            fontsize=annotation_fontsize,
            fontweight="semibold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["plot_label"].tolist(), rotation=20, ha="right", fontsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    if show_axis_labels:
        ax.set_ylabel(metric_label, fontsize=label_fontsize)
    if show_title:
        ax.set_title(title or f"{metric_label} across projector / model conditions", fontsize=title_fontsize)
    ax.grid(axis="y", alpha=0.32, linestyle="--", linewidth=0.9)
    plt.tight_layout()
    return fig, ax, plot_df


def plot_cov_dl_global_metric_panels(
    seed_df: pd.DataFrame,
    row_specs: list,
    metrics: Optional[list] = None,
    title: Optional[str] = None,
    figsize: tuple = (14, 6),
    dpi: int = 180,
    show_title: bool = False,
    show_axis_labels: bool = True,
    show_legend: bool = True,
    title_fontsize: int = 20,
    label_fontsize: int = 16,
    tick_fontsize: int = 14,
    legend_fontsize: int = 12,
    legend_title_fontsize: int = 13,
    annotation_fontsize: int = 12,
    x_buffer_scale: float = 0.35,
    show_errorbars: bool = True,
):
    """Compact horizontal multi-panel comparison for experiment 2."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if metrics is None:
        metrics = ["pearson", "demeaned_pearson", "avg_rank", "top1_acc"]

    rows = []
    for spec in row_specs:
        row = {
            "plot_label": spec.get("plot_label", spec.get("display_model", _display_model_label(spec["model"]))),
            "model": spec["model"],
            "color": _model_plot_color(spec["model"]),
            "_input_order": len(rows),
        }
        spec_cov = _normalize_cov_type_value(spec.get("cov_type"))
        subset = seed_df[
            (seed_df["model"] == spec["model"])
            & (seed_df["source"] == spec.get("source", ""))
            & _cov_type_mask(seed_df["cov_type"], spec_cov)
            & (seed_df["status"] == "complete")
        ].copy()
        for metric in metrics:
            col = f"test_{metric}"
            vals = subset[col].dropna().astype(float).values if col in subset.columns else np.array([])
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_std"] = float(vals.std()) if len(vals) else np.nan
        rows.append(row)

    plot_df = pd.DataFrame(rows)
    if not plot_df.empty:
        model_priority = {
            "NodalGNN": 0,
            "Chen2024GCN": 1,
            "Sarwar2020MLP": 2,
            "Krakencoder_precomputed": 3,
            "CrossModal_PCA_PLS_learnable": 4,
            "CrossModal_PCA_PLS_CovProjector": 5,
        }
        plot_df["_model_priority"] = plot_df["model"].map(model_priority).fillna(100).astype(int)
        plot_df = plot_df.sort_values(
            ["_model_priority", "_input_order"],
            kind="mergesort",
        ).reset_index(drop=True)
        plot_df = plot_df.drop(columns=["_model_priority", "_input_order"])
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, dpi=dpi, sharey=True)
    if n_metrics == 1:
        axes = [axes]

    y = np.arange(len(plot_df))
    bar_height = 0.72

    for ax, metric in zip(axes, metrics):
        means = plot_df[f"{metric}_mean"].astype(float).values
        stds = plot_df[f"{metric}_std"].fillna(0.0).astype(float).values
        colors = plot_df["color"].tolist()
        ax.barh(
            y,
            np.nan_to_num(means, nan=0.0),
            xerr=(stds if show_errorbars else None),
            color=colors,
            capsize=4 if show_errorbars else 0,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.95,
            height=bar_height,
        )

        plotted_x = []
        for mean_val, std_val in zip(means, stds):
            if not np.isnan(mean_val):
                spread = 0.0 if not show_errorbars else std_val
                plotted_x.extend([float(mean_val - spread), float(mean_val + spread)])

        if plotted_x:
            x_min = float(np.nanmin(plotted_x))
            x_max = float(np.nanmax(plotted_x))
            max_std = float(np.nanmax(stds)) if len(stds) else 0.0
            buffer = max(max_std * x_buffer_scale, 0.02 * max(abs(x_max - x_min), 1.0))
            x_lo = x_min - buffer
            x_hi = x_max + buffer
            feasible_lo, feasible_hi = _METRIC_FEASIBLE_BOUNDS.get(metric, (None, None))
            if feasible_lo is not None:
                x_lo = max(x_lo, feasible_lo)
            if feasible_hi is not None:
                x_hi = min(x_hi, feasible_hi)
            if x_hi <= x_lo:
                if feasible_lo is not None and feasible_hi is not None:
                    x_lo, x_hi = feasible_lo, feasible_hi
                else:
                    midpoint = 0.5 * (x_lo + x_hi)
                    x_lo, x_hi = midpoint - 0.5, midpoint + 0.5
            ax.set_xlim(x_lo, x_hi)

        if show_axis_labels:
            ax.set_xlabel(_DISPLAY_METRIC_LABELS.get(metric, metric.replace("_", " ").title()), fontsize=label_fontsize)
        ax.tick_params(axis="x", labelsize=tick_fontsize)
        ax.grid(axis="x", alpha=0.32, linestyle="--", linewidth=0.9)
        ax.invert_yaxis()

        valid = plot_df[f"{metric}_mean"].dropna()
        if not valid.empty:
            direction = _METRIC_OPT_DIRECTION.get(metric, "max")
            best_idx = valid.idxmin() if direction == "min" else valid.idxmax()
            best_val = float(plot_df.loc[best_idx, f"{metric}_mean"])
            x_lo, x_hi = ax.get_xlim()
            x_pad = 0.01 * (x_hi - x_lo)
            ax.text(best_val + x_pad, y[best_idx], f"{best_val:.3f}", va="center", ha="left", fontsize=annotation_fontsize, fontweight="semibold")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(plot_df["plot_label"].tolist(), fontsize=tick_fontsize)
    for ax in axes[1:]:
        ax.tick_params(axis="y", left=False, labelleft=False)

    if show_legend and not plot_df.empty:
        legend_df = plot_df[["plot_label", "color"]].drop_duplicates()
        legend_handles = [
            Patch(facecolor=row.color, edgecolor="white", linewidth=0.8, label=row.plot_label)
            for row in legend_df.itertuples(index=False)
        ]
        fig.legend(
            handles=legend_handles,
            title="Model",
            loc="upper left",
            bbox_to_anchor=(0.865, 0.98),
            frameon=True,
            fontsize=legend_fontsize,
            title_fontsize=legend_title_fontsize,
            borderaxespad=0.0,
        )

    if show_title and title:
        fig.suptitle(title, fontsize=title_fontsize, y=0.98)
    plt.tight_layout(rect=(0.0, 0.0, 0.84 if show_legend else 1.0, 1.0))
    return fig, axes, plot_df


# ---------------------------------------------------------------------------
# Figure registry + export (config-driven experiment scripts)
# ---------------------------------------------------------------------------

# Figure types addressable from an experiment config's `figures:` list. Each entry's remaining
# keys are passed straight through as keyword arguments. Record-level types take `records`;
# cov_dl types take a row-spec seed DataFrame (`seed_df`, `row_specs`).
FIGURE_TYPES = {
    "source_bars": plot_source_metric_bars,
    "metric_scatter": plot_model_metric_scatter,
    "cov_dl_bars": plot_cov_dl_metric_bars,
    "cov_dl_panels": plot_cov_dl_global_metric_panels,
}


def render_figure(spec: dict, records: Optional[list] = None, **defaults):
    """
    Draw one figure from a config entry like {"type": "source_bars", "metric": "pearson"}.

    `records` and `defaults` (e.g. models, seeds, seed_df, row_specs) fill any argument the spec
    leaves unset and that the plotting function accepts. Unknown spec keys raise, so config typos
    fail loudly. Returns the plotting function's (fig, ax, plot_df).
    """
    import inspect

    spec = dict(spec)
    fig_type = spec.pop("type", None)
    spec.pop("name", None)
    if fig_type not in FIGURE_TYPES:
        raise ValueError(f"Unknown figure type {fig_type!r}; choose from {sorted(FIGURE_TYPES)}")
    fn = FIGURE_TYPES[fig_type]
    params = inspect.signature(fn).parameters
    unknown = sorted(set(spec) - set(params))
    if unknown:
        raise ValueError(f"{fig_type}: unknown argument(s) {unknown}; valid: {sorted(params)}")
    available = dict(defaults)
    if records is not None:
        available["records"] = records
    kwargs = {k: v for k, v in available.items() if k in params and k not in spec}
    kwargs.update(spec)
    return fn(**kwargs)


def figure_name(spec: dict) -> str:
    """Stable file stem for a figure spec: explicit `name`, else built from its settings."""
    if spec.get("name"):
        return str(spec["name"])
    parts = [spec.get("type", "figure")]
    if spec.get("type") == "metric_scatter":
        parts += [spec.get("y_metric", "demeaned_pearson"), "vs", spec.get("x_metric", "avg_rank"),
                  spec.get("source", "SC")]
    elif spec.get("type", "").startswith("cov_dl"):
        if spec.get("metric"):
            parts.append(spec["metric"])
        if spec.get("rows"):
            parts.append(spec["rows"])
    else:
        parts.append(spec.get("metric", "demeaned_pearson"))
        parts.append("-".join(spec.get("sources") or ["SC", "SC_r2t", "SC+SC_r2t"]))
    if spec.get("include_reference_lines") or spec.get("include_reference_points"):
        parts.append("refs")
    return "__".join(str(p).replace("+", "p") for p in parts)


def save_figure(fig, path_stem, formats=("png",), dpi: int = 300) -> list:
    """Save `fig` as <path_stem>.<fmt> per format. PNG only by default; pass e.g. ("png", "pdf")
    when a vector copy is needed (PDFs keep editable text and carry no creation timestamp)."""
    import matplotlib.pyplot as plt
    from pathlib import Path

    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    with plt.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none"}):
        for fmt in formats:
            out = path_stem.with_suffix(f".{fmt}")
            # No PDF creation timestamp, so re-rendering unchanged data gives identical files.
            metadata = {"CreationDate": None} if fmt == "pdf" else None
            fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white", metadata=metadata)
            written.append(out)
    return written
