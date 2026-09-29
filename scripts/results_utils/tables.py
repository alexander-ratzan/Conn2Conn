"""
tables.py
=========
RunRecords → summary / status / metric DataFrames.

Includes generic model × source tables, covtype-aware tables, the SC-type
experiment summary, and the covariate-projector / deep-model (cov_dl)
seed-level and summary tables.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .records import _display_model_label, _normalize_cov_type_value


# ---------------------------------------------------------------------------
# Status / metric pivots
# ---------------------------------------------------------------------------

def build_status_table(records: list) -> pd.DataFrame:
    """
    Pivot table for debugging: model × (source, seed) → status string.
    Complete cells show tune-trial count; missing cells show "✗ MISSING".
    """
    rows = []
    for r in records:
        cell = f"✓ {r.n_tune_trials}" if r.status == "complete" else "✗ MISSING"
        rows.append({"model": r.model_name, "source": r.source, "seed": r.seed, "cell": cell})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.pivot_table(
        index="model",
        columns=["source", "seed"],
        values="cell",
        aggfunc="first",
    )


def build_metric_table(
    records: list,
    metric: str = "demeaned_pearson",
    agg: str = "mean±std",
    seeds: Optional[list] = None,
) -> pd.DataFrame:
    """
    Pivot table: model (rows) × source (columns) → aggregated metric.

    Parameters
    ----------
    metric : str
        Key in RunRecord.test_metrics (after 'eval_test/' prefix is stripped).
        Common values: 'demeaned_pearson', 'mse', 'pearson'.
    agg : str
        One of 'mean', 'std', 'mean±std', 'median', 'min', 'max'.
    seeds : list[int] | None
        Restrict to specific seeds (useful when some seeds failed).
    """
    rows = []
    for r in records:
        if r.status != "complete":
            continue
        if seeds is not None and r.seed not in seeds:
            continue
        val = r.test_metrics.get(metric)
        if val is not None:
            rows.append({"model": r.model_name, "source": r.source, "value": float(val)})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if agg == "mean±std":
        def _fmt(vals):
            m, s = np.mean(vals), np.std(vals)
            return f"{m:.4f} ± {s:.4f}  (n={len(vals)})"
        return (
            df.groupby(["model", "source"])["value"]
            .apply(_fmt)
            .unstack("source")
        )

    return (
        df.groupby(["model", "source"])["value"]
        .agg(agg)
        .unstack("source")
    )


def build_sc_type_summary_table(
    records: list,
    models: list,
    sources: list,
    metrics: Optional[list] = None,
    metric_labels: Optional[dict] = None,
    seeds: Optional[list] = None,
    count_col: str = "seeds",
    n_label: str = "n",
) -> pd.DataFrame:
    """
    Build the experiment-1 summary table with one row per (model, source).

    Output columns:
    - model
    - source
    - seeds / count_col   (formatted as f"({n_label}={k})")
    - one formatted mean ± std column per requested metric

    Notes
    -----
    - Aggregation is across completed seeds only.
    - Missing model/source combinations are retained as empty rows.
    - Metrics are read from RunRecord.test_metrics.
    """
    if metrics is None:
        metrics = ["pearson", "demeaned_pearson", "mse", "avg_rank", "top1_acc"]
    metric_labels = metric_labels or {}

    rows = []
    for model_name in models:
        for source in sources:
            subset = [
                r for r in records
                if r.model_name == model_name
                and r.source == source
                and r.status == "complete"
                and (seeds is None or r.seed in seeds)
            ]

            row = {
                "model": model_name,
                "source": source,
                count_col: f"({n_label}={len(subset)})" if subset else "",
            }

            for metric in metrics:
                label = metric_labels.get(metric, metric)
                vals = [
                    float(r.test_metrics[metric])
                    for r in subset
                    if metric in r.test_metrics and r.test_metrics[metric] is not None
                ]
                if vals:
                    arr = np.asarray(vals, dtype=float)
                    row[label] = f"{arr.mean():.4f} +/- {arr.std():.4f}"
                else:
                    row[label] = ""

            rows.append(row)

    return pd.DataFrame(rows)


def build_covtype_status_table(records: list) -> pd.DataFrame:
    """
    Pivot table for covtype experiments:
    (model, source) × (cov_type, seed) -> status string.
    """
    rows = []
    for r in records:
        cell = f"✓ {r.n_tune_trials}" if r.status == "complete" else "✗ MISSING"
        rows.append(
            {
                "model": r.model_name,
                "source": r.source,
                "cov_type": _normalize_cov_type_value(r.cov_type) or "unknown",
                "seed": r.seed,
                "cell": cell,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.pivot_table(
        index=["model", "source"],
        columns=["cov_type", "seed"],
        values="cell",
        aggfunc="first",
    )


def build_covtype_metric_table(
    records: list,
    metric: str = "demeaned_pearson",
    agg: str = "mean±std",
    seeds: Optional[list] = None,
    cov_types: Optional[list] = None,
) -> pd.DataFrame:
    """
    Pivot table: (model, source) rows x cov_type columns -> aggregated metric.
    """
    rows = []
    for r in records:
        if r.status != "complete":
            continue
        if seeds is not None and r.seed not in seeds:
            continue
        cov_type = _normalize_cov_type_value(r.cov_type) or "unknown"
        if cov_types is not None and cov_type not in cov_types:
            continue
        val = r.test_metrics.get(metric)
        if val is not None:
            rows.append(
                {
                    "model": r.model_name,
                    "source": r.source,
                    "cov_type": cov_type,
                    "value": float(val),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if agg == "mean±std":
        def _fmt(vals):
            m, s = np.mean(vals), np.std(vals)
            return f"{m:.4f} ± {s:.4f}  (n={len(vals)})"

        out = (
            df.groupby(["model", "source", "cov_type"])["value"]
            .apply(_fmt)
            .unstack("cov_type")
        )
    else:
        out = (
            df.groupby(["model", "source", "cov_type"])["value"]
            .agg(agg)
            .unstack("cov_type")
        )

    if cov_types is not None:
        cols = [c for c in cov_types if c in out.columns]
        out = out[cols] if cols else out

    return out


# ---------------------------------------------------------------------------
# Covariate-projector / deep-model (cov_dl) comparison
# ---------------------------------------------------------------------------

def _cov_type_mask(series: pd.Series, spec_cov):
    normalized = series.apply(_normalize_cov_type_value)
    if spec_cov is None:
        return normalized.isna()
    return normalized == spec_cov


def build_cov_dl_seed_df(
    records: list,
    row_specs: list,
    metrics: Optional[list] = None,
    seeds: Optional[list] = None,
) -> pd.DataFrame:
    """
    Build a seed-level DataFrame for experiment-2 covariate-projector / deep-model
    comparisons from a declarative row-spec list.

    Each row_spec may include:
      - model: required model name
      - source: required SC/source name
      - cov_type: optional covariate type
      - display_model: optional pretty row label
      - sc_source_label: optional display label for SC/source column
      - cov_source_label: optional display label for covariate-source column
      - plot_label: optional compact label for bar charts
    """
    if metrics is None:
        metrics = ["pearson", "demeaned_pearson", "mse", "avg_rank", "top1_acc"]

    rows = []
    for spec in row_specs:
        for r in records:
            if r.model_name != spec["model"]:
                continue
            if r.source != spec.get("source", r.source):
                continue
            spec_cov = _normalize_cov_type_value(spec.get("cov_type"))
            rec_cov = _normalize_cov_type_value(r.cov_type)
            if spec_cov != rec_cov:
                continue
            if seeds is not None and r.seed not in seeds:
                continue

            row = {
                "display_model": spec.get("display_model", _display_model_label(spec["model"])),
                "plot_label": spec.get("plot_label", spec.get("display_model", _display_model_label(spec["model"]))),
                "model": r.model_name,
                "source": r.source,
                "cov_type": _normalize_cov_type_value(r.cov_type),
                "sc_source": spec.get("sc_source_label", spec.get("source", r.source)),
                "cov_source": spec.get("cov_source_label", spec.get("cov_type", "-") or "-"),
                "seed": r.seed,
                "status": r.status,
                "n_tune_trials": r.n_tune_trials,
                "wandb_run_name": r.wandb_run_name,
                "val_demeaned_r": r.val_metrics.get("val_demeaned_r"),
            }
            for metric in metrics:
                row[f"test_{metric}"] = r.test_metrics.get(metric)
            rows.append(row)

    return pd.DataFrame(rows)


def build_cov_dl_summary_table(
    seed_df: pd.DataFrame,
    row_specs: list,
    metrics: Optional[list] = None,
    metric_labels: Optional[dict] = None,
    count_col: str = "seeds",
    n_label: str = "n",
) -> pd.DataFrame:
    """
    Build the experiment-2 summary table with one row per configured row_spec.
    """
    if metrics is None:
        metrics = ["pearson", "demeaned_pearson", "mse", "avg_rank", "top1_acc"]
    metric_labels = metric_labels or {}

    rows = []
    for spec in row_specs:
        display_model = spec.get("display_model", _display_model_label(spec["model"]))
        sc_source = spec.get("sc_source_label", spec.get("source", ""))
        cov_source = spec.get("cov_source_label", spec.get("cov_type", "-") or "-")
        spec_cov = _normalize_cov_type_value(spec.get("cov_type"))
        subset = seed_df[
            (seed_df["model"] == spec["model"])
            & (seed_df["source"] == spec.get("source", ""))
            & _cov_type_mask(seed_df["cov_type"], spec_cov)
            & (seed_df["status"] == "complete")
        ].copy()

        row = {
            "model": display_model,
            "SC_source": sc_source,
            "cov_source": cov_source,
            count_col: f"({n_label}={len(subset)})" if not subset.empty else "",
        }
        for metric in metrics:
            label = metric_labels.get(metric, metric)
            col = f"test_{metric}"
            vals = subset[col].dropna().astype(float).values if col in subset.columns else np.array([])
            if len(vals):
                row[label] = f"{vals.mean():.4f} +/- {vals.std():.4f}"
            else:
                row[label] = ""
        rows.append(row)

    return pd.DataFrame(rows)
