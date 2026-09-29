from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from .records import LOCAL_RESULTS_DIR

DEFAULT_LOCAL_RESULTS_ROOT = LOCAL_RESULTS_DIR


def _safe_scalar(x):
    if isinstance(x, (np.integer, np.int32, np.int64)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (int, float, bool, str)):
        return x
    return None


def _flatten_dict(d, prefix=""):
    flat = {}
    if not isinstance(d, dict):
        return flat

    for key, value in d.items():
        name = f"{prefix}{key}" if prefix else key

        if key == "ranklist":
            arr = np.asarray(value, dtype=float)
            if arr.size:
                flat[f"{prefix}rank_mean"] = float(arr.mean())
                flat[f"{prefix}rank_median"] = float(np.median(arr))
                flat[f"{prefix}rank_std"] = float(arr.std())
                flat[f"{prefix}rank_min"] = float(arr.min())
                flat[f"{prefix}rank_max"] = float(arr.max())
                flat[f"{prefix}rank_p25"] = float(np.percentile(arr, 25))
                flat[f"{prefix}rank_p75"] = float(np.percentile(arr, 75))
            continue

        if isinstance(value, dict):
            flat.update(_flatten_dict(value, prefix=f"{name}_"))
            continue

        scalar = _safe_scalar(value)
        if scalar is not None:
            flat[name] = scalar

    return flat


def load_local_results(local_root=DEFAULT_LOCAL_RESULTS_ROOT):
    local_root = Path(local_root)
    rows = []

    for final_dir in sorted(local_root.glob("*/final")):
        config_path = final_dir / "config.json"
        metrics_path = final_dir / "metrics_final.json"
        manifest_path = final_dir / "artifact_manifest.json"

        if not config_path.exists() or not metrics_path.exists():
            continue

        config = json.loads(config_path.read_text())
        metrics = json.loads(metrics_path.read_text())
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

        row = {
            "run_name": final_dir.parent.name,
            "artifact_dir": str(final_dir),
            "learned": manifest.get("learned"),
        }
        row.update(_flatten_dict(config.get("data", {}), prefix="data_"))
        row.update(_flatten_dict(config.get("model", {}), prefix="model_"))
        row.update(_flatten_dict(config.get("trainer", {}), prefix="trainer_"))
        row.update(_flatten_dict(config.get("extra", {}), prefix="extra_"))
        row.update(_flatten_dict(metrics.get("train", {}), prefix="train_"))
        row.update(_flatten_dict(metrics.get("val", {}), prefix="val_"))
        row.update(_flatten_dict(metrics.get("test", {}), prefix="test_"))
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    preferred = [
        "run_name",
        "extra_base_model_name",
        "learned",
        "data_source",
        "data_target",
        "val_mse",
        "val_demeaned_pearson",
        "test_base_metrics_mse",
        "test_base_metrics_r2",
        "test_base_metrics_pearson",
        "test_base_metrics_demeaned_pearson",
        "test_heatmaps_raw_top1_acc",
        "test_heatmaps_raw_avg_rank_percentile",
        "test_heatmaps_raw_rank_mean",
        "test_heatmaps_raw_rank_median",
        "test_heatmaps_demeaned_top1_acc",
        "test_heatmaps_demeaned_avg_rank_percentile",
        "test_heatmaps_demeaned_rank_mean",
        "test_heatmaps_demeaned_rank_median",
        "artifact_dir",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]


def best_local_results(
    df,
    sort_metric="test_base_metrics_demeaned_pearson",
    groupby=("extra_base_model_name", "data_source"),
    ascending=False,
):
    if df.empty:
        return df

    available_groupby = [col for col in groupby if col in df.columns]
    out = df.sort_values(sort_metric, ascending=ascending)
    if available_groupby:
        out = out.drop_duplicates(subset=available_groupby, keep="first")
    return out.reset_index(drop=True)


def plot_metric_bar(
    df,
    metric="test_base_metrics_demeaned_pearson",
    label_col="run_name",
    figsize=(12, 5),
    title=None,
    rotation=75,
):
    plot_df = df.sort_values(metric, ascending=False)
    plt.figure(figsize=figsize)
    plt.bar(plot_df[label_col], plot_df[metric])
    plt.xticks(rotation=rotation, ha="right")
    plt.ylabel(metric)
    plt.title(title or metric)
    plt.tight_layout()
    plt.show()


def plot_metric_scatter(
    df,
    x="test_base_metrics_mse",
    y="test_base_metrics_demeaned_pearson",
    label_col="run_name",
    figsize=(7, 5),
    title=None,
):
    plt.figure(figsize=figsize)
    plt.scatter(df[x], df[y])
    for _, row in df.iterrows():
        plt.text(row[x], row[y], row[label_col], fontsize=8)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title or f"{y} vs {x}")
    plt.tight_layout()
    plt.show()
