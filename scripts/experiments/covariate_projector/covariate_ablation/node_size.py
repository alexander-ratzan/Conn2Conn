"""E1 follow-up: how much of the subject-specific SC is parcel size (node scaling)?

Per edge, across training subjects, regress the outcome on the subject's own parcel geometry and apply the fit to
the test subjects (`data.data_viz` edgewise helpers):
  distance         y_e ~ distance_e
  volume           y_e ~ log(4 v_i v_j) + log1p|v_i - v_j|          (parcel volumes, `base.parcel_volume`)
  distance_volume  y_e ~ distance_e + both volume terms
Outcomes: normalised SC (invnodevol, E2.2 / E1 target), unnormalised SC (sift_radius2_count), and FC (contrast with
the FC-outcome analysis in scripts/notebooks/EDA/dist_vol_analysis.ipynb). Glasser, E1 splits (seeds 0-4).

Outputs (results/node_size/):
  per_edge_block.csv  seed 0, train: nested per-edge F-test, volume terms over distance (median / mean delta R2, FDR)
  heldout.csv         every seed: test metrics of each anatomy-only model (same metric code as E1)
  heldout_summary.md  mean ± std over seeds, with E1 PCA-PLS + covariates `none` / `full` for reference
  fig_node_size.png   held-out demeaned r and top-1 per outcome

  python scripts/experiments/covariate_projector/covariate_ablation/node_size.py [--seeds 0 1 2 3 4]
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from conditions import load_config  # noqa: E402
from data.data_viz import (  # noqa: E402
    compare_edgewise_models,
    compute_edgewise_multifeature_regression,
    evaluate_edgewise_predictions,
    predict_edgewise_multifeature_regression,
)
from data.hcp_dataset import HCP_Base  # noqa: E402

OUT_DIR = HERE / "results" / "node_size"
MODELS = {
    "distance": "distance",
    "volume": "log_product_volume_log_volume_mismatch",
    "distance_volume": "distance_log_product_volume_log_volume_mismatch",
}
VOLUME_TERMS = ["log_product_volume", "log_volume_mismatch"]
# (outcome label, y_key, sc_metric key in config.yml; E1 direction / sc_metric for the reference rows)
OUTCOMES = [
    ("SC (invnodevol)", "SC", "invnodevol", ("fc2sc", "invnodevol")),
    ("SC (unnormalised)", "SC", "unnorm", ("fc2sc", "unnorm")),
    ("FC", "FC", "invnodevol", ("sc2fc", "invnodevol")),
]
METRICS = ("demeaned_pearson", "top1_acc", "avg_rank", "pearson")


def load_base(cfg, sc_metric, seed):
    return HCP_Base(
        parcellation=cfg["parcellation"], hemi="both", shuffle_seed=seed, source="FC", target="SC",
        sc_metric_type=cfg["sc_metrics"][sc_metric], sc_apply_log1p=True, data_load_mode="precomputed",
        write_manual_cache=False, expose_node_features=False,
    )


def fit_models(base, y_key):
    return {m: compute_edgewise_multifeature_regression(base, partition="train", y_key=y_key, model_name=name)
            for m, name in MODELS.items()}


def volume_block(fits):
    """Nested per-edge test: volume terms over distance (train)."""
    return compare_edgewise_models(fits["distance"], fits["distance_volume"], added_predictor_names=VOLUME_TERMS)


def block_summary(block, outcome):
    d = block["delta_r2"]
    return {"outcome": outcome, "n_edges": len(block), "median_delta_r2": float(d.median()),
            "mean_delta_r2": float(d.mean()), "q90_delta_r2": float(d.quantile(0.9)),
            "median_r2_distance": float(block["r2_reduced"].median()),
            "median_r2_distance_volume": float(block["r2_full"].median()),
            "n_fdr_significant": int(block["block_significant_fdr"].sum()),
            "frac_fdr_significant": float(block["block_significant_fdr"].mean())}


def heldout_rows(base, fits, outcome, y_key, seed):
    rows = []
    for m, reg_df in fits.items():
        y_pred, y_true = predict_edgewise_multifeature_regression(base, reg_df, partition="test")
        met = evaluate_edgewise_predictions(y_pred, y_true, reg_df[f"{y_key.lower()}_mean"].to_numpy())
        rows.append({"outcome": outcome, "model": m, "seed": seed, "n_test": len(y_true),
                     **{k: float(met[k]) for k in METRICS}})
    return rows


def e1_reference(cfg, seeds):
    rows = []
    for outcome, _, _, (direction, sc_metric) in OUTCOMES:
        for cond in ("none", "full"):
            for s in seeds:
                p = HERE / cfg["runs_dir"] / sc_metric / direction / cond / f"seed{s}.json"
                if p.exists():
                    t = json.loads(p.read_text())["test"]
                    rows.append({"outcome": outcome, "model": f"E1 PCA-PLS + cov, {cond}", "seed": s,
                                 **{k: float(t[k]) for k in METRICS if k in t}})
    return rows


def run(seeds):
    cfg = load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    block_rows, rows = [], []
    for seed in seeds:
        for sc_metric in ("invnodevol", "unnorm"):
            base = load_base(cfg, sc_metric, seed)
            for outcome, y_key, metric_key, _ in OUTCOMES:
                if metric_key != sc_metric:
                    continue
                fits = fit_models(base, y_key)
                if seed == seeds[0]:
                    block_rows.append({"seed": seed, **block_summary(volume_block(fits), outcome)})
                rows += heldout_rows(base, fits, outcome, y_key, seed)
                print(f"[node_size] seed {seed} {outcome}: "
                      + ", ".join(f"{r['model']} dr={r['demeaned_pearson']:.3f} top1={r['top1_acc']:.3f}"
                                  for r in rows[-len(MODELS):]), flush=True)
                del fits
                gc.collect()
            del base
            gc.collect()
    pd.DataFrame(block_rows).to_csv(OUT_DIR / "per_edge_block.csv", index=False)
    heldout = pd.DataFrame(rows + e1_reference(cfg, seeds))
    heldout.to_csv(OUT_DIR / "heldout.csv", index=False)
    write_summary(heldout)
    plot_heldout(heldout, OUT_DIR / "fig_node_size.png")
    return load_results()


def summarize(heldout):
    g = heldout.groupby(["outcome", "model"], sort=False)
    out = g[list(METRICS)].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["n_seeds"] = g.size()
    return out.reset_index()


def write_summary(heldout):
    s = summarize(heldout)
    cols = ["outcome", "model", "n_seeds"] + [f"{m}_{k}" for m in ("demeaned_pearson", "top1_acc") for k in ("mean", "std")]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in s[cols].to_dict("records"):
        lines.append("| " + " | ".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    (OUT_DIR / "heldout_summary.md").write_text("\n".join(lines) + "\n")


def load_results():
    return {"per_edge_block": pd.read_csv(OUT_DIR / "per_edge_block.csv"),
            "heldout": pd.read_csv(OUT_DIR / "heldout.csv"),
            "figure": OUT_DIR / "fig_node_size.png"}


def _model_benchmark_run():
    """scripts/experiments/model_benchmark/run.py (FIG_RC, METRIC_LABEL) under its own module name: E1's run.py
    and this folder both have a `run` module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "model_benchmark_run", REPO_ROOT / "scripts" / "experiments" / "model_benchmark" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def plot_heldout(heldout, path):
    import matplotlib.pyplot as plt

    mb = _model_benchmark_run()
    arms = [("distance", "Distance only", "#D1D5DB"), ("volume", "Parcel volume only", "#93C5FD"),
            ("distance_volume", "Distance + volume", "#0F4D92"),
            ("E1 PCA-PLS + cov, none", "E1 PCA-PLS + cov, none", "#9CA3AF"),
            ("E1 PCA-PLS + cov, full", "E1 PCA-PLS + cov, full", "#C2410C")]
    outcomes = [o for o, *_ in OUTCOMES]
    with plt.rc_context({**mb.FIG_RC, "font.size": 14}):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), layout="constrained")
        for ax, m in zip(axes, ("demeaned_pearson", "top1_acc")):
            w = 0.16
            for k, (arm, label, color) in enumerate(arms):
                for j, o in enumerate(outcomes):
                    v = heldout[(heldout.outcome == o) & (heldout.model == arm)][m]
                    if v.empty:
                        continue
                    x = j + (k - (len(arms) - 1) / 2) * w
                    ax.bar(x, v.mean(), width=w * 0.95, color=color, edgecolor="black", lw=1,
                           label=label if j == 0 else None)
                    if len(v) > 1:
                        ax.errorbar(x, v.mean(), yerr=v.std(ddof=1), color="black", capsize=3, lw=1.2)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xticks(range(len(outcomes)), outcomes)
            ax.set_title(mb.METRIC_LABEL[m], fontweight="bold")
        axes[0].legend(frameon=False, fontsize=10)
        fig.suptitle("Anatomy-Only Edgewise Models vs E1 (test, mean ± SD over seeds)", fontweight="bold")
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    run(ap.parse_args().seeds)
