"""E2.2 benchmark runner (spec v2 E2.2): collect records, write tables and figures for one direction.

    python scripts/experiments/model_benchmark/run.py --direction sc2fc            # scrape logs -> records -> outputs
    python scripts/experiments/model_benchmark/run.py --direction sc2fc --cached   # re-render from <direction>/records.json

Records come from this campaign's task logs only (job names `e2_mse_<Model>_<direction>_<job>_<seed>.out`): each task
prints the best-trial report's JSON summary ("Best Tune trial comprehensive summary:"), which holds the best config and
train / val / test metrics. The latest finished task per (model, seed) wins. Reused rows (Krakencoder, test-retest) are
read from their experiments' tables (config.yml `reuse`).

Outputs (<direction>/):
    records.json                     per (model, seed): config, val / test metrics, log path, ray_tune_id (tracked)
    tables/seed_records.csv          one row per model x seed
    tables/summary.csv, summary.md   per model: mean, SE, n seeds, per metric; model type
    figures/bars_<metric>.png        one bar chart per metric: grouped and coloured by model type, groups sorted by their
                                     mean, bars by model mean; mean ± SE with per-seed points; null / ceiling marked
    figures/bars_all_metrics.png     the four bar charts as one 2 x 2 panel
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
LOG_DIR = REPO_ROOT / "results" / "logs"
MARKER = "Best Tune trial comprehensive summary:"
METRIC_LABEL = {"pearson": "Pearson r", "demeaned_pearson": "Demeaned r", "avg_rank": "Average rank",
                "top1_acc": "Top-1 accuracy", "mse": "MSE"}
LOWER_IS_BETTER = {"mse"}


# ------------------------------------------------------------------------------------------- collection
def parse_log(path):
    """(seed, summary dict) from one task log, or None if the task has no finished best-trial report."""
    text = Path(path).read_text(errors="replace")
    seed = re.search(r"Seed=(\d+)", text)
    i = text.rfind(MARKER)
    if not seed or i < 0:
        return None
    try:
        summary, _ = json.JSONDecoder().raw_decode(text[i + len(MARKER):].lstrip())
    except ValueError:
        return None
    return int(seed.group(1)), summary


def scrape(cfg, direction, log_dir=LOG_DIR):
    records = []
    for model in cfg["directions"][direction]["models"]:
        name = f"{cfg['campaign']}_{model}_{direction}"
        latest = {}
        for path in sorted(glob.glob(str(Path(log_dir) / f"{name}_*_*.out")), key=os.path.getmtime):
            got = parse_log(path)
            if got:
                latest[got[0]] = (path, got[1])
        for seed, (path, s) in sorted(latest.items()):
            if seed not in cfg["seeds"]:
                continue
            m = s.get("metrics") or {}
            records.append({"model": model, "seed": seed, "source": "campaign", "log": os.path.relpath(path, REPO_ROOT),
                            "ray_tune_id": s.get("ray_tune_id"), "selected_val": (s.get("selected_by") or {}).get("value"),
                            "config": (s.get("best_trial") or {}).get("config"),
                            "val": m.get("val") or {}, "test": m.get("test") or {}})
    for name in cfg["directions"][direction].get("reuse", []):
        info = cfg["reuse"][name]
        path = (info.get("records") or {}).get(direction)
        if not path or not (REPO_ROOT / path).exists():
            continue
        df = pd.read_csv(REPO_ROOT / path)
        if "combo_id" in df and info.get("combo_id"):
            df = df[df["combo_id"] == info["combo_id"]]
        if "stage" in df:
            df = df[df["stage"] == "stage2"]
        if "random_seed" in df:
            df = df[df["random_seed"] == df["random_seed"].min()]
        for r in df.to_dict("records"):
            if int(r["seed"]) not in cfg["seeds"]:
                continue
            records.append({"model": name, "seed": int(r["seed"]), "source": f"reuse:{path}",
                            "test": {k[5:]: r[k] for k in r if k.startswith("test_")},
                            "val": {k[4:]: r[k] for k in r if k.startswith("val_")}})
    return records


# ------------------------------------------------------------------------------------------- tables
def model_info(cfg, model):
    return cfg["models"].get(model) or cfg["reuse"].get(model) or {}


def tables(cfg, records):
    rows = []
    for r in records:
        info = model_info(cfg, r["model"])
        row = {"model": r["model"], "label": info.get("label", r["model"]), "type": info.get("type", "other"),
               "role": info.get("role") or "", "objective": info.get("objective") or "", "seed": r["seed"]}
        row.update({f"test_{m}": (float(r["test"][m]) if r["test"].get(m) is not None else np.nan)
                    for m in ("pearson", "demeaned_pearson", "avg_rank", "top1_acc", "mse")})
        row["val_demeaned_r"] = r.get("selected_val") if r.get("selected_val") is not None else r["val"].get("demeaned_r")
        rows.append(row)
    seed_df = pd.DataFrame(rows)
    for col in ("role", "objective"):  # groupby drops NaN keys: keep empty strings
        seed_df[col] = seed_df[col].fillna("").astype(str)
    seed_df = seed_df.sort_values(["type", "model", "seed"]).reset_index(drop=True)
    agg = []
    for (model, label, typ, role, obj), g in seed_df.groupby(["model", "label", "type", "role", "objective"], sort=False):
        a = {"model": model, "label": label, "type": typ, "role": role, "objective": obj, "n_seeds": g["seed"].nunique()}
        for m in ("pearson", "demeaned_pearson", "avg_rank", "top1_acc", "mse"):
            v = g[f"test_{m}"].dropna()
            a[f"{m}_mean"] = v.mean() if len(v) else np.nan
            a[f"{m}_se"] = v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else np.nan
        agg.append(a)
    return seed_df, pd.DataFrame(agg)


# ------------------------------------------------------------------------------------------- figures
def _order(summary, metric, types):
    """Groups (model types) sorted by their mean on the metric, models within a group by their own mean."""
    s = summary[summary["role"] != "ceiling"].dropna(subset=[f"{metric}_mean"])
    sign = 1 if metric in LOWER_IS_BETTER else -1
    group_mean = s.groupby("type")[f"{metric}_mean"].mean()
    groups = sorted(group_mean.index, key=lambda t: sign * group_mean[t])
    order = []
    for t in groups:
        g = s[s["type"] == t].sort_values(f"{metric}_mean", ascending=(sign == 1))
        order += [(t, r) for r in g.to_dict("records")]
    return groups, order


def bar_axes(ax, cfg, summary, seed_df, metric, show_legend=True):
    import matplotlib.pyplot as plt  # noqa: F401
    types = cfg["types"]
    groups, order = _order(summary, metric, types)
    x, gap, xs, labels = 0.0, 0.7, [], []
    group_spans = []
    for t in groups:
        start = x
        for (tt, r) in [o for o in order if o[0] == t]:
            col = types.get(t, {}).get("color", "#777777")
            m, se = r[f"{metric}_mean"], r[f"{metric}_se"]
            hatch = "//" if r.get("objective") else None
            ax.bar(x, m, width=0.8, color=col, alpha=0.85, edgecolor="black" if hatch else col, linewidth=0.6,
                   hatch=hatch, zorder=2)
            if not np.isnan(se):
                ax.errorbar(x, m, yerr=se, fmt="none", ecolor="black", elinewidth=1.4, capsize=4, zorder=4)
            pts = seed_df[seed_df["model"] == r["model"]][f"test_{metric}"].dropna().to_numpy()
            if len(pts):
                jitter = np.linspace(-0.22, 0.22, len(pts)) if len(pts) > 1 else np.zeros(1)
                ax.scatter(x + jitter, pts, s=14, color="white", edgecolor="black", linewidth=0.7, zorder=5)
            xs.append(x)
            labels.append(r["label"] + (" *" if r.get("objective") else ""))
            x += 1.0
        group_spans.append((t, start, x - 1.0))
        x += gap
    ceiling = summary[summary["role"] == "ceiling"]
    ymax = max([summary.loc[summary["role"] != "ceiling", f"{metric}_mean"].max() or 0,
                np.nanmax(seed_df.loc[seed_df["role"] != "ceiling", f"test_{metric}"].to_numpy()) if len(seed_df) else 0])
    null = summary[summary["role"] == "floor"]
    if len(null) and not np.isnan(null.iloc[0][f"{metric}_mean"]):
        ax.axhline(null.iloc[0][f"{metric}_mean"], color="#8C8C8C", ls=":", lw=1.4, zorder=1)
    title_note = ""
    if len(ceiling) and not np.isnan(ceiling.iloc[0][f"{metric}_mean"]):
        c = ceiling.iloc[0][f"{metric}_mean"]
        if c <= ymax * 1.6:
            ax.axhline(c, color="black", ls="--", lw=1.2, zorder=1)
            ax.text(-0.7, c, "test-retest ceiling", va="bottom", ha="left", fontsize=9)
        else:
            title_note = f"  (test-retest ceiling {c:.3g}, off scale)"
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=10)
    ax.set_ylabel(f"{METRIC_LABEL[metric]} (test{', min' if metric in LOWER_IS_BETTER else ', max'})")
    ax.set_title(METRIC_LABEL[metric] + title_note, fontsize=13)
    lo = 0.5 if metric == "avg_rank" else 0.0
    if metric == "pearson":
        vals = summary.loc[summary["role"] != "ceiling", "pearson_mean"].dropna()
        lo = max(0.0, (vals.min() - 0.02) if len(vals) else 0.0)
    top = ymax * 1.08
    if len(ceiling) and not title_note:
        top = max(top, ceiling.iloc[0][f"{metric}_mean"] * 1.03)
    ax.set_ylim(lo, top)
    ax.set_xlim(-0.8, xs[-1] + 0.8)
    ax.grid(axis="y", alpha=0.25, ls="--", zorder=0)
    if show_legend:
        import matplotlib.patches as mpatches
        handles = [mpatches.Patch(color=types[t]["color"], label=types[t]["label"]) for t in groups if t in types]
        ax.legend(handles=handles, fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 1.06), ncol=len(handles),
                  frameon=False, borderaxespad=0.0)


def figures(cfg, summary, seed_df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["DejaVu Sans", "sans-serif"], "font.size": 12, "axes.spines.right": False,
                         "axes.spines.top": False, "axes.linewidth": 1.6})
    out.mkdir(parents=True, exist_ok=True)
    note = "Bars: mean ± SE over seeds (dots = seeds). * = native objective (not plain MSE)."
    written = []
    for metric in cfg["metrics"]:
        fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(summary) + 3), 5.2))
        bar_axes(ax, cfg, summary, seed_df, metric)
        fig.text(0.01, 0.005, note, fontsize=8, color="#555555")
        fig.tight_layout(pad=1.2)
        fig.savefig(out / f"bars_{metric}.png", dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        written.append(f"bars_{metric}.png")
    fig, axes = plt.subplots(2, 2, figsize=(2 * max(8, 0.75 * len(summary) + 3), 11))
    for ax, metric in zip(axes.flat, cfg["metrics"]):
        bar_axes(ax, cfg, summary, seed_df, metric, show_legend=False)
    import matplotlib.patches as mpatches
    present = [t for t in cfg["types"] if t in set(summary["type"])]
    fig.legend(handles=[mpatches.Patch(color=cfg["types"][t]["color"], label=cfg["types"][t]["label"]) for t in present],
               loc="upper center", ncol=len(present), frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.003, note, fontsize=9, color="#555555")
    fig.tight_layout(pad=1.5, rect=(0, 0, 1, 0.97))
    fig.savefig(out / "bars_all_metrics.png", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return written + ["bars_all_metrics.png"]


# ------------------------------------------------------------------------------------------- entry
def build(direction, cached=False, cfg=None, out_root=None, log_dir=LOG_DIR):
    cfg = cfg or yaml.safe_load((HERE / "config.yml").read_text())
    d = Path(out_root or HERE) / direction
    d.mkdir(parents=True, exist_ok=True)
    if cached:
        records = json.loads((d / "records.json").read_text())
    else:
        records = scrape(cfg, direction, log_dir)
        (d / "records.json").write_text(json.dumps(records, indent=1, sort_keys=True, default=str))
    if not records:
        print(f"{direction}: no records yet")
        return 1
    seed_df, summary = tables(cfg, records)
    (d / "tables").mkdir(exist_ok=True)
    seed_df.to_csv(d / "tables" / "seed_records.csv", index=False, float_format="%.6g")
    summary.to_csv(d / "tables" / "summary.csv", index=False, float_format="%.6g")
    md = ["| Model | Type | Seeds | " + " | ".join(METRIC_LABEL[m] for m in cfg["metrics"]) + " |",
          "|---|---|---|" + "---|" * len(cfg["metrics"])]
    for r in summary.sort_values("demeaned_pearson_mean", ascending=False).to_dict("records"):
        cells = [f"{r[f'{m}_mean']:.4f} ± {r[f'{m}_se']:.4f}" if not np.isnan(r[f"{m}_se"]) else f"{r[f'{m}_mean']:.4f}"
                 for m in cfg["metrics"]]
        md.append(f"| {r['label']} | {cfg['types'].get(r['type'], {}).get('label', r['type'])} | {r['n_seeds']} | " + " | ".join(cells) + " |")
    (d / "tables" / "summary.md").write_text("\n".join(md) + "\n")
    written = figures(cfg, summary, seed_df, d / "figures")
    missing = [(m, sorted(set(cfg["seeds"]) - set(seed_df.loc[seed_df["model"] == m, "seed"])))
               for m in cfg["directions"][direction]["models"]]
    missing = [(m, s) for m, s in missing if s]
    print(f"{direction}: {len(records)} records, {summary.shape[0]} models; figures {written}")
    if missing:
        print("missing seeds:", missing)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--direction", required=True, choices=["sc2fc", "fc2sc"])
    ap.add_argument("--cached", action="store_true")
    args = ap.parse_args()
    return build(args.direction, cached=args.cached)


if __name__ == "__main__":
    sys.exit(main())
