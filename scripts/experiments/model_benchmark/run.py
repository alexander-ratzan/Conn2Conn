"""E2.2 benchmark runner (spec v2 E2.2): collect records, write tables and figures for one direction.

    python scripts/experiments/model_benchmark/run.py --direction sc2fc            # scrape logs -> records -> outputs
    python scripts/experiments/model_benchmark/run.py --direction sc2fc --cached   # re-render from records.json

Records come from this campaign's task logs only (job names `e2_mse_<Model>_<direction>_<job>_<seed>.out`): each task
prints the best-trial report's JSON summary ("Best Tune trial comprehensive summary:"), which holds the best config and
train / val / test metrics. The latest finished task per (model, seed) wins. Reused rows (Krakencoder, test-retest) are
read from their experiments' tables (config.yml `reuse`).

Outputs (<results_dir>/<direction>/, e.g. mse/sc2fc/):
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
    gate_failed = cfg["directions"][direction].get("gate_failed") or {}
    for model in [m for m in cfg["directions"][direction]["models"] if m not in gate_failed]:
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
        row = {"model": r["model"], "label": info.get("label", r["model"]) + (" †" if info.get("extra_inputs") else ""), "type": info.get("type", "other"),
               "role": info.get("role") or "", "objective": info.get("objective") or "", "seed": r["seed"],
               "extra_inputs": bool(info.get("extra_inputs"))}
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
def _order(summary, metric, types, with_ceiling=False):
    """Groups (model types) sorted by their mean on the metric, models within a group by their own mean."""
    s = (summary if with_ceiling else summary[summary["role"] != "ceiling"]).dropna(subset=[f"{metric}_mean"])
    sign = 1 if metric in LOWER_IS_BETTER else -1
    group_mean = s.groupby("type")[f"{metric}_mean"].mean()
    groups = sorted(group_mean.index, key=lambda t: sign * group_mean[t])
    order = []
    for t in groups:
        g = s[s["type"] == t].sort_values(f"{metric}_mean", ascending=(sign == 1))
        order += [(t, r) for r in g.to_dict("records")]
    return groups, order


def bar_axes(ax, cfg, summary, seed_df, metric, show_legend=True, with_ceiling=False):
    """with_ceiling: the test-retest ceiling is drawn as its own bar (axis scaled to it) instead of a line / title note."""
    import matplotlib.pyplot as plt  # noqa: F401
    types = cfg["types"]
    groups, order = _order(summary, metric, types, with_ceiling)
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
    shown = (summary["role"] != "ceiling") | with_ceiling
    shown_seeds = (seed_df["role"] != "ceiling") | with_ceiling
    ymax = max([summary.loc[shown, f"{metric}_mean"].max() or 0,
                np.nanmax(seed_df.loc[shown_seeds, f"test_{metric}"].to_numpy()) if len(seed_df) else 0])
    if with_ceiling:
        ceiling = ceiling.iloc[0:0]  # drawn as a bar: no line / off-scale note
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
        vals = summary.loc[shown, "pearson_mean"].dropna()
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


def figures(cfg, summary, seed_df, out, direction):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["DejaVu Sans", "sans-serif"], "font.size": 12, "axes.spines.right": False,
                         "axes.spines.top": False, "axes.linewidth": 1.6})
    out.mkdir(parents=True, exist_ok=True)
    note = "Bars: mean ± SE over seeds (dots = seeds). * = native objective (not plain MSE). † = extra inputs (anatomy + demographics)."
    dropped = {**(cfg["directions"][direction].get("excluded") or {}), **(cfg["directions"][direction].get("gate_failed") or {})}
    if dropped:
        note += " Not shown: " + ", ".join(model_info(cfg, m).get("label", m) for m in dropped) + " (see tables/excluded.md)."
    written = []
    for metric in cfg["metrics"]:
        fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(summary) + 3), 5.2))
        bar_axes(ax, cfg, summary, seed_df, metric)
        fig.text(0.01, 0.005, note, fontsize=8, color="#555555")
        fig.tight_layout(pad=1.2)
        fig.savefig(out / f"bars_{metric}.png", dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        written.append(f"bars_{metric}.png")
    import matplotlib.patches as mpatches
    present = [t for t in cfg["types"] if t in set(summary["type"])]
    variants = [("bars_all_metrics.png", False)]
    if (summary["role"] == "ceiling").any():
        variants.append(("bars_all_metrics_ceiling.png", True))  # same panel with the test-retest ceiling as a bar
    for name, with_ceiling in variants:
        fig, axes = plt.subplots(2, 2, figsize=(2 * max(8, 0.75 * len(summary) + 3), 11))
        for ax, metric in zip(axes.flat, cfg["metrics"]):
            bar_axes(ax, cfg, summary, seed_df, metric, show_legend=False, with_ceiling=with_ceiling)
        fig.legend(handles=[mpatches.Patch(color=cfg["types"][t]["color"], label=cfg["types"][t]["label"]) for t in present],
                   loc="upper center", ncol=len(present), frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.0))
        fig.text(0.01, 0.003, note, fontsize=9, color="#555555")
        fig.tight_layout(pad=1.5, rect=(0, 0, 1, 0.97))
        fig.savefig(out / name, dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        written.append(name)
    return written


def paired(cfg, seed_df, summary):
    """Per-seed paired differences against the best input-matched linear model (by mean test demeaned r; models with
    `extra_inputs` are never the reference), on shared seeds.
    Splits are shared across models, so paired differences remove split-to-split variance."""
    matched = {m for m in summary["model"] if not model_info(cfg, m).get("extra_inputs")}
    lin = summary[summary["type"].str.startswith("linear") & (summary["role"] == "") & summary["model"].isin(matched)]
    if lin.empty:
        return None, pd.DataFrame()
    ref = lin.sort_values("demeaned_pearson_mean", ascending=False).iloc[0]["model"]
    base = seed_df[seed_df["model"] == ref].set_index("seed")
    rows = []
    for model, g in seed_df[seed_df["role"] != "ceiling"].groupby("model", sort=False):
        g = g.set_index("seed")
        shared = sorted(set(g.index) & set(base.index))
        r = {"model": model, "label": g["label"].iloc[0], "type": g["type"].iloc[0], "reference": ref,
             "n_shared_seeds": len(shared)}
        for m in ("pearson", "demeaned_pearson", "avg_rank", "top1_acc"):
            d = (g.loc[shared, f"test_{m}"] - base.loc[shared, f"test_{m}"]).dropna()
            r[f"d_{m}_mean"] = d.mean() if len(d) else np.nan
            r[f"d_{m}_se"] = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else np.nan
        rows.append(r)
    return ref, pd.DataFrame(rows).sort_values("d_demeaned_pearson_mean", ascending=False)


def scatter(cfg, summary, out):
    """Test demeaned r vs average rank, one point per model (mean ± SE), coloured by model type."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = summary[summary["role"] != "ceiling"]
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for r in s.to_dict("records"):
        col = cfg["types"].get(r["type"], {}).get("color", "#777777")
        ax.errorbar(r["avg_rank_mean"], r["demeaned_pearson_mean"], xerr=r["avg_rank_se"], yerr=r["demeaned_pearson_se"],
                    fmt="s" if r["objective"] else "o", ms=8, color=col, ecolor=col, capsize=3,
                    mfc="white" if r["objective"] else col, mew=1.6)
    # labels: nudge apart vertically when points crowd (greedy, bottom-up), with a leader line when moved
    pts = sorted(((r["avg_rank_mean"], r["demeaned_pearson_mean"], r["label"]) for r in s.to_dict("records")),
                 key=lambda t: t[1])
    xr = np.ptp([t[0] for t in pts]) or 1.0
    gap = 0.035 * (np.ptp([t[1] for t in pts]) or 1.0)
    placed = []
    for x, y, lab in pts:
        ly = y
        for px, py in placed:
            if abs(px - x) < 0.25 * xr and ly - py < gap:
                ly = py + gap
        placed.append((x, ly))
        moved = abs(ly - y) > 1e-12
        ax.annotate(lab, (x, y), xytext=(x + 0.012 * xr, ly + 0.25 * gap), fontsize=8, color="#444444",
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6) if moved else None)
    ceil = summary[summary["role"] == "ceiling"]
    note = ""
    if len(ceil):
        c = ceil.iloc[0]
        note = f"test-retest ceiling: demeaned r {c['demeaned_pearson_mean']:.3g}, avg rank {c['avg_rank_mean']:.3g} (off scale)"
    import matplotlib.patches as mpatches
    present = [t for t in cfg["types"] if t in set(s["type"])]
    ax.legend(handles=[mpatches.Patch(color=cfg["types"][t]["color"], label=cfg["types"][t]["label"]) for t in present],
              fontsize=9, frameon=False, loc="lower right")
    ax.set_xlabel("Average rank (test, max)")
    ax.set_ylabel("Demeaned r (test, max)")
    ax.grid(alpha=0.25, ls="--")
    ax.set_title("Mean ± SE over seeds; open squares = native objective" + ("\n" + note if note else ""), fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "scatter_demeaned_vs_rank.png", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return "scatter_demeaned_vs_rank.png"


# ------------------------------------------------------------------------------------------- entry
def build(direction, cached=False, cfg=None, out_root=None, log_dir=LOG_DIR):
    cfg = cfg or yaml.safe_load((HERE / "config.yml").read_text())
    d = Path(out_root or HERE) / cfg.get("results_dir", "") / direction
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
    written = figures(cfg, summary, seed_df, d / "figures", direction)
    written.append(scatter(cfg, summary, d / "figures"))
    ref, pair = paired(cfg, seed_df, summary)
    if ref is not None:
        pair.to_csv(d / "tables" / "paired_vs_best_linear.csv", index=False, float_format="%.6g")
        lines = [f"Paired per-seed differences vs `{ref}` (best input-matched linear model by mean test demeaned r; shared seeds). † = extra inputs, not input-matched.", "",
                 "| Model | Seeds | Δ Pearson r | Δ demeaned r | Δ avg rank | Δ top-1 |", "|---|---|---|---|---|---|"]
        for r in pair.to_dict("records"):
            cell = lambda m: (f"{r[f'd_{m}_mean']:+.4f} ± {r[f'd_{m}_se']:.4f}" if not np.isnan(r[f"d_{m}_se"])
                              else (f"{r[f'd_{m}_mean']:+.4f}" if not np.isnan(r[f"d_{m}_mean"]) else "–"))
            lines.append(f"| {r['label']} | {r['n_shared_seeds']} | " + " | ".join(cell(m) for m in ("pearson", "demeaned_pearson", "avg_rank", "top1_acc")) + " |")
        (d / "tables" / "paired_vs_best_linear.md").write_text("\n".join(lines) + "\n")
    gate_failed = cfg["directions"][direction].get("gate_failed") or {}
    if gate_failed:
        (d / "tables" / "excluded.md").write_text("".join(f"- `{m}`: {why}\n" for m, why in gate_failed.items()))
    missing = [(m, sorted(set(cfg["seeds"]) - set(seed_df.loc[seed_df["model"] == m, "seed"])))
               for m in cfg["directions"][direction]["models"] if m not in gate_failed]
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
