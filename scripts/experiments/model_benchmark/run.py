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


def gate_dropped(cfg, direction):
    """Gate-failed models not overridden for this direction: {model: reason}."""
    dcfg = cfg["directions"][direction]
    override = dcfg.get("gate_override") or {}
    return {m: why for m, why in (dcfg.get("gate_failed") or {}).items() if m not in override}


def scrape(cfg, direction, log_dir=LOG_DIR):
    records = []
    gate_failed = gate_dropped(cfg, direction)
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


def tables(cfg, records, direction=None):
    rows = []
    for r in records:
        info = model_info(cfg, r["model"])
        overridden = direction and r["model"] in (cfg["directions"][direction].get("gate_override") or {})
        row = {"model": r["model"], "label": info.get("label", r["model"]) + (" †" if info.get("extra_inputs") else "")
               + (" ‡" if overridden else ""), "type": info.get("type", "other"),
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
FIG_RC = {"font.family": ["Arial", "Helvetica", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"], "font.size": 14,
          "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 1.8, "xtick.major.width": 1.5,
          "ytick.major.width": 1.5, "legend.frameon": False}  # scientific-figure-making skill preset, sized for print


def _order(cfg, summary, with_ceiling=False):
    """Fixed thematic order for every metric (as panel_grouped): type groups in config order, models within a group by
    test demeaned r, PCA null last (the ceiling first in the null group when drawn as a bar)."""
    s = summary if with_ceiling else summary[summary["role"] != "ceiling"]
    body = s[s["role"] == ""].sort_values("demeaned_pearson_mean", ascending=False)
    groups, order = [], []
    for t in cfg["types"]:
        g = body[body["type"] == t]
        if len(g):
            groups.append(t)
            order += [(t, r) for r in g.to_dict("records")]
    ref = pd.concat([s[s["role"] == "ceiling"], s[s["role"] == "floor"]])
    if len(ref):
        if "null_ceiling" not in groups:
            groups.append("null_ceiling")
        order += [("null_ceiling", r) for r in ref.to_dict("records")]
    return groups, order


def _legend_handles(cfg, groups, with_null=True, with_ceiling_line=False):
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    types = cfg["types"]
    h = [mpatches.Patch(facecolor=types[t]["color"], edgecolor="black", label=types[t]["label"])
         for t in groups if t in types and t != "null_ceiling"]
    h.append(mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="native objective (*)"))
    if with_null:
        h.append(Line2D([0], [0], color="#767676", ls=":", lw=1.8, label="PCA null"))
    if with_ceiling_line:
        h.append(Line2D([0], [0], color="black", ls="--", lw=1.6, label="test-retest ceiling"))
    return h


def bar_axes(ax, cfg, summary, seed_df, metric, show_legend=True, with_ceiling=False):
    """Bars = mean ± SE, white dots = seeds, grouped and coloured by model type in a fixed thematic order.
    with_ceiling: the test-retest ceiling is its own bar (axis scaled to it) instead of a line / title note."""
    types = cfg["types"]
    groups, order = _order(cfg, summary, with_ceiling)
    x, gap, xs, labels = 0.0, 0.6, [], []
    for t in groups:
        for (tt, r) in [o for o in order if o[0] == t]:
            col = "#4D4D4D" if r["role"] == "ceiling" else types.get(t, {}).get("color", "#777777")
            m, se = r[f"{metric}_mean"], r[f"{metric}_se"]
            native = bool(r.get("objective"))
            ax.bar(x, m, width=0.78, color=col, edgecolor="black", linewidth=1.0, hatch="///" if native else None, zorder=2)
            if not np.isnan(se):
                ax.errorbar(x, m, yerr=se, fmt="none", ecolor="black", elinewidth=1.6, capsize=4, capthick=1.6, zorder=4)
            pts = seed_df[seed_df["model"] == r["model"]][f"test_{metric}"].dropna().to_numpy()
            if len(pts):
                jitter = np.linspace(-0.2, 0.2, len(pts)) if len(pts) > 1 else np.zeros(1)
                ax.scatter(x + jitter, pts, s=18, color="white", edgecolor="black", linewidth=0.8, zorder=5)
            xs.append(x)
            labels.append(r["label"] + (" *" if native else ""))
            x += 1.0
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
        ax.axhline(null.iloc[0][f"{metric}_mean"], color="#767676", ls=":", lw=1.8, zorder=1)
    title_note = ""
    if len(ceiling) and not np.isnan(ceiling.iloc[0][f"{metric}_mean"]):
        c = ceiling.iloc[0][f"{metric}_mean"]
        if c <= ymax * 1.6:
            ax.axhline(c, color="black", ls="--", lw=1.6, zorder=1)
        else:
            title_note = f"\n(test-retest ceiling {c:.2f}, off scale)"
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=12.5)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylabel(METRIC_LABEL[metric], fontsize=15)
    ax.set_title(METRIC_LABEL[metric] + title_note, fontsize=16)
    lo = 0.5 if metric == "avg_rank" else 0.0
    if metric == "pearson":  # skill: tighten the axis to the data range (values sit in a narrow band)
        vals = np.r_[summary.loc[shown, "pearson_mean"].dropna().to_numpy(),
                     seed_df.loc[shown_seeds, "test_pearson"].dropna().to_numpy()]
        lo = max(0.0, np.floor((vals.min() - 0.01) * 50) / 50) if len(vals) else 0.0
    top = ymax * 1.06 if metric != "pearson" else ymax + 0.01
    if len(ceiling) and not title_note:
        top = max(top, ceiling.iloc[0][f"{metric}_mean"] * 1.03)
    ax.set_ylim(lo, top)
    ax.set_xlim(-0.7, xs[-1] + 0.7)
    ax.locator_params(axis="y", nbins=5)
    if show_legend:
        ax.legend(handles=_legend_handles(cfg, groups, with_null=len(null) > 0,
                                          with_ceiling_line=len(ceiling) > 0 and not title_note),
                  fontsize=12, loc="lower left", bbox_to_anchor=(0.0, 1.10 if title_note else 1.03),
                  ncol=4, borderaxespad=0.0, handlelength=1.6, columnspacing=1.2)


def _figure_note(cfg, direction):
    note = "Bars: mean ± SE over seeds (dots = seeds). * native objective (not plain MSE). † extra inputs (anatomy + demographics)."
    if cfg["directions"][direction].get("gate_override"):
        note += " ‡ below the screening gate in this direction, included at full budget by decision."
    dropped = {**(cfg["directions"][direction].get("excluded") or {}), **gate_dropped(cfg, direction)}
    if dropped:
        note += " Not shown: " + ", ".join(model_info(cfg, m).get("label", m) for m in dropped) + " (tables/excluded.md)."
    return note


def figures(cfg, summary, seed_df, out, direction):
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    note = _figure_note(cfg, direction)
    n = len(summary)
    w = max(9.0, 0.55 * n + 2.5)
    written = []
    with plt.rc_context(FIG_RC):
        for metric in cfg["metrics"]:
            fig, ax = plt.subplots(figsize=(w, 6.8), layout="constrained")
            bar_axes(ax, cfg, summary, seed_df, metric, show_legend=False)
            groups, _ = _order(cfg, summary)
            ceil_row = summary[summary["role"] == "ceiling"]
            in_range = len(ceil_row) and ceil_row.iloc[0][f"{metric}_mean"] <= 1.6 * summary.loc[summary["role"] != "ceiling", f"{metric}_mean"].max()
            fig.legend(handles=_legend_handles(cfg, groups, with_ceiling_line=bool(in_range)), loc="outside upper center",
                       ncol=4, fontsize=12, handlelength=1.6, columnspacing=1.2)
            fig.get_layout_engine().set(h_pad=0.15)
            fig.supxlabel("\n".join(textwrap.wrap(note, width=int(w * 11))), fontsize=11, color="#4D4D4D",
                          ha="left", x=0.01)
            fig.savefig(out / f"bars_{metric}.png", dpi=300, facecolor="white")
            plt.close(fig)
            written.append(f"bars_{metric}.png")
        variants = [("bars_all_metrics.png", False)]
        if (summary["role"] == "ceiling").any():
            variants.append(("bars_all_metrics_ceiling.png", True))  # same panel with the test-retest ceiling as a bar
        for name, with_ceiling in variants:
            fig, axes = plt.subplots(2, 2, figsize=(2 * w, 13), layout="constrained")
            for ax, metric in zip(axes.flat, cfg["metrics"]):
                bar_axes(ax, cfg, summary, seed_df, metric, show_legend=False, with_ceiling=with_ceiling)
            groups, _ = _order(cfg, summary, with_ceiling)
            has_ceiling_line = (summary["role"] == "ceiling").any() and not with_ceiling
            fig.legend(handles=_legend_handles(cfg, groups, with_ceiling_line=has_ceiling_line), loc="outside upper center",
                       ncol=5, fontsize=13, handlelength=1.6, columnspacing=1.4)
            fig.supxlabel("\n".join(textwrap.wrap(note, width=int(2 * w * 10))), fontsize=12, color="#4D4D4D",
                          ha="left", x=0.01)
            fig.savefig(out / name, dpi=300, facecolor="white")
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


def _place_labels(ax, points, fontsize=11.5):
    """Label each point at the first free spot from a fixed candidate list (right, left, above, below, then farther
    out), avoiding other labels and markers; boxes estimated in axes fractions. Leader line when placed away."""
    fig = ax.figure
    fig.canvas.draw()
    to_ax = ax.transData + ax.transAxes.inverted()
    bbox = ax.get_window_extent()
    px_pt = fig.dpi / 72.0                      # bbox is in pixels, font size in points
    cw = 0.58 * fontsize * px_pt / bbox.width   # approx. character width (axes fraction)
    ch = 1.3 * fontsize * px_pt / bbox.height   # line height (axes fraction)
    mk = [tuple(to_ax.transform((x, y))) for x, y, _ in points]
    rm_x, rm_y = 8 * px_pt / bbox.width, 8 * px_pt / bbox.height   # marker radius
    placed = []

    def free(x0, y0, w):
        if x0 < 0 or x0 + w > 1.0 or y0 < 0 or y0 + ch > 1.0:
            return False
        for (a0, b0, a1, b1) in placed:
            if x0 < a1 and x0 + w > a0 and y0 < b1 and y0 + ch > b0:
                return False
        return all(not (x0 - rm_x < mx < x0 + w + rm_x and y0 - rm_y < my < y0 + ch + rm_y) for mx, my in mk)

    order = sorted(range(len(points)), key=lambda i: -sum(abs(mk[i][0] - m[0]) < 0.15 and abs(mk[i][1] - m[1]) < 0.08 for m in mk))
    for i in order:
        x, y, lab = points[i]
        px, py = mk[i]
        w = cw * len(lab)
        # candidates: beside (right / left) and centred above / below, at growing offsets; nearest free spot wins
        cands = [(px + 0.015, py - ch / 2), (px - 0.015 - w, py - ch / 2), (px - w / 2, py + 0.02),
                 (px - w / 2, py - 0.02 - ch)]
        for k in range(1, 7):
            d = 0.028 * k
            cands += [(px + 0.015, py - ch / 2 + d), (px + 0.015, py - ch / 2 - d),
                      (px - 0.015 - w, py - ch / 2 + d), (px - 0.015 - w, py - ch / 2 - d),
                      (px - w / 2, py + 0.02 + d), (px - w / 2, py - 0.02 - ch - d)]
        dist = lambda c: np.hypot(min(abs(c[0] - px), abs(c[0] + w - px)), c[1] + ch / 2 - py)
        spot = min((c for c in cands if free(c[0], c[1], w)), key=dist, default=cands[0])
        placed.append((spot[0], spot[1], spot[0] + w, spot[1] + ch))
        far = abs(spot[1] + ch / 2 - py) > ch or not (spot[0] - 0.02 <= px <= spot[0] + w + 0.02 or abs(spot[0] - px) < 0.03)
        ax.annotate(lab, (x, y), xytext=(spot[0], spot[1] + 0.15 * ch), textcoords="axes fraction", fontsize=fontsize,
                    color="#272727", va="bottom",
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.7, shrinkA=2, shrinkB=6) if far else None)


def scatter(cfg, summary, out):
    """Test demeaned r vs average rank, one point per model (mean ± SE), coloured by model type."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    s = summary[summary["role"] != "ceiling"]
    with plt.rc_context(FIG_RC):
        fig, ax = plt.subplots(figsize=(9.5, 7.5), layout="constrained")
        for r in s.to_dict("records"):
            col = "#767676" if r["role"] == "floor" else cfg["types"].get(r["type"], {}).get("color", "#777777")
            native = bool(r["objective"])
            ax.errorbar(r["avg_rank_mean"], r["demeaned_pearson_mean"], xerr=r["avg_rank_se"], yerr=r["demeaned_pearson_se"],
                        fmt="none", ecolor=col, elinewidth=1.8, capsize=3, zorder=2)
            ax.scatter([r["avg_rank_mean"]], [r["demeaned_pearson_mean"]], s=110, marker="s" if native else "o", zorder=3,
                       facecolor="white" if native else col, edgecolor=col if native else "black",
                       linewidth=2.0 if native else 0.8)
        _place_labels(ax, [(r["avg_rank_mean"], r["demeaned_pearson_mean"], r["label"]) for r in s.to_dict("records")])
        ceil = summary[summary["role"] == "ceiling"]
        note = ""
        if len(ceil):
            c = ceil.iloc[0]
            note = f"test-retest ceiling: demeaned r {c['demeaned_pearson_mean']:.2f}, average rank {c['avg_rank_mean']:.2f} (off scale)"
        present = [t for t in cfg["types"] if t != "null_ceiling" and t in set(s["type"])]
        handles = [Line2D([0], [0], marker="o", ls="none", markersize=10, markerfacecolor=cfg["types"][t]["color"],
                          markeredgecolor="black", label=cfg["types"][t]["label"]) for t in present]
        handles += [Line2D([0], [0], marker="o", ls="none", markersize=10, markerfacecolor="#767676",
                           markeredgecolor="black", label="PCA null"),
                    Line2D([0], [0], marker="s", ls="none", markersize=10, markerfacecolor="white",
                           markeredgecolor="#4D4D4D", markeredgewidth=2, label="native objective")]
        ax.legend(handles=handles, fontsize=12, loc="lower right", handlelength=1.2)
        ax.set_xlabel("Average rank (test)", fontsize=15)
        ax.set_ylabel("Demeaned r (test)", fontsize=15)
        ax.set_title("Mean ± SE over seeds" + ("\n" + note if note else ""), fontsize=14)
        fig.savefig(out / "scatter_demeaned_vs_rank.png", dpi=300, facecolor="white")
        plt.close(fig)
    return "scatter_demeaned_vs_rank.png"


# ------------------------------------------------------------------------------------------- panel (main figure)
PANEL_FONTS = FIG_RC["font.family"]  # skill: Helvetica-like sans


def _panel_slots(cfg, summary, order):
    """Rows top to bottom: ("model", row) or ("header", type). `performance`: by test demeaned r, null last;
    `grouped`: config type order with a header per group, by demeaned r within a group, null last."""
    s = summary[summary["role"] != "ceiling"]
    body = s[s["role"] != "floor"].sort_values("demeaned_pearson_mean", ascending=False)
    null = s[s["role"] == "floor"]
    slots = []
    if order == "performance":
        slots = [("model", r) for r in body.to_dict("records")]
    else:
        for t in cfg["types"]:
            g = body[body["type"] == t]
            if len(g):
                slots += [("header", t)] + [("model", r) for r in g.to_dict("records")]
    if len(null):
        slots += ([("header", "null_ceiling")] if order == "grouped" else []) + [("model", r) for r in null.to_dict("records")]
    return slots


def panel(cfg, summary, seed_df, out, direction, order):
    """One row per model, the four metrics side by side on a shared model axis (dot and whisker: mean ± SE, faint
    dots = seeds; open squares = native objective). Ceiling: dashed line when in range, else its value in the title."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    rc = {"font.family": PANEL_FONTS, "font.size": 15, "axes.spines.right": False, "axes.spines.top": False,
          "axes.spines.left": False, "axes.linewidth": 1.8, "legend.frameon": False}
    types = cfg["types"]
    slots = _panel_slots(cfg, summary, order)
    ceil = summary[summary["role"] == "ceiling"]
    null = summary[summary["role"] == "floor"]
    shown = [r for kind, r in slots if kind == "model"]
    n = len(slots)
    y = np.arange(n)[::-1]
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, len(cfg["metrics"]), figsize=(16, 0.40 * n + 2.2), sharey=True, layout="constrained")
        fig.get_layout_engine().set(wspace=0.04)
        for ax, m in zip(axes, cfg["metrics"]):
            k = 0
            for yi, (kind, r) in zip(y, slots):
                if kind != "model":
                    continue
                if k % 2 == 0:
                    ax.axhspan(yi - 0.5, yi + 0.5, color="#F2F2F2", zorder=0, lw=0)
                k += 1
                col = "#767676" if r["role"] == "floor" else types.get(r["type"], {}).get("color", "#777777")
                pts = seed_df.loc[seed_df["model"] == r["model"], f"test_{m}"].dropna().to_numpy()
                ax.scatter(pts, np.full(len(pts), yi), s=16, color=col, alpha=0.35, lw=0, zorder=2)
                mean, se = r[f"{m}_mean"], r[f"{m}_se"]
                if np.isfinite(se):
                    ax.errorbar(mean, yi, xerr=se, fmt="none", ecolor=col, elinewidth=2.2, capsize=4, capthick=2, zorder=3)
                native = bool(r["objective"])
                ax.scatter([mean], [yi], s=90, marker="s" if native else "o", zorder=4,
                           facecolor="white" if native else col, edgecolor=col if native else "black",
                           linewidth=2.0 if native else 0.8)
            vals = np.r_[[r[f"{m}_mean"] for r in shown],
                         seed_df.loc[seed_df["model"].isin([r["model"] for r in shown]), f"test_{m}"].dropna().to_numpy()]
            pad = 0.06 * (vals.max() - vals.min())
            lo, hi = vals.min() - pad, vals.max() + pad
            if len(null):
                ax.axvline(null.iloc[0][f"{m}_mean"], color="#767676", ls=":", lw=1.6, zorder=1)
            if m == "avg_rank":
                ax.axvline(0.5, color="#B0B0B0", ls="-", lw=1.0, zorder=1)
            title = METRIC_LABEL[m]
            if len(ceil) and np.isfinite(ceil.iloc[0][f"{m}_mean"]):
                c = ceil.iloc[0][f"{m}_mean"]
                if c <= hi + 2 * pad:
                    hi = max(hi, c + pad)
                    ax.axvline(c, color="black", ls="--", lw=1.6, zorder=1)
                else:
                    title += f"\n(test-retest: {c:.2f}, off scale)"
            ax.set_xlim(lo, hi)
            ax.set_title(title, fontsize=16)
            ax.tick_params(axis="x", labelsize=13, width=1.5, length=5)
            ax.tick_params(axis="y", length=0)
            ax.locator_params(axis="x", nbins=4)
        labels = [(r["label"] + (" *" if r["objective"] else "")) if kind == "model"
                  else types[r].get("header", types[r]["label"]) for kind, r in slots]
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(labels, fontsize=14)
        axes[0].set_ylim(-0.5, n - 0.5)
        for tick, (kind, r) in zip(axes[0].get_yticklabels(), slots):
            if kind == "header":
                tick.set_fontweight("bold")
                tick.set_color("#4D4D4D" if r == "null_ceiling" else types[r].get("header_color", types[r]["color"]))
        # grouped: the headers name the type colours, so the legend keeps only the encodings
        present = [] if order == "grouped" else [t for t in types if t != "null_ceiling" and any(x["type"] == t for x in shown)]
        handles = [Line2D([0], [0], marker="o", ls="none", markersize=9, markerfacecolor=types[t]["color"],
                          markeredgecolor="black", label=types[t]["label"]) for t in present]
        handles += [Line2D([0], [0], marker="s", ls="none", markersize=9, markerfacecolor="white",
                           markeredgecolor="#4D4D4D", markeredgewidth=2, label="native objective (*)"),
                    Line2D([0], [0], color="#767676", ls=":", lw=1.6, label="PCA null"),
                    Line2D([0], [0], color="#B0B0B0", ls="-", lw=1.0, label="chance (average rank)")]
        if len(ceil):
            handles.append(Line2D([0], [0], color="black", ls="--", lw=1.6, label="test-retest ceiling"))
        fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=13, handlelength=1.8, columnspacing=1.6)
        src, tgt = cfg["directions"][direction]["source"], cfg["directions"][direction]["target"]
        how = "grouped by model type, sorted by demeaned r within group" if order == "grouped" else "sorted by demeaned r"
        marks = "; † extra inputs" + ("; ‡ below screening gate, included" if cfg["directions"][direction].get("gate_override") else "")
        fig.suptitle(f"{src} → {tgt}: test set, mean ± SE over {len(cfg['seeds'])} seeds (faint dots = seeds)\n"
                     f"{how}{marks}", fontsize=15)
        name = f"panel_{order}.png"
        fig.savefig(out / name, dpi=300, facecolor="white")
        plt.close(fig)
    return name


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
    seed_df, summary = tables(cfg, records, direction)
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
    written += [panel(cfg, summary, seed_df, d / "figures", direction, order) for order in ("grouped", "performance")]
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
    gate_failed = gate_dropped(cfg, direction)
    dcfg = cfg["directions"][direction]
    lines = [f"- `{m}`: not run ({why})\n" for m, why in (dcfg.get("excluded") or {}).items()]
    lines += [f"- `{m}`: {why}\n" for m, why in gate_failed.items()]
    lines += [f"- `{m}` (included, ‡): {dcfg['gate_failed'].get(m, 'gate failed')}; override: {why}\n"
              for m, why in (dcfg.get("gate_override") or {}).items()]
    if lines:
        (d / "tables" / "excluded.md").write_text("".join(lines))
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
