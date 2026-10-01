"""
Krakencoder loss-grid report: per-cell summary tables and PNG figures from the collected tables.

    python scripts/experiments/composite_loss/krakencoder/grid_runner.py collect --set grid   # (and --set noise)
    python scripts/experiments/composite_loss/krakencoder/report.py [--set grid]

Inputs:  tables/<set>_seed_records.csv, tables/<set>_epoch_history.csv (and tables/noise_seed_records.csv if present)
Outputs: tables/summary.{csv,md}  per cell x direction: test metrics mean +- SD over seeds, and the paired-by-seed
                                   difference from mse_only (mean +- SE)
         tables/noise.md           retrain variability: init-seed SD (fixed split) vs split-seed SD, kraken_default
         figures/dose_response.png, val_trajectories.png, tradeoff_scatter.png
Missing cells or seeds are skipped (the report also runs on the pilot set).
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PALETTE = {"blue_main": "#0F4D92", "green_3": "#8BCF8B", "red_strong": "#B64342", "teal": "#42949E",
           "violet": "#9A4D8E", "neutral": "#CFCECE"}
TERM_COLORS = {"varmatch": PALETTE["teal"], "correye": PALETTE["blue_main"], "neidist": PALETTE["red_strong"],
               "all": PALETTE["violet"]}
TERM_LABELS = {"varmatch": "var (variance match)", "correye": "correye (~ correye_dm)", "neidist": "neidist",
               "all": "all three"}
METRICS = {"test_demeaned_pearson": "test demeaned r", "test_avg_rank": "test avg rank", "test_top1_acc": "test top-1"}
DIRECTIONS = ("SC->FC", "FC->SC")


def style():
    plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 1.6, "legend.frameon": False, "font.family": "DejaVu Sans"})


def read(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path))) if path.exists() else []


def mean_sd(v):
    v = [x for x in v if x is not None and not math.isnan(x)]
    if not v:
        return math.nan, math.nan, 0
    return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else math.nan, len(v)


def summary(seed_rows: list[dict]) -> list[dict]:
    by = defaultdict(dict)  # (cell, direction) -> seed -> row
    meta = {}
    for r in seed_rows:
        by[(r["combo_id"], r["direction"])][int(r["seed"])] = r
        meta[r["combo_id"]] = r
    out = []
    for (cell, d), seeds in sorted(by.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        row = {"combo_id": cell, "block": meta[cell]["block"], "direction": d, "n_seeds": len(seeds),
               **{f"w_{t}": meta[cell][f"w_{t}"] for t in ("varmatch", "correye", "neidist")}}
        base = by.get(("mse_only", d), {})
        for m in METRICS:
            mu, sd, _ = mean_sd([float(s[m]) for s in seeds.values()])
            row[f"{m}_mean"], row[f"{m}_sd"] = mu, sd
            diffs = [float(seeds[s][m]) - float(base[s][m]) for s in seeds if s in base]
            dm, dsd, n = mean_sd(diffs)
            row[f"{m}_vs_mse_only"] = dm
            row[f"{m}_vs_mse_only_se"] = dsd / math.sqrt(n) if n > 1 else math.nan
        out.append(row)
    return out


def fmt(x, nd=4):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def write_summary(rows: list[dict], tables: Path) -> None:
    if not rows:
        return
    with open(tables / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = []
    for d in DIRECTIONS:
        sub = [r for r in rows if r["direction"] == d]
        if not sub:
            continue
        lines += [f"### {d}", "", "| cell | block | n | test demeaned r | Δ vs mse_only | test avg rank | Δ vs mse_only | test top-1 |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in sorted(sub, key=lambda r: -r["test_demeaned_pearson_mean"]):
            lines.append(
                f"| {r['combo_id']} | {r['block']} | {r['n_seeds']} | {fmt(r['test_demeaned_pearson_mean'])} ± "
                f"{fmt(r['test_demeaned_pearson_sd'])} | {fmt(r['test_demeaned_pearson_vs_mse_only'])} ± "
                f"{fmt(r['test_demeaned_pearson_vs_mse_only_se'])} | {fmt(r['test_avg_rank_mean'], 3)} ± "
                f"{fmt(r['test_avg_rank_sd'], 3)} | {fmt(r['test_avg_rank_vs_mse_only'], 3)} ± "
                f"{fmt(r['test_avg_rank_vs_mse_only_se'], 3)} | {fmt(r['test_top1_acc_mean'], 3)} |")
        lines.append("")
    (tables / "summary.md").write_text("\n".join(lines))


def write_noise(grid_rows: list[dict], noise_rows: list[dict], tables: Path) -> None:
    kd = [r for r in grid_rows if r["combo_id"] == "kraken_default"]
    if not noise_rows:
        return
    lines = ["| direction | metric | init-seed SD (split 0, n) | split-seed SD (init 0, n) |", "|---|---|---|---|"]
    for d in DIRECTIONS:
        init = [r for r in noise_rows + kd if r["direction"] == d and int(r["seed"]) == 0]
        split = [r for r in kd if r["direction"] == d]
        for m, label in METRICS.items():
            _, sd_i, n_i = mean_sd([float(r[m]) for r in init])
            _, sd_s, n_s = mean_sd([float(r[m]) for r in split])
            lines.append(f"| {d} | {label} | {fmt(sd_i)} ({n_i}) | {fmt(sd_s)} ({n_s}) |")
    (tables / "noise.md").write_text("\n".join(lines) + "\n")


def dose_figure(rows: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True)
    levels_of = lambda r, t: float(r[f"w_{t}"])  # noqa: E731
    for i, d in enumerate(DIRECTIONS):
        sub = {r["combo_id"]: r for r in rows if r["direction"] == d}
        base = sub.get("mse_only")
        for j, m in enumerate(("test_demeaned_pearson", "test_avg_rank")):
            ax = axes[i, j]
            for term in ("varmatch", "correye", "neidist", "all"):
                pts = []
                for r in sub.values():
                    w = {t: levels_of(r, t) for t in ("varmatch", "correye", "neidist")}
                    active = [t for t, v in w.items() if v > 0]
                    if r["block"] == "reference":
                        continue
                    if term == "all" and len(active) == 3 and len(set(w.values())) == 1:
                        pts.append((w["varmatch"], r))
                    elif term != "all" and active == [term]:
                        pts.append((w[term], r))
                if base is not None:
                    pts.append((0.0, base))
                pts.sort(key=lambda p: p[0])
                if len(pts) < 2:
                    continue
                x = [p[0] for p in pts]
                y = [p[1][f"{m}_mean"] for p in pts]
                e = [p[1][f"{m}_sd"] / math.sqrt(max(p[1]["n_seeds"], 1)) for p in pts]
                ax.errorbar(x, y, yerr=e, marker="o", lw=2, ms=5, capsize=3, color=TERM_COLORS[term],
                            label=TERM_LABELS[term])
            ref = sub.get("kraken_default")
            if ref is not None:
                ax.axhline(ref[f"{m}_mean"], color=PALETTE["neutral"], lw=2, ls="--", label="paper default")
            ax.set_xscale("symlog", linthresh=0.1)
            ax.set_xticks([0, 0.1, 0.5, 1, 2])
            ax.set_xticklabels(["0", "0.1", "0.5", "1", "2"])
            ax.set_title(f"{d}", fontsize=13)
            ax.set_ylabel(METRICS[m])
            if i == 1:
                ax.set_xlabel("grid level (× anchor)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def trajectory_figure(history: list[dict], out: Path) -> None:
    cells = ["mse_only", "ce_1.0", "nd_1.0", "vm_1.0", "all_1.0", "kraken_default"]
    colors = [PALETTE["neutral"], PALETTE["blue_main"], PALETTE["red_strong"], PALETTE["teal"], PALETTE["violet"], "black"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, d in zip(axes, DIRECTIONS):
        for cell, color in zip(cells, colors):
            by_epoch = defaultdict(list)
            for r in history:
                if r["combo_id"] == cell and r["direction"] == d and int(r.get("random_seed", 0) or 0) == 0:
                    by_epoch[int(r["epoch"])].append(float(r["val_demeaned_pearson"]))
            if not by_epoch:
                continue
            ep = sorted(by_epoch)
            ax.plot(ep, [np.mean(by_epoch[e]) for e in ep], lw=2, color=color, label=cell)
        ax.set_title(d, fontsize=13)
        ax.set_xlabel("epoch")
        ax.set_ylabel("val demeaned r (mean over seeds)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def tradeoff_figure(rows: list[dict], out: Path) -> None:
    block_colors = {"factorial": PALETTE["blue_main"], "dose": PALETTE["teal"], "extension": PALETTE["violet"],
                    "reference": "black"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, d in zip(axes, DIRECTIONS):
        for r in [r for r in rows if r["direction"] == d]:
            ax.scatter(r["test_avg_rank_mean"], r["test_demeaned_pearson_mean"], s=55,
                       color=block_colors.get(r["block"], PALETTE["neutral"]), zorder=3)
            ax.annotate(r["combo_id"], (r["test_avg_rank_mean"], r["test_demeaned_pearson_mean"]), fontsize=8,
                        xytext=(4, 3), textcoords="offset points")
        ax.set_title(d, fontsize=13)
        ax.set_xlabel("test avg rank")
        ax.set_ylabel("test demeaned r")
    for block, c in block_colors.items():
        axes[1].scatter([], [], color=c, label=block)
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="set_name", default="grid")
    args = ap.parse_args()
    tables, figures = HERE / "tables", HERE / "figures"
    figures.mkdir(exist_ok=True)
    seed_rows = read(tables / f"{args.set_name}_seed_records.csv")
    history = read(tables / f"{args.set_name}_epoch_history.csv")
    if not seed_rows:
        raise SystemExit(f"no {tables}/{args.set_name}_seed_records.csv; run grid_runner.py collect --set {args.set_name}")
    style()
    rows = summary(seed_rows)
    write_summary(rows, tables)
    write_noise(seed_rows, read(tables / "noise_seed_records.csv"), tables)
    dose_figure(rows, figures / "dose_response.png")
    trajectory_figure(history, figures / "val_trajectories.png")
    tradeoff_figure(rows, figures / "tradeoff_scatter.png")
    print(f"wrote {tables}/summary.(csv|md), noise.md and {figures}/*.png ({len(rows)} cell x direction rows)")


if __name__ == "__main__":
    main()
