"""Spec v3 E0 results: scrape the task logs (job name e0_taskfc_<condition>), write per-seed and summary tables with
paired differences against rest, and draw the four metric bar charts (shared model_benchmark bar code).

    python scripts/experiments/task_fc_to_sc/run.py [--allow-partial]
"""
import argparse
import glob
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
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments" / "model_benchmark"))
import run as mb  # noqa: E402  (model_benchmark/run.py: parse_log, tables, figures)

METRICS = ("pearson", "demeaned_pearson", "avg_rank", "top1_acc", "mse")


def mb_cfg(cfg):
    """The slice of a model_benchmark config that its tables / figures read, with conditions as 'models'."""
    return {"campaign": cfg["campaign"], "seeds": cfg["seeds"], "metrics": cfg["metrics"], "types": cfg["types"],
            "reuse": {}, "directions": {cfg["direction"]: {"models": cfg["conditions"]}},
            "models": {c: {"label": cfg["labels"][c], "type": "rest" if c.startswith("rest") else "task"}
                       for c in cfg["conditions"]}}


def scrape(cfg):
    records = []
    for cond in cfg["conditions"]:
        latest = {}
        # exact <job>_<task> suffix so "rest" does not also match rest_S1 logs
        pat = re.compile(rf"{re.escape(cfg['campaign'])}_{re.escape(cond)}_\d+_\d+\.out")
        paths = glob.glob(str(Path(cfg.get("log_dir") or mb.LOG_DIR) / f"{cfg['campaign']}_{cond}_*_*.out"))
        for path in sorted((p for p in paths if pat.fullmatch(os.path.basename(p))), key=os.path.getmtime):
            got = mb.parse_log(path)
            if got:
                latest[got[0]] = (path, got[1])
        for seed, (path, s) in sorted(latest.items()):
            if seed not in cfg["seeds"]:
                continue
            m = s.get("metrics") or {}
            run = s.get("run") or {}
            want = f"scripts/experiments/task_fc_to_sc/configs/{cond}.yml"
            if run.get("config_path") != want or (run.get("source"), run.get("target")) != ("FC", "SC"):
                raise SystemExit(f"{path}: ran {run.get('config_path')} {run.get('source')}->{run.get('target')}, "
                                 f"expected {want} FC->SC")
            records.append({"model": cond, "seed": seed, "log": os.path.basename(path),
                            "selected_val": (s.get("selected_by") or {}).get("value"),
                            "config": (s.get("best_trial") or {}).get("config"), "val": m.get("val") or {},
                            "test": m.get("test") or {}})
    return records


def paired_vs(seed_df, ref):
    base = seed_df[seed_df["model"] == ref].set_index("seed")
    rows = []
    for cond, g in seed_df.groupby("model", sort=False):
        g = g.set_index("seed")
        shared = sorted(set(g.index) & set(base.index))
        r = {"condition": cond, "label": g["label"].iloc[0], "reference": ref, "n_shared_seeds": len(shared)}
        for m in ("pearson", "demeaned_pearson", "avg_rank", "top1_acc"):
            d = (g.loc[shared, f"test_{m}"] - base.loc[shared, f"test_{m}"]).dropna()
            r[f"d_{m}_mean"] = d.mean() if len(d) else np.nan
            r[f"d_{m}_se"] = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else np.nan
        rows.append(r)
    return pd.DataFrame(rows).sort_values("d_demeaned_pearson_mean", ascending=False)


def md(df, cols, fmt=4):
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "".join("| " + " | ".join(f"{r[c]:.{fmt}f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |\n"
                   for r in df.to_dict("records"))
    return head + body


TITLE = "Rest and Task FC → SC Benchmark: CrossModal PCA-PLS Learnable"
FOOTER = ("Bars: mean ± SE over 5 seeds (dots = seeds). MSE loss, Glasser; test split "
          "of the matched 917-subject cohort; only the source FC differs. Pearson r axis truncated; average-rank axis "
          "starts at chance (0.5).")


def combined_figure(mcfg, summary, seed_df, path):
    """bars_all_metrics.png in the benchmark 2 x 2 layout, restyled for reading (scientific-figure-making skill:
    16 pt base, heavier axes and bar edges, panel letters, one legend in panel A, experiment title, setup in the footer)."""
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rc = {**mb.FIG_RC, "font.size": 16, "axes.linewidth": 2.2, "xtick.major.width": 2.0, "ytick.major.width": 2.0,
          "ytick.major.size": 6}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 2, figsize=(19, 13.5), layout="constrained")
        for letter, ax, metric in zip("ABCD", axes.flat, mcfg["metrics"]):
            mb.bar_axes(ax, mcfg, summary, seed_df, metric, show_legend=False)
            for b in ax.patches:
                b.set_linewidth(1.6)
            for c in ax.collections:  # seed dots (PathCollection); error bars are LineCollections
                if hasattr(c, "set_sizes"):
                    c.set_sizes([34])
            ax.set_title(mb.METRIC_LABEL[metric], fontsize=19, fontweight="bold", pad=10)
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelsize=15, length=6)
            ax.set_xticklabels([t.get_text() for t in ax.get_xticklabels()], rotation=35, ha="right",
                               rotation_mode="anchor", fontsize=15.5)
            ax.text(-0.08, 1.04, letter, transform=ax.transAxes, fontsize=22, fontweight="bold", va="bottom")
        groups, _ = mb._order(mcfg, summary)
        fig.suptitle(TITLE, fontsize=22, fontweight="bold")
        # one legend, in panel A's empty headroom (a figure-level legend collides with the suptitle)
        axes.flat[0].legend(handles=mb._legend_handles(mcfg, groups), loc="upper right", ncol=2, fontsize=16,
                            handlelength=1.8, columnspacing=1.6, frameon=False)
        fig.get_layout_engine().set(h_pad=0.25, w_pad=0.3)
        fig.supxlabel("\n".join(textwrap.wrap(FOOTER, width=215)), fontsize=13.5, color="#4D4D4D", ha="left", x=0.01)
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)



def scan_time_figure(cfg, mcfg, summary, path, tr=0.72):
    """Each metric against the source FC's scan time (log scale), 2 x 2 like bars_all_metrics.png. Dashed: least-squares
    fit of the metric on log scan time over all conditions; Pearson r (on log scan time) and Spearman rho over all
    conditions and over tasks only."""
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator
    s = summary.set_index("model")
    conds = cfg["conditions"]
    minutes = pd.Series({c: cfg["volumes"][c] * tr / 60.0 for c in conds})
    tasks = [c for c in conds if not c.startswith("rest")]
    rc = {**mb.FIG_RC, "font.size": 16, "axes.linewidth": 2.2, "xtick.major.width": 2.0, "ytick.major.width": 2.0,
          "xtick.major.size": 6, "ytick.major.size": 6}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 2, figsize=(19, 13.5), layout="constrained")
        for letter, ax, metric in zip("ABCD", axes.flat, mcfg["metrics"]):
            m, se = s.loc[conds, f"{metric}_mean"], s.loc[conds, f"{metric}_se"]
            lx = np.log(minutes[conds].to_numpy())
            b, a = np.polyfit(lx, m.to_numpy(), 1)
            grid = np.linspace(lx.min() - 0.1, lx.max() + 0.1, 50)
            ax.plot(np.exp(grid), a + b * grid, color="#767676", ls="--", lw=1.8, zorder=1)
            for c in conds:
                col = cfg["types"]["rest" if c.startswith("rest") else "task"]["color"]
                ax.errorbar(minutes[c], m[c], yerr=se[c], fmt="o", ms=11, color=col, mec="black", mew=1.2,
                            ecolor="black", elinewidth=1.6, capsize=4, capthick=1.6, zorder=3)
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(FixedLocator([4, 6, 10, 20, 30, 60]))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xticklabels(["4", "6", "10", "20", "30", "60"])
            ax.set_xlim(3.4, 75)
            lo, hi = (m - se).min(), (m + se).max()
            ax.set_ylim(lo - 0.22 * (hi - lo), hi + 0.22 * (hi - lo))
            ax.locator_params(axis="y", nbins=5)
            ax.tick_params(labelsize=15)
            ax.set_title(mb.METRIC_LABEL[metric], fontsize=19, fontweight="bold", pad=10)
            ax.text(-0.08, 1.04, letter, transform=ax.transAxes, fontsize=22, fontweight="bold", va="bottom")
            logt = np.log(minutes)
            corr = {k: (m.corr(logt[conds], method=k), m[tasks].corr(logt[tasks], method=k))
                    for k in ("pearson", "spearman")}
            ax.text(0.98, 0.04, f"Pearson r (log time): all {corr['pearson'][0]:.2f} \u00b7 tasks {corr['pearson'][1]:.2f}\n"
                                f"Spearman \u03c1: all {corr['spearman'][0]:.2f} \u00b7 tasks {corr['spearman'][1]:.2f}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=14.5, color="#333333", linespacing=1.4)
            mb._place_labels(ax, [(minutes[c], m[c], cfg["labels"][c]) for c in conds], fontsize=12.5)
        groups, _ = mb._order(mcfg, summary)
        handles = mb._legend_handles(mcfg, groups)
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color="#767676", ls="--", lw=1.8, label="fit on log scan time"))
        axes.flat[0].legend(handles=handles, loc="upper left", fontsize=15, handlelength=1.8, frameon=False)
        fig.supxlabel("\n".join(textwrap.wrap(
            "Points: mean ± SE over 5 seeds. x: source FC scan time in minutes (fMRI volumes × TR 0.72 s; log scale). "
            "MSE loss, Glasser; test split of the matched 917-subject cohort; only the source FC differs.", width=215)),
            fontsize=13.5, color="#4D4D4D", ha="left", x=0.01)
        fig.supylabel("")
        for ax in axes[1]:
            ax.set_xlabel("Scan time (min, log scale)", fontsize=16)
        fig.get_layout_engine().set(h_pad=0.25, w_pad=0.3)
        fig.suptitle("Scan Time and FC → SC Performance: CrossModal PCA-PLS Learnable", fontsize=22, fontweight="bold")
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-partial", action="store_true", help="draw even if some condition x seed cells are missing")
    args = ap.parse_args()
    cfg = yaml.safe_load((HERE / "config.yml").read_text())
    mcfg = mb_cfg(cfg)
    records = scrape(cfg)
    have = {(r["model"], r["seed"]) for r in records}
    missing = [(c, s) for c in cfg["conditions"] for s in cfg["seeds"] if (c, s) not in have]
    print(f"{len(records)} condition x seed results; missing {len(missing)}: {missing[:12]}")
    if not records or (missing and not args.allow_partial):
        raise SystemExit("incomplete (use --allow-partial for a partial render)")
    seed_df, summary = mb.tables(mcfg, records)
    out = HERE / cfg["output_dir"]
    (out / "tables").mkdir(parents=True, exist_ok=True)
    seed_df.to_csv(out / "tables" / "seed_records.csv", index=False, float_format="%.6g")
    summary.to_csv(out / "tables" / "summary.csv", index=False, float_format="%.6g")
    pd.DataFrame([{"condition": r["model"], "seed": r["seed"], "log": r["log"], "selected_val": r["selected_val"],
                   "config": r["config"]} for r in records]).to_csv(out / "tables" / "runs.csv", index=False)
    pv = paired_vs(seed_df, cfg["reference"])
    pv.to_csv(out / "tables" / "paired_vs_rest.csv", index=False, float_format="%.6g")
    s = summary.sort_values("demeaned_pearson_mean", ascending=False)
    (out / "tables" / "summary.md").write_text(
        md(s, ["label", "n_seeds", "pearson_mean", "demeaned_pearson_mean", "avg_rank_mean", "top1_acc_mean"])
        + "\nPaired vs " + cfg["reference"] + " (mean over shared seeds):\n\n"
        + md(pv, ["label", "n_shared_seeds", "d_pearson_mean", "d_demeaned_pearson_mean", "d_avg_rank_mean", "d_top1_acc_mean"]))
    for col in ("role", "objective"):
        summary[col] = summary[col].fillna("").astype(str)
    # E0 has no native-objective / null / extra-input / gated bars: keep only the type swatches and replace the footer
    _handles = mb._legend_handles
    mb._legend_handles = lambda c, g, **kw: [h for h in _handles(c, g, **kw)
                                             if h.get_label() not in ("native objective (*)", "PCA null")]
    mb._figure_note = lambda *_: ("Bars: mean ± SE over 5 seeds (dots = seeds). CrossModal PCA-PLS learnable, MSE, FC → SC, "
                                  "Glasser, test split of the matched 917-subject cohort; only the source FC differs.")
    written = mb.figures(mcfg, summary, seed_df, out / "figures", cfg["direction"])
    combined_figure(mcfg, summary, seed_df, out / "figures" / "bars_all_metrics.png")
    scan_time_figure(cfg, mcfg, summary, out / "figures" / "scan_time.png")
    print("wrote", out / "tables", "and", [str(out / "figures" / w) for w in written])


if __name__ == "__main__":
    main()
