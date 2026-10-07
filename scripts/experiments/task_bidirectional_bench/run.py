"""Spec v3 E0 results: scrape one step's task logs (config.yml steps: parcellation x direction), write per-seed and
summary tables with paired differences against rest, and draw the four metric bar charts (shared model_benchmark bar
code) and the scan-time figure into results/<parcellation>/<direction>/. --compare builds results/compare/ across the
finished steps (condition-order agreement, side-by-side differences vs rest, one demeaned-r panel per step).

    python scripts/experiments/task_bidirectional_bench/run.py --step E0.2 [--allow-partial]
    python scripts/experiments/task_bidirectional_bench/run.py --compare
"""
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments" / "model_benchmark"))
import run as mb  # noqa: E402  (model_benchmark/run.py: parse_log, tables, figures)
sys.path.insert(0, str(HERE))
import autopilot  # noqa: E402  (task_logs / verify: the same done-cell rule the autopilot uses)

METRICS = ("pearson", "demeaned_pearson", "avg_rank", "top1_acc", "mse")


def mb_cfg(cfg, step):
    """The slice of a model_benchmark config that its tables / figures read, with conditions as 'models'."""
    st = cfg["steps"][step]
    return {"campaign": cfg["campaign"], "seeds": cfg["seeds"], "metrics": cfg["metrics"], "types": cfg["types"],
            "reuse": {}, "directions": {st["direction"]: {"models": cfg["conditions"], **cfg["directions"][st["direction"]]}},
            "models": {c: {"label": cfg["labels"][c], "type": "rest" if c.startswith("rest") else "task"}
                       for c in cfg["conditions"]}}


def scrape(cfg, step):
    """One record per verified (condition, seed): the latest task whose best-trial summary ran this step's config,
    direction, parcellation, condition and cohort (autopilot.verify); a mismatched finished log is an error."""
    records = []
    for cond in cfg["conditions"]:
        latest = {}
        for log in autopilot.task_logs(cfg, step, cond):
            if log["summary"] and log["bad"]:
                raise SystemExit(f"{log['path']}: {log['bad']}")
            if log["summary"]:
                latest[log["seed"]] = log
        for seed, log in sorted(latest.items()):
            if seed not in cfg["seeds"]:
                continue
            s = log["summary"]
            m = s.get("metrics") or {}
            records.append({"model": cond, "seed": seed, "log": os.path.basename(log["path"]),
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


def texts(cfg, step):
    """Figure titles / footers for one step (E0.1, Glasser FC -> SC, keeps its original wording)."""
    st = cfg["steps"][step]
    parc, n = st["parcellation"], cfg["cohort"][st["parcellation"]]
    fc_role = "source" if st["direction"] == "fc2sc" else "target"
    arrow = "FC → SC" if st["direction"] == "fc2sc" else "SC → FC"
    title = ("Rest and Task FC → SC Benchmark" if st["direction"] == "fc2sc" else "SC → Rest and Task FC Benchmark")
    parc_note = "" if parc == "Glasser" else f" ({parc})"
    return dict(
        title=f"{title}{parc_note}: CrossModal PCA-PLS Learnable",
        footer=(f"Bars: mean ± SE over 5 seeds (dots = seeds). MSE loss, {parc}; test split "
                f"of the matched {n}-subject cohort; only the {fc_role} FC differs. Pearson r axis truncated; average-rank axis "
                "starts at chance (0.5)."),
        note=(f"Bars: mean ± SE over 5 seeds (dots = seeds). CrossModal PCA-PLS learnable, MSE, {arrow}, "
              f"{parc}, test split of the matched {n}-subject cohort; only the {fc_role} FC differs."),
        scan_title=f"Scan Time and {arrow} Performance{parc_note}: CrossModal PCA-PLS Learnable",
        scan_footer=(f"Points: mean ± SE over 5 seeds. x: {fc_role} FC scan time in minutes (fMRI volumes × TR 0.72 s; log scale). "
                     f"MSE loss, {parc}; test split of the matched {n}-subject cohort; only the {fc_role} FC differs."))



def combined_figure(mcfg, summary, seed_df, path, txt):
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
        fig.suptitle(txt["title"], fontsize=22, fontweight="bold")
        # one legend, in panel A's empty headroom (a figure-level legend collides with the suptitle)
        axes.flat[0].legend(handles=mb._legend_handles(mcfg, groups), loc="upper right", ncol=2, fontsize=16,
                            handlelength=1.8, columnspacing=1.6, frameon=False)
        fig.get_layout_engine().set(h_pad=0.25, w_pad=0.3)
        fig.supxlabel("\n".join(textwrap.wrap(txt["footer"], width=215)), fontsize=13.5, color="#4D4D4D", ha="left", x=0.01)
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)



def scan_time_figure(cfg, mcfg, summary, path, txt, tr=0.72):
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
        fig.supxlabel("\n".join(textwrap.wrap(txt["scan_footer"], width=215)),
            fontsize=13.5, color="#4D4D4D", ha="left", x=0.01)
        fig.supylabel("")
        for ax in axes[1]:
            ax.set_xlabel("Scan time (min, log scale)", fontsize=16)
        fig.get_layout_engine().set(h_pad=0.25, w_pad=0.3)
        fig.suptitle(txt["scan_title"], fontsize=22, fontweight="bold")
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)


def out_dir(cfg, step):
    st = cfg["steps"][step]
    return HERE / cfg["output_dir"] / st["parcellation"] / st["direction"]


def build_step(cfg, step, allow_partial=False):
    mcfg, txt = mb_cfg(cfg, step), texts(cfg, step)
    direction = cfg["steps"][step]["direction"]
    records = scrape(cfg, step)
    have = {(r["model"], r["seed"]) for r in records}
    missing = [(c, s) for c in cfg["conditions"] for s in cfg["seeds"] if (c, s) not in have]
    print(f"{step}: {len(records)} condition x seed results; missing {len(missing)}: {missing[:12]}")
    if not records or (missing and not allow_partial):
        raise SystemExit("incomplete (use --allow-partial for a partial render)")
    seed_df, summary = mb.tables(mcfg, records)
    out = out_dir(cfg, step)
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
    mb._figure_note = lambda *_: txt["note"]
    try:
        written = mb.figures(mcfg, summary, seed_df, out / "figures", direction)
        combined_figure(mcfg, summary, seed_df, out / "figures" / "bars_all_metrics.png", txt)
        scan_time_figure(cfg, mcfg, summary, out / "figures" / "scan_time.png", txt)
    finally:
        mb._legend_handles = _handles
    print("wrote", out / "tables", "and", [str(out / "figures" / w) for w in written])
    return seed_df, summary


def build_compare(cfg):
    """results/compare/: steps with a complete summary.csv; condition-order agreement (Spearman over condition means,
    per metric), side-by-side paired differences vs rest, and one demeaned-r panel per step (2 x 2)."""
    steps = [s for s in cfg["steps"] if (out_dir(cfg, s) / "tables" / "summary.csv").exists()]
    if len(steps) < 2:
        raise SystemExit(f"--compare needs >= 2 finished steps; have {steps}")
    out = HERE / cfg["output_dir"] / "compare"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    summ = {s: pd.read_csv(out_dir(cfg, s) / "tables" / "summary.csv").set_index("model") for s in steps}
    pv = {s: pd.read_csv(out_dir(cfg, s) / "tables" / "paired_vs_rest.csv").set_index("condition") for s in steps}
    name = {s: f"{s} {cfg['steps'][s]['parcellation']} {cfg['steps'][s]['direction']}" for s in steps}
    rows = []
    for metric in cfg["metrics"]:
        means = pd.DataFrame({name[s]: summ[s][f"{metric}_mean"] for s in steps}).loc[cfg["conditions"]]
        rho = means.corr(method="spearman")
        tasks = [c for c in cfg["conditions"] if not c.startswith("rest")]
        rho_t = means.loc[tasks].corr(method="spearman")
        for i, a in enumerate(rho.index):
            for b in rho.columns[i + 1:]:
                rows.append({"metric": metric, "step_a": a, "step_b": b, "spearman_all": rho.loc[a, b],
                             "spearman_tasks": rho_t.loc[a, b]})
    agree = pd.DataFrame(rows)
    agree.to_csv(out / "tables" / "order_agreement.csv", index=False, float_format="%.4g")
    side = pd.DataFrame({f"{name[s]} {k}": pv[s][f"d_demeaned_pearson_{k}"] for s in steps for k in ("mean", "se")}).loc[cfg["conditions"]]
    side.insert(0, "label", [cfg["labels"][c] for c in side.index])
    side.reset_index(names="condition").to_csv(out / "tables" / "paired_vs_rest_demeaned_r.csv", index=False, float_format="%.6g")
    means = pd.DataFrame({name[s]: summ[s]["demeaned_pearson_mean"] for s in steps}).loc[cfg["conditions"]]
    means.insert(0, "label", [cfg["labels"][c] for c in means.index])
    (out / "tables" / "summary.md").write_text(
        "Test demeaned r by condition (mean over 5 seeds):\n\n" + md(means.reset_index(names="condition"), ["label"] + [name[s] for s in steps])
        + "\nCondition-order agreement (Spearman over condition means):\n\n"
        + md(agree, ["metric", "step_a", "step_b", "spearman_all", "spearman_tasks"], fmt=2))
    compare_figure(cfg, steps, out / "figures" / "demeaned_r_by_step.png")
    print("wrote", out)


def compare_figure(cfg, steps, path):
    """One demeaned-r bar panel per step (rows parcellation, columns direction), same bar code as the step figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    parcs = list(dict.fromkeys(cfg["steps"][s]["parcellation"] for s in cfg["steps"]))
    dirs = ["fc2sc", "sc2fc"]
    with plt.rc_context({**mb.FIG_RC, "font.size": 14}):
        fig, axes = plt.subplots(len(parcs), 2, figsize=(18, 6.5 * len(parcs)), layout="constrained", squeeze=False)
        for step, st in cfg["steps"].items():
            ax = axes[parcs.index(st["parcellation"])][dirs.index(st["direction"])]
            if step not in steps:
                ax.set_axis_off()
                ax.set_title(f"{step}: not run yet", fontsize=15)
                continue
            mcfg = mb_cfg(cfg, step)
            seed_df = pd.read_csv(out_dir(cfg, step) / "tables" / "seed_records.csv")
            summary = pd.read_csv(out_dir(cfg, step) / "tables" / "summary.csv")
            for col in ("role", "objective"):
                summary[col] = summary[col].fillna("").astype(str)
                seed_df[col] = seed_df[col].fillna("").astype(str)
            mb.bar_axes(ax, mcfg, summary, seed_df, "demeaned_pearson", show_legend=False)
            ax.set_title(f"{step} · {st['parcellation']} · {'FC → SC' if st['direction'] == 'fc2sc' else 'SC → FC'}",
                         fontsize=16, fontweight="bold")
            ax.set_xticklabels([t.get_text() for t in ax.get_xticklabels()], rotation=35, ha="right", rotation_mode="anchor")
        fig.suptitle("Test Demeaned r by FC Condition, Direction and Parcellation: CrossModal PCA-PLS Learnable",
                     fontsize=19, fontweight="bold")
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", help="config.yml step (E0.1 ... E0.4)")
    ap.add_argument("--compare", action="store_true", help="cross-step comparison into results/compare/")
    ap.add_argument("--allow-partial", action="store_true", help="draw even if some condition x seed cells are missing")
    args = ap.parse_args()
    cfg = yaml.safe_load((HERE / "config.yml").read_text())
    if args.compare:
        build_compare(cfg)
    elif args.step in cfg["steps"]:
        build_step(cfg, args.step, args.allow_partial)
    else:
        raise SystemExit(f"give --step one of {list(cfg['steps'])} or --compare")


if __name__ == "__main__":
    main()
