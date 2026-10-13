"""Spec v3 E1: covariate family ablation — pilot gate, tables, figures.

  python scripts/experiments/covariate_ablation/run.py gate      # E1.2 pilot gate (seed 0: none, full, e2_ref)
  python scripts/experiments/covariate_ablation/run.py           # tables + figures (needs every run; --allow_partial)

Reads runs/<sc_metric>/<direction>/<condition>/seed<S>.json (run_cell.py). Writes results/tables/*.{csv,md} and
results/figures/*.png. Deltas are paired by seed (same split, same frozen config apart from cov_sources).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments" / "model_benchmark"))
import run as mb  # noqa: E402  (FIG_RC, METRIC_LABEL)
from conditions import grid_conditions, load_config  # noqa: E402

DIR_LABEL = {"fc2sc": "FC → SC", "sc2fc": "SC → FC"}
FAMILY_LABEL = {"demo": "Demographics", "fs_vol": "FS volumes", "fs_area": "FS cortical area",
                "fs_thick": "FS cortical thickness"}
COLORS = {"only": "#0F4D92", "unique": "#C2410C", "invnodevol": "#0F4D92", "unnorm": "#C2410C",
          "none": "#9CA3AF", "full": "#0F4D92", "learnable": "#4B5563"}
FIG_METRICS = ("demeaned_pearson", "top1_acc")


def load_runs(cfg):
    rows = []
    for p in sorted((HERE / cfg["runs_dir"]).glob("*/*/*/seed*.json")):
        r = json.loads(p.read_text())
        row = {k: r[k] for k in ("sc_metric", "direction", "condition", "seed")}
        row.update({f"test_{m}": r["test"].get(m) for m in cfg["metrics"]})
        row.update({f"val_{m}": r["val"].get(m) for m in cfg["metrics"]})
        row["train_s"] = r.get("train_s")
        rows.append(row)
    return pd.DataFrame(rows)


def expected_cells(cfg):
    cells = []
    for run in cfg["runs"]:
        conds = grid_conditions(cfg) if run["conditions"] == "all" else run["conditions"]
        cells += [(run["sc_metric"], d, c, s) for d in run["directions"] for c in conds for s in cfg["seeds"]]
    return cells


def gate(cfg, df):
    pc = cfg["pilot"]
    ok_all, out = True, {}
    for d in cfg["directions"]:
        s = df[(df.sc_metric == "invnodevol") & (df.direction == d) & (df.seed == pc["seed"])].set_index("condition")
        missing = [c for c in pc["conditions"] if c not in s.index]
        if missing:
            out[d] = {"pass": False, "missing": missing}
            ok_all = False
            continue
        ref = pc["e2_seed0_test_demeaned_r"][d]
        res = {c: {"test_demeaned_r": round(float(s.loc[c, "test_demeaned_pearson"]), 4),
                   "test_top1": round(float(s.loc[c, "test_top1_acc"]), 3),
                   "val_demeaned_r": round(float(s.loc[c, "val_demeaned_pearson"]), 4),
                   "train_s": s.loc[c, "train_s"]} for c in pc["conditions"]}
        checks = {"full_within_tol": abs(res["full"]["test_demeaned_r"] - ref) <= pc["tolerance"],
                  "e2_ref_within_tol": abs(res["e2_ref"]["test_demeaned_r"] - ref) <= pc["tolerance"],
                  "all_finite": bool(np.isfinite(s.loc[pc["conditions"], [f"test_{m}" for m in cfg["metrics"]]].values).all())}
        if d in pc["none_min_val_demeaned_r"]:
            checks["none_val_above_min"] = res["none"]["val_demeaned_r"] > pc["none_min_val_demeaned_r"][d]
        out[d] = {"e2_seed0": ref, "runs": res, "checks": checks, "pass": all(checks.values())}
        ok_all &= out[d]["pass"]
    out["pass"] = bool(ok_all)
    print(json.dumps(out, indent=1, default=float))
    (HERE / cfg["runs_dir"] / "pilot_gate.json").write_text(json.dumps(out, indent=1, default=float))
    return ok_all


def summarize(df, cfg):
    g = df.groupby(["sc_metric", "direction", "condition"])
    rows = []
    for key, s in g:
        row = dict(zip(("sc_metric", "direction", "condition"), key), n=len(s))
        for m in cfg["metrics"]:
            row[f"{m}_mean"], row[f"{m}_std"] = s[f"test_{m}"].mean(), s[f"test_{m}"].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def paired(df, cfg, a, b, sc_metric, direction, m):
    s = df[(df.sc_metric == sc_metric) & (df.direction == direction)]
    x = s[s.condition == a].set_index("seed")[f"test_{m}"]
    y = s[s.condition == b].set_index("seed")[f"test_{m}"]
    return (x - y).dropna()


def deltas(df, cfg):
    rows = []
    fams = list(cfg["families"])
    for (sc_metric, direction), _ in df.groupby(["sc_metric", "direction"]):
        contrasts = [("full − none", "full", "none", "all")]
        if sc_metric == "invnodevol":
            contrasts += [(f"only − none", f"only_{f}", "none", f) for f in fams]
            contrasts += [(f"full − drop", "full", f"drop_{f}", f) for f in fams]
        for label, a, b, fam in contrasts:
            for m in cfg["metrics"]:
                d = paired(df, cfg, a, b, sc_metric, direction, m)
                if len(d) == 0:
                    continue
                rows.append({"sc_metric": sc_metric, "direction": direction, "contrast": label, "family": fam,
                             "metric": m, "n": len(d), "mean": d.mean(), "std": d.std(ddof=1) if len(d) > 1 else np.nan,
                             "n_positive": int((d > 0).sum()), "per_seed": " ".join(f"{v:+.4f}" for v in d.values)})
    return pd.DataFrame(rows)


def e2_rows(cfg):
    rows = []
    for d in cfg["directions"]:
        recs = json.loads((REPO_ROOT / cfg["frozen"]["records"].format(direction=d)).read_text())
        for r in recs:
            if r["model"] in ("CrossModal_PCA_PLS_learnable", cfg["model"]):
                rows.append({"direction": d, "model": r["model"], "seed": int(r["seed"]),
                             **{f"test_{m}": r["test"].get(m) for m in cfg["metrics"]}})
    return pd.DataFrame(rows)


def md(df, cols, fmt=4):
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join(f"{r[c]:.{fmt}f}" if isinstance(r[c], float) else str(r[c]) for c in cols)
                          + " |\n" for r in df.to_dict("records"))


def plt_ctx():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt, plt.rc_context({**mb.FIG_RC, "font.size": 14})


def _dots(ax, x, d, color):
    ax.scatter(np.full(len(d), x) + np.linspace(-0.04, 0.04, len(d)), d, color=color, s=28, edgecolor="black",
               lw=0.6, zorder=3)
    ax.errorbar(x, d.mean(), yerr=d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else 0, fmt="_", ms=20, mew=2.5,
                color="black", capsize=5, zorder=4)


def fig_families(df, cfg, path):
    """Fig 1: per family, only − none (gain on its own) and full − drop (unique gain), both directions."""
    plt, ctx = plt_ctx()
    fams = list(cfg["families"])
    with ctx:
        fig, axes = plt.subplots(len(FIG_METRICS), len(cfg["directions"]), figsize=(13, 9.5), layout="constrained",
                                 squeeze=False)
        for j, d in enumerate(cfg["directions"]):
            for i, m in enumerate(FIG_METRICS):
                ax = axes[i, j]
                for k, f in enumerate(fams):
                    _dots(ax, k - 0.18, paired(df, cfg, f"only_{f}", "none", "invnodevol", d, m), COLORS["only"])
                    _dots(ax, k + 0.18, paired(df, cfg, "full", f"drop_{f}", "invnodevol", d, m), COLORS["unique"])
                full = paired(df, cfg, "full", "none", "invnodevol", d, m)
                if len(full):
                    ax.axhline(full.mean(), color=COLORS["full"], ls="--", lw=1.5)
                ax.axhline(0, color="#555555", lw=1)
                ax.set_xticks(range(len(fams)), [FAMILY_LABEL[f].replace(" ", "\n", 1) for f in fams], fontsize=12)
                ax.set_title(f"{DIR_LABEL[d]}: Δ {mb.METRIC_LABEL[m]}", fontweight="bold")
        handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS["only"], mec="black", label="family alone − none"),
                   plt.Line2D([], [], marker="o", ls="", color=COLORS["unique"], mec="black", label="full − family dropped"),
                   plt.Line2D([], [], ls="--", color=COLORS["full"], label="full − none")]
        axes[0, 0].legend(handles=handles, frameon=False, fontsize=11)
        fig.suptitle("Covariate Family Contributions, PCA-PLS + Covariates (test, frozen E2.2 configs)", fontweight="bold")
        fig.supxlabel("Dots: seeds 0–4 (paired); bar: mean ± SE. Glasser, invnodevol SC.", fontsize=11, color="#4D4D4D")
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)


def fig_normalisation(df, cfg, path):
    """Fig 2: FC → SC full − none gain with normalised vs unnormalised SC target."""
    plt, ctx = plt_ctx()
    with ctx:
        fig, axes = plt.subplots(1, len(FIG_METRICS), figsize=(11, 5), layout="constrained")
        for ax, m in zip(axes, FIG_METRICS):
            for k, sc in enumerate(("invnodevol", "unnorm")):
                d = paired(df, cfg, "full", "none", sc, "fc2sc", m)
                if len(d):
                    _dots(ax, k, d, COLORS[sc])
            ax.axhline(0, color="#555555", lw=1)
            ax.set_xticks([0, 1], ["invnodevol SC\n(E2.2 target)", "unnormalised SC\n(sift count)"], fontsize=12)
            ax.set_xlim(-0.6, 1.6)
            ax.set_title(f"Δ {mb.METRIC_LABEL[m]} (full − none)", fontweight="bold")
        fig.suptitle("FC → SC Covariate Gain With vs Without Node-Volume Normalisation", fontweight="bold")
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)


def fig_context(df, e2, cfg, path):
    """Fig 3: none, full (this run) and PCA-PLS learnable (E2.2) per direction."""
    plt, ctx = plt_ctx()
    arms = [("PCA-PLS learnable (E2.2)", "learnable"), ("PCA-PLS + cov, none", "none"), ("PCA-PLS + cov, full", "full")]
    with ctx:
        fig, axes = plt.subplots(1, len(FIG_METRICS), figsize=(12, 5), layout="constrained")
        for ax, m in zip(axes, FIG_METRICS):
            for j, d in enumerate(cfg["directions"]):
                for k, (label, arm) in enumerate(arms):
                    if arm == "learnable":
                        v = e2[(e2.direction == d) & (e2.model == "CrossModal_PCA_PLS_learnable")][f"test_{m}"]
                    else:
                        v = df[(df.sc_metric == "invnodevol") & (df.direction == d) & (df.condition == arm)][f"test_{m}"]
                    x = j + (k - 1) * 0.27
                    ax.bar(x, v.mean(), width=0.25, color=COLORS[arm], edgecolor="black", lw=1,
                           label=label if j == 0 else None)
                    ax.errorbar(x, v.mean(), yerr=v.std(ddof=1), color="black", capsize=4, lw=1.5)
            ax.set_xticks(range(len(cfg["directions"])), [DIR_LABEL[d] for d in cfg["directions"]])
            ax.set_title(mb.METRIC_LABEL[m], fontweight="bold")
        axes[0].legend(frameon=False, fontsize=11)
        fig.suptitle("Covariates vs No Covariates (test, mean ± SD over seeds 0–4)", fontweight="bold")
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)


def report(cfg, df, allow_partial):
    have = set(map(tuple, df[["sc_metric", "direction", "condition", "seed"]].values.tolist()))
    missing = [c for c in expected_cells(cfg) if c not in have]
    if missing and not allow_partial:
        raise SystemExit(f"{len(missing)} runs missing (e.g. {missing[:3]}); pass --allow_partial")
    out = HERE / cfg["output_dir"]
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    df.sort_values(["sc_metric", "direction", "condition", "seed"]).to_csv(out / "tables" / "seed_records.csv", index=False)
    summ = summarize(df, cfg)
    summ.to_csv(out / "tables" / "summary.csv", index=False)
    cols = ["sc_metric", "direction", "condition", "n"] + [f"{m}_{s}" for m in ("demeaned_pearson", "top1_acc", "avg_rank")
                                                            for s in ("mean", "std")]
    (out / "tables" / "summary.md").write_text(md(summ[cols], cols))
    dl = deltas(df, cfg)
    dl.to_csv(out / "tables" / "paired_deltas.csv", index=False)
    dcols = ["sc_metric", "direction", "contrast", "family", "metric", "n", "mean", "std", "n_positive"]
    (out / "tables" / "paired_deltas.md").write_text(
        md(dl[dl.metric.isin(("demeaned_pearson", "top1_acc", "avg_rank"))][dcols], dcols))
    fig_families(df, cfg, out / "figures" / "fig1_family_contributions.png")
    fig_normalisation(df, cfg, out / "figures" / "fig2_normalisation_control.png")
    fig_context(df, e2_rows(cfg), cfg, out / "figures" / "fig3_context.png")
    print(f"wrote tables + figures to {out}; {len(df)} runs, {len(missing)} missing; "
          f"train time {df.train_s.sum() / 3600:.2f} h")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", nargs="?", default="report", choices=["report", "gate"])
    p.add_argument("--allow_partial", action="store_true")
    a = p.parse_args()
    cfg = load_config()
    df = load_runs(cfg)
    if a.cmd == "gate":
        sys.exit(0 if gate(cfg, df) else 1)
    report(cfg, df, a.allow_partial)


if __name__ == "__main__":
    main()
