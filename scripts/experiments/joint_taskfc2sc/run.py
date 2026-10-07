"""Spec v3 E2 results: read runs/<version>/seed<S>.csv (grid.py), select per seed within each k_task slice on val
demeaned r, and write tables + figures into results/<parcellation>/{raw,predictive_resid,compare}/.

    python scripts/experiments/joint_taskfc2sc/run.py [--allow-partial]

Both versions share the rest-only (k_task = 0) cells, which grid.py scores once under raw.
Figures (spec v3 E2 outputs):
  compare/ablation_curve.png       1  test metrics vs k_task (val-selected per slice), both versions
  <version>/heatmap_rest_task.png  2  k_rest x k_task: val and test demeaned r (val-selected over k_sc, n_pls)
  <version>/weave.png              3  parallel coordinates over every configuration (seed means)
  compare/dimensionality.png       4  test demeaned r vs total input dimension, by k_task
  compare/paired_delta.png         5  per-seed paired difference vs rest-only, per metric
  compare/block_contribution.png   6  block VIP share of each version's val-best with-task config
  compare/selected_k_sc.png        7  selected SC PCs by k_task
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments" / "model_benchmark"))
import run as mb  # noqa: E402  (FIG_RC, METRIC_LABEL)

AXES = ("k_rest", "k_task", "k_sc", "n_pls")
COLORS = {"raw": "#0F4D92", "predictive_resid": "#C2410C"}
N_TRAIN_NOTE = "n_train ≈ 659"


def load(cfg, allow_partial):
    frames = []
    for v in cfg["versions"]:
        for seed in cfg["seeds"]:
            p = HERE / cfg["runs_dir"] / v / f"seed{seed}.csv"
            if p.exists():
                frames.append(pd.read_csv(p))
            elif not allow_partial:
                raise SystemExit(f"missing {p} (use --allow-partial)")
    df = pd.concat(frames, ignore_index=True)
    rest_only = df[(df["version"] == "raw") & (df["k_task"] == 0)]
    df = pd.concat([df, rest_only.assign(version="predictive_resid")], ignore_index=True)  # shared baseline
    g = cfg["grid"]
    for v in cfg["versions"]:
        for seed in cfg["seeds"]:
            n = len(df[(df["version"] == v) & (df["seed"] == seed)].drop_duplicates(list(AXES)))
            want = sum(1 for kr in g["k_rest"] for kt in g["k_task"] for ks in g["k_sc"] for k in g["n_pls"]
                       if k <= kr + 7 * kt)
            if n != want and not allow_partial:
                raise SystemExit(f"{v} seed {seed}: {n} of {want} configurations")
    return df


def select(df, cfg, by=("version", "seed", "k_task")):
    """Per group, the configuration with the best val demeaned r (never test)."""
    key = cfg["selection"]["metric"]
    idx = df.groupby(list(by))[key].idxmax()
    return df.loc[idx].reset_index(drop=True)


def summarize(sel, cfg):
    rows = []
    for (v, kt), g in sel.groupby(["version", "k_task"]):
        r = {"version": v, "k_task": kt, "n_seeds": len(g)}
        base = sel[(sel["version"] == v) & (sel["k_task"] == 0)].set_index("seed")
        gg = g.set_index("seed")
        for m in cfg["metrics"]:
            col = f"test_{m}"
            r[f"{m}_mean"], r[f"{m}_se"] = gg[col].mean(), gg[col].std(ddof=1) / math.sqrt(len(gg)) if len(gg) > 1 else np.nan
            d = (gg[col] - base[col].reindex(gg.index)).dropna()
            r[f"d_{m}_mean"] = d.mean()
            r[f"d_{m}_se"] = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else np.nan
        for a in ("k_rest", "k_sc", "n_pls"):
            r[f"{a}_selected"] = ",".join(str(int(x)) for x in gg[a])
        rows.append(r)
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


def fig_ablation(summ, cfg, path):
    plt, ctx = plt_ctx()
    kts = cfg["grid"]["k_task"]
    with ctx:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), layout="constrained")
        for ax, m in zip(axes.flat, cfg["metrics"]):
            for i, v in enumerate(cfg["versions"]):
                s = summ[summ["version"] == v].set_index("k_task").reindex(kts)
                x = np.arange(len(kts)) + (i - 0.5) * 0.08
                ax.errorbar(x, s[f"{m}_mean"], yerr=s[f"{m}_se"], fmt="-o", ms=8, lw=2, capsize=4, color=COLORS[v],
                            label=cfg["versions"][v]["label"], mec="black", mew=1)
            ax.set_xticks(range(len(kts)), [str(k) for k in kts])
            ax.set_xlabel("task PCs per condition (k_task; 0 = rest only)")
            ax.set_title(mb.METRIC_LABEL[m], fontweight="bold")
        axes.flat[0].legend(frameon=False)
        fig.suptitle("Adding Task-FC Blocks to Rest FC → SC: Multi-Block PCA-PLS (test, val-selected per k_task)",
                     fontweight="bold")
        fig.supxlabel("Points: mean ± SE over 5 seeds; each point is that k_task slice's val-selected config "
                      "(over k_rest, k_sc, n_pls). Glasser, matched 917-subject cohort.", fontsize=11, color="#4D4D4D")
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)


def fig_heatmap(df, cfg, v, path):
    plt, ctx = plt_ctx()
    sel = select(df[df["version"] == v], cfg, by=("seed", "k_rest", "k_task"))
    g = cfg["grid"]
    with ctx:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
        for ax, split in zip(axes, ("val", "test")):
            mat = sel.groupby(["k_rest", "k_task"])[f"{split}_demeaned_pearson"].mean().unstack().reindex(
                index=g["k_rest"], columns=g["k_task"])
            im = ax.imshow(mat.values, cmap="viridis", aspect="auto", origin="lower")
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, f"{mat.values[i, j]:.3f}", ha="center", va="center", fontsize=11,
                            color="white" if mat.values[i, j] < np.nanmean(mat.values) else "black")
            ax.set_xticks(range(len(g["k_task"])), g["k_task"])
            ax.set_yticks(range(len(g["k_rest"])), g["k_rest"])
            ax.set_xlabel("k_task")
            ax.set_ylabel("k_rest")
            ax.set_title(f"{split} demeaned r", fontweight="bold")
            fig.colorbar(im, ax=ax, shrink=0.85)
        fig.suptitle(f"{cfg['versions'][v]['label']}: rest × task PCs (val-selected over k_sc, n_pls; seed mean)",
                     fontweight="bold")
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)


def fig_weave(df, cfg, v, path):
    """Parallel coordinates: one line per configuration (seed means), coloured by val demeaned r; the val-best
    configuration of each k_task slice (seed means) drawn thick."""
    plt, ctx = plt_ctx()
    from matplotlib import cm, colors
    d = df[df["version"] == v].groupby(list(AXES)).agg(
        n_inputs=("n_inputs", "first"), val=("val_demeaned_pearson", "mean"), test=("test_demeaned_pearson", "mean")
    ).reset_index().sort_values("val")
    cols = ["k_rest", "k_task", "n_inputs", "k_sc", "n_pls", "val", "test"]
    labels = ["rest PCs", "task PCs / cond.", "total inputs D", "SC PCs", "PLS comps", "val dem. r", "test dem. r"]

    def norm(c, x):
        if c in ("val", "test"):
            lo, hi = d[["val", "test"]].min().min(), d[["val", "test"]].max().max()
            return (x - lo) / (hi - lo)
        lx = np.log2(np.where(x > 0, x, 4.0))  # k_task = 0 drawn below 8
        lo, hi = np.log2(4.0 if c == "k_task" else d[c].min()), np.log2(d[c].max())
        return (lx - lo) / (hi - lo)

    y = np.stack([norm(c, d[c].to_numpy(dtype=float)) for c in cols], axis=1)
    winners = d.loc[d.groupby("k_task")["val"].idxmax()]
    with ctx:
        fig, ax = plt.subplots(figsize=(15, 7), layout="constrained")
        cmap, nrm = cm.viridis, colors.Normalize(d["val"].min(), d["val"].max())
        for row, val in zip(y, d["val"]):
            ax.plot(range(len(cols)), row, color=cmap(nrm(val)), lw=0.6, alpha=0.35)
        for _, w in winners.iterrows():
            yy = [norm(c, np.array([float(w[c])]))[0] for c in cols]
            ax.plot(range(len(cols)), yy, color="black", lw=2.6)
            ax.text(len(cols) - 0.95, yy[-1], f"k_task={int(w['k_task'])}: {w['test']:.3f}", fontsize=10, va="center")
        for j, c in enumerate(cols):
            ax.axvline(j, color="#555555", lw=1)
            if c in ("val", "test"):
                lo, hi = d[["val", "test"]].min().min(), d[["val", "test"]].max().max()
                for t in np.linspace(lo, hi, 5):
                    ax.text(j - 0.04, (t - lo) / (hi - lo), f"{t:.3f}", ha="right", va="center", fontsize=9)
            else:
                ticks = [t for t in (64, 128, 256, 512, 960) if d[c].min() <= t <= d[c].max()] if c == "n_inputs" \
                    else sorted(d[c].unique())
                for t in ticks:
                    ax.text(j - 0.04, norm(c, np.array([float(t)]))[0], str(int(t)), ha="right", va="center", fontsize=9)
        dpos = norm("n_inputs", np.array([659.0]))[0]
        ax.plot([2 - 0.15, 2 + 0.15], [dpos, dpos], color="#B91C1C", lw=2)
        ax.text(2.18, dpos, N_TRAIN_NOTE, color="#B91C1C", fontsize=10, va="center")
        ax.set_xticks(range(len(cols)), labels)
        ax.set_yticks([])
        ax.set_xlim(-0.4, len(cols) - 0.3)
        for s in ("left", "right", "top"):
            ax.spines[s].set_visible(False)
        fig.colorbar(cm.ScalarMappable(norm=nrm, cmap=cmap), ax=ax, label="val demeaned r", shrink=0.8)
        fig.suptitle(f"{cfg['versions'][v]['label']}: every configuration (seed means); thick = val-best per k_task",
                     fontweight="bold")
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)


def fig_dimensionality(df, cfg, path):
    plt, ctx = plt_ctx()
    with ctx:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), layout="constrained", sharey=True)
        cmap = plt.get_cmap("viridis", len(cfg["grid"]["k_task"]))
        for ax, v in zip(axes, cfg["versions"]):
            sel = select(df[df["version"] == v], cfg, by=("seed", "k_rest", "k_task"))
            s = sel.groupby(["k_rest", "k_task"]).agg(D=("n_inputs", "first"), test=("test_demeaned_pearson", "mean")).reset_index()
            for i, kt in enumerate(cfg["grid"]["k_task"]):
                g = s[s["k_task"] == kt].sort_values("D")
                ax.plot(g["D"], g["test"], "-o", color=cmap(i), ms=8, mec="black", label=f"k_task = {kt}")
            ax.axvline(659, color="#B91C1C", ls="--", lw=1.5)
            ax.text(670, ax.get_ylim()[0], N_TRAIN_NOTE, color="#B91C1C", fontsize=10, va="bottom")
            ax.set_xscale("log")
            ax.set_xlabel("total input dimension D = k_rest + 7 · k_task")
            ax.set_title(cfg["versions"][v]["label"], fontweight="bold")
        axes[0].set_ylabel("test demeaned r (val-selected over k_sc, n_pls)")
        axes[0].legend(frameon=False, fontsize=11)
        fig.suptitle("Input Dimensionality vs SC Prediction (lines: fixed k_task, points: k_rest 64 → 512)", fontweight="bold")
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)


def fig_paired(sel, cfg, path):
    plt, ctx = plt_ctx()
    kts = [k for k in cfg["grid"]["k_task"] if k > 0]
    with ctx:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), layout="constrained")
        for ax, m in zip(axes.flat, cfg["metrics"]):
            for i, v in enumerate(cfg["versions"]):
                s = sel[sel["version"] == v]
                base = s[s["k_task"] == 0].set_index("seed")[f"test_{m}"]
                for j, kt in enumerate(kts):
                    d = s[s["k_task"] == kt].set_index("seed")[f"test_{m}"] - base
                    x = j + (i - 0.5) * 0.25
                    ax.scatter(np.full(len(d), x), d, color=COLORS[v], s=30, edgecolor="black", lw=0.6, zorder=3)
                    ax.errorbar(x, d.mean(), yerr=d.std(ddof=1) / math.sqrt(len(d)), fmt="_", ms=18, mew=2.5,
                                color="black", capsize=5, zorder=4)
            ax.axhline(0, color="#555555", lw=1)
            ax.set_xticks(range(len(kts)), [str(k) for k in kts])
            ax.set_xlabel("k_task")
            ax.set_title(f"Δ {mb.METRIC_LABEL[m]} vs rest only", fontweight="bold")
        handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[v], mec="black", label=cfg["versions"][v]["label"])
                   for v in cfg["versions"]]
        axes.flat[0].legend(handles=handles, frameon=False)
        fig.suptitle("Per-Seed Paired Difference vs Rest Only (test; each arm val-selected within its slice)", fontweight="bold")
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)


def fig_blocks(sel, cfg, path):
    """Block VIP share of each version's best with-task slice (highest mean val), per seed."""
    plt, ctx = plt_ctx()
    blocks = ["rest"] + list(cfg["task_conditions"])
    with ctx:
        fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")
        for i, v in enumerate(cfg["versions"]):
            s = sel[(sel["version"] == v) & (sel["k_task"] > 0)]
            best_kt = s.groupby("k_task")["val_demeaned_pearson"].mean().idxmax()
            s = s[s["k_task"] == best_kt]
            for j, b in enumerate(blocks):
                vals = s[f"vip_{b}"]
                x = j + (i - 0.5) * 0.35
                ax.bar(x, vals.mean(), width=0.33, color=COLORS[v], edgecolor="black",
                       label=f"{cfg['versions'][v]['label']} (k_task = {best_kt})" if j == 0 else None)
                ax.scatter(np.full(len(vals), x), vals, color="white", edgecolor="black", s=18, zorder=3)
        ax.axhline(1 / len(blocks), color="#555555", ls=":", lw=1)
        ax.set_xticks(range(len(blocks)), [b.capitalize() if b != "wm" else "WM" for b in blocks])
        ax.set_ylabel("share of explained SC variance (block VIP)")
        ax.legend(frameon=False)
        ax.set_title("Which FC Blocks Carry the PLS Fit (best with-task slice; dots = seeds; dotted = equal share)",
                     fontweight="bold")
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)


def fig_ksc(sel, cfg, path):
    plt, ctx = plt_ctx()
    kts = cfg["grid"]["k_task"]
    with ctx:
        fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
        rng = np.random.default_rng(0)
        for i, v in enumerate(cfg["versions"]):
            s = sel[sel["version"] == v]
            for j, kt in enumerate(kts):
                k = s[s["k_task"] == kt]["k_sc"]
                x = j + (i - 0.5) * 0.25 + rng.uniform(-0.05, 0.05, len(k))
                ax.scatter(x, k, color=COLORS[v], edgecolor="black", s=40, zorder=3,
                           label=cfg["versions"][v]["label"] if j == 0 else None)
        ax.set_yscale("log", base=2)
        ax.set_yticks(cfg["grid"]["k_sc"], [str(k) for k in cfg["grid"]["k_sc"]])
        ax.set_xticks(range(len(kts)), [str(k) for k in kts])
        ax.set_xlabel("k_task")
        ax.set_ylabel("selected SC PCs (k_sc)")
        ax.legend(frameon=False)
        ax.set_title("Does the Target Dimension Move Up with Task Blocks? (dots = seeds)", fontweight="bold")
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load((HERE / "config.yml").read_text())
    df = load(cfg, args.allow_partial)
    root = HERE / cfg["output_dir"] / cfg["parcellation"]
    sel = select(df, cfg)
    summ = summarize(sel, cfg)
    for v in cfg["versions"]:
        out = root / v
        (out / "tables").mkdir(parents=True, exist_ok=True)
        (out / "figures").mkdir(parents=True, exist_ok=True)
        sel[sel["version"] == v].to_csv(out / "tables" / "selected.csv", index=False, float_format="%.6g")
        s = summ[summ["version"] == v]
        s.to_csv(out / "tables" / "summary.csv", index=False, float_format="%.6g")
        (out / "tables" / "summary.md").write_text(md(s, ["k_task", "n_seeds"] + [f"{m}_mean" for m in cfg["metrics"]]
                                                       + ["d_demeaned_pearson_mean", "d_demeaned_pearson_se",
                                                          "k_rest_selected", "k_sc_selected", "n_pls_selected"]))
        fig_heatmap(df, cfg, v, out / "figures" / "heatmap_rest_task.png")
        fig_weave(df, cfg, v, out / "figures" / "weave.png")
    out = root / "compare"
    for sub in ("tables", "figures"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    summ.to_csv(out / "tables" / "summary.csv", index=False, float_format="%.6g")
    (out / "tables" / "summary.md").write_text(md(summ, ["version", "k_task"] + [f"{m}_mean" for m in cfg["metrics"]]
                                                  + [f"d_{m}_mean" for m in cfg["metrics"]]))
    fig_ablation(summ, cfg, out / "figures" / "ablation_curve.png")
    fig_dimensionality(df, cfg, out / "figures" / "dimensionality.png")
    fig_paired(sel, cfg, out / "figures" / "paired_delta.png")
    fig_blocks(sel, cfg, out / "figures" / "block_contribution.png")
    fig_ksc(sel, cfg, out / "figures" / "selected_k_sc.png")
    print("wrote", root)


if __name__ == "__main__":
    main()
