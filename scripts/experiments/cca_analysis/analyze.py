"""Spec v4 E5.3: tables and figures for H1-H3 (also imported by the notebook, which draws the same figures inline).

Seed-level statistics: every contrast is computed per seed (mean over the cells in scope) and reported as mean ± SE over
the 5 seeds with the number of seeds in favour; the support rule (spec E5) is >= 4 / 5 seeds in favour on average rank
with mean demeaned r not worse by more than 0.005.

    python scripts/experiments/cca_analysis/analyze.py      # -> tables/*.{csv,md}, figures/*.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e5_common import CFG, FC, OUT, VIEWS  # noqa: E402

REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments" / "model_benchmark"))
METRICS = {"demeaned_pearson": "demeaned r", "avg_rank": "average rank", "top1_acc": "top-1"}
REP_LABEL = {"R0": "nodal PCA of the mean connectome", "R1": "condition-specific PCA",
             "R2": "shared block PCA, unscaled", "R3": "shared block PCA, variance-adjusted",
             "CCA": "CCA (full rank)", "CCA-256": "CCA on 256 variance-adjusted block-PCA codes", "edges": "copy"}
MODEL_LABEL = {"ridge": "ridge", "cond_gauss": "cond. Gaussian", "cca_ridge": "CCA + ridge decoder", "copy": "copy",
               "cca_agg": "CCA-agg", "cca_cat": "CCA-cat", "cca_best_single": "best single source (CCA)",
               "ridge_cat": "ridge, concatenated codes"}
COMBO_LABEL = {"ridge": "ridge (closed form)", "mse_only": "MSE", "vm_0.5": "+VM", "cedm_0.5": "+CEdm", "nd_0.5": "+ND",
               "vm_cedm_0.5": "+VM +CEdm", "vm_nd_0.5": "+VM +ND", "cedm_nd_0.5": "+CEdm +ND", "alldm_0.5": "+VM +CEdm +ND"}
MARGIN = 0.005
TAB, FIG = HERE / "tables", HERE / "figures"


# ---------------------------------------------------------------- loading
def load_e51():
    rows, gates, sel = [], {}, {}
    for p in sorted((OUT / "results").glob("seed[0-9].json")):
        r = json.loads(p.read_text())
        rows += r["rows"]
        gates[r["seed"]], sel[r["seed"]] = r["g0"], r["selections"]
    return pd.DataFrame(rows), gates, sel


def load_h3():
    rows, gates, curves, scales = [], {}, {}, {}
    for p in sorted((OUT / "h3").glob("composite_seed[0-9]*.json")):          # whole seeds or parts (_part<p>of<P>)
        r = json.loads(p.read_text())
        s = r["seed"]
        rows += r["rows"]
        g = gates.setdefault(s, {"pass": True, "parts": 0})
        g["parts"] += 1
        g["pass"] = g["pass"] and r["g1"]["pass"]
        for k, v in r["g1"].items():
            if k != "pass":
                g[k] = max(g.get(k, 0.0), v)                                     # worst part
        curves.setdefault(s, {}).update(r["curves"])
        scales.setdefault(s, {}).update(r["scales"])
    return pd.DataFrame(rows), gates, curves, scales


def seed_mean(df, by, metrics=tuple(METRICS)):
    """Mean over cells within seed, then mean ± SE over seeds."""
    s = df.groupby(by + ["seed"])[list(metrics)].mean().reset_index()
    g = s.groupby(by)[list(metrics)]
    out = g.mean().add_suffix("_mean").join(g.sem().add_suffix("_se")).join(g.size().rename("n_seeds"))
    return out.reset_index()


def fmt(m, se):
    return f"{m:.4f} ± {se:.4f}"


def contrast(a, b, label):
    """a, b: per-seed Series (index seed) of a metric; returns the paired difference summary."""
    d = (a - b).dropna()
    return dict(contrast=label, mean=d.mean(), se=d.sem(), n_pos=int((d > 0).sum()), n=len(d))


# ---------------------------------------------------------------- H1
def h1_frame(df):
    h1 = df[df.hyp == "H1"].copy()
    h1["label"] = h1.apply(lambda r: f"{REP_LABEL[r.rep]} · {MODEL_LABEL[r.model]}" if r.model not in ("copy", "cca_ridge")
                           else REP_LABEL[r.rep] if r.model == "cca_ridge" else "copy (FC → FC only)", axis=1)
    return h1


def best_pca(sel):
    """Per seed: the PCA representation with the best mean val average rank (ridge, 72 pairs)."""
    return {s: max(v["H1_pca_val"], key=v["H1_pca_val"].get) for s, v in sel.items()}


def h1_tables(df, sel):
    h1 = h1_frame(df)
    summ = []
    for blk in ("all", "FC -> FC", "SC -> FC", "FC -> SC"):
        sub = h1 if blk == "all" else h1[h1.block == blk]
        t = seed_mean(sub[sub.model != "copy"] if blk != "FC -> FC" else sub, ["label"])
        t.insert(0, "block", blk)
        summ.append(t)
    summ = pd.concat(summ, ignore_index=True)
    bp = best_pca(sel)
    h1["best_pca"] = h1.seed.map(bp)
    cca = h1[(h1.model == "cca_ridge") & (h1.rep == "CCA")]
    pca = h1[(h1.model == "ridge") & (h1.rep == h1.best_pca)]
    con = []
    for blk in ("all", "FC -> FC", "SC -> FC", "FC -> SC"):
        f = (lambda x: x) if blk == "all" else (lambda x: x[x.block == blk])
        for m in METRICS:
            row = contrast(f(cca).groupby("seed")[m].mean(), f(pca).groupby("seed")[m].mean(), "CCA − best PCA (ridge)")
            con.append(dict(block=blk, metric=m, **row))
        for rep in ("R0", "R1", "R2", "R3"):
            other = h1[(h1.model == "ridge") & (h1.rep == rep)]
            row = contrast(f(cca).groupby("seed")["avg_rank"].mean(), f(other).groupby("seed")["avg_rank"].mean(),
                           f"CCA − {REP_LABEL[rep]} (ridge)")
            con.append(dict(block=blk, metric="avg_rank", **row))
        row = contrast(f(cca).groupby("seed")["avg_rank"].mean(),
                       f(h1[(h1.model == "cca_ridge") & (h1.rep == "CCA-256")]).groupby("seed")["avg_rank"].mean(),
                       "CCA − CCA on 256 block-PCA codes")
        con.append(dict(block=blk, metric="avg_rank", **row))
    con = pd.DataFrame(con)
    pairs = cca.merge(pca, on=["seed", "source", "target", "block"], suffixes=("_cca", "_pca"))
    for m in METRICS:
        pairs[f"d_{m}"] = pairs[f"{m}_cca"] - pairs[f"{m}_pca"]
    per_pair = pairs.groupby(["source", "target", "block"])[[f"d_{m}" for m in METRICS]].agg(["mean", "sem"])
    return summ, con, per_pair, bp


def verdict(con, block="all"):
    c = con[(con.block == block) & (con.contrast == "CCA − best PCA (ridge)")].set_index("metric")
    ok = c.loc["avg_rank", "n_pos"] >= 4 and c.loc["demeaned_pearson", "mean"] >= -MARGIN
    return "supported" if ok else "not supported", c


def pair_matrix(df, metric, sel_rows):
    """9 x 9 matrix (rows = source, columns = target) of the seed-mean metric."""
    m = sel_rows.groupby(["source", "target"])[metric].mean().unstack().reindex(index=VIEWS, columns=VIEWS)
    return m


# ---------------------------------------------------------------- H2
def h2_tables(df):
    h2 = df[df.hyp == "H2"].copy()
    h2["label"] = h2.apply(lambda r: f"ridge on concatenated codes: {REP_LABEL[r.rep]}" if r.model == "ridge_cat" else
                           "copy: mean of the other FC conditions' residuals" if r.model == "copy" else MODEL_LABEL[r.model],
                           axis=1)
    summ = seed_mean(h2, ["setting", "label"])
    per_target = seed_mean(h2, ["setting", "target", "label"])
    con = []
    for setting in ("other8", "other7"):
        s = h2[h2.setting == setting]
        get = lambda q: s.query(q).groupby("seed")
        best_cat = s[s.model == "ridge_cat"].groupby(["seed", "rep"]).val_avg_rank.mean().unstack().idxmax(axis=1)
        pca_cat = s[(s.model == "ridge_cat") & (s.rep == s.seed.map(best_cat))]
        for model in ("cca_agg", "cca_cat"):
            for m in ("avg_rank", "demeaned_pearson", "top1_acc"):
                a = get(f"model == '{model}'")[m].mean()
                con.append(dict(setting=setting, metric=m, **contrast(a, get("model == 'cca_best_single'")[m].mean(),
                                                                       f"{MODEL_LABEL[model]} − best single source")))
                con.append(dict(setting=setting, metric=m, **contrast(a, pca_cat.groupby("seed")[m].mean(),
                                                                       f"{MODEL_LABEL[model]} − best PCA concatenated")))
    return summ, per_target, pd.DataFrame(con)


# ---------------------------------------------------------------- H3
def h3_tables(h3):
    h3 = h3.copy()
    h3["group"] = np.where(h3.hyp == "H1", h3.source.eq("SC").map({True: "SC -> FC", False: ""}) , h3.setting)
    h3.loc[h3.hyp == "H1", "group"] = [("SC -> FC" if s == "SC" else "FC -> SC" if t == "SC" else "FC -> FC")
                                       for s, t in zip(h3[h3.hyp == "H1"].source, h3[h3.hyp == "H1"].target)]
    key = ["seed", "model", "setting", "source", "target"]
    ref = h3[h3.combo == "ridge"].set_index(key)
    out = []
    for combo, g in h3[h3.combo != "ridge"].groupby("combo"):
        g = g.set_index(key)
        for m in METRICS:
            g[f"d_{m}"] = g[m] - ref[m].reindex(g.index)
        out.append(g.reset_index())
    d = pd.concat(out, ignore_index=True)
    summ = []
    for grp in ["all"] + sorted(d.group.unique()):
        sub = d if grp == "all" else d[d.group == grp]
        s = sub.groupby(["combo", "seed"])[[f"d_{m}" for m in METRICS] + list(METRICS)].mean().reset_index()
        for combo, c in s.groupby("combo"):
            row = dict(group=grp, combo=combo)
            for m in METRICS:
                row[f"{m}_mean"], row[f"d_{m}_mean"], row[f"d_{m}_se"] = c[m].mean(), c[f"d_{m}"].mean(), c[f"d_{m}"].sem()
            row["n_pos_rank"] = int((c["d_avg_rank"] > 0).sum())
            row["supported"] = ("reference (G1)" if combo == "mse_only" else      # MSE only = the ridge objective
                                "yes" if row["n_pos_rank"] >= 4 and row["d_demeaned_pearson_mean"] >= -MARGIN else "no")
            summ.append(row)
    return d, pd.DataFrame(summ)


# ---------------------------------------------------------------- figures
def fig_rc():
    import run as mb  # noqa: E402  (model_benchmark house style)
    return {**mb.FIG_RC, "font.size": 12, "axes.linewidth": 1.5}


def heatmaps(mats, titles, suptitle, cmap="viridis", diverging=False, fmt_="{:.2f}", path=None, vertical=False):
    import matplotlib.pyplot as plt
    with plt.rc_context(fig_rc()):
        shape, size = ((len(mats), 1), (6.2, 5.2 * len(mats))) if vertical else ((1, len(mats)), (5.2 * len(mats), 5.0))
        fig, axes = plt.subplots(*shape, figsize=size, layout="constrained")
        axes = np.ravel(axes)
        divs = diverging if isinstance(diverging, (list, tuple)) else [diverging] * len(mats)
        for ax, M, t, div in zip(axes, mats, titles, divs):
            A = M.values.astype(float)
            if div:
                v = np.nanmax(np.abs(A))
                im = ax.imshow(A, cmap="RdBu_r", vmin=-v, vmax=v)
            else:
                im = ax.imshow(A, cmap=cmap)
            ax.set_xticks(range(len(M.columns)), M.columns, rotation=60, ha="right")
            ax.set_yticks(range(len(M.index)), M.index)
            ax.set_xlabel("target")
            ax.set_ylabel("source")
            ax.set_title(t, fontsize=12)
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    if np.isfinite(A[i, j]):
                        ax.text(j, i, fmt_.format(A[i, j]), ha="center", va="center", fontsize=7.5,
                                color="white" if (not div and A[i, j] < np.nanmean(A)) else "black")
            ax.spines[:].set_visible(False)
            fig.colorbar(im, ax=ax, shrink=0.75)
        fig.suptitle(suptitle, fontweight="bold")
        if path:
            fig.savefig(path, dpi=200, facecolor="white")
    return fig


def write_md(path, title, df, floatfmt="{:.4f}"):
    cols = list(df.columns)
    lines = [f"# {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    path.write_text("\n".join(lines) + "\n")


def main():
    TAB.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    df, g0, sel = load_e51()
    print("G0:", {s: g["pass"] for s, g in g0.items()})
    summ, con, per_pair, bp = h1_tables(df, sel)
    summ.to_csv(TAB / "h1_summary.csv", index=False)
    con.to_csv(TAB / "h1_contrasts.csv", index=False)
    per_pair.to_csv(TAB / "h1_per_pair.csv")
    write_md(TAB / "h1_contrasts.md", "E5 H1 contrasts (test; per-seed means over pairs; mean ± SE over seeds)", con)
    v, _ = verdict(con)
    print("H1:", v, "best PCA per seed:", bp)
    s2, t2, c2 = h2_tables(df)
    s2.to_csv(TAB / "h2_summary.csv", index=False)
    t2.to_csv(TAB / "h2_per_target.csv", index=False)
    c2.to_csv(TAB / "h2_contrasts.csv", index=False)
    write_md(TAB / "h2_contrasts.md", "E5 H2 contrasts (test; per-seed means over held-out views)", c2)
    ceil = seed_mean(df[df.hyp == "ceiling"], ["rep", "size", "source"])
    ceil.to_csv(TAB / "ceilings.csv", index=False)
    pd.DataFrame([dict(seed=s, **{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in x.items()})
                  for s, x in sel.items()]).to_csv(TAB / "selections.csv", index=False)
    h3, g1, _, _ = load_h3()
    if len(h3):
        print("G1:", {s: g["pass"] for s, g in g1.items()})
        d3, s3 = h3_tables(h3)
        s3.to_csv(TAB / "h3_summary.csv", index=False)
        write_md(TAB / "h3_summary.md", "E5 H3: composite terms vs the ridge decoder (test; Δ = combination − ridge)", s3)
    h1 = h1_frame(df)
    cca = h1[(h1.model == "cca_ridge") & (h1.rep == "CCA")]
    pca = h1[(h1.model == "ridge") & (h1.rep == h1.seed.map(bp))]
    for m, lab in METRICS.items():
        heatmaps([pair_matrix(df, m, cca), pair_matrix(df, m, pca),
                  pair_matrix(df, m, cca) - pair_matrix(df, m, pca)],
                 ["CCA (full rank) + ridge decoder",
                  f"best PCA representation, ridge\n({', '.join(sorted({REP_LABEL[r] for r in bp.values()}))})", "CCA − best PCA"],
                 f"H1: test {lab}, every source → target pair (mean of 5 seeds)", diverging=[False, False, True],
                 fmt_="{:.3f}", path=FIG / f"h1_pairs_{m}.png")
    print("wrote tables/ and figures/")


if __name__ == "__main__":
    main()
