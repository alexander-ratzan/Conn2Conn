#!/usr/bin/env python3
"""Generate MANY chart variants per finding (F1-F13) for a pick-and-choose slideshow.

For each finding we build a tidy (category x parcellation) table (+ per-seed arrays where
available) and render it as several different chart types: grouped bars, horizontal bars,
lollipop, dumbbell, slope, heatmap, and box/strip. Line-shaped findings (F8, F10) get
bespoke variants. Everything lands in figures/slides/F{n}_{variant}.{png,pdf}.

Then build_slides.py (called at the end) auto-assembles a Beamer deck from whatever exists.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from style import (load_recon, load_downstream, load_family, recon_est, down_est,
                   C, PARCS, PARC_LABEL, ESTIMATORS, EST_LABEL, FAM, NLIN, TRACT, SANITY)

r = load_recon(); d = load_downstream(); fam = load_family()
SL = Path(__file__).resolve().parent / "figures" / "slides"
SL.mkdir(parents=True, exist_ok=True)
PCOL = {"Glasser": C["FC"], "4S456Parcels": C["SC"]}
PSHORT = {"Glasser": "Glasser", "4S456Parcels": "4S456"}
COGS = ["CogTotal", "CogFluid", "CogCryst"]


def save(fig, fid, variant):
    for ext in ("pdf", "png"):
        fig.savefig(SL / f"{fid}_{variant}.{ext}", bbox_inches="tight")
    plt.close(fig)


def build(cats):
    """cats: list of (label, fn) with fn(parc)->seed array. Returns (tbl, seeds)."""
    tbl = pd.DataFrame(index=[c[0] for c in cats], columns=PARCS, dtype=float)
    seeds = {}
    for label, fn in cats:
        for parc in PARCS:
            a = np.asarray(fn(parc), float); a = a[np.isfinite(a)]
            tbl.loc[label, parc] = a.mean() if len(a) else np.nan
            seeds[(label, parc)] = a
    return tbl, seeds


# ----------------------------------------------------------- generic renderers
def r_grouped(tbl, meta):
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = list(tbl.index); x = np.arange(len(cats)); w = 0.8 / len(PARCS)
    for pi, parc in enumerate(PARCS):
        ax.bar(x + (pi - (len(PARCS)-1)/2)*w, tbl[parc].values, w, color=PCOL[parc],
               edgecolor="white", label=PARC_LABEL[parc])
    if meta.get("baseline") is not None:
        ax.axhline(meta["baseline"], color=C["base"], ls="--", lw=1.2)
    if meta.get("zero"):
        ax.axhline(0, color=C["ink"], lw=1)
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=meta.get("rot", 0),
                                         ha="center" if not meta.get("rot") else "right", fontsize=9)
    ax.set_ylabel(meta["ylabel"]); ax.set_title(meta["title"] + " — grouped bars", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    save(fig, meta["fid"], "v_groupedbar")


def r_hbar(tbl, meta):
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = list(tbl.index); y = np.arange(len(cats)); h = 0.8 / len(PARCS)
    for pi, parc in enumerate(PARCS):
        ax.barh(y + (pi - (len(PARCS)-1)/2)*h, tbl[parc].values, h, color=PCOL[parc],
                edgecolor="white", label=PARC_LABEL[parc])
    if meta.get("zero"):
        ax.axvline(0, color=C["ink"], lw=1)
    if meta.get("baseline") is not None:
        ax.axvline(meta["baseline"], color=C["base"], ls="--", lw=1.2)
    ax.set_yticks(y); ax.set_yticklabels(cats, fontsize=9); ax.invert_yaxis()
    ax.set_xlabel(meta["ylabel"]); ax.set_title(meta["title"] + " — horizontal bars", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    save(fig, meta["fid"], "v_hbar")


def r_lollipop(tbl, meta):
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = list(tbl.index); y = np.arange(len(cats))
    for pi, parc in enumerate(PARCS):
        off = (pi - (len(PARCS)-1)/2) * 0.18
        for yi, v in zip(y, tbl[parc].values):
            ax.plot([0, v], [yi+off, yi+off], color=PCOL[parc], lw=1.5, zorder=1)
        ax.scatter(tbl[parc].values, y+off, s=80, color=PCOL[parc], zorder=3,
                   edgecolor="white", label=PARC_LABEL[parc])
    ax.axvline(0, color=C["ink"], lw=1)
    if meta.get("baseline") is not None:
        ax.axvline(meta["baseline"], color=C["base"], ls="--", lw=1.2)
    ax.set_yticks(y); ax.set_yticklabels(cats, fontsize=9); ax.invert_yaxis()
    ax.set_xlabel(meta["ylabel"]); ax.set_title(meta["title"] + " — lollipop", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    save(fig, meta["fid"], "v_lollipop")


def r_dumbbell(tbl, meta):
    if len(PARCS) != 2:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = list(tbl.index); y = np.arange(len(cats))
    a, b = tbl[PARCS[0]].values, tbl[PARCS[1]].values
    for yi, va, vb in zip(y, a, b):
        ax.plot([va, vb], [yi, yi], color="#bbb", lw=2, zorder=1)
    ax.scatter(a, y, s=90, color=PCOL[PARCS[0]], zorder=3, edgecolor="white", label=PARC_LABEL[PARCS[0]])
    ax.scatter(b, y, s=90, color=PCOL[PARCS[1]], zorder=3, edgecolor="white", label=PARC_LABEL[PARCS[1]])
    if meta.get("baseline") is not None:
        ax.axvline(meta["baseline"], color=C["base"], ls="--", lw=1.2)
    if meta.get("zero"):
        ax.axvline(0, color=C["ink"], lw=1)
    ax.set_yticks(y); ax.set_yticklabels(cats, fontsize=9); ax.invert_yaxis()
    ax.set_xlabel(meta["ylabel"]); ax.set_title(meta["title"] + " — dumbbell (parcellation gap)",
                                                fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    save(fig, meta["fid"], "v_dumbbell")


def r_slope(tbl, meta):
    if len(PARCS) != 2:
        return
    fig, ax = plt.subplots(figsize=(7, 5.5))
    cmap = plt.cm.tab10(np.linspace(0, 1, len(tbl)))
    for i, cat in enumerate(tbl.index):
        vals = [tbl.loc[cat, p] for p in PARCS]
        ax.plot([0, 1], vals, "-o", color=cmap[i], ms=7, lw=2)
        ax.annotate(cat, (1, vals[1]), textcoords="offset points", xytext=(6, 0),
                    fontsize=8, va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels([PSHORT[p] for p in PARCS])
    ax.set_xlim(-0.2, 1.6)
    if meta.get("zero"):
        ax.axhline(0, color=C["ink"], lw=1)
    ax.set_ylabel(meta["ylabel"]); ax.set_title(meta["title"] + " — slope across parcellations",
                                                fontsize=11, fontweight="bold")
    save(fig, meta["fid"], "v_slope")


def r_heatmap(tbl, meta):
    fig, ax = plt.subplots(figsize=(6.5, 0.7*len(tbl)+1.5))
    M = tbl[PARCS].values.astype(float)
    vmax = np.nanmax(np.abs(M)); diverge = meta.get("zero") or (np.nanmin(M) < 0)
    im = ax.imshow(M, cmap="RdBu_r" if diverge else "viridis",
                   vmin=-vmax if diverge else None, vmax=vmax if diverge else None, aspect="auto")
    ax.set_xticks(range(len(PARCS))); ax.set_xticklabels([PSHORT[p] for p in PARCS])
    ax.set_yticks(range(len(tbl))); ax.set_yticklabels(tbl.index, fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(M[i, j]) > 0.6*vmax else "black")
    ax.set_title(meta["title"] + " — heatmap", fontsize=11, fontweight="bold"); ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.06, pad=0.04, label=meta["ylabel"])
    save(fig, meta["fid"], "v_heatmap")


def r_box(tbl, seeds, meta):
    if not any(len(seeds[(c, PARCS[0])]) > 2 for c in tbl.index):
        return
    fig, ax = plt.subplots(figsize=(8.5, 5))
    cats = list(tbl.index); x = np.arange(len(cats)); w = 0.8/len(PARCS)
    for pi, parc in enumerate(PARCS):
        data = [seeds[(c, parc)] for c in cats]
        pos = x + (pi - (len(PARCS)-1)/2)*w
        bp = ax.boxplot(data, positions=pos, widths=w*0.85, patch_artist=True, showfliers=False,
                        medianprops=dict(color=C["ink"]))
        for patch in bp["boxes"]:
            patch.set_facecolor(PCOL[parc]); patch.set_alpha(0.6)
        for ci, c in enumerate(cats):
            v = seeds[(c, parc)]
            ax.scatter(np.full(len(v), pos[ci]) + np.random.default_rng(ci).uniform(-w*0.2, w*0.2, len(v)),
                       v, s=12, color=C["ink"], alpha=0.4, zorder=3)
    if meta.get("zero"):
        ax.axhline(0, color=C["ink"], lw=1)
    if meta.get("baseline") is not None:
        ax.axhline(meta["baseline"], color=C["base"], ls="--", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=meta.get("rot", 0),
                                         ha="center" if not meta.get("rot") else "right", fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=PCOL[p], alpha=0.6, label=PARC_LABEL[p]) for p in PARCS], fontsize=8)
    ax.set_ylabel(meta["ylabel"]); ax.set_title(meta["title"] + " — box + seed dots", fontsize=11, fontweight="bold")
    save(fig, meta["fid"], "v_box")


def render_all(tbl, seeds, meta):
    r_grouped(tbl, meta); r_hbar(tbl, meta); r_lollipop(tbl, meta)
    r_dumbbell(tbl, meta); r_slope(tbl, meta); r_heatmap(tbl, meta); r_box(tbl, seeds, meta)


# ----------------------------------------------------------- per-finding payloads
def F1():
    tbl, seeds = build([("FC→SC", lambda p: recon_est(r, p, "pca_pls", "FC", "SC")),
                        ("SC→FC", lambda p: recon_est(r, p, "pca_pls", "SC", "FC"))])
    render_all(tbl, seeds, dict(fid="F1", title="F1 directional asymmetry",
                                ylabel="demeaned pearson"))
    # ratio across estimators (bespoke)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ESTIMATORS))
    for parc in PARCS:
        ratios = [recon_est(r, parc, e, "FC", "SC").mean()/recon_est(r, parc, e, "SC", "FC").mean()
                  for e in ESTIMATORS]
        ax.plot(x, ratios, "-o", color=PCOL[parc], ms=8, label=PARC_LABEL[parc])
        for xi, v in zip(x, ratios):
            ax.annotate(f"{v:.2f}×", (xi, v), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, fontweight="bold")
    ax.axhline(1, color=C["bad"], ls="--", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels([EST_LABEL[e] for e in ESTIMATORS])
    ax.set_ylabel("FC→SC / SC→FC ratio"); ax.set_ylim(0.9, None)
    ax.set_title("F1 directional asymmetry — ratio across estimators", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8); save(fig, "F1", "v_ratioline")


def F2():
    cats = []
    for tgt in ["SC", "FC"]:
        for src in ["bv", "demo", "bv+demo"]:
            cats.append((f"{src}→{tgt}", (lambda s, t: (lambda p: recon_est(r, p, "pca_pls", s, t)))(src, tgt)))
    tbl, seeds = build(cats)
    render_all(tbl, seeds, dict(fid="F2", title="F2 double dissociation",
                                ylabel="demeaned pearson", rot=40))


def F3():
    tbl, seeds = build([("pred SC (FC→SC)", lambda p: recon_est(r, p, "pca_pls", "FC", "SC")),
                        ("pred FC (SC→FC)", lambda p: recon_est(r, p, "pca_pls", "SC", "FC"))])
    render_all(tbl, seeds, dict(fid="F3recon", title="F3 imputed connectomes are measurable",
                                ylabel="reconstruction demeaned r"))
    tbl2, s2 = build([("pred SC", lambda p: down_est(d, p, "bayesian_ridge", "pred_SC", "CogCryst", "lift_over_bvdemo")),
                      ("pred FC", lambda p: down_est(d, p, "bayesian_ridge", "pred_FC", "CogCryst", "lift_over_bvdemo"))])
    render_all(tbl2, s2, dict(fid="F3util", title="F3 imputed utility is asymmetric (CogCryst)",
                              ylabel="lift over bv+demo", zero=True))


def F4():
    cats = [(t[3:], (lambda tt: (lambda p: down_est(d, p, "bayesian_ridge", "obs_FC", tt, "lift_over_bvdemo")))(t))
            for t in COGS]
    tbl, seeds = build(cats)
    render_all(tbl, seeds, dict(fid="F4", title="F4 obs FC crystallized-cognition (narrow, FDR-surviving)",
                                ylabel="lift over bv+demo", zero=True))


def F5():
    cats = [("obs SC", lambda p: down_est(d, p, "bayesian_ridge", "obs_SC", "CogCryst", "lift_over_bvdemo")),
            ("pred SC", lambda p: down_est(d, p, "bayesian_ridge", "pred_SC", "CogCryst", "lift_over_bvdemo")),
            ("pred FC", lambda p: down_est(d, p, "bayesian_ridge", "pred_FC", "CogCryst", "lift_over_bvdemo")),
            ("obs SC+bv+demo", lambda p: down_est(d, p, "bayesian_ridge", "obs_SC+bv+demo", "CogCryst", "lift_over_bvdemo"))]
    tbl, seeds = build(cats)
    render_all(tbl, seeds, dict(fid="F5", title="F5 SC/imputation utility wall (CogCryst)",
                                ylabel="lift over bv+demo", zero=True, rot=30))


def F6():
    variants = ["bvdemo_to_SC", "pred_SC_raw", "pred_SC_resid_bvdemo", "obs_SC", "obs_FC"]
    lab = {"bvdemo_to_SC": "bv+demo", "pred_SC_raw": "pred SC raw",
           "pred_SC_resid_bvdemo": "pred SC resid", "obs_SC": "obs SC", "obs_FC": "obs FC"}

    def getter(v):
        def fn(p):
            row = fam[(fam.parcellation == p) & (fam.relation == "sibling") & (fam.variant == v)]
            return np.array([row.auc.iloc[0]]) if len(row) else np.array([np.nan])
        return fn
    tbl, seeds = build([(lab[v], getter(v)) for v in variants])
    render_all(tbl, seeds, dict(fid="F6", title="F6 family (sibling) AUC", ylabel="sibling AUC",
                                baseline=0.5, rot=30))


def F13():
    cats = [("pred SC", lambda p: down_est(d, p, "bayesian_ridge", "pred_SC", "CogCryst", "lift_over_bvdemo")),
            ("pred FC", lambda p: down_est(d, p, "bayesian_ridge", "pred_FC", "CogCryst", "lift_over_bvdemo"))]
    tbl, seeds = build(cats)
    render_all(tbl, seeds, dict(fid="F13", title="F13 imputation harm (matched dim, CogCryst)",
                                ylabel="lift over bv+demo", zero=True))


# ----------------------------------------------------------- bespoke line findings
def F7():
    rels = ["MZ", "DZ", "sibling"]
    for parc in PARCS:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for v, col, lab in [("pred_SC_resid_bvdemo", C["pred"], "residual (identification)"),
                            ("combined_pred_SC", C["accent"], "combined (reconstruction)")]:
            sub = fam[(fam.parcellation == parc) & (fam.variant == v)].set_index("relation")
            ax.plot(range(3), [sub.loc[rl, "auc"] for rl in rels], "-o", color=col, ms=8, lw=2, label=lab)
        ax.axhline(0.5, color=C["bad"], ls="--", lw=1.2)
        ax.set_xticks(range(3)); ax.set_xticklabels(["MZ", "DZ", "sib"]); ax.set_ylim(0.45, 1.0)
        ax.set_ylabel("separation AUC"); ax.legend(fontsize=8)
        ax.set_title(f"F7 objective tradeoff — {PARC_LABEL[parc]}", fontsize=11, fontweight="bold")
        save(fig, "F7", f"v_{PSHORT[parc]}")


def F8():
    pp = pd.read_csv(FAM / "f8_per_pc.csv")
    g = pp.groupby(["parcellation", "pc"]).median(numeric_only=True).reset_index()
    sel = {"Glasser": 3, "4S456Parcels": 4}
    for metric, ylab, vk in [("FC_to_PC_R2", "FC→PC R²", "fcr2"),
                             ("AUC_sibling", "sibling AUC", "auc"),
                             ("confound_R2_test", "confound R² (sex/vol)", "conf")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for parc in PARCS:
            s = g[g.parcellation == parc]
            ax.plot(s.pc, s[metric], "-o", color=PCOL[parc], ms=6, label=PARC_LABEL[parc])
            ax.axvline(sel[parc], color=C["accent"], lw=8, alpha=0.15, zorder=0)
        ax.set_xlabel("principal component"); ax.set_xticks(range(1, 11)); ax.set_ylabel(ylab)
        ax.set_title(f"F8 mechanism — {ylab} per PC", fontsize=11, fontweight="bold"); ax.legend(fontsize=8)
        save(fig, "F8", f"v_{vk}")


def F9():
    ts = pd.read_csv(TRACT / "tractography_synthesis.csv").set_index("row")["value"]
    feats = [("SC", "E1: SC -> FC median dp"), ("SC+r2t", "E1: SC_r2t -> FC median dp"),
             ("kitchen", "E1: kitchen_sink -> FC median dp"), ("r2t", "E1: r2t -> FC median dp"),
             ("r2t corr", "E1: r2t_corr -> FC median dp")]
    vals = [float(ts[k]) for _, k in feats]; labs = [f for f, _ in feats]
    # bar
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(labs)), vals, color=[C["SC"], "#5DADE2", "#85C1E9", C["bv"], "#B2BABB"], edgecolor="white")
    ax.axhline(vals[0], color=C["SC"], ls="--", lw=1)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=30, ha="right")
    ax.set_ylabel("→FC demeaned r"); ax.set_title("F9 tractography → FC (Glasser) — bars", fontsize=11, fontweight="bold")
    save(fig, "F9", "v_recon_bar")
    # hbar
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(labs)), vals, color=[C["SC"], "#5DADE2", "#85C1E9", C["bv"], "#B2BABB"], edgecolor="white")
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs); ax.invert_yaxis()
    ax.set_xlabel("→FC demeaned r"); ax.set_title("F9 tractography → FC (Glasser) — hbar", fontsize=11, fontweight="bold")
    save(fig, "F9", "v_recon_hbar")
    # downstream
    e5 = pd.read_csv(TRACT / "e5_downstream_summary.csv")
    order = ["FC", "bv+demo", "SC", "r2t", "SC_r2t", "r2t_corr", "r2t->synthFC"]
    sub = e5[e5.target == "CogCrystalComp_Unadj"].set_index("rep")
    vv = [sub.loc[o, "pearson_raw"] if o in sub.index else np.nan for o in order]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(range(len(order)), vv, color=[C["FC"] if o == "FC" else C["base"] if o == "bv+demo" else C["bv"] for o in order],
           edgecolor="white")
    ax.axhline(sub.loc["bv+demo", "pearson_raw"], color=C["base"], ls="--", lw=1.2)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("CogCryst pearson"); ax.set_title("F9 tractography downstream (Glasser)", fontsize=11, fontweight="bold")
    save(fig, "F9", "v_downstream")


def F10():
    sc = pd.read_csv(NLIN / "n6_scaling_summary.csv")
    # gap vs n
    fig, ax = plt.subplots(figsize=(8, 5))
    for task, col in [("reconstruction", C["SC"]), ("cognition", C["FC"])]:
        s = sc[(sc.task == task) & (sc.n_seeds >= 5)].sort_values("n_sub")
        ax.plot(s.n_sub, s.median_gap, "-o", color=col, ms=6, label=task)
        ax.fill_between(s.n_sub, s.gap_min, s.gap_max, color=col, alpha=0.12)
    ax.axhline(0, color=C["ink"], lw=1); ax.set_xlabel("training subjects"); ax.set_ylabel("nonlinear − linear gap")
    ax.legend(fontsize=8); ax.set_title("F10 nonlinear gap vs n", fontsize=11, fontweight="bold")
    save(fig, "F10", "v_gap")
    # linear vs nonlinear
    fig, ax = plt.subplots(figsize=(8, 5))
    for task, col in [("reconstruction", C["SC"]), ("cognition", C["FC"])]:
        s = sc[(sc.task == task) & (sc.n_seeds >= 5)].sort_values("n_sub")
        ax.plot(s.n_sub, s.median_linear, "-o", color=col, ms=5, label=f"{task} linear")
        ax.plot(s.n_sub, s.median_final, "--s", color=col, ms=5, alpha=0.6, label=f"{task} nonlinear")
    ax.set_xlabel("training subjects"); ax.set_ylabel("score"); ax.legend(fontsize=7)
    ax.set_title("F10 linear vs nonlinear vs n", fontsize=11, fontweight="bold")
    save(fig, "F10", "v_linvsnon")
    # reliability ceiling
    NOISE = SANITY / "noise_sanity_check" / "outputs"
    a = pd.read_csv(NOISE / "a_reliability_ceiling.csv")
    e = pd.read_csv(NOISE / "e_crossmodal_disattenuation.csv")
    ach = e[(e.source == "SC->FC") & (e.metric == "demeaned_pearson")]["achieved"].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 5)); x = np.arange(len(PARCS)); w = 0.38
    ceil = [a[(a.parc == p) & (a.comparison == "between_session")]["demeaned_pearson"].iloc[0] for p in PARCS]
    ax.bar(x - w/2, ceil, w, color=C["fc_lt"], edgecolor="white", label="FC reliability ceiling")
    ax.bar(x + w/2, [ach]*len(PARCS), w, color=C["SC"], edgecolor="white", label="SC→FC achieved")
    for i, c in enumerate(ceil):
        ax.text(i, c + 0.01, f"{ach/c*100:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([PARC_LABEL[p] for p in PARCS]); ax.set_ylabel("demeaned pearson")
    ax.legend(fontsize=8); ax.set_title("F10 SC→FC vs FC reliability ceiling", fontsize=11, fontweight="bold")
    save(fig, "F10", "v_ceiling")


def F11_F12():
    # reuse the already-built composite figures as slide variants
    import shutil
    base = Path(__file__).resolve().parent / "figures"
    for src, dst in [("F11_statistical_fragility", "F11_v_composite"),
                     ("F12_bayesian_corroboration", "F12_v_composite")]:
        for ext in ("pdf", "png"):
            s = base / f"{src}.{ext}"
            if s.exists():
                shutil.copy(s, SL / f"{dst}.{ext}")


if __name__ == "__main__":
    for fn in [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F13, F11_F12]:
        fn()
        print("  built", fn.__name__)
    n = len(list(SL.glob("*.pdf")))
    print(f"done — {n} variant figures in figures/slides/")
