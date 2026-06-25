#!/usr/bin/env python3
"""Build the simplified, table-driven talk deck (Glasser main + 4S456 appendix).
Generates bar charts into figures/simplified/ and writes simplified_slides.tex
with native LaTeX tables whose numbers come from the same CSVs as the charts.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from style import load_recon, load_downstream, load_family, recon_est, down_est, C

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures" / "simplified"
FIG.mkdir(parents=True, exist_ok=True)
r = load_recon(); d = load_downstream(); fam = load_family()
PARC_MAIN = "Glasser"
PARC_APP = "4S456Parcels"
PLABEL = {"Glasser": "Glasser (360 regions)", "4S456Parcels": "4S456Parcels (456 regions)"}

CM = C["SC"]      # cross-modal colour
BL = C["base"]    # baseline colour
GOOD = C["good"]; BAD = C["bad"]; FC = C["FC"]; PRED = C["pred"]
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.titleweight": "bold"})


# ---------- data accessors (means over 10 seeds) ----------
def rc(parc, src, tgt, metric="demeaned_pearson"):
    blk = src in ("FC+bv+demo", "SC+bv+demo")   # connectome+subject-info combos stored as block inputs
    return float(np.nanmean(recon_est(r, parc, "pca_pls", src, tgt, col=metric, block=blk)))


def lift(parc, inp, t="CogCryst"):
    return float(np.nanmean(down_est(d, parc, "bayesian_ridge", inp, t, "lift_over_bvdemo")))


def cogp(parc, inp, t="CogCryst"):
    return float(np.nanmean(down_est(d, parc, "bayesian_ridge", inp, t, "pearson")))


def famauc(parc, variant):
    s = fam[(fam.parcellation == parc) & (fam.relation == "sibling") & (fam.variant == variant)]
    return float(s.auc.iloc[0]) if len(s) else np.nan


def famrel(parc, variant, relation):
    s = fam[(fam.parcellation == parc) & (fam.relation == relation) & (fam.variant == variant)]
    return float(s.auc.iloc[0]) if len(s) else np.nan


def rce(parc, src, tgt, est, metric="demeaned_pearson"):
    return float(np.nanmean(recon_est(r, parc, est, src, tgt, col=metric)))


def dle(parc, inp, est, t="CogCryst"):
    return float(np.nanmean(down_est(d, parc, est, inp, t, "lift_over_bvdemo")))


def savefig(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


# fixed model palette: PCA->PLS blue, BayesianRidge orange, KernelRidge green
MODC = {"pca_pls": "#2471A3", "bayesian_ridge": "#E67E22", "kernel_ridge": "#1E8449"}
MODL = {"pca_pls": "PCA→PLS", "bayesian_ridge": "BayesianRidge", "kernel_ridge": "KernelRidge"}


def iqr_whisk(ax, x, arr, w=0.25):
    """Draw IQR whisker (25-75 pct) + a white circle at the mean. Returns the mean."""
    arr = np.asarray(arr, float); arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    mn = float(arr.mean())
    if len(arr) >= 2:
        q1, q3 = np.percentile(arr, [25, 75])
        ax.vlines(x, q1, q3, color=C["ink"], lw=1.2, zorder=4)
        ax.hlines([q1, q3], x - w*0.13, x + w*0.13, color=C["ink"], lw=1.2, zorder=4)
    ax.scatter([x], [mn], s=20, facecolor="white", edgecolor=C["ink"], lw=1.0, zorder=5)
    return mn


def ci_whisk(ax, x, mid, lo, hi, w=0.25):
    """Whisker from a precomputed CI (lo,hi) + circle at mid."""
    ax.vlines(x, lo, hi, color=C["ink"], lw=1.2, zorder=4)
    ax.hlines([lo, hi], x - w*0.13, x + w*0.13, color=C["ink"], lw=1.2, zorder=4)
    ax.scatter([x], [mid], s=20, facecolor="white", edgecolor=C["ink"], lw=1.0, zorder=5)


# ============================ bar charts ============================
def _models_grouped(parc, conds, ylabel, title, fname, w=0.25, legend_loc="best"):
    """conds: list of (label, src, tgt). Grouped bars: 3 models per condition, IQR whiskers."""
    fig, ax = plt.subplots(figsize=(max(7.5, 2.2*len(conds)+2), 4.8))
    x = np.arange(len(conds))
    for mi, est in enumerate(EST):
        for ci, (lab, s, t) in enumerate(conds):
            arr = recon_est(r, parc, est, s, t)
            mn = float(np.nanmean(arr))
            pos = x[ci] + (mi - 1) * w
            ax.bar(pos, mn, w, color=MODC[est], edgecolor="white",
                   label=MODL[est] if ci == 0 else None)
            iqr_whisk(ax, pos, arr, w)
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in conds])
    ax.set_ylabel(ylabel); ax.set_ylim(0, None)
    ax.set_title(title); ax.legend(fontsize=10, loc=legend_loc)
    savefig(fig, fname)


def chart_asym(parc):
    _models_grouped(parc, [("FC→SC", "FC", "SC"), ("SC→FC", "SC", "FC")],
                    "demeaned pearson",
                    f"Cross-modal reconstruction by model — {PLABEL[parc]}", f"asym_{parc}")


def chart_baseline(parc):
    # full input ladder, single PCA->PLS, grouped by target (3-model view is on slide 6)
    sources = [("cross-modal", {"SC": "FC", "FC": "SC"}, CM),
               ("brain volume", {"SC": "bv", "FC": "bv"}, C["bv"]),
               ("demographics", {"SC": "demo", "FC": "demo"}, C["demo"]),
               ("bv+demo", {"SC": "bv+demo", "FC": "bv+demo"}, BL),
               ("connectome+bv+demo", {"SC": "FC+bv+demo", "FC": "SC+bv+demo"}, C["accent"])]
    tgts = ["SC", "FC"]; x = np.arange(len(tgts)); w = 0.16
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for si, (lab, srcmap, col) in enumerate(sources):
        for ti, tgt in enumerate(tgts):
            arr = recon_est(r, parc, "pca_pls", srcmap[tgt], tgt,
                            block=srcmap[tgt] in ("FC+bv+demo", "SC+bv+demo"))
            pos = x[ti] + (si - 2) * w
            ax.bar(pos, float(np.nanmean(arr)), w, color=col, edgecolor="white",
                   label=lab if ti == 0 else None)
            iqr_whisk(ax, pos, arr, w)
    ax.set_xticks(x); ax.set_xticklabels(["predict SC", "predict FC"])
    ax.set_ylabel("demeaned pearson"); ax.set_ylim(0, None)
    ax.set_title(f"Reconstruction: every input (PCA→PLS) — {PLABEL[parc]}")
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    savefig(fig, f"baseline_{parc}")


def chart_metrics(parc, tgt):
    # comparable metrics only (all in [0,1]); raw pearson + mse stay in the table
    mets = ["demeaned_pearson", "top1_acc", "avg_rank"]
    mlab = ["demeaned r", "top1 (fingerprint)", "avg_rank"]
    cm_src = "FC" if tgt == "SC" else "SC"
    combo = "FC+bv+demo" if tgt == "SC" else "SC+bv+demo"
    inputs = [(f"{cm_src}→{tgt} (connectome)", cm_src, CM),
              (f"bv+demo→{tgt} (baseline)", "bv+demo", BL),
              (f"connectome+bv+demo→{tgt}", combo, C["accent"])]
    x = np.arange(len(mets)); w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for ii, (lab, src, col) in enumerate(inputs):
        blk = src in ("FC+bv+demo", "SC+bv+demo")
        for mi, mm in enumerate(mets):
            arr = recon_est(r, parc, "pca_pls", src, tgt, col=mm, block=blk)
            pos = x[mi] + (ii - 1) * w
            ax.bar(pos, float(np.nanmean(arr)), w, color=col, edgecolor="white",
                   label=lab if mi == 0 else None)
            iqr_whisk(ax, pos, arr, w)
    ax.set_xticks(x); ax.set_xticklabels(mlab)
    ax.set_ylabel("score (higher = better)"); ax.set_ylim(0, 1.0)
    ax.set_title(f"Predicting {tgt}: comparable metrics — {PLABEL[parc]}")
    ax.legend(fontsize=9, loc="upper left")
    savefig(fig, f"metrics_{tgt}_{parc}")


def chart_cog(parc):
    inputs = ["obs_FC", "obs_FC+bv+demo", "obs_SC", "obs_SC+bv+demo", "pred_SC", "pred_FC"]
    labs = ["obs FC", "obs FC\n+bv+demo", "obs SC", "obs SC\n+bv+demo", "pred SC", "pred FC"]
    vals = [lift(parc, i) for i in inputs]
    cols = [FC, FC, CM, CM, PRED, PRED]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(range(len(inputs)), vals, color=cols, edgecolor="white", width=0.62)
    for i, inp in enumerate(inputs):
        iqr_whisk(ax, i, down_est(d, parc, "bayesian_ridge", inp, "CogCryst", "lift_over_bvdemo"), 0.62)
    ax.axhline(0, color=C["ink"], lw=1)
    ax.set_xticks(range(len(inputs))); ax.set_xticklabels(labs, fontsize=10)
    ax.set_ylabel("CogCryst lift over bv+demo")
    ax.set_title(f"Downstream cognition (BayesianRidge) — {PLABEL[parc]}")
    savefig(fig, f"cog_{parc}")


def chart_family(parc):
    variants = ["bvdemo_to_SC", "pred_SC_resid_bvdemo", "obs_SC"]
    labs = ["bv+demo\n(baseline)", "predicted SC\n(demographics removed)", "observed SC\n(reference)"]
    vals = [famauc(parc, v) for v in variants]
    cols = [BL, PRED, CM]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(range(len(variants)), vals, color=cols, edgecolor="white", width=0.6)
    for i, v in enumerate(variants):
        row = fam[(fam.parcellation == parc) & (fam.relation == "sibling") & (fam.variant == v)]
        if len(row):
            ci_whisk(ax, i, row.auc.iloc[0], row.auc_lo.iloc[0], row.auc_hi.iloc[0], 0.6)
    ax.axhline(0.5, color=BAD, ls="--", lw=1.3)
    ax.text(len(variants)-0.5, 0.515, "chance", color=BAD, fontsize=10, ha="right")
    ax.set_xticks(range(len(variants))); ax.set_xticklabels(labs, fontsize=10)
    ax.set_ylabel("sibling-separation AUC"); ax.set_ylim(0.45, 0.95)
    ax.set_title(f"Downstream family / heritability — {PLABEL[parc]}")
    savefig(fig, f"family_{parc}")


EST = ["pca_pls", "bayesian_ridge", "kernel_ridge"]
ESTLAB = {"pca_pls": "PCA→PLS", "bayesian_ridge": "BayesianRidge", "kernel_ridge": "KernelRidge"}
ESTC = {"pca_pls": C["SC"], "bayesian_ridge": C["FC"], "kernel_ridge": C["good"]}


def chart_models(parc):
    inputs = ["obs_FC", "obs_SC", "pred_FC"]
    labs = ["obs FC", "obs SC", "pred FC"]
    x = np.arange(len(inputs)); w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for ei, est in enumerate(EST):
        for ci, inp in enumerate(inputs):
            arr = down_est(d, parc, est, inp, "CogCryst", "lift_over_bvdemo")
            pos = x[ci] + (ei - 1)*w
            ax.bar(pos, float(np.nanmean(arr)), w, color=MODC[est], edgecolor="white",
                   label=MODL[est] if ci == 0 else None)
            iqr_whisk(ax, pos, arr, w)
    ax.axhline(0, color=C["ink"], lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_ylabel("CogCryst lift over bv+demo")
    ax.set_title(f"Downstream cognition by model — {PLABEL[parc]}")
    ax.legend(fontsize=10, loc="upper right")
    savefig(fig, f"models_{parc}")


def chart_q1(parc):
    groups = ["SC alone", "SC + bv+demo"]
    obs = [lift(parc, "obs_SC"), lift(parc, "obs_SC+bv+demo")]
    pred = [lift(parc, "pred_SC"), lift(parc, "pred_SC+bv+demo")]
    x = np.arange(2); w = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    obs_inp = ["obs_SC", "obs_SC+bv+demo"]; pred_inp = ["pred_SC", "pred_SC+bv+demo"]
    ax.bar(x - w/2, obs, w, color=CM, edgecolor="white", label="observed SC")
    ax.bar(x + w/2, pred, w, color=PRED, edgecolor="white", label="predicted SC (from FC)")
    for i in range(2):
        iqr_whisk(ax, x[i]-w/2, down_est(d, parc, "bayesian_ridge", obs_inp[i], "CogCryst", "lift_over_bvdemo"), w)
        iqr_whisk(ax, x[i]+w/2, down_est(d, parc, "bayesian_ridge", pred_inp[i], "CogCryst", "lift_over_bvdemo"), w)
    ref = lift(parc, "obs_FC+bv+demo")
    ax.axhline(ref, color=FC, ls="--", lw=1.5)
    ax.text(1.4, ref+0.004, f"observed FC+bv+demo (source) = {ref:+.3f}", color=FC,
            fontsize=9, ha="right")
    ax.axhline(0, color=C["ink"], lw=1)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("CogCryst lift over bv+demo")
    ax.set_title(f"Predicted SC beats observed SC — cognition (BayesianRidge, {PLABEL[parc]})")
    ax.legend(fontsize=10, loc="upper left")
    savefig(fig, f"q1_{parc}")


def chart_q2(parc):
    rels = ["MZ", "DZ", "sibling"]
    series = [("observed SC", "obs_SC", CM),
              ("predicted SC", "pred_SC_resid_bvdemo", PRED),
              ("combined predicted SC", "combined_pred_SC", C["accent"]),
              ("bv+demo baseline", "bvdemo_to_SC", BL)]
    fig, ax = plt.subplots(figsize=(9, 5))
    for lab, v, col in series:
        ax.plot(range(3), [famrel(parc, v, rl) for rl in rels], "-o", color=col, ms=8, lw=2, label=lab)
    ax.axhline(0.5, color=BAD, ls="--", lw=1.3); ax.text(2, 0.515, "chance", color=BAD, fontsize=9, ha="right")
    ax.set_xticks(range(3)); ax.set_xticklabels(["MZ twins", "DZ twins", "siblings"])
    ax.set_ylabel("separation AUC"); ax.set_ylim(0.45, 1.02)
    ax.set_title(f"Family signal by relatedness — {PLABEL[parc]}")
    ax.legend(fontsize=9, loc="lower left")
    savefig(fig, f"q2_{parc}")


def chart_q3(parc):
    targets = ["CogTotal", "CogFluid", "CogCryst"]; tl = ["Total", "Fluid", "Cryst"]
    x = np.arange(3); w = 0.26
    series = [("bv+demo (baseline)", "bv+demo", BL),
              ("observed SC (the source)", "obs_SC", CM),
              ("predicted FC (from SC)", "pred_FC", PRED)]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for si, (lab, inp, col) in enumerate(series):
        for ti, t in enumerate(targets):
            arr = down_est(d, parc, "bayesian_ridge", inp, t, "pearson")
            pos = x[ti] + (si - 1) * w
            ax.bar(pos, float(np.nanmean(arr)), w, color=col, edgecolor="white",
                   label=lab if ti == 0 else None)
            iqr_whisk(ax, pos, arr, w)
    ax.set_xticks(x); ax.set_xticklabels(tl)
    ax.set_ylabel("cognition prediction (pearson r)"); ax.set_ylim(0, None)
    ax.set_title(f"Cognition: baseline vs SC vs predicted FC (BayesianRidge, {PLABEL[parc]})")
    ax.legend(fontsize=9, loc="upper right")
    savefig(fig, f"q3_{parc}")


def build_charts(parc):
    chart_asym(parc); chart_baseline(parc); chart_metrics(parc, "SC")
    chart_metrics(parc, "FC"); chart_cog(parc); chart_family(parc); chart_models(parc)
    chart_q1(parc); chart_q2(parc); chart_q3(parc)


# ============================ LaTeX tables ============================
def f(x, dec=3, sign=False):
    return f"{x:+.{dec}f}" if sign else f"{x:.{dec}f}"


def tbl_asym(parc):
    a, b = rc(parc, "FC", "SC"), rc(parc, "SC", "FC")
    return (r"\begin{tabular}{@{}lc@{}}\toprule \textbf{direction} & \textbf{demeaned r} \\\midrule "
            rf"FC$\rightarrow$SC & {f(a)} \\ SC$\rightarrow$FC & {f(b)} \\\midrule "
            rf"\textbf{{ratio}} & \textbf{{{a/b:.2f}$\times$}} \\\bottomrule\end{{tabular}}")


def tbl_baseline(parc):
    """FULL: all subject-info sources + cross-modal, both targets."""
    rows = [("cross-modal connectome", ("FC", "SC"), ("SC", "FC"), False),
            ("brain volume (bv)", ("bv", "SC"), ("bv", "FC"), False),
            ("demographics (demo)", ("demo", "SC"), ("demo", "FC"), False),
            ("bv+demo", ("bv+demo", "SC"), ("bv+demo", "FC"), False),
            ("connectome + bv+demo", ("FC+bv+demo", "SC"), ("SC+bv+demo", "FC"), True)]
    body = ""
    for lab, sc, fc, bold in rows:
        a, b = f(rc(parc, *sc)), f(rc(parc, *fc))
        if bold:
            body += rf"\textbf{{{lab}}} & \textbf{{{a}}} & \textbf{{{b}}} \\ "
        else:
            body += rf"{lab} & {a} & {b} \\ "
    return (r"\begin{tabular}{@{}lcc@{}}\toprule \textbf{source} & "
            r"\textbf{$\rightarrow$ SC} & \textbf{$\rightarrow$ FC} \\\midrule "
            + body + r"\bottomrule\end{tabular}")


def tbl_metrics(parc, tgt):
    """FULL: every input (cross-modal, bv, demo, bv+demo) x all six metrics."""
    mets = ["demeaned_pearson", "pearson", "top1_acc", "avg_rank", "mse", "r2"]
    head = (r"\textbf{input} & \textbf{demeaned r} & \textbf{raw r} & \textbf{top1} & "
            r"\textbf{avg rank} & \textbf{mse} & \textbf{r$^2$}")
    cm_src = "FC" if tgt == "SC" else "SC"
    combo = "FC+bv+demo" if tgt == "SC" else "SC+bv+demo"
    inputs = [cm_src, "bv", "demo", "bv+demo", combo]
    body = ""
    for s in inputs:
        lab = rf"{s}$\rightarrow${tgt}"
        vals = [f(rc(parc, s, tgt, mm)) for mm in mets]
        if s == combo:
            body += rf"\textbf{{{lab}}} & " + " & ".join(rf"\textbf{{{v}}}" for v in vals) + r" \\ "
        else:
            body += rf"{lab} & " + " & ".join(vals) + r" \\ "
    return (r"\begin{tabular}{@{}lcccccc@{}}\toprule " + head + r" \\\midrule "
            + body + r"\bottomrule\end{tabular}")


def tbl_cog(parc):
    """FULL: all ten downstream inputs x (CogCryst r, lift on Total/Fluid/Cryst)."""
    rows = [("bv+demo (baseline)", "bv+demo", False),
            ("observed FC", "obs_FC", True),
            ("observed SC", "obs_SC", False),
            ("observed FC + SC", "obs_FC+obs_SC", False),
            ("observed FC + bv+demo", "obs_FC+bv+demo", True),
            ("observed SC + bv+demo", "obs_SC+bv+demo", False),
            ("predicted SC", "pred_SC", False),
            ("predicted FC", "pred_FC", False),
            ("predicted SC + bv+demo", "pred_SC+bv+demo", False),
            ("predicted FC + bv+demo", "pred_FC+bv+demo", False)]
    body = ""
    for lab, inp, bold in rows:
        cells = [f(cogp(parc, inp)), f(lift(parc, inp, "CogTotal"), sign=True),
                 f(lift(parc, inp, "CogFluid"), sign=True), f(lift(parc, inp, "CogCryst"), sign=True)]
        if bold:
            body += rf"\textbf{{{lab}}} & " + " & ".join(rf"\textbf{{{c}}}" for c in cells) + r" \\ "
        else:
            body += rf"{lab} & " + " & ".join(cells) + r" \\ "
    return (r"\begin{tabular}{@{}lcccc@{}}\toprule \textbf{input} & \textbf{CogCryst r} & "
            r"\textbf{Total lift} & \textbf{Fluid lift} & \textbf{Cryst lift} \\\midrule "
            + body + r"\bottomrule\end{tabular}")


def tbl_family(parc):
    """FULL: all eight predictors x MZ/DZ/sibling AUC."""
    rows = [("observed SC", "obs_SC", False), ("observed FC", "obs_FC", False),
            ("predicted SC (raw)", "pred_SC_raw", False),
            ("predicted SC (demog removed)", "pred_SC_resid_bvdemo", True),
            ("predicted FC (raw)", "pred_FC_raw", False),
            ("predicted FC (demog removed)", "pred_FC_resid_bvdemo", False),
            ("combined predicted SC", "combined_pred_SC", True),
            ("bv+demo baseline", "bvdemo_to_SC", False)]
    body = ""
    for lab, v, bold in rows:
        cells = [f(famrel(parc, v, "MZ")), f(famrel(parc, v, "DZ")), f(famrel(parc, v, "sibling"))]
        if bold:
            body += rf"\textbf{{{lab}}} & " + " & ".join(rf"\textbf{{{c}}}" for c in cells) + r" \\ "
        else:
            body += rf"{lab} & " + " & ".join(cells) + r" \\ "
    return (r"\begin{tabular}{@{}lccc@{}}\toprule \textbf{predictor} & \textbf{MZ} & "
            r"\textbf{DZ} & \textbf{sibling} \\\midrule " + body + r"\bottomrule\end{tabular}")


def tbl_models_recon(parc):
    def row(lab, fn):
        return rf"{lab} & " + " & ".join(fn(e) for e in EST) + r" \\ "
    body = row("FC$\\rightarrow$SC", lambda e: f(rce(parc, "FC", "SC", e)))
    body += row("SC$\\rightarrow$FC", lambda e: f(rce(parc, "SC", "FC", e)))
    body += (r"\textbf{asymmetry ratio} & "
             + " & ".join(rf"\textbf{{{rce(parc,'FC','SC',e)/rce(parc,'SC','FC',e):.2f}$\times$}}" for e in EST) + r" \\ ")
    body += row("SC$\\rightarrow$SC oracle", lambda e: f(rce(parc, "SC", "SC", e)))
    return (r"\begin{tabular}{@{}lccc@{}}\toprule \textbf{reconstruction} & "
            r"\textbf{PCA$\rightarrow$PLS} & \textbf{BayesianRidge} & \textbf{KernelRidge} \\\midrule "
            + body + r"\bottomrule\end{tabular}")


def tbl_models_down(parc):
    rows = [("obs FC", "obs_FC"), ("obs SC", "obs_SC"), ("pred FC", "pred_FC")]
    body = "".join(rf"{lab} & " + " & ".join(f(dle(parc, inp, e), sign=True) for e in EST) + r" \\ "
                   for lab, inp in rows)
    return (r"\begin{tabular}{@{}lccc@{}}\toprule \textbf{CogCryst lift} & "
            r"\textbf{PCA$\rightarrow$PLS} & \textbf{BayesianRidge} & \textbf{KernelRidge} \\\midrule "
            + body + r"\bottomrule\end{tabular}")


def tbl_q1_flip():
    rows = [("observed SC", "obs_SC"), ("predicted SC", "pred_SC"),
            ("observed SC + bv+demo", "obs_SC+bv+demo"),
            ("predicted SC + bv+demo", "pred_SC+bv+demo")]
    body = ""
    for lab, inp in rows:
        bold = inp == "pred_SC+bv+demo"
        g = f(lift("Glasser", inp), sign=True); s = f(lift("4S456Parcels", inp), sign=True)
        if bold:
            body += rf"\textbf{{{lab}}} & \textbf{{{g}}} & \textbf{{{s}}} \\ "
        else:
            body += rf"{lab} & {g} & {s} \\ "
    body += r"\midrule "
    body += (rf"\textit{{(source) observed FC + bv+demo}} & \textit{{{f(lift('Glasser','obs_FC+bv+demo'),sign=True)}}} "
             rf"& \textit{{{f(lift('4S456Parcels','obs_FC+bv+demo'),sign=True)}}} \\ ")
    return (r"\begin{tabular}{@{}lcc@{}}\toprule \textbf{input} & \textbf{Glasser} & \textbf{4S456} "
            r"\\\midrule " + body + r"\bottomrule\end{tabular}")


def tbl_q1_caveat():
    P = "Glasser"
    body = ""
    for lab, inp in [("predicted SC + bv+demo", "pred_SC+bv+demo"),
                     ("observed SC + bv+demo", "obs_SC+bv+demo")]:
        body += rf"{lab} & " + " & ".join(f(dle(P, inp, e), sign=True) for e in EST) + r" \\ "
    verdict = " & ".join((r"\textbf{yes}" if dle(P, "pred_SC+bv+demo", e) > dle(P, "obs_SC+bv+demo", e)
                          else "no") for e in EST)
    body += r"\midrule predicted $>$ observed? & " + verdict + r" \\ "
    return (r"\begin{tabular}{@{}lccc@{}}\toprule \textbf{CogCryst lift (Glasser)} & "
            r"\textbf{PCA$\rightarrow$PLS} & \textbf{BayesianRidge} & \textbf{KernelRidge} \\\midrule "
            + body + r"\bottomrule\end{tabular}")


def tbl_q2(parc):
    rows = [("observed SC", "obs_SC"), ("predicted SC", "pred_SC_resid_bvdemo"),
            ("predicted FC", "pred_FC_resid_bvdemo"),
            ("combined predicted SC", "combined_pred_SC"), ("bv+demo baseline", "bvdemo_to_SC")]
    body = ""
    for lab, v in rows:
        body += rf"{lab} & {f(famrel(parc,v,'MZ'))} & {f(famrel(parc,v,'DZ'))} & {f(famrel(parc,v,'sibling'))} \\ "
    return (r"\begin{tabular}{@{}lccc@{}}\toprule \textbf{predictor} & \textbf{MZ twins} & "
            r"\textbf{DZ twins} & \textbf{siblings} \\\midrule " + body + r"\bottomrule\end{tabular}")


def tbl_q3():
    """Glasser FULL: raw CogXxx pearson r for the baseline + every source/prediction
    (incl. +bv+demo combos). Bold where a prediction beats the source it came from."""
    P = "Glasser"; targets = ["CogTotal", "CogFluid", "CogCryst"]
    rows = [("bv+demo (baseline)", "bv+demo", None),
            ("MID", None, None),
            ("observed FC", "obs_FC", None),
            ("$\\rightarrow$ predicted SC", "pred_SC", "obs_FC"),
            ("observed FC + bv+demo", "obs_FC+bv+demo", None),
            ("$\\rightarrow$ predicted SC + bv+demo", "pred_SC+bv+demo", "obs_FC+bv+demo"),
            ("MID", None, None),
            ("observed SC", "obs_SC", None),
            ("$\\rightarrow$ predicted FC", "pred_FC", "obs_SC"),
            ("observed SC + bv+demo", "obs_SC+bv+demo", None),
            ("$\\rightarrow$ predicted FC + bv+demo", "pred_FC+bv+demo", "obs_SC+bv+demo")]
    body = ""
    for lab, inp, src in rows:
        if lab == "MID":
            body += r"\midrule "
            continue
        cells = []
        for t in targets:
            v = cogp(P, inp, t)
            if src is not None and v > cogp(P, src, t):
                cells.append(rf"\textbf{{{f(v)}}}")
            else:
                cells.append(f(v))
        body += rf"{lab} & " + " & ".join(cells) + r" \\ "
    return (r"\begin{tabular}{@{}lccc@{}}\toprule \textbf{input (CogXxx pearson r)} & \textbf{CogTotal} & "
            r"\textbf{CogFluid} & \textbf{CogCryst} \\\midrule " + body + r"\bottomrule\end{tabular}")


# ============================ deck assembly ============================
CAP = {
 "asym": r"Across all three models the FC$\rightarrow$SC bars are $\sim$1.6$\times$ taller than SC$\rightarrow$FC: FC reconstructs SC's individual deviations far better than the reverse, and the ratio barely changes by model, so it is not an estimator artifact. Whisker = 25th--75th percentile across the 10 seeds; white circle = mean.",
 "baseline": r"For \emph{both} targets bv+demo matches or beats the cross-modal connectome: a cheap age+sex+volume vector reconstructs the missing connectome as well as the other modality does. Adding the connectome on top of bv+demo (gold) gives only a marginal extra, so the connectome adds little beyond subject information. Note the dissociation: bv (volume) wins for SC, demo wins for FC.",
 "metrics_SC": r"Raw r is $\sim$0.9 for every input (dominated by the shared group-mean connectome, so uninformative); demeaned r is the honest individual-deviation metric, where bv+demo leads. Whisker = 25th--75th percentile over the 10 seeds.",
 "metrics_FC": r"Same picture predicting FC, with one twist: the real connectome (SC) beats bv+demo on the fingerprint metrics (top1, avg rank), i.e.\ it carries subject-identity information the cheap baseline lacks.",
 "cog": r"Only observed FC and observed FC+bv+demo rise above zero (beat the baseline). Observed SC and both predicted connectomes sit at or below zero; predicted FC is actively harmful. Wide whiskers (25--75th pct over seeds) show even the FC lift is variable.",
 "family": r"Predicted SC separates siblings far above the bv+demo baseline and close to observed SC: the predicted connectome keeps heritable family structure. Whisker = bootstrap 95\% CI (family is a single pipeline with no seed split).",
 "models": r"The observed-FC cognition lift is positive under all three models, but the negatives (obs SC, pred FC) flip sign under KernelRidge (green): KernelRidge's scalar regression is numerically unstable, which is why cognition is reported with BayesianRidge.",
 "q1": r"Predicted SC (purple) beats observed SC (blue) because predicted SC is a projection of FC, the modality that carries cognition. It never reaches observed FC itself (dashed line): you would always do better using FC directly.",
 "q2": r"Every predictor weakens from MZ twins to DZ twins to siblings (a genetic dose-response). The combined predictor (gold) falls to the chance line at siblings: optimizing for reconstruction discards the fine-grained family signal.",
 "q3": r"Every connectome row sits below the bv+demo baseline (grey, tallest). `Beating the source' just means less far below: predicted FC edges observed SC on CogFluid, and predicted FC+bv+demo beats observed SC+bv+demo on CogTotal and CogFluid (see table) --- a mild denoising of the weak modality. Predicted SC never beats observed FC, the strong modality.",
}


def section_frames(parc, appendix=False):
    P = parc
    tag = " (4S456 backup)" if appendix else ""
    fr = []

    def table_frame(title, table, size=r"\large"):
        fr.append(r"\begin{frame}{%s%s}" % (title, tag) +
                  r"\vfill\centering" + size + "\n" + table + r"\vfill\end{frame}")

    def chart_frame(title, img, cap=""):
        capt = (r"\\[3pt]\begin{minipage}{0.95\linewidth}\scriptsize " + cap +
                r"\end{minipage}") if cap else ""
        fr.append(r"\begin{frame}{%s%s}" % (title, tag) +
                  r"\centering\includegraphics[height=0.66\textheight,width=\linewidth,keepaspectratio]{figures/simplified/%s.pdf}" % img +
                  capt + r"\end{frame}")

    PLS = r"\,{\footnotesize[PCA$\rightarrow$PLS]}"
    BR = r"\,{\footnotesize[BayesianRidge]}"
    fr.append(r"\section{%s%s}" % ("Asymmetry" if not appendix else "Appendix: 4S456", tag))
    table_frame("1 · Cross-modal prediction is directional" + PLS, tbl_asym(P))
    chart_frame("1 · Cross-modal prediction is directional" + PLS, f"asym_{P}", CAP["asym"])
    table_frame("2 · A cheap baseline beats cross-modal reconstruction" + PLS, tbl_baseline(P))
    chart_frame("2 · A cheap baseline beats cross-modal reconstruction" + PLS, f"baseline_{P}", CAP["baseline"])
    table_frame("3a · Predicting SC: every metric" + PLS, tbl_metrics(P, "SC"), r"\small")
    chart_frame("3a · Predicting SC: comparable metrics" + PLS, f"metrics_SC_{P}", CAP["metrics_SC"])
    table_frame("3b · Predicting FC: every metric" + PLS, tbl_metrics(P, "FC"), r"\small")
    chart_frame("3b · Predicting FC: comparable metrics" + PLS, f"metrics_FC_{P}", CAP["metrics_FC"])
    table_frame("4 · Downstream cognition: only observed FC helps" + BR, tbl_cog(P), r"\footnotesize")
    chart_frame("4 · Downstream cognition: only observed FC helps" + BR, f"cog_{P}", CAP["cog"])
    table_frame("5 · Downstream family: the predicted connectome carries it" + PLS, tbl_family(P), r"\small")
    chart_frame("5 · Downstream family: the predicted connectome carries it" + PLS, f"family_{P}", CAP["family"])
    # 6 · model robustness (two stacked tables on one frame)
    fr.append(r"\begin{frame}{6 · Are the results model-dependent?%s}" % tag +
              r"\vfill\centering\footnotesize Headlines replicate across models; "
              r"the only divergences are PLS's low oracle and KernelRidge's unstable scalar rows.\\[6pt]"
              + tbl_models_recon(P) + r"\\[8pt]" + tbl_models_down(P) + r"\vfill\end{frame}")
    chart_frame("6 · Are the results model-dependent?", f"models_{P}", CAP["models"])
    return fr


def extras_frames():
    """Q1 (predicted beats observed) then Q2 (family relatedness), Glasser."""
    P = "Glasser"
    fr = [r"\section{Backup: deeper dives (Glasser)}"]
    # Q1
    fr.append(r"\begin{frame}{Q1 · Can a predicted connectome beat the real one?\,{\footnotesize[BayesianRidge]}}"
              r"\vfill\centering\footnotesize Predicted SC beats \emph{observed} SC for cognition --- "
              r"because predicted SC is a projection of FC (the strong modality). It still loses to FC itself.\\[6pt]"
              + tbl_q1_flip() + r"\\[8pt]" + r"\footnotesize …but the flip is estimator-specific:\\[3pt]"
              + tbl_q1_caveat() + r"\vfill\end{frame}")
    fr.append(r"\begin{frame}{Q1 · Predicted SC beats observed SC --- but never the source\,{\footnotesize[BayesianRidge]}}"
              r"\centering\includegraphics[height=0.66\textheight,width=\linewidth,keepaspectratio]{figures/simplified/q1_%s.pdf}"
              r"\\[3pt]\begin{minipage}{0.95\linewidth}\scriptsize %s\end{minipage}\end{frame}" % (P, CAP["q1"]))
    # Q2
    fr.append(r"\begin{frame}{Q2 · Family signal across relatedness (MZ / DZ / sibling)\,{\footnotesize[PCA$\rightarrow$PLS]}}"
              r"\vfill\centering\footnotesize A genetic dose-response; the reconstruction-optimized "
              r"\emph{combined} predictor collapses to chance for siblings.\\[6pt]"
              + tbl_q2(P) + r"\vfill\end{frame}")
    fr.append(r"\begin{frame}{Q2 · Heritability gradient \& the objective tradeoff\,{\footnotesize[PCA$\rightarrow$PLS]}}"
              r"\centering\includegraphics[height=0.66\textheight,width=\linewidth,keepaspectratio]{figures/simplified/q2_%s.pdf}"
              r"\\[3pt]\begin{minipage}{0.95\linewidth}\scriptsize %s\end{minipage}\end{frame}" % (P, CAP["q2"]))
    # Q3
    fr.append(r"\begin{frame}{Q3 · Can a prediction beat the connectome it came from?\,{\footnotesize[BayesianRidge]}}"
              r"\vfill\centering\footnotesize Predicted FC beats its source SC on the noisy targets (a denoising effect) --- "
              r"but both stay below the baseline; predicted SC never beats its source FC.\\[8pt]"
              + tbl_q3() + r"\vfill\end{frame}")
    fr.append(r"\begin{frame}{Q3 · Prediction beats source only as `less harmful'\,{\footnotesize[BayesianRidge]}}"
              r"\centering\includegraphics[height=0.66\textheight,width=\linewidth,keepaspectratio]{figures/simplified/q3_%s.pdf}"
              r"\\[3pt]\begin{minipage}{0.95\linewidth}\scriptsize %s\end{minipage}\end{frame}" % (P, CAP["q3"]))
    return fr


def main():
    build_charts(PARC_MAIN)
    build_charts(PARC_APP)
    L = [
        r"\documentclass[aspectratio=169]{beamer}",
        r"\usetheme{default}\usecolortheme{seahorse}",
        r"\setbeamertemplate{navigation symbols}{}",
        r"\setbeamertemplate{footline}[frame number]",
        r"\usepackage{graphicx}\usepackage{booktabs}",
        r"\setbeamertemplate{section in toc}{\inserttocsectionnumber.~\inserttocsection}",
        r"\title{What Cross-Modal Connectome Prediction Does and Does Not Buy}",
        r"\subtitle{Reconstruction, baselines, and downstream utility (Glasser; 4S456 in backup)}",
        r"\author{Adel Sahuc}\date{}",
        r"\begin{document}",
        r"\frame{\titlepage}",
    ]
    L += section_frames(PARC_MAIN, appendix=False)
    # takeaways + thank you
    L.append(r"\section{Takeaways}\begin{frame}{Takeaways}\large\begin{enumerate}"
             r"\item Cross-modal prediction is \textbf{real and directional}: FC$\rightarrow$SC $>$ SC$\rightarrow$FC (1.61$\times$)."
             r"\item But a \textbf{cheap bv+demo baseline beats it} at reconstruction --- only the "
             r"\emph{fingerprint/identity} signal is uniquely connectomic."
             r"\item Downstream, signal splits cleanly: \textbf{observed FC} carries (crystallized) cognition; "
             r"\textbf{predicted SC} carries family/heritable signal."
             r"\end{enumerate}\end{frame}")
    L.append(r"\begin{frame}\centering\vfill{\Huge Thank you}\\[8pt]"
             r"\large Questions?\\[12pt]\normalsize Backup: deeper dives (Q1, Q2), then 4S456 replication.\vfill\end{frame}")
    L += extras_frames()
    L += section_frames(PARC_APP, appendix=True)
    L.append(r"\end{document}")
    (HERE / "simplified_slides.tex").write_text("\n".join(L))
    print("wrote simplified_slides.tex and figures/simplified/*.pdf")


if __name__ == "__main__":
    main()
