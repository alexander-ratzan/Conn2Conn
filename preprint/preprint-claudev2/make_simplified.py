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
    return float(np.nanmean(recon_est(r, parc, "pca_pls", src, tgt, col=metric)))


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


# ============================ bar charts ============================
def chart_asym(parc):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    vals = [rc(parc, "FC", "SC"), rc(parc, "SC", "FC")]
    ax.bar(["FC→SC", "SC→FC"], vals, color=[FC, CM], edgecolor="white", width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=13, fontweight="bold")
    ax.text(0.5, max(vals) + 0.02, f"{vals[0]/vals[1]:.2f}× asymmetry", ha="center",
            fontsize=14, fontweight="bold", color=C["ink"])
    ax.set_ylabel("demeaned pearson"); ax.set_ylim(0, max(vals) + 0.04)
    ax.set_title(f"Cross-modal reconstruction — {PLABEL[parc]}")
    savefig(fig, f"asym_{parc}")


def chart_baseline(parc):
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    groups = ["predict SC", "predict FC"]
    cross = [rc(parc, "FC", "SC"), rc(parc, "SC", "FC")]
    base = [rc(parc, "bv+demo", "SC"), rc(parc, "bv+demo", "FC")]
    x = np.arange(2); w = 0.36
    b1 = ax.bar(x - w/2, cross, w, color=CM, edgecolor="white", label="cross-modal connectome")
    b2 = ax.bar(x + w/2, base, w, color=BL, edgecolor="white", label="bv+demo (cheap baseline)")
    for bars in (b1, b2):
        for rect in bars:
            ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.004,
                    f"{rect.get_height():.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("demeaned pearson"); ax.set_ylim(0, max(base+cross)+0.03)
    ax.set_title(f"Cheap baseline vs cross-modal — {PLABEL[parc]}")
    ax.legend(fontsize=11)
    savefig(fig, f"baseline_{parc}")


def chart_metrics(parc, tgt):
    # comparable metrics only (all in [0,1]); raw pearson + mse stay in the table
    mets = ["demeaned_pearson", "top1_acc", "avg_rank"]
    mlab = ["demeaned r", "top1 (fingerprint)", "avg_rank"]
    cm_src = "FC" if tgt == "SC" else "SC"
    cm = [rc(parc, cm_src, tgt, mm) for mm in mets]
    bl = [rc(parc, "bv+demo", tgt, mm) for mm in mets]
    x = np.arange(len(mets)); w = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(x - w/2, cm, w, color=CM, edgecolor="white", label=f"{cm_src}→{tgt} (connectome)")
    ax.bar(x + w/2, bl, w, color=BL, edgecolor="white", label=f"bv+demo→{tgt} (baseline)")
    for i in range(len(mets)):
        ax.text(x[i]-w/2, cm[i]+0.008, f"{cm[i]:.3f}", ha="center", fontsize=10)
        ax.text(x[i]+w/2, bl[i]+0.008, f"{bl[i]:.3f}", ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(mlab)
    ax.set_ylabel("score (higher = better)"); ax.set_ylim(0, 1.0)
    ax.set_title(f"Predicting {tgt}: all comparable metrics — {PLABEL[parc]}")
    ax.legend(fontsize=11, loc="upper left")
    savefig(fig, f"metrics_{tgt}_{parc}")


def chart_cog(parc):
    inputs = ["obs_FC", "obs_FC+bv+demo", "obs_SC", "obs_SC+bv+demo", "pred_SC", "pred_FC"]
    labs = ["obs FC", "obs FC\n+bv+demo", "obs SC", "obs SC\n+bv+demo", "pred SC", "pred FC"]
    vals = [lift(parc, i) for i in inputs]
    cols = [FC, FC, CM, CM, PRED, PRED]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(range(len(inputs)), vals, color=cols, edgecolor="white", width=0.62)
    for rect, v in zip(bars, vals):
        ax.text(rect.get_x()+rect.get_width()/2, v + (0.004 if v >= 0 else -0.012),
                f"{v:+.3f}", ha="center", fontsize=11, fontweight="bold",
                va="bottom" if v >= 0 else "top")
    ax.axhline(0, color=C["ink"], lw=1)
    ax.set_ylim(min(vals) - 0.05, max(vals) + 0.045)
    ax.set_xticks(range(len(inputs))); ax.set_xticklabels(labs, fontsize=10)
    ax.set_ylabel("CogCryst lift over bv+demo")
    ax.set_title(f"Downstream cognition — {PLABEL[parc]}")
    savefig(fig, f"cog_{parc}")


def chart_family(parc):
    variants = ["bvdemo_to_SC", "pred_SC_resid_bvdemo", "obs_SC"]
    labs = ["bv+demo\n(baseline)", "predicted SC\n(demographics removed)", "observed SC\n(reference)"]
    vals = [famauc(parc, v) for v in variants]
    cols = [BL, PRED, CM]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(range(len(variants)), vals, color=cols, edgecolor="white", width=0.6)
    for rect, v in zip(bars, vals):
        ax.text(rect.get_x()+rect.get_width()/2, v + 0.008, f"{v:.3f}", ha="center",
                fontsize=12, fontweight="bold")
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
        vals = [dle(parc, inp, est) for inp in inputs]
        bars = ax.bar(x + (ei - 1)*w, vals, w, color=ESTC[est], edgecolor="white",
                      label=ESTLAB[est])
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x()+rect.get_width()/2, v + (0.004 if v >= 0 else -0.004),
                    f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5)
    ax.axhline(0, color=C["ink"], lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_ylabel("CogCryst lift over bv+demo")
    ax.set_title(f"Downstream cognition by model — {PLABEL[parc]}")
    ax.legend(fontsize=10, loc="upper right")
    ax.annotate("KernelRidge scalar rows\nflip sign (ill-conditioned)",
                xy=(1 + w, dle(parc, "obs_SC", "kernel_ridge")), xytext=(1.42, 0.135),
                fontsize=8.5, color=C["good"], fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=C["good"]))
    savefig(fig, f"models_{parc}")


def chart_q1(parc):
    groups = ["SC alone", "SC + bv+demo"]
    obs = [lift(parc, "obs_SC"), lift(parc, "obs_SC+bv+demo")]
    pred = [lift(parc, "pred_SC"), lift(parc, "pred_SC+bv+demo")]
    x = np.arange(2); w = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - w/2, obs, w, color=CM, edgecolor="white", label="observed SC")
    ax.bar(x + w/2, pred, w, color=PRED, edgecolor="white", label="predicted SC (from FC)")
    for i in range(2):
        ax.text(x[i]-w/2, obs[i] + (0.004 if obs[i] >= 0 else -0.004), f"{obs[i]:+.3f}",
                ha="center", va="bottom" if obs[i] >= 0 else "top", fontsize=10)
        ax.text(x[i]+w/2, pred[i] + (0.004 if pred[i] >= 0 else -0.004), f"{pred[i]:+.3f}",
                ha="center", va="bottom" if pred[i] >= 0 else "top", fontsize=10)
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
              ("predicted SC (demog removed)", "pred_SC_resid_bvdemo", PRED),
              ("combined predicted SC", "combined_pred_SC", C["accent"]),
              ("bv+demo baseline", "bvdemo_to_SC", BL)]
    fig, ax = plt.subplots(figsize=(9, 5))
    for lab, v, col in series:
        ax.plot(range(3), [famrel(parc, v, rl) for rl in rels], "-o", color=col, ms=8, lw=2, label=lab)
    ax.axhline(0.5, color=BAD, ls="--", lw=1.3); ax.text(2, 0.515, "chance", color=BAD, fontsize=9, ha="right")
    ax.annotate("combined predictor\ncollapses to chance\nfor siblings",
                xy=(2, famrel(parc, "combined_pred_SC", "sibling")), xytext=(1.1, 0.6),
                fontsize=9, color=C["accent"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C["accent"]))
    ax.set_xticks(range(3)); ax.set_xticklabels(["MZ twins", "DZ twins", "siblings"])
    ax.set_ylabel("separation AUC"); ax.set_ylim(0.45, 1.02)
    ax.set_title(f"Family signal by relatedness — {PLABEL[parc]}")
    ax.legend(fontsize=9, loc="lower left")
    savefig(fig, f"q2_{parc}")


def build_charts(parc):
    chart_asym(parc); chart_baseline(parc); chart_metrics(parc, "SC")
    chart_metrics(parc, "FC"); chart_cog(parc); chart_family(parc); chart_models(parc)
    chart_q1(parc); chart_q2(parc)


# ============================ LaTeX tables ============================
def f(x, dec=3, sign=False):
    return f"{x:+.{dec}f}" if sign else f"{x:.{dec}f}"


def tbl_asym(parc):
    a, b = rc(parc, "FC", "SC"), rc(parc, "SC", "FC")
    return (r"\begin{tabular}{@{}lc@{}}\toprule \textbf{direction} & \textbf{demeaned r} \\\midrule "
            rf"FC$\rightarrow$SC & {f(a)} \\ SC$\rightarrow$FC & {f(b)} \\\midrule "
            rf"\textbf{{ratio}} & \textbf{{{a/b:.2f}$\times$}} \\\bottomrule\end{{tabular}}")


def tbl_baseline(parc):
    return (r"\begin{tabular}{@{}lcc@{}}\toprule \textbf{target} & \textbf{cross-modal} & \textbf{bv+demo} \\\midrule "
            rf"$\rightarrow$ SC & {f(rc(parc,'FC','SC'))} & \textbf{{{f(rc(parc,'bv+demo','SC'))}}} \\ "
            rf"$\rightarrow$ FC & {f(rc(parc,'SC','FC'))} & \textbf{{{f(rc(parc,'bv+demo','FC'))}}} \\\bottomrule\end{{tabular}}")


def tbl_metrics(parc, tgt):
    mets = ["demeaned_pearson", "pearson", "top1_acc", "avg_rank", "mse", "r2"]
    head = r"\textbf{input} & \textbf{demeaned r} & \textbf{raw r} & \textbf{top1} & \textbf{avg rank} & \textbf{mse} & \textbf{r$^2$}"
    cm_src = "FC" if tgt == "SC" else "SC"
    cm = " & ".join(f(rc(parc, cm_src, tgt, mm)) for mm in mets)
    bl = " & ".join(f(rc(parc, "bv+demo", tgt, mm)) for mm in mets)
    return (r"\begin{tabular}{@{}lcccccc@{}}\toprule " + head + r" \\\midrule "
            rf"{cm_src}$\rightarrow${tgt} & {cm} \\ "
            rf"bv+demo$\rightarrow${tgt} & {bl} \\\bottomrule\end{{tabular}}")


def tbl_cog(parc):
    rows = [("observed FC", "obs_FC"), ("observed FC + bv+demo", "obs_FC+bv+demo"),
            ("observed SC", "obs_SC"), ("observed SC + bv+demo", "obs_SC+bv+demo"),
            ("predicted SC", "pred_SC"), ("predicted FC", "pred_FC")]
    body = ""
    for lab, inp in rows:
        body += rf"{lab} & {f(cogp(parc,inp))} & {f(lift(parc,inp),sign=True)} \\ "
    return (r"\begin{tabular}{@{}lcc@{}}\toprule \textbf{input} & \textbf{CogCryst r} & \textbf{lift vs bv+demo} \\\midrule "
            f"{body}" + r"\bottomrule\end{tabular}")


def tbl_family(parc):
    rows = [("bv+demo (baseline)", "bvdemo_to_SC"),
            ("predicted SC (demographics removed)", "pred_SC_resid_bvdemo"),
            ("observed SC (reference)", "obs_SC")]
    body = "".join(rf"{lab} & {f(famauc(parc,v))} \\ " for lab, v in rows)
    return (r"\begin{tabular}{@{}lc@{}}\toprule \textbf{predictor} & \textbf{sibling AUC} \\\midrule "
            f"{body}" + r"\bottomrule\end{tabular}")


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
    rows = [("observed SC", "obs_SC"), ("predicted SC (demog removed)", "pred_SC_resid_bvdemo"),
            ("predicted FC (demog removed)", "pred_FC_resid_bvdemo"),
            ("combined predicted SC", "combined_pred_SC"), ("bv+demo baseline", "bvdemo_to_SC")]
    body = ""
    for lab, v in rows:
        body += rf"{lab} & {f(famrel(parc,v,'MZ'))} & {f(famrel(parc,v,'DZ'))} & {f(famrel(parc,v,'sibling'))} \\ "
    return (r"\begin{tabular}{@{}lccc@{}}\toprule \textbf{predictor} & \textbf{MZ twins} & "
            r"\textbf{DZ twins} & \textbf{siblings} \\\midrule " + body + r"\bottomrule\end{tabular}")


# ============================ deck assembly ============================
def section_frames(parc, appendix=False):
    P = parc
    tag = " (4S456 backup)" if appendix else ""
    fr = []

    def table_frame(title, table):
        fr.append(r"\begin{frame}{%s%s}" % (title, tag) +
                  r"\vfill\centering\large" + "\n" + table + r"\vfill\end{frame}")

    def chart_frame(title, img):
        fr.append(r"\begin{frame}{%s%s}" % (title, tag) +
                  r"\centering\includegraphics[height=0.78\textheight,width=\linewidth,keepaspectratio]{figures/simplified/%s.pdf}" % img +
                  r"\end{frame}")

    PLS = r"\,{\footnotesize[PCA$\rightarrow$PLS]}"
    BR = r"\,{\footnotesize[BayesianRidge]}"
    fr.append(r"\section{%s%s}" % ("Asymmetry" if not appendix else "Appendix: 4S456", tag))
    table_frame("1 · Cross-modal prediction is directional" + PLS, tbl_asym(P))
    chart_frame("1 · Cross-modal prediction is directional" + PLS, f"asym_{P}")
    table_frame("2 · A cheap baseline beats cross-modal reconstruction" + PLS, tbl_baseline(P))
    chart_frame("2 · A cheap baseline beats cross-modal reconstruction" + PLS, f"baseline_{P}")
    table_frame("3a · Predicting SC: every metric" + PLS, tbl_metrics(P, "SC"))
    chart_frame("3a · Predicting SC: comparable metrics" + PLS, f"metrics_SC_{P}")
    table_frame("3b · Predicting FC: every metric" + PLS, tbl_metrics(P, "FC"))
    chart_frame("3b · Predicting FC: comparable metrics" + PLS, f"metrics_FC_{P}")
    table_frame("4 · Downstream cognition: only observed FC helps" + BR, tbl_cog(P))
    chart_frame("4 · Downstream cognition: only observed FC helps" + BR, f"cog_{P}")
    table_frame("5 · Downstream family: the predicted connectome carries it" + PLS, tbl_family(P))
    chart_frame("5 · Downstream family: the predicted connectome carries it" + PLS, f"family_{P}")
    # 6 · model robustness (two stacked tables on one frame)
    fr.append(r"\begin{frame}{6 · Are the results model-dependent?%s}" % tag +
              r"\vfill\centering\footnotesize Headlines replicate across models; "
              r"the only divergences are PLS's low oracle and KernelRidge's unstable scalar rows.\\[6pt]"
              + tbl_models_recon(P) + r"\\[8pt]" + tbl_models_down(P) + r"\vfill\end{frame}")
    chart_frame("6 · Are the results model-dependent?", f"models_{P}")
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
              r"\centering\includegraphics[height=0.78\textheight,width=\linewidth,keepaspectratio]{figures/simplified/q1_%s.pdf}\end{frame}" % P)
    # Q2
    fr.append(r"\begin{frame}{Q2 · Family signal across relatedness (MZ / DZ / sibling)\,{\footnotesize[PCA$\rightarrow$PLS]}}"
              r"\vfill\centering\footnotesize A genetic dose-response; the reconstruction-optimized "
              r"\emph{combined} predictor collapses to chance for siblings.\\[6pt]"
              + tbl_q2(P) + r"\vfill\end{frame}")
    fr.append(r"\begin{frame}{Q2 · Heritability gradient \& the objective tradeoff\,{\footnotesize[PCA$\rightarrow$PLS]}}"
              r"\centering\includegraphics[height=0.78\textheight,width=\linewidth,keepaspectratio]{figures/simplified/q2_%s.pdf}\end{frame}" % P)
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
        r"\begin{frame}{The question}\Large\begin{itemize}"
        r"\item If FC predicts SC (or vice versa), can the \emph{predicted} connectome stand in for the missing one?"
        r"\item Two sub-questions: (1) can we \emph{reconstruct} it? (2) is the reconstruction \emph{useful}?"
        r"\item All numbers: HCP-YA, 10 frozen family-aware splits; reconstruction in demeaned pearson, "
        r"downstream in BayesianRidge lift over a cheap \texttt{bv+demo} baseline.\end{itemize}\end{frame}",
    ]
    L += section_frames(PARC_MAIN, appendix=False)
    # takeaways + thank you
    L.append(r"\section{Takeaways}\begin{frame}{Takeaways}\large\begin{enumerate}"
             r"\item Cross-modal prediction is \textbf{real and directional}: FC$\rightarrow$SC $>$ SC$\rightarrow$FC (1.61$\times$)."
             r"\item But a \textbf{cheap bv+demo baseline beats it} at reconstruction --- only the "
             r"\emph{fingerprint/identity} signal is uniquely connectomic."
             r"\item Downstream, signal splits cleanly: \textbf{observed FC} carries (crystallized) cognition; "
             r"\textbf{predicted SC} carries family/heritable signal."
             r"\item \textbf{Always beat the cheap baseline} before claiming a connectome is useful."
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
