#!/usr/bin/env python3
"""Finding-by-finding figures F1-F10, each across both parcellations and (where it
applies) across estimators. Writes figures/F{n}_*.pdf/.png.

Run with the dev-env interpreter from this directory.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from style import (load_recon, load_downstream, load_family,
                   recon_est, down_est, m_sd, C, PARCS, PARC_LABEL,
                   ESTIMATORS, EST_LABEL, EST_COLOR, panel_tag, savefig,
                   FAM, SANITY, NLIN, TRACT)

r = load_recon()
d = load_downstream()
fam = load_family()
COGS = ["CogTotal", "CogFluid", "CogCryst"]
PSHORT = {"Glasser": "Glasser", "4S456Parcels": "4S456"}


def barwerr(ax, x, vals, err, **kw):
    return ax.bar(x, vals, yerr=err, capsize=2.5, edgecolor="white", **kw)


# ============================================================ F1
def F1():
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.24,
                          left=0.07, right=0.98, top=0.9, bottom=0.08)
    # A: FC->SC vs SC->FC across estimators x parc
    axA = fig.add_subplot(gs[0, 0])
    x = np.arange(len(ESTIMATORS)); w = 0.18
    for pi, parc in enumerate(PARCS):
        for di, (src, tgt, hatch) in enumerate([("FC", "SC", ""), ("SC", "FC", "//")]):
            vals, errs = [], []
            for est in ESTIMATORS:
                m, s = m_sd(recon_est(r, parc, est, src, tgt))
                vals.append(m); errs.append(s)
            off = (pi * 2 + di - 1.5) * w
            col = C["FC"] if src == "FC" else C["SC"]
            alpha = 1.0 if pi == 0 else 0.55
            axA.bar(x + off, vals, w, yerr=errs, capsize=2, color=col, alpha=alpha,
                    hatch=hatch, edgecolor="white",
                    label=f"{'FC→SC' if di==0 else 'SC→FC'} · {PSHORT[parc]}")
    axA.set_xticks(x); axA.set_xticklabels([EST_LABEL[e] for e in ESTIMATORS], fontsize=8)
    axA.set_ylabel("demeaned pearson"); axA.set_title("A · FC→SC vs SC→FC by estimator × parcellation")
    axA.legend(fontsize=6.5, ncol=2)
    panel_tag(axA, "A")
    # B: ratio per estimator x parc
    axB = fig.add_subplot(gs[0, 1])
    for pi, parc in enumerate(PARCS):
        ratios = []
        for est in ESTIMATORS:
            a = m_sd(recon_est(r, parc, est, "FC", "SC"))[0]
            b = m_sd(recon_est(r, parc, est, "SC", "FC"))[0]
            ratios.append(a / b)
        axB.plot(range(len(ESTIMATORS)), ratios, "-o", color=C["SC"] if pi else C["FC"],
                 label=PARC_LABEL[parc], ms=7)
        for xi, rr in enumerate(ratios):
            axB.annotate(f"{rr:.2f}×", (xi, rr), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8, fontweight="bold")
    axB.axhline(1, color=C["bad"], ls="--", lw=1.2)
    axB.set_xticks(range(len(ESTIMATORS))); axB.set_xticklabels([EST_LABEL[e] for e in ESTIMATORS], fontsize=8)
    axB.set_ylabel("FC→SC / SC→FC ratio"); axB.set_ylim(0.9, None)
    axB.set_title("B · Asymmetry ratio is estimator-robust"); axB.legend(fontsize=8)
    panel_tag(axB, "B")
    # C: reduction-axis robustness (Glasser)
    axC = fig.add_subplot(gs[1, 0])
    summ = pd.read_csv(SANITY / "preprocessing_check" / "reduction_axis_summary.csv")
    syn = pd.read_csv(SANITY / "preprocessing_check" / "reduction_axis_synthesis.csv")
    syn["k"] = syn.apply(lambda x: x["method"] if isinstance(x["jl_variant"], float)
                         else f"{x['method']}:{x['jl_variant']}", axis=1)
    nm = {"FULL_PLS": "full PLS", "PCA_PLS_PCA": "PCA→PLS",
          "JL_PLS_PCA:gaussian_dense": "JL gauss", "JL_PLS_PCA:sparse_auto": "JL sp-auto",
          "JL_PLS_PCA:sparse_third": "JL sp-1/3"}
    summ = summ.sort_values("median_ratio")
    for i, (_, row) in enumerate(summ.iterrows()):
        k = row["method"] if pd.isna(row["jl_variant"]) else f"{row['method']}:{row['jl_variant']}"
        pts = syn[syn.k == k]["ratio"].values
        axC.scatter(pts, np.full_like(pts, i) + np.random.default_rng(i).uniform(-.12, .12, len(pts)),
                    s=18, color=C["FC"], alpha=0.5, edgecolor="white", zorder=3)
        axC.scatter([row["median_ratio"]], [i], marker="|", s=200, color=C["ink"], lw=2.5, zorder=4)
        axC.text(2.32, i, f"{row['median_ratio']:.2f}× p={row['wilcoxon_p_vs_1']:.0e}",
                 va="center", fontsize=6.5, color="#555")
    axC.axvline(1, color=C["bad"], ls="--", lw=1.2)
    axC.set_yticks(range(len(summ))); axC.set_yticklabels(
        [nm.get(row["method"] if pd.isna(row["jl_variant"]) else f"{row['method']}:{row['jl_variant']}",
                "") for _, row in summ.iterrows()], fontsize=7.5)
    axC.set_xlim(0.9, 3.1); axC.set_xlabel("ratio (per seed)")
    axC.set_title("C · Reduction-axis robustness (Glasser)")
    panel_tag(axC, "C")
    # D: per-seed ratio both parc (pca_pls)
    axD = fig.add_subplot(gs[1, 1])
    for pi, parc in enumerate(PARCS):
        sub = r[(r.parcellation == parc) & (r.estimator == "pca_pls") & (~r.is_block)]
        a = sub[(sub.source == "FC") & (sub.target == "SC")].set_index("seed")["demeaned_pearson"]
        b = sub[(sub.source == "SC") & (sub.target == "FC")].set_index("seed")["demeaned_pearson"]
        j = a.index.intersection(b.index)
        ratios = (a.loc[j] / b.loc[j]).values
        jit = np.random.default_rng(pi).uniform(-0.08, 0.08, len(ratios))
        axD.scatter(np.full_like(ratios, pi) + jit, ratios, s=36, color=C["FC"],
                    alpha=0.75, edgecolor="white", zorder=3)
        axD.hlines(np.median(ratios), pi - 0.2, pi + 0.2, color=C["ink"], lw=2.2, zorder=4)
    axD.axhline(1, color=C["bad"], ls="--", lw=1.2)
    axD.set_xticks(range(len(PARCS))); axD.set_xticklabels([PARC_LABEL[p] for p in PARCS])
    axD.set_ylabel("FC→SC / SC→FC ratio"); axD.set_title("D · Every seed > 1 (PCA→PLS)")
    panel_tag(axD, "D")
    fig.suptitle("F1 · Cross-modal prediction is directional: FC→SC > SC→FC "
                 "(both parcellations, all estimators, all reductions)",
                 fontsize=12, fontweight="bold", y=0.975)
    savefig(fig, "F1_directional_asymmetry")


# ============================================================ F2
def F2():
    fig = plt.figure(figsize=(13, 7.6))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.24, left=0.07, right=0.98,
                          top=0.9, bottom=0.08)
    srcs = [("bv", C["bv"]), ("demo", C["demo"]), ("bv+demo", C["base"])]
    for pi, parc in enumerate(PARCS):
        ax = fig.add_subplot(gs[0, pi])
        x = np.arange(2); w = 0.26
        for si, (src, col) in enumerate(srcs):
            vals = [m_sd(recon_est(r, parc, "pca_pls", src, t))[0] for t in ["SC", "FC"]]
            errs = [m_sd(recon_est(r, parc, "pca_pls", src, t))[1] for t in ["SC", "FC"]]
            ax.bar(x + (si - 1) * w, vals, w, yerr=errs, capsize=2, color=col,
                   edgecolor="white", label=src)
        ax.set_xticks(x); ax.set_xticklabels(["→ SC", "→ FC"])
        ax.set_ylabel("demeaned pearson"); ax.set_ylim(0, 0.21)
        ax.set_title(f"{'A' if pi==0 else 'B'} · {PARC_LABEL[parc]}")
        if pi == 0:
            ax.legend(fontsize=8, title="input")
        panel_tag(ax, "A" if pi == 0 else "B")
    # C: dissociation index (bv-demo) for ->SC vs ->FC across estimators
    axC = fig.add_subplot(gs[1, 0])
    x = np.arange(len(ESTIMATORS)); w = 0.18
    for pi, parc in enumerate(PARCS):
        for ti, tgt in enumerate(["SC", "FC"]):
            di = []
            for est in ESTIMATORS:
                bv = m_sd(recon_est(r, parc, est, "bv", tgt))[0]
                de = m_sd(recon_est(r, parc, est, "demo", tgt))[0]
                di.append(bv - de)
            off = (pi * 2 + ti - 1.5) * w
            col = C["SC"] if tgt == "SC" else C["FC"]
            axC.bar(x + off, di, w, color=col, alpha=1.0 if pi == 0 else 0.55,
                    edgecolor="white",
                    label=f"→{tgt} · {PSHORT[parc]}")
    axC.axhline(0, color=C["ink"], lw=1)
    axC.set_xticks(x); axC.set_xticklabels([EST_LABEL[e] for e in ESTIMATORS], fontsize=8)
    axC.set_ylabel("bv − demo  (demeaned r)")
    axC.set_title("C · Dissociation index: + means anatomy wins, − means demographics win")
    axC.legend(fontsize=6.5, ncol=2)
    panel_tag(axC, "C")
    # D: heatmap input x target x parc
    axD = fig.add_subplot(gs[1, 1])
    inputs = ["bv", "demo", "bv+demo"]
    M = np.zeros((len(inputs), 4))
    cols = []
    ci = 0
    for parc in PARCS:
        for tgt in ["SC", "FC"]:
            for ri, src in enumerate(inputs):
                M[ri, ci] = m_sd(recon_est(r, parc, "pca_pls", src, tgt))[0]
            cols.append(f"{PSHORT[parc]}→{tgt}")
            ci += 1
    im = axD.imshow(M, cmap="viridis", aspect="auto")
    axD.set_xticks(range(4)); axD.set_xticklabels(cols, rotation=30, ha="right", fontsize=7.5)
    axD.set_yticks(range(len(inputs))); axD.set_yticklabels(inputs)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            axD.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", fontsize=7,
                     color="white" if M[i, j] < 0.12 else "black")
    fig.colorbar(im, ax=axD, fraction=0.046, pad=0.02)
    axD.set_title("D · Subject-info → connectome (demeaned r)"); axD.grid(False)
    panel_tag(axD, "D")
    fig.suptitle("F2 · Double dissociation: anatomy predicts SC, demographics predict FC "
                 "(both parcellations, estimator-robust)", fontsize=12, fontweight="bold", y=0.975)
    savefig(fig, "F2_double_dissociation")


# ============================================================ F3
def F3():
    fig = plt.figure(figsize=(13, 4.8))
    gs = fig.add_gridspec(1, 3, wspace=0.3, left=0.06, right=0.985, top=0.84, bottom=0.14)
    # A: imputed connectomes ARE measurable (recon quality)
    axA = fig.add_subplot(gs[0, 0])
    x = np.arange(2); w = 0.36
    for pi, parc in enumerate(PARCS):
        vals = [m_sd(recon_est(r, parc, "pca_pls", "FC", "SC"))[0],
                m_sd(recon_est(r, parc, "pca_pls", "SC", "FC"))[0]]
        errs = [m_sd(recon_est(r, parc, "pca_pls", "FC", "SC"))[1],
                m_sd(recon_est(r, parc, "pca_pls", "SC", "FC"))[1]]
        axA.bar(x + (pi - 0.5) * w, vals, w, yerr=errs, capsize=2.5,
                color=[C["SC"], C["FC"]], alpha=1.0 if pi == 0 else 0.55,
                edgecolor="white")
    axA.set_xticks(x); axA.set_xticklabels(["pred SC\n(FC→SC)", "pred FC\n(SC→FC)"])
    axA.set_ylabel("reconstruction demeaned r")
    axA.set_title("A · Imputed connectomes are measurable")
    axA.text(0.5, 0.9, "solid=Glasser, faded=4S456", transform=axA.transAxes,
             fontsize=7, ha="center", color="#666")
    panel_tag(axA, "A")
    # B: but downstream utility is asymmetric (lift)
    axB = fig.add_subplot(gs[0, 1])
    for pi, parc in enumerate(PARCS):
        vals = [m_sd(down_est(d, parc, "bayesian_ridge", "pred_SC", "CogCryst", "lift_over_bvdemo"))[0],
                m_sd(down_est(d, parc, "bayesian_ridge", "pred_FC", "CogCryst", "lift_over_bvdemo"))[0]]
        axB.bar(x + (pi - 0.5) * w, vals, w, color=[C["SC"], C["pred"]],
                alpha=1.0 if pi == 0 else 0.55, edgecolor="white")
    axB.axhline(0, color=C["ink"], lw=1)
    axB.set_xticks(x); axB.set_xticklabels(["pred SC", "pred FC"])
    axB.set_ylabel("CogCryst lift over bv+demo")
    axB.set_title("B · …but their utility is asymmetric")
    panel_tag(axB, "B")
    # C: recon quality vs utility disconnect
    axC = fig.add_subplot(gs[0, 2])
    for parc in PARCS:
        for inp, src, tgt, col in [("pred_SC", "FC", "SC", C["SC"]),
                                   ("pred_FC", "SC", "FC", C["pred"])]:
            rq = m_sd(recon_est(r, parc, "pca_pls", src, tgt))[0]
            ut = m_sd(down_est(d, parc, "bayesian_ridge", inp, "CogCryst", "lift_over_bvdemo"))[0]
            mk = "o" if parc == "Glasser" else "s"
            axC.scatter([rq], [ut], s=90, color=col, marker=mk, edgecolor="white", zorder=3)
            axC.annotate(f"{inp.replace('pred_','pred ')}\n{PSHORT[parc]}", (rq, ut),
                         textcoords="offset points", xytext=(6, 4), fontsize=6.5)
    axC.axhline(0, color=C["ink"], lw=1)
    axC.set_xlabel("reconstruction quality (demeaned r)")
    axC.set_ylabel("cognition utility (lift)")
    axC.set_title("C · Measurable ≠ useful")
    panel_tag(axC, "C")
    fig.suptitle("F3 · Imputed connectomes are reconstructable, but downstream utility is "
                 "asymmetric (interpreted through F5)", fontsize=11.5, fontweight="bold", y=0.97)
    savefig(fig, "F3_imputation_usable_asymmetric")


# ============================================================ F4
def F4():
    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(1, 3, wspace=0.3, left=0.06, right=0.985, top=0.84, bottom=0.13)
    # A: obs_FC vs bv+demo, 3 targets, both parc (BR)
    axA = fig.add_subplot(gs[0, 0])
    x = np.arange(len(COGS)); w = 0.2
    for pi, parc in enumerate(PARCS):
        base = [m_sd(down_est(d, parc, "bayesian_ridge", "bv+demo", t, "pearson"))[0] for t in COGS]
        fc = [m_sd(down_est(d, parc, "bayesian_ridge", "obs_FC", t, "pearson"))[0] for t in COGS]
        axA.bar(x + (pi*2-1.5)*w, base, w, color=C["base"], alpha=1 if pi == 0 else 0.55,
                edgecolor="white", label=f"bv+demo·{PSHORT[parc]}")
        axA.bar(x + (pi*2-0.5)*w, fc, w, color=C["FC"], alpha=1 if pi == 0 else 0.55,
                edgecolor="white", label=f"obs FC·{PSHORT[parc]}")
    axA.set_xticks(x); axA.set_xticklabels([c[3:] for c in COGS])
    axA.set_ylabel("cognition pearson"); axA.set_title("A · obs FC vs bv+demo (BayesianRidge)")
    axA.legend(fontsize=6.5, ncol=2); panel_tag(axA, "A")
    # B: residualized pearson (after bv+demo)
    axB = fig.add_subplot(gs[0, 1])
    for pi, parc in enumerate(PARCS):
        rp = [m_sd(down_est(d, parc, "bayesian_ridge", "obs_FC", t, "residualized_pearson"))[0] for t in COGS]
        axB.bar(x + (pi - 0.5)*0.36, rp, 0.36, color=C["FC"], alpha=1 if pi == 0 else 0.55,
                edgecolor="white", label=PARC_LABEL[parc])
    axB.axhline(0, color=C["ink"], lw=1)
    axB.set_xticks(x); axB.set_xticklabels([c[3:] for c in COGS])
    axB.set_ylabel("residualized pearson")
    axB.set_title("B · obs FC signal AFTER removing bv+demo"); axB.legend(fontsize=7.5)
    panel_tag(axB, "B")
    # C: across estimators obs_FC CogCryst lift
    axC = fig.add_subplot(gs[0, 2])
    x2 = np.arange(len(ESTIMATORS))
    for pi, parc in enumerate(PARCS):
        vals = [m_sd(down_est(d, parc, e, "obs_FC", "CogCryst", "lift_over_bvdemo"))[0] for e in ESTIMATORS]
        axC.bar(x2 + (pi - 0.5)*0.36, vals, 0.36, color=C["FC"], alpha=1 if pi == 0 else 0.55,
                edgecolor="white", label=PARC_LABEL[parc])
    axC.axhline(0, color=C["ink"], lw=1)
    axC.set_xticks(x2); axC.set_xticklabels([EST_LABEL[e] for e in ESTIMATORS], fontsize=7.5)
    axC.set_ylabel("CogCryst lift"); axC.set_title("C · obs FC lift across estimators")
    axC.legend(fontsize=7.5); panel_tag(axC, "C")
    fig.suptitle("F4 · Observed FC carries a crystallized-cognition signal (narrow, "
                 "FDR-surviving cell); broader lift is multiplicity-fragile (F11)",
                 fontsize=10.5, fontweight="bold", y=0.97)
    savefig(fig, "F4_obs_fc_cognition")


# ============================================================ F5
def F5():
    fig = plt.figure(figsize=(13, 7.8))
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.22, left=0.09, right=0.97,
                          top=0.9, bottom=0.08, height_ratios=[1.1, 1])
    inputs = ["obs_FC", "obs_SC", "obs_FC+obs_SC", "pred_SC", "pred_FC",
              "pred_SC+bv+demo", "pred_FC+bv+demo"]
    ylab = ["obs FC", "obs SC", "obs FC+SC", "pred SC", "pred FC",
            "pred SC+bv+demo", "pred FC+bv+demo"]
    for pi, parc in enumerate(PARCS):
        ax = fig.add_subplot(gs[0, pi])
        M = np.array([[m_sd(down_est(d, parc, "bayesian_ridge", inp, t, "lift_over_bvdemo"))[0]
                       for t in COGS] for inp in inputs])
        im = ax.imshow(M, cmap="RdBu_r", vmin=-0.17, vmax=0.17, aspect="auto")
        ax.set_xticks(range(3)); ax.set_xticklabels([c[3:] for c in COGS], fontsize=8)
        ax.set_yticks(range(len(inputs))); ax.set_yticklabels(ylab, fontsize=7.5)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(M[i, j]) > 0.09 else "#333")
        ax.set_title(f"{'A' if pi==0 else 'B'} · lift over bv+demo — {PARC_LABEL[parc]}")
        ax.grid(False); panel_tag(ax, "A" if pi == 0 else "B")
    fig.colorbar(im, ax=fig.axes[:2], fraction=0.025, pad=0.02, label="Δ pearson")
    # C: obs_SC vs pred_FC harm across estimators (CogCryst)
    axC = fig.add_subplot(gs[1, 0])
    x = np.arange(len(ESTIMATORS)); w = 0.18
    for pi, parc in enumerate(PARCS):
        for ii, (inp, col) in enumerate([("obs_SC", C["SC"]), ("pred_FC", C["pred"])]):
            vals = [m_sd(down_est(d, parc, e, inp, "CogCryst", "lift_over_bvdemo"))[0] for e in ESTIMATORS]
            axC.bar(x + (pi*2 + ii - 1.5)*w, vals, w, color=col, alpha=1 if pi == 0 else 0.55,
                    edgecolor="white", label=f"{inp}·{PSHORT[parc]}")
    axC.axhline(0, color=C["ink"], lw=1)
    axC.set_xticks(x); axC.set_xticklabels([EST_LABEL[e] for e in ESTIMATORS], fontsize=7.5)
    axC.set_ylabel("CogCryst lift"); axC.set_title("C · SC underperforms & pred FC harms (all estimators)")
    axC.legend(fontsize=6.5, ncol=2); panel_tag(axC, "C")
    # D: pred_SC vs pred_FC contrast (the imputation asymmetry) all targets, both parc
    axD = fig.add_subplot(gs[1, 1])
    x = np.arange(len(COGS)); w = 0.2
    for pi, parc in enumerate(PARCS):
        ps = [m_sd(down_est(d, parc, "bayesian_ridge", "pred_SC", t, "lift_over_bvdemo"))[0] for t in COGS]
        pf = [m_sd(down_est(d, parc, "bayesian_ridge", "pred_FC", t, "lift_over_bvdemo"))[0] for t in COGS]
        axD.bar(x + (pi*2-1.5)*w, ps, w, color=C["SC"], alpha=1 if pi == 0 else 0.55,
                edgecolor="white", label=f"pred SC·{PSHORT[parc]}")
        axD.bar(x + (pi*2-0.5)*w, pf, w, color=C["pred"], alpha=1 if pi == 0 else 0.55,
                edgecolor="white", label=f"pred FC·{PSHORT[parc]}")
    axD.axhline(0, color=C["ink"], lw=1)
    axD.set_xticks(x); axD.set_xticklabels([c[3:] for c in COGS])
    axD.set_ylabel("lift over bv+demo")
    axD.set_title("D · Imputation asymmetry: pred FC consistently harmful")
    axD.legend(fontsize=6.5, ncol=2); panel_tag(axD, "D")
    fig.suptitle("F5 · SC underperforms bv+demo, imputation does not transfer utility, "
                 "and pred FC is harmful (both parcellations, all estimators)",
                 fontsize=11.5, fontweight="bold", y=0.975)
    savefig(fig, "F5_utility_wall")


# ============================================================ F6
def F6():
    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(1, 3, wspace=0.3, left=0.06, right=0.985, top=0.84, bottom=0.16)
    variants = ["bvdemo_to_SC", "pred_SC_raw", "pred_SC_resid_bvdemo", "obs_SC", "obs_FC"]
    vlab = ["bv+demo", "pred SC\nraw", "pred SC\nresid", "obs SC", "obs FC"]
    vcol = [C["base"], "#C39BD3", C["pred"], C["SC"], C["FC"]]
    # A: sibling AUC both parc
    axA = fig.add_subplot(gs[0, 0])
    x = np.arange(len(variants)); w = 0.38
    for pi, parc in enumerate(PARCS):
        sub = fam[(fam.parcellation == parc) & (fam.relation == "sibling")].set_index("variant")
        vals = [sub.loc[v, "auc"] for v in variants]
        lo = [sub.loc[v, "auc"] - sub.loc[v, "auc_lo"] for v in variants]
        hi = [sub.loc[v, "auc_hi"] - sub.loc[v, "auc"] for v in variants]
        axA.bar(x + (pi - 0.5)*w, vals, w, yerr=[lo, hi], capsize=2,
                color=vcol, alpha=1 if pi == 0 else 0.6,
                hatch="" if pi == 0 else "///", edgecolor="white")
    axA.axhline(0.5, color=C["bad"], ls="--", lw=1.2)
    axA.set_xticks(x); axA.set_xticklabels(vlab, fontsize=7.5)
    axA.set_ylabel("sibling AUC"); axA.set_ylim(0.45, 0.95)
    axA.set_title("A · Family (sibling) signal by predictor")
    from matplotlib.patches import Patch
    axA.legend(handles=[Patch(facecolor="#bbb", label="Glasser"),
                        Patch(facecolor="#bbb", hatch="///", label="4S456")], fontsize=7)
    panel_tag(axA, "A")
    # B: MZ/DZ/sib gradient for pred_SC_resid both parc
    axB = fig.add_subplot(gs[0, 1])
    rels = ["MZ", "DZ", "sibling"]
    for pi, parc in enumerate(PARCS):
        sub = fam[(fam.parcellation == parc) & (fam.variant == "pred_SC_resid_bvdemo")].set_index("relation")
        vals = [sub.loc[rl, "auc"] for rl in rels]
        axB.plot(range(3), vals, "-o", color=C["pred"] if pi == 0 else "#C39BD3",
                 ms=7, label=PARC_LABEL[parc])
    axB.axhline(0.5, color=C["bad"], ls="--", lw=1.2)
    axB.set_xticks(range(3)); axB.set_xticklabels(["MZ", "DZ", "sib"])
    axB.set_ylabel("AUC"); axB.set_ylim(0.45, 1.0)
    axB.set_title("B · pred SC (resid): relatedness gradient"); axB.legend(fontsize=7.5)
    panel_tag(axB, "B")
    # C: lift over baseline (resid - bvdemo) both parc
    axC = fig.add_subplot(gs[0, 2])
    for pi, parc in enumerate(PARCS):
        sub = fam[(fam.parcellation == parc) & (fam.relation == "sibling")].set_index("variant")
        gain = sub.loc["pred_SC_resid_bvdemo", "auc"] - sub.loc["bvdemo_to_SC", "auc"]
        axC.bar([pi], [gain], 0.5, color=C["pred"], edgecolor="white")
        axC.text(pi, gain + 0.005, f"+{gain:.2f}", ha="center", fontsize=9, fontweight="bold")
    axC.set_xticks(range(len(PARCS))); axC.set_xticklabels([PARC_LABEL[p] for p in PARCS])
    axC.set_ylabel("sibling AUC gain over bv+demo")
    axC.set_title("C · Heritable signal beyond baseline"); panel_tag(axC, "C")
    fig.suptitle("F6 · Predicted connectomes carry heritable family signal beyond bv+demo "
                 "(both parcellations)", fontsize=11.5, fontweight="bold", y=0.97)
    savefig(fig, "F6_family_signal")


# ============================================================ F7
def F7():
    fig = plt.figure(figsize=(11.5, 5))
    gs = fig.add_gridspec(1, 2, wspace=0.28, left=0.08, right=0.97, top=0.84, bottom=0.13)
    rels = ["MZ", "DZ", "sibling"]
    for pi, parc in enumerate(PARCS):
        ax = fig.add_subplot(gs[0, pi])
        for variant, col, lab in [("pred_SC_resid_bvdemo", C["pred"], "residual (identification)"),
                                  ("combined_pred_SC", C["accent"], "combined (reconstruction)")]:
            sub = fam[(fam.parcellation == parc) & (fam.variant == variant)].set_index("relation")
            vals = [sub.loc[rl, "auc"] for rl in rels]
            lo = [sub.loc[rl, "auc"] - sub.loc[rl, "auc_lo"] for rl in rels]
            hi = [sub.loc[rl, "auc_hi"] - sub.loc[rl, "auc"] for rl in rels]
            ax.errorbar(range(3), vals, yerr=[lo, hi], marker="o", ms=8, color=col,
                        lw=2, capsize=3, label=lab)
        ax.axhline(0.5, color=C["bad"], ls="--", lw=1.2)
        ax.set_xticks(range(3)); ax.set_xticklabels(["MZ", "DZ", "sib"])
        ax.set_ylabel("separation AUC"); ax.set_ylim(0.45, 1.0)
        ax.set_title(f"{'A' if pi==0 else 'B'} · {PARC_LABEL[parc]}")
        if pi == 0:
            ax.legend(fontsize=7.5, loc="upper right")
        ax.annotate("combined collapses\nto chance for siblings", (2, 0.51), (1.1, 0.62),
                    fontsize=7.5, color=C["accent"], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=C["accent"]))
        panel_tag(ax, "A" if pi == 0 else "B")
    fig.suptitle("F7 · Reconstruction and identification objectives trade off: the "
                 "reconstruction-optimized predictor loses sibling signal (both parcellations)",
                 fontsize=11, fontweight="bold", y=0.97)
    savefig(fig, "F7_objective_tradeoff")


# ============================================================ F8
def F8():
    pp = pd.read_csv(FAM / "f8_per_pc.csv")
    g = pp.groupby(["parcellation", "pc"]).median(numeric_only=True).reset_index()
    st = pd.read_csv(FAM / "f8_stability.csv")
    enr = pd.read_csv(FAM / "f8_pc3_enrichment_agg.csv")
    sel = {"Glasser": 3, "4S456Parcels": 4}
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.26, left=0.07, right=0.98,
                          top=0.9, bottom=0.08)
    # A: FC->PC R2 and sibling AUC per PC, both parc
    axA = fig.add_subplot(gs[0, 0])
    for parc, mk in zip(PARCS, ["o", "s"]):
        s = g[g.parcellation == parc]
        axA.plot(s.pc, s.FC_to_PC_R2, "-"+mk, color=C["SC"] if parc == "Glasser" else "#85C1E9",
                 label=f"{PSHORT[parc]} FC→PC R²", ms=5)
        axA.plot(s.pc, s.AUC_sibling, "--"+mk, color=C["pred"] if parc == "Glasser" else "#C39BD3",
                 label=f"{PSHORT[parc]} sib AUC", ms=5)
        axA.axvline(sel[parc], color=C["accent"], lw=8, alpha=0.15, zorder=0)
    axA.set_xlabel("principal component"); axA.set_xticks(range(1, 11))
    axA.set_ylabel("value"); axA.set_title("A · FC-predictability & family AUC per PC")
    axA.legend(fontsize=6.5); panel_tag(axA, "A")
    # B: confound R2 per PC (PC1 confound)
    axB = fig.add_subplot(gs[0, 1])
    for parc, mk in zip(PARCS, ["o", "s"]):
        s = g[g.parcellation == parc]
        axB.plot(s.pc, s.confound_R2_test, "-"+mk, color=C["bad"] if parc == "Glasser" else "#E59866",
                 label=PARC_LABEL[parc], ms=5)
        axB.axvline(sel[parc], color=C["accent"], lw=8, alpha=0.15, zorder=0)
    axB.set_xlabel("principal component"); axB.set_xticks(range(1, 11))
    axB.set_ylabel("confound R² (sex/volume)")
    axB.set_title("B · PC1 is a confound; selected mode is not"); axB.legend(fontsize=7.5)
    panel_tag(axB, "B")
    # C: cross-seed stability (median |cos|) per anchor PC, both parc
    axC = fig.add_subplot(gs[1, 0])
    for parc, mk in zip(PARCS, ["o", "s"]):
        s = st[st.parcellation == parc]
        axC.plot(s.anchor_pc, s.median_abs_cos, "-"+mk,
                 color=C["SC"] if parc == "Glasser" else "#85C1E9", label=PARC_LABEL[parc], ms=5)
        axC.axvline(sel[parc], color=C["accent"], lw=8, alpha=0.15, zorder=0)
    axC.axhline(0.8, color="#888", ls=":", lw=1)
    axC.set_xlabel("anchor PC"); axC.set_xticks(range(1, 11))
    axC.set_ylabel("cross-seed median |cos|")
    axC.set_title("C · Component stability across seeds"); axC.legend(fontsize=7.5)
    panel_tag(axC, "C")
    # D: enrichment top pairs both parc
    axD = fig.add_subplot(gs[1, 1])
    top = enr[enr.parcellation == "Glasser"].nlargest(6, "median_enrichment")
    axD.barh(range(len(top)), top.median_enrichment, color=C["SC"], edgecolor="white")
    axD.set_yticks(range(len(top))); axD.set_yticklabels(top.net_pair, fontsize=7)
    axD.invert_yaxis(); axD.axvline(1, color=C["ink"], ls=":", lw=1)
    axD.set_xlabel("median enrichment (× chance)")
    axD.set_title("D · Selected-mode network localization (Glasser)")
    panel_tag(axD, "D")
    fig.suptitle("F8 · Exploratory FC-predictable low-variance SC mode (component index NOT "
                 "stable: 3rd in Glasser, 4th in 4S456); PC1 is a sex/volume confound",
                 fontsize=11, fontweight="bold", y=0.975)
    savefig(fig, "F8_pc_mechanism")


# ============================================================ F9
def F9():
    ts = pd.read_csv(TRACT / "tractography_synthesis.csv").set_index("row")["value"]
    e5 = pd.read_csv(TRACT / "e5_downstream_summary.csv")
    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(1, 3, wspace=0.34, left=0.07, right=0.985, top=0.84, bottom=0.2)
    # A: E1 recon by rep
    axA = fig.add_subplot(gs[0, 0])
    feats = [("SC", "E1: SC -> FC median dp"), ("SC+r2t", "E1: SC_r2t -> FC median dp"),
             ("kitchen", "E1: kitchen_sink -> FC median dp"), ("r2t", "E1: r2t -> FC median dp"),
             ("r2t corr", "E1: r2t_corr -> FC median dp")]
    vals = [float(ts[k]) for _, k in feats]
    axA.bar(range(len(feats)), vals, color=[C["SC"], "#5DADE2", "#85C1E9", C["bv"], "#B2BABB"],
            edgecolor="white")
    axA.axhline(float(ts["E1: SC -> FC median dp"]), color=C["SC"], ls="--", lw=1)
    axA.set_xticks(range(len(feats))); axA.set_xticklabels([f for f, _ in feats], rotation=30,
                                                           ha="right", fontsize=7.5)
    axA.set_ylabel("→FC demeaned r"); axA.set_title("A · Predicting FC (Glasser)")
    panel_tag(axA, "A")
    # B: E5 downstream by rep (CogCryst)
    axB = fig.add_subplot(gs[0, 1])
    order = ["FC", "bv+demo", "SC", "r2t", "SC_r2t", "r2t_corr", "r2t->synthFC"]
    sub = e5[e5.target == "CogCrystalComp_Unadj"].set_index("rep")
    vals = [sub.loc[o, "pearson_raw"] if o in sub.index else np.nan for o in order]
    floor = sub.loc["bv+demo", "pearson_raw"]
    cols = [C["FC"] if o == "FC" else C["base"] if o == "bv+demo" else C["bv"] for o in order]
    axB.bar(range(len(order)), vals, color=cols, edgecolor="white")
    axB.axhline(floor, color=C["base"], ls="--", lw=1.2)
    axB.set_xticks(range(len(order))); axB.set_xticklabels(order, rotation=35, ha="right", fontsize=7)
    axB.set_ylabel("CogCryst pearson"); axB.set_title("B · Only FC clears the floor (Glasser)")
    panel_tag(axB, "B")
    # C: asymmetry across reps
    axC = fig.add_subplot(gs[0, 2])
    rrs = [("FC↔SC", "E2: FC<->SC median ratio"), ("FC↔r2t", "E2: FC<->r2t median ratio"),
           ("FC↔r2t_corr", "E2: FC<->r2t_corr median ratio")]
    vals = [float(ts[k]) for _, k in rrs]
    axC.bar(range(len(rrs)), vals, color=C["FC"], edgecolor="white")
    axC.axhline(1, color=C["bad"], ls="--", lw=1.2)
    axC.set_xticks(range(len(rrs))); axC.set_xticklabels([n for n, _ in rrs], fontsize=7.5)
    axC.set_ylabel("asymmetry ratio"); axC.set_title("C · Asymmetry holds across reps")
    for i, v in enumerate(vals):
        axC.text(i, v + 0.03, f"{v:.2f}×", ha="center", fontsize=8, fontweight="bold")
    panel_tag(axC, "C")
    fig.suptitle("F9 · Richer tractography (r2t) does not help: worse FC prediction, no "
                 "cognition signal, asymmetry preserved (Glasser, exploratory)",
                 fontsize=11, fontweight="bold", y=0.97)
    savefig(fig, "F9_tractography")


# ============================================================ F10
def F10():
    sc = pd.read_csv(NLIN / "n6_scaling_summary.csv")
    NOISE = SANITY / "noise_sanity_check" / "outputs"
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.26, left=0.07, right=0.98,
                          top=0.9, bottom=0.08)
    # A: nonlinear gap vs n
    axA = fig.add_subplot(gs[0, 0])
    for task, col in [("reconstruction", C["SC"]), ("cognition", C["FC"])]:
        s = sc[(sc.task == task) & (sc.n_seeds >= 5)].sort_values("n_sub")
        axA.plot(s.n_sub, s.median_gap, "-o", color=col, label=task, ms=5)
        axA.fill_between(s.n_sub, s.gap_min, s.gap_max, color=col, alpha=0.12)
    axA.axhline(0, color=C["ink"], lw=1)
    axA.set_xlabel("training subjects"); axA.set_ylabel("nonlinear − linear gap")
    axA.set_title("A · No growing nonlinear gap with n"); axA.legend(fontsize=8)
    panel_tag(axA, "A")
    # B: linear vs nonlinear scores vs n
    axB = fig.add_subplot(gs[0, 1])
    for task, col in [("reconstruction", C["SC"]), ("cognition", C["FC"])]:
        s = sc[(sc.task == task) & (sc.n_seeds >= 5)].sort_values("n_sub")
        axB.plot(s.n_sub, s.median_linear, "-o", color=col, ms=4, label=f"{task} linear")
        axB.plot(s.n_sub, s.median_final, "--s", color=col, ms=4, alpha=0.6, label=f"{task} nonlinear")
    axB.set_xlabel("training subjects"); axB.set_ylabel("score")
    axB.set_title("B · Linear improves with n; nonlinear adds nothing")
    axB.legend(fontsize=6.5); panel_tag(axB, "B")
    # C: KR hyperparameter flatness (recon FC->SC, 9 variants spread per seed)
    axC = fig.add_subplot(gs[1, 0])
    for pi, parc in enumerate(PARCS):
        kr = r[(r.parcellation == parc) & (r.estimator == "kernel_ridge") &
               (r.source == "FC") & (r.target == "SC") & (~r.is_block)]
        spread = kr.groupby("seed")["demeaned_pearson"].agg(lambda v: v.max() - v.min())
        axC.scatter(np.full(len(spread), pi) + np.random.default_rng(pi).uniform(-.08, .08, len(spread)),
                    spread.values, s=30, color=C["good"], alpha=0.7, edgecolor="white")
        axC.hlines(spread.median(), pi - 0.2, pi + 0.2, color=C["ink"], lw=2)
    axC.set_xticks(range(len(PARCS))); axC.set_xticklabels([PARC_LABEL[p] for p in PARCS])
    axC.set_ylabel("KR 9-variant spread (demeaned r)")
    axC.set_title("C · KernelRidge HP sweep is nearly degenerate (FC→SC)")
    panel_tag(axC, "C")
    # D: FC reliability ceiling vs SC->FC achieved (both parc)
    axD = fig.add_subplot(gs[1, 1])
    a = pd.read_csv(NOISE / "a_reliability_ceiling.csv")
    e = pd.read_csv(NOISE / "e_crossmodal_disattenuation.csv")
    x = np.arange(len(PARCS)); w = 0.38
    ceil = [a[(a.parc == p) & (a.comparison == "between_session")]["demeaned_pearson"].iloc[0] for p in PARCS]
    ach = e[(e.source == "SC->FC") & (e.metric == "demeaned_pearson")]["achieved"].iloc[0]
    axD.bar(x - w/2, ceil, w, color=C["fc_lt"], edgecolor="white", label="FC reliability ceiling")
    axD.bar(x + w/2, [ach]*len(PARCS), w, color=C["SC"], edgecolor="white", label="SC→FC achieved")
    for i, c in enumerate(ceil):
        axD.text(i, c + 0.01, f"{ach/c*100:.0f}% of ceiling", ha="center", fontsize=7.5)
    axD.set_xticks(x); axD.set_xticklabels([PARC_LABEL[p] for p in PARCS])
    axD.set_ylabel("demeaned pearson"); axD.set_title("D · SC→FC reaches only ~17% of FC ceiling")
    axD.legend(fontsize=7.5); panel_tag(axD, "D")
    fig.suptitle("F10 · The ceiling is structural: nonlinear models, more data, KR tuning, "
                 "and FC denoising do not break it (Glasser scaling + both-parc reliability)",
                 fontsize=11, fontweight="bold", y=0.975)
    savefig(fig, "F10_structural_ceiling")


if __name__ == "__main__":
    for fn in [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10]:
        fn()
    print("done F1-F10")
