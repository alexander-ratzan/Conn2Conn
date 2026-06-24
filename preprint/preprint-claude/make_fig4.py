#!/usr/bin/env python3
"""Fig 4 - The ceiling is structural in this regime (closed escape routes).

A: SC->FC achieves only ~17% of the FC between-session reliability ceiling.
B: reduction-axis robustness - FC->SC/SC->FC ratio > 1 for every reduction method.
C: sample-size scaling - nonlinear minus linear gap stays ~0 as n grows.
D: richer structure (tractography) does not beat streamline-count SC at predicting FC.
E: per-subject SC->FC quality is uncorrelated with that subject's FC reliability.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from style import (C, panel_tag, savefig, SANITY, NLIN, TRACT)

NOISE = SANITY / "noise_sanity_check" / "outputs"

fig = plt.figure(figsize=(13, 7.6))
gs = fig.add_gridspec(2, 3, hspace=0.46, wspace=0.32,
                      left=0.06, right=0.985, top=0.9, bottom=0.09)

# ---------------------------------------------------------------- A
axA = fig.add_subplot(gs[0, 0])
e = pd.read_csv(NOISE / "e_crossmodal_disattenuation.csv")
row = e[(e.source == "SC->FC") & (e.metric == "demeaned_pearson")].iloc[0]
ceiling, achieved = row["ceiling"], row["achieved"]
axA.bar([0], [ceiling], 0.55, color=C["fc_lt"], edgecolor="white",
        label="FC reliability ceiling")
axA.bar([0], [achieved], 0.55, color=C["SC"], edgecolor="white",
        label="SC→FC achieved")
axA.annotate("", xy=(0.42, achieved), xytext=(0.42, ceiling),
             arrowprops=dict(arrowstyle="<->", color=C["ink"], lw=1.3))
axA.text(0.5, (ceiling + achieved) / 2,
         f"only {row['fraction_of_ceiling']*100:.0f}%\nof ceiling",
         fontsize=9, fontweight="bold", va="center")
axA.text(0, ceiling + 0.012, f"{ceiling:.3f}", ha="center", fontsize=8)
axA.text(0, achieved + 0.012, f"{achieved:.3f}", ha="center", fontsize=8,
         color="white")
axA.set_xlim(-0.6, 1.1)
axA.set_xticks([])
axA.set_ylabel("demeaned pearson (Glasser)")
axA.set_ylim(0, 0.56)
axA.set_title("A · SC→FC vs FC reliability ceiling")
axA.legend(loc="upper right", fontsize=7.5)
panel_tag(axA, "A")

# ---------------------------------------------------------------- B
axB = fig.add_subplot(gs[0, 1])
rax = pd.read_csv(SANITY / "preprocessing_check" / "reduction_axis_summary.csv")
syn = pd.read_csv(SANITY / "preprocessing_check" / "reduction_axis_synthesis.csv")


def key(r):
    return r["method"] if pd.isna(r["jl_variant"]) else f"{r['method']}:{r['jl_variant']}"


rax = rax.sort_values("median_ratio")
names = {"FULL_PLS": "full PLS\n(raw edges)",
         "PCA_PLS_PCA": "PCA→PLS",
         "JL_PLS_PCA:gaussian_dense": "JL gaussian",
         "JL_PLS_PCA:sparse_auto": "JL sparse-auto",
         "JL_PLS_PCA:sparse_third": "JL sparse-1/3"}
syn["k"] = syn.apply(lambda r: r["method"] if (isinstance(r["jl_variant"], float))
                     else f"{r['method']}:{r['jl_variant']}", axis=1)
ys = []
labs = []
for i, (_, r) in enumerate(rax.iterrows()):
    k = key(r)
    labs.append(names.get(k, k))
    pts = syn[syn.k == k]["ratio"].values
    jit = np.random.default_rng(i).uniform(-0.12, 0.12, len(pts))
    axB.scatter(pts, np.full_like(pts, i) + jit, s=20, color=C["FC"],
                alpha=0.55, edgecolor="white", zorder=3)
    axB.hlines(i, r["min_ratio"], r["max_ratio"], color="#aaa", lw=1, zorder=1)
    axB.scatter([r["median_ratio"]], [i], s=70, color=C["ink"], zorder=4,
                marker="|", linewidth=2.5)
    axB.text(r["max_ratio"] + 0.05, i, f"p={r['wilcoxon_p_vs_1']:.0e}",
             va="center", fontsize=6.5, color="#666")
axB.axvline(1.0, color=C["bad"], ls="--", lw=1.2)
axB.set_yticks(range(len(labs)))
axB.set_yticklabels(labs, fontsize=7.5)
axB.set_xlabel("FC→SC / SC→FC ratio")
axB.set_xlim(0.9, 2.5)
axB.set_title("B · Asymmetry survives every reduction")
panel_tag(axB, "B")

# ---------------------------------------------------------------- C
axC = fig.add_subplot(gs[0, 2])
sc = pd.read_csv(NLIN / "n6_scaling_summary.csv")
for task, col in [("reconstruction", C["SC"]), ("cognition", C["FC"])]:
    s = sc[(sc.task == task) & (sc.n_seeds >= 5)].sort_values("n_sub")
    axC.plot(s.n_sub, s.median_gap, "-o", color=col, label=task, ms=5)
    axC.fill_between(s.n_sub, s.gap_min, s.gap_max, color=col, alpha=0.12)
axC.axhline(0, color=C["ink"], lw=1)
axC.set_xlabel("training subjects (n)")
axC.set_ylabel("nonlinear − linear  (Δ demeaned r / r)")
axC.set_title("C · No growing nonlinear gap with n")
axC.legend(loc="lower right", fontsize=8)
axC.text(0.02, 0.04, "gap stays ≤ 0:\nmore data / nonlinearity\ndoes not unlock signal",
         transform=axC.transAxes, fontsize=7, color="#666", va="bottom")
panel_tag(axC, "C")

# ---------------------------------------------------------------- D
axD = fig.add_subplot(gs[1, 0])
ts = pd.read_csv(TRACT / "tractography_synthesis.csv").set_index("row")["value"]
feats = [("SC (counts)", "E1: SC -> FC median dp", C["SC"]),
         ("SC + r2t", "E1: SC_r2t -> FC median dp", "#5DADE2"),
         ("kitchen sink", "E1: kitchen_sink -> FC median dp", "#85C1E9"),
         ("r2t bundles", "E1: r2t -> FC median dp", C["bv"]),
         ("r2t corr", "E1: r2t_corr -> FC median dp", "#B2BABB")]
vals = [float(ts[k]) for _, k, _ in feats]
cols = [c for _, _, c in feats]
yy = np.arange(len(feats))
axD.barh(yy, vals, color=cols, edgecolor="white")
axD.axvline(float(ts["E1: SC -> FC median dp"]), color=C["SC"], ls="--", lw=1)
axD.set_yticks(yy)
axD.set_yticklabels([f for f, _, _ in feats], fontsize=8)
axD.invert_yaxis()
for i, v in enumerate(vals):
    axD.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=7)
axD.set_xlabel("→ FC  (median demeaned r)")
axD.set_title("D · Richer tractography ≯ streamline counts")
axD.set_xlim(0, 0.115)
panel_tag(axD, "D")

# ---------------------------------------------------------------- E
axE = fig.add_subplot(gs[1, 1])
h = pd.read_csv(NOISE / "h_per_subject_achieved_vs_ceiling.csv")
h = h[h.source == "SC"]
hc = pd.read_csv(NOISE / "h_correlations.csv")
rr = hc[hc.source == "SC"]["pearson_achieved_vs_ceiling"].iloc[0]
axE.scatter(h.ceiling, h.achieved_mean, s=12, color=C["SC"], alpha=0.35,
            edgecolor="none")
# trend line
m, b = np.polyfit(h.ceiling, h.achieved_mean, 1)
xs = np.linspace(h.ceiling.min(), h.ceiling.max(), 50)
axE.plot(xs, m * xs + b, color=C["ink"], lw=1.5)
axE.set_xlabel("subject FC reliability (ceiling)")
axE.set_ylabel("SC→FC achieved (demeaned r)")
axE.set_title("E · SC→FC quality independent of FC reliability")
axE.text(0.04, 0.93, f"r = {rr:.2f}  (n.s.)", transform=axE.transAxes,
         fontsize=9, fontweight="bold", va="top")
panel_tag(axE, "E")

# ---------------------------------------------------------------- F (summary text)
axF = fig.add_subplot(gs[1, 2])
axF.axis("off")
checks = [
    ("PCA reduction artifact", "ratio > 1 for raw-PLS, PCA, 3× JL"),
    ("Bigger / nonlinear model", "KernelRidge & scaling gaps ≈ 0"),
    ("Crude structural features", "r2t bundles predict FC worse"),
    ("More subjects", "no growing gap to n≈680"),
    ("FC measurement noise", "SC→FC indep. of FC reliability (r=0.01)"),
]
axF.text(0.0, 1.0, "Escape routes closed", fontsize=11, fontweight="bold",
         va="top")
for i, (a, b) in enumerate(checks):
    y = 0.86 - i * 0.17
    axF.text(0.0, y, "✗", color=C["good"], fontsize=14, fontweight="bold",
             va="top")
    axF.text(0.10, y, a, fontsize=9, fontweight="bold", va="top")
    axF.text(0.10, y - 0.06, b, fontsize=7.8, color="#555", va="top")
axF.text(0.0, -0.04, "Regime: healthy young adults, cross-sectional,\n"
         "normal-range cognition.", fontsize=7.5, color="#888",
         style="italic", va="top")

fig.suptitle("Figure 4 · The missing utility is not recovered by reduction, "
             "model class, structural detail, sample size, or FC denoising",
             fontsize=12, fontweight="bold", y=0.975)
savefig(fig, "fig4_closed_escape_routes")
