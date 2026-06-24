#!/usr/bin/env python3
"""Fig 2 - Directional translation: FC->SC > SC->FC.

A: FC->SC vs SC->FC demeaned-pearson by parcellation, with x-ratio.
B: seed-level ratios (all >1) across parcellations.
C: within-modality oracle ceilings (FC->FC, SC->SC) - similar self-predictability.
D: anatomy/demographics double dissociation (bv->SC, demo->FC).
"""
import numpy as np
import matplotlib.pyplot as plt
from style import (load_recon, recon_cell, C, PARCS, PARC_LABEL,
                   panel_tag, savefig)

r = load_recon()
fig = plt.figure(figsize=(11.5, 7.4))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.26,
                      left=0.07, right=0.985, top=0.9, bottom=0.08)

# ---------------------------------------------------------------- A
axA = fig.add_subplot(gs[0, 0])
x = np.arange(len(PARCS))
w = 0.36
for i, parc in enumerate(PARCS):
    fcsc = recon_cell(r, parc, "pca_pls", "FC", "SC")
    scfc = recon_cell(r, parc, "pca_pls", "SC", "FC")
    axA.bar(i - w/2, fcsc.mean(), w, yerr=fcsc.std(), capsize=3,
            color=C["FC"], edgecolor="white",
            label="FC→SC" if i == 0 else None)
    axA.bar(i + w/2, scfc.mean(), w, yerr=scfc.std(), capsize=3,
            color=C["SC"], edgecolor="white",
            label="SC→FC" if i == 0 else None)
    ratio = fcsc.mean() / scfc.mean()
    top = max(fcsc.mean(), scfc.mean()) + 0.022
    axA.annotate(f"{ratio:.2f}×", (i, top), ha="center",
                 fontsize=11, fontweight="bold", color=C["ink"])
axA.set_xticks(x)
axA.set_xticklabels([PARC_LABEL[p] for p in PARCS])
axA.set_ylabel("demeaned pearson")
axA.set_ylim(0, 0.20)
axA.set_title("Cross-modal reconstruction (PCA→PLS)")
axA.legend(loc="upper right")
panel_tag(axA, "A")

# ---------------------------------------------------------------- B
axB = fig.add_subplot(gs[0, 1])
rng = np.random.default_rng(0)


def per_seed_ratio(parc):
    sub = r[(r.parcellation == parc) & (r.estimator == "pca_pls") &
            (r.is_block == False)]
    a = sub[(sub.source == "FC") & (sub.target == "SC")].set_index("seed")["demeaned_pearson"]
    b = sub[(sub.source == "SC") & (sub.target == "FC")].set_index("seed")["demeaned_pearson"]
    j = a.index.intersection(b.index)
    return (a.loc[j] / b.loc[j]).values


for i, parc in enumerate(PARCS):
    ratios = per_seed_ratio(parc)
    jitter = rng.uniform(-0.08, 0.08, size=len(ratios))
    axB.scatter(np.full_like(ratios, i) + jitter, ratios, s=34,
                color=C["FC"], alpha=0.75, edgecolor="white", zorder=3)
    axB.hlines(np.median(ratios), i - 0.22, i + 0.22, color=C["ink"],
               lw=2.2, zorder=4)
    axB.annotate(f"med {np.median(ratios):.2f}×", (i, ratios.max() + 0.06),
                 ha="center", fontsize=8.5, fontweight="bold")
axB.axhline(1.0, color=C["bad"], ls="--", lw=1.2)
axB.text(1.45, 1.02, "ratio = 1 (no asymmetry)", color=C["bad"],
         fontsize=7.5, ha="right", va="bottom")
axB.set_xticks(x)
axB.set_xticklabels([PARC_LABEL[p] for p in PARCS])
axB.set_ylabel("FC→SC / SC→FC ratio")
axB.set_ylim(0.9, None)
axB.set_title("Per-seed asymmetry (every seed > 1)")
panel_tag(axB, "B")

# ---------------------------------------------------------------- C
axC = fig.add_subplot(gs[1, 0])
for i, parc in enumerate(PARCS):
    ff = recon_cell(r, parc, "bayesian_ridge", "FC", "FC")
    ss = recon_cell(r, parc, "bayesian_ridge", "SC", "SC")
    fcsc = recon_cell(r, parc, "pca_pls", "FC", "SC")
    scfc = recon_cell(r, parc, "pca_pls", "SC", "FC")
    xs = np.array([0, 1, 2, 3]) + i * 5
    vals = [ff.mean(), ss.mean(), fcsc.mean(), scfc.mean()]
    errs = [ff.std(), ss.std(), fcsc.std(), scfc.std()]
    cols = [C["FC"], C["SC"], C["fc_lt"], C["sc_lt"]]
    axC.bar(xs, vals, 0.8, yerr=errs, capsize=2.5, color=cols,
            edgecolor="white")
labels = ["FC→FC", "SC→SC", "FC→SC", "SC→FC"]
axC.set_xticks(list(range(4)) + [c + 5 for c in range(4)])
axC.set_xticklabels(labels * 2, rotation=30, ha="right", fontsize=7.5)
axC.set_ylabel("demeaned pearson")
axC.set_ylim(0, 0.82)
axC.set_title("Within-modality oracle ceilings vs cross-modal")
for i, parc in enumerate(PARCS):
    axC.text(1.5 + i * 5, 0.79, parc.split("Parcels")[0],
             ha="center", fontsize=8.5, color="#555", fontweight="bold")
panel_tag(axC, "C")

# ---------------------------------------------------------------- D
axD = fig.add_subplot(gs[1, 1])
tg = ["SC", "FC"]
xx = np.arange(2)
for i, parc in enumerate(PARCS):
    off = (i - 0.5) * 0.0  # overlay both parcs as paired groups below
for k, parc in enumerate(PARCS):
    bv = [recon_cell(r, parc, "pca_pls", "bv", t).mean() for t in tg]
    de = [recon_cell(r, parc, "pca_pls", "demo", t).mean() for t in tg]
    base = k * 3
    axD.bar(base + 0 - 0.18, bv[0], 0.36, color=C["bv"], edgecolor="white",
            label="brain volume" if k == 0 else None)
    axD.bar(base + 0 + 0.18, de[0], 0.36, color=C["demo"], edgecolor="white",
            label="demographics" if k == 0 else None)
    axD.bar(base + 1 - 0.18, bv[1], 0.36, color=C["bv"], edgecolor="white")
    axD.bar(base + 1 + 0.18, de[1], 0.36, color=C["demo"], edgecolor="white")
axD.set_xticks([0, 1, 3, 4])
axD.set_xticklabels(["→SC", "→FC", "→SC", "→FC"])
axD.text(0.5, -0.052, "Glasser", ha="center", fontsize=8, color="#555")
axD.text(3.5, -0.052, "4S456", ha="center", fontsize=8, color="#555")
axD.set_ylabel("demeaned pearson")
axD.set_ylim(0, 0.21)
axD.set_title("Double dissociation: anatomy→SC, demographics→FC")
axD.legend(loc="upper right")
panel_tag(axD, "D")

fig.suptitle("Figure 2 · Cross-modal connectome prediction is directional: "
             "FC predicts SC better than SC predicts FC",
             fontsize=12, fontweight="bold", y=0.975)
savefig(fig, "fig2_directional_translation")
