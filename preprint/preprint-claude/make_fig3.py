#!/usr/bin/env python3
"""Fig 3 - Reconstruction does not imply cognitive utility.

A: cognition pearson by input vs the bv+demo bar (Glasser), 3 cognitive targets.
B: lift over bv+demo heatmap, both parcellations, with significance stars.
C: observed vs imputed contrast for CogCryst (the strongest target), both parcs.
All downstream numbers are BayesianRidge.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from style import (load_downstream, down_cell, C, PARCS, PARC_LABEL,
                   panel_tag, savefig)

d = load_downstream()
COGS = ["CogTotal", "CogFluid", "CogCryst"]

fig = plt.figure(figsize=(12, 7.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.46, wspace=0.22,
                      left=0.08, right=0.97, top=0.9, bottom=0.09)

# ---------------------------------------------------------------- A
axA = fig.add_subplot(gs[0, 0])
inputs_A = ["bv+demo", "obs_FC", "obs_FC+bv+demo", "obs_SC", "pred_SC", "pred_FC"]
labels_A = ["bv+demo\n(baseline)", "obs FC", "obs FC\n+bv+demo",
            "obs SC", "pred SC\n(FC→SC)", "pred FC\n(SC→FC)"]
cols_A = [C["base"], C["FC"], C["fc_lt"], C["SC"], C["pred"], "#C39BD3"]
parc = "Glasser"
xx = np.arange(len(inputs_A))
w = 0.26
for j, t in enumerate(COGS):
    vals = [down_cell(d, parc, inp, t, "pearson").mean() for inp in inputs_A]
    axA.bar(xx + (j - 1) * w, vals, w, label=t,
            color=plt.cm.Greys(0.35 + 0.25 * j), edgecolor="white")
base = down_cell(d, parc, "bv+demo", "CogCryst", "pearson").mean()
axA.axhline(base, color=C["base"], ls="--", lw=1.1)
axA.text(len(inputs_A) - 0.5, base + 0.005, "bv+demo (CogCryst)",
         ha="right", fontsize=7, color=C["base"])
axA.set_xticks(xx)
axA.set_xticklabels(labels_A, fontsize=7)
axA.set_ylabel("cognition prediction (pearson r)")
axA.set_title("A · Cognition prediction by input — Glasser")
axA.legend(title="target", loc="upper right", fontsize=7.5, ncols=1)
panel_tag(axA, "A")

# ---------------------------------------------------------------- B
axB = fig.add_subplot(gs[0, 1])
inputs_B = ["obs_FC", "obs_FC+obs_SC", "obs_FC+bv+demo", "obs_SC",
            "obs_SC+bv+demo", "pred_SC", "pred_SC+bv+demo", "pred_FC",
            "pred_FC+bv+demo"]
ylab = ["obs FC", "obs FC+SC", "obs FC+bv+demo", "obs SC", "obs SC+bv+demo",
        "pred SC", "pred SC+bv+demo", "pred FC", "pred FC+bv+demo"]
cols = []
for parc in PARCS:
    for t in COGS:
        cols.append(f"{parc[:3]}·{t[3:]}")
M = np.zeros((len(inputs_B), len(PARCS) * len(COGS)))
P = np.zeros_like(M)
for r_i, inp in enumerate(inputs_B):
    c_i = 0
    for parc in PARCS:
        for t in COGS:
            M[r_i, c_i] = down_cell(d, parc, inp, t, "lift_over_bvdemo").mean()
            P[r_i, c_i] = np.median(down_cell(d, parc, inp, t, "lift_perm_p"))
            c_i += 1
im = axB.imshow(M, cmap="RdBu_r", vmin=-0.17, vmax=0.17, aspect="auto")
axB.set_xticks(range(len(cols)))
axB.set_xticklabels(cols, rotation=45, ha="right", fontsize=6.8)
axB.set_yticks(range(len(ylab)))
axB.set_yticklabels(ylab, fontsize=7.5)
axB.axvline(2.5, color="white", lw=2)
for r_i in range(M.shape[0]):
    for c_i in range(M.shape[1]):
        star = "*" if P[r_i, c_i] < 0.05 else ""
        txt = f"{M[r_i, c_i]:+.2f}{star}"
        axB.text(c_i, r_i, txt, ha="center", va="center", fontsize=5.6,
                 color="white" if abs(M[r_i, c_i]) > 0.09 else "#333")
axB.set_title("B · Lift over bv+demo (BayesianRidge)  ·  * p<0.05")
cb = fig.colorbar(im, ax=axB, fraction=0.046, pad=0.02)
cb.set_label("Δ pearson vs bv+demo", fontsize=7.5)
axB.grid(False)
panel_tag(axB, "B")

# ---------------------------------------------------------------- C
axC = fig.add_subplot(gs[1, :])
# Slope-style: observed FC helps, observed SC / imputed do not, vs baseline.
order = ["bv+demo", "obs_FC", "obs_SC", "pred_SC", "pred_FC"]
nice = {"bv+demo": "bv+demo", "obs_FC": "obs FC", "obs_SC": "obs SC",
        "pred_SC": "pred SC (FC→SC)", "pred_FC": "pred FC (SC→FC)"}
mk = {"Glasser": "o", "4S456Parcels": "s"}
for parc in PARCS:
    base = down_cell(d, parc, "bv+demo", "CogCryst", "pearson").mean()
    for k, inp in enumerate(order):
        v = down_cell(d, parc, inp, "CogCryst", "pearson")
        col = (C["base"] if inp == "bv+demo" else
               C["FC"] if inp == "obs_FC" else
               C["SC"] if inp == "obs_SC" else C["pred"])
        axC.scatter([k], [v.mean()], s=90, marker=mk[parc], color=col,
                    edgecolor="white", zorder=3,
                    label=PARC_LABEL[parc] if k == 0 else None)
        axC.errorbar([k], [v.mean()], yerr=[v.std()/np.sqrt(len(v))],
                     color=col, capsize=3, zorder=2)
axC.axhspan(0, down_cell(d, "Glasser", "bv+demo", "CogCryst", "pearson").mean(),
            color=C["base"], alpha=0.06)
axC.axhline(down_cell(d, "Glasser", "bv+demo", "CogCryst", "pearson").mean(),
            color=C["base"], ls="--", lw=1.1)
axC.text(4.4, down_cell(d, "Glasser", "bv+demo", "CogCryst", "pearson").mean()+0.004,
         "cheap subject-info baseline", ha="right", fontsize=8, color=C["base"])
axC.set_xticks(range(len(order)))
axC.set_xticklabels([nice[o] for o in order])
axC.set_ylabel("CogCryst prediction (pearson r)")
axC.set_ylim(0.18, 0.54)
axC.set_title("C · Only observed FC clears the baseline for CogCryst; "
              "imputed connectomes fall below it")
axC.legend(loc="lower left")
panel_tag(axC, "C")

fig.suptitle("Figure 3 · Reconstruction does not imply cognitive utility: "
             "observed FC helps, SC and imputed connectomes do not beat bv+demo",
             fontsize=12, fontweight="bold", y=0.975)
savefig(fig, "fig3_utility_checkpoint")
