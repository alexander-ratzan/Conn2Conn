#!/usr/bin/env python3
"""Fig 5 - Predicted connectomes carry family signal, but objectives diverge.

A: sibling-separation AUC for baseline / raw / residualized / combined predicted SC.
B: the objective tradeoff - the reconstruction-optimized 'combined' predictor keeps
   MZ/DZ signal but collapses to chance for siblings.
C: property-selected mechanism mode - FC-predictability vs family AUC across PCs;
   PC1 is a confound, the selected low-variance mode (PC3 Glasser / PC4 4S456) is
   FC-predictable AND family-discriminative with near-zero confound.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from style import (load_family, C, PARCS, PARC_LABEL, panel_tag, savefig, FAM)

fam = load_family()
fig = plt.figure(figsize=(13, 5.3))
gs = fig.add_gridspec(1, 3, wspace=0.30, left=0.06, right=0.985,
                      top=0.83, bottom=0.14)

# ---------------------------------------------------------------- A
axA = fig.add_subplot(gs[0, 0])
variants = ["bvdemo_to_SC", "pred_SC_raw", "pred_SC_resid_bvdemo",
            "combined_pred_SC", "obs_SC"]
vlab = ["bv+demo\nbaseline", "pred SC\n(raw)", "pred SC\n(resid)",
        "pred SC\n(combined)", "obs SC\n(reference)"]
vcol = [C["base"], "#C39BD3", C["pred"], C["accent"], C["SC"]]
x = np.arange(len(variants))
w = 0.38
for k, parc in enumerate(PARCS):
    sub = fam[(fam.parcellation == parc) & (fam.relation == "sibling")]
    sub = sub.set_index("variant")
    vals = [sub.loc[v, "auc"] for v in variants]
    lo = [sub.loc[v, "auc"] - sub.loc[v, "auc_lo"] for v in variants]
    hi = [sub.loc[v, "auc_hi"] - sub.loc[v, "auc"] for v in variants]
    off = (k - 0.5) * w
    axA.bar(x + off, vals, w, yerr=[lo, hi], capsize=2.5,
            color=[vcol[i] for i in range(len(variants))],
            edgecolor="white", alpha=0.75 if k == 0 else 1.0,
            hatch="" if k == 0 else "///")
axA.axhline(0.5, color=C["bad"], ls="--", lw=1.2)
axA.text(len(variants) - 0.5, 0.51, "chance", color=C["bad"], fontsize=7.5,
         ha="right")
axA.set_xticks(x)
axA.set_xticklabels(vlab, fontsize=7.5)
axA.set_ylabel("sibling-separation AUC")
axA.set_ylim(0.45, 0.92)
axA.set_title("A · Family signal survives residualized prediction")
# parcellation legend (solid vs hatch)
from matplotlib.patches import Patch
axA.legend(handles=[Patch(facecolor="#bbb", alpha=0.75, label="Glasser"),
                    Patch(facecolor="#bbb", hatch="///", label="4S456Parcels")],
           loc="upper left", fontsize=7.5)
panel_tag(axA, "A")

# ---------------------------------------------------------------- B
axB = fig.add_subplot(gs[0, 1])
rels = ["MZ", "DZ", "sibling"]
parc = "Glasser"
for variant, col, lab in [("pred_SC_resid_bvdemo", C["pred"], "residual (identification obj.)"),
                          ("combined_pred_SC", C["accent"], "combined (reconstruction obj.)")]:
    sub = fam[(fam.parcellation == parc) & (fam.variant == variant)].set_index("relation")
    vals = [sub.loc[r, "auc"] for r in rels]
    lo = [sub.loc[r, "auc"] - sub.loc[r, "auc_lo"] for r in rels]
    hi = [sub.loc[r, "auc_hi"] - sub.loc[r, "auc"] for r in rels]
    axB.errorbar(range(len(rels)), vals, yerr=[lo, hi], marker="o", ms=8,
                 color=col, lw=2, capsize=3, label=lab)
axB.axhline(0.5, color=C["bad"], ls="--", lw=1.2)
axB.annotate("collapses to chance\nfor siblings", (2, 0.505), (1.25, 0.6),
             fontsize=8, color=C["accent"], fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=C["accent"]))
axB.set_xticks(range(len(rels)))
axB.set_xticklabels(["MZ twins", "DZ twins", "siblings"])
axB.set_ylabel("separation AUC (Glasser)")
axB.set_ylim(0.45, 1.0)
axB.set_title("B · Objective decides what signal survives")
axB.legend(loc="upper right", fontsize=7.5)
panel_tag(axB, "B")

# ---------------------------------------------------------------- C
axC = fig.add_subplot(gs[0, 2])
pp = pd.read_csv(FAM / "f8_per_pc.csv")
g = pp.groupby(["parcellation", "pc"]).median(numeric_only=True).reset_index()
sel = {"Glasser": 3, "4S456Parcels": 4}
mk = {"Glasser": "o", "4S456Parcels": "s"}
for parc in PARCS:
    sub = g[g.parcellation == parc]
    for _, row in sub.iterrows():
        is_pc1 = row.pc == 1
        is_sel = row.pc == sel[parc]
        size = 60 + 1400 * row.explained_var_ratio
        if is_pc1:
            col, ec, z = C["bad"], "black", 5
        elif is_sel:
            col, ec, z = C["accent"], "black", 6
        else:
            col, ec, z = "#BDC3C7", "white", 3
        axC.scatter(row.FC_to_PC_R2, row.AUC_sibling, s=size, marker=mk[parc],
                    color=col, edgecolor=ec, linewidth=1.1, zorder=z, alpha=0.9)
        if is_pc1 or is_sel:
            axC.annotate(f"PC{int(row.pc)}", (row.FC_to_PC_R2, row.AUC_sibling),
                         (6, 6), textcoords="offset points", fontsize=8,
                         fontweight="bold")
axC.axhline(0.5, color=C["bad"], ls=":", lw=1)
axC.set_xlabel("FC→PC predictability (R²)")
axC.set_ylabel("sibling-separation AUC")
axC.set_title("C · Property-selected SC mode")
axC.text(0.03, 0.04,
         "red = PC1 (sex/volume confound)\ngold = selected FC-predictable mode\n"
         "size ∝ variance explained",
         transform=axC.transAxes, fontsize=7, color="#555", ha="left",
         va="bottom")
axC.legend(handles=[plt.Line2D([], [], marker="o", ls="", color="#999",
                               label="Glasser"),
                    plt.Line2D([], [], marker="s", ls="", color="#999",
                               label="4S456Parcels")],
           loc="upper right", fontsize=7.5)
panel_tag(axC, "C")

fig.suptitle("Figure 5 · Predicted connectomes carry heritable family signal — "
             "but reconstruction and identification select different information",
             fontsize=12, fontweight="bold", y=0.97)
savefig(fig, "fig5_objective_divergence")
