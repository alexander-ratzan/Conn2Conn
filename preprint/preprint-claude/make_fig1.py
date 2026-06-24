#!/usr/bin/env python3
"""Fig 1 - Evaluation spine (schematic).

A: inputs -> targets.  B: the two branches (reconstruction vs downstream utility)
with the bv+demo checkpoint.  C: frozen family-aware splits x both parcellations.
This is a hand-drawn diagram, not a data plot.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from style import C, savefig, panel_tag


def box(ax, x, y, w, h, text, fc, tc="white", fs=8.5, lw=0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.012,rounding_size=0.02",
                 facecolor=fc, edgecolor="white", linewidth=lw, zorder=2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", color=tc,
            fontsize=fs, fontweight="bold", zorder=3)


def arrow(ax, x1, y1, x2, y2, color="#566573", lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=12, color=color, lw=lw, zorder=1))


fig = plt.figure(figsize=(13, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.25, 0.95], wspace=0.12,
                      left=0.01, right=0.99, top=0.86, bottom=0.04)

# ---------------------------------------------------------------- A inputs/targets
axA = fig.add_subplot(gs[0, 0]); axA.axis("off")
axA.set_xlim(0, 1); axA.set_ylim(0, 1)
axA.text(0.5, 0.97, "Inputs", ha="center", fontsize=10, fontweight="bold")
inputs = [("FC", C["FC"]), ("SC", C["SC"]), ("brain volume", C["bv"]),
          ("demographics", C["demo"]), ("bv + demo", C["base"])]
for i, (t, c) in enumerate(inputs):
    box(axA, 0.04, 0.80 - i*0.155, 0.42, 0.11, t, c)
axA.text(0.75, 0.97, "Targets", ha="center", fontsize=10, fontweight="bold")
targets = [("SC", C["SC"]), ("FC", C["FC"]), ("Cognition\n(Total/Fluid/Cryst)", "#34495E"),
           ("sex / age\n(leak check)", "#95A5A6")]
for i, (t, c) in enumerate(targets):
    box(axA, 0.56, 0.74 - i*0.19, 0.40, 0.14, t, c, fs=8)
for i in range(len(inputs)):
    arrow(axA, 0.47, 0.855 - i*0.155, 0.55, 0.5, color="#CCD1D1", lw=0.8)
panel_tag(axA, "A")
axA.set_title("Inputs and targets", fontsize=10)

# ---------------------------------------------------------------- B two branches
axB = fig.add_subplot(gs[0, 1]); axB.axis("off")
axB.set_xlim(0, 1); axB.set_ylim(0, 1)
box(axB, 0.34, 0.86, 0.32, 0.11, "Connectome\ninput", C["ink"], fs=8)
# reconstruction branch
box(axB, 0.02, 0.58, 0.44, 0.13, "RECONSTRUCTION\nFC↔SC", C["SC"], fs=8.5)
box(axB, 0.02, 0.36, 0.44, 0.13, "metric:\ndemeaned pearson", "#5499C7", fs=8)
box(axB, 0.02, 0.14, 0.44, 0.13, "+ oracle ceilings\nFC→FC, SC→SC", "#7FB3D5", fs=8)
arrow(axB, 0.4, 0.86, 0.24, 0.715)
arrow(axB, 0.24, 0.58, 0.24, 0.495)
arrow(axB, 0.24, 0.36, 0.24, 0.275)
# downstream branch
box(axB, 0.54, 0.58, 0.44, 0.13, "DOWNSTREAM UTILITY\ncognition", C["FC"], fs=8.5)
box(axB, 0.54, 0.33, 0.44, 0.16, "CHECKPOINT\nbeat bv+demo?", C["accent"],
    tc=C["ink"], fs=9)
box(axB, 0.54, 0.11, 0.44, 0.13, "lift over baseline\n+ paired permutation", "#E59866",
    fs=8)
arrow(axB, 0.6, 0.86, 0.76, 0.715)
arrow(axB, 0.76, 0.58, 0.76, 0.495)
arrow(axB, 0.76, 0.33, 0.76, 0.245)
axB.text(0.5, 0.015, "observed  vs  imputed connectomes evaluated on both branches",
         ha="center", fontsize=7.5, color="#666", style="italic")
panel_tag(axB, "B")
axB.set_title("Reconstruction is separated from utility", fontsize=10)

# ---------------------------------------------------------------- C splits/grid
axC = fig.add_subplot(gs[0, 2]); axC.axis("off")
axC.set_xlim(0, 1); axC.set_ylim(0, 1)
axC.text(0.5, 0.95, "10 frozen family-aware splits", ha="center",
         fontsize=9.5, fontweight="bold")
for i in range(10):
    xx = 0.05 + (i % 5) * 0.185
    yy = 0.74 - (i // 5) * 0.13
    box(axC, xx, yy, 0.15, 0.09, f"s{i}", C["pred"], fs=7.5)
axC.text(0.5, 0.5, "siblings & twins kept on the same side",
         ha="center", fontsize=7.5, color="#666", style="italic")
# two parcellations
box(axC, 0.05, 0.27, 0.42, 0.13, "Glasser\n360 nodes\n64,620 edges", C["SC"], fs=7)
box(axC, 0.53, 0.27, 0.42, 0.13, "4S456Parcels\n456 nodes\n103,740 edges", "#2E86C1", fs=7)
box(axC, 0.20, 0.05, 0.60, 0.12, "reused across recon · downstream ·\n"
    "imputation · leak · family", C["ink"], fs=7.5)
arrow(axC, 0.26, 0.27, 0.4, 0.17)
arrow(axC, 0.74, 0.27, 0.6, 0.17)
panel_tag(axC, "C")
axC.set_title("Frozen splits × two parcellations", fontsize=10)

fig.suptitle("Figure 1 · Evaluation spine: one frozen grid that separates connectome "
             "reconstruction from downstream cognitive utility",
             fontsize=12, fontweight="bold", y=0.99)
savefig(fig, "fig1_evaluation_spine")
