#!/usr/bin/env python3
"""Figures for the grid exploration. Writes PNGs to exploration/figures/.
Run from reproduction/:  python exploration/make_figures.py
"""
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
r = pd.read_csv(OUT / "reconstruction.csv"); d = pd.read_csv(OUT / "downstream.csv")
PARCS = sorted(r.parcellation.unique())


def cell(df, parc, est, a, b, col="demeaned_pearson"):
    s = df[(df.parcellation == parc) & (df.estimator == est) & (df.input_set == a) & (df.target == b)][col]
    s = s[np.isfinite(s)]; return s.mean(), s.std()


# ---- Fig 1: asymmetry across estimators ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
ests = ["pca_pls", "bayesian_ridge", "kernel_ridge"]
x = np.arange(len(ests)); w = 0.35
for ax, parc in zip(axes, PARCS):
    fcsc = [cell(r, parc, e, "FC", "SC") for e in ests]
    scfc = [cell(r, parc, e, "SC", "FC") for e in ests]
    ax.bar(x - w/2, [m for m, _ in fcsc], w, yerr=[s for _, s in fcsc], label="FC→SC", capsize=3)
    ax.bar(x + w/2, [m for m, _ in scfc], w, yerr=[s for _, s in scfc], label="SC→FC", capsize=3)
    for i in range(len(ests)):
        ax.text(i, max(fcsc[i][0], scfc[i][0]) + 0.012,
                f"{fcsc[i][0]/scfc[i][0]:.2f}×", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([e.split("_")[0] for e in ests])
    ax.set_title(parc); ax.set_ylabel("demeaned pearson"); ax.legend()
fig.suptitle("F1 — cross-modal asymmetry FC→SC > SC→FC (robust across estimators)")
fig.tight_layout(); fig.savefig(FIG / "f1_asymmetry.png", dpi=130); plt.close(fig)

# ---- Fig 2: F2 double dissociation (bv vs demo into SC vs FC) ----
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, parc in zip(axes, PARCS):
    tg = ["SC", "FC"]; xx = np.arange(2)
    bv = [cell(r, parc, "pca_pls", "bv", t) for t in tg]
    de = [cell(r, parc, "pca_pls", "demo", t) for t in tg]
    ax.bar(xx - w/2, [m for m, _ in bv], w, yerr=[s for _, s in bv], label="bv (anatomy)", capsize=3)
    ax.bar(xx + w/2, [m for m, _ in de], w, yerr=[s for _, s in de], label="demo (demographics)", capsize=3)
    ax.set_xticks(xx); ax.set_xticklabels(["→SC", "→FC"]); ax.set_title(parc)
    ax.set_ylabel("demeaned pearson"); ax.legend()
fig.suptitle("F2 — double dissociation: anatomy→structure, demographics→function")
fig.tight_layout(); fig.savefig(FIG / "f2_dissociation.png", dpi=130); plt.close(fig)

# ---- Fig 3: downstream lift heatmap (bayesian_ridge) ----
inputs = ["obs_FC", "obs_FC+obs_SC", "obs_FC+bv+demo", "obs_SC", "pred_SC", "pred_FC",
          "pred_SC+bv+demo", "pred_FC+bv+demo"]
cogs = ["CogTotal", "CogFluid", "CogCryst"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, parc in zip(axes, PARCS):
    M = np.array([[cell(d, parc, "bayesian_ridge", inp, t, "lift_over_bvdemo")[0] for t in cogs] for inp in inputs])
    im = ax.imshow(M, cmap="RdBu_r", vmin=-0.16, vmax=0.16, aspect="auto")
    ax.set_xticks(range(len(cogs))); ax.set_xticklabels([c[3:] for c in cogs])
    ax.set_yticks(range(len(inputs))); ax.set_yticklabels(inputs, fontsize=8)
    for i in range(len(inputs)):
        for j in range(len(cogs)):
            ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(parc); fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("Downstream lift over bv+demo (bayesian_ridge) — obs_FC helps, pred_FC hurts")
fig.tight_layout(); fig.savefig(FIG / "f3_downstream_lift.png", dpi=130); plt.close(fig)

# ---- Fig 4: cross-parcellation scatter ----
fig, ax = plt.subplots(figsize=(5.6, 5.6))
pls = r[r.estimator == "pca_pls"]
pairs = [("FC", "SC"), ("SC", "FC"), ("bv", "SC"), ("bv", "FC"), ("demo", "SC"), ("demo", "FC"),
         ("bv+demo", "SC"), ("bv+demo", "FC"), ("FC+bv+demo", "SC"), ("SC+bv+demo", "FC"),
         ("FC", "FC"), ("SC", "SC")]
gx = [cell(pls, "Glasser", "pca_pls", a, b)[0] for a, b in pairs]
sy = [cell(pls, "4S456Parcels", "pca_pls", a, b)[0] for a, b in pairs]
ax.scatter(gx, sy, s=40)
for (a, b), xg, ys in zip(pairs, gx, sy):
    ax.annotate(f"{a}→{b}", (xg, ys), fontsize=7, xytext=(3, 3), textcoords="offset points")
lim = [0, max(gx + sy) * 1.1]; ax.plot(lim, lim, "k--", alpha=0.4)
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel("Glasser demeaned_r"); ax.set_ylabel("4S456 demeaned_r")
ax.set_title("Cross-parcellation replication (pca_pls recon)\nabove line = 4S456 higher (→SC pairs)")
fig.tight_layout(); fig.savefig(FIG / "f4_cross_parcellation.png", dpi=130); plt.close(fig)

# ---- Fig 5: cross-modal vs oracle ceiling ----
fig, ax = plt.subplots(figsize=(7, 4.2))
labels, xmod, orac = [], [], []
for parc in PARCS:
    for src, tgt in [("FC", "SC"), ("SC", "FC")]:
        labels.append(f"{parc[:4]}\n{src}→{tgt}")
        xmod.append(cell(r, parc, "pca_pls", src, tgt)[0])
        orac.append(cell(r, parc, "bayesian_ridge", tgt, tgt)[0])  # oracle of the TARGET modality
xx = np.arange(len(labels))
ax.bar(xx, orac, 0.6, label="within-modal oracle (Ceiling B, BR)", color="lightgray")
ax.bar(xx, xmod, 0.6, label="cross-modal (pca_pls)", color="tab:blue")
for i in range(len(labels)):
    ax.text(i, xmod[i] + 0.01, f"{100*xmod[i]/orac[i]:.0f}%", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(xx); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel("demeaned pearson")
ax.set_title("Cross-modal reaches only ~20–24% of the within-modal ceiling"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "f5_ceiling_gap.png", dpi=130); plt.close(fig)

print("wrote 5 figures to", FIG)
