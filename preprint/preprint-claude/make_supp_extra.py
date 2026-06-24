#!/usr/bin/env python3
"""Extra detailed sanity-check figures for the long supplement.
Writes PNG+PDF to figures/supp/. Run with the dev-env interpreter.
All values read from source CSVs.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from style import (C, PARCS, PARC_LABEL, panel_tag, FIGDIR, FAM, SANITY, TRACT)

SUPP = FIGDIR / "supp"
SUPP.mkdir(exist_ok=True)
NOISE = SANITY / "noise_sanity_check" / "outputs"
TCHECK = SANITY / "tract_check"


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(SUPP / f"{name}.{ext}")
    print(f"  wrote figures/supp/{name}.png")
    plt.close(fig)


# ================================================================ SX1
# FC variance decomposition + reliability-ceiling rungs.
b = pd.read_csv(NOISE / "b_variance_decomposition.csv")
a = pd.read_csv(NOISE / "a_reliability_ceiling.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
ax = axes[0]
comps = ["trait_frac_mean", "state_frac_mean", "within_sess_frac_mean", "noise_frac_mean"]
clabs = ["trait\n(signal)", "state\n(day)", "within\nsession", "noise"]
ccols = [C["good"], C["accent"], "#5DADE2", C["bad"]]
xx = np.arange(len(PARCS)); w = 0.6
bottom = np.zeros(len(PARCS))
for comp, lab, col in zip(comps, clabs, ccols):
    vals = [b[b.parc == p][comp].iloc[0] for p in PARCS]
    ax.bar(xx, vals, w, bottom=bottom, label=lab, color=col, edgecolor="white")
    for i, v in enumerate(vals):
        if v > 0.04:
            ax.text(i, bottom[i] + v/2, f"{v*100:.0f}%", ha="center",
                    va="center", fontsize=8, color="white", fontweight="bold")
    bottom += np.array(vals)
ax.set_xticks(xx); ax.set_xticklabels([PARC_LABEL[p] for p in PARCS])
ax.set_ylabel("fraction of between-subject edge variance")
ax.set_title("A · Single-edge FC variance: ~64% noise")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
ax = axes[1]
rungs = ["within_session_run1", "between_session"]
rlab = ["within-session\n(LR/RL, distortion-confounded)", "between-session\n(REST1/REST2, valid ceiling)"]
for i, parc in enumerate(PARCS):
    vals = [a[(a.parc == parc) & (a.comparison == r)]["demeaned_pearson"].iloc[0] for r in rungs]
    ax.bar(np.arange(2) + i*0.4 - 0.2, vals, 0.38,
           color=[C["sc_lt"], C["FC"]] if i == 0 else [C["bv"], C["bad"]],
           edgecolor="white", label=PARC_LABEL[parc])
ax.set_xticks(range(2)); ax.set_xticklabels(rlab, fontsize=7.5)
ax.set_ylabel("demeaned pearson (reliability)")
ax.set_title("B · FC reliability ceiling (between-session = 0.49)")
ax.text(0.5, 0.46, "G(avg connectome)\n≈ 0.52–0.59", fontsize=7.5, color="#666",
        ha="center", style="italic")
panel_tag(axes[0], "A"); panel_tag(axes[1], "B")
fig.suptitle("SX1 · FC is mostly edge-noise but reliable as a whole connectome",
             fontweight="bold", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93)); save(fig, "SX1_variance_reliability")

# ================================================================ SX2
# Tractography downstream cognition: raw + residualized, by representation.
e5 = pd.read_csv(TRACT / "e5_downstream_summary.csv")
e5["rep"] = e5["rep"].astype(str)
order = ["FC", "bv+demo", "SC", "r2t", "SC_r2t", "r2t_corr", "r2t->synthFC"]
tgt = "CogCrystalComp_Unadj"
fig, ax = plt.subplots(figsize=(9, 4.4))
sub = e5[e5.target == tgt].set_index("rep")
vals_raw = [sub.loc[r, "pearson_raw"] if r in sub.index else np.nan for r in order]
floor = sub.loc["bv+demo", "pearson_raw"]
cols = []
for r in order:
    cols.append(C["FC"] if r == "FC" else C["base"] if r == "bv+demo"
                else C["SC"] if r == "SC" else C["pred"])
ax.bar(range(len(order)), vals_raw, color=cols, edgecolor="white")
ax.axhline(floor, color=C["base"], ls="--", lw=1.2)
ax.text(len(order)-0.5, floor+0.005, "bv+demo floor", ha="right", fontsize=8, color=C["base"])
for i, v in enumerate(vals_raw):
    if not np.isnan(v):
        ax.text(i, v+0.006, f"{v:.2f}", ha="center", fontsize=7.5)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(["FC", "bv+demo", "SC", "r2t", "SC+r2t", "r2t corr", "r2t→\nsynthFC"],
                   fontsize=8)
ax.set_ylabel("CogCryst prediction (pearson r)")
ax.set_title("SX2 · Only FC clears the cognition floor; no tractography rep does")
ax.set_ylim(0, 0.5)
fig.tight_layout(); save(fig, "SX2_tractography_downstream")

# ================================================================ SX3
# Per-PC stability: FC->PC R2, confound R2, sibling AUC (both parcellations).
pp = pd.read_csv(FAM / "f8_per_pc.csv")
g = pp.groupby(["parcellation", "pc"]).median(numeric_only=True).reset_index()
sel = {"Glasser": 3, "4S456Parcels": 4}
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True)
for ax, parc in zip(axes, PARCS):
    s = g[g.parcellation == parc]
    ax.bar(s.pc - 0.27, s.FC_to_PC_R2, 0.27, label="FC→PC R²", color=C["SC"], edgecolor="white")
    ax.bar(s.pc, s.confound_R2_test, 0.27, label="confound R² (sex/vol)", color=C["bad"], edgecolor="white")
    ax.bar(s.pc + 0.27, s.AUC_sibling - 0.5, 0.27, bottom=0.5,
           label="sibling AUC (−0.5 base)", color=C["pred"], edgecolor="white")
    ax.axvline(sel[parc], color=C["accent"], lw=8, alpha=0.18, zorder=0)
    ax.set_title(f"{PARC_LABEL[parc]}  (selected: PC{sel[parc]})")
    ax.set_xlabel("principal component"); ax.set_xticks(range(1, 11))
axes[0].set_ylabel("value"); axes[0].legend(fontsize=7.5, loc="upper right")
fig.suptitle("SX3 · PC1 is FC-predictable but a sex/volume confound; the selected "
             "low-variance mode is FC-predictable, family-discriminative, low-confound",
             fontweight="bold", fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.93)); save(fig, "SX3_per_pc_confound")

# ================================================================ SX4
# PC3 network enrichment: raw vs reliability-residualized (tract_check).
enr = pd.read_csv(TCHECK / "enrichment_residual_top200.csv")
enr = enr.sort_values("enrichment_raw", ascending=False).head(8)
fig, ax = plt.subplots(figsize=(9, 4.6))
yy = np.arange(len(enr))[::-1]
ax.barh(yy + 0.2, enr.enrichment_raw, 0.38, label="raw", color=C["bv"], edgecolor="white")
ax.barh(yy - 0.2, enr.enrichment_resid, 0.38, label="residualized (− strength, distance)",
        color=C["SC"], edgecolor="white")
ax.axvline(1.0, color=C["ink"], ls=":", lw=1)
ax.set_yticks(yy); ax.set_yticklabels(enr.net_pair, fontsize=7.5)
ax.set_xlabel("network-pair enrichment in selected SC mode (×chance)")
ax.set_title("SX4 · Visual/DAN localization survives reliability partialling\n"
             "(strength+distance explain 41% of |PC3|; residual stays visual/DAN)")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout(); save(fig, "SX4_pc_enrichment_partialled")

print("\nExtra supplement figures in", SUPP)
