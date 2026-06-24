#!/usr/bin/env python3
"""Supplementary figures S1-S10. Each writes PNG+PDF to figures/supp/.

Run:  /Users/user/dev-env/bin/python make_supp.py
All values read from source CSVs at render time.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from style import (load_recon, load_downstream, load_family, C, PARCS,
                   PARC_LABEL, panel_tag, FIGDIR, FAM, SANITY, NLIN, TRACT)

SUPP = FIGDIR / "supp"
SUPP.mkdir(exist_ok=True)
NOISE = SANITY / "noise_sanity_check" / "outputs"


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(SUPP / f"{name}.{ext}")
    print(f"  wrote figures/supp/{name}.png")
    plt.close(fig)


r = load_recon()
d = load_downstream()

# ================================================================ S1
# raw vs demeaned pearson: raw is dominated by the population mean.
fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
for ax, parc in zip(axes, PARCS):
    rows = [("FC→SC", "FC", "SC"), ("SC→FC", "SC", "FC")]
    x = np.arange(len(rows)); w = 0.36
    raw = [r[(r.parcellation == parc) & (r.estimator == "pca_pls") &
             (r.source == s) & (r.target == t) & (~r.is_block)]["pearson"].mean()
           for _, s, t in rows]
    dem = [r[(r.parcellation == parc) & (r.estimator == "pca_pls") &
             (r.source == s) & (r.target == t) & (~r.is_block)]["demeaned_pearson"].mean()
           for _, s, t in rows]
    ax.bar(x - w/2, raw, w, color="#B2BABB", edgecolor="white", label="raw pearson")
    ax.bar(x + w/2, dem, w, color=C["FC"], edgecolor="white", label="demeaned pearson")
    for i in range(len(rows)):
        ax.text(i - w/2, raw[i] + 0.01, f"{raw[i]:.2f}", ha="center", fontsize=8)
        ax.text(i + w/2, dem[i] + 0.01, f"{dem[i]:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([n for n, _, _ in rows])
    ax.set_title(PARC_LABEL[parc]); ax.set_ylim(0, 1.05)
axes[0].set_ylabel("correlation")
axes[0].legend(loc="center right")
fig.suptitle("S1 · Raw pearson is dominated by the shared population-mean connectome;\n"
             "demeaned pearson isolates individual-deviation reconstruction",
             fontweight="bold", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.92)); save(fig, "S1_raw_vs_demeaned")

# ================================================================ S2
# seed-level reconstruction distributions (cross-modal + oracle).
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
combos = [("FC→SC", "FC", "SC", "pca_pls", C["FC"]),
          ("SC→FC", "SC", "FC", "pca_pls", C["SC"]),
          ("FC→FC", "FC", "FC", "bayesian_ridge", C["fc_lt"]),
          ("SC→SC", "SC", "SC", "bayesian_ridge", C["sc_lt"])]
for ax, parc in zip(axes, PARCS):
    data = []; labs = []; cols = []
    for name, s, t, est, c in combos:
        v = r[(r.parcellation == parc) & (r.estimator == est) &
              (r.source == s) & (r.target == t) & (~r.is_block)]["demeaned_pearson"].values
        data.append(v); labs.append(name); cols.append(c)
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color(C["ink"])
    for i, v in enumerate(data):
        ax.scatter(np.full_like(v, i + 1) + np.random.default_rng(i).uniform(-0.1, 0.1, len(v)),
                   v, s=12, color=C["ink"], alpha=0.4, zorder=3)
    ax.set_xticklabels(labs); ax.set_title(PARC_LABEL[parc])
axes[0].set_ylabel("demeaned pearson (10 seeds)")
fig.suptitle("S2 · Seed-level reconstruction distributions (n=10 frozen splits)",
             fontweight="bold", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94)); save(fig, "S2_seed_reconstruction")

# ================================================================ S3
# downstream seed-level lifts for key inputs (CogCryst).
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
inputs = ["obs_FC", "obs_SC", "pred_SC", "pred_FC"]
icol = [C["FC"], C["SC"], C["pred"], "#C39BD3"]
for ax, parc in zip(axes, PARCS):
    for i, (inp, c) in enumerate(zip(inputs, icol)):
        v = d[(d.parcellation == parc) & (d.estimator == "bayesian_ridge") &
              (d.input_set == inp) & (d.target == "CogCryst")]["lift_over_bvdemo"].values
        ax.scatter(np.full_like(v, i) + np.random.default_rng(i).uniform(-0.1, 0.1, len(v)),
                   v, s=22, color=c, alpha=0.7, edgecolor="white", zorder=3)
        ax.hlines(np.median(v), i - 0.25, i + 0.25, color=C["ink"], lw=2)
    ax.axhline(0, color=C["base"], ls="--", lw=1.1)
    ax.set_xticks(range(len(inputs)))
    ax.set_xticklabels(["obs FC", "obs SC", "pred SC", "pred FC"])
    ax.set_title(PARC_LABEL[parc])
axes[0].set_ylabel("CogCryst lift over bv+demo (per seed)")
fig.suptitle("S3 · Seed-level cognition lift: only observed FC sits above zero",
             fontweight="bold", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94)); save(fig, "S3_downstream_lifts")

# ================================================================ S4
# leak check: sex (balanced_acc) and age (pearson) by input.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
parc = "Glasser"
li = ["bv+demo", "obs_FC", "obs_SC", "pred_FC", "pred_SC",
      "pred_FC+bv+demo", "pred_SC+bv+demo"]
ax = axes[0]
sex = [d[(d.parcellation == parc) & (d.estimator == "bayesian_ridge") &
         (d.input_set == i) & (d.target == "sex")]["balanced_acc"].mean() for i in li]
ax.barh(range(len(li)), sex, color=C["demo"], edgecolor="white")
ax.axvline(0.5, color=C["bad"], ls="--", lw=1); ax.set_xlim(0, 1.05)
ax.set_yticks(range(len(li))); ax.set_yticklabels(li, fontsize=8); ax.invert_yaxis()
ax.set_xlabel("sex balanced accuracy"); ax.set_title("sex (leak check)")
ax = axes[1]
age = [d[(d.parcellation == parc) & (d.estimator == "bayesian_ridge") &
         (d.input_set == i) & (d.target == "age")]["pearson"].mean() for i in li]
ax.barh(range(len(li)), age, color="#AF7AC5", edgecolor="white")
ax.set_yticks(range(len(li))); ax.set_yticklabels(li, fontsize=8); ax.invert_yaxis()
ax.set_xlabel("age pearson"); ax.set_title("age (leak check)"); ax.set_xlim(0, 1.05)
ax.text(0.99, len(li) - 0.5, "any input with bv+demo → ~1.0:\nexpected, bv+demo encodes age/sex\n(float64 PCA fix, commit 69add40)",
        ha="right", fontsize=7, color="#666", style="italic")
fig.suptitle("S4 · Leak-check targets (Glasser, BayesianRidge). "
             "Sex/age are diagnostics, not scientific outcomes.",
             fontweight="bold", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93)); save(fig, "S4_leak_checks")

# ================================================================ S5
# reduction-axis robustness (per-seed ratios by method).
syn = pd.read_csv(SANITY / "preprocessing_check" / "reduction_axis_synthesis.csv")
summ = pd.read_csv(SANITY / "preprocessing_check" / "reduction_axis_summary.csv")
syn["k"] = syn.apply(lambda x: x["method"] if isinstance(x["jl_variant"], float)
                     else f"{x['method']}:{x['jl_variant']}", axis=1)
fig, ax = plt.subplots(figsize=(8, 4.4))
order = summ.sort_values("median_ratio")
names = {"FULL_PLS": "full PLS (raw 64,620-d)", "PCA_PLS_PCA": "PCA→PLS",
         "JL_PLS_PCA:gaussian_dense": "JL gaussian", "JL_PLS_PCA:sparse_auto": "JL sparse-auto",
         "JL_PLS_PCA:sparse_third": "JL sparse-1/3"}
for i, (_, row) in enumerate(order.iterrows()):
    k = row["method"] if pd.isna(row["jl_variant"]) else f"{row['method']}:{row['jl_variant']}"
    pts = syn[syn.k == k]["ratio"].values
    ax.scatter(pts, np.full_like(pts, i) + np.random.default_rng(i).uniform(-.12, .12, len(pts)),
               s=28, color=C["FC"], alpha=0.6, edgecolor="white", zorder=3)
    ax.scatter([row["median_ratio"]], [i], marker="|", s=300, color=C["ink"], lw=2.5, zorder=4)
    ax.text(2.35, i, f"med {row['median_ratio']:.2f}×  p={row['wilcoxon_p_vs_1']:.0e}",
            va="center", fontsize=7.5, color="#555")
    ax.text(-0.02, i, names.get(k, k), va="center", ha="right", fontsize=8,
            transform=ax.get_yaxis_transform())
ax.axvline(1.0, color=C["bad"], ls="--", lw=1.2)
ax.set_yticks([]); ax.set_xlim(0.9, 3.4); ax.set_xlabel("FC→SC / SC→FC ratio (per seed)")
fig.suptitle("S5 · FC→SC asymmetry survives every reduction axis (all ratios > 1)",
             fontweight="bold", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93)); save(fig, "S5_reduction_axis")

# ================================================================ S6
# FC reliability histogram + achieved-vs-ceiling scatter.
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
g = pd.read_csv(NOISE / "g_per_subject_reliability_Glasser.csv")
ax = axes[0]
ax.hist(g.rel_between_session, bins=40, color=C["FC"], alpha=0.8, edgecolor="white")
ax.axvline(g.rel_between_session.mean(), color=C["ink"], lw=2,
           label=f"mean={g.rel_between_session.mean():.2f}")
ax.set_xlabel("between-session FC reliability"); ax.set_ylabel("subjects")
ax.set_title("FC whole-connectome reliability"); ax.legend()
ax = axes[1]
h = pd.read_csv(NOISE / "h_per_subject_achieved_vs_ceiling.csv")
h = h[h.source == "SC"]
hc = pd.read_csv(NOISE / "h_correlations.csv")
rr = hc[hc.source == "SC"]["pearson_achieved_vs_ceiling"].iloc[0]
ax.scatter(h.ceiling, h.achieved_mean, s=10, color=C["SC"], alpha=0.35)
m, b = np.polyfit(h.ceiling, h.achieved_mean, 1)
xs = np.linspace(h.ceiling.min(), h.ceiling.max(), 50)
ax.plot(xs, m * xs + b, color=C["ink"], lw=1.5)
ax.set_xlabel("subject FC reliability"); ax.set_ylabel("SC→FC achieved")
ax.set_title(f"achieved vs reliability (r={rr:.2f}, n.s.)")
fig.suptitle("S6 · FC is reliable as a whole connectome, yet SC→FC quality is "
             "uncorrelated with it", fontweight="bold", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93)); save(fig, "S6_fc_reliability")

# ================================================================ S7
# nonlinear / scaling nulls.
sc = pd.read_csv(NLIN / "n6_scaling_summary.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
ax = axes[0]
for task, c in [("reconstruction", C["SC"]), ("cognition", C["FC"])]:
    s = sc[(sc.task == task) & (sc.n_seeds >= 5)].sort_values("n_sub")
    ax.plot(s.n_sub, s.median_linear, "-o", color=c, label=f"{task}: linear", ms=4)
    ax.plot(s.n_sub, s.median_final, "--s", color=c, ms=4, alpha=0.6,
            label=f"{task}: nonlinear")
ax.set_xlabel("training subjects"); ax.set_ylabel("score"); ax.legend(fontsize=7)
ax.set_title("linear vs nonlinear by n")
ax = axes[1]
for task, c in [("reconstruction", C["SC"]), ("cognition", C["FC"])]:
    s = sc[(sc.task == task) & (sc.n_seeds >= 5)].sort_values("n_sub")
    ax.plot(s.n_sub, s.median_gap, "-o", color=c, label=task, ms=5)
    ax.fill_between(s.n_sub, s.gap_min, s.gap_max, color=c, alpha=0.12)
ax.axhline(0, color=C["ink"], lw=1); ax.set_xlabel("training subjects")
ax.set_ylabel("nonlinear − linear gap"); ax.legend(); ax.set_title("gap stays ≤ 0")
fig.suptitle("S7 · Nonlinear models and more data do not unlock the missing signal",
             fontweight="bold", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93)); save(fig, "S7_nonlinear_scaling")

# ================================================================ S8
# tractography / source-representation comparison.
ts = pd.read_csv(TRACT / "tractography_synthesis.csv").set_index("row")["value"]
fig, ax = plt.subplots(figsize=(8, 4.2))
feats = [("SC (streamline counts)", "E1: SC -> FC median dp"),
         ("SC + r2t bundles", "E1: SC_r2t -> FC median dp"),
         ("kitchen sink", "E1: kitchen_sink -> FC median dp"),
         ("r2t bundles only", "E1: r2t -> FC median dp"),
         ("r2t correlation", "E1: r2t_corr -> FC median dp")]
vals = [float(ts[k]) for _, k in feats]
cols = [C["SC"], "#5DADE2", "#85C1E9", C["bv"], "#B2BABB"]
ax.barh(range(len(feats)), vals, color=cols, edgecolor="white")
ax.axvline(float(ts["E1: SC -> FC median dp"]), color=C["SC"], ls="--", lw=1)
for i, v in enumerate(vals):
    ax.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=8)
ax.set_yticks(range(len(feats))); ax.set_yticklabels([f for f, _ in feats])
ax.invert_yaxis(); ax.set_xlabel("→ FC median demeaned r"); ax.set_xlim(0, 0.11)
fig.suptitle("S8 · Richer tractography features do not beat streamline-count SC "
             "at predicting FC", fontweight="bold", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93)); save(fig, "S8_tractography")

# ================================================================ S9
# family AUC across variants and relations (both parcellations).
fam = load_family()
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
variants = ["obs_SC", "obs_FC", "pred_SC_resid_bvdemo", "pred_SC_raw",
            "bvdemo_to_SC", "combined_pred_SC", "pred_FC_resid_bvdemo"]
vlab = ["obs SC", "obs FC", "pred SC\n(resid)", "pred SC\n(raw)",
        "bv+demo", "pred SC\n(comb)", "pred FC\n(resid)"]
rels = ["MZ", "DZ", "sibling"]
rcol = {"MZ": "#1A5276", "DZ": "#2E86C1", "sibling": "#85C1E9"}
for ax, parc in zip(axes, PARCS):
    x = np.arange(len(variants)); w = 0.26
    for j, rel in enumerate(rels):
        sub = fam[(fam.parcellation == parc) & (fam.relation == rel)].set_index("variant")
        vals = [sub.loc[v, "auc"] if v in sub.index else np.nan for v in variants]
        lo = [sub.loc[v, "auc"] - sub.loc[v, "auc_lo"] if v in sub.index else 0 for v in variants]
        hi = [sub.loc[v, "auc_hi"] - sub.loc[v, "auc"] if v in sub.index else 0 for v in variants]
        ax.bar(x + (j - 1) * w, vals, w, yerr=[lo, hi], capsize=2,
               color=rcol[rel], edgecolor="white", label=rel)
    ax.axhline(0.5, color=C["bad"], ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(vlab, fontsize=7.5)
    ax.set_title(PARC_LABEL[parc]); ax.set_ylim(0.45, 1.02)
axes[0].set_ylabel("relatedness-separation AUC"); axes[0].legend(title="relation")
fig.suptitle("S9 · Family separation AUC by predictor and relatedness. "
             "Combined predictor collapses for siblings.",
             fontweight="bold", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93)); save(fig, "S9_family_auc")

# ================================================================ S10
# PC mechanism localization & enrichment, Glasser vs 4S456.
pp = pd.read_csv(FAM / "f8_per_pc.csv")
gp = pp.groupby(["parcellation", "pc"]).median(numeric_only=True).reset_index()
enr = pd.read_csv(FAM / "f8_pc3_enrichment_agg.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
ax = axes[0]
sel = {"Glasser": 3, "4S456Parcels": 4}
for parc, mk in zip(PARCS, ["o", "s"]):
    s = gp[gp.parcellation == parc]
    ax.plot(s.pc, s.FC_to_PC_R2, "-", color=C["SC"] if parc == "Glasser" else "#85C1E9",
            marker=mk, label=f"{parc.split('Parcels')[0]} FC→PC R²")
    ax.plot(s.pc, s.AUC_sibling, "--", color=C["pred"] if parc == "Glasser" else "#C39BD3",
            marker=mk, label=f"{parc.split('Parcels')[0]} sibling AUC")
    ax.axvline(sel[parc], color=C["accent"], ls=":", lw=1.5, alpha=0.6)
ax.set_xlabel("principal component"); ax.set_ylabel("value")
ax.set_title("FC-predictability & family AUC per PC\n(gold = selected mode)")
ax.legend(fontsize=6.5, loc="upper right")
ax = axes[1]
top = enr[enr.parcellation == "Glasser"].nlargest(8, "median_enrichment")
ax.barh(range(len(top)), top.median_enrichment, color=C["SC"], edgecolor="white")
ax.set_yticks(range(len(top))); ax.set_yticklabels(top.net_pair, fontsize=7.5)
ax.invert_yaxis(); ax.set_xlabel("median edge enrichment in selected mode")
ax.set_title("Glasser: network-pair enrichment\n(top 8, visual/DAN dominate)")
fig.suptitle("S10 · The selected SC mode is FC-predictable, family-discriminative, "
             "and visual/dorsal-attention localized", fontweight="bold", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.92)); save(fig, "S10_pc_mechanism")

print("\nAll supplement figures in", SUPP)
