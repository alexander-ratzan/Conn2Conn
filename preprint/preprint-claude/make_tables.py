#!/usr/bin/env python3
"""Generate the main tables (1-3) as Markdown + rendered PNG, from source CSVs.

Run:  /Users/user/dev-env/bin/python make_tables.py
Writes tables/tables.md and tables/tableN.png.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from style import load_recon, load_downstream, C, PARCS, FIGDIR

TBL = FIGDIR.parent / "tables"
TBL.mkdir(exist_ok=True)
r = load_recon()
d = load_downstream()


def render_table(df, name, title, colw=None, fs=9):
    nrow, ncol = df.shape
    fig, ax = plt.subplots(figsize=(min(2 + 1.7 * ncol, 14), 0.5 + 0.36 * (nrow + 1)))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center",
                   cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(fs); tbl.scale(1, 1.45)
    for (rr, cc), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D5D8DC")
        if rr == 0:
            cell.set_facecolor(C["ink"]); cell.set_text_props(color="white", fontweight="bold")
        elif rr % 2 == 0:
            cell.set_facecolor("#F4F6F7")
    if colw:
        for cc, wgt in enumerate(colw):
            for rr in range(nrow + 1):
                tbl[(rr, cc)].set_width(wgt)
    ax.set_title(title, fontweight="bold", fontsize=11, pad=12)
    fig.savefig(TBL / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(TBL / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote tables/{name}.png")


def tex_escape(s):
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("#", r"\#"), ("_", r"\_"), ("→", r"$\rightarrow$"),
                 ("×", r"$\times$"), ("±", r"$\pm$"), ("—", "---"),
                 ("~", r"\textasciitilde{}")]:
        s = s.replace(a, b)
    return s


def emit_tex(df, fname, colspec, mono_cols=()):
    """Write a booktabs tabular body to tables/<fname>.tex for \\input."""
    cols = list(df.columns)
    lines = [r"\begin{tabular}{%s}" % colspec, r"\toprule"]
    lines.append(" & ".join(r"\textbf{%s}" % tex_escape(c) for c in cols) + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        cells = []
        for j, v in enumerate(row.values):
            cells.append(tex_escape(v))
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    (TBL / f"{fname}.tex").write_text("\n".join(lines))
    print(f"  wrote tables/{fname}.tex")


md = ["# Main tables\n",
      "_Generated from the reproduction-grid CSVs; see `make_tables.py`._\n"]

# ---------------------------------------------------------------- Table 1
t1 = pd.DataFrame([
    ["Cohort", "HCP Young Adults (FC + SC + demographics + brain-volume)"],
    ["Subjects (per split)", "~878 (683 train / 195 test, family-aware)"],
    ["Splits", "10 frozen, family-aware (siblings/twins kept on one side)"],
    ["Parcellation A", "Glasser — 360 regions, 64,620 edges"],
    ["Parcellation B", "4S456Parcels — 456 regions, 103,740 edges"],
    ["Connectome rep.", "upper-triangle edge vector"],
    ["Primary recon metric", "demeaned_pearson (per-subject, train-mean removed)"],
    ["Secondary metrics", "avg_rank, top1_acc (fingerprint), r2/mse"],
    ["Recon estimator", "PCA(256)→PLS(64)→inverse-PCA (+ BayesianRidge, KernelRidge controls)"],
    ["Downstream estimator", "BayesianRidge (reportable for scalar targets)"],
], columns=["Field", "Value"])
render_table(t1, "table1_dataset", "Table 1 · Dataset, parcellations, and evaluation",
             colw=[0.28, 0.72], fs=8.5)
emit_tex(t1, "table1_body", r"@{}l X@{}".replace("X", "p{0.66\\linewidth}"))

# ---------------------------------------------------------------- Table 2
t2 = pd.DataFrame([
    ["Reconstruction", "FC", "SC", "PCA→PLS", "demeaned_pearson", "cross-modal FC→SC"],
    ["Reconstruction", "SC", "FC", "PCA→PLS", "demeaned_pearson", "cross-modal SC→FC"],
    ["Oracle", "FC", "FC", "BayesianRidge", "demeaned_pearson", "within-modality ceiling"],
    ["Oracle", "SC", "SC", "BayesianRidge", "demeaned_pearson", "within-modality ceiling"],
    ["Baseline", "bv+demo", "SC/FC", "PCA→PLS", "demeaned_pearson", "subject-info → connectome"],
    ["Downstream", "bv+demo", "Cognition", "BayesianRidge", "pearson / lift", "checkpoint baseline"],
    ["Downstream", "obs_FC", "Cognition", "BayesianRidge", "lift_over_bvdemo", "does FC beat baseline?"],
    ["Downstream", "obs_SC", "Cognition", "BayesianRidge", "lift_over_bvdemo", "does SC beat baseline?"],
    ["Downstream", "pred_SC / pred_FC", "Cognition", "BayesianRidge", "lift_over_bvdemo", "does imputation transfer?"],
    ["Leak check", "any", "sex / age", "BayesianRidge", "bal_acc / pearson", "diagnostic, not outcome"],
    ["Family", "pred_SC variants", "MZ/DZ/sib", "—", "AUC", "identity signal survives?"],
], columns=["Task", "Input", "Target", "Estimator", "Metric", "Claim tested"])
render_table(t2, "table2_grid", "Table 2 · Task grid and the claim each row tests",
             colw=[0.13, 0.17, 0.13, 0.15, 0.18, 0.24], fs=8)
emit_tex(t2, "table2_body", r"@{}lllllp{0.24\linewidth}@{}")


def rmean(parc, est, s, t, col="demeaned_pearson"):
    return r[(r.parcellation == parc) & (r.estimator == est) & (r.source == s) &
             (r.target == t) & (~r.is_block)][col].mean()


def dmean(parc, inp, t, col, est="bayesian_ridge"):
    return d[(d.parcellation == parc) & (d.estimator == est) & (d.input_set == inp) &
             (d.target == t)][col].mean()

# ---------------------------------------------------------------- Table 3
def f(x): return f"{x:.3f}"
rows = []
rows.append(["FC→SC reconstruction", f(rmean("Glasser", "pca_pls", "FC", "SC")),
             f(rmean("4S456Parcels", "pca_pls", "FC", "SC")), "PCA→PLS",
             "FC recovers SC deviations"])
rows.append(["SC→FC reconstruction", f(rmean("Glasser", "pca_pls", "SC", "FC")),
             f(rmean("4S456Parcels", "pca_pls", "SC", "FC")), "PCA→PLS",
             "weaker reverse direction"])
rows.append(["FC→SC / SC→FC ratio",
             f(rmean("Glasser", "pca_pls", "FC", "SC") / rmean("Glasser", "pca_pls", "SC", "FC")) + "×",
             f(rmean("4S456Parcels", "pca_pls", "FC", "SC") / rmean("4S456Parcels", "pca_pls", "SC", "FC")) + "×",
             "PCA→PLS", "directional asymmetry"])
rows.append(["FC→FC oracle ceiling", f(rmean("Glasser", "bayesian_ridge", "FC", "FC")),
             f(rmean("4S456Parcels", "bayesian_ridge", "FC", "FC")), "BayesianRidge",
             "self-predictability ~equal"])
rows.append(["SC→SC oracle ceiling", f(rmean("Glasser", "bayesian_ridge", "SC", "SC")),
             f(rmean("4S456Parcels", "bayesian_ridge", "SC", "SC")), "BayesianRidge",
             "self-predictability ~equal"])
rows.append(["bv+demo → CogCryst", f(dmean("Glasser", "bv+demo", "CogCryst", "pearson")),
             f(dmean("4S456Parcels", "bv+demo", "CogCryst", "pearson")), "BayesianRidge",
             "the checkpoint to beat"])
rows.append(["obs FC CogCryst lift", "+" + f(dmean("Glasser", "obs_FC", "CogCryst", "lift_over_bvdemo")),
             "+" + f(dmean("4S456Parcels", "obs_FC", "CogCryst", "lift_over_bvdemo")), "BayesianRidge",
             "FC clears the baseline"])
rows.append(["obs SC CogCryst lift", f(dmean("Glasser", "obs_SC", "CogCryst", "lift_over_bvdemo")),
             f(dmean("4S456Parcels", "obs_SC", "CogCryst", "lift_over_bvdemo")), "BayesianRidge",
             "SC below baseline"])
rows.append(["pred FC CogCryst lift", f(dmean("Glasser", "pred_FC", "CogCryst", "lift_over_bvdemo")),
             f(dmean("4S456Parcels", "pred_FC", "CogCryst", "lift_over_bvdemo")), "BayesianRidge",
             "imputation is harmful"])
t3 = pd.DataFrame(rows, columns=["Finding", "Glasser", "4S456", "Estimator", "Interpretation"])
render_table(t3, "table3_headline", "Table 3 · Headline findings (values read from CSVs)",
             colw=[0.24, 0.12, 0.12, 0.16, 0.36], fs=8)
emit_tex(t3, "table3_body", r"@{}lrrlp{0.34\linewidth}@{}")

# ---------------------------------------------------------------- markdown
def to_md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


for t, name in [(t1, "Table 1 · Dataset and evaluation"),
                (t2, "Table 2 · Task grid and claims"),
                (t3, "Table 3 · Headline findings")]:
    md.append(f"\n## {name}\n")
    md.append(to_md(t))
    md.append("")
(TBL / "tables.md").write_text("\n".join(md))
print("  wrote tables/tables.md")
