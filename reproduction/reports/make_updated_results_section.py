#!/usr/bin/env python3
"""Generate an updated results section from the completed reproduction grid.

Outputs:
  - reproduction/reports/updated_results_section.md
  - output/pdf/Updated_Results_Section.pdf

The text is intentionally concise: it reconciles the June 18 preliminary PDF with
the completed grid and highlights the useful deltas for write-up.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPRO = ROOT / "reproduction"
REPORTS = REPRO / "reports"
OUTPDF = ROOT / "output" / "pdf"
REPORTS.mkdir(parents=True, exist_ok=True)
OUTPDF.mkdir(parents=True, exist_ok=True)

R = pd.read_csv(REPRO / "outputs" / "reconstruction.csv")
D = pd.read_csv(REPRO / "outputs" / "downstream.csv")


def mean(df: pd.DataFrame, col: str) -> float:
    return float(pd.to_numeric(df[col], errors="coerce").mean())


def rec(parc: str, estimator: str, input_set: str, target: str, col: str = "demeaned_pearson") -> float:
    return mean(
        R[
            (R["parcellation"] == parc)
            & (R["estimator"] == estimator)
            & (R["input_set"] == input_set)
            & (R["target"] == target)
        ],
        col,
    )


def down(
    parc: str,
    input_set: str,
    target: str,
    col: str = "pearson",
    estimator: str = "bayesian_ridge",
) -> float:
    return mean(
        D[
            (D["parcellation"] == parc)
            & (D["estimator"] == estimator)
            & (D["input_set"] == input_set)
            & (D["target"] == target)
        ],
        col,
    )


def med_p(parc: str, input_set: str, target: str, estimator: str = "bayesian_ridge") -> float:
    s = D[
        (D["parcellation"] == parc)
        & (D["estimator"] == estimator)
        & (D["input_set"] == input_set)
        & (D["target"] == target)
    ]["lift_perm_p"]
    return float(pd.to_numeric(s, errors="coerce").median())


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def fmt_signed(x: float, nd: int = 3) -> str:
    return f"{x:+.{nd}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


parcs = ["Glasser", "4S456Parcels"]

f1_rows = []
for parc in parcs:
    fcsc = rec(parc, "pca_pls", "FC", "SC")
    scfc = rec(parc, "pca_pls", "SC", "FC")
    f1_rows.append([parc, fmt(fcsc), fmt(scfc), f"{fcsc / scfc:.2f}x"])

f2_rows = []
for parc in parcs:
    f2_rows.append(
        [
            parc,
            fmt(rec(parc, "pca_pls", "bv", "SC")),
            fmt(rec(parc, "pca_pls", "demo", "SC")),
            fmt(rec(parc, "pca_pls", "bv", "FC")),
            fmt(rec(parc, "pca_pls", "demo", "FC")),
            fmt(rec(parc, "pca_pls", "bv+demo", "SC")),
            fmt(rec(parc, "pca_pls", "bv+demo", "FC")),
        ]
    )

oracle_rows = []
for parc in parcs:
    oracle_rows.append(
        [
            parc,
            fmt(rec(parc, "pca_pls", "SC", "SC")),
            fmt(rec(parc, "bayesian_ridge", "SC", "SC")),
            fmt(rec(parc, "kernel_ridge", "SC", "SC")),
            fmt(rec(parc, "bayesian_ridge", "FC", "FC")),
        ]
    )

f5_rows = []
for parc in parcs:
    for inp in ["bv+demo", "obs_FC", "obs_SC", "obs_FC+bv+demo", "obs_SC+bv+demo"]:
        f5_rows.append(
            [
                parc,
                inp,
                fmt(down(parc, inp, "CogTotal")),
                fmt_signed(down(parc, inp, "CogTotal", "lift_over_bvdemo")),
                f"{med_p(parc, inp, 'CogTotal'):.3g}",
                fmt(down(parc, inp, "CogCryst")),
                fmt_signed(down(parc, inp, "CogCryst", "lift_over_bvdemo")),
                f"{med_p(parc, inp, 'CogCryst'):.3g}",
            ]
        )

f3_rows = []
for parc in parcs:
    for target in ["CogTotal", "CogFluid", "CogCryst"]:
        pred_sc = down(parc, "pred_SC", target)
        obs_sc = down(parc, "obs_SC", target)
        pred_fc = down(parc, "pred_FC", target)
        obs_fc = down(parc, "obs_FC", target)
        f3_rows.append(
            [
                parc,
                target,
                f"{pred_sc / obs_sc:.2f}x",
                f"{pred_fc / obs_fc:.2f}x",
                f"{(pred_sc / obs_sc) / (pred_fc / obs_fc):.2f}x",
            ]
        )

survival_rows = []
for parc in parcs:
    for inp in ["obs_FC", "obs_SC", "pred_SC", "pred_FC"]:
        vals = []
        for target in ["CogTotal", "CogFluid", "CogCryst"]:
            vals.append(down(parc, inp, target, "residualized_pearson") / down(parc, inp, target))
        survival_rows.append([parc, inp] + [f"{100 * v:.0f}%" for v in vals])


md_parts = [
    "# Updated Results Section - Full Reproduction Grid",
    "",
    "This section supersedes the June 18 preliminary-results PDF for the confirmatory grid. "
    "It keeps the same scientific story, but uses the completed reproduction grid: 10 seeds, "
    "Glasser plus 4S456Parcels, PCA->PLS / BayesianRidge / KernelRidge, frozen family-aware "
    "splits, leak checks, and CSV-first provenance.",
    "",
    "## Headline",
    "",
    "The core result survives intact: FC->SC prediction is consistently stronger than SC->FC, "
    "subject-info baselines explain much of the apparent cross-modal signal, and observed FC is "
    "the only connectome representation that reliably adds cognition signal above bv+demo. "
    "The full grid adds two important refinements: (1) use BayesianRidge/KernelRidge same-modality "
    "rows for Ceiling B, not the bottlenecked PLS oracle; (2) state estimator next to every "
    "absolute number, because PLS, BayesianRidge, and KernelRidge can move magnitudes even when "
    "directions are stable.",
    "",
    "## F1 - FC->SC > SC->FC",
    md_table(["Parcellation", "FC->SC", "SC->FC", "ratio"], f1_rows),
    "",
    "The asymmetry is now cross-parcellation evidence, not just Glasser reproduction. It clears "
    "the 1.15x gate in every seed on both parcellations.",
    "",
    "## F2 - Anatomy predicts structure; demographics predict function",
    md_table(
        ["Parcellation", "bv->SC", "demo->SC", "bv->FC", "demo->FC", "bv+demo->SC", "bv+demo->FC"],
        f2_rows,
    ),
    "",
    "This is stronger as a 2x2 dissociation than as a single baseline result: anatomy wins for "
    "SC, demographics wins for FC. The 4S456 grid strengthens the structure side.",
    "",
    "## Ceiling B - model oracle reconciliation",
    md_table(["Parcellation", "PLS SC->SC", "BR SC->SC", "KR SC->SC", "BR FC->FC"], oracle_rows),
    "",
    "The preliminary PDF's SC->SC = 0.647 was not contradicted. It matches the completed grid's "
    "BayesianRidge/KernelRidge same-modality oracle. The lower PLS SC->SC number is a bottlenecked "
    "PCA->PLS->inverse-PCA self-map and should not be used as the disattenuation denominator.",
    "",
    "## F3 - imputation inherits source information, but does not beat bv+demo",
    md_table(["Parcellation", "Target", "pred_SC / obs_SC", "pred_FC / obs_FC", "utility asym"], f3_rows),
    "",
    "The imputation asymmetry is stronger in the full grid than the preliminary PDF. However, "
    "the clean wording is now: pred_SC beats observed SC downstream because it inherits FC-like "
    "information, but pred_SC is roughly neutral against bv+demo. Pred_FC is actively harmful "
    "relative to bv+demo.",
    "",
    "## F4 - residual cognition signal after bv+demo",
    md_table(["Parcellation", "Input", "Total", "Fluid", "Cryst"], survival_rows),
    "",
    "Observed FC retains most of its cognition signal after removing bv+demo, especially for "
    "crystallized cognition. Observed SC mostly collapses. Pred_SC retains some FC-derived "
    "residual signal; pred_FC carries little residual functional cognition signal.",
    "",
    "## F5 - cost-benefit baseline",
    md_table(
        ["Parcellation", "Input", "Total r", "Total lift", "Total p", "Cryst r", "Cryst lift", "Cryst p"],
        f5_rows,
    ),
    "",
    "The clean citable claim is: bv+demo is the required baseline; observed FC adds real cognition "
    "signal, strongest for CogCryst; observed SC underperforms the cheap baseline. The most direct "
    "'add fMRI to cheap baseline' comparison is obs_FC+bv+demo, which is significant for CogTotal "
    "and CogCryst on both parcellations in the grid.",
    "",
    "## New grid takeaways worth exploring",
    "",
    "- The asymmetry is not a target-reliability artifact: BR/KR FC->FC and SC->SC oracles are nearly matched, while cross-modal FC->SC still beats SC->FC by about 1.6-1.7x.",
    "- 4S456 specifically boosts ->SC prediction, especially bv->SC and bv+demo->SC, while ->FC changes little or drops slightly. This looks like finer parcellation adding structural detail rather than generic metric noise.",
    "- SC can be actively counterproductive downstream: obs_SC and obs_SC+bv+demo stay below bv+demo for cognition. That is a stronger message than 'SC adds little.'",
    "- pred_FC is worse than useless for cognition in the grid, consistently below bv+demo and below obs_FC. That sharpens the non-substitutability result.",
    "- The PC3/visual-DAN mechanism remains the main non-confirmatory piece. F1-F5 now replicate across parcellations; F8 still needs a 4S456 mechanism pass before it can be phrased as more than hypothesis-generating.",
    "",
    "## Suggested write-up posture",
    "",
    "Lead with the baseline and ceiling logic rather than biomarker language. The field story is: "
    "simple deterministic models reproduce the FC->SC asymmetry, but subject-level anatomy and "
    "demographics explain a large fraction of apparent connectome signal; observed FC is the only "
    "modality with reliable cognition lift above that floor; richer SC, imputed FC, and nonlinear "
    "models do not rescue the cognition claim in HCP-YA.",
    "",
]

md_text = "\n".join(md_parts)
(REPORTS / "updated_results_section.md").write_text(md_text)


def add_wrapped_text(fig, x: float, y: float, text: str, size: int = 10, weight: str = "normal", width: int = 92, line_gap: float = 0.025) -> float:
    for line in wrap(text, width=width) or [""]:
        fig.text(x, y, line, fontsize=size, fontweight=weight, va="top", ha="left")
        y -= line_gap
    return y


def add_table_page(pdf: PdfPages, title: str, intro: str, headers: list[str], rows: list[list[str]], font_size: int = 7) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.07, 0.95, title, fontsize=17, fontweight="bold", color="#17365d", va="top")
    y = add_wrapped_text(fig, 0.07, 0.905, intro, size=9, width=98, line_gap=0.023)
    ax = fig.add_axes([0.05, 0.08, 0.90, max(0.25, y - 0.10)])
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, loc="upper left", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#b0b7c3")
        cell.set_linewidth(0.45)
        if r == 0:
            cell.set_facecolor("#203864")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f3f6fa")
    fig.text(0.07, 0.035, "Generated from reproduction/outputs/*.csv", fontsize=7, color="#666666")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_bullet_page(pdf: PdfPages, title: str, paragraphs: list[str], bullets: list[str]) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.07, 0.95, title, fontsize=18, fontweight="bold", color="#17365d", va="top")
    y = 0.90
    for paragraph in paragraphs:
        y = add_wrapped_text(fig, 0.07, y, paragraph, size=10, width=96, line_gap=0.026)
        y -= 0.018
    for bullet in bullets:
        y = add_wrapped_text(fig, 0.09, y, "- " + bullet, size=9.5, width=94, line_gap=0.024)
        y -= 0.012
    fig.text(0.07, 0.035, "Updated results section - completed reproduction grid", fontsize=7, color="#666666")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


pdf_path = OUTPDF / "Updated_Results_Section.pdf"
with PdfPages(pdf_path) as pdf:
    add_bullet_page(
        pdf,
        "Updated Results Section",
        [
            "Full-grid reconciliation of the June 18 preliminary PDF. The main story survives: FC->SC is stronger than SC->FC, bv+demo is the required baseline, and observed FC is the only connectome representation that reliably adds cognition signal above that floor.",
            "Two reporting fixes matter: use BayesianRidge/KernelRidge same-modality rows for Ceiling B, and state the estimator next to every absolute number.",
        ],
        [
            "F1 replicates across Glasser and 4S456: FC->SC / SC->FC is 1.61x and 1.68x.",
            "F2 becomes a clean double dissociation: anatomy predicts SC, demographics predicts FC.",
            "F5 is clearest as an add-on test: obs_FC+bv+demo beats bv+demo for CogTotal and CogCryst; observed SC remains below the cheap baseline.",
            "F8 remains hypothesis-generating until the PC3 mechanism is checked on 4S456.",
        ],
    )
    add_table_page(
        pdf,
        "F1 and F2 - Reconstruction Spine",
        "PCA->PLS reconstruction cells. Values are mean demeaned Pearson over 10 frozen seeds.",
        ["Parcellation", "FC->SC", "SC->FC", "ratio"],
        f1_rows,
        font_size=8,
    )
    add_table_page(
        pdf,
        "F2 - Modality Dissociation",
        "Anatomy wins for structural targets; demographics wins for functional targets.",
        ["Parcellation", "bv->SC", "demo->SC", "bv->FC", "demo->FC", "bv+demo->SC", "bv+demo->FC"],
        f2_rows,
        font_size=7,
    )
    add_table_page(
        pdf,
        "Ceiling B - Oracle Reconciliation",
        "The preliminary SC->SC = 0.647 number matches BR/KR same-modality oracles, not bottlenecked PLS.",
        ["Parcellation", "PLS SC->SC", "BR SC->SC", "KR SC->SC", "BR FC->FC"],
        oracle_rows,
        font_size=8,
    )
    add_table_page(
        pdf,
        "F3 - Downstream Imputation Asymmetry",
        "BayesianRidge cognition rows. pred_SC improves over observed SC; pred_FC loses much of observed FC's cognition signal.",
        ["Parcellation", "Target", "pred_SC / obs_SC", "pred_FC / obs_FC", "utility asym"],
        f3_rows,
        font_size=7,
    )
    add_table_page(
        pdf,
        "F4 - Fraction Surviving bv+demo Removal",
        "Residualized cognition signal divided by raw cognition score. FC survives; SC mostly collapses.",
        ["Parcellation", "Input", "Total", "Fluid", "Cryst"],
        survival_rows,
        font_size=7,
    )
    add_table_page(
        pdf,
        "F5 - Cost-Benefit Baseline",
        "BayesianRidge downstream cognition. Lift is Pearson r minus the bv+demo baseline; p is the median paired-permutation p across seeds.",
        ["Parcellation", "Input", "Total r", "Total lift", "Total p", "Cryst r", "Cryst lift", "Cryst p"],
        f5_rows,
        font_size=5.8,
    )
    add_bullet_page(
        pdf,
        "New Takeaways and Write-up Posture",
        [
            "The completed grid is most useful because it turns the preliminary Glasser story into a cross-parcellation confirmatory spine, while cleanly separating confirmed findings from mechanism hypotheses.",
        ],
        [
            "The asymmetry is not a target-reliability artifact: same-modality FC and SC oracles are closely matched under BR/KR.",
            "4S456 specifically boosts ->SC prediction, suggesting finer structural detail rather than generic metric inflation.",
            "SC can be counterproductive downstream; obs_SC+bv+demo stays below bv+demo.",
            "pred_FC is actively harmful for cognition relative to bv+demo, sharpening non-substitutability.",
            "Use modest-effect language throughout: this is a methods and redirect paper, not a diagnostic biomarker paper.",
        ],
    )

print(f"wrote {REPORTS / 'updated_results_section.md'}")
print(f"wrote {pdf_path}")
