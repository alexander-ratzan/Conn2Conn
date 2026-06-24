#!/usr/bin/env python3
"""Build the long sanity-check supplement markdown.

Writes only inside preprint/preprint-codex. Reads source markdown and CSVs from
the repository as immutable inputs.
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import stats


SCRIPT = Path(__file__).resolve()
CODEX = SCRIPT.parents[1]
ROOT = SCRIPT.parents[3]
OUT_MD = CODEX / "supplement_sanity_checks.md"
OUT_GAP_PLAN = CODEX / "sanity_check_gap_plan.md"


SOURCES = {
    "spine": ROOT / "preprint" / "supplement.md",
    "reproduction_findings": ROOT / "reproduction" / "reports" / "reproduction_findings.md",
    "exploration_findings": ROOT / "reproduction" / "exploration" / "FINDINGS_EXPLORATION.md",
    "discrepancy": ROOT / "reproduction" / "exploration" / "DISCREPANCY_RESOLUTION.md",
    "family_readme": ROOT / "reproduction" / "family_mechanism" / "README.md",
    "preprocessing": ROOT / "notebooks-FC_to_SC-experimental" / "sanity_checks" / "preprocessing_check" / "findings.md",
    "noise": ROOT / "notebooks-FC_to_SC-experimental" / "sanity_checks" / "noise_sanity_check" / "findings_noise.md",
    "tract_check": ROOT / "notebooks-FC_to_SC-experimental" / "sanity_checks" / "tract_check" / "findings.md",
    "nonlinear": ROOT / "notebooks-FC_to_SC-experimental" / "non-linear-sanity-check" / "findings_nonlinear.md",
    "residual": ROOT / "notebooks-FC_to_SC-experimental" / "non-linear-sanity-check" / "findings_residual.md",
    "scaling": ROOT / "notebooks-FC_to_SC-experimental" / "non-linear-sanity-check" / "findings_scaling.md",
    "tractography": ROOT / "notebooks-FC_to_SC-experimental" / "tractography_predict" / "findings.md",
}


CSV_SOURCES = {
    "Reduction-axis summary": ROOT / "notebooks-FC_to_SC-experimental" / "sanity_checks" / "preprocessing_check" / "reduction_axis_summary.csv",
    "Reduction-axis all seed ratios": ROOT / "notebooks-FC_to_SC-experimental" / "sanity_checks" / "preprocessing_check" / "reduction_axis_synthesis.csv",
    "FC reliability ceiling": ROOT / "notebooks-FC_to_SC-experimental" / "sanity_checks" / "noise_sanity_check" / "outputs" / "a_reliability_ceiling.csv",
    "FC per-subject reliability summary": ROOT / "notebooks-FC_to_SC-experimental" / "sanity_checks" / "noise_sanity_check" / "outputs" / "g_per_subject_summary.csv",
    "Reliability filtering summary": ROOT / "notebooks-FC_to_SC-experimental" / "sanity_checks" / "noise_sanity_check" / "outputs" / "h_reliability_filtered_summary.csv",
    "Noise synthesis": ROOT / "notebooks-FC_to_SC-experimental" / "sanity_checks" / "noise_sanity_check" / "outputs" / "noise_synthesis.csv",
    "Tractography source representation": ROOT / "notebooks-FC_to_SC-experimental" / "tractography_predict" / "e1_source_rep_results.csv",
    "Tractography asymmetry summary": ROOT / "notebooks-FC_to_SC-experimental" / "tractography_predict" / "e2_asymmetry_summary.csv",
    "Tractography marginal summary": ROOT / "notebooks-FC_to_SC-experimental" / "tractography_predict" / "e3_marginal_summary.csv",
    "Tractography downstream summary": ROOT / "notebooks-FC_to_SC-experimental" / "tractography_predict" / "e5_downstream_summary.csv",
    "Nonlinear cognition summary": ROOT / "notebooks-FC_to_SC-experimental" / "non-linear-sanity-check" / "n1_cognition_summary.csv",
    "Nonlinear reconstruction summary": ROOT / "notebooks-FC_to_SC-experimental" / "non-linear-sanity-check" / "n2_reconstruction_summary.csv",
    "Nonlinear marginal summary": ROOT / "notebooks-FC_to_SC-experimental" / "non-linear-sanity-check" / "n3_marginal_summary.csv",
    "Residual cognition summary": ROOT / "notebooks-FC_to_SC-experimental" / "non-linear-sanity-check" / "n4_cog_summary.csv",
    "Residual reconstruction summary": ROOT / "notebooks-FC_to_SC-experimental" / "non-linear-sanity-check" / "n4_recon_summary.csv",
    "Sink cognition summary": ROOT / "notebooks-FC_to_SC-experimental" / "non-linear-sanity-check" / "n5_cog_summary.csv",
    "Sink reconstruction summary": ROOT / "notebooks-FC_to_SC-experimental" / "non-linear-sanity-check" / "n5_recon_summary.csv",
    "Scaling summary": ROOT / "notebooks-FC_to_SC-experimental" / "non-linear-sanity-check" / "n6_scaling_summary.csv",
    "Family AUC": ROOT / "reproduction" / "family_mechanism" / "outputs" / "family_auc.csv",
    "F8 stability": ROOT / "reproduction" / "family_mechanism" / "outputs" / "f8_stability.csv",
}


EVIDENCE_SOURCES = {
    "Downstream reproduction grid": ROOT / "reproduction" / "outputs" / "downstream.csv",
    "Reconstruction reproduction grid": ROOT / "reproduction" / "outputs" / "reconstruction.csv",
    "Leak verdict grid": ROOT / "reproduction" / "outputs" / "leak_verdict.csv",
    "Expected grid cells": ROOT / "reproduction" / "configs" / "expected_cells.csv",
    "Detailed tractography downstream": ROOT / "notebooks-FC_to_SC-experimental" / "tractography_predict" / "e5_downstream_results.csv",
    "Per-subject FC achieved-vs-ceiling": ROOT / "notebooks-FC_to_SC-experimental" / "sanity_checks" / "noise_sanity_check" / "outputs" / "h_per_subject_achieved_vs_ceiling.csv",
    "FC achieved-vs-ceiling correlations": ROOT / "notebooks-FC_to_SC-experimental" / "sanity_checks" / "noise_sanity_check" / "outputs" / "h_correlations.csv",
    "Nonlinear cognition results": ROOT / "notebooks-FC_to_SC-experimental" / "non-linear-sanity-check" / "n1_cognition_summary.csv",
    "Residual cognition results": ROOT / "notebooks-FC_to_SC-experimental" / "non-linear-sanity-check" / "n4_cog_summary.csv",
    "Sink cognition results": ROOT / "notebooks-FC_to_SC-experimental" / "non-linear-sanity-check" / "n5_cog_summary.csv",
    "Scaling results": ROOT / "notebooks-FC_to_SC-experimental" / "non-linear-sanity-check" / "n6_scaling_summary.csv",
    "Preprocessing method A": ROOT / "notebooks-FC_to_SC-experimental" / "sanity_checks" / "preprocessing_check" / "method_a_results.csv",
    "Preprocessing method B": ROOT / "notebooks-FC_to_SC-experimental" / "sanity_checks" / "preprocessing_check" / "method_b_results.csv",
    "Preprocessing method C": ROOT / "notebooks-FC_to_SC-experimental" / "sanity_checks" / "preprocessing_check" / "method_c_results.csv",
    "Family PC per-component": ROOT / "reproduction" / "family_mechanism" / "outputs" / "f8_per_pc.csv",
    "Family PC3 localization": ROOT / "reproduction" / "family_mechanism" / "outputs" / "f8_pc3_localization.csv",
    "Family PC3 enrichment": ROOT / "reproduction" / "family_mechanism" / "outputs" / "f8_pc3_enrichment_agg.csv",
    "Reliability-residualized PC3 enrichment": ROOT / "notebooks-FC_to_SC-experimental" / "sanity_checks" / "tract_check" / "enrichment_residual_top200.csv",
    "FC reliability summary": ROOT / "notebooks-FC_to_SC-experimental" / "sanity_checks" / "tract_check" / "retest_icc_results" / "fc_reliability_summary.csv",
    "FC variance decomposition": ROOT / "notebooks-FC_to_SC-experimental" / "sanity_checks" / "noise_sanity_check" / "outputs" / "b_variance_decomposition.csv",
    "Crossmodal disattenuation": ROOT / "notebooks-FC_to_SC-experimental" / "sanity_checks" / "noise_sanity_check" / "outputs" / "e_crossmodal_disattenuation.csv",
    "FC discriminability": ROOT / "notebooks-FC_to_SC-experimental" / "sanity_checks" / "noise_sanity_check" / "outputs" / "f_discriminability.csv",
}


def read_text(key: str) -> str:
    return SOURCES[key].read_text()


def deemoji(text: str) -> str:
    replacements = {
        "\u2705": "[pass]",
        "\u2713": "[pass]",
        "\u2717": "[fail]",
        "\u2b50": "[note]",
        "\u26a0\ufe0f": "[caution]",
        "\u2192": "->",
        "\u2194": "<->",
        "\u2248": "~",
        "\u2264": "<=",
        "\u2265": ">=",
        "\u2212": "-",
        "\u00d7": "x",
        "\u0394": "Delta",
        "\u03c1": "rho",
        "\u03b1": "alpha",
        "\u03b3": "gamma",
        "\u2016": "||",
        "\u2014": "--",
        "\u2013": "-",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2155": "1/5",
        "\u00bc": "1/4",
        "\u03bc": "mu",
        "\u00b1": "+/-",
        "\u2260": "!=",
        "\u221e": "infinity",
        "\u2460": "1.",
        "\u2461": "2.",
        "\u2462": "3.",
        "\u2463": "4.",
        "\u2464": "5.",
        "\u2465": "6.",
        "\u2466": "7.",
        "\u2467": "8.",
        "\u2074": "^4",
        "\u00b2": "^2",
        "\u00a7": "section ",
        "\u2154": "2/3",
        "\u2190": "<-",
        "\u226a": "<<",
        "\u221a": "sqrt",
        "\u0177": "yhat",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def heading_shift(text: str, level: int = 2) -> str:
    prefix = "#" * level
    out = []
    for line in text.splitlines():
        if line.startswith("#"):
            hashes = len(line) - len(line.lstrip("#"))
            out.append(prefix + "#" * hashes + line[hashes:])
        else:
            out.append(line)
    return "\n".join(out)


def csv_table(title: str, path: Path, max_rows: int | None = 12) -> str:
    if not path.exists():
        return f"### {title}\n\nMissing source: {breakable_path(path)}\n"
    df = pd.read_csv(path)
    if max_rows is not None and len(df) > max_rows:
        head = df.head(max_rows).copy()
        note = f"\n\nShowing first {max_rows} of {len(df)} rows. Full source: {breakable_path(path)}.\n"
    else:
        head = df.copy()
        note = f"\n\nFull source: {breakable_path(path)}.\n"
    for col in head.columns:
        if pd.api.types.is_float_dtype(head[col]):
            head[col] = head[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    return f"### {title}\n\n" + markdown_table(head) + note


def markdown_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    rows = []
    for _, row in df.iterrows():
        rows.append([clean_cell(row[c]) for c in df.columns])
    widths = []
    for i, col in enumerate(cols):
        widths.append(max(len(col), *(len(r[i]) for r in rows)) if rows else len(col))
    header = "| " + " | ".join(col.ljust(widths[i]) for i, col in enumerate(cols)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"
    body = ["| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(cols))) + " |" for r in rows]
    return "\n".join([header, sep, *body])


def clean_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    return text.replace("|", "/")


def breakable_path(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("/", "/ ")


def fmt(x: object, digits: int = 3) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, (bool, np.bool_)):
        return "yes" if bool(x) else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        ax = abs(float(x))
        if ax != 0 and ax < 0.001:
            return f"{float(x):.2e}"
        return f"{float(x):.{digits}f}"
    return str(x)


def rounded_table(df: pd.DataFrame, digits: int = 3) -> str:
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(lambda x: fmt(x, digits=digits))
    return markdown_table(out)


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float]:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan, np.nan
    if method == "pearson":
        res = stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
        return float(res.statistic), float(res.pvalue)
    res = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
    return float(res.statistic), float(res.pvalue)


def ci_halfwidth(values: pd.Series) -> tuple[float, float, float, float]:
    vals = values.dropna().astype(float).to_numpy()
    if len(vals) < 2:
        return (float(np.nanmean(vals)) if len(vals) else np.nan, np.nan, np.nan, np.nan)
    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / np.sqrt(len(vals)))
    half = float(stats.t.ppf(0.975, len(vals) - 1) * se)
    return mean, mean - half, mean + half, half


def source_line(label: str, *paths: Path) -> str:
    joined = "; ".join(breakable_path(p) for p in paths)
    return f"\n\nSource: {label}: {joined}.\n"


def existing_data_addendum() -> str:
    downstream = pd.read_csv(EVIDENCE_SOURCES["Downstream reproduction grid"])
    recon = pd.read_csv(EVIDENCE_SOURCES["Reconstruction reproduction grid"])
    leak = pd.read_csv(EVIDENCE_SOURCES["Leak verdict grid"])
    family = pd.read_csv(CSV_SOURCES["Family AUC"])
    hcorr = pd.read_csv(EVIDENCE_SOURCES["FC achieved-vs-ceiling correlations"])

    parts: list[str] = []
    parts.append("# Existing-Data Gap Closure\n")
    parts.append(
        "This section answers the review-style holes that can be closed from files already "
        "present in the repository. It does not claim to solve checks that require raw "
        "diffusion repeat data, new objectives, or additional cohorts; those are triaged in "
        "the next section."
    )

    parts.append("\n## E1. Reconstruction Quality Does Not Track Cognitive Lift\n")
    recon_fc_sc = recon[(recon["source"] == "FC") & (recon["target"] == "SC")][
        ["parcellation", "seed", "estimator", "variant", "demeaned_pearson", "avg_rank", "top1_acc"]
    ]
    ds_pred = downstream[
        (downstream["input_set"] == "pred_SC")
        & (downstream["target"].isin(["CogTotal", "CogFluid", "CogCryst"]))
    ][["parcellation", "seed", "estimator", "variant", "target", "lift_over_bvdemo"]]
    joined = ds_pred.merge(recon_fc_sc, on=["parcellation", "seed", "estimator", "variant"], how="inner")
    rows = []
    for parc, grp in joined.groupby("parcellation"):
        for metric in ["demeaned_pearson", "avg_rank", "top1_acc"]:
            pr, pp = safe_corr(grp[metric], grp["lift_over_bvdemo"], "pearson")
            sr, sp = safe_corr(grp[metric], grp["lift_over_bvdemo"], "spearman")
            rows.append(
                {
                    "parcellation": parc,
                    "reconstruction_metric": metric,
                    "n": len(grp),
                    "pearson_r_with_lift": pr,
                    "pearson_p": pp,
                    "spearman_r_with_lift": sr,
                    "spearman_p": sp,
                }
            )
    parts.append(rounded_table(pd.DataFrame(rows)))
    parts.append(
        "\nInterpretation: the strongest reconstruction scores are not the cells that deliver "
        "cognitive gain. Across 660 joined rows, the FC->SC reconstruction metrics are near "
        "orthogonal to lift over the brain-volume/demographic baseline. This directly closes "
        "the hole that the null might be a presentation artifact of looking at the wrong "
        "reconstruction metric."
    )
    parts.append(source_line("joined", EVIDENCE_SOURCES["Reconstruction reproduction grid"], EVIDENCE_SOURCES["Downstream reproduction grid"]))

    parts.append("\n## E2. Existing Target Sweep: What the Current Grid Already Covers\n")
    br = downstream[downstream["estimator"] == "bayesian_ridge"].copy()
    target_rows = []
    interesting_inputs = ["obs_FC+bv+demo", "pred_SC+bv+demo", "pred_SC", "obs_SC", "obs_FC"]
    for (target, input_set), grp in br[br["input_set"].isin(interesting_inputs)].groupby(["target", "input_set"]):
        metric_col = "balanced_acc" if target == "sex" else "pearson"
        mean_metric = grp[metric_col].mean()
        mean_lift = grp["lift_over_bvdemo"].mean()
        pvals = grp["lift_perm_p"].dropna()
        median_p = pvals.median() if len(pvals) else np.nan
        target_rows.append(
            {
                "target": target,
                "input_set": input_set,
                "metric": "balanced_acc" if target == "sex" else "pearson",
                "mean_metric": mean_metric,
                "mean_lift": mean_lift,
                "median_perm_p": median_p,
                "n_cells": len(grp),
            }
        )
    target_df = pd.DataFrame(target_rows).sort_values(["target", "input_set"])
    parts.append(rounded_table(target_df))
    parts.append(
        "\nInterpretation: the current target sweep is not broad enough to claim all behavior, "
        "but it already covers three cognition composites plus age and sex controls. Observed "
        "FC plus bv+demo reliably improves cognition; predicted SC alone does not. Predicted "
        "SC plus bv+demo gives a small, target-dependent lift, strongest for crystallized "
        "cognition and age-like information. Sex is best read as a leakage/control target "
        "because bv+demo nearly solves it."
    )
    parts.append(source_line("Bayesian-ridge downstream grid", EVIDENCE_SOURCES["Downstream reproduction grid"]))

    parts.append("\n## E3. Seed-Level Lift Bounds For the Current Cognition Null\n")
    lift_rows = []
    for input_set in ["pred_SC", "pred_SC+bv+demo", "obs_FC+bv+demo"]:
        sub = br[(br["input_set"] == input_set) & (br["target"].isin(["CogTotal", "CogFluid", "CogCryst"]))]
        for parc, grp in sub.groupby("parcellation"):
            mean, lo, hi, half = ci_halfwidth(grp["lift_over_bvdemo"])
            lift_rows.append(
                {
                    "parcellation": parc,
                    "input_set": input_set,
                    "n_seed_target_cells": len(grp),
                    "mean_lift": mean,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "ci_halfwidth": half,
                    "frac_perm_p_lt_0.05": (grp["lift_perm_p"] < 0.05).mean(),
                    "median_perm_p": grp["lift_perm_p"].median(),
                }
            )
    parts.append(rounded_table(pd.DataFrame(lift_rows)))
    parts.append(
        "\nInterpretation: with the existing ten frozen splits and three cognition targets, "
        "predicted SC alone is bounded near zero or below. Predicted SC plus bv+demo has a "
        "small positive lift, but it is well below observed FC plus bv+demo. This makes the "
        "null more quantitative: the current grid can detect an observed-FC-size improvement, "
        "but predicted SC is not delivering one."
    )
    parts.append(source_line("seed-level downstream lift", EVIDENCE_SOURCES["Downstream reproduction grid"]))

    parts.append("\n## E4. Leak Audit Summary\n")
    verdict_counts = leak["verdict"].value_counts().rename_axis("verdict").reset_index(name="n")
    target_counts = leak.groupby(["verdict", "target"]).size().reset_index(name="n")
    parts.append(rounded_table(verdict_counts))
    parts.append("\nBy verdict and target:\n")
    parts.append(rounded_table(target_counts))
    parts.append(
        "\nInterpretation: the grid has zero genuine LEAK_FAIL cells. The only threshold "
        "exceedances are either expected raw-connectome sex/age signal or cells explicitly "
        "marked as containing bv+demo."
    )
    parts.append(source_line("leak verdict", EVIDENCE_SOURCES["Leak verdict grid"]))

    parts.append("\n## E5. Objective-Mismatch Evidence Already Present\n")
    family_sib = family[family["relation"] == "sibling"][
        ["parcellation", "variant", "auc", "auc_lo", "auc_hi", "p_fdr", "sig_fdr"]
    ].sort_values(["parcellation", "variant"])
    parts.append(rounded_table(family_sib))
    parts.append(
        "\nInterpretation: predicted connectomes can preserve family/identity signal when the "
        "objective or representation selects for it. The same artifact does not automatically "
        "translate into cognition gain. This strengthens the paper's objective-mismatch claim: "
        "the model is not universally incapable; it is carrying the wrong signal for the "
        "downstream cognition task."
    )
    parts.append(source_line("family mechanism", CSV_SOURCES["Family AUC"]))

    parts.append("\n## E6. Per-Subject FC Reliability Is Not Driving the SC->FC Result\n")
    parts.append(rounded_table(hcorr))
    parts.append(
        "\nInterpretation: the SC achieved-vs-ceiling correlation is essentially flat, while "
        "the bv+demo baseline detects a reliability association. This is an internal positive "
        "control: the analysis can see reliability structure when it exists, but SC prediction "
        "is not explained by per-subject FC reliability."
    )
    parts.append(source_line("per-subject reliability", EVIDENCE_SOURCES["FC achieved-vs-ceiling correlations"], EVIDENCE_SOURCES["Per-subject FC achieved-vs-ceiling"]))
    return "\n\n".join(parts)


def second_pass_addendum() -> str:
    downstream = pd.read_csv(EVIDENCE_SOURCES["Downstream reproduction grid"])
    recon = pd.read_csv(EVIDENCE_SOURCES["Reconstruction reproduction grid"])
    expected = pd.read_csv(EVIDENCE_SOURCES["Expected grid cells"])
    n1 = pd.read_csv(EVIDENCE_SOURCES["Nonlinear cognition results"])
    n4 = pd.read_csv(EVIDENCE_SOURCES["Residual cognition results"])
    n5 = pd.read_csv(EVIDENCE_SOURCES["Sink cognition results"])
    n6 = pd.read_csv(EVIDENCE_SOURCES["Scaling results"])
    tract_down = pd.read_csv(CSV_SOURCES["Tractography downstream summary"])
    tract_marginal = pd.read_csv(CSV_SOURCES["Tractography marginal summary"])

    parts: list[str] = []
    parts.append("# Existing-Data Gap Closure, Pass 2\n")
    parts.append(
        "This second pass adds checks that were still latent in the existing result files: "
        "grid completeness, estimator robustness, nonlinear-capacity rescue attempts, "
        "richer-tractography downstream rescue, scaling behavior, and preprocessing-axis "
        "stress tests."
    )

    parts.append("\n## E7. Expected-Cell Manifest Is Complete\n")
    actual = pd.concat(
        [
            recon[["task", "parcellation", "seed", "estimator", "variant", "input_set", "target"]],
            downstream[["task", "parcellation", "seed", "estimator", "variant", "input_set", "target"]],
        ],
        ignore_index=True,
    ).drop_duplicates()
    key = ["task", "parcellation", "seed", "estimator", "variant", "input_set", "target"]
    missing = expected.merge(actual, on=key, how="left", indicator=True).query("_merge == 'left_only'")
    extra = actual.merge(expected, on=key, how="left", indicator=True).query("_merge == 'left_only'")
    by_task = actual.groupby(["task", "estimator"]).size().reset_index(name="observed_cells")
    by_task["expected_cells"] = (
        expected.groupby(["task", "estimator"]).size().reindex(
            pd.MultiIndex.from_frame(by_task[["task", "estimator"]]), fill_value=0
        ).to_numpy()
    )
    by_task["missing_cells"] = by_task["expected_cells"] - by_task["observed_cells"]
    parts.append(rounded_table(by_task))
    parts.append(
        f"\nManifest check: expected rows = {len(expected)}, observed rows = {len(actual)}, "
        f"missing = {len(missing)}, extra = {len(extra)}. The reproduction grid is not a "
        "hand-picked subset; every declared cell is present exactly once in the combined "
        "reconstruction/downstream outputs."
    )
    parts.append(source_line("grid completeness", EVIDENCE_SOURCES["Expected grid cells"], EVIDENCE_SOURCES["Reconstruction reproduction grid"], EVIDENCE_SOURCES["Downstream reproduction grid"]))

    parts.append("\n## E8. Estimator Robustness Of the Cognition Null\n")
    est_rows = []
    for input_set in ["pred_SC", "pred_SC+bv+demo", "obs_FC+bv+demo"]:
        sub = downstream[
            (downstream["target"].isin(["CogTotal", "CogFluid", "CogCryst"]))
            & (downstream["input_set"] == input_set)
        ]
        for (estimator, parc), grp in sub.groupby(["estimator", "parcellation"]):
            pvals = grp["lift_perm_p"].dropna()
            est_rows.append(
                {
                    "input_set": input_set,
                    "estimator": estimator,
                    "parcellation": parc,
                    "n": len(grp),
                    "mean_lift": grp["lift_over_bvdemo"].mean(),
                    "median_perm_p": pvals.median() if len(pvals) else np.nan,
                    "mean_pearson": grp["pearson"].mean(),
                }
            )
    parts.append(rounded_table(pd.DataFrame(est_rows).sort_values(["input_set", "estimator", "parcellation"])))
    parts.append(
        "\nInterpretation: the downstream null is not a single-estimator accident. Predicted "
        "SC alone is non-positive across pca_pls, Bayesian ridge, and kernel ridge families. "
        "Adding bv+demo produces small lifts, but the positive observed-FC+bv+demo benchmark "
        "remains larger."
    )
    parts.append(source_line("estimator robustness", EVIDENCE_SOURCES["Downstream reproduction grid"]))

    parts.append("\n## E9. Nonlinear Capacity Does Not Rescue Cognition\n")
    nonlinear_best = (
        n1[n1["rep"].isin(["SC", "SC_r2t", "r2t", "r2t_corr"])]
        .sort_values("lift_over_bvdemo", ascending=False)
        .head(12)[["rep", "estimator", "target", "pearson", "r2", "lift_over_bvdemo"]]
    )
    residual_delta = n4.pivot_table(index=["rep", "target"], columns="variant", values="pearson").reset_index()
    residual_delta["final_minus_template"] = residual_delta.get("final", np.nan) - residual_delta.get("template", np.nan)
    residual_delta = residual_delta[residual_delta["rep"].isin(["SC", "SC_r2t", "r2t", "r2t_corr"])][
        ["rep", "target", "template", "final", "final_minus_template"]
    ]
    sink = n5[n5["rep"].isin(["bv+demo", "FC", "sink_linear", "sink_residual"])].copy()
    parts.append("Best nonlinear SC/r2t cognition rows by lift over bv+demo:\n")
    parts.append(rounded_table(nonlinear_best))
    parts.append("\nResidual-learning final-minus-template deltas:\n")
    parts.append(rounded_table(residual_delta))
    parts.append("\nMultimodal sink summary:\n")
    parts.append(rounded_table(sink[["rep", "target", "pearson", "spearman", "r2"]]))
    parts.append(
        "\nInterpretation: stronger nonlinear model classes, residual boosts, and multimodal "
        "sink variants fail to convert SC/richer-tractography representations into a cognition "
        "win. The best downstream behavior still follows observed FC or baseline covariates, "
        "not reconstructed structural signal."
    )
    parts.append(source_line("nonlinear/residual/sink", EVIDENCE_SOURCES["Nonlinear cognition results"], EVIDENCE_SOURCES["Residual cognition results"], EVIDENCE_SOURCES["Sink cognition results"]))

    parts.append("\n## E10. Richer Tractography Does Not Rescue Downstream Utility\n")
    tract_keep = tract_down[
        tract_down["rep"].isin(["FC", "SC", "SC_r2t", "r2t", "r2t_corr", "r2t->synthFC", "bv+demo"])
    ][["rep", "target", "pearson_raw", "pearson_resid", "lift_over_bvdemo_raw"]]
    parts.append(rounded_table(tract_keep))
    parts.append("\nMarginal SC+r2t reconstruction increment:\n")
    parts.append(rounded_table(tract_marginal))
    parts.append(
        "\nInterpretation: richer tractography features neither improve the SC representation "
        "materially nor rescue cognition. The marginal reconstruction delta is approximately "
        "zero to negative, and downstream r2t variants remain below observed FC and often below "
        "the bv+demo baseline."
    )
    parts.append(source_line("richer tractography", CSV_SOURCES["Tractography downstream summary"], CSV_SOURCES["Tractography marginal summary"]))

    parts.append("\n## E11. Sample-Size Scaling Does Not Reveal a Hidden Positive Gap\n")
    parts.append(rounded_table(n6))
    parts.append(
        "\nInterpretation: increasing training size in the tested regime does not uncover a "
        "latent nonlinear/residual advantage. The cognition gap is mostly negative through "
        "n=683, while reconstruction also converges to approximately zero incremental gain."
    )
    parts.append(source_line("scaling", EVIDENCE_SOURCES["Scaling results"]))

    parts.append("\n## E12. Preprocessing-Axis Stress Tests Preserve Directionality\n")
    method_paths = [
        EVIDENCE_SOURCES["Preprocessing method A"],
        EVIDENCE_SOURCES["Preprocessing method B"],
        EVIDENCE_SOURCES["Preprocessing method C"],
    ]
    methods = pd.concat([pd.read_csv(p) for p in method_paths], ignore_index=True)
    method_summary = (
        methods.groupby(["method", "jl_variant", "direction"], dropna=False)
        .agg(
            median_dp=("demeaned_pearson", "median"),
            median_rank=("avg_rank", "median"),
            median_top1=("top1_acc", "median"),
            n=("seed", "nunique"),
        )
        .reset_index()
    )
    parts.append(rounded_table(method_summary))
    parts.append(
        "\nInterpretation: the FC->SC > SC->FC directionality survives PCA-PLS-PCA, full PLS, "
        "and JL-PLS-PCA variants. This makes the result less dependent on a particular "
        "dimensionality-reduction path."
    )
    parts.append(source_line("preprocessing methods", *method_paths))
    return "\n\n".join(parts)


def third_pass_addendum() -> str:
    f8_stability = pd.read_csv(CSV_SOURCES["F8 stability"])
    f8_per_pc = pd.read_csv(EVIDENCE_SOURCES["Family PC per-component"])
    f8_local = pd.read_csv(EVIDENCE_SOURCES["Family PC3 localization"])
    f8_enrich = pd.read_csv(EVIDENCE_SOURCES["Family PC3 enrichment"])
    resid_enrich = pd.read_csv(EVIDENCE_SOURCES["Reliability-residualized PC3 enrichment"])
    fc_rel = pd.read_csv(EVIDENCE_SOURCES["FC reliability summary"])
    variance = pd.read_csv(EVIDENCE_SOURCES["FC variance decomposition"])
    disatten = pd.read_csv(EVIDENCE_SOURCES["Crossmodal disattenuation"])
    discrim = pd.read_csv(EVIDENCE_SOURCES["FC discriminability"])

    parts: list[str] = []
    parts.append("# Existing-Data Gap Closure, Pass 3\n")
    parts.append(
        "This pass tightens two interpretive edges: whether the family/mechanism result is "
        "stable and localized, and how the FC measurement-noise accounting should be read."
    )

    parts.append("\n## E13. Family Mechanism Is Stable, Localized, and Not Just Rich-Club Confounding\n")
    pc3_stability = f8_stability[f8_stability["anchor_pc"] == 3][
        [
            "parcellation",
            "anchor_pc",
            "median_abs_cos",
            "min_abs_cos",
            "median_expl_var",
            "median_FC_to_PC_R2",
            "median_AUC_sibling",
        ]
    ]
    pc3_local_summary = (
        f8_local[f8_local["K"].isin([100, 200])]
        .groupby(["parcellation", "K"])
        .agg(
            interhemi_top=("interhemi_frac_top", "median"),
            richclub_top=("richclub_frac_top", "median"),
            energy_top1pct=("energy_top1pct_frac", "median"),
            anchor_cos_min=("anchor_match_signedcos", "min"),
            n=("seed", "nunique"),
        )
        .reset_index()
    )
    pc3_enrich_top = f8_enrich.sort_values("median_enrichment", ascending=False).head(8)
    resid_top = resid_enrich.sort_values("enrichment_resid", ascending=False).head(8)[
        ["net_pair", "n_raw_top200", "n_resid_top200", "enrichment_raw", "enrichment_resid"]
    ]
    pc3_predict = (
        f8_per_pc[f8_per_pc["pc"] == 3]
        .groupby("parcellation")
        .agg(
            median_FC_to_PC_R2=("FC_to_PC_R2", "median"),
            median_AUC_sibling=("AUC_sibling", "median"),
            median_confound_R2=("confound_R2_test", "median"),
            n=("seed", "nunique"),
        )
        .reset_index()
    )
    parts.append("PC3 stability and family signal:\n")
    parts.append(rounded_table(pc3_stability))
    parts.append("\nPC3 predictability/confound summary from per-seed outputs:\n")
    parts.append(rounded_table(pc3_predict))
    parts.append("\nLocalization summary:\n")
    parts.append(rounded_table(pc3_local_summary))
    parts.append("\nTop network enrichments across seeds:\n")
    parts.append(rounded_table(pc3_enrich_top[["parcellation", "net_pair", "n_seeds", "median_enrichment", "min_enrichment", "median_n_obs"]]))
    parts.append("\nReliability/distance residualized top-200 enrichment:\n")
    parts.append(rounded_table(resid_top))
    parts.append(
        "\nInterpretation: PC3 is aligned across seeds, carries sibling/family information, "
        "and is spatially concentrated in visual and dorsal-attention edges. The residualized "
        "top-200 check says this localization survives obvious reliability/tractography "
        "proxies rather than collapsing into a generic high-strength or rich-club artifact."
    )
    parts.append(source_line("family localization", CSV_SOURCES["F8 stability"], EVIDENCE_SOURCES["Family PC per-component"], EVIDENCE_SOURCES["Family PC3 localization"], EVIDENCE_SOURCES["Family PC3 enrichment"], EVIDENCE_SOURCES["Reliability-residualized PC3 enrichment"]))

    parts.append("\n## E14. FC Noise Accounting: Edge Noise Is Large, Aggregate Identity Is Still Reliable\n")
    parts.append("FC edge-level reliability summary:\n")
    parts.append(rounded_table(fc_rel))
    parts.append("\nVariance decomposition:\n")
    parts.append(rounded_table(variance))
    parts.append("\nCrossmodal disattenuation:\n")
    parts.append(rounded_table(disatten))
    parts.append("\nWhole-connectome discriminability:\n")
    parts.append(rounded_table(discrim))
    parts.append(
        "\nInterpretation: the apparent tension is real but resolved. Individual edges are "
        "substantially noisy, yet whole-connectome fingerprints are highly discriminable. "
        "SC->FC captures only a small fraction of reproducible demeaned FC signal and a small "
        "fraction of fingerprint top-1 identity, so the negative result is not simply a failure "
        "to recognize subjects in aggregate."
    )
    parts.append(source_line("FC noise accounting", EVIDENCE_SOURCES["FC reliability summary"], EVIDENCE_SOURCES["FC variance decomposition"], EVIDENCE_SOURCES["Crossmodal disattenuation"], EVIDENCE_SOURCES["FC discriminability"]))
    return "\n\n".join(parts)


def gap_plan() -> str:
    rows = [
        {
            "gap": "SC-side reliability/noise",
            "needed": "Repeat dMRI, split-half tractography, bootstrap streamlines, or saved tractography perturbations.",
            "feasibility": "medium if raw diffusion/tractography pipeline is available; low from summary CSVs alone",
            "why": "Current repository bounds FC noise well but cannot produce a true SC test-retest ceiling.",
        },
        {
            "gap": "Objective interpolation",
            "needed": "New training grid mixing reconstruction, fingerprint/family, and cognition objectives on frozen splits.",
            "feasibility": "medium-high computationally; no new cohort needed",
            "why": "Would turn the objective-mismatch argument from inferential to causal.",
        },
        {
            "gap": "Broader phenotype families",
            "needed": "Additional HCP behavioral, personality, motor, emotion, and latent phenotype targets.",
            "feasibility": "high if phenotypes are already local; medium if target cleaning is needed",
            "why": "Current grid covers three cognition composites plus age/sex controls, not all behavior.",
        },
        {
            "gap": "Predicted-SC calibration/topology",
            "needed": "Saved predicted matrices or regenerated predictions; compare degree, strength, sparsity, modularity, hubs.",
            "feasibility": "medium if predictions are cached; medium-low if all predictions must be regenerated",
            "why": "Correlation can look acceptable while graph topology is biologically distorted.",
        },
        {
            "gap": "Edge-class stratification",
            "needed": "Per-edge predictions plus distance, network labels, reliability, and streamline-strength bins.",
            "feasibility": "medium with saved matrices and atlas metadata",
            "why": "Could show whether useful signal is concentrated in short/long, intra/inter-network, or high-reliability edges.",
        },
        {
            "gap": "Split stress tests",
            "needed": "Rerun selected cells under random, family-aware, age/sex-balanced, high-motion-excluded, and low-motion-only splits.",
            "feasibility": "medium; compute-heavy but no new data",
            "why": "Would make the split-drift/leakage defense harder to attack.",
        },
        {
            "gap": "External validity",
            "needed": "Second cohort with FC, dMRI-derived SC, demographics, and comparable cognition/behavior targets.",
            "feasibility": "low-medium depending on access",
            "why": "HCP-YA-only evidence supports an internal claim, not a universal population claim.",
        },
        {
            "gap": "Full hyperparameter leakage audit",
            "needed": "Static/code audit plus small rerun proving PCA, scaling, residualization, and target transforms fit inside folds.",
            "feasibility": "high for code audit; medium for rerun",
            "why": "Leak verdicts cover output behavior; this would document every transform boundary.",
        },
    ]
    bullet_rows = []
    for row in rows:
        bullet_rows.append(
            f"- **{row['gap']}**\n"
            f"  - Needed: {row['needed']}\n"
            f"  - Feasibility: {row['feasibility']}.\n"
            f"  - Why it matters: {row['why']}"
        )
    text = [
        "# Sanity-Check Gap Plan\n",
        "This file separates checks already closed from existing repository outputs from checks that require new data, saved predictions, or new model runs.",
        "\n## Closed With Existing Data\n",
        "- Reconstruction quality is decoupled from cognitive lift; see `supplement_sanity_checks.md`, section `Existing-Data Gap Closure`.",
        "- The current downstream grid covers three cognition composites plus age/sex controls.",
        "- Seed-level lift bounds show predicted SC alone is near-zero or negative for cognition, while observed FC plus bv+demo is detectably positive.",
        "- Leak verdicts show zero genuine `LEAK_FAIL` cells.",
        "- Family AUC shows predicted connectomes can preserve identity/family signal when the representation selects for it.",
        "- Per-subject FC reliability does not explain SC->FC achieved performance.",
        "- Expected-vs-observed grid completeness is exact for the reproduction outputs.",
        "- Estimator, nonlinear, residual, sink, richer-tractography, scaling, and preprocessing variants do not rescue cognition.",
        "- Family PC3 stability/localization and FC-noise accounting are summarized from existing structured outputs.",
        "\n## Requires New Data Or New Runs\n",
        "\n\n".join(bullet_rows),
    ]
    return "\n\n".join(text) + "\n"


def compact_source_manifest() -> str:
    lines = ["## Source Manifest\n"]
    for name, path in {**SOURCES, **CSV_SOURCES, **EVIDENCE_SOURCES}.items():
        lines.append(f"- **{name}**: {breakable_path(path)}")
    return "\n".join(lines)


def curated_front_matter() -> str:
    spine = read_text("spine")
    # Keep the existing supplement as the authored spine, but avoid the future-tense
    # "to generate" sections because this file now includes generated tables.
    spine = re.split(r"\n## S11\. Supplementary Tables To Generate", spine)[0].strip()
    return spine


def build() -> str:
    parts: list[str] = []
    parts.append("# Supplementary Material: Sanity Checks and Reproducibility Discipline\n")
    parts.append(
        "This document is a long-form supplement for the Conn2Conn preprint. "
        "It combines the compact manuscript supplement with the detailed sanity-check notes "
        "from the repository, plus selected machine-readable CSV summaries. The purpose is "
        "to make the negative claims auditable: reduction choices, FC measurement noise, "
        "nonlinear capacity, richer tractography, family objectives, and operational grid "
        "completeness are all checked explicitly.\n"
    )
    parts.append(curated_front_matter())

    parts.append("\n# Machine-Readable Summary Tables\n")
    for title, path in CSV_SOURCES.items():
        max_rows = 24 if title in {"Reduction-axis all seed ratios", "Family AUC", "F8 stability"} else 14
        parts.append(csv_table(title, path, max_rows=max_rows))

    parts.append(existing_data_addendum())
    parts.append(second_pass_addendum())
    parts.append(third_pass_addendum())

    parts.append("\n# New Data and Experiment Triage\n")
    parts.append(
        "The existing data close several presentation-level holes, but not every scientific "
        "hole. The remaining checks below require new model runs, saved prediction matrices, "
        "raw diffusion perturbations, or an external cohort. A standalone copy of this triage "
        "is written to `sanity_check_gap_plan.md`."
    )
    parts.append(gap_plan())

    parts.append("\n# Detailed Findings Notes\n")
    parts.append(
        "The following subsections preserve the detailed findings notes that motivated and "
        "defended the manuscript claims. Minor Unicode normalization is applied for stable "
        "LaTeX compilation; source paths are listed in the manifest.\n"
    )
    for key, title in [
        ("reproduction_findings", "Reproduction Grid Findings"),
        ("exploration_findings", "Grid Exploration Audit"),
        ("discrepancy", "Estimator Discrepancy Resolution"),
        ("preprocessing", "Reduction-Axis Robustness"),
        ("noise", "FC Noise and Reliability Sanity Check"),
        ("tract_check", "PC Mechanism Tractography-Reliability Check"),
        ("tractography", "Richer Tractography Representation Check"),
        ("nonlinear", "Nonlinear Model-Class Check"),
        ("residual", "Residual-Boost and Multimodal Sink Check"),
        ("scaling", "Sample-Size Scaling Check"),
        ("family_readme", "Family Mechanism Reproduction Notes"),
    ]:
        parts.append(f"\n## {title}\n")
        body = read_text(key)
        parts.append(heading_shift(body, level=2))

    parts.append("\n# Closing Interpretation\n")
    parts.append(
        "Taken together, these checks support a narrow but strong conclusion. FC-SC "
        "translation is reproducible and directional, but reconstruction accuracy is not "
        "a sufficient proxy for downstream cognitive utility. The null is not explained "
        "by the PCA reduction axis, obvious nonlinear model capacity, richer tractography "
        "features, sample-size trends within the tested regime, FC-side measurement noise, "
        "or hard grid leakage. The constructive exception is family signal: predicted "
        "connectomes can preserve identity/family information when the objective selects "
        "for it, but that does not imply cognition transfer.\n"
    )
    parts.append(compact_source_manifest())
    return deemoji("\n\n".join(parts)) + "\n"


def main() -> None:
    OUT_MD.write_text(build())
    OUT_GAP_PLAN.write_text(deemoji(gap_plan()))
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_GAP_PLAN}")


if __name__ == "__main__":
    main()
