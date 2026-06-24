#!/usr/bin/env python3
"""Build the F1-F10 expanded Conn2Conn preprint v2.

Writes only inside preprint/preprint-codexv2. Reads source CSVs and finding
notes from the repository as immutable inputs.
"""
from __future__ import annotations

import os
from pathlib import Path
from textwrap import dedent

SCRIPT = Path(__file__).resolve()
CODEX = SCRIPT.parents[1]
ROOT = SCRIPT.parents[3]
FIG = CODEX / "figures"
MPLCONFIG = CODEX / ".mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TEX = CODEX / "conn2conn_findings_v2.tex"
MANIFEST = CODEX / "findings_v2_manifest.md"

COLORS = {
    "FC->SC": "#2E6F9E",
    "SC->FC": "#C44E52",
    "PCA-PLS": "#4C78A8",
    "BayesRidge": "#59A14F",
    "KernelRidge": "#F28E2B",
    "Glasser": "#4C78A8",
    "4S456Parcels": "#F28E2B",
    "positive": "#2E8B57",
    "negative": "#B55252",
    "neutral": "#7A7A7A",
}


PATHS = {
    "handoff": ROOT / "dev-notes" / "PROJECT-HANDOFF-2026-06-22.md",
    "master_findings": ROOT / "notebooks-FC_to_SC-experimental" / "MASTER_FINDINGS.md",
    "exploration": ROOT / "reproduction" / "exploration" / "FINDINGS_EXPLORATION.md",
    "reproduction_findings": ROOT / "reproduction" / "reports" / "reproduction_findings.md",
    "reconstruction": ROOT / "reproduction" / "outputs" / "reconstruction.csv",
    "downstream": ROOT / "reproduction" / "outputs" / "downstream.csv",
    "leak": ROOT / "reproduction" / "outputs" / "leak_verdict.csv",
    "family_auc": ROOT / "reproduction" / "family_mechanism" / "outputs" / "family_auc.csv",
    "f8_per_pc": ROOT / "reproduction" / "family_mechanism" / "outputs" / "f8_per_pc.csv",
    "f8_stability": ROOT / "reproduction" / "family_mechanism" / "outputs" / "f8_stability.csv",
    "f8_localization": ROOT / "reproduction" / "family_mechanism" / "outputs" / "f8_pc3_localization.csv",
    "f8_enrichment": ROOT / "reproduction" / "family_mechanism" / "outputs" / "f8_pc3_enrichment_agg.csv",
    "tract_e1": ROOT / "notebooks-FC_to_SC-experimental" / "tractography_predict" / "e1_source_rep_results.csv",
    "tract_e2": ROOT / "notebooks-FC_to_SC-experimental" / "tractography_predict" / "e2_asymmetry_summary.csv",
    "tract_e3": ROOT / "notebooks-FC_to_SC-experimental" / "tractography_predict" / "e3_marginal_summary.csv",
    "tract_e5": ROOT / "notebooks-FC_to_SC-experimental" / "tractography_predict" / "e5_downstream_summary.csv",
    "tract_findings": ROOT / "notebooks-FC_to_SC-experimental" / "tractography_predict" / "findings.md",
    "nl_n1": ROOT / "notebooks-FC_to_SC-experimental" / "non-linear-sanity-check" / "n1_cognition_summary.csv",
    "nl_n2": ROOT / "notebooks-FC_to_SC-experimental" / "non-linear-sanity-check" / "n2_reconstruction_summary.csv",
    "nl_n3": ROOT / "notebooks-FC_to_SC-experimental" / "non-linear-sanity-check" / "n3_marginal_summary.csv",
    "nl_n4_cog": ROOT / "notebooks-FC_to_SC-experimental" / "non-linear-sanity-check" / "n4_cog_summary.csv",
    "nl_n4_recon": ROOT / "notebooks-FC_to_SC-experimental" / "non-linear-sanity-check" / "n4_recon_summary.csv",
    "nl_n5_cog": ROOT / "notebooks-FC_to_SC-experimental" / "non-linear-sanity-check" / "n5_cog_summary.csv",
    "nl_n5_recon": ROOT / "notebooks-FC_to_SC-experimental" / "non-linear-sanity-check" / "n5_recon_summary.csv",
    "nl_n6": ROOT / "notebooks-FC_to_SC-experimental" / "non-linear-sanity-check" / "n6_scaling_summary.csv",
    "nl_findings": ROOT / "notebooks-FC_to_SC-experimental" / "non-linear-sanity-check" / "findings_nonlinear.md",
    "residual_findings": ROOT / "notebooks-FC_to_SC-experimental" / "non-linear-sanity-check" / "findings_residual.md",
    "scaling_findings": ROOT / "notebooks-FC_to_SC-experimental" / "non-linear-sanity-check" / "findings_scaling.md",
    "reduction_axis": ROOT / "notebooks-FC_to_SC-experimental" / "sanity_checks" / "preprocessing_check" / "reduction_axis_summary.csv",
    "noise_hcorr": ROOT / "notebooks-FC_to_SC-experimental" / "sanity_checks" / "noise_sanity_check" / "outputs" / "h_correlations.csv",
    "noise_hfilter": ROOT / "notebooks-FC_to_SC-experimental" / "sanity_checks" / "noise_sanity_check" / "outputs" / "h_reliability_filtered_summary.csv",
    "tract_check": ROOT / "notebooks-FC_to_SC-experimental" / "sanity_checks" / "tract_check" / "findings.md",
    "noise_findings": ROOT / "notebooks-FC_to_SC-experimental" / "sanity_checks" / "noise_sanity_check" / "findings_noise.md",
    "preprocessing_findings": ROOT / "notebooks-FC_to_SC-experimental" / "sanity_checks" / "preprocessing_check" / "findings.md",
}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def esc(text: object) -> str:
    s = "" if pd.isna(text) else str(text)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in repl.items():
        s = s.replace(old, new)
    return s


def fmt(x: object, digits: int = 3) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, (bool, np.bool_)):
        return "yes" if bool(x) else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if x != 0 and abs(x) < 0.001:
            return f"{x:.2e}"
        return f"{x:.{digits}f}"
    return str(x)


def tex_table(df: pd.DataFrame, caption: str, label: str, digits: int = 3, max_rows: int | None = None) -> str:
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows).copy()
        caption = f"{caption} Showing first {max_rows} rows."
    cols = list(df.columns)
    align = "l" * len(cols)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        rf"\caption{{{esc(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(esc(c) for c in cols) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(esc(fmt(row[c], digits)) for c in cols) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def fig_tex(filename: str, caption: str, label: str, width: str = r"0.95\linewidth") -> str:
    return "\n".join(
        [
            r"\begin{figure}[H]",
            r"\centering",
            rf"\includegraphics[width={width}]{{figures/{filename}}}",
            rf"\caption{{{esc(caption)}}}",
            rf"\label{{{label}}}",
            r"\end{figure}",
        ]
    )


def est_label(est: str) -> str:
    return {
        "pca_pls": "PCA-PLS",
        "bayesian_ridge": "BayesRidge",
        "kernel_ridge": "KernelRidge",
        "linear_BR": "Linear BR",
        "linear_PLS": "Linear PLS",
        "KR": "KernelRidge",
        "HGB": "HGB",
    }.get(est, est)


def savefig(name: str) -> None:
    for ext in ["png", "pdf"]:
        plt.savefig(FIG / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close()


def base_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def add_bar_labels(ax, rotation: int = 0) -> None:
    for patch in ax.patches:
        h = patch.get_height()
        if not np.isfinite(h):
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            h + (0.01 if h >= 0 else -0.02),
            f"{h:.2f}",
            ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=7,
            rotation=rotation,
        )


def load_data() -> dict[str, pd.DataFrame]:
    data = {}
    for key, path in PATHS.items():
        if path.suffix == ".csv":
            data[key] = pd.read_csv(path)
    return data


def recon_summary(rec: pd.DataFrame) -> pd.DataFrame:
    df = rec.copy()
    df["model"] = df["estimator"].map(est_label)
    rows = []
    for (parc, model, source, target), grp in df.groupby(["parcellation", "model", "source", "target"]):
        rows.append(
            {
                "parcellation": parc,
                "model": model,
                "source": source,
                "target": target,
                "direction": f"{source}->{target}",
                "demeaned_pearson": grp["demeaned_pearson"].median(),
                "avg_rank": grp["avg_rank"].median(),
                "top1_acc": grp["top1_acc"].median(),
                "n": len(grp),
            }
        )
    return pd.DataFrame(rows)


def make_f1(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, str]:
    rs = recon_summary(data["reconstruction"])
    cross = rs[rs["direction"].isin(["FC->SC", "SC->FC"])].copy()
    pivot = cross.pivot_table(index=["parcellation", "model"], columns="direction", values="demeaned_pearson")
    pivot["ratio_FC_to_SC_over_SC_to_FC"] = pivot["FC->SC"] / pivot["SC->FC"]
    out = pivot.reset_index().sort_values(["parcellation", "model"])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    for ax, parc in zip(axes, ["Glasser", "4S456Parcels"]):
        sub = cross[cross["parcellation"] == parc]
        models = ["PCA-PLS", "BayesRidge", "KernelRidge"]
        x = np.arange(len(models))
        width = 0.36
        for i, direction in enumerate(["FC->SC", "SC->FC"]):
            vals = [
                sub[(sub["model"] == m) & (sub["direction"] == direction)]["demeaned_pearson"].median()
                for m in models
            ]
            ax.bar(x + (i - 0.5) * width, vals, width, label=direction, color=COLORS[direction])
        ax.set_title(parc)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.axhline(0, color="#444", lw=0.8)
        ax.set_ylabel("Median demeaned Pearson")
        ax.legend(frameon=False)
    fig.suptitle("F1: FC->SC exceeds SC->FC across parcellations and model families")
    savefig("f1_asymmetry_models")
    return out, "f1_asymmetry_models.png"


def make_f2(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, str]:
    rs = recon_summary(data["reconstruction"])
    sub = rs[(rs["source"].isin(["bv", "demo"])) & (rs["target"].isin(["SC", "FC"]))].copy()
    order = ["bv->SC", "bv->FC", "demo->SC", "demo->FC"]
    models = ["PCA-PLS", "BayesRidge", "KernelRidge"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, parc in zip(axes, ["Glasser", "4S456Parcels"]):
        ss = sub[sub["parcellation"] == parc]
        x = np.arange(len(order))
        width = 0.24
        for i, model in enumerate(models):
            vals = [ss[(ss["model"] == model) & (ss["direction"] == d)]["demeaned_pearson"].median() for d in order]
            ax.bar(x + (i - 1) * width, vals, width, label=model, color=COLORS[model])
        ax.set_title(parc)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=25, ha="right")
        ax.set_ylabel("Median demeaned Pearson")
        ax.legend(frameon=False)
    fig.suptitle("F2: anatomy predicts SC, demographics predict FC")
    savefig("f2_double_dissociation_models")

    table = sub[["parcellation", "model", "direction", "demeaned_pearson", "avg_rank", "top1_acc"]]
    return table.sort_values(["parcellation", "model", "direction"]), "f2_double_dissociation_models.png"


def downstream_summary(down: pd.DataFrame) -> pd.DataFrame:
    df = down.copy()
    df["model"] = df["estimator"].map(est_label)
    rows = []
    for (parc, model, input_set, target), grp in df.groupby(["parcellation", "model", "input_set", "target"]):
        rows.append(
            {
                "parcellation": parc,
                "model": model,
                "input_set": input_set,
                "target": target,
                "pearson": grp["pearson"].mean(),
                "lift_over_bvdemo": grp["lift_over_bvdemo"].mean(),
                "median_perm_p": grp["lift_perm_p"].median(),
                "residualized_pearson": grp["residualized_pearson"].mean(),
                "balanced_acc": grp["balanced_acc"].mean(),
                "n": len(grp),
            }
        )
    return pd.DataFrame(rows)


def make_f3_f5(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    ds = downstream_summary(data["downstream"])
    cog = ds[(ds["model"] == "BayesRidge") & (ds["target"].isin(["CogTotal", "CogFluid", "CogCryst"]))].copy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, parc in zip(axes, ["Glasser", "4S456Parcels"]):
        sub = cog[(cog["parcellation"] == parc) & (cog["input_set"].isin(["obs_FC", "obs_SC", "pred_SC", "pred_FC"]))]
        order = ["obs_FC", "obs_SC", "pred_SC", "pred_FC"]
        targets = ["CogTotal", "CogFluid", "CogCryst"]
        x = np.arange(len(order))
        width = 0.24
        for i, target in enumerate(targets):
            vals = [sub[(sub["input_set"] == inp) & (sub["target"] == target)]["lift_over_bvdemo"].mean() for inp in order]
            ax.bar(x + (i - 1) * width, vals, width, label=target)
        ax.axhline(0, color="#333", lw=0.8)
        ax.set_title(parc)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=25, ha="right")
        ax.set_ylabel("Lift over bv+demo")
        ax.legend(frameon=False)
    fig.suptitle("F3/F5: observed FC helps cognition; imputed/structural connectomes do not transfer utility")
    savefig("f3_f5_imputation_utility")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, parc in zip(axes, ["Glasser", "4S456Parcels"]):
        sub = cog[(cog["parcellation"] == parc) & (cog["input_set"].isin(["obs_FC+bv+demo", "pred_SC+bv+demo", "pred_FC+bv+demo", "obs_SC+bv+demo"]))]
        order = ["obs_FC+bv+demo", "pred_SC+bv+demo", "pred_FC+bv+demo", "obs_SC+bv+demo"]
        vals = [sub[sub["input_set"] == inp]["lift_over_bvdemo"].mean() for inp in order]
        cols = [COLORS["positive"] if v > 0 else COLORS["negative"] for v in vals]
        ax.bar(np.arange(len(order)), vals, color=cols)
        ax.axhline(0, color="#333", lw=0.8)
        ax.set_title(parc)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(order, rotation=25, ha="right")
        ax.set_ylabel("Mean lift across cognition targets")
    fig.suptitle("F5: adding real FC beats the floor; predicted FC and observed SC underperform")
    savefig("f5_sc_predfc_underperform")

    f3_table = cog[cog["input_set"].isin(["obs_FC", "obs_SC", "pred_SC", "pred_FC"])][
        ["parcellation", "input_set", "target", "pearson", "lift_over_bvdemo", "median_perm_p", "residualized_pearson"]
    ].sort_values(["parcellation", "target", "input_set"])
    f5_table = cog[cog["input_set"].isin(["obs_FC+bv+demo", "pred_SC+bv+demo", "pred_FC+bv+demo", "obs_SC+bv+demo"])][
        ["parcellation", "input_set", "target", "pearson", "lift_over_bvdemo", "median_perm_p"]
    ].sort_values(["parcellation", "target", "input_set"])
    return f3_table, f5_table, "f3_f5_imputation_utility.png", "f5_sc_predfc_underperform.png"


def make_f4(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, str]:
    ds = downstream_summary(data["downstream"])
    sub = ds[(ds["model"] == "BayesRidge") & (ds["target"].isin(["CogTotal", "CogFluid", "CogCryst"]))].copy()
    sub = sub[sub["input_set"].isin(["obs_FC", "obs_FC+bv+demo", "bv+demo"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), sharey=True)
    targets = ["CogTotal", "CogFluid", "CogCryst"]
    order = ["bv+demo", "obs_FC", "obs_FC+bv+demo"]
    for ax, parc in zip(axes, ["Glasser", "4S456Parcels"]):
        ss = sub[sub["parcellation"] == parc]
        x = np.arange(len(targets))
        width = 0.24
        for i, inp in enumerate(order):
            vals = [ss[(ss["target"] == target) & (ss["input_set"] == inp)]["pearson"].mean() for target in targets]
            ax.bar(x + (i - 1) * width, vals, width, label=inp)
        ax.set_title(parc)
        ax.set_xticks(x)
        ax.set_xticklabels(targets)
        ax.set_ylabel("Mean test Pearson")
        ax.legend(frameon=False)
    fig.suptitle("F4: observed FC carries cognition signal beyond the free baseline")
    savefig("f4_observed_fc_cognition")
    table = sub[["parcellation", "input_set", "target", "pearson", "lift_over_bvdemo", "median_perm_p", "residualized_pearson"]]
    return table.sort_values(["parcellation", "target", "input_set"]), "f4_observed_fc_cognition.png"


def make_f6_f7(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    family = data["family_auc"].copy()
    sib = family[family["relation"] == "sibling"].copy()
    order = ["bvdemo_to_SC", "pred_SC_raw", "pred_SC_resid_bvdemo", "combined_pred_SC", "obs_SC", "obs_FC"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, parc in zip(axes, ["Glasser", "4S456Parcels"]):
        ss = sib[(sib["parcellation"] == parc) & (sib["variant"].isin(order))]
        ss["variant"] = pd.Categorical(ss["variant"], categories=order, ordered=True)
        ss = ss.sort_values("variant")
        ax.bar(np.arange(len(ss)), ss["auc"], color="#4C78A8")
        ax.errorbar(np.arange(len(ss)), ss["auc"], yerr=[ss["auc"] - ss["auc_lo"], ss["auc_hi"] - ss["auc"]], fmt="none", ecolor="#333", capsize=2)
        ax.axhline(0.5, color="#B55252", ls="--", lw=0.9)
        ax.set_title(parc)
        ax.set_xticks(np.arange(len(ss)))
        ax.set_xticklabels(ss["variant"], rotation=25, ha="right")
        ax.set_ylabel("Sibling AUC")
    fig.suptitle("F6/F7: predicted connectomes can carry family signal, but reconstruction and identification trade off")
    savefig("f6_f7_family_tradeoff")

    fig, ax = plt.subplots(figsize=(9.5, 4.1))
    rels = ["MZ", "DZ", "sibling"]
    variants = ["obs_SC", "pred_SC_resid_bvdemo", "combined_pred_SC", "bvdemo_to_SC"]
    x = np.arange(len(variants))
    width = 0.25
    ss = family[(family["parcellation"] == "Glasser") & (family["variant"].isin(variants))]
    for i, reln in enumerate(rels):
        vals = [ss[(ss["variant"] == var) & (ss["relation"] == reln)]["auc"].mean() for var in variants]
        ax.bar(x + (i - 1) * width, vals, width, label=reln)
    ax.axhline(0.5, color="#B55252", ls="--", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=20, ha="right")
    ax.set_ylabel("AUC")
    ax.set_title("F6 family-gradient check, shown for Glasser")
    ax.legend(frameon=False)
    savefig("f6_family_gradient")

    f6_table = family[family["variant"].isin(["obs_SC", "obs_FC", "pred_SC_resid_bvdemo", "bvdemo_to_SC"])][
        ["parcellation", "variant", "relation", "auc", "auc_lo", "auc_hi", "p_fdr", "sig_fdr"]
    ].sort_values(["parcellation", "variant", "relation"])
    f7_table = sib[sib["variant"].isin(["pred_SC_raw", "pred_SC_resid_bvdemo", "combined_pred_SC", "bvdemo_to_SC"])][
        ["parcellation", "variant", "auc", "auc_lo", "auc_hi", "p_fdr", "sig_fdr"]
    ].sort_values(["parcellation", "variant"])
    return f6_table, f7_table, "f6_family_gradient.png", "f6_f7_family_tradeoff.png"


def make_f8(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    stab = data["f8_stability"].copy()
    enrich = data["f8_enrichment"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for parc in ["Glasser", "4S456Parcels"]:
        ss = stab[stab["parcellation"] == parc]
        axes[0].plot(ss["anchor_pc"], ss["median_FC_to_PC_R2"], marker="o", label=parc, color=COLORS[parc])
        axes[1].plot(ss["anchor_pc"], ss["median_AUC_sibling"], marker="o", label=parc, color=COLORS[parc])
    axes[0].set_title("FC predictability by SC PC")
    axes[0].set_xlabel("SC PC")
    axes[0].set_ylabel("Median FC->PC R2")
    axes[1].axhline(0.5, color="#B55252", ls="--", lw=0.8)
    axes[1].set_title("Sibling signal by SC PC")
    axes[1].set_xlabel("SC PC")
    axes[1].set_ylabel("Median sibling AUC")
    top = enrich[enrich["net_pair"].isin(["visual || visual", "dorsal attention || dorsal attention", "dorsal attention || visual"])].copy()
    xlabels = top["net_pair"].drop_duplicates().tolist()
    x = np.arange(len(xlabels))
    width = 0.35
    for i, parc in enumerate(["Glasser", "4S456Parcels"]):
        vals = [top[(top["parcellation"] == parc) & (top["net_pair"] == lab)]["median_enrichment"].mean() for lab in xlabels]
        axes[2].bar(x + (i - 0.5) * width, vals, width, label=parc, color=COLORS[parc])
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(["visual-visual", "DAN-DAN", "DAN-visual"], rotation=20, ha="right")
    axes[2].set_title("Mechanism-mode enrichment")
    axes[2].set_ylabel("Median enrichment")
    for ax in axes:
        ax.legend(frameon=False)
    fig.suptitle("F8: a small FC-predictable, family-informative structural PC exists, but the PC index is atlas-sensitive")
    savefig("f8_pc_mechanism")
    table1 = stab[["parcellation", "anchor_pc", "median_abs_cos", "median_expl_var", "median_FC_to_PC_R2", "median_AUC_sibling"]]
    table2 = top[["parcellation", "net_pair", "median_enrichment", "min_enrichment", "median_n_obs"]]
    return table1.sort_values(["parcellation", "anchor_pc"]), table2.sort_values(["parcellation", "net_pair"]), "f8_pc_mechanism.png"


def make_f9(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    e1 = data["tract_e1"].copy()
    e5 = data["tract_e5"].copy()
    rep_order = ["FC", "bv+demo", "SC", "SC_r2t", "r2t", "r2t_corr", "r2t_synthFC", "kitchen_sink"]
    med = e1.groupby("rep", as_index=False)["demeaned_pearson"].median()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    m1 = med[med["rep"].isin(["SC", "SC_r2t", "r2t", "r2t_corr", "kitchen_sink"])].copy()
    m1["rep"] = pd.Categorical(m1["rep"], categories=["SC", "SC_r2t", "r2t", "r2t_corr", "kitchen_sink"], ordered=True)
    m1 = m1.sort_values("rep")
    axes[0].bar(np.arange(len(m1)), m1["demeaned_pearson"], color="#4C78A8")
    axes[0].set_xticks(np.arange(len(m1)))
    axes[0].set_xticklabels(m1["rep"], rotation=20, ha="right")
    axes[0].set_ylabel("Median demeaned Pearson to FC")
    axes[0].set_title("Structural representations predicting FC")
    lift = e5[e5["rep"].isin(rep_order)].copy()
    lift_total = lift[lift["target"].str.contains("Total")].copy()
    axes[1].bar(np.arange(len(lift_total)), lift_total["lift_over_bvdemo_raw"], color=[COLORS["positive"] if v > 0 else COLORS["negative"] for v in lift_total["lift_over_bvdemo_raw"]])
    axes[1].axhline(0, color="#333", lw=0.8)
    axes[1].set_xticks(np.arange(len(lift_total)))
    axes[1].set_xticklabels(lift_total["rep"], rotation=25, ha="right")
    axes[1].set_ylabel("CogTotal lift over bv+demo")
    axes[1].set_title("Downstream cognition")
    fig.suptitle("F9: richer tractography does not rescue FC prediction or cognition")
    savefig("f9_tractography_negative")
    e1_table = e1.groupby("rep", as_index=False).agg(
        demeaned_pearson=("demeaned_pearson", "median"),
        pearson=("pearson", "median"),
        top1_acc=("top1_acc", "median"),
        avg_rank=("avg_rank", "median"),
    )
    e5_table = e5[["rep", "target", "pearson_raw", "pearson_resid", "lift_over_bvdemo_raw"]]
    return e1_table.sort_values("demeaned_pearson", ascending=False), e5_table.sort_values(["target", "rep"]), "f9_tractography_negative.png"


def make_f10(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    n1 = data["nl_n1"].copy()
    n2 = data["nl_n2"].copy()
    n4 = data["nl_n4_recon"].copy()
    n5 = data["nl_n5_cog"].copy()
    n6 = data["nl_n6"].copy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    sc = n2[(n2["rep"] == "SC") & (n2["metric"] == "demeaned_pearson")].copy()
    axes[0].bar(np.arange(len(sc)), sc["asym_FCwins"], color="#4C78A8")
    axes[0].axhline(1, color="#333", lw=0.8)
    axes[0].set_xticks(np.arange(len(sc)))
    axes[0].set_xticklabels(sc["estimator"], rotation=20, ha="right")
    axes[0].set_ylabel("FC-wins ratio")
    axes[0].set_title("Asymmetry survives nonlinear KR")
    cog = n1[(n1["target"].str.contains("Total")) & (n1["rep"].isin(["FC", "SC", "r2t", "SC_r2t"]))].copy()
    xlabels = cog["rep"].drop_duplicates().tolist()
    ests = ["linear_BR", "KR", "HGB"]
    x = np.arange(len(xlabels))
    width = 0.25
    for i, est in enumerate(ests):
        vals = [cog[(cog["rep"] == rep) & (cog["estimator"] == est)]["lift_over_bvdemo"].mean() for rep in xlabels]
        axes[1].bar(x + (i - 1) * width, vals, width, label=est_label(est))
    axes[1].axhline(0, color="#333", lw=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(xlabels, rotation=20, ha="right")
    axes[1].set_ylabel("CogTotal lift over bv+demo")
    axes[1].set_title("No tractography nonlinear unlock")
    axes[1].legend(frameon=False)
    for task, color in [("cognition", "#4C78A8"), ("reconstruction", "#F28E2B")]:
        ss = n6[n6["task"] == task]
        axes[2].plot(ss["n_sub"], ss["median_gap"], marker="o", label=task, color=color)
    axes[2].axhline(0, color="#333", lw=0.8)
    axes[2].set_xlabel("Training n")
    axes[2].set_ylabel("Nonlinear residual gap")
    axes[2].set_title("Gap does not grow with n")
    axes[2].legend(frameon=False)
    fig.suptitle("F10: nonlinear models and more data do not break the ceiling")
    savefig("f10_nonlinear_scaling")

    n4_dp = n4[(n4["metric"] == "demeaned_pearson")][["rep", "direction", "median_template", "median_final", "median_improvement", "wilcoxon_p_improve"]]
    n6_table = n6[["task", "n_sub", "n_seeds", "median_linear", "median_final", "median_gap", "wilcoxon_p_gap_gt0"]]
    return n4_dp.sort_values(["rep", "direction"]), n6_table.sort_values(["task", "n_sub"]), "f10_nonlinear_scaling.png"


def make_sanity_figure(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    red = data["reduction_axis"].copy()
    hc = data["noise_hcorr"].copy()
    hf = data["noise_hfilter"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1))
    labels = red["method"].fillna("") + red["jl_variant"].fillna("").map(lambda x: f" {x}" if x else "")
    axes[0].bar(np.arange(len(red)), red["median_ratio"], color="#4C78A8")
    axes[0].axhline(1, color="#333", lw=0.8)
    axes[0].set_xticks(np.arange(len(red)))
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].set_title("Reduction-axis robustness")
    axes[0].set_ylabel("FC->SC / SC->FC ratio")
    axes[1].bar(np.arange(len(hc)), hc["pearson_achieved_vs_ceiling"], color=["#B55252", "#59A14F"])
    axes[1].set_xticks(np.arange(len(hc)))
    axes[1].set_xticklabels(hc["source"])
    axes[1].set_title("Achieved vs FC reliability")
    axes[1].set_ylabel("Pearson r")
    sc = hf[hf["source"] == "SC"]
    axes[2].plot(np.arange(len(sc)), sc["fraction_of_ceiling"], marker="o", color="#B55252")
    axes[2].set_xticks(np.arange(len(sc)))
    axes[2].set_xticklabels(sc["filter"], rotation=25, ha="right")
    axes[2].set_title("Reliability filtering does not rescue SC")
    axes[2].set_ylabel("Fraction of FC ceiling")
    fig.suptitle("Cross-cutting sanity checks supporting F1 and F10")
    savefig("sanity_reduction_noise")
    return red, hc, "sanity_reduction_noise.png"


def write_manifest() -> None:
    lines = ["# Findings v2 Source Manifest\n"]
    for key, path in PATHS.items():
        lines.append(f"- **{key}**: `{rel(path)}`")
    MANIFEST.write_text("\n".join(lines) + "\n")


def make_figures_and_tables() -> dict[str, object]:
    base_style()
    FIG.mkdir(exist_ok=True)
    data = load_data()
    f1_table, f1_fig = make_f1(data)
    f2_table, f2_fig = make_f2(data)
    f3_table, f5_table, f3_fig, f5_fig = make_f3_f5(data)
    f4_table, f4_fig = make_f4(data)
    f6_table, f7_table, f6_fig, f7_fig = make_f6_f7(data)
    f8_table, f8_enrich, f8_fig = make_f8(data)
    f9_table, f9_down, f9_fig = make_f9(data)
    f10_table, f10_scaling, f10_fig = make_f10(data)
    red_table, noise_table, sanity_fig = make_sanity_figure(data)
    write_manifest()
    return locals()


def source_refs(*keys: str) -> str:
    return " Sources: " + "; ".join(rel(PATHS[k]) for k in keys) + "."


def document(ctx: dict[str, object]) -> str:
    return rf"""
\documentclass{{article}}

\PassOptionsToPackage{{numbers,compress}}{{natbib}}
\PassOptionsToPackage{{hypertexnames=false}}{{hyperref}}
\usepackage[eandd,preprint]{{neurips_2026}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{hyperref}}
\usepackage{{url}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{graphicx}}
\usepackage{{longtable}}
\usepackage{{array}}
\usepackage{{caption}}
\usepackage{{float}}
\usepackage{{microtype}}
\usepackage{{xcolor}}
\usepackage{{enumitem}}
\usepackage{{placeins}}

\setlist[itemize]{{itemsep=2pt, topsep=3pt, parsep=0pt, partopsep=0pt}}
\setlength{{\abovecaptionskip}}{{4pt}}
\setlength{{\belowcaptionskip}}{{3pt}}
\setlength{{\textfloatsep}}{{10pt plus 2pt minus 2pt}}
\setlength{{\floatsep}}{{8pt plus 1pt minus 1pt}}

\title{{Conn2Conn Findings Ledger v2: A Section-by-Section F1--F10 Preprint}}
\author{{Adel Sahuc\\Conn2Conn Preprint Draft v2}}

\begin{{document}}
\maketitle

\begin{{abstract}}
This version rewrites the Conn2Conn preprint as a finding-by-finding ledger. Each section
tracks one finding, F1 through F10, across the two parcellations where confirmatory data
exist and across the model families or objective variants that bear on the claim. The
main distinction is between confirmatory findings replicated on both Glasser and
4S456Parcels, and exploratory checks that are still Glasser-only or hypothesis-generating.
The central conclusion is unchanged: cross-modal connectome prediction is real and
directional, but reconstruction quality does not transfer into cognition utility. FC is
the useful cognitive modality in this HCP-YA regime; structural connectomes, richer
tractography, nonlinear models, and imputed connectomes do not clear the bv+demo floor.
\end{{abstract}}

\section{{Reader's Map}}

This draft is intentionally longer than the compact preprint. It is a ledger rather than
a teaser: every finding gets its own section, its own figure, a parcellation/model table
where the repository contains such data, and an explicit status label.

\begin{{itemize}}
\item F1--F5 come from the confirmatory reproduction spine: two parcellations, ten frozen
family-aware splits, and three model families where applicable.
\item F6--F7 come from the family-mechanism grid: both parcellations, objective variants,
permutation tests, and bootstrap intervals.
\item F8 is intentionally hedged. The mechanism exists in both parcellations in the
property sense, but the PC index migrates across atlases, so the claim is a mechanism
hypothesis rather than a settled biomarker.
\item F9 and F10 remain Glasser/exploratory sanity checks. They are still important
because they close concrete escape routes: richer tractography, nonlinear model class,
residual learning, cross-modal interactions, and sample-size scaling.
\end{{itemize}}

{tex_table(pd.DataFrame([
    ["F1", "FC->SC > SC->FC asymmetry", "replicated both", "reconstruction grid"],
    ["F2", "anatomy->SC / demographics->FC double dissociation", "replicated both", "reconstruction grid"],
    ["F3", "imputed connectomes are usable but utility is asymmetric", "replicated both", "downstream grid"],
    ["F4", "observed FC carries cognition signal", "replicated both", "downstream grid"],
    ["F5", "SC underperforms; imputation does not transfer cognition utility", "replicated both", "downstream grid"],
    ["F6", "predicted connectomes carry heritable family signal", "replicated both", "family grid"],
    ["F7", "reconstruction objective and identification objective trade off", "replicated both", "family grid"],
    ["F8", "exploratory PC mechanism, PC3 Glasser / PC4 4S456", "partial", "family/mechanism grid"],
    ["F9", "richer tractography representation does not help", "Glasser/exploratory", "tractography sanity"],
    ["F10", "nonlinear models and more data do not break ceiling", "Glasser/exploratory", "nonlinear sanity"],
], columns=["Finding", "Meaning", "Status", "Primary evidence"]), "F1-F10 status ledger.", "tab:status", max_rows=None)}

\section{{F1: FC->SC Exceeds SC->FC}}

\textbf{{Status: replicated on both parcellations.}} The directional asymmetry is the
scaffolding result: FC predicts individual SC deviations better than SC predicts
individual FC deviations. The compact summary quotes the PCA-PLS workhorse because it is
the original reconstruction model; v2 shows the model ledger explicitly. The direction
survives PCA-PLS, Bayesian ridge, and the kernel-ridge family. The important point is not
that the absolute value is identical across estimators. It is not. The important point is
that every reasonable estimator family keeps the same ordering.

{fig_tex(ctx["f1_fig"], "F1 across parcellations and model families. Bars show median demeaned Pearson for FC->SC and SC->FC. The FC->SC bar is higher in both atlases and across PCA-PLS, Bayesian ridge, and the kernel-ridge sweep.", "fig:f1")}

{tex_table(ctx["f1_table"], "F1 median reconstruction metrics by parcellation and model family.", "tab:f1", max_rows=None)}

The 4S456 parcellation gives slightly higher structure-target prediction and the Glasser
atlas gives nearly the same asymmetry ratio. This matters because the atlas comparison is
within-cohort and split-matched. The finer atlas does not invent the effect; it shifts the
absolute structural resolution while preserving the direction. {esc(source_refs("reconstruction", "reproduction_findings", "exploration", "reduction_axis", "preprocessing_findings"))}

\section{{F2: Anatomy Predicts SC, Demographics Predict FC}}

\textbf{{Status: replicated on both parcellations.}} F2 is a crossed dissociation, not a
single nuisance observation. Brain-volume/anatomy variables better predict structural
connectivity, while demographics better predict functional connectivity. The clean way to
write it is as a 2x2: anatomy has its natural target in SC; demographics have their
natural target in FC. The same family-aware split logic and model families are used.

{fig_tex(ctx["f2_fig"], "F2 across parcellations and model families. Anatomy/brain-volume predicts SC more strongly than FC, while demographics shift toward FC. The crossover is visible in both atlases.", "fig:f2")}

{tex_table(ctx["f2_table"], "F2 model ledger: source variables to structural or functional targets.", "tab:f2", max_rows=18)}

The dissociation is central because it turns bv+demo from a nuisance into a required
scientific baseline. A connectome that cannot beat cheap subject-level covariates should
not be advertised as a useful cognitive biomarker. {esc(source_refs("reconstruction", "exploration", "master_findings"))}

\section{{F3: Imputed Connectomes Are Usable, But Utility Is Asymmetric}}

\textbf{{Status: replicated on both parcellations, interpreted through F5.}} F3 is the
bridge between reconstruction and downstream utility. Predicted connectomes are not
random. They carry source-modality information and can support some downstream tasks. But
the utility is not symmetric: imputed SC from FC is closer to neutral or mildly useful,
whereas imputed FC from SC is often harmful for cognition. This is why the paper should
not stop at reconstruction metrics.

{fig_tex(ctx["f3_fig"], "F3/F5 downstream utility ledger. Bayesian-ridge cognition lift over bv+demo is shown for observed and predicted FC/SC. Observed FC carries the positive signal; structural and imputed representations do not transfer it reliably.", "fig:f3")}

{tex_table(ctx["f3_table"], "F3 downstream utility of observed and imputed connectomes, Bayesian ridge.", "tab:f3", max_rows=18)}

The model caveat is important. The scalar downstream grid includes PCA-PLS and kernel
ridge rows, but the project ledger treats Bayesian ridge as the reportable scalar
estimator because the discrepancy audit showed it is stable for cognition targets. The
other model families are still useful as sensitivity checks; they do not rescue the
imputation story. {esc(source_refs("downstream", "exploration", "reproduction_findings"))}

\section{{F4: Observed FC Carries Real Cognition Signal}}

\textbf{{Status: replicated on both parcellations.}} F4 is the positive control for the
entire cognition argument. If the pipeline could not detect any connectome-cognition
signal at all, the SC/imputation null would be much less meaningful. Observed FC does
clear the bv+demo floor, especially for crystallized cognition. This gives the negative
claims teeth: the downstream framework can find signal when signal exists.

{fig_tex(ctx["f4_fig"], "F4 observed FC cognition signal. Mean Bayesian-ridge Pearson is shown for bv+demo, observed FC, and observed FC plus bv+demo across cognition targets and parcellations.", "fig:f4")}

{tex_table(ctx["f4_table"], "F4 observed FC vs bv+demo cognition ledger.", "tab:f4", max_rows=18)}

The result is intentionally modest. These are research-grade correlations, not a clinical
biomarker. The point is comparative: FC is the modality that adds cognitive information
above the cheap floor in this dataset; SC does not. {esc(source_refs("downstream", "master_findings", "exploration"))}

\section{{F5: SC Underperforms, Imputation Does Not Transfer Cognition Utility}}

\textbf{{Status: replicated on both parcellations.}} F5 is the paper's main redirect.
Observed SC is below the bv+demo floor for cognition, predicted SC is at best small and
target-dependent, and predicted FC is often actively harmful. This is stronger than
"imputation is useless." It says reconstruction can manufacture a plausible connectome
whose downstream cognitive content is misaligned with what real FC carries.

{fig_tex(ctx["f5_fig"], "F5 cognition lift when each connectome is added to bv+demo. Observed FC gives a positive lift; predicted SC is smaller; predicted FC and observed SC underperform.", "fig:f5")}

{tex_table(ctx["f5_table"], "F5 connectome plus bv+demo cognition ledger, Bayesian ridge.", "tab:f5", max_rows=18)}

This section should be written as an objective mismatch, not a model failure. F6 and F7
show why: predicted connectomes can carry real biological/family signal, but the signal
selected by reconstruction is not the one needed for cognition. {esc(source_refs("downstream", "exploration", "leak"))}

\section{{F6: Predicted Connectomes Carry Heritable Family Signal}}

\textbf{{Status: replicated on both parcellations.}} F6 prevents an over-simple reading of
F5. Predicted connectomes are not empty artifacts. When the analysis asks whether they
preserve family structure, the answer is yes. The strongest sibling result is the
predicted SC residualized against bv+demo, which separates siblings from unrelated pairs
well above the bv+demo-only baseline in both parcellations.

{fig_tex(ctx["f6_fig"], "F6 family-gradient check. AUCs are shown for MZ, DZ, and sibling separation in selected variants. Predicted SC residualized against bv+demo preserves family signal beyond the subject-info baseline.", "fig:f6")}

{tex_table(ctx["f6_table"], "F6 family AUCs with confidence intervals and FDR-adjusted permutation tests.", "tab:f6", max_rows=24)}

The family result is constructive: cross-modal prediction can preserve a heritable
connectome signature. The lesson is narrower and more interesting than "prediction is
bad." Prediction preserves the signal selected by its objective and representation.
{esc(source_refs("family_auc", "handoff", "master_findings"))}

\section{{F7: Reconstruction and Identification Objectives Trade Off}}

\textbf{{Status: replicated on both parcellations.}} F7 is the objective warning. The
combined reconstruction-oriented predicted SC looks like a better reconstruction object,
but sibling identification collapses to chance. The residualized predicted SC keeps
family signal, while the combined reconstruction variant does not. Thus one cannot
assume that optimizing reconstruction, identification, and cognition are aligned.

{fig_tex(ctx["f7_fig"], "F6/F7 sibling AUC across objective variants. The combined predicted SC variant collapses to chance for sibling separation, while pred_SC_resid_bvdemo remains strongly above chance.", "fig:f7")}

{tex_table(ctx["f7_table"], "F7 sibling AUC objective tradeoff.", "tab:f7", max_rows=None)}

This is the best conceptual support for the manuscript's redirect: the missing cognition
transfer is not because the predicted connectomes are vacuous. It is because connectome
objectives are not interchangeable. {esc(source_refs("family_auc", "exploration", "master_findings"))}

\section{{F8: Exploratory PC Mechanism, PC3 Glasser / PC4 4S456}}

\textbf{{Status: partial and hypothesis-generating.}} F8 should be written carefully. The
mechanism-like structural PC exists in the property sense: a low-variance SC mode is
moderately FC-predictable, carries weak family signal, and localizes to visual and dorsal
attention edges. But the index is atlas-sensitive: the Glasser story is PC3, while the
4S456 mechanism shifts to a neighboring PC. Therefore the right claim is not "PC3 is the
biomarker." The right claim is "there is a small, heritable, FC-predictable structural
mode whose exact PC index depends on the atlas."

{fig_tex(ctx["f8_fig"], "F8 mechanism ledger. FC predictability, sibling AUC, and network enrichment are shown across PCs and parcellations. The mechanism property replicates more clearly than the literal PC index.", "fig:f8")}

{tex_table(ctx["f8_table"], "F8 PC stability, explained variance, FC predictability, and sibling AUC.", "tab:f8pcs", max_rows=20)}

{tex_table(ctx["f8_enrich"], "F8 enrichment of the mechanism mode in visual and dorsal-attention network pairs.", "tab:f8enrich", max_rows=12)}

The hedging is not weakness; it is the scientific point. The analysis already caught a
single-seed false positive in an earlier PC label. V2 therefore treats F8 as a forward map
for future work rather than a load-bearing claim. {esc(source_refs("f8_per_pc", "f8_stability", "f8_enrichment", "f8_localization", "tract_check"))}

\section{{F9: Richer Tractography Representation Does Not Help}}

\textbf{{Status: Glasser/exploratory, but internally strong.}} F9 asks whether SC counts
are simply too crude. The answer in the current data is no. The named-bundle r2t
representation predicts FC worse than raw streamline counts, adds no marginal signal on
top of count-SC, and fails downstream cognition. This closes the most obvious "better
structural features would rescue the story" escape route for the tested representation.

{fig_tex(ctx["f9_fig"], "F9 tractography sanity check. Count-SC beats r2t bundle features for FC prediction, and all structural/tractography representations fall below the bv+demo cognition floor.", "fig:f9")}

{tex_table(ctx["f9_table"], "F9 source representation prediction of FC, median across seeds.", "tab:f9e1", max_rows=None)}

{tex_table(ctx["f9_down"], "F9 downstream cognition for tractography representations.", "tab:f9e5", max_rows=21)}

Because F9 is Glasser-only, it should not be overclaimed as a universal tractography
result. But it is exactly the right sanity check for this preprint: richer structural
features do not recover FC's cognition signal in the tested regime. {esc(source_refs("tract_e1", "tract_e2", "tract_e3", "tract_e5", "tract_findings"))}

\section{{F10: Nonlinear Models and More Data Do Not Break the Ceiling}}

\textbf{{Status: Glasser/exploratory, but broad within that scope.}} F10 closes the
"bigger model" and "more data in this regime" escape routes. Kernel ridge preserves FC's
known cognition signal but does not unlock tractography. Residual boosting gives the
nonlinear model the linear answer for free and still finds no cognition residual.
Cross-modal sink models dilute FC rather than improve it. The sample-size curve shows the
nonlinear gap does not grow into positive territory through the tested n.

{fig_tex(ctx["f10_fig"], "F10 nonlinear and scaling checks. KernelRidge preserves the asymmetry but does not unlock tractography cognition. The residual nonlinear gap stays at or below zero as n increases.", "fig:f10")}

{tex_table(ctx["f10_table"], "F10 residual-boost reconstruction improvements. The small positive reconstruction sliver is far below the practical threshold.", "tab:f10n4", max_rows=18)}

{tex_table(ctx["f10_scaling"], "F10 sample-size scaling summary.", "tab:f10n6", max_rows=None)}

F10 should be framed as a structural ceiling for this HCP-YA, normal-range cognition,
cross-sectional setting. It does not prove there is no signal in clinical, longitudinal,
or much larger cohorts. It does prove that the ordinary escape routes do not explain the
current null. {esc(source_refs("nl_n1", "nl_n2", "nl_n3", "nl_n4_cog", "nl_n4_recon", "nl_n5_cog", "nl_n5_recon", "nl_n6", "nl_findings", "residual_findings", "scaling_findings", "noise_findings"))}

\section{{Cross-Cutting Sanity Checks}}

Several sanity checks are not standalone findings but protect the ledger. The
reduction-axis analysis shows F1 is not a PCA artifact. The FC reliability analysis shows
the SC->FC gap is not explained by per-subject FC reliability. The leak verdict grid shows
zero genuine leak failures.

{fig_tex(ctx["sanity_fig"], "Cross-cutting sanity checks. Reduction choice preserves the F1 ratio; per-subject FC reliability does not explain SC->FC achieved performance; reliability filtering does not rescue SC.", "fig:sanity")}

{tex_table(ctx["red_table"], "Reduction-axis robustness supporting F1.", "tab:reduction", max_rows=None)}

{tex_table(ctx["noise_table"], "Per-subject FC achieved-vs-ceiling correlations supporting F10.", "tab:noise", max_rows=None)}

\section{{Limitations and Exact Next Experiments}}

\begin{{itemize}}
\item \textbf{{SC-side reliability remains data-blocked.}} FC reliability is well
measured, but a true SC test-retest or tractography perturbation ceiling is still needed.
\item \textbf{{F9 and F10 should be ported to both parcellations.}} The Glasser results
are internally coherent, but the v2 ledger should label them exploratory until the same
nonlinear/tractography checks run under 4S456Parcels.
\item \textbf{{F8 needs property-selected localization.}} The mechanism mode should be
selected by properties such as non-confounded FC predictability and family signal, not by
hard-coded PC index.
\item \textbf{{Objective interpolation is the cleanest new experiment.}} Train along a
ladder from reconstruction to cognition to fingerprint/family objectives, then ask where
the signals diverge.
\item \textbf{{External validity is not solved.}} HCP-YA is healthy, cross-sectional, and
normal-range cognition. The protocol should be repeated in clinical, longitudinal, or
larger cohorts with the bv+demo baseline carried along.
\end{{itemize}}

\section{{Conclusion}}

The F1--F10 ledger supports a sharper, more disciplined manuscript than the compact draft.
The data do not say that cross-modal prediction is impossible. They say that cross-modal
prediction is directional, objectively real, and biologically structured, but that
reconstruction is the wrong proxy for downstream cognition. Observed FC is the useful
cognition modality in this regime. SC and richer structural representations are valuable
for other biological questions, including family structure and mechanism hypotheses, but
they do not transfer FC's cognitive utility through imputation. The scientific redirect is
therefore concrete: report bv+demo, stop treating reconstruction as utility, and move any
future cognition claim to regimes where structural variation plausibly exceeds the healthy
young-adult floor.

\end{{document}}
"""


def main() -> None:
    FIG.mkdir(exist_ok=True)
    MPLCONFIG.mkdir(exist_ok=True)
    ctx = make_figures_and_tables()
    TEX.write_text(document(ctx))
    print(f"Wrote {TEX}")
    print(f"Wrote {MANIFEST}")
    print(f"Wrote figures to {FIG}")


if __name__ == "__main__":
    main()
