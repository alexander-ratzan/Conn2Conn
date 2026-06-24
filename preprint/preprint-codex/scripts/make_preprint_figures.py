#!/usr/bin/env python3
"""Generate manuscript-facing Conn2Conn figures.

This script is intentionally self-contained and writes only inside
preprint/preprint-codex/figures. Source CSVs elsewhere in the repository are
read-only inputs.
"""
from __future__ import annotations

from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


SCRIPT = Path(__file__).resolve()
CODEX_DIR = SCRIPT.parents[1]
ROOT = SCRIPT.parents[3]
FIG_DIR = CODEX_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RECON = ROOT / "reproduction" / "outputs" / "reconstruction.csv"
DOWNSTREAM = ROOT / "reproduction" / "outputs" / "downstream.csv"
FAMILY_AUC = ROOT / "reproduction" / "family_mechanism" / "outputs" / "family_auc.csv"
F8_STABILITY = ROOT / "reproduction" / "family_mechanism" / "outputs" / "f8_stability.csv"
REDUCTION = (
    ROOT
    / "notebooks-FC_to_SC-experimental"
    / "sanity_checks"
    / "preprocessing_check"
    / "reduction_axis_synthesis.csv"
)
TRACT_E1 = (
    ROOT
    / "notebooks-FC_to_SC-experimental"
    / "tractography_predict"
    / "e1_source_rep_results.csv"
)
NONLIN_N2 = (
    ROOT
    / "notebooks-FC_to_SC-experimental"
    / "non-linear-sanity-check"
    / "n2_reconstruction_summary.csv"
)
SCALING = (
    ROOT
    / "notebooks-FC_to_SC-experimental"
    / "non-linear-sanity-check"
    / "n6_scaling_summary.csv"
)
NOISE_RELIABILITY = (
    ROOT
    / "notebooks-FC_to_SC-experimental"
    / "sanity_checks"
    / "noise_sanity_check"
    / "outputs"
    / "a_reliability_ceiling.csv"
)
NOISE_FILTER = (
    ROOT
    / "notebooks-FC_to_SC-experimental"
    / "sanity_checks"
    / "noise_sanity_check"
    / "outputs"
    / "h_reliability_filtered_summary.csv"
)
NOISE_CORR = (
    ROOT
    / "notebooks-FC_to_SC-experimental"
    / "sanity_checks"
    / "noise_sanity_check"
    / "outputs"
    / "h_correlations.csv"
)

PARC_ORDER = ["Glasser", "4S456Parcels"]
PARC_LABEL = {"Glasser": "Glasser", "4S456Parcels": "4S456"}
COG_ORDER = ["CogTotal", "CogFluid", "CogCryst"]
COG_LABEL = {"CogTotal": "Total", "CogFluid": "Fluid", "CogCryst": "Cryst"}

COL = {
    "fc": "#3b7fb6",
    "sc": "#c95f3e",
    "bv": "#6c9a3e",
    "demo": "#8b6fb4",
    "base": "#6e7681",
    "gold": "#d49c2f",
    "dark": "#273043",
    "pale": "#eef2f6",
    "green": "#2f8c6b",
    "red": "#b84a51",
}


def load() -> dict[str, pd.DataFrame]:
    return {
        "recon": pd.read_csv(RECON),
        "downstream": pd.read_csv(DOWNSTREAM),
        "family_auc": pd.read_csv(FAMILY_AUC),
        "f8": pd.read_csv(F8_STABILITY),
        "reduction": pd.read_csv(REDUCTION),
        "tract": pd.read_csv(TRACT_E1),
        "nonlin": pd.read_csv(NONLIN_N2),
        "scaling": pd.read_csv(SCALING),
        "noise_reliability": pd.read_csv(NOISE_RELIABILITY),
        "noise_filter": pd.read_csv(NOISE_FILTER),
        "noise_corr": pd.read_csv(NOISE_CORR),
    }


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "font.family": "DejaVu Sans",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=240)
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.13,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def seed_values(
    df: pd.DataFrame,
    parc: str,
    estimator: str,
    input_set: str,
    target: str,
    metric: str,
) -> np.ndarray:
    q = df[
        (df.parcellation == parc)
        & (df.estimator == estimator)
        & (df.input_set == input_set)
        & (df.target == target)
    ]
    if estimator in {"pca_pls", "bayesian_ridge"}:
        q = q[q.variant == estimator]
    values = q[metric].to_numpy(dtype=float)
    return values[np.isfinite(values)]


def mean_std(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], text: str, color: str) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.0,
        facecolor=color,
        edgecolor="#2f3542",
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        "\n".join(textwrap.wrap(text, 23)),
        ha="center",
        va="center",
        fontsize=9,
        color="#111827",
    )


def arrow(ax: plt.Axes, xy0: tuple[float, float], xy1: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=xy1,
        xytext=xy0,
        arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#293241", shrinkA=4, shrinkB=4),
    )


def fig1_evaluation_spine(_: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.03, 0.96, "Figure 1. Evaluation spine", fontsize=15, fontweight="bold")
    ax.text(
        0.03,
        0.915,
        "The design separates reconstruction from downstream utility and treats bv+demo as the utility checkpoint.",
        fontsize=10,
        color="#394150",
    )

    box(ax, (0.04, 0.70), (0.16, 0.12), "Observed FC edge vectors", "#dceefa")
    box(ax, (0.04, 0.52), (0.16, 0.12), "Observed SC edge vectors", "#fde2d4")
    box(ax, (0.04, 0.34), (0.16, 0.12), "Brain volumes + demographics", "#e8eadc")

    box(ax, (0.29, 0.57), (0.17, 0.14), "10 frozen family-aware splits", "#f1f5f9")
    box(ax, (0.29, 0.37), (0.17, 0.12), "Two parcellations: Glasser and 4S456", "#f1f5f9")

    for y0 in [0.76, 0.58, 0.40]:
        arrow(ax, (0.20, y0), (0.29, 0.64 if y0 > 0.45 else 0.43))
    arrow(ax, (0.375, 0.57), (0.375, 0.49))

    box(ax, (0.56, 0.71), (0.18, 0.13), "Reconstruction: FC->SC, SC->FC, oracles", "#e5f0fb")
    box(ax, (0.56, 0.51), (0.18, 0.13), "Downstream cognition: lift over bv+demo", "#fff2cf")
    box(ax, (0.56, 0.31), (0.18, 0.13), "Reviewer defenses: reduction, nonlinear, noise", "#e9f4ec")
    box(ax, (0.56, 0.13), (0.18, 0.13), "Family signal: identification objective", "#efe7f8")

    for y1 in [0.775, 0.575, 0.375, 0.195]:
        arrow(ax, (0.46, 0.63 if y1 > 0.45 else 0.43), (0.56, y1))

    box(ax, (0.82, 0.71), (0.15, 0.13), "Metric: demeaned Pearson", "#f8fafc")
    box(ax, (0.82, 0.51), (0.15, 0.13), "Estimator: BayesianRidge for scalar targets", "#f8fafc")
    box(ax, (0.82, 0.31), (0.15, 0.13), "Claim status: escape route closed or caveated", "#f8fafc")
    box(ax, (0.82, 0.13), (0.15, 0.13), "AUC by relation and objective", "#f8fafc")

    for y in [0.775, 0.575, 0.375, 0.195]:
        arrow(ax, (0.74, y), (0.82, y))

    save(fig, "fig1_evaluation_spine")


def fig2_directional_translation(data: dict[str, pd.DataFrame]) -> None:
    r = data["recon"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # A. Directional reconstruction.
    x = np.arange(len(PARC_ORDER))
    width = 0.34
    for offset, src, tgt, color, label in [
        (-width / 2, "FC", "SC", COL["fc"], "FC -> SC"),
        (width / 2, "SC", "FC", COL["sc"], "SC -> FC"),
    ]:
        means, errs = [], []
        for parc in PARC_ORDER:
            vals = seed_values(r, parc, "pca_pls", src, tgt, "demeaned_pearson")
            m, s = mean_std(vals)
            means.append(m)
            errs.append(s)
            jitter = np.linspace(-0.055, 0.055, len(vals))
            ax_a.scatter(
                np.full_like(vals, x[PARC_ORDER.index(parc)] + offset) + jitter,
                vals,
                s=17,
                color=color,
                alpha=0.55,
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )
        ax_a.bar(x + offset, means, width, yerr=errs, capsize=3, color=color, alpha=0.78, label=label)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([PARC_LABEL[p] for p in PARC_ORDER])
    ax_a.set_ylabel("demeaned Pearson")
    ax_a.set_title("Cross-modal reconstruction is directional")
    ax_a.legend(frameon=False, loc="upper left")
    panel_label(ax_a, "A")

    # B. Ratio by seed.
    ratio_data = []
    for parc in PARC_ORDER:
        fcsc = seed_values(r, parc, "pca_pls", "FC", "SC", "demeaned_pearson")
        scfc = seed_values(r, parc, "pca_pls", "SC", "FC", "demeaned_pearson")
        ratio_data.append(fcsc / scfc)
    bp = ax_b.boxplot(ratio_data, patch_artist=True, tick_labels=[PARC_LABEL[p] for p in PARC_ORDER])
    for patch, color in zip(bp["boxes"], [COL["fc"], COL["gold"]]):
        patch.set(facecolor=color, alpha=0.35, edgecolor="#293241")
    for i, vals in enumerate(ratio_data, start=1):
        ax_b.scatter(np.full_like(vals, i) + np.linspace(-0.06, 0.06, len(vals)), vals, s=18, color="#293241")
        ax_b.text(i, np.max(vals) + 0.05, f"mean {np.mean(vals):.2f}x", ha="center", fontsize=9)
    ax_b.axhline(1.0, color="#6b7280", lw=1, ls="--")
    ax_b.set_ylabel("FC->SC / SC->FC")
    ax_b.set_title("The ratio is stable across frozen splits")
    panel_label(ax_b, "B")

    # C. Same-modality oracle.
    width = 0.34
    for offset, src_tgt, color, label in [
        (-width / 2, ("FC", "FC"), COL["fc"], "FC -> FC"),
        (width / 2, ("SC", "SC"), COL["sc"], "SC -> SC"),
    ]:
        means, errs = [], []
        for parc in PARC_ORDER:
            vals = seed_values(r, parc, "bayesian_ridge", src_tgt[0], src_tgt[1], "demeaned_pearson")
            m, s = mean_std(vals)
            means.append(m)
            errs.append(s)
        ax_c.bar(x + offset, means, width, yerr=errs, capsize=3, color=color, alpha=0.72, label=label)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([PARC_LABEL[p] for p in PARC_ORDER])
    ax_c.set_ylabel("demeaned Pearson")
    ax_c.set_title("Within-modality targets have similar oracle ceilings")
    ax_c.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.03), ncol=2)
    panel_label(ax_c, "C")

    # D. Anatomy/demo double dissociation.
    labels = []
    matrix = []
    for parc in PARC_ORDER:
        for tgt in ["SC", "FC"]:
            labels.append(f"{PARC_LABEL[parc]} -> {tgt}")
            row = []
            for inp in ["bv", "demo"]:
                row.append(np.mean(seed_values(r, parc, "pca_pls", inp, tgt, "demeaned_pearson")))
            matrix.append(row)
    matrix = np.asarray(matrix)
    im = ax_d.imshow(matrix, cmap="YlGnBu", vmin=0.03, vmax=0.20, aspect="auto")
    ax_d.set_xticks([0, 1])
    ax_d.set_xticklabels(["brain volumes", "demographics"])
    ax_d.set_yticks(range(len(labels)))
    ax_d.set_yticklabels(labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax_d.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=9)
    ax_d.set_title("Subject information is not one generic confound")
    fig.colorbar(im, ax=ax_d, fraction=0.045, pad=0.02)
    panel_label(ax_d, "D")

    fig.suptitle("Figure 2. Directional translation", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout()
    save(fig, "fig2_directional_translation")


def fig3_utility_checkpoint(data: dict[str, pd.DataFrame]) -> None:
    d = data["downstream"]
    br = d[(d.estimator == "bayesian_ridge") & (d.variant == "bayesian_ridge")]
    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 1.0], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[:, 2])
    ax_d = fig.add_subplot(gs[1, 0:2])

    # A. Raw CogCryst Pearson: the reader-facing cognition bar.
    focus_inputs = ["bv+demo", "obs_FC", "obs_SC", "pred_SC", "pred_FC", "obs_FC+bv+demo"]
    focus_labels = ["bv+demo", "obs FC", "obs SC", "pred SC", "pred FC", "obs FC+bv+demo"]
    color_map = {
        "bv+demo": COL["base"],
        "obs_FC": COL["fc"],
        "obs_SC": COL["sc"],
        "pred_SC": "#8fb8de",
        "pred_FC": "#e3a08b",
        "obs_FC+bv+demo": COL["gold"],
    }
    xpos = np.arange(len(focus_inputs))
    w = 0.36
    for j, parc in enumerate(PARC_ORDER):
        vals = []
        errs = []
        for inp in focus_inputs:
            s = br[(br.parcellation == parc) & (br.input_set == inp) & (br.target == "CogCryst")][
                "pearson"
            ].to_numpy(float)
            m, sd = mean_std(s)
            vals.append(m)
            errs.append(sd)
        ax_a.bar(
            xpos + (j - 0.5) * w,
            vals,
            w,
            yerr=errs,
            capsize=2,
            label=PARC_LABEL[parc],
            color=[color_map[i] for i in focus_inputs],
            alpha=0.82 if j == 0 else 0.48,
            edgecolor="#293241",
            linewidth=0.4,
        )
    ax_a.set_xticks(xpos)
    ax_a.set_xticklabels(focus_labels, rotation=35, ha="right")
    ax_a.set_ylabel("test Pearson, CogCryst")
    ax_a.set_title("Observed FC clears the cognition bar")
    ax_a.legend(frameon=False)
    panel_label(ax_a, "A")

    # B. CogCryst lift with paired-permutation p labels.
    lift_inputs = ["obs_FC", "obs_SC", "pred_SC", "pred_FC", "obs_FC+bv+demo", "pred_SC+bv+demo"]
    lift_labels = ["obs FC", "obs SC", "pred SC", "pred FC", "obs FC+\nbv+demo", "pred SC+\nbv+demo"]
    means = []
    pvals = []
    for inp in lift_inputs:
        q = br[(br.input_set == inp) & (br.target == "CogCryst")]
        means.append(q.groupby("parcellation")["lift_over_bvdemo"].mean().reindex(PARC_ORDER).mean())
        pvals.append(q.groupby("parcellation")["lift_perm_p"].median().reindex(PARC_ORDER).mean())
    bars = ax_b.bar(np.arange(len(lift_inputs)), means, color=[color_map.get(i, "#9db4c0") for i in lift_inputs])
    ax_b.axhline(0, color="#293241", lw=1)
    ax_b.set_ylim(-0.14, 0.20)
    for i, (bar, p) in enumerate(zip(bars, pvals)):
        y = bar.get_height()
        ax_b.text(
            bar.get_x() + bar.get_width() / 2,
            y + (0.012 if y >= 0 else -0.018),
            f"p~{p:.3f}",
            ha="center",
            va="bottom" if y >= 0 else "top",
            fontsize=8,
        )
    ax_b.set_xticks(np.arange(len(lift_inputs)))
    ax_b.set_xticklabels(lift_labels, rotation=25, ha="right")
    ax_b.set_ylabel("lift over bv+demo")
    ax_b.set_title("Utility is judged against subject information")
    panel_label(ax_b, "B")

    # C. Lift heatmap across all cognitive outcomes and parcellations.
    heat_inputs = ["obs_FC", "obs_SC", "pred_SC", "pred_FC", "obs_FC+bv+demo", "pred_FC+bv+demo"]
    rows = []
    ylabels = []
    for inp in heat_inputs:
        ylabels.append(inp.replace("_", " "))
        vals = []
        for parc in PARC_ORDER:
            for cog in COG_ORDER:
                vals.append(
                    br[
                        (br.parcellation == parc)
                        & (br.input_set == inp)
                        & (br.target == cog)
                    ]["lift_over_bvdemo"].mean()
                )
        rows.append(vals)
    M = np.asarray(rows)
    im = ax_c.imshow(M, cmap="RdBu_r", vmin=-0.16, vmax=0.16, aspect="auto")
    ax_c.set_yticks(np.arange(len(ylabels)))
    ax_c.set_yticklabels(ylabels)
    xlabels = [f"{PARC_LABEL[p]}\n{COG_LABEL[c]}" for p in PARC_ORDER for c in COG_ORDER]
    ax_c.set_xticks(np.arange(len(xlabels)))
    ax_c.set_xticklabels(xlabels, rotation=45, ha="right")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax_c.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    ax_c.set_title("Lift landscape")
    fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.02)
    panel_label(ax_c, "C")

    # D. Imputation contrast by target.
    idx = np.arange(len(COG_ORDER))
    for offset, inp, color, label in [
        (-0.27, "obs_FC", COL["fc"], "observed FC"),
        (-0.09, "obs_SC", COL["sc"], "observed SC"),
        (0.09, "pred_SC", "#8fb8de", "SC imputed from FC"),
        (0.27, "pred_FC", "#e3a08b", "FC imputed from SC"),
    ]:
        vals = [
            br[(br.input_set == inp) & (br.target == cog)]["lift_over_bvdemo"].mean()
            for cog in COG_ORDER
        ]
        ax_d.bar(idx + offset, vals, 0.18, color=color, label=label, alpha=0.86)
    ax_d.axhline(0, color="#293241", lw=1)
    ax_d.set_xticks(idx)
    ax_d.set_xticklabels([COG_LABEL[c] for c in COG_ORDER])
    ax_d.set_ylabel("mean lift over bv+demo")
    ax_d.set_title("Imputation does not transfer FC's downstream signal through SC")
    ax_d.legend(frameon=False, ncol=2, fontsize=8)
    panel_label(ax_d, "D")

    fig.suptitle("Figure 3. Utility checkpoint", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout()
    save(fig, "fig3_utility_checkpoint")


def fig4_closed_escape_routes(data: dict[str, pd.DataFrame]) -> None:
    r = data["recon"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.4))
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.ravel()

    # A. Cross-modal fraction of same-modality oracle.
    labels, frac = [], []
    for parc in PARC_ORDER:
        for src, tgt in [("FC", "SC"), ("SC", "FC")]:
            cross = np.mean(seed_values(r, parc, "pca_pls", src, tgt, "demeaned_pearson"))
            oracle = np.mean(seed_values(r, parc, "bayesian_ridge", tgt, tgt, "demeaned_pearson"))
            labels.append(f"{PARC_LABEL[parc]}\n{src}->{tgt}")
            frac.append(cross / oracle)
    ax_a.bar(np.arange(len(frac)), frac, color=[COL["fc"], COL["sc"], COL["fc"], COL["sc"]], alpha=0.82)
    ax_a.set_xticks(np.arange(len(frac)))
    ax_a.set_xticklabels(labels)
    ax_a.set_ylim(0, 0.32)
    ax_a.set_ylabel("cross-modal / oracle")
    for i, y in enumerate(frac):
        ax_a.text(i, y + 0.012, f"{100*y:.0f}%", ha="center", fontsize=9)
    ax_a.set_title("Real signal, far below same-modality oracle")
    panel_label(ax_a, "A")

    # B. Reduction-axis robustness.
    red = data["reduction"].copy()
    red["method_label"] = red["method"].where(red["jl_variant"].isna(), red["jl_variant"])
    method_order = ["FULL_PLS", "PCA_PLS_PCA", "gaussian_dense", "sparse_auto", "sparse_third"]
    method_label = {
        "FULL_PLS": "full PLS",
        "PCA_PLS_PCA": "PCA->PLS",
        "gaussian_dense": "JL dense",
        "sparse_auto": "JL sparse",
        "sparse_third": "JL 1/3",
    }
    plot_data = [red.loc[red.method_label == m, "ratio"].dropna().to_numpy(float) for m in method_order]
    bp = ax_b.boxplot(plot_data, patch_artist=True, tick_labels=[method_label[m] for m in method_order])
    for patch in bp["boxes"]:
        patch.set(facecolor="#dfe8f3", edgecolor="#293241")
    for i, vals in enumerate(plot_data, start=1):
        ax_b.scatter(np.full_like(vals, i) + np.linspace(-0.05, 0.05, len(vals)), vals, s=13, color=COL["dark"], alpha=0.65)
        ax_b.text(i, np.median(vals) + 0.12, f"{np.median(vals):.2f}x", ha="center", fontsize=8)
    ax_b.axhline(1.0, color="#6b7280", ls="--", lw=1)
    ax_b.set_ylabel("FC->SC / SC->FC")
    ax_b.set_title("Asymmetry survives reduction choices")
    panel_label(ax_b, "B")

    # C. Richer tractography representations.
    tract = data["tract"]
    reps = ["SC", "r2t", "r2t_corr", "SC_r2t", "kitchen_sink"]
    rep_labels = ["count SC", "bundle r2t", "bundle sim", "SC+r2t", "sink"]
    vals = [tract.loc[tract.rep == rep, "demeaned_pearson"].median() for rep in reps]
    colors = [COL["sc"], "#9fb5c8", "#b8c6d2", "#cab2d6", "#8dd3c7"]
    ax_c.bar(np.arange(len(reps)), vals, color=colors)
    ax_c.set_xticks(np.arange(len(reps)))
    ax_c.set_xticklabels(rep_labels, rotation=25, ha="right")
    ax_c.set_ylabel("median demeaned Pearson, -> FC")
    ax_c.set_title("Richer structural representations do not rescue FC")
    panel_label(ax_c, "C")

    # D. Nonlinear reconstruction gain.
    nonlin = data["nonlin"]
    q = nonlin[(nonlin.rep == "SC") & (nonlin.metric == "demeaned_pearson")]
    pls = q[q.estimator == "linear_PLS"].iloc[0]
    kr = q[q.estimator == "KR"].iloc[0]
    gains = [
        kr.median_FC_to_X - pls.median_FC_to_X,
        kr.median_X_to_FC - pls.median_X_to_FC,
    ]
    ax_d.bar([0, 1], gains, color=[COL["fc"], COL["sc"]], alpha=0.85)
    ax_d.axhline(0, color="#293241", lw=1)
    ax_d.axhspan(-0.0025, 0.0025, color="#edf2f7", zorder=-1)
    ax_d.set_xticks([0, 1])
    ax_d.set_xticklabels(["FC->SC", "SC->FC"])
    ax_d.set_ylabel("KernelRidge - linear PLS")
    ax_d.set_title("Nonlinear probe changes almost nothing")
    for i, y in enumerate(gains):
        ax_d.text(i, y + (0.00035 if y >= 0 else -0.00035), f"{y:+.4f}", ha="center", va="bottom" if y >= 0 else "top", fontsize=9)
    panel_label(ax_d, "D")

    # E. Scaling: nonlinear gap as n grows.
    scaling = data["scaling"].copy()
    scaling = scaling[scaling.n_sub != 682]
    for task, color in [("cognition", COL["gold"]), ("reconstruction", COL["green"])]:
        q = scaling[scaling.task == task].sort_values("n_sub")
        ax_e.plot(q.n_sub, q.median_gap, marker="o", color=color, label=task)
    ax_e.axhline(0, color="#293241", lw=1)
    ax_e.set_xticks([100, 200, 400, 683])
    ax_e.set_ylabel("nonlinear residual gap")
    ax_e.set_xlabel("training subjects")
    ax_e.set_title("No growing nonlinear gap with sample size")
    ax_e.legend(frameon=False, fontsize=8)
    panel_label(ax_e, "E")

    # F. FC noise accounting.
    rel = data["noise_reliability"]
    ceiling = float(
        rel[(rel.parc == "Glasser") & (rel.comparison == "between_session")]["demeaned_pearson"].iloc[0]
    )
    filt = data["noise_filter"]
    sc_all = float(filt[(filt.source == "SC") & (filt["filter"] == "all")]["mean_achieved"].iloc[0])
    bvdemo_all = float(filt[(filt.source == "bv+demo") & (filt["filter"] == "all")]["mean_achieved"].iloc[0])
    corr = float(data["noise_corr"].loc[data["noise_corr"].source == "SC", "pearson_achieved_vs_ceiling"].iloc[0])
    names = ["FC test-retest", "SC->FC", "bv+demo->FC"]
    vals = [ceiling, sc_all, bvdemo_all]
    ax_f.bar(np.arange(3), vals, color=["#9aa6b2", COL["sc"], COL["base"]], alpha=0.85)
    ax_f.set_xticks(np.arange(3))
    ax_f.set_xticklabels(names, rotation=20, ha="right")
    ax_f.set_ylabel("demeaned Pearson")
    ax_f.set_title("SC->FC is not limited by FC-side noise")
    ax_f.text(1, sc_all + 0.03, f"{100*sc_all/ceiling:.0f}% of ceiling", ha="center", fontsize=9)
    ax_f.text(1.9, max(vals) * 0.88, f"per-subject r={corr:.2f}", fontsize=9, color="#293241")
    panel_label(ax_f, "F")

    fig.suptitle("Figure 4. Closed escape routes", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout()
    save(fig, "fig4_closed_escape_routes")


def fig5_objective_signal(data: dict[str, pd.DataFrame]) -> None:
    fam = data["family_auc"]
    f8 = data["f8"]
    r = data["recon"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # A. Sibling AUC by objective-relevant predicted SC variants.
    variants = ["bvdemo_to_SC", "pred_SC_raw", "pred_SC_resid_bvdemo", "combined_pred_SC", "obs_SC"]
    labels = ["bv+demo\nbaseline", "raw pred SC", "residual\npred SC", "combined\npred SC", "observed SC"]
    colors = [COL["base"], "#8fb8de", COL["fc"], COL["gold"], COL["sc"]]
    x = np.arange(len(variants))
    w = 0.36
    for j, parc in enumerate(PARC_ORDER):
        vals = [
            fam[
                (fam.parcellation == parc) & (fam.variant == variant) & (fam.relation == "sibling")
            ]["auc"].iloc[0]
            for variant in variants
        ]
        ax_a.bar(x + (j - 0.5) * w, vals, w, color=colors, alpha=0.84 if j == 0 else 0.52, label=PARC_LABEL[parc])
    ax_a.axhline(0.5, color="#293241", lw=1, ls="--")
    ax_a.set_ylim(0.45, 0.91)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels)
    ax_a.set_ylabel("sibling AUC")
    ax_a.set_title("Predicted SC can preserve family signal")
    ax_a.legend(frameon=False)
    panel_label(ax_a, "A")

    # B. Reconstruction-objective score versus family objective score.
    trade = []
    mappings = [
        ("bv+demo", "SC", "bvdemo_to_SC", "bv+demo"),
        ("FC", "SC", "pred_SC_raw", "FC->SC raw"),
        ("FC+bv+demo", "SC", "combined_pred_SC", "combined"),
    ]
    for parc in PARC_ORDER:
        for inp, tgt, variant, label in mappings:
            recon_score = np.mean(seed_values(r, parc, "pca_pls", inp, tgt, "demeaned_pearson"))
            auc = fam[
                (fam.parcellation == parc) & (fam.variant == variant) & (fam.relation == "sibling")
            ]["auc"].iloc[0]
            trade.append((parc, label, recon_score, auc))
    for parc, marker in zip(PARC_ORDER, ["o", "s"]):
        q = [row for row in trade if row[0] == parc]
        ax_b.scatter(
            [row[2] for row in q],
            [row[3] for row in q],
            s=85,
            marker=marker,
            label=PARC_LABEL[parc],
            color=[COL["base"], COL["fc"], COL["gold"]],
            edgecolor="#293241",
        )
        for _, label, xs, ys in q:
            ax_b.text(xs + 0.002, ys + 0.006, label, fontsize=8)
    ax_b.axhline(0.5, color="#293241", lw=1, ls="--")
    ax_b.set_xlabel("reconstruction demeaned Pearson")
    ax_b.set_ylabel("sibling AUC")
    ax_b.set_title("Reconstruction and identification select different signal")
    ax_b.legend(frameon=False, fontsize=8)
    panel_label(ax_b, "B")

    # C. Property-selected mechanism mode.
    for parc, color in [("Glasser", COL["fc"]), ("4S456Parcels", COL["green"])]:
        q = f8[f8.parcellation == parc]
        ax_c.scatter(
            q.median_FC_to_PC_R2,
            q.median_AUC_sibling,
            s=80 + q.median_expl_var * 3600,
            color=color,
            alpha=0.65,
            edgecolor="#293241",
            label=PARC_LABEL[parc],
        )
        selected_pc = 3 if parc == "Glasser" else 4
        sel = q[q.anchor_pc == selected_pc].iloc[0]
        ax_c.scatter(
            [sel.median_FC_to_PC_R2],
            [sel.median_AUC_sibling],
            s=230,
            facecolors="none",
            edgecolors=COL["red"],
            linewidth=2.0,
        )
        ax_c.text(
            sel.median_FC_to_PC_R2 + 0.012,
            sel.median_AUC_sibling + 0.006,
            f"{PARC_LABEL[parc]} PC{selected_pc}",
            fontsize=9,
            fontweight="bold",
        )
    ax_c.axhline(0.5, color="#293241", lw=1, ls="--")
    ax_c.set_xlabel("FC predictability of SC PC")
    ax_c.set_ylabel("sibling AUC")
    ax_c.set_title("A low-variance FC-predictable mode carries family signal")
    ax_c.legend(frameon=False)
    panel_label(ax_c, "C")

    # D. PC summary: PC1 is confounded; selected modes are smaller and useful.
    summary_rows = []
    for parc in PARC_ORDER:
        for pc in [1, 3, 4]:
            row = f8[(f8.parcellation == parc) & (f8.anchor_pc == pc)]
            if not row.empty:
                row = row.iloc[0]
                summary_rows.append(
                    [
                        f"{PARC_LABEL[parc]} PC{pc}",
                        row.median_expl_var,
                        row.median_FC_to_PC_R2,
                        row.median_AUC_sibling,
                    ]
                )
    table = ax_d.table(
        cellText=[[f"{a}", f"{b:.3f}", f"{c:.3f}", f"{d:.3f}"] for a, b, c, d in summary_rows],
        colLabels=["mode", "var", "FC R2", "sib AUC"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.35)
    ax_d.set_axis_off()
    ax_d.set_title("Mechanism result is property-selected, not fixed-index")
    panel_label(ax_d, "D")

    fig.suptitle("Figure 5. Signal changes with objective", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout()
    save(fig, "fig5_signal_changes_with_objective")


def write_manifest() -> None:
    manifest = CODEX_DIR / "figure_manifest.md"
    text = f"""# Preprint Figure Manifest

Generated with:

```bash
/Users/user/dev-env/bin/python preprint/preprint-codex/scripts/make_preprint_figures.py
```

Output directory:

`{FIG_DIR}`

## Main Figures

| Figure | File stem | Manuscript role |
|---|---|---|
| Figure 1 | `fig1_evaluation_spine` | Evaluation design, frozen splits, reconstruction vs utility branches, and sanity checks |
| Figure 2 | `fig2_directional_translation` | FC->SC vs SC->FC asymmetry, seed ratios, same-modality oracles, anatomy/demo double dissociation |
| Figure 3 | `fig3_utility_checkpoint` | Cognition prediction and lift over `bv+demo`, observed vs imputed connectome contrast |
| Figure 4 | `fig4_closed_escape_routes` | Oracle fractions, reduction robustness, tractography/null nonlinear checks, scaling, and FC-noise accounting |
| Figure 5 | `fig5_signal_changes_with_objective` | Family AUCs, reconstruction/identification tradeoff, and property-selected mechanism mode |

Each figure is written as both PNG and PDF.

## Read-Only Sources

- `reproduction/outputs/reconstruction.csv`
- `reproduction/outputs/downstream.csv`
- `reproduction/family_mechanism/outputs/family_auc.csv`
- `reproduction/family_mechanism/outputs/f8_stability.csv`
- `notebooks-FC_to_SC-experimental/sanity_checks/preprocessing_check/reduction_axis_synthesis.csv`
- `notebooks-FC_to_SC-experimental/tractography_predict/e1_source_rep_results.csv`
- `notebooks-FC_to_SC-experimental/non-linear-sanity-check/n2_reconstruction_summary.csv`
- `notebooks-FC_to_SC-experimental/non-linear-sanity-check/n6_scaling_summary.csv`
- `notebooks-FC_to_SC-experimental/sanity_checks/noise_sanity_check/outputs/a_reliability_ceiling.csv`
- `notebooks-FC_to_SC-experimental/sanity_checks/noise_sanity_check/outputs/h_reliability_filtered_summary.csv`
- `notebooks-FC_to_SC-experimental/sanity_checks/noise_sanity_check/outputs/h_correlations.csv`

## Notes

- The figures are manuscript-facing composites, not replacements for the exploratory figures in `reproduction/exploration/figures`.
- Figure 5 follows the manuscript caveat: the mechanism panel highlights Glasser PC3 and 4S456 PC4 as property-selected low-variance modes rather than treating one PC index as universal.
- Figure 4 keeps FC test-retest reliability as an FC-side accounting result and uses the same-modality oracle separately.
"""
    manifest.write_text(text)


def main() -> None:
    configure_style()
    data = load()
    fig1_evaluation_spine(data)
    fig2_directional_translation(data)
    fig3_utility_checkpoint(data)
    fig4_closed_escape_routes(data)
    fig5_objective_signal(data)
    write_manifest()
    print(f"Wrote preprint figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
