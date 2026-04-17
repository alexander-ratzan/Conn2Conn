"""Visualization and markdown report helpers for evaluator outputs."""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from scipy.stats import false_discovery_control, pearsonr, ttest_ind
from sklearn.metrics import r2_score

from data.data_utils import *
from models.eval.eval_utils import (
    extract_hungarian_metrics,
    extract_sample_size_metrics,
    extract_violin_metrics,
    generate_mean_baseline,
    generate_noise_baseline,
    generate_report,
)
from models.eval.fc_distance import (
    pairwise_fc_distance,
    prepare_fc_matrices,
    reconstruct_fc_matrices,
)
from models.eval.metrics import (
    compute_corr_matrix,
    compute_identifiability,
    corr_avg_rank,
    corr_topn_accuracy,
    distance_avg_rank,
    distance_top1_accuracy,
    hungarian_matching,
    hungarian_matching_subsample,
)
from models.eval.pca_analysis import compute_normalized_sse_pc


def generate_markdown_report(figures_dict, all_metrics, filepath, verbose=False):
    """
    Generate a Markdown report with embedded PNG figures.

    Args:
        figures_dict: dict mapping figure names to matplotlib figure objects
        all_metrics: dict with all computed metrics
        filepath: base path for output without extension
        verbose: whether to include verbose report sections
    """
    model_type = all_metrics.get("model_name")
    if not model_type:
        filepath_base = os.path.basename(filepath)
        if filepath_base.endswith("_results"):
            model_type = filepath_base[:-8]
        else:
            model_type = filepath_base

    plots_dir = f"{filepath}_plots"
    os.makedirs(plots_dir, exist_ok=True)
    plots_dirname = os.path.basename(plots_dir)

    saved_figures = {}
    for name, fig in figures_dict.items():
        if fig is not None:
            fig_path = os.path.join(plots_dir, f"{name}.png")
            fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
            saved_figures[name] = f"{plots_dirname}/{name}.png"
            plt.close(fig)

    md_lines = []

    md_lines.append("# FC Prediction Evaluation Report")
    md_lines.append("")
    md_lines.append(
        f"**Model:** {model_type} | "
        f"**Partition:** {all_metrics.get('partition', 'N/A')} | "
        f"**N subjects:** {all_metrics.get('n_subjects', 'N/A')} | "
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    md_lines.append("## Identifiability Heatmaps")
    md_lines.append("")
    md_lines.append(
        "Pairwise correlation matrices between predicted and target connectomes. "
        "Diagonal dominance indicates subject-specific predictions."
    )
    md_lines.append("")
    md_lines.append("**Top-1 Accuracy:** Fraction of subjects whose own prediction is the best match.")
    md_lines.append("")
    md_lines.append(r"$$\text{Top-1 Acc} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\arg\max_j \, r(\hat{y}_i, y_j) = i]$$")
    md_lines.append("")
    md_lines.append("**Avg Rank %ile:** For each subject, count how many other subjects' predictions have lower correlation with their target than their own prediction does.")
    md_lines.append("")
    md_lines.append(r"$$\text{avgrank} = \frac{1}{N}\sum_{s=1}^{N}\left(\frac{1}{N}\sum_{a \neq s}^{N} \mathbb{1}[r(X_s, \hat{X}_a) < r(X_s, \hat{X}_s)]\right)$$")
    md_lines.append("")

    if verbose and "identifiability_heatmaps" in saved_figures:
        md_lines.append("<table><tr>")
        md_lines.append(f'<td align="center"><b>Raw</b><br/><img src="{saved_figures.get("identifiability_heatmaps", "")}" width="450"/></td>')
        md_lines.append(f'<td align="center"><b>Demeaned</b><br/><img src="{saved_figures.get("identifiability_heatmaps_demeaned", "")}" width="450"/></td>')
        md_lines.append("</tr></table>")
    elif "identifiability_heatmaps_demeaned" in saved_figures:
        md_lines.append(f"![]({saved_figures['identifiability_heatmaps_demeaned']})")
    md_lines.append("")

    if verbose and "heatmaps_raw" in all_metrics:
        md_lines.append("| Metric | Raw | Demeaned |")
        md_lines.append("|--------|-----|----------|")
        raw = all_metrics.get("heatmaps_raw", {})
        dem = all_metrics.get("heatmaps_demeaned", {})
        md_lines.append(f"| Mean Corr | {raw.get('raw_mean_corr', raw.get('mean_corr', np.nan)):.3f} | {dem.get('raw_mean_corr', dem.get('mean_corr', np.nan)):.3f} |")
        md_lines.append(f"| Demeaned Mean Corr | {raw.get('demeaned_mean_corr', np.nan):.3f} | {dem.get('demeaned_mean_corr', np.nan):.3f} |")
        md_lines.append(f"| Top-1 Acc | {raw.get('top1_acc', '-'):.3f} | {dem.get('top1_acc', '-'):.3f} |")
        md_lines.append(f"| Avg Rank %ile | {raw.get('avg_rank_percentile', '-'):.3f} | {dem.get('avg_rank_percentile', '-'):.3f} |")
    elif "heatmaps_demeaned" in all_metrics:
        dem = all_metrics.get("heatmaps_demeaned", {})
        md_lines.append("| Metric | Value |")
        md_lines.append("|--------|-------|")
        md_lines.append(f"| Mean Corr | {dem.get('raw_mean_corr', dem.get('mean_corr', np.nan)):.3f} |")
        md_lines.append(f"| Demeaned Mean Corr | {dem.get('demeaned_mean_corr', np.nan):.3f} |")
        md_lines.append(f"| Top-1 Acc | {dem.get('top1_acc', '-'):.3f} |")
        md_lines.append(f"| Avg Rank %ile | {dem.get('avg_rank_percentile', '-'):.3f} |")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    if "geodesic_heatmaps" in saved_figures:
        md_lines.append("## SPD Geodesic Heatmaps")
        md_lines.append("")
        md_lines.append(
            "Pairwise distances between full reconstructed FC matrices after SPD projection. "
            "These plots use an SPD-aware matrix distance instead of edgewise correlation."
        )
        md_lines.append("")
        md_lines.append(f"![]({saved_figures['geodesic_heatmaps']})")
        md_lines.append("")
        geo = all_metrics.get("geodesic", {})
        md_lines.append("| Metric | Value |")
        md_lines.append("|--------|-------|")
        md_lines.append(f"| Method | {geo.get('method', '-')} |")
        md_lines.append(f"| Mean Self Distance | {geo.get('mean_self_distance', np.nan):.3f} |")
        md_lines.append(f"| Top-1 Acc | {geo.get('top1_acc', np.nan):.3f} |")
        md_lines.append(f"| Avg Rank %ile | {geo.get('avg_rank_percentile', np.nan):.3f} |")
        meta = geo.get("metadata", {})
        if isinstance(meta, dict):
            md_lines.append(f"| Target SPD Projection Rate | {meta.get('targets_fraction_projected', np.nan):.2%} |")
            md_lines.append(f"| Pred SPD Projection Rate | {meta.get('preds_fraction_projected', np.nan):.2%} |")
            md_lines.append(f"| Target Mean Clipped Eig Fraction | {meta.get('targets_mean_clipped_fraction', np.nan):.2%} |")
            md_lines.append(f"| Pred Mean Clipped Eig Fraction | {meta.get('preds_mean_clipped_fraction', np.nan):.2%} |")
            md_lines.append(f"| Target Mean Clip Mass | {meta.get('targets_mean_clip_mass', np.nan):.3f} |")
            md_lines.append(f"| Pred Mean Clip Mass | {meta.get('preds_mean_clip_mass', np.nan):.3f} |")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    md_lines.append("## Identifiability Violin")
    md_lines.append("")
    md_lines.append(
        "Tests whether intraindividual correlations (subject's prediction vs their own target) "
        "exceed interindividual correlations (vs other subjects' targets) using a **one-sample t-test**."
    )
    md_lines.append("")
    md_lines.append("For each subject $i$, compute $d_i = r_{intra}(i) - r_{inter}(i)$, then test:")
    md_lines.append("")
    md_lines.append(r"$$H_0: \frac{1}{N}\sum_i d_i = 0 \quad \text{(one-sample t-test)}$$")
    md_lines.append("")
    md_lines.append("Significance ($*$) indicates the model captures individual-specific features beyond group average.")
    md_lines.append("")

    if verbose and "identifiability_violin" in saved_figures:
        md_lines.append("<table><tr>")
        md_lines.append(f'<td align="center"><b>Raw</b><br/><img src="{saved_figures.get("identifiability_violin", "")}" width="400"/></td>')
        md_lines.append(f'<td align="center"><b>Demeaned</b><br/><img src="{saved_figures.get("identifiability_violin_demeaned", "")}" width="400"/></td>')
        md_lines.append("</tr></table>")
    elif "identifiability_violin_demeaned" in saved_figures:
        md_lines.append(f"![]({saved_figures['identifiability_violin_demeaned']})")
    md_lines.append("")

    if verbose and "violin_raw" in all_metrics:
        md_lines.append("| Metric | Raw | Demeaned |")
        md_lines.append("|--------|-----|----------|")
        raw = all_metrics.get("violin_raw", {})
        dem = all_metrics.get("violin_demeaned", {})
        md_lines.append(f"| pFC r_intra | {raw.get('pfc_mean_r_intra', '-'):.3f} | {dem.get('pfc_mean_r_intra', '-'):.3f} |")
        md_lines.append(f"| pFC r_inter | {raw.get('pfc_mean_r_inter', '-'):.3f} | {dem.get('pfc_mean_r_inter', '-'):.3f} |")
        md_lines.append(f"| pFC Cohen's d | {raw.get('pfc_cohen_d', '-'):.2f} | {dem.get('pfc_cohen_d', '-'):.2f} |")
        md_lines.append(f"| pFC p-value (t-test) | {raw.get('pfc_p_value', '-'):.2e} | {dem.get('pfc_p_value', '-'):.2e} |")
    elif "violin_demeaned" in all_metrics:
        dem = all_metrics.get("violin_demeaned", {})
        md_lines.append("| Metric | Value |")
        md_lines.append("|--------|-------|")
        md_lines.append(f"| pFC r_intra | {dem.get('pfc_mean_r_intra', '-'):.3f} |")
        md_lines.append(f"| pFC r_inter | {dem.get('pfc_mean_r_inter', '-'):.3f} |")
        md_lines.append(f"| pFC Cohen's d | {dem.get('pfc_cohen_d', '-'):.2f} |")
        md_lines.append(f"| pFC p-value (t-test) | {dem.get('pfc_p_value', '-'):.2e} |")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    if verbose:
        md_lines.append("## Hungarian Matching")
        md_lines.append("")
        md_lines.append(
            "The Hungarian algorithm derives an optimal **one-to-one** mapping between target (eFC) and predicted (pFC) matrices "
            "that maximizes total similarity. Unlike greedy matching (which permits one-to-many assignments), "
            "Hungarian matching ensures each prediction is assigned to exactly one target."
        )
        md_lines.append("")
        md_lines.append("**Procedure:**")
        md_lines.append("1. Compute similarity matrix $R_{ij} = r(X_i, \\hat{X}_j)$ between all target-prediction pairs")
        md_lines.append("2. Find permutation $\\pi^*$ that maximizes total similarity:")
        md_lines.append("")
        md_lines.append(r"$$\pi^* = \arg\max_{\pi \in S_N} \sum_{i=1}^{N} R_{i,\pi(i)}$$")
        md_lines.append("")
        md_lines.append("3. **Top-1 Acc** = fraction of subjects assigned to themselves: $\\frac{1}{N}\\sum_i \\mathbb{1}[\\pi^*(i) = i]$")
        md_lines.append("")
        md_lines.append("**Null conditions:**")
        md_lines.append("- **Null (noise):** Predictions replaced with mean + Gaussian noise")
        md_lines.append("- **Null (permute):** Columns of similarity matrix randomly permuted (chance baseline)")
        md_lines.append("")

        if "hungarian_heatmaps" in saved_figures:
            md_lines.append("<table><tr>")
            md_lines.append(f'<td align="center"><b>Raw</b><br/><img src="{saved_figures.get("hungarian_heatmaps", "")}" width="500"/></td>')
            md_lines.append(f'<td align="center"><b>Demeaned</b><br/><img src="{saved_figures.get("hungarian_heatmaps_demeaned", "")}" width="500"/></td>')
            md_lines.append("</tr></table>")
        md_lines.append("")

        if "hungarian_raw" in all_metrics:
            md_lines.append("| Condition | Raw Top-1 Acc | Demeaned Top-1 Acc |")
            md_lines.append("|-----------|---------------|---------------------|")
            raw = all_metrics.get("hungarian_raw", {})
            dem = all_metrics.get("hungarian_demeaned", {})
            md_lines.append(f"| pFC | {raw.get('pfc_accuracy', '-'):.3f} | {dem.get('pfc_accuracy', '-'):.3f} |")
            md_lines.append(f"| Null (noise) | {raw.get('null_noise_accuracy', '-'):.3f} | {dem.get('null_noise_accuracy', '-'):.3f} |")
            md_lines.append(f"| Null (permute) | {raw.get('null_permute_accuracy', '-'):.3f} | {dem.get('null_permute_accuracy', '-'):.3f} |")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

        md_lines.append("## Hungarian Sample Size Analysis")
        md_lines.append("")
        md_lines.append(
            "Matching accuracy as a function of subset sample size. "
            "Stars indicate sample sizes where pFC significantly exceeds both null baselines (FDR-corrected, two-sample t-test)."
        )
        md_lines.append("")

        if "hungarian_sample_size" in saved_figures:
            md_lines.append("<table><tr>")
            md_lines.append(f'<td align="center"><b>Raw</b><br/><img src="{saved_figures.get("hungarian_sample_size", "")}" width="450"/></td>')
            md_lines.append(f'<td align="center"><b>Demeaned</b><br/><img src="{saved_figures.get("hungarian_sample_size_demeaned", "")}" width="450"/></td>')
            md_lines.append("</tr></table>")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    md_lines.append("## PCA Structure")
    md_lines.append("")
    md_lines.append(
        "Subject-mode PCA captures the main modes of inter-subject variation in connectivity. "
        "High PC score correlations indicate the model preserves individual differences along each mode."
    )
    md_lines.append("")
    md_lines.append("**Procedure:**")
    md_lines.append("1. Mean-center data: $\\tilde{X} = X - \\bar{X}_{train}$ where $X \\in \\mathbb{R}^{N \\times E}$ (subjects $\\times$ edges)")
    md_lines.append("2. Transpose for subject-mode PCA: $\\tilde{X}^T \\in \\mathbb{R}^{E \\times N}$")
    md_lines.append("3. Compute eigenvectors $B_k$ (loadings) and project: $C_k = \\tilde{X}^T B_k$ (PC scores, length $E$)")
    md_lines.append("4. Project predictions into same basis: $C^{pred}_k = \\tilde{\\hat{X}}^T B_k$")
    md_lines.append("5. Correlate scores: $\\text{PC Corr}_k = r(C^{target}_k, C^{pred}_k)$")
    md_lines.append("")
    md_lines.append(r"$$\text{PC Corr}_k = \text{corr}(C^{target}_k, C^{pred}_k)$$")
    md_lines.append("")

    if "pca_line" in saved_figures:
        md_lines.append(f"![]({saved_figures['pca_line']})")
        md_lines.append("")
    if "pca_spatial" in saved_figures:
        md_lines.append(f"![]({saved_figures['pca_spatial']})")
        md_lines.append("")

    if "pca" in all_metrics:
        pca = all_metrics["pca"]
        md_lines.append("| Metric | Value |")
        md_lines.append("|--------|-------|")
        if "cutoff_idx_95" in pca:
            md_lines.append(f"| PCs for 95% variance | {pca['cutoff_idx_95'] + 1} |")
        if "pc_corrs" in pca and len(pca["pc_corrs"]) > 0:
            md_lines.append(f"| PC1 Corr | {pca['pc_corrs'][0]:.3f} |")
            if len(pca["pc_corrs"]) > 4:
                md_lines.append(f"| PC5 Corr | {pca['pc_corrs'][4]:.3f} |")
        if pca.get("exp_fit_params"):
            exp = pca["exp_fit_params"]
            md_lines.append(f"| Exp. decay rate (b) | {exp.get('b', '-'):.4f} |")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    if "prediction_subset" in saved_figures:
        md_lines.append("## Prediction Subset Viewer")
        md_lines.append("")
        md_lines.append(
            "Compact row-wise viewer of selected subjects: target matrix, prediction matrix, "
            "and edge-wise scatter with per-subject metrics."
        )
        md_lines.append("")
        md_lines.append(f"![]({saved_figures['prediction_subset']})")
        md_lines.append("")
        subset = all_metrics.get("prediction_subset", {})
        md_lines.append("| Metric | Value |")
        md_lines.append("|--------|-------|")
        md_lines.append(f"| Subjects Shown | {subset.get('n_subjects_shown', '-')} |")
        modes = subset.get("display_modes", [])
        if isinstance(modes, list):
            md_lines.append(f"| Display Modes | {', '.join(map(str, modes))} |")
        md_lines.append(f"| Include Best/Worst | {subset.get('include_best_worst', '-')} |")
        selected_ids = subset.get("selected_subject_ids", [])
        if isinstance(selected_ids, list):
            md_lines.append(f"| Selected Subject IDs | {', '.join(map(str, selected_ids))} |")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    md_lines.append("## Summary Metrics")
    md_lines.append("")
    md_lines.append("| Category | Metric | Value |")
    md_lines.append("|----------|--------|-------|")

    base = all_metrics.get("base_metrics", {})
    md_lines.append(f"| Base | MSE | {base.get('mse', '-'):.4f} |")
    md_lines.append(f"| Base | R2 | {base.get('r2', '-'):.4f} |")
    md_lines.append(f"| Base | Pearson Corr | {base.get('pearson', '-'):.4f} |")
    md_lines.append(f"| Base | Demeaned Pearson | {base.get('demeaned_pearson', '-'):.4f} |")
    for geo_prefix, geo_label in (("geodesic_demeaned", "Geodesic Demeaned"), ("geodesic_raw", "Geodesic Raw")):
        if f"{geo_prefix}_top1_acc" in base:
            md_lines.append(
                f"| Base (optional) | {geo_label} Top-1 Acc ({base.get(f'{geo_prefix}_method', '-')}) | "
                f"{base.get(f'{geo_prefix}_top1_acc', np.nan):.4f} |"
            )
        if f"{geo_prefix}_avg_rank" in base:
            md_lines.append(
                f"| Base (optional) | {geo_label} Avg Rank %ile ({base.get(f'{geo_prefix}_method', '-')}) | "
                f"{base.get(f'{geo_prefix}_avg_rank', np.nan):.4f} |"
            )

    raw_hm = all_metrics.get("heatmaps_raw", all_metrics.get("heatmaps_demeaned", {}))
    md_lines.append(f"| Identifiability | Top-1 Acc | {raw_hm.get('top1_acc', '-'):.3f} |")
    md_lines.append(f"| Identifiability | Avg Rank %ile | {raw_hm.get('avg_rank_percentile', '-'):.3f} |")

    raw_vio = all_metrics.get("violin_raw", all_metrics.get("violin_demeaned", {}))
    md_lines.append(f"| Violin | Cohen's d | {raw_vio.get('pfc_cohen_d', '-'):.2f} |")
    md_lines.append(f"| Violin | p-value | {raw_vio.get('pfc_p_value', '-'):.2e} |")

    if verbose and ("hungarian_raw" in all_metrics or "hungarian_demeaned" in all_metrics):
        raw_hung = all_metrics.get("hungarian_raw", all_metrics.get("hungarian_demeaned", {}))
        md_lines.append(f"| Hungarian | pFC Top-1 Acc | {raw_hung.get('pfc_accuracy', '-'):.3f} |")

    if "pca" in all_metrics:
        pca = all_metrics["pca"]
        if "pc_corrs" in pca and len(pca["pc_corrs"]) >= 5:
            pc_str = ", ".join([f"{pca['pc_corrs'][i]:.3f}" for i in range(5)])
            md_lines.append(f"| PCA | PC1-5 Corr | {pc_str} |")
        elif "pc_corrs" in pca and len(pca["pc_corrs"]) > 0:
            pc_str = ", ".join([f"{c:.3f}" for c in pca["pc_corrs"][:5]])
            md_lines.append(f"| PCA | PC Corrs | {pc_str} |")
        if pca.get("exp_fit_params"):
            md_lines.append(f"| PCA | Exp. decay rate (b) | {pca['exp_fit_params'].get('b', '-'):.4f} |")

    md_lines.append("")

    md_path = f"{filepath}.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"Markdown report saved to: {md_path}")
    print(f"Figures saved to: {plots_dir}/")


# ============================================================================
# Global font configuration for better visibility
# ============================================================================
FONT_CONFIG = {
    'title': 14,
    'label': 12,
    'tick': 11,
    'legend': 11,
    'annotation': 11,
}

def set_plot_defaults():
    """Set global matplotlib defaults for better visibility."""
    plt.rcParams.update({
        'font.size': FONT_CONFIG['label'],
        'axes.titlesize': FONT_CONFIG['title'],
        'axes.labelsize': FONT_CONFIG['label'],
        'xtick.labelsize': FONT_CONFIG['tick'],
        'ytick.labelsize': FONT_CONFIG['tick'],
        'legend.fontsize': FONT_CONFIG['legend'],
    })

# Apply defaults on import
set_plot_defaults()

class EvaluatorVizMarkdownMixin:
    """Visualization and report methods mixed into Evaluator."""

    def evaluate_pca_structure(self, num_pcs=5, show_first_pcs=5, diagval=0, show_sse=True, show=True):
        """
        Evaluate and visualize PCA structure comparing predictions to targets.
        
        Plots two visualizations:
        1. Corr(line plot) of PC scores for num_pcs PCs, drawing dashed red line at first num of PCs that reaches >= .95
        2. PC spatial maps (as before), for show_first_pcs PCs (smaller subplots than before).
        
        Args:
            num_pcs: int, number of principal components to plot in correlation line (must be <= n_components)
            show_first_pcs: int, number of PC squares to show in spatial map grid (must be <= num_pcs)
            diagval: float, value to set on diagonal of square matrix (default 0)
            show_sse: bool, whether to display normalized SSE (default True)
            show: bool, whether to display the plots (default True)
        
        Returns:
            tuple: ((fig_line, fig_grid), metrics_dict) with figures and PCA metrics
        """
        # Fit PCA if not already done
        if self._pca_targets is None or self._pca_preds is None:
            self._fit_pca()
        
        pca_preds = self._pca_preds
        pca_targets = self._pca_targets
        explained_var_targets = pca_targets.explained_variance_ratio_

        # Validate PC arguments
        total_pcs = len(explained_var_targets)
        if num_pcs > total_pcs:
            raise ValueError(f"num_pcs ({num_pcs}) cannot be greater than number of available PCs ({total_pcs})")
        if show_first_pcs > num_pcs:
            raise ValueError(f"show_first_pcs ({show_first_pcs}) cannot be greater than num_pcs ({num_pcs})")
        
        # Prepare centered matrices, shape: (features, subjects)
        targets_centered_T = (self.targets - self.train_mean).T
        preds_centered_T   = (self.preds - self.train_mean).T
        
        # Project into targets PCA basis
        C_targets_scores = pca_targets.transform(targets_centered_T)  # (n_features, n_components)
        C_preds_scores = pca_targets.transform(preds_centered_T)      # (n_features, n_components)
        # C_targets_scores[:, i] is PCi for all features (i-th PC, subjects)
        # But in scikit-learn, transform returns shape (samples, components),
        # so here (features, PCs), want to correlate each column

        # Compute PC score correlations for the first num_pcs
        pc_corrs = []
        for i in range(num_pcs):
            t = C_targets_scores[:, i]
            p = C_preds_scores[:, i]
            if np.std(t) > 0 and np.std(p) > 0:
                pc_corrs.append(np.corrcoef(t, p)[0, 1])
            else:
                pc_corrs.append(float('nan'))

        # Correlation line plot for the first num_pcs
        fig_line = plt.figure(figsize=(8, 4))
        plt.plot(np.arange(1, num_pcs+1), pc_corrs, marker='o', color='b', label='PC score corr (targets, preds)')
        
        # Fit exponential decay curve: r(k) = a * exp(-b * k) + c
        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        exp_fit_params = None
        try:
            x_data = np.arange(1, num_pcs+1)
            y_data = np.array(pc_corrs)
            # Filter out NaN values for fitting
            valid_mask = ~np.isnan(y_data)
            if np.sum(valid_mask) > 3:  # Need at least 3 points
                popt, _ = curve_fit(exp_decay, x_data[valid_mask], y_data[valid_mask], 
                                   p0=[0.5, 0.05, 0.0], maxfev=5000,
                                   bounds=([0, 0, -1], [2, 1, 1]))
                exp_fit_params = {'a': popt[0], 'b': popt[1], 'c': popt[2]}
                # Plot smooth exponential curve
                x_smooth = np.linspace(1, num_pcs, 200)
                y_smooth = exp_decay(x_smooth, *popt)
                plt.plot(x_smooth, y_smooth, '-', color='lightblue', alpha=0.8, linewidth=2,
                        label=f'Exp. fit: {popt[0]:.2f}·exp(-{popt[1]:.3f}·k) + {popt[2]:.2f}')
        except Exception:
            pass  # Skip if fitting fails
        
        plt.xlabel("Principal Component", fontsize=FONT_CONFIG['label'])
        plt.ylabel("Correlation between scores", fontsize=FONT_CONFIG['label'])
        plt.title(f"Correlation between Predicted & Target PC Scores (First {num_pcs} PCs)", fontsize=FONT_CONFIG['title'])

        # Find index where cumulative variance explained crosses 0.95
        cum_var = np.cumsum(explained_var_targets)
        cutoff_idx = np.argmax(cum_var >= 0.95)
        if cutoff_idx < num_pcs:
            plt.axvline(cutoff_idx+1, color='r', linestyle='--', linewidth=1, alpha=0.8,
                        label='95% target var explained')
        plt.ylim(-1, 1)
        plt.xlim(0.5, num_pcs + 0.5)
        plt.grid(alpha=0.25)
        plt.legend(fontsize=FONT_CONFIG['legend'])
        plt.tight_layout()
        if show:
            plt.show()

        # Now PC score spatial maps for show_first_pcs
        fig_grid, axs = plt.subplots(show_first_pcs, 3, figsize=(8, show_first_pcs * 2.3), dpi=200)
        if show_first_pcs == 1:
            axs = axs[None, :]  # Always 2D
        
        pca_metrics = {
            'pc_corrs': pc_corrs,
            'explained_variance_ratio': explained_var_targets[:num_pcs].tolist(),
            'cumulative_variance': cum_var[:num_pcs].tolist(),
            'cutoff_idx_95': int(cutoff_idx),
            'exp_fit_params': exp_fit_params,
        }
        
        for i in range(show_first_pcs):
            C_targets_pc = C_targets_scores[:, i]
            C_preds_pc = C_preds_scores[:, i]

            # Project to square (ROI x ROI) format
            C_targets_sq = tri2square(C_targets_pc, numroi=self.numrois, diagval=diagval)
            C_preds_sq   = tri2square(C_preds_pc, numroi=self.numrois, diagval=diagval)
            
            vmax_targets = np.max(np.abs(C_targets_sq))
            vmin_targets = -vmax_targets
            vmax_preds   = np.max(np.abs(C_preds_sq))
            vmin_preds   = -vmax_preds

            # Target PC
            im0 = axs[i, 0].imshow(C_targets_sq, cmap='RdBu_r', vmin=vmin_targets, vmax=vmax_targets, aspect='equal')
            axs[i, 0].set_title(f'Target PC{i+1}\n(Scores)', fontsize=FONT_CONFIG['label'])
            axs[i, 0].set_xticks([])
            axs[i, 0].set_yticks([])
            plt.colorbar(im0, ax=axs[i, 0], fraction=0.045, pad=0.03)

            # Predicted PC
            im1 = axs[i, 1].imshow(C_preds_sq, cmap='RdBu_r', vmin=vmin_preds, vmax=vmax_preds, aspect='equal')
            axs[i, 1].set_title(f'Predicted PC{i+1}\n(Scores)', fontsize=FONT_CONFIG['label'])
            axs[i, 1].set_xticks([])
            axs[i, 1].set_yticks([])
            plt.colorbar(im1, ax=axs[i, 1], fraction=0.045, pad=0.03)

            pc_corr = pc_corrs[i]
            var_exp = explained_var_targets[i] * 100

            label = f"PC{i+1}\nVariance explained: {var_exp:.1f}%\n"
            label += f"Corr(target, pred): r={pc_corr:.3f}"

            if show_sse:
                sse_norm_targets = compute_normalized_sse_pc(self.targets.T, pca_targets, i)
                sse_norm_preds   = compute_normalized_sse_pc(self.preds.T, pca_preds, i)
                label += f"\nNorm. SSE (targ): {sse_norm_targets:.4f}"
                label += f"\nNorm. SSE (pred): {sse_norm_preds:.4f}"
                pca_metrics[f'pc{i+1}_sse_targets'] = sse_norm_targets
                pca_metrics[f'pc{i+1}_sse_preds'] = sse_norm_preds

            axs[i, 2].axis('off')
            axs[i, 2].text(0.05, 0.5, label, va='center', ha='left', fontsize=FONT_CONFIG['annotation'])

        axs[0, 2].figure.text(0.79, 0.97, "Metrics", fontsize=FONT_CONFIG['title'], ha='center', va='center')
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        
        if show:
            plt.show()

        return (fig_line, fig_grid), pca_metrics

    def _get_spd_reconstructions(self, demeaned=False, diag_value=1.0, eps=1e-6):
        """Reconstruct target/prediction FC matrices and project them to SPD, with caching."""
        cache_key = (bool(demeaned), float(diag_value), float(eps))
        if cache_key in self._spd_matrix_cache:
            return self._spd_matrix_cache[cache_key]

        if demeaned:
            preds_data = self.preds - self.train_mean
            targets_data = self.targets - self.train_mean
        else:
            preds_data = self.preds
            targets_data = self.targets

        target_mats, target_meta = prepare_fc_matrices(
            targets_data,
            numrois=self.numrois,
            diag_value=diag_value,
            eps=eps,
        )
        pred_mats, pred_meta = prepare_fc_matrices(
            preds_data,
            numrois=self.numrois,
            diag_value=diag_value,
            eps=eps,
        )
        out = {
            "targets": target_mats,
            "preds": pred_mats,
            "metadata": {
                "demeaned": bool(demeaned),
                "diag_value": float(diag_value),
                "eps": float(eps),
                "targets_n_projected": int(target_meta["n_projected"]),
                "preds_n_projected": int(pred_meta["n_projected"]),
                "targets_fraction_projected": float(target_meta["fraction_projected"]),
                "preds_fraction_projected": float(pred_meta["fraction_projected"]),
                "targets_min_eig_before": float(np.min(target_meta["min_eig_before"])),
                "preds_min_eig_before": float(np.min(pred_meta["min_eig_before"])),
                "n_eigs_per_matrix": int(target_meta["n_eigs_per_matrix"]),
                "targets_mean_clipped_fraction": float(target_meta["mean_clipped_fraction"]),
                "preds_mean_clipped_fraction": float(pred_meta["mean_clipped_fraction"]),
                "targets_mean_negative_fraction": float(target_meta["mean_negative_fraction"]),
                "preds_mean_negative_fraction": float(pred_meta["mean_negative_fraction"]),
                "targets_mean_clip_mass": float(target_meta["mean_clip_mass"]),
                "preds_mean_clip_mass": float(pred_meta["mean_clip_mass"]),
                "targets_median_clip_mass": float(target_meta["median_clip_mass"]),
                "preds_median_clip_mass": float(pred_meta["median_clip_mass"]),
                "targets_mean_negative_mass": float(target_meta["mean_negative_mass"]),
                "preds_mean_negative_mass": float(pred_meta["mean_negative_mass"]),
            },
        }
        self._spd_matrix_cache[cache_key] = out
        return out

    def _get_fc_reconstructions(self, demeaned=False, diag_value=1.0):
        """Reconstruct target/prediction FC matrices without SPD projection, with caching."""
        cache_key = (bool(demeaned), float(diag_value))
        if cache_key in self._fc_matrix_cache:
            return self._fc_matrix_cache[cache_key]

        if demeaned:
            preds_data = self.preds - self.train_mean
            targets_data = self.targets - self.train_mean
        else:
            preds_data = self.preds
            targets_data = self.targets

        out = {
            "targets": reconstruct_fc_matrices(targets_data, numrois=self.numrois, diag_value=diag_value),
            "preds": reconstruct_fc_matrices(preds_data, numrois=self.numrois, diag_value=diag_value),
            "metadata": {
                "demeaned": bool(demeaned),
                "diag_value": float(diag_value),
                "projection_applied": False,
            },
        }
        self._fc_matrix_cache[cache_key] = out
        return out

    def compute_geodesic_metrics(
        self,
        order_by='original',
        demeaned=False,
        method='log_euclidean',
        diag_value=1.0,
        eps=1e-6,
    ):
        """
        Compute FC matrix distance metrics without creating plots.

        This is the metric-only counterpart to `plot_geodesic_heatmaps(...)`,
        intended for optional inclusion in reports or W&B summaries without
        changing the default evaluation surface.
        """
        order = self._compute_subject_order(order_by)
        if method == "frobenius":
            recon_data = self._get_fc_reconstructions(demeaned=demeaned, diag_value=diag_value)
            targets_mats = recon_data["targets"][order]
            preds_mats = recon_data["preds"][order]
            base_metadata = dict(recon_data["metadata"])
            base_metadata.update({
                "eps": float(eps),
                "targets_n_projected": 0,
                "preds_n_projected": 0,
                "targets_fraction_projected": 0.0,
                "preds_fraction_projected": 0.0,
                "targets_min_eig_before": float(np.nanmin(np.linalg.eigvalsh(targets_mats))),
                "preds_min_eig_before": float(np.nanmin(np.linalg.eigvalsh(preds_mats))),
                "targets_mean_clipped_fraction": 0.0,
                "preds_mean_clipped_fraction": 0.0,
                "targets_mean_negative_fraction": float(np.mean(np.linalg.eigvalsh(targets_mats) < 0.0)),
                "preds_mean_negative_fraction": float(np.mean(np.linalg.eigvalsh(preds_mats) < 0.0)),
                "targets_mean_clip_mass": 0.0,
                "preds_mean_clip_mass": 0.0,
                "projection_applied": False,
            })
        else:
            spd_data = self._get_spd_reconstructions(demeaned=demeaned, diag_value=diag_value, eps=eps)
            targets_mats = spd_data["targets"][order]
            preds_mats = spd_data["preds"][order]
            base_metadata = dict(spd_data["metadata"])
            base_metadata["projection_applied"] = True

        tt_dist = pairwise_fc_distance(targets_mats, targets_mats, method=method, eps=eps)
        pp_dist = pairwise_fc_distance(preds_mats, preds_mats, method=method, eps=eps)
        tp_dist = pairwise_fc_distance(targets_mats, preds_mats, method=method, eps=eps)

        mean_self_distance = float(np.mean(np.diag(tp_dist)))
        avg_rank_percentile, ranklist = distance_avg_rank(tp_dist, return_ranklist=True)
        top1_acc = distance_top1_accuracy(tp_dist)

        metadata = {
            **base_metadata,
            "method": method,
            "order_by": order_by,
            "mean_self_distance": mean_self_distance,
            "top1_acc": top1_acc,
            "avg_rank_percentile": avg_rank_percentile,
        }
        return {
            "method": method,
            "mean_self_distance": mean_self_distance,
            "top1_acc": top1_acc,
            "avg_rank_percentile": avg_rank_percentile,
            "demeaned": bool(demeaned),
            "diag_value": float(diag_value),
            "eps": float(eps),
            "ranklist": ranklist,
            "metadata": metadata,
            "distance_matrices": {
                "targets_targets": tt_dist,
                "preds_preds": pp_dist,
                "targets_preds": tp_dist,
            },
        }

    def plot_geodesic_heatmaps(
        self,
        order_by='original',
        demeaned=False,
        method='log_euclidean',
        diag_value=1.0,
        eps=1e-6,
        geodesic_eps=None,
        color_scale_mode='per_plot',
        include_black_circles=True,
        dpi=300,
        figsize=(17, 5),
        show=True,
        print_metadata=True,
    ):
        """
        Plot one-row SPD distance heatmaps using reconstructed FC matrices.

        Uses full square FC reconstructions, projects them to SPD, then computes
        pairwise distances under the selected SPD geometry.
        """
        # Allow analyze_results-style naming when called directly from notebooks.
        if geodesic_eps is not None:
            eps = geodesic_eps
        geodesic_metrics = self.compute_geodesic_metrics(
            order_by=order_by,
            demeaned=demeaned,
            method=method,
            diag_value=diag_value,
            eps=eps,
        )
        tp_dist = geodesic_metrics["distance_matrices"]["targets_preds"]
        tt_dist = geodesic_metrics["distance_matrices"]["targets_targets"]
        pp_dist = geodesic_metrics["distance_matrices"]["preds_preds"]
        mean_self_distance = geodesic_metrics["mean_self_distance"]
        avg_rank_percentile = geodesic_metrics["avg_rank_percentile"]
        top1_acc = geodesic_metrics["top1_acc"]
        ranklist = geodesic_metrics["ranklist"]
        metadata = geodesic_metrics["metadata"]

        global_vmax = float(np.max([np.max(tt_dist), np.max(pp_dist), np.max(tp_dist)]))
        if not np.isfinite(global_vmax) or np.isclose(global_vmax, 0.0):
            global_vmax = 1.0

        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = GridSpec(1, 4, figure=fig, width_ratios=[1.0, 1.0, 1.0, 0.58], wspace=0.34)
        axs = [fig.add_subplot(gs[0, i]) for i in range(4)]
        comparisons = [
            (pp_dist, "Predicted", "Predicted"),
            (tt_dist, "Target", "Target"),
            (tp_dist, "Target", "Predicted"),
        ]
        demean_label = " (demeaned)" if demeaned else ""

        for iax, (dist, label1, label2) in enumerate(comparisons):
            ax = axs[iax]
            if color_scale_mode == 'global':
                vmax = global_vmax
            elif color_scale_mode == 'per_plot':
                vmax = float(np.nanmax(dist))
                if not np.isfinite(vmax) or np.isclose(vmax, 0.0):
                    vmax = global_vmax
            else:
                raise ValueError(f"Unknown color_scale_mode: {color_scale_mode}. Use 'global' or 'per_plot'.")
            im = ax.imshow(
                dist,
                vmin=0.0,
                vmax=vmax,
                cmap='viridis',
                interpolation='none',
                resample=False,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{label1} vs {label2}{demean_label}", fontsize=FONT_CONFIG['title'])
            ax.set_xlabel(label2, fontsize=FONT_CONFIG['label'])
            ax.set_ylabel(label1, fontsize=FONT_CONFIG['label'])

            if label1 == "Target" and label2 == "Predicted" and include_black_circles:
                marker_kwargs = dict(markersize=4.5, markerfacecolor='none', alpha=0.55)
                for i in range(dist.shape[0]):
                    min_idx = int(np.argmin(dist[i]))
                    ax.plot(min_idx, i, 'ko', **marker_kwargs)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=FONT_CONFIG['tick'])
            cbar.set_label('Distance', fontsize=FONT_CONFIG['label'] - 1)

        summary_ax = axs[3]
        summary_ax.axis('off')
        partition_label = str(getattr(self.dataset_partition, "partition", "unknown")).capitalize()
        summary_text = (
            f"Geodesic Summary ({partition_label})\n"
            f"{'─' * 32}\n"
            f"Method: {method}\n"
            f"Mean self distance: {mean_self_distance:.3f}\n"
            f"Top-1 accuracy: {top1_acc:.3f}\n"
            f"Avg rank %ile: {avg_rank_percentile:.3f}\n"
            f"Demeaned: {demeaned}\n"
            f"Order: {order_by}\n"
            f"N subjects: {tp_dist.shape[0]}"
        )
        summary_ax.text(
            0.5,
            0.5,
            summary_text,
            transform=summary_ax.transAxes,
            fontsize=FONT_CONFIG['legend'],
            verticalalignment='center',
            horizontalalignment='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.85),
            family='monospace',
        )
        plt.tight_layout()

        metadata = dict(metadata)
        metadata["color_scale_mode"] = color_scale_mode
        metrics = {
            "method": method,
            "mean_self_distance": mean_self_distance,
            "top1_acc": top1_acc,
            "avg_rank_percentile": avg_rank_percentile,
            "demeaned": bool(demeaned),
            "diag_value": float(diag_value),
            "eps": float(eps),
            "ranklist": ranklist,
            "metadata": metadata,
        }

        if show:
            plt.show()
        if print_metadata:
            print("\nGeodesic FC distance metadata:")
            print("=" * 44)
            print(f"Partition: {partition_label}")
            print(f"Method: {method}")
            print(f"Demeaned: {demeaned}")
            print(f"Order: {order_by}")
            print(f"Color scale mode: {color_scale_mode}")
            print(f"Diagonal value: {diag_value:.3f}")
            if method != "frobenius":
                print(f"SPD eps: {eps:.1e}")
            print(f"Targets projected: {metadata['targets_n_projected']}/{tp_dist.shape[0]} "
                  f"({metadata['targets_fraction_projected']:.2%})")
            print(f"Predictions projected: {metadata['preds_n_projected']}/{tp_dist.shape[0]} "
                  f"({metadata['preds_fraction_projected']:.2%})")
            print(f"Min target eig before projection: {metadata['targets_min_eig_before']:.3e}")
            print(f"Min pred eig before projection: {metadata['preds_min_eig_before']:.3e}")
            print(f"Mean clipped eig fraction (targets): {metadata['targets_mean_clipped_fraction']:.2%}")
            print(f"Mean clipped eig fraction (preds): {metadata['preds_mean_clipped_fraction']:.2%}")
            print(f"Mean negative eig fraction (targets): {metadata['targets_mean_negative_fraction']:.2%}")
            print(f"Mean negative eig fraction (preds): {metadata['preds_mean_negative_fraction']:.2%}")
            print(f"Mean clip mass (targets): {metadata['targets_mean_clip_mass']:.3f}")
            print(f"Mean clip mass (preds): {metadata['preds_mean_clip_mass']:.3f}")
            print(f"Mean self distance: {mean_self_distance:.4f}")
            print(f"Top-1 accuracy: {top1_acc:.4f}")
            print(f"Avg rank %ile: {avg_rank_percentile:.4f}")
            print("=" * 44)

        return fig, metrics

    def _compute_subject_order(self, order_by='original'):
        """
        Return reordered subject indices for visualization.
        
        Args:
            order_by: str, one of:
                - 'original': keep original order
                - 'family': group by Family_ID from dataset.metadata_df
                - 'demographic': group by unique (sex × race_eth) categories
                - 'age': sort by z-scored age (youngest to oldest)
        
        Returns:
            np.ndarray: indices for reordering subjects
        """
        n_subjects = self.preds.shape[0]
        
        if order_by == 'original':
            return np.arange(n_subjects)
        
        elif order_by == 'family':
            # Get Family_ID for subjects in this partition
            metadata_df = self.dataset.metadata_df
            partition_subject_ids = metadata_df.index[self.subject_indices]
            family_ids = metadata_df.loc[partition_subject_ids, 'Family_ID'].values
            
            # Sort by Family_ID to group families together
            sort_order = np.argsort(family_ids)
            return sort_order
        
        elif order_by == 'demographic':
            # Sort subjects by (sex, race_eth) demographic group.
            # Concatenate sex and race_eth one-hot arrays, find unique rows, then sort.
            base = self.dataset
            sex_np      = np.asarray(base.sex_oh)[self.subject_indices]
            race_eth_np = np.asarray(base.race_eth_oh)[self.subject_indices]
            concat_covariates = np.concatenate([sex_np, race_eth_np], axis=1)
            unique_rows, categories = np.unique(concat_covariates, axis=0, return_inverse=True)
            print(f"Number of unique demographic categories: {len(unique_rows)}")
            sort_order = np.argsort(categories)
            return sort_order

        elif order_by == 'age':
            base = self.dataset
            age_vals = np.asarray(base.age_z)[self.subject_indices].ravel()
            sort_order = np.argsort(age_vals)
            return sort_order

        else:
            raise ValueError(f"Unknown order_by: {order_by}. Use 'original', 'family', 'demographic', or 'age'.")

    def plot_identifiability_heatmaps(self, order_by='original', include_black_circles=True,
                                       include_blue_dots=True, demeaned=False,
                                       top_row_scale_mode='fixed',
                                       bottom_row_scale_mode='per_plot',
                                       dpi=400, figsize=(14, 14), show=True):
        """
        Plot identifiability heatmaps for target and predicted connectomes.
        
        Generates a figure with:
        - Row 1: Target vs Target, Predicted vs Predicted, Target vs Predicted
          using the full correlation scale [-1, 1]
        - Row 2: The same three plots using a tighter shared color scale derived
          from the off-diagonal entries of the Target vs Target matrix
        - Row 3: Summary metrics panel spanning the full width
        
        Args:
            order_by: str, subject ordering method:
                - 'original': original order (default)
                - 'family': group by Family_ID
                - 'demographic': group by (sex × race_eth) categories
                - 'age': sort by age (youngest to oldest)
            include_black_circles: bool, if True, plot black circles for max
                similarity prediction per target (default True)
            include_blue_dots: bool, if True, plot blue dots for predictions
                with similarity greater than correct match (default True)
            demeaned: bool, whether to demean using training mean (default False)
            dpi: int, figure resolution (default 200)
            figsize: tuple, figure size (default (14, 10))
            show: bool, whether to display the plot (default True)
        
        Returns:
            tuple: (fig, metrics_dict) where metrics includes mean_corr, top1_acc, avg_rank_percentile
        """
        # Get subject ordering
        order = self._compute_subject_order(order_by)
        
        # Reorder data
        targets_ordered = self.targets[order]
        preds_ordered = self.preds[order]
        
        # Optionally demean using training mean
        if demeaned:
            targets_data = targets_ordered - self.train_mean
            preds_data = preds_ordered - self.train_mean
            demean_label = " (demeaned)"
        else:
            targets_data = targets_ordered
            preds_data = preds_ordered
            demean_label = ""
        
        # Compute correlation matrices
        corr_matrix = compute_corr_matrix(targets_data, preds_data)
        target_vs_target = compute_corr_matrix(targets_data, targets_data)
        pred_vs_pred = compute_corr_matrix(preds_data, preds_data)
        mean_corr = np.mean(np.diag(corr_matrix))

        # Always compute both raw and demeaned target-vs-pred mean correlation for display
        raw_corr_matrix = compute_corr_matrix(targets_ordered, preds_ordered)
        demeaned_corr_matrix = compute_corr_matrix(
            targets_ordered - self.train_mean,
            preds_ordered - self.train_mean,
        )
        raw_mean_corr = np.mean(np.diag(raw_corr_matrix))
        demeaned_mean_corr = np.mean(np.diag(demeaned_corr_matrix))

        # Shared tighter scale can come from off-diagonal target-target correlations.
        off_diag_mask = ~np.eye(target_vs_target.shape[0], dtype=bool)
        off_diag_values = target_vs_target[off_diag_mask]
        tight_abs_max = float(np.nanmax(np.abs(off_diag_values)))
        if not np.isfinite(tight_abs_max) or np.isclose(tight_abs_max, 0.0):
            tight_vmin, tight_vmax = -1.0, 1.0
        else:
            tight_vmin, tight_vmax = -tight_abs_max, tight_abs_max

        # Create figure layout: 2 heatmap rows + summary row
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1.0, 1.0, 0.4], hspace=0.28, wspace=0.34)
        heatmap_axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[1, 2]),
        ]
        summary_ax = fig.add_subplot(gs[2, :])

        observed_vs_pred_ranklist = None

        comparisons = [
            (pred_vs_pred, "Predicted", "Predicted"),
            (target_vs_target, "Target", "Target"),
            (corr_matrix, "Target", "Predicted"),
        ]

        def _overlay_target_pred_markers(ax, sim, draw_black_circles=True):
            ranklist = []
            marker_kwargs = dict(markersize=5, markerfacecolor='none', alpha=0.5)
            tiny_blue_kwargs = dict(marker='o', color='blue', markersize=1,
                                   alpha=0.7, linestyle='None')

            for i in range(sim.shape[0]):
                cci = sim[i, :]
                cci_min = np.nanmin(cci)
                cci_max = np.nanmax(cci)
                if np.isclose(cci_max, cci_min):
                    cci_scaled = np.zeros_like(cci)
                else:
                    cci_scaled = (cci - cci_min) / (cci_max - cci_min)
                    cci_scaled = (cci_scaled - 0.5) * 0.8

                cci_maxidx = np.argmax(cci)
                cci_sortidx = np.argsort(np.argsort(cci)[::-1])

                if include_black_circles and draw_black_circles:
                    ax.plot(cci_maxidx, i - cci_scaled[cci_maxidx], 'ko', **marker_kwargs)

                if include_blue_dots:
                    closer_pred_idxs = np.where(cci_scaled > cci_scaled[i])[0]
                    if closer_pred_idxs.size > 0:
                        ax.plot(
                            closer_pred_idxs,
                            np.repeat(i, len(closer_pred_idxs)) - cci_scaled[closer_pred_idxs],
                            **tiny_blue_kwargs
                        )

                ranklist.append(cci_sortidx[i] + 1)

            return np.array(ranklist)

        def _centered_limits(sim, fallback=1.0, ignore_diagonal=False):
            arr = np.asarray(sim)
            if ignore_diagonal and arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] > 1:
                mask = ~np.eye(arr.shape[0], dtype=bool)
                vals = arr[mask]
            else:
                vals = arr
            absmax = float(np.nanmax(np.abs(vals)))
            if not np.isfinite(absmax) or np.isclose(absmax, 0.0):
                absmax = fallback
            return -absmax, absmax

        for row_idx in range(2):
            for col_idx, (sim, label1, label2) in enumerate(comparisons):
                ax = heatmap_axes[row_idx * 3 + col_idx]
                ignore_diag_for_scaling = (label1 == label2)
                if row_idx == 0:
                    if top_row_scale_mode == 'fixed':
                        vmin, vmax = -1.0, 1.0
                    elif top_row_scale_mode == 'per_plot':
                        vmin, vmax = _centered_limits(
                            sim,
                            fallback=1.0,
                            ignore_diagonal=ignore_diag_for_scaling,
                        )
                    else:
                        raise ValueError(
                            f"Unknown top_row_scale_mode: {top_row_scale_mode}. Use 'fixed' or 'per_plot'."
                        )
                else:
                    if bottom_row_scale_mode == 'per_plot':
                        vmin, vmax = _centered_limits(
                            sim,
                            fallback=1.0,
                            ignore_diagonal=ignore_diag_for_scaling,
                        )
                    elif bottom_row_scale_mode == 'shared':
                        vmin, vmax = tight_vmin, tight_vmax
                    else:
                        raise ValueError(
                            f"Unknown bottom_row_scale_mode: {bottom_row_scale_mode}. Use 'per_plot' or 'shared'."
                        )
                im = ax.imshow(
                    sim,
                    vmin=vmin,
                    vmax=vmax,
                    cmap='RdBu_r',
                    interpolation='none',
                    resample=False,
                )

                ax.set_xticks([])
                ax.set_yticks([])

                titlestr = ""
                if row_idx == 0:
                    titlestr = f'{label1} vs {label2}{demean_label}'

                if label1 == "Target" and label2 == "Predicted":
                    ranklist = _overlay_target_pred_markers(
                        ax,
                        sim,
                        draw_black_circles=(row_idx == 0),
                    )
                    observed_vs_pred_ranklist = ranklist
                    if row_idx == 0:
                        avgrank = corr_avg_rank(cc=sim)
                        avgrank_index = np.mean(ranklist)
                        titlestr += f'\nAvg Rank {avgrank_index:.1f} out of {sim.shape[0]}, %ile: {avgrank:.3f}'

                if titlestr:
                    ax.set_title(titlestr, fontsize=FONT_CONFIG['title'])
                ax.set_xlabel(label2, fontsize=FONT_CONFIG['label'])
                ax.set_ylabel(label1, fontsize=FONT_CONFIG['label'])

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cbar = fig.colorbar(im, cax=cax)
                cbar.ax.tick_params(labelsize=FONT_CONFIG['tick'])
        
        # Compute summary metrics
        n_subjects = len(observed_vs_pred_ranklist)
        n_top1 = np.sum(observed_vs_pred_ranklist == 1)
        top1_acc = n_top1 / n_subjects
        avgrank_percentile = 1 - (np.mean(observed_vs_pred_ranklist) / n_subjects)
        
        summary_ax.axis('off')
        
        # Compute chance levels
        chance_top1 = 1 / n_subjects
        chance_avgrank = 0.5
        
        # Add summary text in fourth panel
        partition_label = str(getattr(self.dataset_partition, "partition", "unknown")).capitalize()
        summary_text = (
            f"Summary Metrics ({partition_label})\n"
            f"{'─' * 35}\n"
            f"Mean corr: {raw_mean_corr:.3f}\n"
            f"Demeaned Mean corr: {demeaned_mean_corr:.3f}\n"
            f"Demeaned: {demeaned}\n"
            f"Top-1 accuracy: {top1_acc:.3f} (chance={chance_top1:.3f})\n"
            f"  ({n_top1} of {n_subjects} subjects had rank 1)\n"
            f"Avg rank %ile: {avgrank_percentile:.3f} (chance={chance_avgrank})\n"
            f"{'─' * 35}\n"
            f"Order: {order_by}\n"
            f"N subjects: {n_subjects}"
        )
        summary_ax.text(0.5, 0.3, summary_text, transform=summary_ax.transAxes,
                        fontsize=FONT_CONFIG['legend'], verticalalignment='center',
                        horizontalalignment='center',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                                  edgecolor='gray', alpha=0.8),
                        family='monospace')

        plt.tight_layout()
        
        metrics = {
            'mean_corr': mean_corr,
            'raw_mean_corr': raw_mean_corr,
            'demeaned_mean_corr': demeaned_mean_corr,
            'demeaned': demeaned,
            'top1_acc': top1_acc,
            'avg_rank_percentile': avgrank_percentile,
            'ranklist': observed_vs_pred_ranklist,
            'chance_top1': chance_top1,
            'chance_avgrank': chance_avgrank,
            'tight_scale_vmin': tight_vmin,
            'tight_scale_vmax': tight_vmax,
        }
        
        if show:
            print(f"Top-1 accuracy: {top1_acc:.3f}  ({n_top1} of {n_subjects} subjects had rank 1) (chance={chance_top1:.3f})")
            print(f"Average rank percentile: {avgrank_percentile:.3f} (chance={chance_avgrank})")
            plt.show()
        
        return fig, metrics

    def plot_single_corr_heatmap(self, comparison='targets_vs_preds', order_by='original',
                                  include_black_circles=True, include_blue_dots=True,
                                  demeaned=False, dpi=200, figsize=(8, 8), show=True):
        """
        Plot a single correlation matrix heatmap.
        
        Args:
            comparison: str, which comparison to plot:
                - 'targets_vs_targets': Target correlations
                - 'preds_vs_preds': Prediction correlations  
                - 'targets_vs_preds': Target vs Prediction correlations (default)
            order_by: str, subject ordering ('original', 'family', 'demographic')
            include_black_circles: bool, show black circles for max similarity (only for targets_vs_preds)
            include_blue_dots: bool, show blue dots for closer predictions
            demeaned: bool, whether to demean using training mean (default False)
            dpi: int, figure resolution (default 200)
            figsize: tuple, figure size (default (8, 8))
            show: bool, whether to display the plot (default True)
        
        Returns:
            tuple: (fig, metrics_dict) for this comparison
        """
        # Get subject ordering
        order = self._compute_subject_order(order_by)
        
        # Reorder and optionally demean
        targets_ordered = self.targets[order]
        preds_ordered = self.preds[order]
        
        if demeaned:
            targets_data = targets_ordered - self.train_mean
            preds_data = preds_ordered - self.train_mean
            demean_label = " (demeaned)"
        else:
            targets_data = targets_ordered
            preds_data = preds_ordered
            demean_label = ""
        
        # Select data based on comparison type
        if comparison == 'targets_vs_targets':
            data1, data2 = targets_data, targets_data
            label1, label2 = "Target", "Target"
        elif comparison == 'preds_vs_preds':
            data1, data2 = preds_data, preds_data
            label1, label2 = "Predicted", "Predicted"
        elif comparison == 'targets_vs_preds':
            data1, data2 = targets_data, preds_data
            label1, label2 = "Target", "Predicted"
        else:
            raise ValueError(f"Unknown comparison: {comparison}")
        
        # Compute correlation matrix
        sim = compute_corr_matrix(data1, data2)
        mean_corr = np.mean(np.diag(sim))
        avgrank = corr_avg_rank(cc=sim)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        im = ax.imshow(sim, vmin=-1, vmax=1, cmap='RdBu_r')
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        titlestr = f'{label1} vs {label2}{demean_label}'
        
        ranklist = None
        # Add markers for targets_vs_preds
        if comparison == 'targets_vs_preds' and (include_black_circles or include_blue_dots):
            ranklist = []
            marker_kwargs = dict(markersize=5, markerfacecolor='none')
            tiny_blue_kwargs = dict(marker='o', color='blue', markersize=1, 
                                   alpha=0.7, linestyle='None')
            
            for i in range(sim.shape[0]):
                cci = sim[i, :]
                cci_scaled = (cci - np.nanmin(cci)) / (np.nanmax(cci) - np.nanmin(cci))
                cci_scaled = (cci_scaled - 0.5) * 0.8
                
                cci_maxidx = np.argmax(cci)
                cci_sortidx = np.argsort(np.argsort(cci)[::-1])
                
                # Black circle for most similar prediction
                if include_black_circles:
                    ax.plot(cci_maxidx, i - cci_scaled[cci_maxidx], 'ko', **marker_kwargs)
                
                if include_blue_dots:
                    closer_pred_idxs = np.where(cci_scaled > cci_scaled[i])[0]
                    if closer_pred_idxs.size > 0:
                        ax.plot(
                            closer_pred_idxs,
                            np.repeat(i, len(closer_pred_idxs)) - cci_scaled[closer_pred_idxs],
                            **tiny_blue_kwargs
                        )
                
                ranklist.append(cci_sortidx[i] + 1)
            
            ranklist = np.array(ranklist)
            avgrank_index = np.mean(ranklist)
            titlestr += f'\nAvg Rank {avgrank_index:.1f}, %ile: {avgrank:.3f}'
        
        ax.set_title(titlestr, fontsize=FONT_CONFIG['title'] + 2)
        ax.set_xlabel(label2, fontsize=FONT_CONFIG['label'] + 2)
        ax.set_ylabel(label1, fontsize=FONT_CONFIG['label'] + 2)
        
        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=FONT_CONFIG['tick'])
        
        plt.tight_layout()
        
        metrics = {
            'mean_corr': mean_corr,
            'avg_rank': avgrank,
        }
        if ranklist is not None:
            n_subjects = len(ranklist)
            metrics['top1_acc'] = np.sum(ranklist == 1) / n_subjects
            metrics['avg_rank_percentile'] = 1 - (np.mean(ranklist) / n_subjects)
            metrics['ranklist'] = ranklist
        
        if show:
            plt.show()
        
        return fig, metrics

    def compute_identifiability_stats(self, demeaned=False):
        """
        Compute identifiability statistics for predictions vs targets.
        
        Args:
            demeaned: bool, whether to demean data before computing correlations
        
        Returns:
            dict with identifiability statistics (see compute_identifiability)
        """
        if demeaned:
            preds_data = self.preds - self.train_mean
            targets_data = self.targets - self.train_mean
        else:
            preds_data = self.preds
            targets_data = self.targets
        
        return compute_identifiability(preds_data, targets_data)

    def plot_identifiability_violin(self, include_mean_baseline=True, include_noise_baseline=True,
                                     demeaned=False, seed=42,
                                     p_threshold=0.001, dpi=150, figsize=(8, 6), show=True):
        """
        Plot identifiability violin plot comparing intraindividual vs interindividual correlations.
        
        Shows distributions for:
        - pFC (model predictions)
        - Mean eFC baseline (optional, not shown if demeaned=True) - # demeaned = False: true and null identical (consider what target vs predicted matrix looks like; constant horizontally)
        - Mean eFC + noise baseline (optional) - # demeaned = False: true and null near identical here (consider that inter is correlation to noise minus self noise estimate); demeaned = True: true has more variance than null (consider precision in estimate when correlating with noise over many samples)
        
        Args:
            include_mean_baseline: bool, include mean eFC baseline condition
            include_noise_baseline: bool, include mean eFC + noise baseline
            demeaned: bool, whether to demean data (if True, mean baseline is excluded)
            seed: int, random seed for noise baseline
            p_threshold: float, p-value threshold for significance annotation (default 0.001)
            dpi: int, figure resolution
            figsize: tuple, figure size
            show: bool, whether to display the plot (default True)
        
        Returns:
            tuple: (fig, results_dict) with identifiability results for each condition
        """
        results = {}
        conditions = []
        r_intra_all = []
        r_inter_all = []
        stats_all = []
        
        # Prepare data
        if demeaned:
            preds_data = self.preds - self.train_mean
            targets_data = self.targets - self.train_mean
            include_mean_baseline = False  # No mean baseline when demeaned
        else:
            preds_data = self.preds
            targets_data = self.targets
        
        n_subjects = preds_data.shape[0]
        
        # 1. Model predictions (pFC)
        pfc_stats = compute_identifiability(preds_data, targets_data)
        results['pFC'] = pfc_stats
        conditions.append('pFC')
        r_intra_all.append(pfc_stats['r_intra'])
        r_inter_all.append(pfc_stats['r_inter'])
        stats_all.append(pfc_stats)
        
        # 2. Mean eFC baseline (only if not demeaned)
        if include_mean_baseline:
            mean_preds = generate_mean_baseline(self.train_mean, n_subjects)
            mean_stats = compute_identifiability(mean_preds, targets_data)
            results['Mean eFC'] = mean_stats
            conditions.append('Mean eFC')
            r_intra_all.append(mean_stats['r_intra'])
             # technically these are the exact same
             # if the null test is that every prediction is truly the population mean
            r_inter_all.append(mean_stats['r_inter'])
            stats_all.append(mean_stats)
        
        # 3. Noise baseline
        if include_noise_baseline:
            noise_preds = generate_noise_baseline(self.train_mean, self.train_std, n_subjects, 
                                                   seed=seed)
            
            if demeaned:
                noise_preds = noise_preds - self.train_mean
            noise_stats = compute_identifiability(noise_preds, targets_data)
            results['Null'] = noise_stats
            conditions.append('Null')
            r_intra_all.append(noise_stats['r_intra'])
            r_inter_all.append(noise_stats['r_inter'])
            stats_all.append(noise_stats)
        
        # Create violin plot
        n_conditions = len(conditions)
        positions = np.arange(n_conditions)
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Colors
        color_intra = '#6699CC'  # blue
        color_inter = '#CC6666'  # red
        
        width = 0.35
        
        for i, (cond, r_intra, r_inter, stats) in enumerate(zip(conditions, r_intra_all, r_inter_all, stats_all)):
            pos = positions[i]
            
            # Create split violin (left=intra, right=inter)
            # Intraindividual (blue) - left side
            parts_intra = ax.violinplot([r_intra], positions=[pos - width/2], 
                                         widths=width, showmeans=False, showextrema=False)
            for pc in parts_intra['bodies']:
                pc.set_facecolor(color_intra)
                pc.set_edgecolor('black')
                pc.set_alpha(0.7)
            
            # Interindividual (red) - right side  
            parts_inter = ax.violinplot([r_inter], positions=[pos + width/2], 
                                         widths=width, showmeans=False, showextrema=False)
            for pc in parts_inter['bodies']:
                pc.set_facecolor(color_inter)
                pc.set_edgecolor('black')
                pc.set_alpha(0.7)
            
            # Add mean markers
            ax.scatter([pos - width/2], [np.mean(r_intra)], color='white', 
                      s=40, zorder=3, edgecolor='black', linewidth=1.5)
            ax.scatter([pos + width/2], [np.mean(r_inter)], color='white', 
                      s=40, zorder=3, edgecolor='black', linewidth=1.5)
            
            # Add quartile boxes
            q1_intra, q3_intra = np.percentile(r_intra, [25, 75])
            q1_inter, q3_inter = np.percentile(r_inter, [25, 75])
            
            ax.vlines(pos - width/2, q1_intra, q3_intra, color=color_intra, linewidth=5, zorder=2)
            ax.vlines(pos + width/2, q1_inter, q3_inter, color=color_inter, linewidth=5, zorder=2)
            
            # Add significance annotation
            p_val = stats['p_value']
            if p_val < p_threshold:
                sig_text = '*'
            else:
                sig_text = 'n.s.'
            
            y_max = max(np.max(r_intra), np.max(r_inter))
            ax.text(pos, y_max + 0.01, sig_text, ha='center', va='bottom', fontsize=FONT_CONFIG['title'], fontweight='bold')
        
        # Formatting
        ax.set_xticks(positions)
        ax.set_xticklabels(conditions, fontsize=FONT_CONFIG['label'] + 2)
        ax.set_ylabel('Correlation', fontsize=FONT_CONFIG['label'] + 2)
        ax.set_xlabel('')
        
        # Legend
        legend_elements = [
            Patch(facecolor=color_intra, edgecolor='black', alpha=0.7, label='Intraindividual'),
            Patch(facecolor=color_inter, edgecolor='black', alpha=0.7, label='Interindividual'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='white', 
                   markeredgecolor='black', markersize=8, label='Mean', linestyle='None'),
            Line2D([0], [0], color='none', label=''),  # spacer
            Line2D([0], [0], color='none', 
                   label=r'$H_0: \frac{1}{N}\sum_i (r_{intra}(i) - r_{inter}(i)) = 0$')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=FONT_CONFIG['legend'])
        
        # Title
        demean_str = " (demeaned)" if demeaned else ""
        ax.set_title(f'FC Prediction Identifiability{demean_str}', fontsize=FONT_CONFIG['title'] + 2)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set y-axis limit to extend to 1
        ax.set_ylim(top=1.0)
        
        plt.tight_layout()
        
        if show:
            plt.show()
            # Print summary statistics
            print("\nIdentifiability Summary:")
            print("=" * 70)
            print(f"{'Condition':<12} {'r_intra':<10} {'r_inter':<10} {'d':<10} {'t':<10} {'p':<12} {'Cohen d':<10}")
            print("-" * 70)
            for cond, stats in results.items():
                print(f"{cond:<12} {stats['mean_r_intra']:<10.4f} {stats['mean_r_inter']:<10.4f} "
                      f"{stats['mean_d']:<10.4f} {stats['t_stat']:<10.2f} {stats['p_value']:<12.2e} "
                      f"{stats['cohen_d']:<10.2f}")
            print("=" * 70)
        
        return fig, results

    def plot_hungarian_heatmaps(self, include_noise_baseline=True, include_permute_baseline=True,
                                 demeaned=False, seed=42,
                                 dpi=150, figsize=(15, 5), show=True):
        """
        Plot Hungarian matching heatmaps comparing pFC, noised null, and permuted null.
        
        Shows side-by-side correlation heatmaps with black dots indicating 
        Hungarian optimal assignments and Top-1 accuracy for each condition.
        
        Args:
            include_noise_baseline: bool, include noised baseline
            include_permute_baseline: bool, include permuted baseline
            demeaned: bool, whether to demean data
            seed: int, random seed
            dpi: int, figure resolution
            figsize: tuple, figure size
            show: bool, whether to display the plot (default True)
        
        Returns:
            tuple: (fig, results_dict) with Hungarian matching results for each condition
        """
        results = {}
        
        # Prepare data
        if demeaned:
            preds_data = self.preds - self.train_mean
            targets_data = self.targets - self.train_mean
        else:
            preds_data = self.preds
            targets_data = self.targets
        
        n_subjects = preds_data.shape[0]
        
        # Compute conditions
        conditions = []
        sim_matrices = []
        hungarian_results = []
        
        # 1. pFC (model predictions) - rows=targets, cols=preds
        sim_pfc = compute_corr_matrix(targets_data, preds_data)
        hung_pfc = hungarian_matching(sim_pfc)
        results['pFC'] = hung_pfc
        conditions.append('pFC')
        sim_matrices.append(sim_pfc)
        hungarian_results.append(hung_pfc)
        
        # 2. Noised baseline
        if include_noise_baseline:
            noise_preds = generate_noise_baseline(self.train_mean, self.train_std, n_subjects,
                                                   seed=seed)
            if demeaned:
                noise_preds = noise_preds - self.train_mean
            sim_noise = compute_corr_matrix(targets_data, noise_preds)
            hung_noise = hungarian_matching(sim_noise)
            results['Null (noise)'] = hung_noise
            conditions.append('Null (noise)')
            sim_matrices.append(sim_noise)
            hungarian_results.append(hung_noise)
        
        # 3. Permuted baseline (chance)
        if include_permute_baseline:
            # Use pFC similarity but permute for chance
            rng = np.random.default_rng(seed + 1)
            perm = rng.permutation(n_subjects)
            sim_permuted = sim_pfc[:, perm]
            hung_permuted = hungarian_matching(sim_permuted)
            results['Null (permute)'] = hung_permuted
            conditions.append('Null (permute)')
            sim_matrices.append(sim_permuted)
            hungarian_results.append(hung_permuted)
        
        # Create figure
        n_plots = len(conditions)
        fig, axs = plt.subplots(1, n_plots, figsize=figsize, dpi=dpi)
        if n_plots == 1:
            axs = [axs]
        
        for ax, cond, sim, hung in zip(axs, conditions, sim_matrices, hungarian_results):
            # Plot heatmap
            im = ax.imshow(sim, vmin=-1, vmax=1, cmap='RdBu_r')
            
            # Plot Hungarian assignments as black dots
            row_ind, col_ind = hung['row_ind'], hung['col_ind']
            ax.scatter(col_ind, row_ind, c='black', s=15, marker='o', zorder=3)
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            acc = hung['accuracy'] * 100
            n_correct = hung['n_correct']
            n_total = hung['n']
            demean_str = " (demeaned)" if demeaned else ""
            ax.set_title(f'{cond}{demean_str}\nTop-1 Acc: {acc:.1f}% ({n_correct}/{n_total})', fontsize=FONT_CONFIG['title'])
            ax.set_xlabel('Predicted', fontsize=FONT_CONFIG['label'])
            ax.set_ylabel('Target', fontsize=FONT_CONFIG['label'])
            
            # Colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=FONT_CONFIG['tick'] - 2)
            cbar.set_label('Corr', fontsize=FONT_CONFIG['label'] - 2)
        
        plt.tight_layout()
        
        if show:
            plt.show()
        
        return fig, results

    def plot_hungarian_sample_size_analysis(self, n_min=2, n_max=20, step=1, n_iterations=2500,
                                             demeaned=False, 
                                             fdr_alpha=0.05, seed=42,
                                             dpi=150, figsize=(10, 6), show=True):
        """
        Plot Hungarian matching accuracy across different sample sizes.
        
        Repeatedly samples subsets of size n (for n=n_min, n_min+step, ..., n_max) and computes
        average matching accuracy. Compares pFC vs null conditions with FDR-corrected
        significance testing.
        
        Args:
            n_min: int, minimum sample size (default 2)
            n_max: int, maximum sample size (default 20)
            step: int, step size between sample sizes (default 1)
            n_iterations: int, number of iterations per sample size (M, default 2500)
            demeaned: bool, whether to demean data
            fdr_alpha: float, FDR threshold (default 0.05)
            seed: int, random seed
            dpi: int, figure resolution
            figsize: tuple, figure size
            show: bool, whether to display the plot (default True)
        
        Returns:
            tuple: (fig, results_dict) with results for each condition and sample size
        """
        # Prepare data
        if demeaned:
            preds_data = self.preds - self.train_mean
            targets_data = self.targets - self.train_mean
        else:
            preds_data = self.preds
            targets_data = self.targets

        n_subjects = preds_data.shape[0]
        sample_sizes = np.arange(n_min, n_max + 1, step)
        n_sizes = len(sample_sizes)

        # Compute similarity matrices (targets, preds: rows=targets, cols=preds)
        sim_pfc = compute_corr_matrix(targets_data, preds_data)

        # Noised baseline
        noise_preds = generate_noise_baseline(self.train_mean, self.train_std, n_subjects,
                                               seed=seed)
        if demeaned:
            noise_preds = noise_preds - self.train_mean
        sim_noise = compute_corr_matrix(targets_data, noise_preds)

        # Storage for results
        results = {
            'sample_sizes': sample_sizes,
            'pFC': {'mean': np.zeros(n_sizes), 'std': np.zeros(n_sizes), 'all': []},
            'Null (noise)': {'mean': np.zeros(n_sizes), 'std': np.zeros(n_sizes), 'all': []},
            'Null (permute)': {'mean': np.zeros(n_sizes), 'std': np.zeros(n_sizes), 'all': []},
            'p_values_noise': np.zeros(n_sizes),
            'p_values_permute': np.zeros(n_sizes),
            'significant_noise': np.zeros(n_sizes, dtype=bool),
            'significant_permute': np.zeros(n_sizes, dtype=bool),
        }

        rng = np.random.default_rng(seed)

        print(f"Running Hungarian matching analysis for n={n_min} to {n_max} (step={step}) with M={n_iterations} iterations...")

        for i, n in enumerate(sample_sizes):
            if n > n_subjects:
                print(f"Warning: n={n} exceeds n_subjects={n_subjects}, skipping")
                continue

            # pFC
            acc_pfc = hungarian_matching_subsample(sim_pfc, n, n_iterations, seed=seed + i)
            results['pFC']['mean'][i] = np.mean(acc_pfc)
            results['pFC']['std'][i] = np.std(acc_pfc)
            results['pFC']['all'].append(acc_pfc)

            # Noised null
            acc_noise = hungarian_matching_subsample(sim_noise, n, n_iterations, seed=seed + i + 1000)
            results['Null (noise)']['mean'][i] = np.mean(acc_noise)
            results['Null (noise)']['std'][i] = np.std(acc_noise)
            results['Null (noise)']['all'].append(acc_noise)

            # Permuted null - for each iteration, permute the subsampled matrix
            acc_permute = np.zeros(n_iterations)
            for m in range(n_iterations):
                indices = rng.choice(n_subjects, size=n, replace=False)
                sub_sim = sim_pfc[np.ix_(indices, indices)]
                # Permute columns
                perm = rng.permutation(n)
                sub_sim_permuted = sub_sim[:, perm]
                hung_result = hungarian_matching(sub_sim_permuted)
                acc_permute[m] = hung_result['accuracy']

            results['Null (permute)']['mean'][i] = np.mean(acc_permute)
            results['Null (permute)']['std'][i] = np.std(acc_permute)
            results['Null (permute)']['all'].append(acc_permute)

            # Two-sample t-test: pFC vs noise
            t_noise, p_noise = ttest_ind(acc_pfc, acc_noise)
            results['p_values_noise'][i] = p_noise

            # Two-sample t-test: pFC vs permute
            t_permute, p_permute = ttest_ind(acc_pfc, acc_permute)
            results['p_values_permute'][i] = p_permute

            print(f"  n={n}: pFC={results['pFC']['mean'][i]*100:.1f}%, "
                  f"Noise={results['Null (noise)']['mean'][i]*100:.1f}%, "
                  f"Permute={results['Null (permute)']['mean'][i]*100:.1f}%")

        # FDR correction using Benjamini-Hochberg
        adjusted_p_noise = false_discovery_control(results['p_values_noise'], method='bh')
        adjusted_p_permute = false_discovery_control(results['p_values_permute'], method='bh')
        results['significant_noise'] = adjusted_p_noise < fdr_alpha
        results['significant_permute'] = adjusted_p_permute < fdr_alpha

        # Plot
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Colors
        color_pfc = '#CC6666'  # red
        color_noise = '#6699CC'  # blue  
        color_permute = '#333333'  # black

        # Plot only the 3 main lines (no std bands)
        ax.plot(sample_sizes, results['pFC']['mean'] * 100, '-', color=color_pfc, 
                linewidth=2, label='pFC')
        ax.plot(sample_sizes, results['Null (noise)']['mean'] * 100, '-', color=color_noise,
                linewidth=2, label='Null (noise)')
        ax.plot(sample_sizes, results['Null (permute)']['mean'] * 100, '-', color=color_permute,
                linewidth=2, label='Null (permute)')

        ax.set_xlabel('Number of Individuals', fontsize=FONT_CONFIG['label'] + 2)
        ax.set_ylabel('% Individuals Correctly Matched', fontsize=FONT_CONFIG['label'] + 2)
        ax.set_xlim(n_min - 0.5, n_max + 0.5)
        ax.set_ylim(0, None)

        # Set x-axis ticks at the specified interval
        ax.set_xticks(sample_sizes)
        ax.set_xticklabels(sample_sizes)

        # Add significance markers above x-axis tick labels
        y_min, y_max = ax.get_ylim()
        for i, n in enumerate(sample_sizes):
            # Plot asterisk only if significant for BOTH nulls
            if results['significant_permute'][i] and results['significant_noise'][i]:
                ax.text(n, y_min+0.05, '*', ha='center', va='top', fontsize=FONT_CONFIG['tick'], 
                        fontweight='bold', transform=ax.get_xaxis_transform())

        # Build legend, clarifying the meaning of *
        handles, labels = ax.get_legend_handles_labels()
        asterisk_patch = Line2D([0], [0], color='none', marker='*', markersize=12, 
                                markerfacecolor='k', label="Significant (p < {:.3g}) vs both nulls".format(fdr_alpha))
        handles.append(asterisk_patch)
        ax.legend(handles=handles, fontsize=FONT_CONFIG['legend'], loc='upper right')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        demean_str = " (demeaned)" if demeaned else ""
        ax.set_title(f'Hungarian Matching Accuracy vs Subsetted Sample Size{demean_str}', fontsize=FONT_CONFIG['title'] + 2)

        plt.tight_layout()
        
        if show:
            plt.show()
            # Print summary
            print(f"\nFDR-corrected significance (alpha={fdr_alpha}):")
            print(f"  pFC vs Null (permute): {np.sum(results['significant_permute'])}/{n_sizes} sample sizes significant")
            print(f"  pFC vs Null (noise): {np.sum(results['significant_noise'])}/{n_sizes} sample sizes significant")
        
        return fig, results

    def get_pca_objects(self):
        """
        Get the fitted PCA objects for predictions and targets.
        
        Returns:
            tuple: (pca_preds, pca_targets) or (None, None) if not yet fitted
        """
        if self._pca_targets is None or self._pca_preds is None:
            self._fit_pca()
        return self._pca_preds, self._pca_targets

    
    def _output_metrics(self):
        """
        Print all evaluation metrics in a neat and human-readable format.
        Includes the data partition (train/val/test) from dataset_partition.partition.
        """
        print("=" * 50)
        print(f" Evaluation Metrics for Partition: '{self.dataset_partition.partition}'")
        print("-" * 50)
        pretty_names = {
            "mse": "Mean Squared Error",
            "r2": "R2 Score",
            "pearson": "Pearson Corr.",
            "demeaned_pearson": "Demeaned Pearson Corr.",
            "avg_rank": "Average Rank",
            "top1_acc": "Top-1 Accuracy"
        }
        for key in ["mse", "r2", "pearson", "demeaned_pearson", "avg_rank", "top1_acc"]:
            val = self._metrics.get(key, None)
            if val is not None:
                # Format value with 4 decimals, handle numpy/scalar cases
                if isinstance(val, (float, np.floating)):
                    print(f"{pretty_names.get(key, key):25s}: {float(val):.4f}")
                elif isinstance(val, (list, np.ndarray)):
                    mean_val = float(np.mean(val))
                    print(f"{pretty_names.get(key, key):25s}: Mean={mean_val:.4f} (All: {np.array2string(np.asarray(val), precision=4, separator=', ')})")
                else:
                    print(f"{pretty_names.get(key, key):25s}: {val}")
        print("=" * 50)

    def plot_prediction_subset_compact(
        self,
        n_subjects=5,
        include_best_worst=True,
        random_seed=42,
        demeaned=False,
        show_both=True,
        subject_selection_metric="demeaned_pearson",
        dpi=360,
        figsize=None,
        show=True,
        point_alpha=0.20,
        point_size=1.4,
        scatter_max_points=12000,
    ):
        """
        Compact prediction viewer with 1-2 rows per subject:
        [target matrix | prediction matrix | scatter + metrics].

        Subject selection defaults to a random subset of size n_subjects with optional
        best/worst straddling based on per-subject correlation metric.
        """
        n_total = int(self.preds.shape[0])
        if n_total == 0:
            raise ValueError("No subjects available in this evaluator partition.")
        n_subjects = int(max(1, min(n_subjects, n_total)))

        def _safe_corr(x, y):
            x = np.asarray(x)
            y = np.asarray(y)
            if x.size == 0 or y.size == 0:
                return np.nan
            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                return np.nan
            return float(np.corrcoef(x, y)[0, 1])

        # Ranking metrics for optional best/worst straddling.
        dm_true_all = self.targets - self.train_mean
        dm_pred_all = self.preds - self.train_mean
        raw_r_all = np.array([_safe_corr(self.targets[i], self.preds[i]) for i in range(n_total)], dtype=np.float64)
        dm_r_all = np.array([_safe_corr(dm_true_all[i], dm_pred_all[i]) for i in range(n_total)], dtype=np.float64)
        if subject_selection_metric == "demeaned_pearson":
            ranking_scores = dm_r_all
        elif subject_selection_metric == "pearson":
            ranking_scores = raw_r_all
        else:
            raise ValueError(
                f"Unknown subject_selection_metric: {subject_selection_metric}. "
                "Use 'demeaned_pearson' or 'pearson'."
            )
        valid_rank = np.where(np.isfinite(ranking_scores))[0]
        if valid_rank.size == 0:
            valid_rank = np.arange(n_total)

        rng = np.random.default_rng(random_seed)
        selected = []
        roles = []
        if include_best_worst and n_subjects >= 2:
            best_idx = int(valid_rank[np.nanargmax(ranking_scores[valid_rank])])
            worst_idx = int(valid_rank[np.nanargmin(ranking_scores[valid_rank])])
            if worst_idx == best_idx:
                remaining_valid = valid_rank[valid_rank != best_idx]
                worst_idx = int(remaining_valid[0]) if remaining_valid.size > 0 else best_idx

            middle_n = n_subjects - 2
            pool = np.setdiff1d(np.arange(n_total), np.array([best_idx, worst_idx], dtype=int), assume_unique=False)
            if middle_n > 0:
                if pool.size <= middle_n:
                    middle = pool.tolist()
                else:
                    middle = rng.choice(pool, size=middle_n, replace=False).tolist()
            else:
                middle = []
            selected = [best_idx] + middle + [worst_idx]
            roles = ["best"] + ["random"] * len(middle) + ["worst"]
        elif include_best_worst and n_subjects == 1:
            best_idx = int(valid_rank[np.nanargmax(ranking_scores[valid_rank])])
            selected = [best_idx]
            roles = ["best"]
        else:
            if n_total <= n_subjects:
                selected = list(range(n_total))
            else:
                selected = rng.choice(np.arange(n_total), size=n_subjects, replace=False).tolist()
            roles = ["random"] * len(selected)

        display_modes = (
            [
                ("raw", self.targets, self.preds),
                ("demeaned", self.targets - self.train_mean, self.preds - self.train_mean),
            ]
            if show_both
            else [("demeaned" if demeaned else "raw",
                   self.targets - self.train_mean if demeaned else self.targets,
                   self.preds - self.train_mean if demeaned else self.preds)]
        )
        n_rows = len(selected)
        n_modes = len(display_modes)
        gap_width = 0.24
        mode_starts = []
        width_ratios = []
        col_cursor = 0
        for mm in range(n_modes):
            mode_starts.append(col_cursor)
            width_ratios.extend([1.0, 1.0, 0.22, 1.02])  # true, pred, matrix-scatter gap, scatter
            col_cursor += 4
            if mm < (n_modes - 1):
                width_ratios.append(gap_width)  # inter-mode spacer
                col_cursor += 1
        n_cols = len(width_ratios)
        if figsize is None:
            figsize = (7.3 * n_modes + 0.2 * max(0, n_modes - 1), max(2.0 * n_rows + 0.75, 3.8))

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize,
            dpi=dpi,
            squeeze=False,
            gridspec_kw={"width_ratios": width_ratios},
        )
        fig.subplots_adjust(wspace=0.30, hspace=0.30, top=0.88, bottom=0.055, left=0.11, right=0.985)

        # Turn off spacer columns.
        if n_modes > 1:
            spacer_cols = []
            for mm in range(n_modes - 1):
                spacer_cols.append(mode_starts[mm] + 4)
            for rr in range(n_rows):
                for cc in spacer_cols:
                    axes[rr, cc].axis('off')

        # Build shared square scatter limits per mode for consistent spacing/scale.
        scatter_limits = {}
        for mode_name, targets_view, preds_view in display_modes:
            xs = np.concatenate([np.asarray(targets_view[i], dtype=np.float64) for i in selected])
            ys = np.concatenate([np.asarray(preds_view[i], dtype=np.float64) for i in selected])
            lo = float(np.nanmin([np.nanmin(xs), np.nanmin(ys)]))
            hi = float(np.nanmax([np.nanmax(xs), np.nanmax(ys)]))
            if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
                lo, hi = -1.0, 1.0
            pad = 0.03 * (hi - lo)
            scatter_limits[mode_name] = (lo - pad, hi + pad)

        per_subject = []
        for subj_pos, idx in enumerate(selected):
            role = roles[subj_pos]
            role_suffix = ""
            if role == "best":
                role_suffix = " (best)"
            elif role == "worst":
                role_suffix = " (worst)"
            subject_label = f"ID {self.dataset_partition.ids[idx]}{role_suffix}"

            y_true_raw = np.asarray(self.targets[idx], dtype=np.float64)
            y_pred_raw = np.asarray(self.preds[idx], dtype=np.float64)
            r_raw = _safe_corr(y_true_raw, y_pred_raw)
            r_dm = _safe_corr(y_true_raw - self.train_mean, y_pred_raw - self.train_mean)
            mse_raw = float(np.mean((y_pred_raw - y_true_raw) ** 2))
            r2_raw = float(r2_score(y_true_raw, y_pred_raw))
            y_true_dm = y_true_raw - self.train_mean
            y_pred_dm = y_pred_raw - self.train_mean
            mse_dm = float(np.mean((y_pred_dm - y_true_dm) ** 2))
            r2_dm = float(r2_score(y_true_dm, y_pred_dm))

            per_subject.append(
                {
                    "idx": int(idx),
                    "subject_id": str(self.dataset_partition.ids[idx]),
                    "role": role,
                    "pearson_r": r_raw,
                    "demeaned_pearson_r": r_dm,
                    "r2_raw": r2_raw,
                    "mse_raw": mse_raw,
                    "r2_demeaned": r2_dm,
                    "mse_demeaned": mse_dm,
                }
            )

            for mode_idx, (mode_name, targets_view, preds_view) in enumerate(display_modes):
                row = subj_pos
                col0 = mode_starts[mode_idx]
                y_true = np.asarray(targets_view[idx], dtype=np.float64)
                y_pred = np.asarray(preds_view[idx], dtype=np.float64)

                mat_true = tri2square(y_true, numroi=self.numrois, diagval=0)
                mat_pred = tri2square(y_pred, numroi=self.numrois, diagval=0)
                # Scaling rule:
                # raw mode -> fixed to true range (applied to both true/pred),
                # demeaned mode -> dynamic per connectome (true and pred each get their own).
                absmax_true = float(np.nanmax(np.abs(mat_true)))
                absmax_pred = float(np.nanmax(np.abs(mat_pred)))
                if not np.isfinite(absmax_true) or np.isclose(absmax_true, 0.0):
                    absmax_true = 1.0
                if not np.isfinite(absmax_pred) or np.isclose(absmax_pred, 0.0):
                    absmax_pred = 1.0
                if mode_name == "raw":
                    vmin_true, vmax_true = -absmax_true, absmax_true
                    vmin_pred, vmax_pred = -absmax_true, absmax_true
                else:
                    vmin_true, vmax_true = -absmax_true, absmax_true
                    vmin_pred, vmax_pred = -absmax_pred, absmax_pred

                ax_t = axes[row, col0]
                ax_p = axes[row, col0 + 1]
                ax_c = axes[row, col0 + 2]
                ax_s = axes[row, col0 + 3]
                ax_c.axis('off')

                im_t = ax_t.imshow(
                    mat_true,
                    cmap='RdBu_r',
                    vmin=vmin_true,
                    vmax=vmax_true,
                    aspect='equal',
                    interpolation='nearest',
                )
                ax_t.set_xticks([])
                ax_t.set_yticks([])
                if row == 0:
                    ax_t.set_title("True" if mode_name == "raw" else "True (demeaned)", fontsize=FONT_CONFIG['title'])
                if mode_idx == 0:
                    ax_t.text(
                        -0.17,
                        0.50,
                        subject_label,
                        transform=ax_t.transAxes,
                        rotation=90,
                        va='center',
                        ha='right',
                        fontsize=FONT_CONFIG['tick'] - 2,
                    )
                cax_t = ax_t.inset_axes([1.01, 0.06, 0.026, 0.88])
                cbar_t = fig.colorbar(im_t, cax=cax_t, ticks=[vmin_true, 0.0, vmax_true])
                cbar_t.ax.tick_params(labelsize=FONT_CONFIG['tick'] - 4)

                im_p = ax_p.imshow(
                    mat_pred,
                    cmap='RdBu_r',
                    vmin=vmin_pred,
                    vmax=vmax_pred,
                    aspect='equal',
                    interpolation='nearest',
                )
                ax_p.set_xticks([])
                ax_p.set_yticks([])
                if row == 0:
                    ax_p.set_title("Pred" if mode_name == "raw" else "Pred (demeaned)", fontsize=FONT_CONFIG['title'])
                cax_p = ax_p.inset_axes([1.01, 0.06, 0.026, 0.88])
                cbar_p = fig.colorbar(im_p, cax=cax_p, ticks=[vmin_pred, 0.0, vmax_pred])
                cbar_p.ax.tick_params(labelsize=FONT_CONFIG['tick'] - 4)

                lo, hi = scatter_limits[mode_name]
                if y_true.size > scatter_max_points:
                    rng_scatter = np.random.default_rng((int(random_seed) + 1009 * int(idx) + 37 * mode_idx) % (2**32 - 1))
                    keep_idx = rng_scatter.choice(y_true.size, size=int(scatter_max_points), replace=False)
                    x_scatter = y_true[keep_idx]
                    y_scatter = y_pred[keep_idx]
                else:
                    x_scatter = y_true
                    y_scatter = y_pred
                ax_s.scatter(
                    x_scatter,
                    y_scatter,
                    s=point_size,
                    alpha=point_alpha,
                    color=('#1f77b4' if mode_name == "raw" else '#a64d79'),
                    edgecolors='none',
                    linewidths=0.0,
                    rasterized=True,
                )
                if np.std(y_true) > 1e-12:
                    slope, intercept = np.polyfit(y_true, y_pred, 1)
                    xx = np.array([lo, hi], dtype=np.float64)
                    ax_s.plot(xx, slope * xx + intercept, color='#444444', linewidth=0.95, alpha=0.75)
                ax_s.plot([lo, hi], [lo, hi], 'k--', linewidth=0.9, alpha=0.8)
                ax_s.set_xlim(lo, hi)
                ax_s.set_ylim(lo, hi)
                ax_s.set_aspect('equal', adjustable='box')
                if row == 0:
                    ax_s.set_title("Scatter" if mode_name == "raw" else "Scatter (demeaned)", fontsize=FONT_CONFIG['title'])
                if row == n_rows - 1:
                    ax_s.set_xlabel("True", fontsize=FONT_CONFIG['tick'])
                else:
                    ax_s.set_xlabel("")
                ax_s.set_ylabel("")
                ax_s.tick_params(labelsize=FONT_CONFIG['tick'] - 2)
                ax_s.grid(alpha=0.18)

                if mode_name == "raw":
                    metrics_text = (
                        f"r: {r_raw:.3f}\n"
                        f"R2: {r2_raw:.3f}\n"
                        f"MSE: {mse_raw:.3f}"
                    )
                else:
                    metrics_text = (
                        f"dm r: {r_dm:.3f}\n"
                        f"dm R2: {r2_dm:.3f}\n"
                        f"dm MSE: {mse_dm:.3f}"
                    )
                ax_s.text(
                    0.98,
                    0.02,
                    metrics_text,
                    transform=ax_s.transAxes,
                    va='bottom',
                    ha='right',
                    fontsize=FONT_CONFIG['tick'] - 5,
                    bbox=dict(boxstyle='round,pad=0.06', facecolor='white', edgecolor='gray', alpha=0.58),
                )

        partition_label = str(getattr(self.dataset_partition, "partition", "unknown")).capitalize()
        fig.suptitle(
            f"Prediction Subset Viewer ({partition_label})",
            fontsize=FONT_CONFIG['title'],
            y=0.955,
        )

        metrics = {
            "n_subjects_shown": int(len(selected)),
            "display_modes": [name for name, _, _ in display_modes],
            "include_best_worst": bool(include_best_worst),
            "subject_selection_metric": subject_selection_metric,
            "selected_indices": [int(i) for i in selected],
            "selected_subject_ids": [str(self.dataset_partition.ids[i]) for i in selected],
            "roles": roles,
            "subjects": per_subject,
        }
        if show:
            plt.show()
        return fig, metrics

    def analyze_results(
        self,
        verbose=False,
        filepath=None,
        output_format='md',
        model_name=None,
        order_by='demographic',
        corr_top_row_scale_mode='fixed',
        corr_bottom_row_scale_mode='per_plot',
        include_geodesic=False,
        include_prediction_subset=True,
        prediction_subset_n_subjects=5,
        prediction_subset_include_best_worst=True,
        prediction_subset_selection_metric='demeaned_pearson',
        prediction_subset_seed=42,
        prediction_subset_demeaned=False,
        prediction_subset_show_both=True,
        include_geodesic_metrics=False,
        geodesic_metric_method='log_euclidean',
        geodesic_metric_demeaned=True,
        geodesic_metric_order_by='demographic',
        geodesic_metric_diag_value=1.0,
        geodesic_metric_eps=1e-6,
        geodesic_demeaned=False,
        geodesic_method='log_euclidean',
        geodesic_color_scale_mode='per_plot',
        geodesic_diag_value=1.0,
        geodesic_eps=1e-6,
        geodesic_print_metadata=True,
    ):
        """
        Main analysis function to generate a comprehensive prediction analysis report.
        
        Organizes plots into a report either at a filepath or displays inline in notebook.
        Stores all analysis metrics in a comprehensive dict for wandb tracking.
        
        Args:
            verbose: bool, if True shows both demeaned and non-demeaned results plus Hungarian plots.
                     If False (default), shows only demeaned results without Hungarian plots.
            filepath: str or None, if specified saves report to this path (without extension).
                      If None, displays plots inline in notebook.
            output_format: str, output format when filepath is specified:
                      - 'md' (default): Markdown file with separate PNG figures in {filepath}_plots/
                      - 'jpg': Single combined JPEG image (legacy behavior)
        
        Returns:
            dict: Comprehensive metrics dictionary containing all analysis results,
                  suitable for wandb logging.
        
        Report Layout (verbose=True):
            Row 1: Identifiability Heatmaps (non-demeaned | demeaned) - side by side
            Row 2: Identifiability Violin (non-demeaned | demeaned) - side by side
            Row 3: Hungarian Heatmaps (non-demeaned, then demeaned) - stacked
            Row 4: Hungarian Sample Size (non-demeaned | demeaned) - side by side
            Row 5: PCA Structure (line plot | spatial maps)
        
        Report Layout (verbose=False):
            Row 1: Identifiability Heatmaps (demeaned only)
            Row 2: Identifiability Violin (demeaned only)
            Row 3: PCA Structure (line plot | spatial maps)
        """
        # Initialize comprehensive metrics dict
        all_metrics = {
            'partition': self.dataset_partition.partition,
            'n_subjects': len(self.subject_indices),
            'base_metrics': self._metrics.copy(),
            'model_name': model_name,
        }
        if include_geodesic_metrics:
            geo_base = self.compute_geodesic_metrics(
                order_by=geodesic_metric_order_by,
                demeaned=geodesic_metric_demeaned,
                method=geodesic_metric_method,
                diag_value=geodesic_metric_diag_value,
                eps=geodesic_metric_eps,
            )
            geo_prefix = "geodesic_demeaned" if geodesic_metric_demeaned else "geodesic_raw"
            base_metrics = all_metrics['base_metrics']
            base_metrics[f'{geo_prefix}_top1_acc'] = float(geo_base['top1_acc'])
            base_metrics[f'{geo_prefix}_avg_rank'] = float(geo_base['avg_rank_percentile'])
            base_metrics[f'{geo_prefix}_method'] = str(geo_base['method'])
        
        # Determine whether we're saving to file or displaying inline
        save_to_file = filepath is not None
        show_inline = not save_to_file
        
        # Collect all figures for report generation
        figures_to_combine = []  # For JPEG output (legacy)
        figure_labels = []       # For JPEG output (legacy)
        figures_dict = {}        # For Markdown output (new)
        
        # ========================================================================
        # Analysis 1: Identifiability Heatmaps
        # ========================================================================
        if verbose:
            # Non-demeaned heatmaps
            fig_hm_raw, metrics_hm_raw = self.plot_identifiability_heatmaps(
                order_by=order_by, demeaned=False, 
                include_black_circles=True, include_blue_dots=False, 
                top_row_scale_mode=corr_top_row_scale_mode,
                bottom_row_scale_mode=corr_bottom_row_scale_mode,
                dpi=150, figsize=(12, 9), show=show_inline
            )
            all_metrics['heatmaps_raw'] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                           for k, v in metrics_hm_raw.items() if k != 'ranklist'}
            all_metrics['heatmaps_raw']['ranklist'] = metrics_hm_raw.get('ranklist', [])
            if isinstance(all_metrics['heatmaps_raw']['ranklist'], np.ndarray):
                all_metrics['heatmaps_raw']['ranklist'] = all_metrics['heatmaps_raw']['ranklist'].tolist()
            
            if save_to_file:
                figures_dict['identifiability_heatmaps'] = fig_hm_raw
                figures_to_combine.append(('side_by_side', [fig_hm_raw, None]))
                figure_labels.append('Identifiability Heatmaps')
        
        # Demeaned heatmaps (always)
        fig_hm_dm, metrics_hm_dm = self.plot_identifiability_heatmaps(
            order_by=order_by, demeaned=True, 
            include_black_circles=True, include_blue_dots=False, 
            top_row_scale_mode=corr_top_row_scale_mode,
            bottom_row_scale_mode=corr_bottom_row_scale_mode,
            dpi=150, figsize=(12, 9), show=show_inline
        )
        all_metrics['heatmaps_demeaned'] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                             for k, v in metrics_hm_dm.items() if k != 'ranklist'}
        all_metrics['heatmaps_demeaned']['ranklist'] = metrics_hm_dm.get('ranklist', [])
        if isinstance(all_metrics['heatmaps_demeaned']['ranklist'], np.ndarray):
            all_metrics['heatmaps_demeaned']['ranklist'] = all_metrics['heatmaps_demeaned']['ranklist'].tolist()
        
        if save_to_file:
            figures_dict['identifiability_heatmaps_demeaned'] = fig_hm_dm
            if verbose:
                figures_to_combine[-1] = ('side_by_side', [fig_hm_raw, fig_hm_dm])
            else:
                figures_to_combine.append(('single', fig_hm_dm))
                figure_labels.append('Identifiability Heatmaps (demeaned)')

        # ========================================================================
        # Analysis 1b: Optional SPD Geodesic Heatmaps
        # ========================================================================
        if include_geodesic:
            fig_geo, metrics_geo = self.plot_geodesic_heatmaps(
                order_by=order_by,
                demeaned=geodesic_demeaned,
                method=geodesic_method,
                color_scale_mode=geodesic_color_scale_mode,
                diag_value=geodesic_diag_value,
                eps=geodesic_eps,
                dpi=220,
                figsize=(14, 4.5),
                show=show_inline,
                print_metadata=geodesic_print_metadata,
            )
            all_metrics['geodesic'] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in metrics_geo.items()
                if k != 'ranklist'
            }
            all_metrics['geodesic']['ranklist'] = (
                metrics_geo['ranklist'].tolist()
                if isinstance(metrics_geo.get('ranklist'), np.ndarray)
                else metrics_geo.get('ranklist', [])
            )
            if save_to_file:
                figures_dict['geodesic_heatmaps'] = fig_geo
                figures_to_combine.append(('single', fig_geo))
                figure_labels.append(
                    f"SPD Geodesic Heatmaps{' (demeaned)' if geodesic_demeaned else ''}"
                )
        
        # ========================================================================
        # Analysis 2: Identifiability Violin
        # ========================================================================
        if verbose:
            # Non-demeaned violin
            fig_vio_raw, metrics_vio_raw = self.plot_identifiability_violin(
                include_mean_baseline=True, include_noise_baseline=True,
                demeaned=False, p_threshold=0.05,
                dpi=150, figsize=(7, 5), show=show_inline
            )
            all_metrics['violin_raw'] = extract_violin_metrics(metrics_vio_raw)
            
            if save_to_file:
                figures_dict['identifiability_violin'] = fig_vio_raw
                figures_to_combine.append(('side_by_side', [fig_vio_raw, None]))
                figure_labels.append('Identifiability Violin')
        
        # Demeaned violin (always)
        fig_vio_dm, metrics_vio_dm = self.plot_identifiability_violin(
            include_mean_baseline=True, include_noise_baseline=True,
            demeaned=True, p_threshold=0.05,
            dpi=150, figsize=(7, 5), show=show_inline
        )
        all_metrics['violin_demeaned'] = extract_violin_metrics(metrics_vio_dm)
        
        if save_to_file:
            figures_dict['identifiability_violin_demeaned'] = fig_vio_dm
            if verbose:
                figures_to_combine[-1] = ('side_by_side', [fig_vio_raw, fig_vio_dm])
            else:
                figures_to_combine.append(('single', fig_vio_dm))
                figure_labels.append('Identifiability Violin (demeaned)')
        
        # ========================================================================
        # Analysis 3: Hungarian Matching (verbose only)
        # ========================================================================
        if verbose:
            # Non-demeaned Hungarian heatmaps
            fig_hung_raw, metrics_hung_raw = self.plot_hungarian_heatmaps(
                demeaned=False, dpi=150, figsize=(14, 4), show=show_inline
            )
            all_metrics['hungarian_raw'] = extract_hungarian_metrics(metrics_hung_raw)
            
            # Demeaned Hungarian heatmaps
            fig_hung_dm, metrics_hung_dm = self.plot_hungarian_heatmaps(
                demeaned=True, dpi=150, figsize=(14, 4), show=show_inline
            )
            all_metrics['hungarian_demeaned'] = extract_hungarian_metrics(metrics_hung_dm)
            
            if save_to_file:
                figures_dict['hungarian_heatmaps'] = fig_hung_raw
                figures_dict['hungarian_heatmaps_demeaned'] = fig_hung_dm
                figures_to_combine.append(('stacked', [fig_hung_raw, fig_hung_dm]))
                figure_labels.append('Hungarian Matching')
            
            # Hungarian sample size analysis
            fig_ss_raw, metrics_ss_raw = self.plot_hungarian_sample_size_analysis(
                n_min=2, n_max=75, step=10, n_iterations=1000, 
                demeaned=False, dpi=150, figsize=(9, 5), show=show_inline
            )
            all_metrics['hungarian_sample_size_raw'] = extract_sample_size_metrics(metrics_ss_raw)
            
            fig_ss_dm, metrics_ss_dm = self.plot_hungarian_sample_size_analysis(
                n_min=2, n_max=75, step=10, n_iterations=1000, 
                demeaned=True, dpi=150, figsize=(9, 5), show=show_inline
            )
            all_metrics['hungarian_sample_size_demeaned'] = extract_sample_size_metrics(metrics_ss_dm)
            
            if save_to_file:
                figures_dict['hungarian_sample_size'] = fig_ss_raw
                figures_dict['hungarian_sample_size_demeaned'] = fig_ss_dm
                figures_to_combine.append(('side_by_side', [fig_ss_raw, fig_ss_dm]))
                figure_labels.append('Hungarian Sample Size Analysis')
        
        # ========================================================================
        # Analysis 4: PCA Structure (always)
        # ========================================================================
        (fig_pca_line, fig_pca_grid), metrics_pca = self.evaluate_pca_structure(
            num_pcs=len(self.subject_indices), show_first_pcs=5, show=show_inline
        )
        all_metrics['pca'] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                              for k, v in metrics_pca.items()}
        
        if save_to_file:
            figures_dict['pca_line'] = fig_pca_line
            figures_dict['pca_spatial'] = fig_pca_grid
            figures_to_combine.append(('side_by_side', [fig_pca_line, fig_pca_grid]))
            figure_labels.append('PCA Structure')

        # ========================================================================
        # Analysis 5: Compact Prediction Subset Viewer (optional, default on)
        # ========================================================================
        if include_prediction_subset:
            fig_pred_subset, metrics_pred_subset = self.plot_prediction_subset_compact(
                n_subjects=prediction_subset_n_subjects,
                include_best_worst=prediction_subset_include_best_worst,
                subject_selection_metric=prediction_subset_selection_metric,
                random_seed=prediction_subset_seed,
                demeaned=prediction_subset_demeaned,
                show_both=prediction_subset_show_both,
                dpi=200,
                figsize=None,
                show=show_inline,
            )
            all_metrics['prediction_subset'] = metrics_pred_subset
            if save_to_file:
                figures_dict['prediction_subset'] = fig_pred_subset
                figures_to_combine.append(('single', fig_pred_subset))
                figure_labels.append('Prediction Subset Viewer')
        
        # ========================================================================
        # Generate report if filepath specified
        # ========================================================================
        if save_to_file:
            if output_format == 'md':
                # New markdown output
                generate_markdown_report(figures_dict, all_metrics, filepath, verbose=verbose)
            else:
                # Legacy JPEG output
                generate_report(figures_to_combine, figure_labels, filepath)
                # Close all figures to free memory
                for item in figures_to_combine:
                    layout_type, figs = item
                    if layout_type == 'single':
                        plt.close(figs)
                    else:
                        for f in figs:
                            if f is not None:
                                plt.close(f)
        
        # Store metrics as instance attribute
        self.analysis_metrics = all_metrics
        
        return all_metrics
    

    def visualize_individual_prediction(self, idx=None, subject_id=None, scale='self', 
                                         dpi=150, figsize=(15, 10), show=True):
        """
        Visualize the prediction for a single subject with comparison to ground truth.
        
        Shows two rows of plots:
        - Row 1 (Original): True matrix, Predicted matrix, Scatter plot with regression line
        - Row 2 (Demeaned): Same plots but with training mean subtracted
        
        Args:
            idx: int, optional. Index into preds/targets arrays (position in partition).
                 Must be < number of subjects in this partition.
            subject_id: str or int, optional. Subject ID as defined in partition_ids.
                        The position of the ID in partition_ids equals the idx.
            scale: str, colorbar scaling mode (default 'self'):
                - 'self': Each matrix uses its own colorbar range
                - 'true': Both true and predicted matrices use the true target's range
            dpi: int, figure resolution (default 150)
            figsize: tuple, figure size (default (15, 10))
            show: bool, whether to display the plot (default True)
        
        Returns:
            tuple: (fig, metrics_dict) where metrics includes Pearson correlations for
                   both original and demeaned comparisons
        
        Note:
            Exactly one of idx or subject_id must be provided.
        """
        # Validate inputs
        if idx is None and subject_id is None:
            raise ValueError("Must provide either idx or subject_id")
        if idx is not None and subject_id is not None:
            raise ValueError("Provide only one of idx or subject_id, not both")
        
        n_subjects = self.preds.shape[0]
        partition_ids = self.dataset_partition.ids
        
        # Resolve idx and subject_id
        if subject_id is not None:
            # Convert to string if int for matching
            subject_id_str = str(subject_id)
            if subject_id_str not in partition_ids:
                raise ValueError(f"subject_id '{subject_id}' not found in partition. "
                                f"Available IDs: {partition_ids[:5]}... (showing first 5)")
            idx = partition_ids.index(subject_id_str)
        else:
            if idx < 0 or idx >= n_subjects:
                raise ValueError(f"idx must be in range [0, {n_subjects}), got {idx}")
            subject_id = partition_ids[idx]
        
        # Get prediction and target for this subject
        y_true = self.targets[idx]
        y_pred = self.preds[idx]
        
        # Compute demeaned versions
        y_true_demeaned = y_true - self.train_mean
        y_pred_demeaned = y_pred - self.train_mean
        
        # Compute number of ROIs and convert to square matrices
        n_edges = y_true.size
        n_roi = int((1 + np.sqrt(1 + 8 * n_edges)) / 2)
        
        mat_true = tri2square(y_true, numroi=n_roi)
        mat_pred = tri2square(y_pred, numroi=n_roi)
        mat_true_dm = tri2square(y_true_demeaned, numroi=n_roi)
        mat_pred_dm = tri2square(y_pred_demeaned, numroi=n_roi)
        
        # Look up subject metadata
        metadata_df = self.dataset.metadata_df
        subj_meta = metadata_df[metadata_df['subject'] == int(subject_id)]
        
        if len(subj_meta) == 0:
            # Try string match
            subj_meta = metadata_df[metadata_df['subject'].astype(str) == str(subject_id)]
        
        # Format metadata string
        if len(subj_meta) > 0:
            row = subj_meta.iloc[0]
            meta_str = (
                f"Subject: {subject_id}  |  "
                f"Age: {row.get('age', 'N/A')}  |  "
                f"Sex: {row.get('sex', 'N/A')}  |  "
                f"Race/Ethnicity: {row.get('Race_Ethnicity', 'N/A')}  |  "
                f"Family: {row.get('Family_Relation', 'N/A')}"
            )
        else:
            meta_str = f"Subject: {subject_id}  |  (Metadata not found)"
        
        # Compute Pearson correlations
        r_orig, _ = pearsonr(y_true, y_pred)
        r_demeaned, _ = pearsonr(y_true_demeaned, y_pred_demeaned)
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=dpi)
        
        # Add metadata as super title
        fig.suptitle(meta_str, fontsize=FONT_CONFIG['title'], fontweight='bold', y=0.98)
        
        # Dynamic colorbar scaling
        vmax_true = np.abs(mat_true).max()
        vmax_pred = np.abs(mat_pred).max()
        vmax_true_dm = np.abs(mat_true_dm).max()
        vmax_pred_dm = np.abs(mat_pred_dm).max()
        
        # Apply scaling mode
        if scale == 'self':
            # Each matrix uses its own colorbar range (do nothing)
            pass
        elif scale == 'true':
            # Use true target's range for both matrices
            vmax_pred = vmax_true
            vmax_pred_dm = vmax_true_dm
        else:
            raise ValueError(f"scale must be 'self' or 'true', got '{scale}'")
        
        # ========== Row 0: Original ==========
        # True matrix
        im00 = axes[0, 0].imshow(mat_true, aspect='equal', cmap='RdBu_r', vmin=-vmax_true, vmax=vmax_true)
        axes[0, 0].set_title(f"True Target", fontsize=FONT_CONFIG['title'])
        cbar00 = fig.colorbar(im00, ax=axes[0, 0], orientation='vertical', fraction=0.046, pad=0.04)
        cbar00.ax.tick_params(labelsize=FONT_CONFIG['tick'] - 2)
        axes[0, 0].set_xticks([])
        axes[0, 0].set_yticks([])
        
        # Predicted matrix
        im01 = axes[0, 1].imshow(mat_pred, aspect='equal', cmap='RdBu_r', vmin=-vmax_pred, vmax=vmax_pred)
        axes[0, 1].set_title(f"Predicted Target", fontsize=FONT_CONFIG['title'])
        cbar01 = fig.colorbar(im01, ax=axes[0, 1], orientation='vertical', fraction=0.046, pad=0.04)
        cbar01.ax.tick_params(labelsize=FONT_CONFIG['tick'] - 2)
        axes[0, 1].set_xticks([])
        axes[0, 1].set_yticks([])
        
        # Scatter plot with regression
        axes[0, 2].scatter(y_true, y_pred, alpha=0.3, s=5, c='steelblue')
        # Regression line
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        x_line = np.linspace(y_true.min(), y_true.max(), 100)
        axes[0, 2].plot(x_line, p(x_line), 'r-', linewidth=2, label=f'r = {r_orig:.3f}')
        axes[0, 2].set_xlabel('True', fontsize=FONT_CONFIG['label'])
        axes[0, 2].set_ylabel('Predicted', fontsize=FONT_CONFIG['label'])
        axes[0, 2].set_title(f'True vs Predicted (Original)', fontsize=FONT_CONFIG['title'])
        axes[0, 2].legend(loc='upper left', fontsize=FONT_CONFIG['legend'])
        axes[0, 2].tick_params(labelsize=FONT_CONFIG['tick'] - 2)
        
        # ========== Row 1: Demeaned ==========
        # True matrix (demeaned)
        im10 = axes[1, 0].imshow(mat_true_dm, aspect='equal', cmap='RdBu_r', vmin=-vmax_true_dm, vmax=vmax_true_dm)
        axes[1, 0].set_title(f"True Target (demeaned)", fontsize=FONT_CONFIG['title'])
        cbar10 = fig.colorbar(im10, ax=axes[1, 0], orientation='vertical', fraction=0.046, pad=0.04)
        cbar10.ax.tick_params(labelsize=FONT_CONFIG['tick'] - 2)
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])
        
        # Predicted matrix (demeaned)
        im11 = axes[1, 1].imshow(mat_pred_dm, aspect='equal', cmap='RdBu_r', vmin=-vmax_pred_dm, vmax=vmax_pred_dm)
        axes[1, 1].set_title(f"Predicted Target (demeaned)", fontsize=FONT_CONFIG['title'])
        cbar11 = fig.colorbar(im11, ax=axes[1, 1], orientation='vertical', fraction=0.046, pad=0.04)
        cbar11.ax.tick_params(labelsize=FONT_CONFIG['tick'] - 2)
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
        
        # Scatter plot with regression (demeaned)
        axes[1, 2].scatter(y_true_demeaned, y_pred_demeaned, alpha=0.3, s=5, c='steelblue')
        # Regression line
        z_dm = np.polyfit(y_true_demeaned, y_pred_demeaned, 1)
        p_dm = np.poly1d(z_dm)
        x_line_dm = np.linspace(y_true_demeaned.min(), y_true_demeaned.max(), 100)
        axes[1, 2].plot(x_line_dm, p_dm(x_line_dm), 'r-', linewidth=2, label=f'r = {r_demeaned:.3f}')
        axes[1, 2].set_xlabel('True (demeaned)', fontsize=FONT_CONFIG['label'])
        axes[1, 2].set_ylabel('Predicted (demeaned)', fontsize=FONT_CONFIG['label'])
        axes[1, 2].set_title(f'True vs Predicted (Demeaned)', fontsize=FONT_CONFIG['title'])
        axes[1, 2].legend(loc='upper left', fontsize=FONT_CONFIG['legend'])
        axes[1, 2].tick_params(labelsize=FONT_CONFIG['tick'] - 2)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        metrics = {
            'subject_id': subject_id,
            'idx': idx,
            'pearson_original': float(r_orig),
            'pearson_demeaned': float(r_demeaned),
        }
        
        if show:
            plt.show()
        
        return fig, metrics
