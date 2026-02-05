import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr, ttest_1samp, ttest_ind, false_discovery_control
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import io
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import warnings

from data.data_utils import *
from eval_utils import *

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


class Evaluator:
    """
    Evaluation class for prediction vs target analysis.

    Args:
        preds: torch.Tensor or np.ndarray, predictions (n_subjects x n_features)
        targets: torch.Tensor or np.ndarray, targets (n_subjects x n_features)
        dataset_partition: Dataset partition (e.g., train/val/test split) with .indices attribute
        dataset: Full dataset object with train_mean attributes (fc_train_avg or sc_train_avg)
    """
    
    def __init__(self, preds, targets, dataset_partition, dataset):
        # Convert to numpy arrays
        self.preds = preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else np.asarray(preds)
        self.targets = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)
    
        # Store dataset references
        self.dataset_partition = dataset_partition
        self.dataset = dataset
        self.subject_indices = dataset_partition.indices
        self.numrois = dataset.fc_matrices.shape[1]
        
        # Get training mean and std (depends on target modality)
        # Use upper_triangles (vectorized) since that's what the model predicts
        train_indices = dataset.trainvaltest_partition_indices["train"]
        if dataset.target == "FC":
            self.train_mean = dataset.fc_train_avg
            # fc_upper_triangles shape: (N_subjects, n_edges)
            train_data = dataset.fc_upper_triangles[train_indices]
            if isinstance(train_data, torch.Tensor):
                train_data = train_data.cpu().numpy()
            self.train_std = np.std(train_data, axis=0)
        else:
            self.train_mean = dataset.sc_train_avg
            train_data = dataset.sc_upper_triangles[train_indices]
            if isinstance(train_data, torch.Tensor):
                train_data = train_data.cpu().numpy()
            self.train_std = np.std(train_data, axis=0)
        # Compute correlation matrices (targets, preds) so rows=targets, cols=preds
        # This answers: "for each target, which prediction matches best?"
        self.corr_matrix = compute_corr_matrix(self.targets, self.preds)
        self.corr_matrix_demeaned = compute_corr_matrix(
            self.targets - self.train_mean, 
            self.preds - self.train_mean
        )
    
        self._metrics = self._compute_metrics()
        
        # PCA objects (computed lazily when evaluate_pca_structure is called)
        self._pca_preds = None
        self._pca_targets = None    
    

    def _compute_metrics(self):
        """Compute all evaluation metrics."""
        metrics = {}
        
        # MSE and R2
        mse_values = mean_squared_error(self.targets, self.preds, multioutput='raw_values')
        r2_values = r2_score(self.targets, self.preds, multioutput='raw_values')
        
        metrics['mse'] = np.mean(mse_values)
        metrics['r2'] = np.mean(r2_values)
        
        # Correlation-based metrics
        metrics['pearson'] = np.mean(np.diag(self.corr_matrix))
        metrics['demeaned_pearson'] = np.mean(np.diag(self.corr_matrix_demeaned))
        metrics['avg_rank'] = corr_avg_rank(cc=self.corr_matrix)
        metrics['top1_acc'] = corr_topn_accuracy(cc=self.corr_matrix, topn=1)
        
        return metrics

    
    def _fit_pca(self):
        """Fit PCA models for predictions and targets (subject-mode PCA)."""
        # Mean center
        targets_centered = self.targets - self.train_mean  # (n, d)
        preds_centered = self.preds - self.train_mean  # (n, d)
        
        # Transpose to features x subjects (d, n) for subject-mode PCA
        targets_transposed = targets_centered.T
        preds_transposed = preds_centered.T
        
        # PCA for targets
        self._pca_targets = PCA()
        self._pca_targets.fit(targets_transposed)
        
        # PCA for predictions
        self._pca_preds = PCA()
        self._pca_preds.fit(preds_transposed)
        
        # Orthonormality check for bases (across subjects)
        B_targets = self._pca_targets.components_
        B_preds = self._pca_preds.components_
        assert np.allclose(B_targets @ B_targets.T, np.eye(B_targets.shape[0]), atol=1e-6), \
            "B_targets is not orthonormal"
        assert np.allclose(B_preds @ B_preds.T, np.eye(B_preds.shape[0]), atol=1e-6), \
            "B_preds is not orthonormal"

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

    def _compute_subject_order(self, order_by='original'):
        """
        Return reordered subject indices for visualization.
        
        Args:
            order_by: str, one of:
                - 'original': keep original order
                - 'family': group by Family_ID from dataset.metadata_df
                - 'demographic': group by unique demographic categories from one_hot_covariate
        
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
            # Get one_hot_covariate tuple and concatenate to form unique demographic category
            one_hot_tuple = self.dataset.covariate_one_hot_tuple
            # Subset each array in the one_hot_tuple to correspond to subjects in this partition
            partition_covariates = []
            for cov_array in one_hot_tuple:
                cov_array_np = cov_array.numpy() if isinstance(cov_array, torch.Tensor) else np.asarray(cov_array)
                # self.subject_indices comes from the current dataset partition
                partition_covariates.append(cov_array_np[self.subject_indices])
            
            # Concatenate all one-hot arrays to make a [num_subjects, total_covariate_dim] array
            concat_covariates = np.concatenate(partition_covariates, axis=1)
            
            # Assign a categorical demographic value (integer) to each subject by unique row
            # This produces: categories - an array of labels; unique_rows - the set of unique category rows
            unique_rows, categories = np.unique(concat_covariates, axis=0, return_inverse=True)
            print(f"Number of unique categories: {len(unique_rows)}")
            # Now sort by group/category so that all same-category subjects are together
            sort_order = np.argsort(categories)
            return sort_order
        
        else:
            raise ValueError(f"Unknown order_by: {order_by}. Use 'original', 'family', or 'demographic'.")

    def plot_identifiability_heatmaps(self, order_by='original', include_black_circles=True,
                                       include_blue_dots=True, demeaned=False, 
                                       dpi=200, figsize=(14, 10), show=True):
        """
        Plot identifiability heatmaps for target and predicted connectomes.
        
        Generates a 2x2 figure with:
        - Target vs Target
        - Predicted vs Predicted
        - Target vs Predicted with optional black dots for max similarity
          and optional blue dots for predictions closer than correct match
        
        Args:
            order_by: str, subject ordering method:
                - 'original': original order (default)
                - 'family': group by Family_ID
                - 'demographic': group by demographic categories
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
        
        # Compute correlation matrix for ordered data (targets, preds: rows=targets, cols=preds)
        corr_matrix = compute_corr_matrix(targets_data, preds_data)
        mean_corr = np.mean(np.diag(corr_matrix))
        
        # Create figure
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize, dpi=dpi)
        axs = axs.flatten()
        
        # Store ranklist for summary
        observed_vs_pred_ranklist = None
        
        # Define the three heatmap comparisons
        comparisons = [
            (targets_data, targets_data, "Target", "Target"),
            (preds_data, preds_data, "Predicted", "Predicted"),
            (targets_data, preds_data, "Target", "Predicted"),
        ]
        
        for iax, (data1, data2, label1, label2) in enumerate(comparisons):
            sim = compute_corr_matrix(data1, data2)
            
            avgrank = corr_avg_rank(cc=sim)
            
            ax = axs[iax]
            im = ax.imshow(sim, vmin=-1, vmax=1, cmap='RdBu_r')
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            titlestr = f'{label1} vs {label2}{demean_label}'
            
            # Special handling for Target vs Predicted
            if label1 == "Target" and label2 == "Predicted":
                ranklist = []
                marker_kwargs = dict(markersize=5, markerfacecolor='none')
                tiny_blue_kwargs = dict(marker='o', color='blue', markersize=1, 
                                       alpha=0.7, linestyle='None')
                
                for i in range(sim.shape[0]):
                    cci = sim[i, :]  # row for this target
                    # Scale row from 0-1
                    cci_scaled = (cci - np.nanmin(cci)) / (np.nanmax(cci) - np.nanmin(cci))
                    cci_scaled = (cci_scaled - 0.5) * 0.8  # scale for display
                    
                    cci_maxidx = np.argmax(cci)
                    cci_sortidx = np.argsort(np.argsort(cci)[::-1])
                    
                    # Black circle for most similar prediction
                    if include_black_circles:
                        ax.plot(cci_maxidx, i - cci_scaled[cci_maxidx], 'ko', **marker_kwargs)
                    
                    # Blue dots for predictions closer than correct match
                    if include_blue_dots:
                        closer_pred_idxs = np.where(cci_scaled > cci_scaled[i])[0]
                        if closer_pred_idxs.size > 0:
                            ax.plot(
                                closer_pred_idxs,
                                np.repeat(i, len(closer_pred_idxs)) - cci_scaled[closer_pred_idxs],
                                **tiny_blue_kwargs
                            )
                    
                    ranklist.append(cci_sortidx[i] + 1)
                
                observed_vs_pred_ranklist = np.array(ranklist)
                avgrank_index = np.mean(ranklist)
                titlestr += f'\nAvg Rank {avgrank_index:.1f} out of {sim.shape[0]}, %ile: {avgrank:.3f}'
            
            ax.set_title(titlestr, fontsize=FONT_CONFIG['title'])
            ax.set_xlabel(label2, fontsize=FONT_CONFIG['label'])
            ax.set_ylabel(label1, fontsize=FONT_CONFIG['label'])
            
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=FONT_CONFIG['tick'])
        
        # Compute summary metrics
        n_subjects = len(observed_vs_pred_ranklist)
        n_top1 = np.sum(observed_vs_pred_ranklist == 1)
        top1_acc = n_top1 / n_subjects
        avgrank_percentile = 1 - (np.mean(observed_vs_pred_ranklist) / n_subjects)
        
        # Fourth panel: leave blank or use for additional info
        ax = axs[3]
        ax.axis('off')
        
        # Compute chance levels
        chance_top1 = 1 / n_subjects
        chance_avgrank = 0.5
        
        # Add summary text in fourth panel
        summary_text = (
            f"Summary Metrics\n"
            f"{'─' * 35}\n"
            f"Mean corr (target vs pred): {mean_corr:.3f}\n"
            f"Demeaned: {demeaned}\n"
            f"Top-1 accuracy: {top1_acc:.3f} (chance={chance_top1:.3f})\n"
            f"  ({n_top1} of {n_subjects} subjects had rank 1)\n"
            f"Avg rank %ile: {avgrank_percentile:.3f} (chance={chance_avgrank})\n"
            f"{'─' * 35}\n"
            f"Order: {order_by}\n"
            f"N subjects: {n_subjects}"
        )
        ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=FONT_CONFIG['title'],
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor='gray', alpha=0.8),
                family='monospace')
        
        plt.tight_layout()
        
        metrics = {
            'mean_corr': mean_corr,
            'demeaned': demeaned,
            'top1_acc': top1_acc,
            'avg_rank_percentile': avgrank_percentile,
            'ranklist': observed_vs_pred_ranklist,
            'chance_top1': chance_top1,
            'chance_avgrank': chance_avgrank,
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
                                     noise_scale=1.0, demeaned=False, seed=42,
                                     p_threshold=0.001, dpi=150, figsize=(8, 6), show=True):
        """
        Plot identifiability violin plot comparing intraindividual vs interindividual correlations.
        
        Shows distributions for:
        - pFC (model predictions)
        - Mean eFC baseline (optional, not shown if demeaned=True)
        - Mean eFC + noise baseline (optional)
        
        Args:
            include_mean_baseline: bool, include mean eFC baseline condition
            include_noise_baseline: bool, include mean eFC + noise baseline
            noise_scale: float, noise scale relative to training std
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
            r_inter_all.append(mean_stats['r_inter'])
            stats_all.append(mean_stats)
        
        # 3. Noise baseline
        if include_noise_baseline:
            noise_preds = generate_noise_baseline(self.train_mean, self.train_std, n_subjects, 
                                                   noise_scale=noise_scale, seed=seed)
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
                                 noise_scale=1.0, demeaned=False, seed=42,
                                 dpi=150, figsize=(15, 5), show=True):
        """
        Plot Hungarian matching heatmaps comparing pFC, noised null, and permuted null.
        
        Shows side-by-side correlation heatmaps with black dots indicating 
        Hungarian optimal assignments and Top-1 accuracy for each condition.
        
        Args:
            include_noise_baseline: bool, include noised baseline
            include_permute_baseline: bool, include permuted baseline
            noise_scale: float, noise scale for null baseline
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
                                                   noise_scale=noise_scale, seed=seed)
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
                                             noise_scale=1.0, demeaned=False, 
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
            noise_scale: float, noise scale for null baseline
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
                                               noise_scale=noise_scale, seed=seed)
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

    def analyze_results(self, verbose=False, filepath=None, output_format='md'):
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
        }
        
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
                order_by='family', demeaned=False, 
                include_black_circles=True, include_blue_dots=False, 
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
            order_by='family', demeaned=True, 
            include_black_circles=True, include_blue_dots=False, 
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
        # Analysis 2: Identifiability Violin
        # ========================================================================
        if verbose:
            # Non-demeaned violin
            fig_vio_raw, metrics_vio_raw = self.plot_identifiability_violin(
                include_mean_baseline=True, include_noise_baseline=True,
                noise_scale=1.0, demeaned=False, p_threshold=0.05,
                dpi=150, figsize=(7, 5), show=show_inline
            )
            all_metrics['violin_raw'] = self._extract_violin_metrics(metrics_vio_raw)
            
            if save_to_file:
                figures_dict['identifiability_violin'] = fig_vio_raw
                figures_to_combine.append(('side_by_side', [fig_vio_raw, None]))
                figure_labels.append('Identifiability Violin')
        
        # Demeaned violin (always)
        fig_vio_dm, metrics_vio_dm = self.plot_identifiability_violin(
            include_mean_baseline=True, include_noise_baseline=True,
            noise_scale=1.0, demeaned=True, p_threshold=0.05,
            dpi=150, figsize=(7, 5), show=show_inline
        )
        all_metrics['violin_demeaned'] = self._extract_violin_metrics(metrics_vio_dm)
        
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
            all_metrics['hungarian_raw'] = self._extract_hungarian_metrics(metrics_hung_raw)
            
            # Demeaned Hungarian heatmaps
            fig_hung_dm, metrics_hung_dm = self.plot_hungarian_heatmaps(
                demeaned=True, dpi=150, figsize=(14, 4), show=show_inline
            )
            all_metrics['hungarian_demeaned'] = self._extract_hungarian_metrics(metrics_hung_dm)
            
            if save_to_file:
                figures_dict['hungarian_heatmaps'] = fig_hung_raw
                figures_dict['hungarian_heatmaps_demeaned'] = fig_hung_dm
                figures_to_combine.append(('stacked', [fig_hung_raw, fig_hung_dm]))
                figure_labels.append('Hungarian Matching')
            
            # Hungarian sample size analysis
            fig_ss_raw, metrics_ss_raw = self.plot_hungarian_sample_size_analysis(
                n_min=2, n_max=100, step=5, n_iterations=1000, 
                demeaned=False, dpi=150, figsize=(9, 5), show=show_inline
            )
            all_metrics['hungarian_sample_size_raw'] = self._extract_sample_size_metrics(metrics_ss_raw)
            
            fig_ss_dm, metrics_ss_dm = self.plot_hungarian_sample_size_analysis(
                n_min=2, n_max=100, step=5, n_iterations=1000, 
                demeaned=True, dpi=150, figsize=(9, 5), show=show_inline
            )
            all_metrics['hungarian_sample_size_demeaned'] = self._extract_sample_size_metrics(metrics_ss_dm)
            
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
        # Generate report if filepath specified
        # ========================================================================
        if save_to_file:
            if output_format == 'md':
                # New markdown output
                self._generate_markdown_report(figures_dict, all_metrics, filepath, verbose=verbose)
            else:
                # Legacy JPEG output
                self._generate_report(figures_to_combine, figure_labels, filepath)
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
    
    def _extract_violin_metrics(self, results):
        """Extract wandb-friendly metrics from violin plot results."""
        extracted = {}
        for condition, stats in results.items():
            prefix = condition.lower().replace(' ', '_').replace('(', '').replace(')', '')
            extracted[f'{prefix}_mean_r_intra'] = float(stats['mean_r_intra'])
            extracted[f'{prefix}_mean_r_inter'] = float(stats['mean_r_inter'])
            extracted[f'{prefix}_mean_d'] = float(stats['mean_d'])
            extracted[f'{prefix}_t_stat'] = float(stats['t_stat'])
            extracted[f'{prefix}_p_value'] = float(stats['p_value'])
            extracted[f'{prefix}_cohen_d'] = float(stats['cohen_d'])
        return extracted
    
    def _extract_hungarian_metrics(self, results):
        """Extract wandb-friendly metrics from Hungarian matching results."""
        extracted = {}
        for condition, hung in results.items():
            prefix = condition.lower().replace(' ', '_').replace('(', '').replace(')', '')
            extracted[f'{prefix}_accuracy'] = float(hung['accuracy'])
            extracted[f'{prefix}_n_correct'] = int(hung['n_correct'])
        return extracted
    
    def _extract_sample_size_metrics(self, results):
        """Extract wandb-friendly metrics from sample size analysis results."""
        return {
            'sample_sizes': results['sample_sizes'].tolist(),
            'pfc_mean_accuracy': results['pFC']['mean'].tolist(),
            'null_noise_mean_accuracy': results['Null (noise)']['mean'].tolist(),
            'null_permute_mean_accuracy': results['Null (permute)']['mean'].tolist(),
            'n_significant_vs_permute': int(np.sum(results['significant_permute'])),
            'n_significant_vs_noise': int(np.sum(results['significant_noise'])),
        }
    
    def _generate_report(self, figures_to_combine, figure_labels, filepath):
        """
        Generate a combined .jpg report from collected figures.
        
        Args:
            figures_to_combine: list of tuples (layout_type, figs)
                - 'single': single figure
                - 'side_by_side': [left_fig, right_fig]
                - 'stacked': [top_fig, bottom_fig]
            figure_labels: list of str, section titles for each figure group
            filepath: output path for the report
        """
        # Configuration
        section_spacing = 120  # pixels between sections
        title_height = 100  # height of title banner
        title_bg_color = (230, 230, 235)  # light gray-blue background
        title_text_color = (40, 40, 50)  # dark gray text
        
        # Convert figures to images
        images = []
        
        for idx, (layout_type, figs) in enumerate(figures_to_combine):
            if layout_type == 'single':
                img = self._fig_to_image(figs)
                images.append(img)
            
            elif layout_type == 'side_by_side':
                left_img = self._fig_to_image(figs[0]) if figs[0] is not None else None
                right_img = self._fig_to_image(figs[1]) if figs[1] is not None else None
                
                if left_img is not None and right_img is not None:
                    # Resize to same height
                    max_height = max(left_img.height, right_img.height)
                    left_img = self._resize_to_height(left_img, max_height)
                    right_img = self._resize_to_height(right_img, max_height)
                    
                    # Combine horizontally with small gap
                    gap = 20
                    combined = Image.new('RGB', 
                                        (left_img.width + gap + right_img.width, max_height),
                                        (255, 255, 255))
                    combined.paste(left_img, (0, 0))
                    combined.paste(right_img, (left_img.width + gap, 0))
                    images.append(combined)
                elif left_img is not None:
                    images.append(left_img)
                elif right_img is not None:
                    images.append(right_img)
            
            elif layout_type == 'stacked':
                top_img = self._fig_to_image(figs[0]) if figs[0] is not None else None
                bottom_img = self._fig_to_image(figs[1]) if figs[1] is not None else None
                
                if top_img is not None and bottom_img is not None:
                    # Resize to same width
                    max_width = max(top_img.width, bottom_img.width)
                    top_img = self._resize_to_width(top_img, max_width)
                    bottom_img = self._resize_to_width(bottom_img, max_width)
                    
                    # Combine vertically with small gap
                    gap = 15
                    combined = Image.new('RGB',
                                        (max_width, top_img.height + gap + bottom_img.height),
                                        (255, 255, 255))
                    combined.paste(top_img, (0, 0))
                    combined.paste(bottom_img, (0, top_img.height + gap))
                    images.append(combined)
                elif top_img is not None:
                    images.append(top_img)
                elif bottom_img is not None:
                    images.append(bottom_img)
        
        # Combine all rows vertically with titles and spacing
        if not images:
            warnings.warn("No images to combine for report")
            return
        
        # Find max width for final combining
        max_width = max(img.width for img in images)
        
        # Resize all to same width
        resized_images = [self._resize_to_width(img, max_width) for img in images]
        
        # Create title banners using matplotlib (has bundled fonts)
        title_images = []
        for idx in range(len(resized_images)):
            title = figure_labels[idx] if idx < len(figure_labels) else f"Section {idx + 1}"
            title_img = self._create_title_banner(title, max_width, title_height, title_bg_color, title_text_color)
            title_images.append(title_img)
        
        # Calculate total height including titles and spacing
        n_sections = len(resized_images)
        total_height = (
            sum(img.height for img in resized_images) +
            sum(img.height for img in title_images) +
            (n_sections - 1) * section_spacing  # spacing between sections (not after last)
        )
        
        final_report = Image.new('RGB', (max_width, total_height), (255, 255, 255))
        
        y_offset = 0
        for idx, img in enumerate(resized_images):
            # Paste title banner
            final_report.paste(title_images[idx], (0, y_offset))
            y_offset += title_images[idx].height
            
            # Paste the image
            final_report.paste(img, (0, y_offset))
            y_offset += img.height
            
            # Add spacing after section (except for last section)
            if idx < n_sections - 1:
                y_offset += section_spacing
        
        # Ensure filepath has .jpg extension
        if not filepath.lower().endswith('.jpg') and not filepath.lower().endswith('.jpeg'):
            filepath = filepath + '.jpg'
        
        # Save with high quality
        final_report.save(filepath, 'JPEG', quality=95)
        print(f"Report saved to: {filepath}")
    
    def _fig_to_image(self, fig):
        """Convert matplotlib figure to PIL Image."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img
    
    def _resize_to_height(self, img, target_height):
        """Resize image to target height maintaining aspect ratio."""
        if img.height == target_height:
            return img
        ratio = target_height / img.height
        new_width = int(img.width * ratio)
        return img.resize((new_width, target_height), Image.Resampling.LANCZOS)
    
    def _resize_to_width(self, img, target_width):
        """Resize image to target width maintaining aspect ratio."""
        if img.width == target_width:
            return img
        ratio = target_width / img.width
        new_height = int(img.height * ratio)
        return img.resize((target_width, new_height), Image.Resampling.LANCZOS)
    
    def _create_title_banner(self, title, width, height, bg_color, text_color):
        """
        Create a title banner image using matplotlib for reliable font rendering.
        
        Args:
            title: str, the title text
            width: int, width of the banner in pixels
            height: int, height of the banner in pixels
            bg_color: tuple, RGB background color (0-255)
            text_color: tuple, RGB text color (0-255)
        
        Returns:
            PIL.Image: the title banner image
        """
        # Convert colors from 0-255 to 0-1 for matplotlib
        bg_color_mpl = tuple(c / 255 for c in bg_color)
        text_color_mpl = tuple(c / 255 for c in text_color)
        
        # Create figure with exact pixel dimensions
        dpi = 100
        fig_width = width / dpi
        fig_height = height / dpi
        
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi, facecolor=bg_color_mpl)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor(bg_color_mpl)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Add centered title text
        ax.text(0.5, 0.5, title, transform=ax.transAxes,
                fontsize=28, fontweight='bold', color=text_color_mpl,
                ha='center', va='center')
        
        # Convert to PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, facecolor=bg_color_mpl, 
                    edgecolor='none', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        
        # Resize to exact dimensions (bbox_inches='tight' may alter size slightly)
        if img.width != width or img.height != height:
            img = img.resize((width, height), Image.Resampling.LANCZOS)
        
        return img

    def _format_metrics_table(self, metrics_dict, columns, headers=None):
        """
        Format metrics as a markdown table.
        
        Args:
            metrics_dict: dict with metric values
            columns: list of column keys to include
            headers: optional list of header names (defaults to column keys)
        
        Returns:
            str: markdown table string
        """
        if headers is None:
            headers = columns
        
        # Build header row
        header_row = "| Metric | " + " | ".join(headers) + " |"
        separator = "|--------|" + "|".join(["-------"] * len(columns)) + "|"
        
        # Build data rows
        rows = [header_row, separator]
        
        # Define metric display names
        metric_names = {
            'mean_corr': 'Mean Corr',
            'top1_acc': 'Top-1 Accuracy',
            'avg_rank_percentile': 'Avg Rank %ile',
            'pfc_mean_r_intra': 'pFC r_intra',
            'pfc_mean_r_inter': 'pFC r_inter',
            'pfc_cohen_d': 'pFC Cohen\'s d',
            'pfc_p_value': 'pFC p-value',
            'null_mean_r_intra': 'Null r_intra',
            'null_mean_r_inter': 'Null r_inter',
            'pfc_accuracy': 'pFC Accuracy',
            'null_noise_accuracy': 'Null (noise) Accuracy',
            'null_permute_accuracy': 'Null (permute) Accuracy',
        }
        
        # Collect all unique metrics across columns
        all_metrics = set()
        for col in columns:
            if col in metrics_dict and isinstance(metrics_dict[col], dict):
                all_metrics.update(metrics_dict[col].keys())
        
        # Filter to common important metrics
        important_metrics = ['mean_corr', 'top1_acc', 'avg_rank_percentile']
        
        for metric in important_metrics:
            if any(metric in metrics_dict.get(col, {}) for col in columns if isinstance(metrics_dict.get(col), dict)):
                display_name = metric_names.get(metric, metric)
                values = []
                for col in columns:
                    if col in metrics_dict and isinstance(metrics_dict[col], dict):
                        val = metrics_dict[col].get(metric, '-')
                        if isinstance(val, float):
                            val = f"{val:.3f}"
                        values.append(str(val))
                    else:
                        values.append('-')
                rows.append(f"| {display_name} | " + " | ".join(values) + " |")
        
        return "\n".join(rows)

    def _generate_markdown_report(self, figures_dict, all_metrics, filepath, verbose=False):
        """
        Generate a Markdown report with embedded PNG figures.
        
        Args:
            figures_dict: dict mapping figure names to matplotlib figure objects
            all_metrics: dict with all computed metrics
            filepath: base path for output (without extension)
            verbose: bool, whether verbose mode is enabled
        """
        # Create plots directory
        plots_dir = f"{filepath}_plots"
        os.makedirs(plots_dir, exist_ok=True)
        plots_dirname = os.path.basename(plots_dir)
        
        # Save all figures as PNGs
        saved_figures = {}
        for name, fig in figures_dict.items():
            if fig is not None:
                fig_path = os.path.join(plots_dir, f"{name}.png")
                fig.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
                saved_figures[name] = f"{plots_dirname}/{name}.png"
                plt.close(fig)
        
        # Build markdown content
        md_lines = []
        
        # Header
        md_lines.append("# FC Prediction Evaluation Report")
        md_lines.append("")
        md_lines.append(f"**Partition:** {all_metrics.get('partition', 'N/A')} | "
                       f"**N subjects:** {all_metrics.get('n_subjects', 'N/A')} | "
                       f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        
        # ========================================================================
        # Section 1: Identifiability Heatmaps
        # ========================================================================
        md_lines.append("## Identifiability Heatmaps")
        md_lines.append("")
        md_lines.append("Pairwise correlation matrices between predicted and target connectomes. "
                       "Diagonal dominance indicates subject-specific predictions.")
        md_lines.append("")
        md_lines.append(r"$$\text{Top-1 Acc} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\arg\max_j \, r(\hat{y}_i, y_j) = i]$$")
        md_lines.append("")
        
        if verbose and 'identifiability_heatmaps' in saved_figures:
            md_lines.append("| Raw | Demeaned |")
            md_lines.append("|:---:|:---:|")
            md_lines.append(f"| ![]({saved_figures.get('identifiability_heatmaps', '')}) | "
                           f"![]({saved_figures.get('identifiability_heatmaps_demeaned', '')}) |")
        else:
            if 'identifiability_heatmaps_demeaned' in saved_figures:
                md_lines.append(f"![]({saved_figures['identifiability_heatmaps_demeaned']})")
        md_lines.append("")
        
        # Metrics table for heatmaps
        if verbose and 'heatmaps_raw' in all_metrics:
            md_lines.append("| Metric | Raw | Demeaned |")
            md_lines.append("|--------|-----|----------|")
            raw = all_metrics.get('heatmaps_raw', {})
            dem = all_metrics.get('heatmaps_demeaned', {})
            md_lines.append(f"| Mean Corr | {raw.get('mean_corr', '-'):.3f} | {dem.get('mean_corr', '-'):.3f} |")
            md_lines.append(f"| Top-1 Accuracy | {raw.get('top1_acc', '-'):.3f} | {dem.get('top1_acc', '-'):.3f} |")
            md_lines.append(f"| Avg Rank %ile | {raw.get('avg_rank_percentile', '-'):.3f} | {dem.get('avg_rank_percentile', '-'):.3f} |")
        elif 'heatmaps_demeaned' in all_metrics:
            dem = all_metrics.get('heatmaps_demeaned', {})
            md_lines.append("| Metric | Value |")
            md_lines.append("|--------|-------|")
            md_lines.append(f"| Mean Corr | {dem.get('mean_corr', '-'):.3f} |")
            md_lines.append(f"| Top-1 Accuracy | {dem.get('top1_acc', '-'):.3f} |")
            md_lines.append(f"| Avg Rank %ile | {dem.get('avg_rank_percentile', '-'):.3f} |")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        
        # ========================================================================
        # Section 2: Identifiability Violin
        # ========================================================================
        md_lines.append("## Identifiability Violin")
        md_lines.append("")
        md_lines.append("Tests whether intraindividual correlations (subject's prediction vs their own target) "
                       "exceed interindividual correlations (vs other subjects' targets).")
        md_lines.append("")
        md_lines.append(r"$$H_0: \frac{1}{N}\sum_i (r_{intra}(i) - r_{inter}(i)) = 0$$")
        md_lines.append("")
        
        if verbose and 'identifiability_violin' in saved_figures:
            md_lines.append("| Raw | Demeaned |")
            md_lines.append("|:---:|:---:|")
            md_lines.append(f"| ![]({saved_figures.get('identifiability_violin', '')}) | "
                           f"![]({saved_figures.get('identifiability_violin_demeaned', '')}) |")
        else:
            if 'identifiability_violin_demeaned' in saved_figures:
                md_lines.append(f"![]({saved_figures['identifiability_violin_demeaned']})")
        md_lines.append("")
        
        # Metrics table for violin
        if verbose and 'violin_raw' in all_metrics:
            md_lines.append("| Metric | Raw | Demeaned |")
            md_lines.append("|--------|-----|----------|")
            raw = all_metrics.get('violin_raw', {})
            dem = all_metrics.get('violin_demeaned', {})
            md_lines.append(f"| pFC r_intra | {raw.get('pfc_mean_r_intra', '-'):.3f} | {dem.get('pfc_mean_r_intra', '-'):.3f} |")
            md_lines.append(f"| pFC r_inter | {raw.get('pfc_mean_r_inter', '-'):.3f} | {dem.get('pfc_mean_r_inter', '-'):.3f} |")
            md_lines.append(f"| pFC Cohen's d | {raw.get('pfc_cohen_d', '-'):.2f} | {dem.get('pfc_cohen_d', '-'):.2f} |")
            md_lines.append(f"| pFC p-value | {raw.get('pfc_p_value', '-'):.2e} | {dem.get('pfc_p_value', '-'):.2e} |")
        elif 'violin_demeaned' in all_metrics:
            dem = all_metrics.get('violin_demeaned', {})
            md_lines.append("| Metric | Value |")
            md_lines.append("|--------|-------|")
            md_lines.append(f"| pFC r_intra | {dem.get('pfc_mean_r_intra', '-'):.3f} |")
            md_lines.append(f"| pFC r_inter | {dem.get('pfc_mean_r_inter', '-'):.3f} |")
            md_lines.append(f"| pFC Cohen's d | {dem.get('pfc_cohen_d', '-'):.2f} |")
            md_lines.append(f"| pFC p-value | {dem.get('pfc_p_value', '-'):.2e} |")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        
        # ========================================================================
        # Section 3: Hungarian Matching (verbose only)
        # ========================================================================
        if verbose:
            md_lines.append("## Hungarian Matching")
            md_lines.append("")
            md_lines.append("Optimal 1-to-1 assignment between predictions and targets that maximizes total similarity. "
                           "Black dots show the optimal assignment; accuracy measures diagonal matches.")
            md_lines.append("")
            md_lines.append(r"$$\max_{\pi \in S_N} \sum_{i=1}^{N} r(\hat{y}_i, y_{\pi(i)})$$")
            md_lines.append("")
            
            if 'hungarian_heatmaps' in saved_figures:
                md_lines.append("| Raw | Demeaned |")
                md_lines.append("|:---:|:---:|")
                md_lines.append(f"| ![]({saved_figures.get('hungarian_heatmaps', '')}) | "
                               f"![]({saved_figures.get('hungarian_heatmaps_demeaned', '')}) |")
            md_lines.append("")
            
            # Metrics table
            if 'hungarian_raw' in all_metrics:
                md_lines.append("| Condition | Raw Accuracy | Demeaned Accuracy |")
                md_lines.append("|-----------|--------------|-------------------|")
                raw = all_metrics.get('hungarian_raw', {})
                dem = all_metrics.get('hungarian_demeaned', {})
                md_lines.append(f"| pFC | {raw.get('pfc_accuracy', '-'):.3f} | {dem.get('pfc_accuracy', '-'):.3f} |")
                md_lines.append(f"| Null (noise) | {raw.get('null_noise_accuracy', '-'):.3f} | {dem.get('null_noise_accuracy', '-'):.3f} |")
                md_lines.append(f"| Null (permute) | {raw.get('null_permute_accuracy', '-'):.3f} | {dem.get('null_permute_accuracy', '-'):.3f} |")
            md_lines.append("")
            md_lines.append("---")
            md_lines.append("")
            
            # ========================================================================
            # Section 4: Hungarian Sample Size Analysis (verbose only)
            # ========================================================================
            md_lines.append("## Hungarian Sample Size Analysis")
            md_lines.append("")
            md_lines.append("Matching accuracy as a function of subset sample size. "
                           "Stars indicate sample sizes where pFC significantly exceeds both null baselines (FDR-corrected).")
            md_lines.append("")
            
            if 'hungarian_sample_size' in saved_figures:
                md_lines.append("| Raw | Demeaned |")
                md_lines.append("|:---:|:---:|")
                md_lines.append(f"| ![]({saved_figures.get('hungarian_sample_size', '')}) | "
                               f"![]({saved_figures.get('hungarian_sample_size_demeaned', '')}) |")
            md_lines.append("")
            md_lines.append("---")
            md_lines.append("")
        
        # ========================================================================
        # Section 5: PCA Structure
        # ========================================================================
        md_lines.append("## PCA Structure")
        md_lines.append("")
        md_lines.append("Correlation between predicted and target PC scores in subject-mode PCA. "
                       "High correlations indicate preserved modes of inter-subject variation.")
        md_lines.append("")
        md_lines.append(r"$$\text{PC Corr}_k = \text{corr}(C^{pred}_k, C^{target}_k)$$")
        md_lines.append("")
        
        if 'pca_line' in saved_figures:
            md_lines.append(f"![]({saved_figures['pca_line']})")
            md_lines.append("")
        
        if 'pca_spatial' in saved_figures:
            md_lines.append(f"![]({saved_figures['pca_spatial']})")
            md_lines.append("")
        
        # PCA metrics
        if 'pca' in all_metrics:
            pca = all_metrics['pca']
            md_lines.append("| Metric | Value |")
            md_lines.append("|--------|-------|")
            if 'cutoff_idx_95' in pca:
                md_lines.append(f"| PCs for 95% variance | {pca['cutoff_idx_95'] + 1} |")
            if 'pc_corrs' in pca and len(pca['pc_corrs']) > 0:
                md_lines.append(f"| PC1 Corr | {pca['pc_corrs'][0]:.3f} |")
                if len(pca['pc_corrs']) > 4:
                    md_lines.append(f"| PC5 Corr | {pca['pc_corrs'][4]:.3f} |")
        md_lines.append("")
        
        # ========================================================================
        # Write markdown file
        # ========================================================================
        md_path = f"{filepath}.md"
        with open(md_path, 'w') as f:
            f.write("\n".join(md_lines))
        
        print(f"Markdown report saved to: {md_path}")
        print(f"Figures saved to: {plots_dir}/")

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