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
from data.data_utils import *


def compute_corr_matrix(preds, targets, axis=1):
    """
    Compute correlation between all pairs of rows in preds and targets (or columns if axis=0).
    Adapted from krakencoder loss.py
    
    Args:
        preds: torch tensor or numpy array (Nsubj x M), generally the predicted data for N subjects
        targets: torch tensor or numpy array (Nsubj x M), generally the measured/true data for N subjects
        axis: int (optional, default=1), 1 for row-wise, 0 for column-wise
    
    Returns:
        torch tensor or numpy array (Nsubj x Nsubj)
    
    Note: top1acc, which uses argmax(compute_corr_matrix(targets, preds), axis=1) is:
    for every TRUE output, which subject's PREDICTED output is the best match
    """
    if torch.is_tensor(preds):
        c_preds = preds - preds.mean(keepdims=True, axis=axis)
        c_targets = targets - targets.mean(keepdims=True, axis=axis)
        c_preds = c_preds / torch.sqrt(torch.sum(c_preds ** 2, keepdims=True, axis=axis))
        c_targets = c_targets / torch.sqrt(torch.sum(c_targets ** 2, keepdims=True, axis=axis))
        cc = torch.matmul(c_preds, c_targets.t())
    else:
        c_preds = preds - preds.mean(keepdims=True, axis=axis)
        c_targets = targets - targets.mean(keepdims=True, axis=axis)
        c_preds = c_preds / np.sqrt(np.sum(c_preds ** 2, keepdims=True, axis=axis))
        c_targets = c_targets / np.sqrt(np.sum(c_targets ** 2, keepdims=True, axis=axis))
        cc = np.matmul(c_preds, c_targets.T)
    return cc


def corr_topn_accuracy(preds=None, targets=None, cc=None, topn=1):
    """
    Compute top-N accuracy for compute_corr_matrix(preds, targets).
    Adapted from krakencoder loss.py

    Logical walkthrough:
    - For each subject, compute the full similarity (correlation) matrix between true and predicted outputs.
    - For each subject (row), identify the indices of the top-N most similar predicted subjects (highest correlation coefficients).
    - For each subject, check if their correct match (diagonal entry, i.e., self-match) is included among those top-N indices.
    - Collect the boolean result for all subjects, then compute the mean (proportion of correct self-matches in the top-N).

    Args:
        preds: torch tensor or numpy array (Nsubj x M), predicted data
        targets: torch tensor or numpy array (Nsubj x M), target/true data
        cc: precomputed correlation matrix (optional)
        topn: int, number of top matches to consider

    Returns:
        The proportion of subjects whose true match (diagonal element) is among their top N most similar predictions.
    """
    if cc is None:
        cc = compute_corr_matrix(preds, targets)
    # For torch tensors
    if torch.is_tensor(cc):
        topidx = torch.argsort(cc, axis=1, descending=True)[:, :topn]
        selfidx = torch.atleast_2d(torch.arange(cc.shape[0], device=topidx.device)).t()
        ccmatch = torch.any(topidx == selfidx, axis=1).double()
    # For numpy arrays
    else:
        topidx = np.argsort(cc, axis=1)[:, -topn:]
        selfidx = np.atleast_2d(np.arange(cc.shape[0])).T
        ccmatch = np.any(topidx == selfidx, axis=1)
    return ccmatch.mean()


def corr_avg_rank(preds=None, targets=None, cc=None, sort_descending=True, return_ranklist=False):
    """
    Compute average rank of each row in compute_corr_matrix(preds, targets).
    Adapted from krakencoder loss.py
    
    Perfect match is 1.0, meaning every row i in preds has the best match with row i in targets.
    Chance is roughly 0.5, meaning every row i in preds has a random match with row i in targets.
    
    Args:
        preds: torch tensor or numpy array (Nsubj x M) (ignored if cc is provided)
        targets: torch tensor or numpy array (Nsubj x M) (ignored if cc is provided)
        cc: torch tensor or numpy array (Nsubj x Nsubj), (optional precomputed cc matrix) 
        sort_descending: bool, (optional, default=True), use True for correlation, False for distance
        return_ranklist: bool, whether to return the full ranklist
    
    Returns:
        float (or FloatTensor), average rank percentile (0.0-1.0)
        optionally: ranklist if return_ranklist=True
    """
    if cc is None:
        cc = compute_corr_matrix(preds, targets)
    if torch.is_tensor(cc):
        sidx = torch.argsort(cc, axis=1, descending=sort_descending)
        selfidx = torch.atleast_2d(torch.arange(cc.shape[0], device=sidx.device)).t()
        srank = torch.argmax((sidx == selfidx).double(), axis=1).double()
        ranklist = 1 - srank / cc.shape[0]
        avgrank = 1 - torch.mean(srank) / cc.shape[0]  # percentile
    else:
        if sort_descending:
            sidx = np.argsort(cc, axis=1)[:, ::-1]
        else:
            sidx = np.argsort(cc, axis=1)
        selfidx = np.atleast_2d(np.arange(cc.shape[0])).T
        srank = np.argmax(sidx == selfidx, axis=1)
        ranklist = 1 - srank / cc.shape[0]
        avgrank = 1 - np.mean(srank) / cc.shape[0]  # percentile
    if return_ranklist:
        return avgrank, ranklist
    else:
        return avgrank


def compute_normalized_sse_pc(data, pca, pc_idx):
    """
    Compute normalized SSE of the rank-1 approximation for a given PC index,
    using a fitted sklearn PCA object and data in shape (n_features, n_samples)
    (matching subject-mode PCA: features x subjects).
    
    Args:
        data: array-like (n_features, n_samples), data matrix in subject-mode format
        pca: fitted sklearn PCA object
        pc_idx: int, index of the principal component
    
    Returns:
        float, normalized sum of squared errors
    """
    b_pc = pca.components_[pc_idx]  # shape: (n_samples,)
    # Project data onto all PCs, then select current PC scores
    # Note: in subject-mode, data is (features x subjects)
    c_pc = pca.transform(data)[..., pc_idx]  # shape: (n_features,)
    data_recon_pc = np.outer(c_pc, b_pc)  # (n_features, n_subjects)
    sse_numer = np.sum((data - data_recon_pc) ** 2)
    sse_denom = np.sum(data ** 2)
    sse_norm = sse_numer / sse_denom if sse_denom != 0 else np.nan
    return sse_norm


def compute_identifiability(preds, targets, return_full=False):
    """
    Compute identifiability statistics using one-sample t-test.
    
    Tests whether each subject's predicted FC is more similar to their own 
    empirical FC than to other subjects' empirical FCs.
    
    Args:
        preds: np.ndarray (N x M), predicted connectomes (N subjects, M edges)
        targets: np.ndarray (N x M), empirical/target connectomes
        return_full: bool, if True return additional arrays for plotting
    
    Returns:
        dict with:
            - r_matrix: (N x N) correlation matrix between preds and targets
            - r_intra: (N,) intraindividual correlations (diagonal)
            - r_inter: (N,) mean interindividual correlations per subject
            - d: (N,) per-subject differences d_i = r_intra(i) - r_inter(i)
            - mean_r_intra: mean intraindividual correlation
            - mean_r_inter: mean interindividual correlation
            - mean_d: mean difference
            - t_stat: t-statistic from one-sample t-test
            - p_value: two-sided p-value
            - cohen_d: effect size (Cohen's d)
            - n_subjects: number of subjects
            - df: degrees of freedom (N-1)
    """
    preds_np = preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else np.asarray(preds)
    targets_np = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)
    
    n_subjects = preds_np.shape[0]
    
    # Step 1: Compute similarity matrix (N x N)
    r_matrix = compute_corr_matrix(preds_np, targets_np)
    
    # Step 2: Extract intra- and inter-individual similarities
    r_intra = np.diag(r_matrix)  # r[i,i] for each subject
    
    # r_inter(i) = mean of r[i,j] for j != i
    r_inter = np.zeros(n_subjects)
    for i in range(n_subjects):
        mask = np.ones(n_subjects, dtype=bool)
        mask[i] = False
        r_inter[i] = np.mean(r_matrix[i, mask])
    
    # Per-subject difference
    d = r_intra - r_inter
    
    # Step 3: One-sample t-test on d
    # H0: mean(d) = 0
    t_stat, p_value = ttest_1samp(d, 0)
    
    # Effect size: Cohen's d for one-sample
    mean_d = np.mean(d)
    std_d = np.std(d, ddof=1)
    cohen_d = mean_d / std_d if std_d > 0 else np.nan
    
    results = {
        'r_matrix': r_matrix,
        'r_intra': r_intra,
        'r_inter': r_inter,
        'd': d,
        'mean_r_intra': np.mean(r_intra),
        'mean_r_inter': np.mean(r_inter),
        'mean_d': mean_d,
        't_stat': t_stat,
        'p_value': p_value,
        'cohen_d': cohen_d,
        'n_subjects': n_subjects,
        'df': n_subjects - 1,
    }
    
    return results


def generate_mean_baseline(train_mean, n_subjects):
    """
    Generate mean eFC baseline predictor.
    
    For each subject, the prediction is the same: the training set mean.
    Expect weak/no identifiability because all subjects share the same prediction.
    
    Args:
        train_mean: np.ndarray (M,), mean connectome from training set
        n_subjects: int, number of subjects to generate predictions for
    
    Returns:
        np.ndarray (n_subjects x M), replicated mean for each subject
    """
    train_mean_np = train_mean.detach().cpu().numpy() if isinstance(train_mean, torch.Tensor) else np.asarray(train_mean)
    return np.tile(train_mean_np, (n_subjects, 1))


def generate_noise_baseline(train_mean, train_std, n_subjects, noise_scale=1.0, seed=None):
    """
    Generate mean eFC + independent Gaussian noise baseline.
    
    For each subject, prediction = train_mean + epsilon where epsilon is 
    i.i.d. Gaussian noise per edge.
    
    Args:
        train_mean: np.ndarray (M,), mean connectome from training set
        train_std: np.ndarray (M,) or float, edgewise std from training set
        n_subjects: int, number of subjects
        noise_scale: float, multiplier for noise (sigma = noise_scale * train_std)
        seed: int, random seed for reproducibility
    
    Returns:
        np.ndarray (n_subjects x M), mean + noise predictions
    """
    train_mean_np = train_mean.detach().cpu().numpy() if isinstance(train_mean, torch.Tensor) else np.asarray(train_mean)
    train_std_np = train_std.detach().cpu().numpy() if isinstance(train_std, torch.Tensor) else np.asarray(train_std)
    
    rng = np.random.default_rng(seed)
    n_edges = train_mean_np.shape[0]
    
    # Generate independent noise for each subject and edge
    train_min = np.min(train_mean_np)
    train_max = np.max(train_mean_np)
    mean_std = np.mean(train_std_np)
    tol = mean_std

    # Add i.i.d. random Gaussian noise on a per edge basis
    noise = rng.normal(0, 1, size=(n_subjects, n_edges))
    noise = noise * (noise_scale * train_std_np)
    noised_preds = np.tile(train_mean_np, (n_subjects, 1)) + noise

    # Clip to reasonable range based on training mean and std
    clip_min = train_min - tol
    clip_max = train_max + tol
    noised_preds = np.clip(noised_preds, clip_min, clip_max)
    
    return noised_preds


def hungarian_matching(similarity_matrix):
    """
    Perform Hungarian algorithm for optimal 1-to-1 matching.
    
    Uses the similarity matrix (correlation-based) and finds the assignment
    that maximizes total similarity (by negating for the cost minimization).
    
    Args:
        similarity_matrix: np.ndarray (n x n), correlation/similarity matrix
    
    Returns:
        dict with:
            - row_ind: row indices of optimal assignment
            - col_ind: column indices of optimal assignment  
            - assignment_matrix: binary (n x n) assignment matrix
            - accuracy: proportion of correct matches (trace / n)
            - n_correct: number of correct matches (diagonal assignments)
    """
    n = similarity_matrix.shape[0]
    
    # Negate similarity to convert to cost matrix (Hungarian minimizes cost)
    cost_matrix = -similarity_matrix
    
    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create binary assignment matrix
    assignment_matrix = np.zeros((n, n))
    assignment_matrix[row_ind, col_ind] = 1
    
    # Compute accuracy (proportion on diagonal)
    n_correct = np.sum(row_ind == col_ind)
    accuracy = n_correct / n
    
    return {
        'row_ind': row_ind,
        'col_ind': col_ind,
        'assignment_matrix': assignment_matrix,
        'accuracy': accuracy,
        'n_correct': n_correct,
        'n': n
    }


def hungarian_matching_subsample(similarity_matrix, n_sample, n_iterations=2500, seed=None):
    """
    Repeated subsampling Hungarian matching for a given sample size.
    
    Args:
        similarity_matrix: np.ndarray (N x N), full similarity matrix
        n_sample: int, subset size to sample
        n_iterations: int, number of random subsamples (M)
        seed: int, random seed
    
    Returns:
        np.ndarray of shape (n_iterations,), accuracy for each iteration
    """
    rng = np.random.default_rng(seed)
    N = similarity_matrix.shape[0]
    accuracies = np.zeros(n_iterations)
    
    for m in range(n_iterations):
        # Sample without replacement
        indices = rng.choice(N, size=n_sample, replace=False)
        # Extract submatrix
        sub_sim = similarity_matrix[np.ix_(indices, indices)]
        # Run Hungarian
        result = hungarian_matching(sub_sim)
        accuracies[m] = result['accuracy']
    
    return accuracies


def hungarian_matching_permuted(similarity_matrix, seed=None):
    """
    Hungarian matching on a permuted similarity matrix (chance baseline).
    
    Randomly permutes columns of the similarity matrix before applying
    Hungarian algorithm to establish chance-level matching.
    
    Args:
        similarity_matrix: np.ndarray (n x n)
        seed: int, random seed
    
    Returns:
        dict with Hungarian matching results on permuted matrix
    """
    rng = np.random.default_rng(seed)
    n = similarity_matrix.shape[0]
    
    # Randomly permute columns
    perm = rng.permutation(n)
    permuted_sim = similarity_matrix[:, perm]
    
    return hungarian_matching(permuted_sim)


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
        # Compute correlation matrices
        self.corr_matrix = compute_corr_matrix(self.preds, self.targets)
        self.corr_matrix_demeaned = compute_corr_matrix(
            self.preds - self.train_mean, 
            self.targets - self.train_mean
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

    def evaluate_pca_structure(self, num_pcs=5, show_first_pcs=5, diagval=0, show_sse=True):
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
        
        Returns:
            tuple: (pca_preds, pca_targets) fitted PCA objects
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
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(1, num_pcs+1), pc_corrs, marker='o', color='b', label='PC score corr (targets, preds)')
        plt.xlabel("Principal Component")
        plt.ylabel("Correlation between scores")
        plt.title(f"Correlation between Predicted & Target PC Scores (First {num_pcs} PCs)")

        # Find index where cumulative variance explained crosses 0.95
        cum_var = np.cumsum(explained_var_targets)
        cutoff_idx = np.argmax(cum_var >= 0.95)
        if cutoff_idx < num_pcs:
            plt.axvline(cutoff_idx+1, color='r', linestyle='--', linewidth=1, alpha=0.8,
                        label='95% target var explained')
        plt.ylim(-1, 1)
        plt.xlim(0.5, num_pcs + 0.5)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Now PC score spatial maps for show_first_pcs
        fig, axs = plt.subplots(show_first_pcs, 3, figsize=(8, show_first_pcs * 2.3), dpi=200)
        if show_first_pcs == 1:
            axs = axs[None, :]  # Always 2D
        
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
            axs[i, 0].set_title(f'Target PC{i+1}\n(Scores)', fontsize=10)
            axs[i, 0].set_xticks([])
            axs[i, 0].set_yticks([])
            plt.colorbar(im0, ax=axs[i, 0], fraction=0.045, pad=0.03)

            # Predicted PC
            im1 = axs[i, 1].imshow(C_preds_sq, cmap='RdBu_r', vmin=vmin_preds, vmax=vmax_preds, aspect='equal')
            axs[i, 1].set_title(f'Predicted PC{i+1}\n(Scores)', fontsize=10)
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

            axs[i, 2].axis('off')
            axs[i, 2].text(0.05, 0.5, label, va='center', ha='left', fontsize=11)

        axs[0, 2].figure.text(0.79, 0.97, "Metrics", fontsize=12, ha='center', va='center')
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.show()

        return pca_preds, pca_targets

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
                                       dpi=200, figsize=(14, 10)):
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
        
        Returns:
            dict: metrics including mean_corr, top1_acc, avg_rank_percentile
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
        
        # Compute correlation matrix for ordered data (non-demeaned for reference)
        corr_matrix = compute_corr_matrix(preds_data, targets_data)
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
            
            ax.set_title(titlestr, fontsize=14)
            ax.set_xlabel(label2, fontsize=12)
            ax.set_ylabel(label1, fontsize=12)
            
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=10)
        
        # Compute summary metrics
        n_subjects = len(observed_vs_pred_ranklist)
        n_top1 = np.sum(observed_vs_pred_ranklist == 1)
        top1_acc = n_top1 / n_subjects
        avgrank_percentile = 1 - (np.mean(observed_vs_pred_ranklist) / n_subjects)
        
        # Print summary
        print(f"Top-1 accuracy: {top1_acc:.3f}  ({n_top1} of {n_subjects} subjects had rank 1)")
        print(f"Average rank percentile: {avgrank_percentile:.3f}")
        
        # Fourth panel: leave blank or use for additional info
        ax = axs[3]
        ax.axis('off')
        
        # Add summary text in fourth panel
        summary_text = (
            f"Summary Metrics\n"
            f"{'─' * 25}\n"
            f"Mean corr (target vs pred): {mean_corr:.3f}\n"
            f"Demeaned: {demeaned}\n"
            f"Top-1 accuracy: {top1_acc:.3f}\n"
            f"Avg rank percentile: {avgrank_percentile:.3f}\n"
            f"{'─' * 25}\n"
            f"Order: {order_by}\n"
            f"N subjects: {n_subjects}"
        )
        ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=14,
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor='gray', alpha=0.8),
                family='monospace')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'mean_corr': mean_corr,
            'demeaned': demeaned,
            'top1_acc': top1_acc,
            'avg_rank_percentile': avgrank_percentile,
            'ranklist': observed_vs_pred_ranklist
        }

    def plot_single_corr_heatmap(self, comparison='targets_vs_preds', order_by='original',
                                  include_black_circles=True, include_blue_dots=True,
                                  demeaned=False, dpi=200, figsize=(8, 8)):
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
        
        Returns:
            dict: metrics for this comparison
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
        
        ax.set_title(titlestr, fontsize=16)
        ax.set_xlabel(label2, fontsize=14)
        ax.set_ylabel(label1, fontsize=14)
        
        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        plt.show()
        
        result = {
            'mean_corr': mean_corr,
            'avg_rank': avgrank,
        }
        if ranklist is not None:
            n_subjects = len(ranklist)
            result['top1_acc'] = np.sum(ranklist == 1) / n_subjects
            result['avg_rank_percentile'] = 1 - (np.mean(ranklist) / n_subjects)
            result['ranklist'] = ranklist
        
        return result

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
                                     p_threshold=0.001, dpi=150, figsize=(8, 6)):
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
        
        Returns:
            dict with identifiability results for each condition
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
            ax.text(pos, y_max + 0.01, sig_text, ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Formatting
        ax.set_xticks(positions)
        ax.set_xticklabels(conditions, fontsize=14)
        ax.set_ylabel('Correlation', fontsize=14)
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
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
        
        # Title
        demean_str = " (demeaned)" if demeaned else ""
        ax.set_title(f'FC Prediction Identifiability{demean_str}', fontsize=16)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set y-axis limit to extend to 1
        ax.set_ylim(top=1.0)
        
        plt.tight_layout()
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
        
        return results

    def plot_hungarian_heatmaps(self, include_noise_baseline=True, include_permute_baseline=True,
                                 noise_scale=1.0, demeaned=False, seed=42,
                                 dpi=150, figsize=(15, 5)):
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
        
        Returns:
            dict with Hungarian matching results for each condition
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
        
        # 1. pFC (model predictions)
        sim_pfc = compute_corr_matrix(preds_data, targets_data)
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
            sim_noise = compute_corr_matrix(noise_preds, targets_data)
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
            demean_str = " (demeaned)" if demeaned else ""
            ax.set_title(f'{cond}{demean_str}\nTop-1 Acc: {acc:.1f}%', fontsize=12)
            ax.set_xlabel('Predicted', fontsize=11)
            ax.set_ylabel('Target', fontsize=11)
            
            # Colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=9)
            cbar.set_label('Corr', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print("\nHungarian Matching Summary:")
        print("=" * 50)
        for cond, hung in results.items():
            print(f"{cond:<20} Top-1 Acc: {hung['accuracy']*100:.2f}% ({hung['n_correct']}/{hung['n']})")
        print("=" * 50)
        
        # return results

    def plot_hungarian_sample_size_analysis(self, n_min=2, n_max=20, step=1, n_iterations=2500,
                                             noise_scale=1.0, demeaned=False, 
                                             fdr_alpha=0.05, seed=42,
                                             dpi=150, figsize=(10, 6)):
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
        
        Returns:
            dict with results for each condition and sample size
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

        # Compute similarity matrices
        sim_pfc = compute_corr_matrix(preds_data, targets_data)

        # Noised baseline
        noise_preds = generate_noise_baseline(self.train_mean, self.train_std, n_subjects,
                                               noise_scale=noise_scale, seed=seed)
        if demeaned:
            noise_preds = noise_preds - self.train_mean
        sim_noise = compute_corr_matrix(noise_preds, targets_data)

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

        ax.set_xlabel('Number of Individuals', fontsize=14)
        ax.set_ylabel('% Individuals Correctly Matched', fontsize=14)
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
                ax.text(n, y_min+0.05, '*', ha='center', va='top', fontsize=12, 
                        fontweight='bold', transform=ax.get_xaxis_transform())

        # Build legend, clarifying the meaning of *
        from matplotlib.lines import Line2D
        handles, labels = ax.get_legend_handles_labels()
        asterisk_patch = Line2D([0], [0], color='none', marker='*', markersize=12, 
                                markerfacecolor='k', label="Significant (p < {:.3g}) vs both nulls".format(fdr_alpha))
        handles.append(asterisk_patch)
        ax.legend(handles=handles, fontsize=12, loc='upper right')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        demean_str = " (demeaned)" if demeaned else ""
        ax.set_title(f'Hungarian Matching Accuracy vs Subsetted Sample Size{demean_str}', fontsize=14)

        plt.tight_layout()
        plt.show()

        # Print summary
        print(f"\nFDR-corrected significance (alpha={fdr_alpha}):")
        print(f"  pFC vs Null (permute): {np.sum(results['significant_permute'])}/{n_sizes} sample sizes significant")
        print(f"  pFC vs Null (noise): {np.sum(results['significant_noise'])}/{n_sizes} sample sizes significant")
        # return results

    def get_pca_objects(self):
        """
        Get the fitted PCA objects for predictions and targets.
        
        Returns:
            tuple: (pca_preds, pca_targets) or (None, None) if not yet fitted
        """
        if self._pca_targets is None or self._pca_preds is None:
            self._fit_pca()
        return self._pca_preds, self._pca_targets
    
    def __repr__(self):
        return (
            f"Evaluator(n_subjects={self.preds.shape[0]}, "
            f"n_features={self.preds.shape[1]}, "
            f"mse={self._metrics['mse']:.4f}, "
            f"r2={self._metrics['r2']:.4f})"
        )