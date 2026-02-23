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
    
    Exact translation of Sarwar et al. MATLAB code:
        r_matrix = corr(efc', preds')  % N x N correlation matrix
        r_intra = diag(r_matrix)       % intraindividual: corr(targets[i], preds[i])
        r_inter(i) = mean(r_matrix(i, [1:i-1, i+1:N]))  % interindividual
    
    Tests whether each subject's target (empirical FC) is more similar to their own 
    prediction than to other subjects' predictions.
    
    Args:
        preds: np.ndarray (N x M), predicted connectomes (N subjects, M edges)
        targets: np.ndarray (N x M), empirical/target connectomes
        return_full: bool, if True return additional arrays for plotting
    
    Returns:
        dict with:
            - r_matrix: (N x N) correlation matrix, r_matrix[i,j] = corr(targets[i], preds[j])
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
    # IMPORTANT: Order is (targets, preds) to match paper's corr(efc', preds')
    # r_matrix[i,j] = corr(targets[i], preds[j])
    r_matrix = compute_corr_matrix(targets_np, preds_np)
    
    # Step 2: Extract intra- and inter-individual similarities
    # r_intra[i] = r_matrix[i,i] = corr(targets[i], preds[i])
    r_intra = np.diag(r_matrix)
    
    # r_inter[i] = mean of r_matrix[i,j] for j != i
    # "How well does subject i's target correlate with OTHER subjects' predictions?"
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


def generate_noise_baseline(train_mean, train_std, n_subjects, seed=None):
    """
    Generate mean eFC + independent Gaussian noise baseline.
    
    Exact translation of Sarwar et al. MATLAB code:
        null = repmat(std(efc), N, 1) .* randn(N, J) + mean_efc
    
    For each subject i, each edge j:
        null[i,j] = train_mean[j] + train_std[j] * randn()
    
    Args:
        train_mean: np.ndarray (M,), mean connectome from training set
        train_std: np.ndarray (M,), edgewise std across subjects from training set
        n_subjects: int, number of subjects (N)
        seed: int, random seed for reproducibility
    
    Returns:
        np.ndarray (n_subjects x M), null predictions with noise
    """
    train_mean_np = train_mean.detach().cpu().numpy() if isinstance(train_mean, torch.Tensor) else np.asarray(train_mean)
    train_std_np = train_std.detach().cpu().numpy() if isinstance(train_std, torch.Tensor) else np.asarray(train_std)
    
    rng = np.random.default_rng(seed)
    n_edges = train_mean_np.shape[0]
    
    # Exact translation: std(efc) .* randn(N, J) + mean_efc
    # randn generates standard normal (mean=0, std=1)
    # Then scale by train_std per edge and add mean
    noise = rng.standard_normal(size=(n_subjects, n_edges)) * train_std_np
    null_preds = train_mean_np + noise
    
    return null_preds


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