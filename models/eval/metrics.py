"""Shared training and evaluation metrics.

Contains scalar scoring functions used by both training logs and evaluation analyses.
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import ttest_1samp
from sklearn.metrics import mean_squared_error, r2_score


def compute_pearson_r(y_pred, y_true):
    """
    Compute mean Pearson correlation across samples.
    Each sample's prediction is correlated with its target across features.
    """
    y_pred_centered = y_pred - y_pred.mean(dim=1, keepdim=True)
    y_true_centered = y_true - y_true.mean(dim=1, keepdim=True)

    numerator = (y_pred_centered * y_true_centered).sum(dim=1)
    denom_pred = torch.sqrt((y_pred_centered ** 2).sum(dim=1))
    denom_true = torch.sqrt((y_true_centered ** 2).sum(dim=1))

    r = numerator / (denom_pred * denom_true + 1e-10)
    return r.mean().item()


def compute_demeaned_pearson_r(y_pred, y_true, target_train_mean):
    """
    Compute mean Pearson correlation after subtracting the target training mean.
    """
    y_pred_demeaned = y_pred - target_train_mean
    y_true_demeaned = y_true - target_train_mean

    numerator = (y_pred_demeaned * y_true_demeaned).sum(dim=1)
    denom_pred = torch.sqrt((y_pred_demeaned ** 2).sum(dim=1))
    denom_true = torch.sqrt((y_true_demeaned ** 2).sum(dim=1))

    r = numerator / (denom_pred * denom_true + 1e-10)
    return r.mean().item()


def compute_prediction_variance_ratio(y_pred, y_true):
    """Compare mean across-feature subject variance in predictions vs targets."""
    pred_var = torch.var(y_pred, dim=0, unbiased=False).mean()
    true_var = torch.var(y_true, dim=0, unbiased=False).mean()
    return (pred_var / (true_var + 1e-10)).item()


def compute_prediction_norm_ratio(y_pred, y_true):
    """Compare average per-subject prediction norm to target norm."""
    pred_norm = torch.norm(y_pred, dim=1).mean()
    true_norm = torch.norm(y_true, dim=1).mean()
    return (pred_norm / (true_norm + 1e-10)).item()


def compute_corr_matrix(preds, targets, axis=1):
    """
    Compute correlation between all pairs of rows in preds and targets.

    Args:
        preds: tensor/array with shape (n_subjects, n_features)
        targets: tensor/array with shape (n_subjects, n_features)
        axis: 1 for row-wise features, 0 for column-wise features
    """
    if torch.is_tensor(preds):
        c_preds = preds - preds.mean(keepdims=True, axis=axis)
        c_targets = targets - targets.mean(keepdims=True, axis=axis)
        c_preds = c_preds / torch.sqrt(torch.sum(c_preds ** 2, keepdims=True, axis=axis))
        c_targets = c_targets / torch.sqrt(torch.sum(c_targets ** 2, keepdims=True, axis=axis))
        return torch.matmul(c_preds, c_targets.t())

    c_preds = preds - preds.mean(keepdims=True, axis=axis)
    c_targets = targets - targets.mean(keepdims=True, axis=axis)
    c_preds = c_preds / np.sqrt(np.sum(c_preds ** 2, keepdims=True, axis=axis))
    c_targets = c_targets / np.sqrt(np.sum(c_targets ** 2, keepdims=True, axis=axis))
    return np.matmul(c_preds, c_targets.T)


def corr_topn_accuracy(preds=None, targets=None, cc=None, topn=1):
    """Top-N self-match accuracy for a correlation matrix."""
    if cc is None:
        cc = compute_corr_matrix(preds, targets)
    if torch.is_tensor(cc):
        topidx = torch.argsort(cc, axis=1, descending=True)[:, :topn]
        selfidx = torch.atleast_2d(torch.arange(cc.shape[0], device=topidx.device)).t()
        ccmatch = torch.any(topidx == selfidx, axis=1).double()
    else:
        topidx = np.argsort(cc, axis=1)[:, -topn:]
        selfidx = np.atleast_2d(np.arange(cc.shape[0])).T
        ccmatch = np.any(topidx == selfidx, axis=1)
    return ccmatch.mean()


def corr_avg_rank(preds=None, targets=None, cc=None, sort_descending=True, return_ranklist=False):
    """
    Average self-match rank percentile for a correlation/similarity matrix.
    Perfect match is 1.0 and chance is roughly 0.5.
    """
    if cc is None:
        cc = compute_corr_matrix(preds, targets)
    if torch.is_tensor(cc):
        sidx = torch.argsort(cc, axis=1, descending=sort_descending)
        selfidx = torch.atleast_2d(torch.arange(cc.shape[0], device=sidx.device)).t()
        srank = torch.argmax((sidx == selfidx).double(), axis=1).double()
        ranklist = 1 - srank / cc.shape[0]
        avgrank = 1 - torch.mean(srank) / cc.shape[0]
    else:
        sidx = np.argsort(cc, axis=1)[:, ::-1] if sort_descending else np.argsort(cc, axis=1)
        selfidx = np.atleast_2d(np.arange(cc.shape[0])).T
        srank = np.argmax(sidx == selfidx, axis=1)
        ranklist = 1 - srank / cc.shape[0]
        avgrank = 1 - np.mean(srank) / cc.shape[0]
    if return_ranklist:
        return avgrank, ranklist
    return avgrank


def distance_top1_accuracy(dist_matrix):
    """Top-1 self-match accuracy for a distance matrix where smaller is better."""
    dist = np.asarray(dist_matrix, dtype=np.float64)
    topidx = np.argmin(dist, axis=1)
    return float(np.mean(topidx == np.arange(dist.shape[0])))


def distance_avg_rank(dist_matrix, return_ranklist=False):
    """
    Average self-match rank percentile for a distance matrix where smaller is better.
    """
    dist = np.asarray(dist_matrix, dtype=np.float64)
    sidx = np.argsort(dist, axis=1)
    selfidx = np.arange(dist.shape[0])[:, None]
    srank = np.argmax(sidx == selfidx, axis=1)
    ranklist = 1 - srank / dist.shape[0]
    avgrank = 1 - np.mean(srank) / dist.shape[0]
    if return_ranklist:
        return float(avgrank), ranklist
    return float(avgrank)


def compute_identifiability(preds, targets, return_full=False):
    """
    Compute subject identifiability statistics using a one-sample t-test.
    """
    preds_np = preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else np.asarray(preds)
    targets_np = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)

    n_subjects = preds_np.shape[0]
    r_matrix = compute_corr_matrix(targets_np, preds_np)
    r_intra = np.diag(r_matrix)

    r_inter = np.zeros(n_subjects)
    for i in range(n_subjects):
        mask = np.ones(n_subjects, dtype=bool)
        mask[i] = False
        r_inter[i] = np.mean(r_matrix[i, mask])

    d = r_intra - r_inter
    t_stat, p_value = ttest_1samp(d, 0)

    mean_d = np.mean(d)
    std_d = np.std(d, ddof=1)
    cohen_d = mean_d / std_d if std_d > 0 else np.nan

    return {
        "r_matrix": r_matrix,
        "r_intra": r_intra,
        "r_inter": r_inter,
        "d": d,
        "mean_r_intra": np.mean(r_intra),
        "mean_r_inter": np.mean(r_inter),
        "mean_d": mean_d,
        "t_stat": t_stat,
        "p_value": p_value,
        "cohen_d": cohen_d,
        "n_subjects": n_subjects,
        "df": n_subjects - 1,
    }


def compute_basic_regression_metrics(preds, targets, corr_matrix=None, corr_matrix_demeaned=None):
    """Compute scalar MSE, R2, Pearson, demeaned Pearson, rank, and top-1 metrics."""
    mse_values = mean_squared_error(targets, preds, multioutput="raw_values")
    r2_values = r2_score(targets, preds, multioutput="raw_values")
    corr_matrix = compute_corr_matrix(targets, preds) if corr_matrix is None else corr_matrix

    metrics = {
        "mse": np.mean(mse_values),
        "r2": np.mean(r2_values),
        "pearson": np.mean(np.diag(corr_matrix)),
        "avg_rank": corr_avg_rank(cc=corr_matrix),
        "top1_acc": corr_topn_accuracy(cc=corr_matrix, topn=1),
    }
    if corr_matrix_demeaned is not None:
        metrics["demeaned_pearson"] = np.mean(np.diag(corr_matrix_demeaned))
    return metrics


def hungarian_matching(similarity_matrix):
    """Perform Hungarian matching that maximizes total similarity."""
    n = similarity_matrix.shape[0]
    row_ind, col_ind = linear_sum_assignment(-similarity_matrix)

    assignment_matrix = np.zeros((n, n))
    assignment_matrix[row_ind, col_ind] = 1

    n_correct = np.sum(row_ind == col_ind)
    accuracy = n_correct / n

    return {
        "row_ind": row_ind,
        "col_ind": col_ind,
        "assignment_matrix": assignment_matrix,
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n": n,
    }


def hungarian_matching_subsample(similarity_matrix, n_sample, n_iterations=2500, seed=None):
    """Repeated subsampling Hungarian matching for a given sample size."""
    rng = np.random.default_rng(seed)
    n_subjects = similarity_matrix.shape[0]
    accuracies = np.zeros(n_iterations)

    for i in range(n_iterations):
        indices = rng.choice(n_subjects, size=n_sample, replace=False)
        sub_sim = similarity_matrix[np.ix_(indices, indices)]
        accuracies[i] = hungarian_matching(sub_sim)["accuracy"]

    return accuracies


def hungarian_matching_permuted(similarity_matrix, seed=None):
    """Hungarian matching on a column-permuted similarity matrix."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(similarity_matrix.shape[0])
    return hungarian_matching(similarity_matrix[:, perm])
