import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from data.data_utils import *

def predict_from_loader(model, data_loader, device=None):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in data_loader:
            x = batch["x"]
            y = batch["y"]
            if device is not None:
                x = x.to(device)
                y = y.to(device)
            preds = model(x)
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return all_preds, all_targets

def xycorr(x,y,axis=1):
    """ from krakencoder loss.py
    Compute correlation between all pairs of rows in x and y (or columns if axis=0)
    
    x: torch tensor or numpy array (Nsubj x M), generally the measured data for N subjects
    y: torch tensor or numpy array (Nsubj x M), generally the predicted data for N subjects
    axis: int (optional, default=1), 1 for row-wise, 0 for column-wise
    
    Returns: torch tensor or numpy array (Nsubj x Nsubj)
    
    top1acc, which uses argmax(xycorr(true,predicted),axis=1) is:
    for every TRUE output, which subject's PREDICTED output is the best match
    """
    if torch.is_tensor(x):
        cx=x-x.mean(keepdims=True,axis=axis)
        cy=y-y.mean(keepdims=True,axis=axis)
        cx=cx/torch.sqrt(torch.sum(cx ** 2,keepdims=True,axis=axis))
        cy=cy/torch.sqrt(torch.sum(cy ** 2,keepdims=True,axis=axis))
        cc=torch.matmul(cx,cy.t())
    else:
        cx=x-x.mean(keepdims=True,axis=axis)
        cy=y-y.mean(keepdims=True,axis=axis)
        cx=cx/np.sqrt(np.sum(cx ** 2,keepdims=True,axis=axis))
        cy=cy/np.sqrt(np.sum(cy ** 2,keepdims=True,axis=axis))
        cc=np.matmul(cx,cy.T)
    return cc

def corrtopNacc(x=None, y=None, cc=None, topn=1):
    """ from krakencoder loss.py
    Compute top-N accuracy for xycorr(x, y). See corrtop1acc.

    Logical walkthrough:
    - For each subject, compute the full similarity (correlation) matrix between true and predicted outputs.
    - For each subject (row), identify the indices of the top-N most similar predicted subjects (highest correlation coefficients).
    - For each subject, check if their correct match (diagonal entry, i.e., self-match) is included among those top-N indices.
    - Collect the boolean result for all subjects, then compute the mean (proportion of correct self-matches in the top-N).

    Returns:
        The proportion of subjects whose true match (diagonal element) is among their top N most similar predictions.
    """
    if cc is None:
        cc = xycorr(x, y)
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

def corravgrank(x=None, y=None ,cc=None, sort_descending=True, return_ranklist=False):
    """ from krakencoder loss.py
    Compute average rank of each row in xycorr(x,y).
    Perfect match is 1.0, meaning every row i in x has the best match with row i in y
    Chance is roughly 0.5, meaning every row i in x has a random match with row i in y
    
    Inputs: either x and y must be provided, or cc must be provided
    x: torch tensor or numpy array (Nsubj x M) (ignored if cc is provided)
    y: torch tensor or numpy array (Nsubj x M) (ignored if cc is provided)
    cc: torch tensor or numpy array (Nsubj x Nsubj), (optional precomputed cc matrix) 
    sort_descending: bool, (optional, default=True), use True for correlation, False for distance
    
    Returns: float (or FloatTensor), average rank percentile (0.0-1.0)
    """
    if cc is None:
        cc=xycorr(x,y)
    if torch.is_tensor(cc):
        sidx=torch.argsort(cc,axis=1,descending=sort_descending)
        selfidx=torch.atleast_2d(torch.arange(cc.shape[0],device=sidx.device)).t()
        srank=torch.argmax((sidx==selfidx).double(),axis=1).double()
        ranklist=1-srank/cc.shape[0]
        avgrank=1-torch.mean(srank)/cc.shape[0] #percentile
    else:
        if sort_descending:
            sidx=np.argsort(cc,axis=1)[:,::-1]
        else:
            sidx=np.argsort(cc,axis=1)
        selfidx=np.atleast_2d(np.arange(cc.shape[0])).T
        srank=np.argmax(sidx==selfidx,axis=1)
        ranklist=1-srank/cc.shape[0]
        avgrank=1-np.mean(srank)/cc.shape[0] #percentile
    if return_ranklist:
        return avgrank,ranklist
    else:
        return avgrank

def evaluate_predictions(preds, targets, train_mean):
    preds_np = preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else np.asarray(preds)
    targets_np = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)
    assert preds_np.shape == targets_np.shape

    mse_values = mean_squared_error(targets_np, preds_np, multioutput='raw_values')
    r2_values = r2_score(targets_np, preds_np, multioutput='raw_values')
    xycorr_preds_targets = xycorr(preds_np, targets_np)
    xycorr_preds_targets_demeaned = xycorr(preds_np - train_mean, targets_np - train_mean)
    
    metrics = {}
    metrics['pearson'] = np.mean(np.diag(xycorr_preds_targets))
    metrics['demeaned_pearson'] = np.mean(np.diag(xycorr_preds_targets_demeaned))
    metrics['mse'] = np.mean(mse_values)
    metrics['r2'] = np.mean(r2_values)
    metrics['avg_rank'] = corravgrank(cc=xycorr_preds_targets)
    metrics['top1_acc'] = corrtopNacc(cc=xycorr_preds_targets, topn=1)

    return metrics

def evaluate_pca_structure(preds, targets, train_mean):
    """
    Evaluate PCA structure in prediction and target matrices, in analogy to krakencoder_example_toy.ipynb

    Args:
        preds: array-like (n, d), predicted features (subjects x features)
        targets: array-like (n, d), target/true features
        train_mean: array-like (d,), mean to use for centering
        n_components: int, how many PCs to use
        return_full_pca: bool, if True, also return PCA objects and extra details

    Returns:
        results: dict with target and pred PCA results
    """
    preds_np = preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else np.asarray(preds)
    targets_np = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)
    train_mean_np = train_mean.detach().cpu().numpy() if isinstance(train_mean, torch.Tensor) else np.asarray(train_mean)

    # mean center
    X = targets_np - train_mean_np  # (n, d)
    X_hat = preds_np - train_mean_np  # (n, d)

    # transpose to features x subjects (d, n)
    Y = X.T
    Y_hat = X_hat.T

    # --- PCA for targets ---
    pca_targets = PCA()
    # When fit_transform is applied to (d, n), we get "component weights for each subject" (n_samples = d, n_features = n)
    # But we want to analyze components across subjects (columns are subjects)
    # So subject-mode PCA: d by n (features x subjects), so columns = subjects; each PC is an axis across subjects
    # This will give us B: (k x n), C: (d x k)
    C_targets = pca_targets.fit_transform(Y)      # shape (d, k)
    B_targets = pca_targets.components_           # shape (k, n)
    explained_var_targets = pca_targets.explained_variance_ratio_
    cum_var_targets = np.cumsum(explained_var_targets)

    # -- PCA for predictions --
    pca_preds = PCA()
    C_preds = pca_preds.fit_transform(Y_hat)      # (d, k)
    B_preds = pca_preds.components_               # (k, n)
    explained_var_preds = pca_preds.explained_variance_ratio_
    cum_var_preds = np.cumsum(explained_var_preds)

    # Orthonormality check for bases (across subjects)
    assert np.allclose(B_targets @ B_targets.T, np.eye(B_targets.shape[0]), atol=1e-6), "B_targets is not orthonormal"
    assert np.allclose(B_preds @ B_preds.T, np.eye(B_preds.shape[0]), atol=1e-6), "B_preds is not orthonormal"

    return pca_preds, pca_targets

def compute_normalized_sse_pc(Xobs, pca, pc_idx):
    """
    Compute normalized SSE of the rank-1 approximation for a given PC index,
    using a fitted sklearn PCA object and data in shape (n_features, n_samples)
    (matching subject-mode PCA: features x subjects).
    """
    b_pc = pca.components_[pc_idx]  # shape: (n_samples,)
    # Project Xobs onto all PCs, then select current PC scores
    # Note: in subject-mode, Xobs is (features x subjects)
    c_pc = pca.transform(Xobs)[..., pc_idx]  # shape: (n_features,)
    Xobs_recon_pc = np.outer(c_pc, b_pc)  # (n_features, n_subjects)
    sse_numer = np.sum((Xobs - Xobs_recon_pc) ** 2)
    sse_denom = np.sum(Xobs ** 2)
    sse_norm = sse_numer / sse_denom if sse_denom != 0 else np.nan
    return sse_norm

def plot_pc_modes_comparison(pca_preds, pca_targets, preds, targets, numroi=360, num_pcs=5, diagval=0, show_sse=True):
    """
    Show the top num_pc PC component maps of targets (true) and preds (predicted) side-by-side as square matrices,
    along with correlation, variance explained, and SSE summary for each, with clear labeling.
    
    Parameters
    ----------
    pca_preds   : fitted PCA object (for predictions), e.g., output of evaluate_pca_structure
    pca_targets : fitted PCA object (for targets), e.g., output of evaluate_pca_structure
    preds       : array-like (n, d), predicted features (subjects x features)
    targets     : array-like (n, d), target/true features
    numroi      : int, Number of ROIs (default 360), for tri2square reshaping
    num_pc      : int, Number of principal components to plot (default 6)
    diagval     : float, Value to set on diagonal of square matrix (default 0)
    show_sse    : bool, Whether to display normalized SSE (default True; needs pca_targets and Xobs)
    """
    preds_np = preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else np.asarray(preds)
    targets_np = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)
    explained_var_targets = pca_targets.explained_variance_ratio_
    
    # For plotting, use PCA components_ (PCs as vectors across subjects)
    # sklearn PCA.components_: (n_components, n_features), so with subject-mode (features x subjects),
    # components_: (n_components, n_subjects)
    # Components_ is (n_components, n_features); here scores are (d, k), so we want the PC scores for k
    # Visualize principal component "scores" (i.e., spatial weights) for each PC,
    # for both targets and preds. 
    # Here, pca.transform gives the subject-mode PC scores: shape (n_features, n_components)
    # We extract columns by PC (i.e. across edges/ROIs for a given PC).
    C_true_scores = pca_targets.transform(targets_np.T)   # (n_features, n_components)
    C_pred_scores = pca_preds.transform(preds_np.T)       # (n_features, n_components)

    fig, axs = plt.subplots(num_pcs, 3, figsize=(10, num_pcs * 3))
    if num_pcs == 1:
        axs = axs[None, :]  # Always 2D even for 1 PC

    for i in range(num_pcs):
        # For each principal component, visualize its "spatial map" score across all features/edges
        C_true_pc = C_true_scores[:, i]   # Take the i-th PC's scores (length = n_features)
        C_pred_pc = C_pred_scores[:, i]

        # Project each PC's scores back to the original square (ROI x ROI) format
        C_true_sq = tri2square(C_true_pc, numroi=numroi, diagval=diagval)
        C_pred_sq = tri2square(C_pred_pc, numroi=numroi, diagval=diagval)

        # Use separate colorbars/ranges for each map, as was done originally
        vmax_true = np.max(np.abs(C_true_sq))
        vmin_true = -vmax_true
        vmax_pred = np.max(np.abs(C_pred_sq))
        vmin_pred = -vmax_pred

        # Plot True PC
        im0 = axs[i, 0].imshow(C_true_sq, cmap='RdBu_r', vmin=vmin_true, vmax=vmax_true, aspect='equal')
        axs[i, 0].set_title(f'True PC{i+1}\n(Scores)', fontsize=11)
        axs[i, 0].set_xticks([])
        axs[i, 0].set_yticks([])
        plt.colorbar(im0, ax=axs[i, 0], fraction=0.046, pad=0.04)

        # Plot Predicted PC
        im1 = axs[i, 1].imshow(C_pred_sq, cmap='RdBu_r', vmin=vmin_pred, vmax=vmax_pred, aspect='equal')
        axs[i, 1].set_title(f'Predicted PC{i+1}\n(Scores)', fontsize=11)
        axs[i, 1].set_xticks([])
        axs[i, 1].set_yticks([])
        plt.colorbar(im1, ax=axs[i, 1], fraction=0.046, pad=0.04)

        # Quantitative metrics & summary (as in original)
        pc_corr = np.corrcoef(C_true_pc, C_pred_pc)[0, 1] if np.std(C_true_pc) > 0 and np.std(C_pred_pc) > 0 else float('nan')
        var_exp = explained_var_targets[i] * 100

        label = f"PC{i+1}\nVariance explained (true): {var_exp:.1f}%\n"
        label += f"Corr(true, pred): r = {pc_corr:.3f}"
        if show_sse:
            sse_norm_true = compute_normalized_sse_pc(targets_np.T, pca_targets, i)
            sse_norm_pred = compute_normalized_sse_pc(preds_np.T, pca_preds, i)
            label += f"\nNormalized SSE (true): {sse_norm_true:.4f}"
            label += f"\nNormalized SSE (pred): {sse_norm_pred:.4f}"
        axs[i, 2].axis('off')
        axs[i, 2].text(0.1, 0.5, label, va='center', ha='left', fontsize=12)

    # Super-labels for columns remain as in the original
    axs[0, 2].figure.text(0.80, 0.94, "Metrics", fontsize=13, ha='center', va='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()