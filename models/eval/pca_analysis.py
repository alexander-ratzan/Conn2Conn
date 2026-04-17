"""PCA-specific evaluation helpers.

Contains PCA computations used by evaluator analyses and plots.
"""

import numpy as np


def compute_normalized_sse_pc(data, pca, pc_idx):
    """
    Compute normalized SSE of the rank-1 approximation for one principal component.
    """
    b_pc = pca.components_[pc_idx]
    c_pc = pca.transform(data)[..., pc_idx]
    data_recon_pc = np.outer(c_pc, b_pc)
    sse_numer = np.sum((data - data_recon_pc) ** 2)
    sse_denom = np.sum(data ** 2)
    return sse_numer / sse_denom if sse_denom != 0 else np.nan
