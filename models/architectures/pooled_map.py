"""Pooled cross-condition map (fingerprint filter), closed form (spec v4 E4 / E6; derivation in
scripts/experiments/cross_condition_translation/pooled_map.md, section 5).

For M conditions of the same subjects in one code space (each condition centred on its train mean), the least-squares
map shared by all M (M - 1) ordered condition pairs is B = (M / (M - 1)) W - I / (M - 1), W = (M S_f + S_d)^-1 M S_f,
with S_f the scatter of the subjects' mean codes (fingerprints) and S_d the scatter of the conditions around them. It
is computed without a regression from the generalized eigenproblem M S_f v = w (M S_f + S_d) v: w_j in [0, 1] is the
fingerprint share of direction v_j (Fisher LDA with subjects as classes: w = lambda / (lambda + 1)), and
B = V diag((M w - 1) / (M - 1)) V^-1 with V^-1 = V^T (M S_f + S_d). Row-vector convention: embed z as z @ B.
"""
import numpy as np
from scipy.linalg import eigh


def fingerprint(Z):
    """Z: sequence of (n, k) code matrices, one per condition, rows = the same subjects. Mean over conditions."""
    return sum(Z) / len(Z)


def pooled_map(Z):
    """Z: sequence of M (n, k) train code matrices (same subjects, same order). Returns B (k, k), the fingerprint shares
    w (k,) sorted descending, and the directions V (k, k), columns matching w (normalised so V^T (M S_f + S_d) V = I)."""
    M = len(Z)
    f = fingerprint(Z)
    Sf = f.T @ f
    Sd = sum((z - f).T @ (z - f) for z in Z)
    C = M * Sf + Sd
    w, V = eigh(M * Sf, C)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    B = V @ np.diag((M * w - 1) / (M - 1)) @ V.T @ C
    return B, w, V
