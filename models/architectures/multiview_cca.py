"""Closed-form multiview CCA (MAXVAR) and ridge readout on code matrices (spec v4 I2; used by E2 L4 and E1).

MAXVAR: shared scores G (n, d), G^T G = I, and per-view regularized encoders A_m minimizing
sum_m ||G - Z_m A_m||^2 + lambda_m ||A_m||^2. G is the top-d eigenvectors of sum_m Z_m (Z_m^T Z_m + lambda_m I)^-1 Z_m^T,
A_m = (Z_m^T Z_m + lambda_m I)^-1 Z_m^T G, and the decoders D_m = G^T Z_m map scores back to view codes. lambda_m is
lam_rel times the mean eigenvalue of Z_m^T Z_m. Inputs are train-centered codes.
"""
import numpy as np


def maxvar(Z, d, lam_rel):
    """Z: {view: (n, k)} train codes. Returns encoders {view: (k, d)}, decoders {view: (d, k)}, G (n, d)."""
    inv = {}
    M = 0
    for v, X in Z.items():
        C = X.T @ X
        lam = lam_rel * np.trace(C) / C.shape[0]
        inv[v] = np.linalg.inv(C + lam * np.eye(C.shape[0]))
        M = M + X @ inv[v] @ X.T
    w, V = np.linalg.eigh(M)
    G = V[:, np.argsort(w)[::-1][:d]]
    A = {v: inv[v] @ Z[v].T @ G for v in Z}
    D = {v: G.T @ Z[v] for v in Z}
    return A, D, G


def ridge_gcv(X, Y, rel_alphas):
    """Multi-output ridge without intercept (inputs centered); alpha = rel * mean eigenvalue of X^T X, chosen by
    generalized cross-validation. Returns (B, chosen rel alpha)."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    UtY = U.T @ Y
    n = X.shape[0]
    best = None
    for a_rel in rel_alphas:
        a = a_rel * np.mean(s ** 2)
        f = s ** 2 / (s ** 2 + a)
        gcv = np.square(Y - U @ (f[:, None] * UtY)).sum() / (n - f.sum()) ** 2
        if best is None or gcv < best[0]:
            best = (gcv, a_rel, a)
    _, a_rel, a = best
    return Vt.T @ ((s / (s ** 2 + a))[:, None] * UtY), a_rel
