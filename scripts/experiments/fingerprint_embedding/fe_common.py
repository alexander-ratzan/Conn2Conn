"""Spec v4 E6 (fingerprint_embedding): shared helpers.

The embedding is E4's pooled map (`pooled_map.md` in cross_condition_translation): one least-squares map B fit on the
train subjects over all 56 ordered FC condition pairs in E2's pooled block-PCA codes (k = 256; the penalty is irrelevant,
E4). It is computed by the library's closed form (models/architectures/pooled_map.py: one generalized
eigendecomposition, no regression); `pooled_map` here also solves the stacked least squares and records the difference.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
E4 = REPO_ROOT / "scripts" / "experiments" / "cross_condition_translation"
sys.path[:0] = [str(REPO_ROOT), str(E4)]
from e4_common import FC, SeedData, score  # noqa: E402,F401  (E2 cache and scoring, via E4)
from models.architectures import pooled_map as pm  # noqa: E402
from models.eval.metrics import compute_corr_matrix, corr_avg_rank  # noqa: E402

K = 256
SEEDS = range(5)
CACHE = REPO_ROOT / "results" / "cache" / "experiments" / "fingerprint_embedding"
CONFOUNDS = CACHE / "confounds_Glasser.csv"
COG = ["CogTotalComp_AgeAdj", "CogFluidComp_AgeAdj", "CogCrystalComp_AgeAdj", "PMAT24_A_CR"]
UNRESTRICTED = Path("/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200/HCP1200_UNRESTRICTED.csv")


def codes(sd, split, k=K):
    return {v: sd.code(v, split, k).astype(np.float64) for v in FC}


def pooled_map(sd, k=K):
    """The pooled map from the train codes (library closed form), its fingerprint shares w (descending) and directions V;
    checked against E4's stacked least squares (B1-pooled at penalty 0)."""
    Z = codes(sd, "train", k)
    B, w, V = pm.pooled_map([Z[v] for v in FC])
    G = (len(FC) - 1) * sum(Z[v].T @ Z[v] for v in FC)
    C = sum(Z[s].T @ sum(Z[t] for t in FC if t != s) for s in FC)
    B_ls = np.linalg.solve(G, C)
    return B, dict(closed_form_max_abs_diff=float(np.abs(B - B_ls).max()), w=w, V=V)


def fingerprint(Z):
    """Subject means over the 8 conditions (codes)."""
    return pm.fingerprint([Z[v] for v in FC])


def ident(X, Y):
    """Rows of Y (targets) matched against rows of X (predictions): average rank and top-1 (E2's convention)."""
    cc = compute_corr_matrix(np.asarray(Y, dtype=np.float64), np.asarray(X, dtype=np.float64))
    return float(corr_avg_rank(cc=cc)), float((np.asarray(cc).argmax(1) == np.arange(len(cc))).mean())


def ridge_fit(X, Y, rel_alphas):
    """Ridge coefficients for every relative penalty (x mean eigenvalue of X^T X); dual form when p > n."""
    n, p = X.shape
    out = {}
    if p > n:
        Kx = X @ X.T
        w, U = np.linalg.eigh(Kx)
        UtY = U.T @ Y
        for a in rel_alphas:                       # eigenvalues of X X^T = squared singular values of X
            out[a] = X.T @ (U @ (UtY / (w + a * w.mean())[:, None]))
    else:
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        UtY = U.T @ Y
        for a in rel_alphas:
            out[a] = Vt.T @ ((s / (s ** 2 + a * np.mean(s ** 2)))[:, None] * UtY)
    return out
