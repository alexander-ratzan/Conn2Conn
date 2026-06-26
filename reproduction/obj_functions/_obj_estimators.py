"""Objective-function estimators (Phase 1: 1A + 2C), BR backbone, in PCA-latent space.

Each estimator maps source FC -> imputed SC, differing only in OBJECTIVE:
  - BR        : per-component BayesianRidge (reconstruction; the reference, = capped_bayesian_ridge)
  - PLS       : max FC<->SC covariance (the reference, = capped_pca_pls)
  - obj1a     : BR + reliability-gated per-PC amplitude RESTORATION (identity objective)
  - obj2c     : BR + cognition-weighted reconstruction (cognition objective; CogCryst-resid)
  - obj2c_raw : obj2c but supervised on RAW CogCryst (side-check: was residualizing needed?)

All share the source/target PCA(256) front-end. See dev-notes/objective-functions/PLAN.md.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.model_selection import KFold

K = 256
EPS = 1e-9


def _src_latents(FC_tr, FC_te):
    p = PCA(n_components=min(K, FC_tr.shape[1]), random_state=0).fit(FC_tr)
    return p.transform(FC_tr).astype(np.float64), p.transform(FC_te).astype(np.float64)


def _tgt_pca(SC_tr):
    p = PCA(n_components=min(K, SC_tr.shape[1]), random_state=0).fit(SC_tr)
    return p, p.transform(SC_tr).astype(np.float64)


def _br_latent(Z_tr, Z_te, W_tr, oof=False, n_splits=5):
    """Per-PC BayesianRidge: Z -> each target latent column. Returns What_te (n_te,K) and,
    if oof, the 5-fold out-of-fold train predictions What_tr_oof (n_tr,K) for honest gains."""
    k = W_tr.shape[1]
    What_te = np.zeros((Z_te.shape[0], k), np.float64)
    for j in range(k):
        What_te[:, j] = BayesianRidge(max_iter=300).fit(Z_tr, W_tr[:, j]).predict(Z_te)
    What_tr_oof = None
    if oof:
        What_tr_oof = np.zeros_like(W_tr)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        for tr, va in kf.split(Z_tr):
            for j in range(k):
                What_tr_oof[va, j] = BayesianRidge(max_iter=300).fit(Z_tr[tr], W_tr[tr, j]).predict(Z_tr[va])
    return What_te, What_tr_oof


def _colcorr(A, B):
    Az = A - A.mean(0); Bz = B - B.mean(0)
    return (Az * Bz).sum(0) / (np.sqrt((Az**2).sum(0) * (Bz**2).sum(0)) + EPS)


# --- Objective 1A: reliability-gated per-PC amplitude restoration --------------
def obj1a_amplitude_restore(FC_tr, FC_te, SC_tr, ctx=None):
    Z_tr, Z_te = _src_latents(FC_tr, FC_te)
    pca, W_tr = _tgt_pca(SC_tr)
    What_te, What_oof = _br_latent(Z_tr, Z_te, W_tr, oof=True)
    r = np.clip(_colcorr(What_oof, W_tr), 0, None)            # reliability per PC (>=0)
    g = r * (W_tr.std(0) / (What_oof.std(0) + EPS))           # gain = reliability * amplitude-gap
    return pca.inverse_transform(What_te * g).astype(np.float32)


# --- Objective 2C: cognition-weighted reconstruction ---------------------------
def _oof_cog_beta(W_tr, c_tr, n_splits=5):
    """OOF-averaged ridge coefficients of cognition ~ target latents -> per-PC relevance beta."""
    ok = ~np.isnan(c_tr)
    Wk, ck = W_tr[ok], c_tr[ok]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    coefs = []
    for tr, _ in kf.split(Wk):
        coefs.append(Ridge(alpha=1.0).fit(Wk[tr], ck[tr]).coef_)
    return np.mean(coefs, axis=0)                            # beta (K,)


def _obj2c(FC_tr, FC_te, SC_tr, c_tr):
    Z_tr, Z_te = _src_latents(FC_tr, FC_te)
    pca, W_tr = _tgt_pca(SC_tr)
    beta = _oof_cog_beta(W_tr, np.asarray(c_tr, np.float64))
    w = beta**2
    a = w / (w.mean() + EPS)                                 # emphasis, mean 1 (cog PCs up, others down)
    What_te, _ = _br_latent(Z_tr, Z_te, W_tr, oof=False)
    return pca.inverse_transform(What_te * a).astype(np.float32)


def obj2c_cog_weighted(FC_tr, FC_te, SC_tr, ctx):
    return _obj2c(FC_tr, FC_te, SC_tr, ctx["c_resid_tr"])     # CogCryst residualized over bv+demo


def obj2c_cog_weighted_raw(FC_tr, FC_te, SC_tr, ctx):
    return _obj2c(FC_tr, FC_te, SC_tr, ctx["c_raw_tr"])       # RAW CogCryst (side-check)


# estimator registry (BR/PLS imported lazily from _grid_common to avoid torch at import time)
def estimator_registry():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from _grid_common import capped_bayesian_ridge, capped_pca_pls
    return {
        "BR":        lambda a, b, c, ctx=None: capped_bayesian_ridge(a, b, c),
        "PLS":       lambda a, b, c, ctx=None: capped_pca_pls(a, b, c),
        "obj1a":     obj1a_amplitude_restore,
        "obj2c":     obj2c_cog_weighted,
        "obj2c_raw": obj2c_cog_weighted_raw,
    }
