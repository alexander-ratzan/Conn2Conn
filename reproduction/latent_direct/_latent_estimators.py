"""Latent-direct estimators — the previous obj_functions objectives, but returning the
TASK-TUNED LATENT (W-hat ⊙ objective-weight) instead of inverse-transforming to a connectome.

Same math as reproduction/obj_functions/_obj_estimators.py up to (and including) the per-PC
objective weighting; we simply STOP before `pca.inverse_transform(...)`. Downstream then runs
directly on this latent (no inverse-PCA, no re-PCA). Comparing these numbers to the
obj_functions round-trip scorecard isolates exactly what the "PCA glow-up" costs.

Each fn maps source latents -> predicted TARGET latents (n_te × k_tgt), generic over direction
(FC->SC or SC->FC). Internals are reused verbatim from _obj_estimators.
"""
import sys
from pathlib import Path
import numpy as np
from sklearn.cross_decomposition import PLSRegression

# reuse the parked objective internals verbatim (single source of truth)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "obj_functions"))
import _obj_estimators as oe   # noqa: E402  (_src_latents, _tgt_pca, _br_latent, _colcorr, _oof_cog_beta, EPS, K)


def br_latent(src_tr, src_te, tgt_tr, ctx=None):
    """Per-component BayesianRidge target latents (reconstruction objective)."""
    Z_tr, Z_te = oe._src_latents(src_tr, src_te)
    _, W_tr = oe._tgt_pca(tgt_tr)
    What_te, _ = oe._br_latent(Z_tr, Z_te, W_tr, oof=False)
    return What_te


def pls_latent(src_tr, src_te, tgt_tr, ctx=None):
    """PLS-predicted target latents (max source<->target covariance objective)."""
    Z_tr, Z_te = oe._src_latents(src_tr, src_te)
    _, W_tr = oe._tgt_pca(tgt_tr)
    k = min(64, Z_tr.shape[1], W_tr.shape[1])
    pls = PLSRegression(n_components=k, scale=True, max_iter=2000).fit(Z_tr, W_tr)
    return pls.predict(Z_te)


def obj1a_latent(src_tr, src_te, tgt_tr, ctx=None):
    """BR + reliability-gated per-PC amplitude restoration (identity objective)."""
    Z_tr, Z_te = oe._src_latents(src_tr, src_te)
    _, W_tr = oe._tgt_pca(tgt_tr)
    What_te, What_oof = oe._br_latent(Z_tr, Z_te, W_tr, oof=True)
    r = np.clip(oe._colcorr(What_oof, W_tr), 0, None)
    g = r * (W_tr.std(0) / (What_oof.std(0) + oe.EPS))
    return What_te * g


def _obj2c_latent(src_tr, src_te, tgt_tr, c_tr):
    Z_tr, Z_te = oe._src_latents(src_tr, src_te)
    _, W_tr = oe._tgt_pca(tgt_tr)
    beta = oe._oof_cog_beta(W_tr, np.asarray(c_tr, np.float64))
    a = beta**2 / (np.mean(beta**2) + oe.EPS)
    What_te, _ = oe._br_latent(Z_tr, Z_te, W_tr, oof=False)
    return What_te * a


def obj2c_latent(src_tr, src_te, tgt_tr, ctx):
    return _obj2c_latent(src_tr, src_te, tgt_tr, ctx["c_resid_tr"])


def obj2c_raw_latent(src_tr, src_te, tgt_tr, ctx):
    return _obj2c_latent(src_tr, src_te, tgt_tr, ctx["c_raw_tr"])


# cross-modal latent estimators (FC->SC and SC->FC arms)
LATENT_ESTIMATORS = {
    "BR": br_latent, "PLS": pls_latent, "obj1a": obj1a_latent,
    "obj2c": obj2c_latent, "obj2c_raw": obj2c_raw_latent,
}
