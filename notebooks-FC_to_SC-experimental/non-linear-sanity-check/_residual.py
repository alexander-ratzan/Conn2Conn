"""Additive residual (boosted) learning helpers — architecture (A) from
DESIGN_residual_learning.md.

  template    = OOF linear prediction on train (honest residual, no leakage)
  resid_train = y_train - template
  NL : source -> resid_train         (NL's loss IS the improvement above template)
  final       = linear(full-train)->test  +  NL(source_test)

Two entry points: reconstruction (PLS base, multi-output target latents) and
cognition (BayesianRidge base, scalar target). Both return (final, template) so the
caller can evaluate the improvement on the full metric panel.
"""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nl_common import PCA, BayesianRidge
from _tract_setup import PLSRegression, StandardScaler, _median_gamma  # noqa
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold

K_PCA = 256
K_PLS = 64
N_FOLDS = 5


def residual_reconstruct(X_src_train, X_src_test, Y_train,
                         k_src=K_PCA, k_tgt=K_PCA, k_pls=K_PLS,
                         n_folds=N_FOLDS, alpha=1.0):
    """Additive residual reconstruction. Returns (final_test, template_test) in the
    ORIGINAL target edge space (inverse-PCA'd), so caller runs full_panel_eval on both.

    template = OOF-then-full-train PLS;  final = template + KernelRidge(residual)."""
    pca_src = PCA(n_components=min(k_src, X_src_train.shape[1]), random_state=0).fit(X_src_train)
    Z_tr = pca_src.transform(X_src_train)
    Z_te = pca_src.transform(X_src_test)
    pca_tgt = PCA(n_components=min(k_tgt, Y_train.shape[1]), random_state=0).fit(Y_train)
    Y_lat = pca_tgt.transform(Y_train)

    # --- OOF PLS template on train (in target-latent space) ---
    oof = np.zeros_like(Y_lat)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    for tr_idx, va_idx in kf.split(Z_tr):
        pls = PLSRegression(n_components=k_pls, scale=True, max_iter=2000)
        pls.fit(Z_tr[tr_idx], Y_lat[tr_idx])
        oof[va_idx] = pls.predict(Z_tr[va_idx])
    resid_lat = Y_lat - oof   # honest residual

    # --- NL (KernelRidge RBF) on residual; input = standardized source latents ---
    sc = StandardScaler().fit(Z_tr)
    Z_tr_s = sc.transform(Z_tr)
    Z_te_s = sc.transform(Z_te)
    gamma = _median_gamma(Z_tr_s)
    kr = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma).fit(Z_tr_s, resid_lat)

    # --- full-train PLS for the test template ---
    pls_full = PLSRegression(n_components=k_pls, scale=True, max_iter=2000).fit(Z_tr, Y_lat)
    template_te_lat = pls_full.predict(Z_te)
    final_te_lat = template_te_lat + kr.predict(Z_te_s)

    final_te = pca_tgt.inverse_transform(final_te_lat).astype(np.float32)
    template_te = pca_tgt.inverse_transform(template_te_lat).astype(np.float32)
    return final_te, template_te


def _blocks_to_latents(blocks_train, blocks_test, k_per_block=K_PCA):
    """Per-block PCA (scale-fair) + per-block standardize -> concat. Narrow blocks
    (bv, demo) standardized and passed through raw."""
    Ztr_parts, Zte_parts = [], []
    for Xtr, Xte in zip(blocks_train, blocks_test):
        if Xtr.shape[1] <= k_per_block:
            sc = StandardScaler().fit(Xtr)
            Ztr_parts.append(sc.transform(Xtr)); Zte_parts.append(sc.transform(Xte))
        else:
            p = PCA(n_components=k_per_block, random_state=0).fit(Xtr)
            ztr, zte = p.transform(Xtr), p.transform(Xte)
            sc = StandardScaler().fit(ztr)
            Ztr_parts.append(sc.transform(ztr)); Zte_parts.append(sc.transform(zte))
    return np.concatenate(Ztr_parts, axis=1), np.concatenate(Zte_parts, axis=1)


def residual_cognition_blocks(blocks_train, blocks_test, y_train, n_folds=N_FOLDS, alpha=1.0):
    """Multimodal-sink additive residual cognition. Per-block PCA -> concat -> OOF-BR
    template + KernelRidge residual. Returns (final, template). template = linear sink."""
    Z_tr, Z_te = _blocks_to_latents(blocks_train, blocks_test)
    ok = ~np.isnan(y_train)
    Z_ok, y_ok = Z_tr[ok], y_train[ok]
    oof = np.zeros_like(y_ok)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    for tr_idx, va_idx in kf.split(Z_ok):
        br = BayesianRidge(max_iter=500).fit(Z_ok[tr_idx], y_ok[tr_idx])
        oof[va_idx] = br.predict(Z_ok[va_idx])
    resid = y_ok - oof
    gamma = _median_gamma(Z_ok)
    kr = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma).fit(Z_ok, resid)
    br_full = BayesianRidge(max_iter=500).fit(Z_ok, y_ok)
    template_te = br_full.predict(Z_te)
    final_te = template_te + kr.predict(Z_te)
    return final_te, template_te


def residual_reconstruct_blocks(blocks_train, blocks_test, Y_train,
                                k_tgt=K_PCA, k_pls=K_PLS, n_folds=N_FOLDS, alpha=1.0):
    """Multimodal-sink additive residual reconstruction. Per-block PCA(src) -> concat ->
    OOF-PLS template + KernelRidge residual -> inverse-PCA. Returns (final, template)
    in original target edge space."""
    Z_tr, Z_te = _blocks_to_latents(blocks_train, blocks_test)
    pca_tgt = PCA(n_components=min(k_tgt, Y_train.shape[1]), random_state=0).fit(Y_train)
    Y_lat = pca_tgt.transform(Y_train)
    oof = np.zeros_like(Y_lat)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    for tr_idx, va_idx in kf.split(Z_tr):
        pls = PLSRegression(n_components=k_pls, scale=True, max_iter=2000)
        pls.fit(Z_tr[tr_idx], Y_lat[tr_idx])
        oof[va_idx] = pls.predict(Z_tr[va_idx])
    resid_lat = Y_lat - oof
    gamma = _median_gamma(Z_tr)
    kr = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma).fit(Z_tr, resid_lat)
    pls_full = PLSRegression(n_components=k_pls, scale=True, max_iter=2000).fit(Z_tr, Y_lat)
    tmpl_lat = pls_full.predict(Z_te)
    final_lat = tmpl_lat + kr.predict(Z_te)
    return (pca_tgt.inverse_transform(final_lat).astype(np.float32),
            pca_tgt.inverse_transform(tmpl_lat).astype(np.float32))


def _reduce_scalar(X_tr, X_te, k=K_PCA):
    if X_tr.shape[1] <= k:
        sc = StandardScaler().fit(X_tr)
        return sc.transform(X_tr), sc.transform(X_te)
    p = PCA(n_components=k, random_state=0).fit(X_tr)
    Ztr, Zte = p.transform(X_tr), p.transform(X_te)
    sc = StandardScaler().fit(Ztr)
    return sc.transform(Ztr), sc.transform(Zte)


def residual_cognition(X_train, X_test, y_train, k=K_PCA, n_folds=N_FOLDS, alpha=1.0):
    """Additive residual cognition. base = BayesianRidge, NL = KernelRidge(RBF).
    Returns (final_test, template_test) scalar predictions. NaN y rows dropped."""
    Z_tr, Z_te = _reduce_scalar(X_train, X_test, k=k)
    ok = ~np.isnan(y_train)
    Z_ok, y_ok = Z_tr[ok], y_train[ok]

    # OOF BR template on train.
    oof = np.zeros_like(y_ok)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    for tr_idx, va_idx in kf.split(Z_ok):
        br = BayesianRidge(max_iter=500).fit(Z_ok[tr_idx], y_ok[tr_idx])
        oof[va_idx] = br.predict(Z_ok[va_idx])
    resid = y_ok - oof

    gamma = _median_gamma(Z_ok)
    kr = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma).fit(Z_ok, resid)
    br_full = BayesianRidge(max_iter=500).fit(Z_ok, y_ok)

    template_te = br_full.predict(Z_te)
    final_te = template_te + kr.predict(Z_te)
    return final_te, template_te
