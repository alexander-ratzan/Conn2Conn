"""Spec v3 E2 feasibility: time one closed-form multi-block PCA-PLS grid cell at real shapes and check that
truncating one n_pls=128 fit reproduces separate fits for every smaller n_pls (NIPALS components are sequential).

Synthetic data with the Glasser / E0-cohort shapes (659 train, 74 val, 184 test subjects, 64,620 edges); timing only.
    python scripts/experiments/joint_taskfc2sc/checks/bench_pls_grid.py
"""
import time

import numpy as np
from numpy.linalg import pinv
from sklearn.cross_decomposition import PLSRegression

N_TR, N_VA, N_TE, E = 659, 74, 184, 64620
NPLS = [4, 8, 16, 32, 64, 128]


def truncated_predict(pls, X, k):
    """Predictions of the first k components of a fitted PLSRegression (same algebra as sklearn's fit)."""
    W, P, Q = pls.x_weights_[:, :k], pls.x_loadings_[:, :k], pls.y_loadings_[:, :k]
    rot = W @ pinv(P.T @ W)
    coef = ((rot @ Q.T) * pls._y_std).T / pls._x_std
    return (X - pls._x_mean) @ coef.T + pls.intercept_


def metrics(pred_e, true_e, train_mean):
    """Pearson r (per subject, mean), demeaned r, average rank and top-1 of the subject x subject corr matrix."""
    def rowcorr(a, b):
        a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
        return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
    pd, td = pred_e - train_mean, true_e - train_mean
    pz = (pd - pd.mean(1, keepdims=True)) / pd.std(1, keepdims=True)
    tz = (td - td.mean(1, keepdims=True)) / td.std(1, keepdims=True)
    cc = pz @ tz.T / pd.shape[1]
    rank = (cc < np.diag(cc)[None, :]).mean(0).mean()
    return dict(pearson=rowcorr(pred_e, true_e).mean(), demeaned=rowcorr(pd, td).mean(), avg_rank=rank,
                top1=(cc.argmax(0) == np.arange(len(cc))).mean())


def main():
    rng = np.random.default_rng(0)
    # SC target: low-rank + noise so PLS has structure to find; PCA basis from the train split
    sc = (rng.standard_normal((N_TR + N_VA + N_TE, 300)) @ rng.standard_normal((300, E)) * 0.05
          + rng.standard_normal((N_TR + N_VA + N_TE, E))).astype(np.float32)
    tr, ev = slice(0, N_TR), slice(N_TR, None)
    t0 = time.time()
    mu = sc[tr].mean(0)
    _, _, vt = np.linalg.svd(sc[tr] - mu, full_matrices=False)
    print(f"SC PCA (659 x 64620 SVD): {time.time() - t0:.1f} s", flush=True)
    for d_in, k_sc in [(960, 256), (960, 32), (544, 32), (64, 32)]:
        X = rng.standard_normal((N_TR + N_VA + N_TE, d_in))
        L = vt[:k_sc].T                                   # edges x k_sc
        Y = (sc - mu) @ L
        Y[tr] += X[tr, :min(d_in, k_sc)] * 0.5            # make X informative
        t0 = time.time()
        pls = PLSRegression(n_components=min(128, d_in), scale=True).fit(X[tr], Y[tr])
        t_fit = time.time() - t0
        t0 = time.time()
        for k in [k for k in NPLS if k <= min(128, d_in)]:
            yhat = truncated_predict(pls, X[ev], k)
            m = metrics(yhat @ L.T + mu, sc[ev], mu)       # reconstruct edges and score val + test together
        t_eval = time.time() - t0
        # exactness: separate n_pls = 8 / 32 fits vs truncation
        err = []
        for k in (8, 32):
            sep = PLSRegression(n_components=k, scale=True).fit(X[tr], Y[tr]).predict(X[ev])
            err.append(np.abs(sep - truncated_predict(pls, X[ev], k)).max() / np.abs(sep).max())
        print(f"D={d_in:4d} k_sc={k_sc:3d}: fit(128) {t_fit:5.1f} s, 6 truncations + edge metrics {t_eval:5.1f} s, "
              f"max rel. diff vs separate fits {max(err):.1e}", flush=True)


if __name__ == "__main__":
    main()
