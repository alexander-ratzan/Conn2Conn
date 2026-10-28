"""Spec v4 E4.0c: B1 / B2 (and B3 at k = 256, 512) re-selected on val average rank (D9), at k = 256, 512 and full rank.

Code spaces
  k = 256, 512  E2's pooled block FC PCA codes (D2; as E4.0)
  full          each condition's own train PCA with every component (rank <= n_train - 1): ridge / regularised CCA on
                these codes equal ridge / regularised CCA on the edges (the fits lie in the train span), i.e. CCA without
                PCA truncation; built from E2's loader (`load_views`), decoded with each view's own basis
Selection (D9: validation average rank, decoded edges, rows = targets as v3:E0)
  B1 ridge per cell: penalty from E2.0's grid (x mean eigenvalue), SC code size for +SC cells
  B2 MAXVAR over the 9 views: (d, lambda), d in {16, 32, 64, 128, 256} (<= k), on the mean over the 56 single-source cells
  B3 (k = 256, 512): B1's selected mean + the cell's residual covariance (cross-fitted), 64 samples
References: B0 copy (edges), B0-codes (copy in pooled codes, k = 256 / 512). Check: B1 with GCV selection at k = 256 /
512 reproduces E4.0's B1 rows (same fits, other selection rule).

    python scripts/experiments/cross_condition_translation/run_e40c.py --seed 0 --k 256|512|full
        -> results/e40c/k<k>/seed<S>.json
"""
import argparse
import json
import time

import numpy as np

from e4_common import (CFG, FC, HERE, OTHER7, SC_ONLY, VIEWS, Scorer, SeedData, cells, energy_metrics, maxvar,
                       ridge_gcv, score, sources_of)
from run_baselines import crossfit_ridge
from models.eval.metrics import compute_corr_matrix, corr_avg_rank

SPLITS = ("val", "test")
D_GRID = [16, 32, 64, 128, 256]


class Space:
    """Codes per view and split, decoding to edges, and the edges / train means to score against."""

    def __init__(self, seed, k):
        self.k = k
        if k != "full":
            sd = SeedData(seed, "main")
            self.sd = sd
            self.Z = {s: {v: sd.code(v, s, int(k)).astype(np.float64) for v in VIEWS} for s in ("train",) + SPLITS}
            self.Zsc = {s: sd.code("SC", s, max(CFG["k_sc"])).astype(np.float64) for s in ("train",) + SPLITS}
            self._edges = lambda split, v: sd.edges[(split, v)]
            self._mean = lambda v: sd.train_mean[v]
            self._dec = lambda v, z: sd.decode(v, z)
            return
        from common import load_views          # E2 loader: all views on the matched cohort, E2's splits
        data, splits, _ = load_views(seed)
        self.Z = {s: {} for s in ("train",) + SPLITS}
        self.basis, self.mu, self.E = {}, {}, {}
        for v in VIEWS:
            X = data[v].astype(np.float64)
            tr = splits["train"]
            mu = X[tr].mean(0)
            Xc = X[tr] - mu
            w, U = np.linalg.eigh(Xc @ Xc.T)
            keep = w > 1e-9 * w.max()
            V = (Xc.T @ U[:, keep]) / np.sqrt(w[keep])            # (E, r) orthonormal train basis
            self.basis[v], self.mu[v] = V, mu
            for s in ("train",) + SPLITS:
                self.Z[s][v] = (X[splits[s]] - mu) @ V
                if s != "train":
                    self.E[(s, v)] = X[splits[s]].astype(np.float32)
        self.Zsc = {s: self.Z[s]["SC"] for s in self.Z}
        self._edges = lambda split, v: self.E[(split, v)]
        self._mean = lambda v: self.mu[v]
        self._dec = lambda v, z: self.mu[v] + z @ self.basis[v].T

    def decode(self, v, z):
        return self._dec(v, z)

    def score(self, v, split, z=None, edges=None):
        pred = self.decode(v, z) if edges is None else edges
        return score(pred, self._edges(split, v), self._mean(v))

    def val_rank(self, v, z):
        pred = np.asarray(self.decode(v, z), dtype=np.float64)
        return float(corr_avg_rank(cc=compute_corr_matrix(np.asarray(self._edges("val", v), dtype=np.float64), pred)))

    def edges(self, split, v):
        return self._edges(split, v)

    def train_mean(self, v):
        return self._mean(v)


def ridge_path(X, Y, rel_alphas):
    """Ridge coefficient matrices for every relative penalty (one SVD)."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    UtY = U.T @ Y
    out = {}
    for a_rel in rel_alphas:
        a = a_rel * np.mean(s ** 2)
        out[a_rel] = Vt.T @ ((s / (s ** 2 + a))[:, None] * UtY)
    return out


def run(seed, k):
    t0 = time.time()
    S = Space(seed, k)
    alphas = [float(a) for a in CFG["ridge_alphas"]]
    rows, dist, hps, check = [], [], {}, []
    add = lambda model, src, tgt, with_sc, split, m, **kw: rows.append(
        dict(seed=seed, k=str(k), model=model, source=src, target=tgt, with_sc=with_sc, split=split, **kw, **m))

    # references
    for split in SPLITS:
        for src, tgt, with_sc in cells():
            if with_sc or src == SC_ONLY:
                continue
            srcs = sources_of(src, tgt)
            dev = np.mean([S.edges(split, a) - S.train_mean(a) for a in srcs], axis=0)
            add("B0", src, tgt, False, split, S.score(tgt, split, edges=S.train_mean(tgt) + dev))
            if k != "full":
                add("B0-codes", src, tgt, False, split, S.score(tgt, split, np.mean([S.Z[split][a] for a in srcs], axis=0)))
    print(f"seed {seed} k {k}: references ({time.time() - t0:.0f} s)", flush=True)

    # B1 ridge (+SC), selected on val average rank; B3 on its mean (pooled codes only)
    ksc_grid = CFG["k_sc"] if k != "full" else [None]          # full rank: all SC components
    for src, tgt, with_sc in cells():
        srcs = sources_of(src, tgt)
        base = {s: np.concatenate([S.Z[s][a] for a in srcs], axis=1) if srcs else np.zeros((len(S.Z[s][tgt]), 0)) for s in S.Z}
        Y = S.Z["train"][tgt]
        best, gcv_pick = None, None
        for ksc in (ksc_grid if with_sc else [0]):
            Xk = {s: np.concatenate([base[s], S.Zsc[s][:, :ksc] if ksc is not None else S.Zsc[s]], axis=1) for s in base}
            path = ridge_path(Xk["train"], Y, alphas)
            for a_rel, B in path.items():
                v = S.val_rank(tgt, Xk["val"] @ B)
                if best is None or v > best[0]:
                    best = (v, ksc, a_rel, B, Xk)
            if k != "full" and not with_sc:                          # consistency with E4.0 (GCV choice)
                B_g, a_g = ridge_gcv(Xk["train"], Y, alphas)
                gcv_pick = (Xk, B_g)
        _, ksc, a_rel, B, Xk = best
        name = "B1+SC" if with_sc else "B1"
        hps[f"{name}|{src}|{tgt}"] = dict(k_sc=ksc, alpha_rel=a_rel, val_avg_rank=best[0])
        for split in SPLITS:
            add(name, src, tgt, with_sc, split, S.score(tgt, split, Xk[split] @ B), k_sc=ksc, alpha_rel=a_rel)
        if gcv_pick is not None:
            Xg, Bg = gcv_pick
            m = S.score(tgt, "test", Xg["test"] @ Bg)
            check.append(dict(source=src, target=tgt, demeaned_pearson=m["demeaned_pearson"], avg_rank=m["avg_rank"]))
        if k != "full":
            resid = Y - crossfit_ridge(Xk["train"], Y, a_rel, CFG["crossfit_folds"], seed)
            cov = np.cov(resid, rowvar=False)
            rng = np.random.default_rng(seed)
            for split in SPLITS:
                P = Xk[split] @ B
                smp = P[None] + rng.multivariate_normal(np.zeros(P.shape[1]), cov, size=(CFG["n_samples"], len(P)))
                dist.append(dict(seed=seed, k=str(k), model="B3+SC" if with_sc else "B3", source=src, target=tgt,
                                 with_sc=with_sc, split=split, **energy_metrics(smp, S.Z[split][tgt])))
    print(f"seed {seed} k {k}: B1 / B3 ({time.time() - t0:.0f} s)", flush=True)

    # B2 MAXVAR over the 9 views, (d, lambda) on mean val average rank over the 56 single-source cells
    single = [(a, b) for a in FC for b in FC if a != b]
    kdim = min(S.Z["train"][v].shape[1] for v in VIEWS)
    best = None
    for d in [x for x in D_GRID if x <= kdim]:
        for lam in CFG["maxvar_lambda"]:
            A, D, _ = maxvar(S.Z["train"], d, float(lam))
            val = float(np.mean([S.val_rank(b, S.Z["val"][a] @ A[a] @ D[b]) for a, b in single]))
            if best is None or val > best[0]:
                best = (val, d, float(lam), A, D)
    _, d, lam, A, D = best
    hps["B2"] = dict(d=d, lambda_rel=lam, val_mean_avg_rank=best[0])
    for split in SPLITS:
        for src, tgt, with_sc in cells():
            views = sources_of(src, tgt) + (["SC"] if with_sc else [])
            g = np.mean([S.Z[split][v] @ A[v] for v in views], axis=0)
            add("B2+SC" if with_sc and src != SC_ONLY else "B2", src, tgt, with_sc, split, S.score(tgt, split, g @ D[tgt]))
    print(f"seed {seed} k {k}: B2 d {d} lambda {lam} ({time.time() - t0:.0f} s)", flush=True)

    out = dict(seed=seed, k=str(k), hyperparams=hps, gcv_check=check, rows=rows, dist=dist, seconds=time.time() - t0)
    res = HERE / "results" / "e40c" / f"k{k}"
    res.mkdir(parents=True, exist_ok=True)
    (res / f"seed{seed}.json").write_text(json.dumps(out, default=float))
    print(f"wrote results/e40c/k{k}/seed{seed}.json in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--k", required=True, choices=["256", "512", "full"])
    a = ap.parse_args()
    run(a.seed, a.k if a.k == "full" else int(a.k))
