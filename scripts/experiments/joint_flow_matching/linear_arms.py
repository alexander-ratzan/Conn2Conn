"""Spec v4 E1 arm A1 (closed-form linear references) and the per-pair references, for one seed, at every k.

  L1 per-pair ridge (penalty by generalized cross-validation on train);
  L2 L1 with each output dimension rescaled to the target's train SD (scale from 5-fold CV predictions on train);
  L3 Gaussian OT map between the train source and target code distributions (Ledoit-Wolf; no pairing);
  L4 multiview CCA translator (MAXVAR over all views; (d, lambda) on mean val demeaned r over all pairs);
  references: identity (FC -> FC, in edge space), ceiling (the target's own k-dim reconstruction), N0 (train mean).
Also L4 multi-source SC: the 8 FC encodings averaged, then decoded as SC (pair `FCall -> SC`).

Writes `records_dir/linear/seed<S>_<variant>.csv` (one row per ref x k x pair x split).

    python scripts/experiments/joint_flow_matching/linear_arms.py --seed 0 [--variant main] [--ks 64 128 256 512]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf

from common import CFG, EXP_DIR, REPO_ROOT, SeedData, demeaned_r_torch, pairs, score
from models.architectures.multiview_cca import maxvar, ridge_gcv  # noqa: F401 (I2; re-exported for checks)

SPLITS = ("val", "test")


def cv_scale(X, Y, a_rel, n_folds=5, seed=0):
    """Per-output SD(target) / SD(5-fold CV ridge prediction) on train: the L2 de-shrinkage factors."""
    idx = np.random.default_rng(seed).permutation(len(X))
    pred = np.zeros_like(Y)
    for f in np.array_split(idx, n_folds):
        tr = np.setdiff1d(idx, f)
        U, s, Vt = np.linalg.svd(X[tr], full_matrices=False)
        a = a_rel * np.mean(s ** 2)
        B = Vt.T @ ((s / (s ** 2 + a))[:, None] * (U.T @ Y[tr]))
        pred[f] = X[f] @ B
    return Y.std(0) / np.maximum(pred.std(0), 1e-12)


def _sqrtm_psd(S, inv=False):
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 1e-12, None)
    return (V * (w ** (-0.5 if inv else 0.5))) @ V.T


def gaussian_ot(X, Y):
    """Linear OT map between N(mu_x, S_x) and N(mu_y, S_y): T(z) = mu_y + (z - mu_x) A. Returns (mu_x, A, mu_y)."""
    Sx = LedoitWolf().fit(X).covariance_
    Sy = LedoitWolf().fit(Y).covariance_
    Sx_h, Sx_ih = _sqrtm_psd(Sx), _sqrtm_psd(Sx, inv=True)
    A = Sx_ih @ _sqrtm_psd(Sx_h @ Sy @ Sx_h) @ Sx_ih
    return X.mean(0), A, Y.mean(0)


class Scorer:
    """Decodes predicted codes of a target view and scores them against that view's val / test edges."""

    def __init__(self, sd):
        self.sd = sd

    def __call__(self, view, split, pred_codes=None, pred_edges=None):
        if pred_edges is None:
            pred_edges = self.sd.decode(view, pred_codes)
        return score(pred_edges, self.sd.edges[(split, view)], self.sd.train_mean[view])

    def fast_dr(self, view, split, pred_codes):
        t = torch.as_tensor
        return demeaned_r_torch(t(self.sd.decode(view, pred_codes)), t(self.sd.edges[(split, view)]),
                                t(self.sd.train_mean[view]))


def run(seed, variant, ks, out=None):
    sd = SeedData(seed, variant)
    sc = Scorer(sd)
    views, lin = CFG["views"], {k: [float(x) for x in v] for k, v in CFG["linear"].items()}
    P = pairs(views)
    fc_views = [v for v in views if v != "SC"]
    rows = []

    def add(ref, k, src, tgt, split, metrics, hp=None):
        rows.append(dict(seed=seed, variant=variant, ref=ref, k=k, source=src, target=tgt, split=split,
                         hyperparams=json.dumps(hp or {}), **metrics))

    for split in SPLITS:
        for v in views:
            mu = sd.train_mean[v].astype(np.float64)
            true = sd.edges[(split, v)].astype(np.float64)
            add("N0", 0, "-", v, split, dict(pearson=float(np.mean([np.corrcoef(mu, t)[0, 1] for t in true]))))
            for k in ks:
                add("ceiling", k, v, v, split, sc(v, split, sd.code(v, split, k)))
        for a, b in P:
            if a != "SC" and b != "SC":
                ident = sd.train_mean[b] + (sd.edges[(split, a)] - sd.train_mean[a])
                add("identity", 0, a, b, split, sc(b, split, pred_edges=ident))

    for k in ks:
        Z = {v: sd.code(v, "train", k).astype(np.float64) for v in views}
        for a, b in P:
            B, a_rel = ridge_gcv(Z[a], Z[b], lin["ridge_alphas"])
            scale = cv_scale(Z[a], Z[b], a_rel, seed=seed)
            mx, A, my = gaussian_ot(Z[a], Z[b])
            for split in SPLITS:
                x = sd.code(a, split, k).astype(np.float64)
                p1 = x @ B
                add("L1", k, a, b, split, sc(b, split, p1), dict(alpha_rel=a_rel))
                add("L2", k, a, b, split, sc(b, split, p1 * scale), dict(alpha_rel=a_rel))
                add("L3", k, a, b, split, sc(b, split, my + (x - mx) @ A))
            print(f"seed {seed} k {k}: {a} -> {b} L1-L3 done", flush=True)

        best = None
        for d in [int(d) for d in lin["maxvar_d"] if d <= k]:
            for lam in lin["maxvar_lambda"]:
                Aenc, D, _ = maxvar(Z, d, lam)
                val = np.mean([sc.fast_dr(b, "val", sd.code(a, "val", k) @ Aenc[a] @ D[b]) for a, b in P])
                if best is None or val > best[0]:
                    best = (val, d, lam, Aenc, D)
        _, d, lam, Aenc, D = best
        hp = dict(d=d, lambda_rel=lam, val_mean_dr=best[0])
        for split in SPLITS:
            for a, b in P:
                add("L4", k, a, b, split, sc(b, split, sd.code(a, split, k) @ Aenc[a] @ D[b]), hp)
            g = np.mean([sd.code(a, split, k) @ Aenc[a] for a in fc_views], axis=0)
            add("L4", k, "FCall", "SC", split, sc("SC", split, g @ D["SC"]), hp)
        print(f"seed {seed} k {k}: L4 d={d} lambda={lam} val {best[0]:.4f}", flush=True)

    out = Path(out) if out else REPO_ROOT / CFG["records_dir"] / "linear" / f"seed{seed}_{variant}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--variant", default="main")
    ap.add_argument("--ks", type=int, nargs="*", default=CFG["k_grid"])
    ap.add_argument("--out", help="output CSV (default: records_dir/linear/seed<S>_<variant>.csv)")
    args = ap.parse_args()
    torch.set_num_threads(8)
    run(args.seed, args.variant, args.ks, args.out)


if __name__ == "__main__":
    main()
