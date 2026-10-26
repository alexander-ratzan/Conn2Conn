"""Spec v4 E2.1 shared pieces: the latent-target cache, linear references, decoding and metrics.

Target: E1's SC-blind latent (MAXVAR over the 8 FC conditions on block FC PCA codes; per seed the E1 selection, which
was k_fc 256, d 32, lambda 0.1 in every seed). For every subject and condition m, the encoding g^m = z^m A_m (32-d);
the consensus is the mean over the 8 conditions. Input: SC codes (E2's equal-variance SC basis, up to 512).

Linear references (closed form, CPU), stored in the cache:
  ridge_cons  SC -> consensus; per condition the SC code size chosen on val decoded demeaned r (E1's rule; gate G0b)
  ridge_cond  SC -> each condition's encoding; SC code size per condition chosen on val; train predictions cross-fitted
              (5 folds) so they look like test-time predictions (start point of the corrector and the bridge)
  gauss       linear-Gaussian: ridge_cond means + one residual covariance (cross-fitted train residuals, pooled)

    python scripts/experiments/joint_flow_matching/e21_common.py --seeds 0 1 2 3 4 [--check]   # build caches (+ G0b)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CFG as E2CFG, REPO_ROOT, SeedData, score  # noqa: E402
from models.architectures.multiview_cca import maxvar, ridge_gcv  # noqa: E402

FC = ["rest", "emotion", "gambling", "language", "motor", "relational", "social", "wm"]
K_SC = [64, 128, 256, 512]
ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4]
E1_DIR = REPO_ROOT / "scripts" / "experiments" / "cca_latent_bench" / "results" / "Glasser"
CACHE = Path(E2CFG["cache_dir"]) / "e21"


def e1_latent(seed):
    sel = pd.read_csv(E1_DIR / "tables" / "latent_selection.csv")
    r = sel[(sel.seed == seed) & (sel.selected_sc_blind.fillna(False).astype(bool))].iloc[0]
    return int(r.k_fc), int(r.d), float(r.lambda_rel)


def ridge_fit(X, Y):
    xm, ym = X.mean(0), Y.mean(0)
    B, _ = ridge_gcv(X - xm, Y - ym, ALPHAS)
    return lambda Z: ym + (Z - xm) @ B


def crossfit(X, Y, n_folds=5, seed=0):
    idx = np.random.default_rng(seed).permutation(len(X))
    out = np.zeros_like(Y)
    for f in np.array_split(idx, n_folds):
        tr = np.setdiff1d(idx, f)
        out[f] = ridge_fit(X[tr], Y[tr])(X[f])
    return out


def fast_dr(pred, true, mu):
    a, b = pred - mu, true - mu
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return float(np.mean((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)))


class Latent:
    """Everything E2.1 needs for one seed (built from E2's SeedData; cached under CACHE)."""

    def __init__(self, seed):
        self.seed = seed
        self.sd = SeedData(seed, "main")
        self.k, self.d, self.lam = e1_latent(seed)
        self.splits = self.sd.splits
        Z = {s: {v: self.sd.code(v, s, self.k).astype(np.float64) for v in FC} for s in ("train", "val", "test")}
        self.A, self.D, _ = maxvar(Z["train"], self.d, self.lam)
        self.enc = {s: np.stack([Z[s][v] @ self.A[v] for v in FC]) for s in Z}        # (8, n, d)
        self.sc = {s: self.sd.code("SC", s, max(K_SC)).astype(np.float64) for s in Z}  # (n, 512)

    def decode(self, m, g):
        """Latent (n, d) of condition m -> FC edges of condition m."""
        return self.sd.decode(FC[m], g @ self.D[FC[m]])

    def score_cond(self, m, split, g):
        v = FC[m]
        return score(self.decode(m, g), self.sd.edges[(split, v)], self.sd.train_mean[v])

    def val_dr(self, m, g):
        v = FC[m]
        return fast_dr(self.decode(m, g), self.sd.edges[("val", v)], self.sd.train_mean[v])

    # ---- linear references
    def ridge_cons(self):
        """E1's SC -> consensus latent ridge, SC code size chosen per condition on val decoded demeaned r."""
        cons = {s: self.enc[s].mean(0) for s in self.enc}
        rows, preds = [], {}
        fits = {k: ridge_fit(self.sc["train"][:, :k], cons["train"]) for k in K_SC}
        for m in range(len(FC)):
            best = max(K_SC, key=lambda k: self.val_dr(m, fits[k](self.sc["val"][:, :k])))
            for split in ("val", "test"):
                p = fits[best](self.sc[split][:, :best])
                rows.append(dict(model="ridge_cons", condition=FC[m], split=split, k_sc=best,
                                 **self.score_cond(m, split, p)))
                preds[(m, split)] = p
        return rows, preds

    def ridge_cond(self):
        """SC -> each condition's encoding; SC code size per condition on val. Returns predictions (8, n, d) per
        split (train cross-fitted) and the chosen sizes."""
        out = {s: np.zeros_like(self.enc[s]) for s in self.enc}
        ks = []
        for m in range(len(FC)):
            fits = {k: ridge_fit(self.sc["train"][:, :k], self.enc["train"][m]) for k in K_SC}
            best = max(K_SC, key=lambda k: self.val_dr(m, fits[k](self.sc["val"][:, :k])))
            ks.append(best)
            for split in ("val", "test"):
                out[split][m] = fits[best](self.sc[split][:, :best])
            out["train"][m] = crossfit(self.sc["train"][:, :best], self.enc["train"][m], seed=self.seed)
        return out, ks

    def build(self):
        rc_rows, _ = self.ridge_cons()
        rcond, ks = self.ridge_cond()
        resid = (self.enc["train"] - rcond["train"]).reshape(-1, self.d)
        cov = np.cov(resid, rowvar=False)
        out = dict(enc_train=self.enc["train"], enc_val=self.enc["val"], enc_test=self.enc["test"],
                   sc_train=self.sc["train"], sc_val=self.sc["val"], sc_test=self.sc["test"],
                   rc_train=rcond["train"], rc_val=rcond["val"], rc_test=rcond["test"], rc_k=np.array(ks),
                   gauss_cov=cov, D=np.stack([self.D[v] for v in FC]), latent=np.array([self.k, self.d, self.lam]))
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE / f"seed{self.seed}.npz", **out)
        return rc_rows


def build_sessions(seed):
    """Rest test-retest in the latent: each subject's rest session 1 and session 2 FC, encoded with rest's CCA encoder
    (sessions have their own train mean / scale and rest's block basis; E2 cache). Saved as seed<S>_sessions.npz."""
    L = Latent(seed)
    out = {}
    for split in ("train", "val", "test"):
        for sess in ("rest_S1", "rest_S2"):
            out[f"{sess}_{split}"] = L.sd.code(sess, split, L.k).astype(np.float64) @ L.A["rest"]
    np.savez(CACHE / f"seed{seed}_sessions.npz", **out)


def check_g0b(seed, rc_rows):
    """G0b: ridge -> consensus test demeaned r per condition equals E1's 'via consensus latent' ridge rows."""
    e1 = pd.read_csv(E1_DIR / "records" / f"seed{seed}.csv")
    e1 = e1[(e1.direction == "sc2fc") & (e1.input == "via consensus latent") & (e1.readout == "ridge") & (e1.split == "test")]
    mine = pd.DataFrame(rc_rows)
    mine = mine[mine.split == "test"].set_index("condition").demeaned_pearson
    ref = e1.set_index("condition").demeaned_pearson
    diff = float((mine - ref.loc[mine.index]).abs().max())
    ok = diff < 1e-6
    print(f"{'OK  ' if ok else 'FAIL'} G0b seed {seed}: ridge -> consensus reproduces E1 (max |diff| {diff:.2e})", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--sessions-only", action="store_true", help="only build the rest test-retest encodings")
    args = ap.parse_args()
    torch.set_num_threads(8)
    if args.sessions_only:
        for s in args.seeds:
            build_sessions(s)
            print(f"seed {s}: sessions written", flush=True)
        return
    ok = True
    for s in args.seeds:
        L = Latent(s)
        rows = L.build()
        print(f"seed {s}: cache written; latent k {L.k} d {L.d} lambda {L.lam}", flush=True)
        if args.check:
            ok &= check_g0b(s, rows)
    if args.check:
        print("ALL_OK" if ok else "FAILED")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
