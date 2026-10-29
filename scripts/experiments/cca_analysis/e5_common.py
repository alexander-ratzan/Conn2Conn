"""Spec v4 E5 shared pieces (`e5_common`, not `common`: E2's module of that name is imported here): representations,
scoring, ridge and the conditional Gaussian.

Representations (each view encoded separately; codes are train-centred, float64; SC keeps its own basis throughout):
  R0 nodal   eigenvectors V (360 x q) of the view's mean train connectome (PCA of its rows); codes vec((X - Xbar) V),
             stored component-major so the first 360 q' columns are the q' < q code; decoded by the symmetric
             projection Xhat = Xbar + F V^T + V F^T - V V^T F V^T
  R1 cond    each view's own train PCA (sorted columns of the full-rank basis)
  R2 block   one PCA of the stacked, per-condition-centred FC blocks, no scaling
  R3 blockva E2's code space (D2: each block divided by its total norm), read from E2's per-seed cache
  full       each view's own PCA at full rank (rank <= n_train - 1): fits on these codes equal fits on the edges
Selection score (D9, as E4.0c): average rank of the raw correlation matrix, rows = true val subjects. For affine
decoders (R1-R3, full) it is computed in code space (`FastScorer`, algebraically exact); R0 decodes explicitly.
"""
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
E2 = REPO_ROOT / "scripts" / "experiments" / "joint_flow_matching"
sys.path[:0] = [str(REPO_ROOT), str(E2)]
from common import SeedData, load_views, pairs, score  # noqa: E402,F401  (E2)
from models.architectures.multiview_cca import maxvar  # noqa: E402,F401
from models.eval.metrics import compute_corr_matrix, corr_avg_rank  # noqa: E402

CFG = yaml.safe_load((HERE / "config.yml").read_text())
FC = CFG["fc_views"]
VIEWS = ["SC"] + FC
PAIRS = pairs(VIEWS)                       # 72 ordered pairs
OUT = Path(CFG["out_dir"])
SPLITS = ("train", "val", "test")
N_NODES = 360
TRIU = np.triu_indices(N_NODES, k=1)
ALPHAS = [float(a) for a in CFG["ridge_alphas"]]


def block(a, b):
    """Pair block: FC -> FC, SC -> FC, FC -> SC."""
    return "SC -> FC" if a == "SC" else "FC -> SC" if b == "SC" else "FC -> FC"


def to_matrix(E):
    """(n, 64620) upper triangles -> (n, 360, 360) symmetric matrices, zero diagonal."""
    M = np.zeros((len(E), N_NODES, N_NODES), dtype=np.float64)
    M[:, TRIU[0], TRIU[1]] = E
    return M + M.transpose(0, 2, 1)


# ---------------------------------------------------------------- scoring
def avg_rank_cc(cc):
    return float(corr_avg_rank(cc=cc))


class FastScorer:
    """Raw Pearson correlations between true rows T (n, E) and affine predictions mu + Z W^T, without decoding.
    Z may use the first r' <= r columns of W."""

    def __init__(self, T, mu, W):
        T, mu = np.asarray(T, dtype=np.float64), np.asarray(mu, dtype=np.float64)
        self.E = T.shape[1]
        self.TW = T @ W
        self.Tmu = T @ mu
        tbar = T.mean(1)
        self.tbar, self.tss = tbar, (T ** 2).sum(1) - self.E * tbar ** 2
        self.W1, self.Wmu, self.WW = W.sum(0), W.T @ mu, W.T @ W
        self.mu_sum, self.mumu = mu.sum(), mu @ mu

    def cc(self, Z):
        r = Z.shape[1]
        dot = self.Tmu[:, None] + self.TW[:, :r] @ Z.T
        psum = self.mu_sum + Z @ self.W1[:r]
        pss = self.mumu + 2 * Z @ self.Wmu[:r] + np.einsum("ij,jk,ik->i", Z, self.WW[:r, :r], Z)
        cov = dot - self.tbar[:, None] * psum[None, :]
        return cov / np.sqrt(self.tss[:, None] * (pss - psum ** 2 / self.E)[None, :])


# ---------------------------------------------------------------- linear maps
def ridge_path(X, Y, rel_alphas=ALPHAS):
    """{relative penalty: B} for multi-output ridge without intercept (one SVD); penalty = rel x mean eigenvalue."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    UtY = U.T @ Y
    return {a: Vt.T @ ((s / (s ** 2 + a * np.mean(s ** 2)))[:, None] * UtY) for a in rel_alphas}


def ridge_abs(X, Y, a):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt.T @ ((s / (s ** 2 + a))[:, None] * (U.T @ Y))


def lw_penalty(X, Y):
    """Conditional mean of the Ledoit-Wolf-shrunk joint Gaussian of (x, y) (train-centred codes) as a ridge.

    Shrunk covariance (1 - rho) S + rho mu I, so E[y | x] = x (S_xx + rho mu / (1 - rho) I)^-1 S_xy: ridge with penalty
    a = n rho mu / (1 - rho) in sum-of-squares units. rho is sklearn's Ledoit-Wolf intensity, computed from the n x n Gram
    matrix. Returns (a, rho)."""
    J = np.concatenate([X, Y], axis=1)
    n, p = J.shape
    K = J @ J.T
    sq = np.diag(K).copy()
    mu = sq.sum() / (n * p)
    delta_ = np.square(K).sum() / n ** 2
    beta = ((np.square(sq).sum() / n) - delta_) / (p * n)
    delta = (delta_ - 2 * mu * sq.sum() / n + p * mu ** 2) / p
    rho = min(beta, delta) / delta
    return n * rho * mu / (1 - rho), float(rho)


# ---------------------------------------------------------------- representations
class Rep:
    """Codes Z[split][view] (n, r_max) and decoders of one representation."""

    def __init__(self, name, sizes, Z, mu, T, W=None, nodal=None):
        self.name, self.sizes, self.Z, self.mu, self.T = name, sizes, Z, mu, T
        self.W, self.nodal = W, nodal          # W[view]: (E, r_max) float64; nodal[view]: (Xbar (360, 360), V (360, q_max))
        self.fast = {}

    def code(self, view, split, size):
        return self.Z[split][view][:, :self.ncols(size)]

    def ncols(self, size):
        return N_NODES * size if self.nodal is not None else size

    def decode(self, view, Z):
        if self.nodal is None:
            return self.mu[view] + Z @ self.W[view][:, :Z.shape[1]].T
        Xbar, V = self.nodal[view]
        q = Z.shape[1] // N_NODES
        V = V[:, :q]
        out = np.empty((len(Z), len(TRIU[0])))
        for s in range(0, len(Z), 64):
            F = Z[s:s + 64].reshape(-1, q, N_NODES).transpose(0, 2, 1)            # (m, 360, q)
            G = F @ V.T
            C = np.einsum("iq,miz->mqz", V, F)                                     # V^T F (m, q, q)
            X = Xbar + G + G.transpose(0, 2, 1) - V @ C @ V.T
            out[s:s + 64] = X[:, TRIU[0], TRIU[1]]
        return out

    def val_cc(self, view, Z):
        """Raw correlation matrix, rows = true val subjects, columns = predictions."""
        if self.nodal is not None:
            return compute_corr_matrix(self.T[("val", view)], self.decode(view, Z))
        if view not in self.fast:
            self.fast[view] = FastScorer(self.T[("val", view)], self.mu[view], self.W[view])
        return self.fast[view].cc(Z)

    def val_rank(self, view, Z):
        return avg_rank_cc(self.val_cc(view, Z))

    def test_score(self, view, Z=None, edges=None):
        pred = self.decode(view, Z) if edges is None else edges
        return score(pred, self.T[("test", view)], self.mu[view])


def own_basis(Xtr):
    """Full-rank orthonormal basis of the train rows (E, r), columns sorted by variance (descending)."""
    mu = Xtr.mean(0)
    Xc = Xtr - mu
    w, U = np.linalg.eigh(Xc @ Xc.T)
    keep = w > 1e-9 * w.max()                       # E4.0c's rank rule
    w, U = w[keep][::-1], U[:, keep][:, ::-1]
    return mu, (Xc.T @ U) / np.sqrt(w)


def pooled_basis(blocks, k):
    """Top-k right singular vectors of the stacked (already centred) blocks."""
    S = np.vstack(blocks)
    w, U = np.linalg.eigh(S @ S.T)
    w, U = w[::-1][:k], U[:, ::-1][:, :k]
    return (S.T @ U) / np.sqrt(w)


def build_reps(seed, log=print):
    """All representations for one seed, plus the E2 cache (for G0a). Returns ({name: Rep}, SeedData, splits)."""
    data, splits, _ = load_views(seed)
    sd = SeedData(seed, "main")
    T = {(s, v): data[v][splits[s]].astype(np.float64) for s in ("val", "test") for v in VIEWS}
    g0a = max(float(np.abs(T[(s, v)] - sd.edges[(s, v)]).max()) for s in ("val", "test") for v in VIEWS)
    tr = splits["train"]
    mu = {v: data[v][tr].astype(np.float64).mean(0) for v in VIEWS}
    kmax = max(CFG["representations"]["k"])
    reps = {}

    # full rank and R1 (own bases)
    Wf, Zf = {}, {s: {} for s in SPLITS}
    for v in VIEWS:
        X = data[v].astype(np.float64)
        _, Wf[v] = own_basis(X[tr])
        for s in SPLITS:
            Zf[s][v] = (X[splits[s]] - mu[v]) @ Wf[v]
    reps["full"] = Rep("full", ["full"], Zf, mu, T, W=Wf)
    reps["R1"] = Rep("R1", CFG["representations"]["k"], Zf, mu, T, W=Wf)
    log(f"seed {seed}: full rank / R1 (ranks {sorted({w.shape[1] for w in Wf.values()})})")

    # R2 block PCA, unscaled (SC: own basis)
    Wp = pooled_basis([data[v][tr].astype(np.float64) - mu[v] for v in FC], kmax)
    W2 = {v: (Wp if v in FC else Wf[v][:, :kmax]) for v in VIEWS}
    Z2 = {s: {v: (data[v][splits[s]].astype(np.float64) - mu[v]) @ W2[v] for v in VIEWS} for s in SPLITS}
    reps["R2"] = Rep("R2", CFG["representations"]["k"], Z2, mu, T, W=W2)
    log(f"seed {seed}: R2")

    # R3 block PCA, variance-adjusted (E2's code space)
    cs = sd.cs
    W3 = {v: cs.scale[v] * cs.basis(v)[:, :kmax].astype(np.float64) for v in VIEWS}
    Z3 = {s: {v: sd.code(v, s, kmax).astype(np.float64) for v in VIEWS} for s in SPLITS}
    mu3 = {v: sd.train_mean[v].astype(np.float64) for v in VIEWS}
    reps["R3"] = Rep("R3", CFG["representations"]["k"], Z3, mu3, T, W=W3)

    # R0 nodal PCA of each view's mean train connectome
    qmax = max(CFG["representations"]["nodal_q"])
    nodal, Z0 = {}, {s: {} for s in SPLITS}
    for v in VIEWS:
        Xbar = to_matrix(mu[v][None])[0]
        _, _, Vt = np.linalg.svd(Xbar - Xbar.mean(0), full_matrices=False)
        V = Vt[:qmax].T
        nodal[v] = (Xbar, V)
        for s in SPLITS:
            E = data[v][splits[s]]
            F = np.empty((len(E), qmax * N_NODES))
            for i in range(0, len(E), 64):
                D = to_matrix(E[i:i + 64].astype(np.float64)) - Xbar
                F[i:i + 64] = (D @ V).transpose(0, 2, 1).reshape(len(D), -1)      # component-major
            Z0[s][v] = F
    reps["R0"] = Rep("R0", CFG["representations"]["nodal_q"], Z0, mu, T, nodal=nodal)
    log(f"seed {seed}: R0; G0a max |edges - E2 cache| = {g0a:.2e}")
    del data
    return reps, sd, splits, g0a
