"""windowed_fc_gain G7 shared pieces: training units per arm, the arm's closed-form fits (ridge means, Gaussian
covariances), and the G7.0 checks. Built on E2.1 (scripts/experiments/joint_flow_matching/e21_common.py, e21_run.py).

Routes (E2.1): `cca` = E1's 32-d CCA latent of the 256 block FC PCA codes; `pca` = the 256 block FC PCA codes.
Arms (training targets per train subject; val / test always the full targets, as E2.1):
    none                  full target                                  (= E2.1)
    aug_<level>           full target + the subject's part samples     (level: halves 2, quarters 4)
    token_<level>         as aug, flows also see a sample-type token (full / half / quarter) and sample as "full";
                          closed form: the Gaussian's covariance from full-target residuals only
    parts_<level>         part samples only
A unit = one training sample (subject, sample type); it carries all 8 conditions. Part samples exist for subjects
included in gate 6 for all 8 conditions; other train subjects keep their full target only (parts arms: dropped).

Closed form per arm and condition: ridge on each subject's mean over its units (equivalent to ridge on the duplicated
rows up to the penalty, and keeps GCV from leaving a subject's other units in), SC code size on val decoded demeaned r
(E2.1's ridge_cond rule); train predictions cross-fitted by subject (5 folds, as E2.1). Gaussian covariance: pooled
unit residuals (own) or full-unit residuals only (fullcov).
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
E2 = REPO_ROOT / "scripts" / "experiments" / "joint_flow_matching"
sys.path[:0] = [str(REPO_ROOT), str(E2)]
from e21_common import CACHE, FC, K_SC, Latent, crossfit, fast_dr, ridge_fit  # noqa: E402

PARTS_DIR = REPO_ROOT / "results" / "cache" / "experiments" / "windowed_fc_gain" / "parts"
G7_CACHE = REPO_ROOT / "results" / "cache" / "experiments" / "windowed_fc_gain" / "g7"
ARMS = ["none", "aug_halves", "token_halves", "parts_halves", "aug_quarters", "token_quarters", "parts_quarters"]
TYPE = {"full": 0, "halves": 1, "quarters": 2}
PART_ROWS = {"halves": [1, 2], "quarters": [3, 4, 5, 6]}
ROUTES = {"cca": "", "pca": "_pca"}


def load_parts(seed, route):
    """Per cohort subject: rebuilt full (8, n, d), parts {level: (8, n_parts, n, d)}, included in all 8 conditions (n,).
    cca route: codes @ E1's MAXVAR encoder A_m (recomputed from the seed's train codes, as E2.1's cache build)."""
    A = Latent(seed).A if route == "cca" else None
    full, parts, ok = [], {lv: [] for lv in PART_ROWS}, None
    for c in FC:
        z = np.load(PARTS_DIR / f"{c}.npz")
        codes = z[f"codes_seed{seed}"].astype(np.float64)                       # (7, n, 256)
        enc = codes @ A[c] if route == "cca" else codes
        full.append(enc[0])
        for lv, rows in PART_ROWS.items():
            parts[lv].append(enc[rows])
        ok = z["included"] if ok is None else ok & z["included"]
    return np.stack(full), {lv: np.stack(v) for lv, v in parts.items()}, ok


def units(arm, enc_full_train, parts_train, ok_train):
    """Training units of an arm: subject index (U,), sample type (U,), targets (8, U, d)."""
    n = enc_full_train.shape[1]
    subj, typ, tgt = [], [], []
    if not arm.startswith("parts"):
        subj.append(np.arange(n)); typ.append(np.zeros(n, int)); tgt.append(enc_full_train)
    if arm != "none":
        lv = arm.split("_")[1]
        P = parts_train[lv]                                                     # (8, k, n, d)
        idx = np.flatnonzero(ok_train)
        for k in range(P.shape[1]):
            subj.append(idx); typ.append(np.full(len(idx), TYPE[lv])); tgt.append(P[:, k, idx])
    return np.concatenate(subj), np.concatenate(typ), np.concatenate(tgt, axis=1)


class Route:
    """E2.1 cache for one seed and route (numpy), with E2.1's decoding / val scoring."""

    def __init__(self, seed, route):
        self.seed, self.route = seed, route
        self.z = z = np.load(CACHE / f"seed{seed}{ROUTES[route]}.npz")
        self.L = Latent.__new__(Latent)
        from common import SeedData
        self.L.sd = SeedData(seed, "main")
        self.L.D = {v: z["D"][i] for i, v in enumerate(FC)}
        self.enc = {s: z[f"enc_{s}"] for s in ("train", "val", "test")}
        self.sc = {s: z[f"sc_{s}"] for s in ("train", "val", "test")}
        self.d = self.enc["train"].shape[2]

    def val_dr(self, m, g):
        v = FC[m]
        return fast_dr(self.L.decode(m, g), self.L.sd.edges[("val", v)], self.L.sd.train_mean[v])

    def fit_arm(self, subj, typ, tgt):
        """Arm ridge means and covariances. Returns rc (train per unit, val, test; (8, ., d)), k per condition,
        covariances {own, fullcov}."""
        n = self.sc["train"].shape[0]
        cnt = np.bincount(subj, minlength=n)
        tr = np.flatnonzero(cnt > 0)
        mean_t = np.zeros((8, n, self.d))                                       # each subject's mean over its units
        for m in range(8):
            np.add.at(mean_t[m], subj, tgt[m])
        mean_t /= np.maximum(cnt, 1)[None, :, None]
        rc = {s: np.zeros((8, self.sc[s].shape[0], self.d)) for s in ("val", "test")}
        rc_sub = np.zeros((8, n, self.d))
        ks = []
        for m in range(8):
            fits = {k: ridge_fit(self.sc["train"][tr, :k], mean_t[m, tr]) for k in K_SC}
            best = max(K_SC, key=lambda k: self.val_dr(m, fits[k](self.sc["val"][:, :k])))
            ks.append(best)
            for s in ("val", "test"):
                rc[s][m] = fits[best](self.sc[s][:, :best])
            rc_sub[m, tr] = crossfit(self.sc["train"][tr, :best], mean_t[m, tr], seed=self.seed)
        rc["train"] = rc_sub[:, subj]                                           # per unit
        resid = tgt - rc["train"]
        cov = {"own": np.cov(resid.reshape(-1, self.d), rowvar=False)}
        if (typ == 0).any():
            cov["fullcov"] = np.cov(resid[:, typ == 0].reshape(-1, self.d), rowvar=False)
        return rc, ks, cov


def g7_checks(seed, route, R, full_rebuilt):
    """G7.0: (a) full FC rebuilt from the timeseries, encoded, matches the cached targets; (b) the `none` arm's ridge
    reproduces E2.1's cached ridge_cond predictions."""
    enc_all = np.concatenate([R.enc["train"], R.enc["val"], R.enc["test"]], axis=1)
    idx = np.concatenate([R.L.sd.splits[s] for s in ("train", "val", "test")])
    reb = full_rebuilt[:, idx]
    a = float(np.abs(reb - enc_all).max() / np.abs(enc_all).max())
    r_own, top1 = [], []
    for m in range(8):                                     # alignment: each rebuilt target matches its own cached one
        X = reb[m] - reb[m].mean(0)
        Y = enc_all[m] - enc_all[m].mean(0)
        C = (X / np.linalg.norm(X, axis=1, keepdims=True)) @ (Y / np.linalg.norm(Y, axis=1, keepdims=True)).T
        r_own.append(np.diag(C))
        top1.append(float((C.argmax(1) == np.arange(len(C))).mean()))
    subj, typ, tgt = units("none", R.enc["train"], None, None)
    rc, _, _ = R.fit_arm(subj, typ, tgt)
    b = max(float(np.abs(rc[s] - R.z[f"rc_{s}"]).max()) for s in ("train", "val", "test")) / float(np.abs(R.z["rc_test"]).max())
    return {"rebuilt_full_max_rel": a, "rebuilt_full_median_r": float(np.median(r_own)),
            "rebuilt_full_min_r": float(np.min(r_own)), "rebuilt_full_top1": float(np.min(top1)), "none_ridge_rel": b}
