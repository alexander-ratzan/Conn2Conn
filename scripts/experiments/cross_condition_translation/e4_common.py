"""Spec v4 E4 shared pieces (`e4_common`, not `common`: E2's module of that name is imported here): cells, data access,

Cells (per seed): single source a -> b (56 ordered FC pairs), SC only -> b (8), the other 7 FC conditions -> b (8);
FC cells with and without SC. Point metrics on decoded edges (E2's `score`: Pearson r, demeaned r, average rank,
top-1, as v3:E0 / E2); distribution metrics in code space for one observed target per subject: energy score
(E||X - y|| - 0.5 E||X - X'||, lower is better), 90 % coverage and the spread-error correlation.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
E2 = REPO_ROOT / "scripts" / "experiments" / "joint_flow_matching"
sys.path[:0] = [str(REPO_ROOT), str(E2)]
from common import CFG as E2CFG, SeedData, pairs, score  # noqa: E402  (E2)
from linear_arms import Scorer  # noqa: E402  (E2.0: decode codes, score against val / test edges)
from models.architectures.multiview_cca import maxvar, ridge_gcv  # noqa: E402,F401

CFG = yaml.safe_load((HERE / "config.yml").read_text())
FC = CFG["fc_views"]
VIEWS = ["SC"] + FC                       # E2.0's 9 views (L4)
E20_LINEAR = REPO_ROOT / E2CFG["records_dir"] / "linear"
OTHER7, SC_ONLY = "other7", "SC"


def kdir(base, k=None):
    """Per-code-size subfolder (results/, tables/, figures/): base / k<k>."""
    return base / f"k{k or CFG['k']}"


def cells():
    """(source, target, with_sc) for every evaluated cell."""
    out = []
    for b in FC:
        out.append((SC_ONLY, b, True))
        for a in FC + [OTHER7]:
            if a != b:
                out += [(a, b, False), (a, b, True)]
    return out


def sources_of(src, tgt):
    return [v for v in FC if v != tgt] if src == OTHER7 else ([] if src == SC_ONLY else [src])


def energy_metrics(X, Y):
    """X: (K, n, d) samples; Y: (n, d) one observed target per subject. Energy score, coverage, spread-error r."""
    Xt, Yt = torch.as_tensor(X, dtype=torch.float64), torch.as_tensor(Y, dtype=torch.float64)
    K = X.shape[0]
    es = []
    for i in range(X.shape[1]):
        x = Xt[:, i]
        d_xy = torch.linalg.norm(x - Yt[i], dim=1).mean()
        d_xx = torch.cdist(x, x).sum() / (K * (K - 1))
        es.append(float(d_xy - 0.5 * d_xx))
    mean = X.mean(0)
    q90 = np.quantile(np.linalg.norm(X - mean[None], axis=2), 0.9, axis=0)
    err = np.linalg.norm(Y - mean, axis=1)
    spread = X.var(axis=0, ddof=1).sum(1)
    sr = float(np.corrcoef(spread, err ** 2)[0, 1]) if spread.std() > 0 else float("nan")
    return dict(energy_score=float(np.mean(es)), coverage90=float((err <= q90).mean()), spread_error_r=sr,
                pred_spread=float(spread.mean()), sq_error=float((err ** 2).mean()))
