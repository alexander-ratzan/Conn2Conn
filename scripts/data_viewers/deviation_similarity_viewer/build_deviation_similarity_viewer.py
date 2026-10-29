"""Deviation similarity viewer: for any pair of FC conditions (a, b), how similar is a subject's deviation from the
condition mean in a to their own deviation in b, relative to other subjects' deviations in b?

Same quantities as scripts/notebooks/EDA/FC_deviation_similarity.py (Glasser, subjects with every condition, deviation
d_i^a = FC_i^a - mean_a, correlation over edges), whose loaders this builder reuses:
  C_ab[i, j] = corr(d_i^a, d_j^b)       (n x n; C_ba = C_ab^T)
  own r      C_ab[i, i]
  others     mean / SD / max of C_ab[i, j], j != i
  rank       fraction of other subjects j with C_ab[i, j] < C_ab[i, i]  (identification; average over i = average rank)
Packed per ordered pair: those per-subject values for all n subjects, and histograms of own r and of all off-diagonal
r. Per unordered pair: C_ab on a fixed random sample of --n-sub subjects (default 48; the full n x n matrices would be
~60 MB). Per condition: the deviation connectomes d_i^a of the sampled subjects (int8 with a scale per connectome,
~3 MB per condition at 48 subjects), so clicking a matrix entry shows the two deviations side by side.
The overview table (average rank, within-subject r, top-1) is checked against the EDA script's CSV when present.

Output, next to this file (generated, not tracked): deviation_similarity_viewer.html and its data cache.
Loads ~2 GB of FC: run through sbatch (CPU), not on the login node.

    python scripts/data_viewers/deviation_similarity_viewer/build_deviation_similarity_viewer.py [--from-cache] [--smoke]
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "main.py").exists())
EDA = REPO_ROOT / "scripts" / "notebooks" / "EDA"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EDA))
from FC_deviation_similarity import CONDS, LABEL, OUT, P, folder, load, zrows  # noqa: E402
from scripts.data_viewers.viewer_kit import Viewer, int16_unit, packed, round_list  # noqa: E402


def int8_rows(X):
    """Rows as int8 codes with one scale per row (x ~= q * scale): enough for a 256-colour display."""
    scale = np.abs(X).max(1) / 127.0
    scale[scale == 0] = 1.0
    return np.round(X / scale[:, None]).astype(np.int8), scale

N_BINS, SEED = 60, 0


def subject_stats(C):
    """Row-wise statistics of C (n x n, rows = subjects in condition a, columns = subjects in condition b)."""
    n = len(C)
    own = np.diag(C).astype(np.float64)
    off = C.astype(np.float64).copy()
    np.fill_diagonal(off, np.nan)
    rank = (C < own[:, None]).sum(1) / (n - 1)
    return dict(own=own, other_mean=np.nanmean(off, 1), other_sd=np.nanstd(off, 1, ddof=1), other_max=np.nanmax(off, 1),
                rank=rank), off[np.isfinite(off)]


def pair_set(st, off):
    lo, hi = float(min(st["own"].min(), off.min())), float(max(st["own"].max(), off.max()))
    edges = np.linspace(lo, hi, N_BINS + 1)
    h_own = np.histogram(st["own"], edges)[0] / len(st["own"])
    h_off = np.histogram(off, edges)[0] / len(off)
    return dict(**{k: packed(int16_unit(v)) for k, v in st.items()}, hist_lo=lo, hist_hi=hi,
                hist_own=round_list(h_own, 6), hist_other=round_list(h_off, 6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sub", type=int, default=48, help="subjects in the matrix sample (connectomes: ~0.5 MB each x 8)")
    Viewer.add_args(ap)
    args = ap.parse_args()
    viewer = Viewer(__file__)
    if args.from_cache:
        viewer.write(*viewer.load_cache(), cache=False, smoke=args.smoke)
        return

    ids = None
    for c in CONDS:
        s = set(np.load(folder(c) / "subject_ids.npy").tolist())
        ids = s if ids is None else ids & s
    ids = np.array(sorted(ids))
    n = len(ids)
    sub = np.sort(np.random.default_rng(SEED).choice(n, min(args.n_sub, n), replace=False))
    Z, sets = {}, {}
    for c in CONDS:
        X = load(folder(c), ids)
        dev = X - X.mean(0)
        q, scale = int8_rows(dev[sub])
        sets[f"X:{c}"] = dict(q=packed(q), scale=round_list(scale))
        Z[c] = zrows(dev)
        print(f"loaded {c}", flush=True)
    E = Z[CONDS[0]].shape[1]

    overview = []
    for a, b in combinations(CONDS, 2):
        C = Z[a] @ Z[b].T / E
        for x, y, M in ((a, b, C), (b, a, C.T)):
            st, off = subject_stats(M)
            sets[f"{x}|{y}"] = pair_set(st, off)
            overview.append(dict(a=x, b=y, avg_rank=st["rank"].mean(), self_mean=st["own"].mean(),
                                 self_sd=st["own"].std(ddof=1), other_mean=st["other_mean"].mean(),
                                 top1=float((M.argmax(1) == np.arange(n)).mean())))
        sets[f"M:{a}|{b}"] = dict(C=packed(int16_unit(C[np.ix_(sub, sub)])))
        print(f"{a} x {b}", flush=True)
    ov = pd.DataFrame(overview)

    ref = OUT / f"deviation_similarity_pairs_{P}.csv"
    if ref.exists():
        m = ov.merge(pd.read_csv(ref), on=["a", "b"], suffixes=("", "_eda"))
        worst = max(float((m[c] - m[f"{c}_eda"]).abs().max()) for c in ("avg_rank", "self_mean", "top1"))
        print(f"overview vs the EDA script's CSV: max |diff| {worst:.2e}")
        if worst > 1e-3:
            raise ValueError(f"overview disagrees with {ref} (max |diff| {worst})")

    n_parcels = int((1 + np.sqrt(1 + 8 * E)) // 2)
    assert n_parcels * (n_parcels - 1) // 2 == E
    meta = dict(parcellation=P, n=n, n_parcels=n_parcels, conds=CONDS, labels=LABEL, subject_ids=ids.tolist(), sub=sub.tolist(),
                overview={f"{r.a}|{r.b}": dict(avg_rank=r.avg_rank, self_mean=r.self_mean, self_sd=r.self_sd,
                                               other_mean=r.other_mean, top1=r.top1) for r in ov.itertuples()})
    viewer.write(meta, sets, smoke=args.smoke)


if __name__ == "__main__":
    main()
