"""Spec v4 E4 control for the shared-map linear flow (fm_shared, 2026-10-28): closed-form ridge with the flow's sharing.

  B1-pooled      one ridge map for all 56 single-source FC -> FC pairs: the train subjects' (source codes, target codes)
                 pairs stacked over every pair (codes are centred per condition, so no intercepts); penalty on the mean
                 val average rank over the 56 cells (D9)
  B1-pooled-tgt  one map per target, pooled over its 7 sources; penalty per target on its 7 cells' mean val average rank
Same code space, scoring and penalty grid as E4.0c's B1 (run_e40c.py); no SC. If B1-pooled matches fm_shared, the flow's
gain over per-pair ridge comes from sharing the map across pairs, not from flow matching.
The same map without a regression (closed form, generalized eigendecomposition): models/architectures/pooled_map.py
(pooled_map.md section 5); this script stays as E4's recorded fit (penalty grid, per-target variant).

    python scripts/experiments/cross_condition_translation/run_pooled_ridge.py --seed 0 [--k 256]
        -> results/pooled_ridge/k<k>/seed<S>.json
"""
import argparse
import json
import time

import numpy as np

from e4_common import CFG, FC, HERE
from run_e40c import SPLITS, Space, ridge_path


def run(seed, k):
    t0 = time.time()
    S = Space(seed, k)
    alphas = [float(a) for a in CFG["ridge_alphas"]]
    pairs = [(s, t) for t in FC for s in FC if s != t]
    rows, hps = [], {}

    def fit(cells):
        X = np.concatenate([S.Z["train"][s] for s, _ in cells])
        Y = np.concatenate([S.Z["train"][t] for _, t in cells])
        path = ridge_path(X, Y, alphas)
        val = {a: np.mean([S.val_rank(t, S.Z["val"][s] @ B) for s, t in cells]) for a, B in path.items()}
        a = max(val, key=val.get)
        return path[a], a, val[a]

    def emit(name, cells, B, a):
        for s, t in cells:
            for split in SPLITS:
                rows.append(dict(seed=seed, k=str(k), model=name, source=s, target=t, with_sc=False, split=split, alpha_rel=a,
                                 **S.score(t, split, S.Z[split][s] @ B)))

    B, a, v = fit(pairs)
    hps["B1-pooled"] = dict(alpha_rel=a, val_avg_rank=v)
    emit("B1-pooled", pairs, B, a)
    print(f"seed {seed}: B1-pooled alpha {a:g}, val avg rank {v:.4f} ({time.time() - t0:.0f} s)", flush=True)
    for t in FC:
        cells = [(s, tt) for s, tt in pairs if tt == t]
        B, a, v = fit(cells)
        hps[f"B1-pooled-tgt|{t}"] = dict(alpha_rel=a, val_avg_rank=v)
        emit("B1-pooled-tgt", cells, B, a)
    print(f"seed {seed}: B1-pooled-tgt done ({time.time() - t0:.0f} s)", flush=True)
    out = HERE / "results" / "pooled_ridge" / f"k{k}"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"seed{seed}.json").write_text(json.dumps(dict(seed=seed, k=k, hyperparams=hps, rows=rows), default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--k", type=int, default=256)
    a = ap.parse_args()
    run(a.seed, a.k)


if __name__ == "__main__":
    main()
