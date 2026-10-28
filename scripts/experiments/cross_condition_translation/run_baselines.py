"""Spec v4 E4.0: closed-form baselines for every cross-condition cell, one seed.

  B0  identity: the source's deviation from its train mean copied into the target (edge space, as E2.0); multi-source:
      mean of the sources' deviations
  B1  block-PCA ridge per cell (E2.0's L1: penalty by GCV on train): source codes -> target codes; multi-source: the
      sources' codes concatenated; +SC: SC codes appended (SC code size chosen on val); SC only: SC codes alone
  B2  9-view multiview CCA (E2.0's L4: MAXVAR over SC + 8 FC conditions, (d, lambda) on mean val demeaned r over the 72
      pairs): encode every available view, average the encodings, decode to the target
  B3  linear-Gaussian: B1's mean + the cell's residual covariance (cross-fitted train residuals); 64 samples per subject
References: each target's own k-code reconstruction (ceiling); rest session 1 <-> 2 copied (retest ceiling).
G0: B1 and B2 without SC reproduce E2.0's stored L1 / L4 rows at k (single source, and SC -> target for B2).

    python scripts/experiments/cross_condition_translation/run_baselines.py --seed 0 [--k 512]   # -> results/baselines_seed0[_k512].json
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from e4_common import (CFG, E20_LINEAR, FC, HERE, OTHER7, SC_ONLY, VIEWS, Scorer, SeedData, cells, energy_metrics, maxvar,
                    pairs, ridge_gcv, score, sources_of, suffix)

SPLITS = ("val", "test")


def crossfit_ridge(X, Y, a_rel, n_folds, seed):
    """Cross-fitted train predictions with the full fit's penalty (rows look like test-time predictions)."""
    idx = np.random.default_rng(seed).permutation(len(X))
    out = np.zeros_like(Y)
    for f in np.array_split(idx, n_folds):
        tr = np.setdiff1d(idx, f)
        B, _ = ridge_gcv(X[tr], Y[tr], [a_rel])
        out[f] = X[f] @ B
    return out


def run(seed, k=None):
    t0 = time.time()
    sd = SeedData(seed, "main")
    sc = Scorer(sd)
    k = k or CFG["k"]
    Z = {s: {v: sd.code(v, s, k).astype(np.float64) for v in VIEWS} for s in ("train",) + SPLITS}
    Zsc = {s: sd.code("SC", s, max(CFG["k_sc"])).astype(np.float64) for s in ("train",) + SPLITS}
    rows, dist, hps = [], [], {}
    add = lambda model, src, tgt, with_sc, split, m, **kw: rows.append(
        dict(seed=seed, model=model, source=src, target=tgt, with_sc=with_sc, split=split, **kw, **m))

    # references
    for split in SPLITS:
        for v in FC:
            add("ceiling", v, v, False, split, sc(v, split, Z[split][v]))
        for a, b in (("rest_S1", "rest_S2"), ("rest_S2", "rest_S1")):
            pred = sd.train_mean[b] + (sd.edges[(split, a)] - sd.train_mean[a])
            add("retest", a, b, False, split, score(pred, sd.edges[(split, b)], sd.train_mean[b]))

    # B0 identity
    for split in SPLITS:
        for src, tgt, with_sc in cells():
            if with_sc or src == SC_ONLY:
                continue
            dev = np.mean([sd.edges[(split, a)] - sd.train_mean[a] for a in sources_of(src, tgt)], axis=0)
            add("B0", src, tgt, False, split, sc(tgt, split, pred_edges=sd.train_mean[tgt] + dev))
    print(f"seed {seed}: references + B0 ({time.time() - t0:.0f} s)", flush=True)

    # B1 ridge (+SC) and B3 linear-Gaussian on B1's mean
    for src, tgt, with_sc in cells():
        srcs = sources_of(src, tgt)
        X = {s: np.concatenate([Z[s][a] for a in srcs], axis=1) if srcs else np.zeros((len(Z[s][tgt]), 0)) for s in Z}
        Y = Z["train"][tgt]
        best = None
        for ksc in (CFG["k_sc"] if with_sc else [0]):
            Xk = {s: np.concatenate([X[s], Zsc[s][:, :ksc]], axis=1) for s in X}
            B, a_rel = ridge_gcv(Xk["train"], Y, CFG["ridge_alphas"])
            v = sc.fast_dr(tgt, "val", Xk["val"] @ B)
            if best is None or v > best[0]:
                best = (v, ksc, a_rel, B, Xk)
        _, ksc, a_rel, B, Xk = best
        name = "B1+SC" if with_sc else "B1"
        hps[f"{name}|{src}|{tgt}"] = dict(k_sc=ksc, alpha_rel=a_rel)
        resid = Y - crossfit_ridge(Xk["train"], Y, a_rel, CFG["crossfit_folds"], seed)
        cov = np.cov(resid, rowvar=False)
        rng = np.random.default_rng(seed)
        for split in SPLITS:
            P = Xk[split] @ B
            add(name, src, tgt, with_sc, split, sc(tgt, split, P), k_sc=ksc)
            S = P[None] + rng.multivariate_normal(np.zeros(k), cov, size=(CFG["n_samples"], len(P)))
            dist.append(dict(seed=seed, model="B3+SC" if with_sc else "B3", source=src, target=tgt, with_sc=with_sc,
                             split=split, **energy_metrics(S, Z[split][tgt])))
    print(f"seed {seed}: B1 / B3 ({time.time() - t0:.0f} s)", flush=True)

    # B2 multiview CCA over the 9 views (E2.0 L4 selection), averaged encodings for any input subset
    P72 = pairs(VIEWS)
    best = None
    for d in [int(x) for x in CFG["maxvar_d"] if x <= k]:
        for lam in CFG["maxvar_lambda"]:
            A, D, _ = maxvar(Z["train"], d, float(lam))
            val = np.mean([sc.fast_dr(b, "val", Z["val"][a] @ A[a] @ D[b]) for a, b in P72])
            if best is None or val > best[0]:
                best = (val, d, float(lam), A, D)
    _, d, lam, A, D = best
    hps["B2"] = dict(d=d, lambda_rel=lam, val_mean_dr=best[0])
    for split in SPLITS:
        for src, tgt, with_sc in cells():
            views = sources_of(src, tgt) + (["SC"] if with_sc else [])
            g = np.mean([Z[split][v] @ A[v] for v in views], axis=0)
            add("B2+SC" if with_sc and src != SC_ONLY else "B2", src, tgt, with_sc, split, sc(tgt, split, g @ D[tgt]))
    print(f"seed {seed}: B2 d {d} lambda {lam} ({time.time() - t0:.0f} s)", flush=True)

    # G0: reproduce E2.0's L1 / L4 at k
    e20 = pd.read_csv(E20_LINEAR / f"seed{seed}_main.csv")
    e20 = e20[(e20.k == k) & e20.ref.isin(["L1", "L4"])].set_index(["ref", "source", "target", "split"])
    mine = pd.DataFrame(rows)
    diffs = {"L1": [], "L4": []}
    for r in mine.itertuples():
        if r.model == "B1" and r.source in FC:
            key = ("L1", r.source, r.target, r.split)
        elif r.model == "B2" and (r.source in FC or r.source == SC_ONLY):
            key = ("L4", r.source, r.target, r.split)
        else:
            continue
        if key in e20.index:
            ref = e20.loc[key]
            diffs[key[0]].append(max(abs(r.demeaned_pearson - ref.demeaned_pearson), abs(r.avg_rank - ref.avg_rank)))
    g0 = {f"{k_}_max_abs_diff": float(max(v)) if v else None for k_, v in diffs.items()}
    g0.update({f"{k_}_n": len(v) for k_, v in diffs.items()})
    g0["pass"] = bool(all(v and max(v) < CFG["gates"]["g0_tol"] for v in diffs.values()))
    print(f"seed {seed}: G0 {'PASS' if g0['pass'] else 'FAIL'} {g0}", flush=True)

    out = dict(seed=seed, k=k, g0=g0, hyperparams=hps, rows=rows, dist=dist, seconds=time.time() - t0)
    (HERE / "results").mkdir(exist_ok=True)
    name = f"baselines_seed{seed}{suffix(k)}.json"
    (HERE / "results" / name).write_text(json.dumps(out, default=float))
    print(f"wrote results/{name} ({len(rows)} rows, {len(dist)} distribution rows) in "
          f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--k", type=int, default=None, help="block FC codes per condition (default config k); results get a _k<k> suffix")
    a = ap.parse_args()
    run(a.seed, a.k)
