"""Spec v4 E6: the four checks of the fingerprint embedding (E4's pooled map) for one seed. Test split reported, val
used for selection; family-aware E2 splits; k = 256 codes.

  1 retest     rest session 1 <-> session 2 (different days): copy (edges), copy (codes), embedding (codes x B). The map
               is fit across conditions only (combined rest + tasks), never on the sessions.
  2 confounds  (a) R^2 of the confound proxies (build_confounds.py) for the representation (rest codes / their
               embedding; subject fingerprint = mean of 8 codes / its embedding), fit on train, test R^2;
               (b) cross-condition identification (mean over 56 ordered pairs, code space) before and after removing
               the confound-explained part (regression fit on train) from codes and from the embedding.
  3 behaviour  ridge (penalty on val r) from rest edges, rest codes, rest embedding, fingerprint, fingerprint embedding
               to HCP cognition (total, fluid, crystallised composites, PMAT); test Pearson r.
  4 SC routes  four ways from SC to rest FC, all selected on val average rank against measured rest FC:
               (1) ridge SC codes -> rest's block-PCA codes; (2) multiview CCA (MAXVAR over SC + 8 FC conditions, as
               E4's B2 but (d, lambda) selected for SC -> rest): SC's latent encoding decoded with rest's decoder;
               (3) ridge -> the fingerprint (mean of the 8 conditions' codes); (4) ridge -> the fingerprint embedding;
               (5) E1's consensus CCA latent: MAXVAR over the 8 FC conditions only, consensus = mean of a subject's 8
               encodings, ridge SC -> consensus, rest's CCA decoder ((d, lambda) selected as the others; also at E1's
               fixed d 32, lambda 0.1). Ceiling per route: the subject's true target decoded the same way.
               Stage 1: pairing with the route's own target (codes; CCA: SC's vs rest's latent encoding); stage 2:
               decoded to edges against each condition's measured FC (rest is the main readout).
Also: B's closed form and the eigenvalues w of W (per-direction fingerprint share).

    python scripts/experiments/fingerprint_embedding/run_checks.py --seed 0   -> results/seed<S>.json
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from fe_common import (CONFOUNDS, COG, FC, HERE, K, UNRESTRICTED, SeedData, codes, fingerprint, ident, pooled_map,
                       ridge_fit, score)
from e4_common import CFG as E4CFG, maxvar  # noqa: E402
from run_e40c import ridge_path  # noqa: E402  (E4: one SVD, every penalty)

ALPHAS = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4]
K_SC = [64, 128, 256]
CONF = ["dvars", "gs_sd", "mean_fc", "icv", "parcel_vol"]
SPLITS = ("val", "test")
ALL = ("train",) + SPLITS


def retest(sd, B):
    rows = []
    for a, b in (("rest_S1", "rest_S2"), ("rest_S2", "rest_S1")):
        for split in SPLITS:
            za = sd.codes[a][sd.splits[split], :K].astype(np.float64)
            true, mu = sd.edges[(split, b)], sd.train_mean[b]
            preds = {"copy (edges)": sd.edges[(split, a)] - sd.train_mean[a] + mu,
                     "copy (codes)": sd.decode(b, za), "embedding": sd.decode(b, za @ B)}
            for m, p in preds.items():
                rows.append(dict(check="retest", source=a, target=b, split=split, model=m, **score(p, true, mu)))
    return rows


def confounds(sd, B, conf, Z):
    ids = sd.subject_ids
    C = {s: conf.loc[ids[sd.splits[s]], CONF].to_numpy(float) for s in ALL}
    mu, sdv = C["train"].mean(0), C["train"].std(0)
    C = {s: np.c_[np.ones(len(v)), (v - mu) / sdv] for s, v in C.items()}
    reps = {"codes": lambda z: z, "embedding": lambda z: z @ B}
    rows = []
    for name, f in reps.items():
        for what, get in (("rest", lambda Zs: Zs["rest"]), ("fingerprint", fingerprint)):
            Ytr = f(get(Z["train"]))
            beta = np.linalg.lstsq(C["train"], Ytr, rcond=None)[0]
            for split in SPLITS:
                Y = f(get(Z[split]))
                r2 = 1 - ((Y - C[split] @ beta) ** 2).sum() / ((Y - Ytr.mean(0)) ** 2).sum()
                rows.append(dict(check="confound_r2", representation=name, of=what, split=split, r2=float(r2)))
    for name, f in reps.items():
        E = {s: {v: f(Z[s][v]) for v in FC} for s in ALL}
        beta = {v: np.linalg.lstsq(C["train"], E["train"][v], rcond=None)[0] for v in FC}
        for split in SPLITS:
            for resid in (False, True):
                X = {v: E[split][v] - (C[split] @ beta[v] if resid else 0) for v in FC}
                ar, t1 = zip(*[ident(X[s], X[t]) for s in FC for t in FC if s != t])
                rows.append(dict(check="confound_ident", representation=name, residualised=resid, split=split,
                                 avg_rank=float(np.mean(ar)), top1=float(np.mean(t1))))
    return rows


def behaviour(sd, B, rest_edges, cog, Z):
    ids = sd.subject_ids
    feats = {"rest edges": {s: rest_edges[sd.splits[s]].astype(np.float64) for s in ALL},
             "rest codes": {s: Z[s]["rest"] for s in ALL},
             "rest embedding": {s: Z[s]["rest"] @ B for s in ALL},
             "fingerprint": {s: fingerprint(Z[s]) for s in ALL},
             "fingerprint embedding": {s: fingerprint(Z[s]) @ B for s in ALL}}
    rows = []
    for target in COG:
        y = {s: cog.loc[ids[sd.splits[s]], target].to_numpy(float) for s in ALL}
        ok = {s: np.isfinite(y[s]) for s in ALL}
        ytr = y["train"][ok["train"]]
        for fname, F in feats.items():
            Xtr = F["train"][ok["train"]]
            mx = Xtr.mean(0)
            path = ridge_fit(Xtr - mx, ((ytr - ytr.mean()) / ytr.std())[:, None], ALPHAS)
            r = {s: {a: float(np.corrcoef(((F[s][ok[s]] - mx) @ b).ravel(), y[s][ok[s]])[0, 1]) for a, b in path.items()} for s in SPLITS}
            a = max(r["val"], key=r["val"].get)
            for split in SPLITS:
                rows.append(dict(check="behaviour", target=target, features=fname, split=split, alpha_rel=a, r=r[split][a],
                                 n=int(ok[split].sum())))
    return rows


def sc_targets(sd, B, Z):
    targets = {"rest codes": lambda Zs: Zs["rest"], "fingerprint": fingerprint,
               "fingerprint embedding": lambda Zs: fingerprint(Zs) @ B}
    rows = []
    for tname, tf in targets.items():
        Ytr = tf(Z["train"])
        best = None
        for ksc in K_SC:
            X = {s: sd.code("SC", s, ksc).astype(np.float64) for s in ALL}
            for a, Bsc in ridge_path(X["train"], Ytr, ALPHAS).items():
                v = score(sd.decode("rest", X["val"] @ Bsc), sd.edges[("val", "rest")], sd.train_mean["rest"])["avg_rank"]
                if best is None or v > best[0]:
                    best = (v, ksc, a, Bsc, X)
        _, ksc, a, Bsc, X = best
        for split in SPLITS:
            for m in FC:                       # ceiling: the true target decoded the same way
                rows.append(dict(check="sc_ceiling", target=tname, against=m, split=split,
                                 **score(sd.decode(m, tf(Z[split])), sd.edges[(split, m)], sd.train_mean[m])))
        for split in SPLITS:
            P, own = X[split] @ Bsc, tf(Z[split])
            r2 = 1 - ((own - P) ** 2).sum() / ((own - Ytr.mean(0)) ** 2).sum()
            ar, t1 = ident(P, own)
            rows.append(dict(check="sc_target", target=tname, against="own target (codes)", split=split, k_sc=ksc,
                             alpha_rel=a, r2=float(r2), avg_rank=ar, top1_acc=t1))
            for m in FC:
                rows.append(dict(check="sc_target", target=tname, against=m, split=split, k_sc=ksc, alpha_rel=a,
                                 **score(sd.decode(m, P), sd.edges[(split, m)], sd.train_mean[m])))
    rows += sc_cca(sd, Z)
    rows += sc_consensus(sd, Z, select=True) + sc_consensus(sd, Z, select=False)
    return rows


def sc_consensus(sd, Z, select):
    """Route 5 (E1): MAXVAR over the 8 FC conditions (no SC); target = mean of a subject's 8 encodings; ridge SC ->
    consensus; decoded with each condition's CCA decoder. select=False: E1's latent (d 32, lambda 0.1)."""
    name = "CCA consensus" if select else "CCA consensus (E1 setting)"
    grid = [(d, float(l)) for d in [16, 32, 64, 128, 256] for l in E4CFG["maxvar_lambda"]] if select else [(32, 0.1)]
    X = {k_: {s: sd.code("SC", s, k_).astype(np.float64) for s in ALL} for k_ in K_SC}
    best = None
    for d, lam in grid:
        A, D, _ = maxvar({v: Z["train"][v] for v in FC}, d, lam)
        cons = {s: sum(Z[s][v] @ A[v] for v in FC) / len(FC) for s in ALL}
        for k_ in K_SC:
            for a, Bsc in ridge_path(X[k_]["train"], cons["train"], ALPHAS).items():
                v = score(sd.decode("rest", X[k_]["val"] @ Bsc @ D["rest"]), sd.edges[("val", "rest")], sd.train_mean["rest"])["avg_rank"]
                if best is None or v > best[0]:
                    best = (v, d, lam, k_, a, Bsc, D, cons)
    _, d, lam, k_, a, Bsc, D, cons = best
    rows = []
    for split in SPLITS:
        P = X[k_][split] @ Bsc
        r2 = 1 - ((cons[split] - P) ** 2).sum() / ((cons[split] - cons["train"].mean(0)) ** 2).sum()
        ar, t1 = ident(P, cons[split])
        hp = dict(d=d, lambda_rel=lam, k_sc=k_, alpha_rel=a)
        rows.append(dict(check="sc_target", target=name, against="own target (codes)", split=split, r2=float(r2), avg_rank=ar, top1_acc=t1, **hp))
        for m in FC:
            rows.append(dict(check="sc_target", target=name, against=m, split=split, **hp,
                             **score(sd.decode(m, P @ D[m]), sd.edges[(split, m)], sd.train_mean[m])))
            rows.append(dict(check="sc_ceiling", target=name, against=m, split=split,
                             **score(sd.decode(m, cons[split] @ D[m]), sd.edges[(split, m)], sd.train_mean[m])))
    return rows


def sc_cca(sd, Z):
    """Route 2: MAXVAR over SC + the 8 FC conditions (k = 256 codes each); SC encoded into the latent, decoded with each
    condition's decoder; (d, lambda) on val average rank of SC -> rest against measured rest FC."""
    V = {s: dict(Z[s], SC=sd.code("SC", s, K).astype(np.float64)) for s in ALL}
    best = None
    for d in [16, 32, 64, 128, 256]:
        for lam in E4CFG["maxvar_lambda"]:
            A, D, _ = maxvar(V["train"], d, float(lam))
            v = score(sd.decode("rest", V["val"]["SC"] @ A["SC"] @ D["rest"]), sd.edges[("val", "rest")], sd.train_mean["rest"])["avg_rank"]
            if best is None or v > best[0]:
                best = (v, d, float(lam), A, D)
    _, d, lam, A, D = best
    rows = []
    gtr = V["train"]["rest"] @ A["rest"]
    for split in SPLITS:
        g_sc, g_rest = V[split]["SC"] @ A["SC"], V[split]["rest"] @ A["rest"]
        r2 = 1 - ((g_rest - g_sc) ** 2).sum() / ((g_rest - gtr.mean(0)) ** 2).sum()
        ar, t1 = ident(g_sc, g_rest)
        rows.append(dict(check="sc_target", target="CCA latent", against="own target (codes)", split=split, d=d, lambda_rel=lam,
                         r2=float(r2), avg_rank=ar, top1_acc=t1))
        for m in FC:
            rows.append(dict(check="sc_target", target="CCA latent", against=m, split=split, d=d, lambda_rel=lam,
                             **score(sd.decode(m, g_sc @ D[m]), sd.edges[(split, m)], sd.train_mean[m])))
            rows.append(dict(check="sc_ceiling", target="CCA latent", against=m, split=split,
                             **score(sd.decode(m, V[split][m] @ A[m] @ D[m]), sd.edges[(split, m)], sd.train_mean[m])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    t0 = time.time()
    sd = SeedData(a.seed, "main")
    B, info = pooled_map(sd)
    print(f"seed {a.seed}: B closed form max |diff| {info['closed_form_max_abs_diff']:.2e}; w > 0.5 in "
          f"{int((info['w'] > 0.5).sum())} / {K} directions", flush=True)
    Z = {s: codes(sd, s) for s in ALL}
    conf = pd.read_csv(CONFOUNDS).set_index("subject")
    conf = conf.fillna(conf.mean())
    cog = pd.read_csv(UNRESTRICTED, usecols=["Subject"] + COG).set_index("Subject")
    rows = retest(sd, B)
    print(f"retest done ({time.time() - t0:.0f} s)", flush=True)
    rows += confounds(sd, B, conf, Z)
    print(f"confounds done ({time.time() - t0:.0f} s)", flush=True)
    rows += sc_targets(sd, B, Z)
    print(f"SC targets done ({time.time() - t0:.0f} s)", flush=True)
    from common import load_views          # E2 loader (raw rest edges for every subject)
    data, splits, ids = load_views(a.seed)
    assert np.array_equal(ids, sd.subject_ids) and all(np.array_equal(splits[s], sd.splits[s]) for s in ALL)
    rows += behaviour(sd, B, data["rest"], cog, Z)
    print(f"behaviour done ({time.time() - t0:.0f} s)", flush=True)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / f"seed{a.seed}.json").write_text(json.dumps(dict(seed=a.seed, k=K, closed_form_max_abs_diff=info["closed_form_max_abs_diff"],
                                                            w=info["w"].tolist(), rows=rows), default=float))


if __name__ == "__main__":
    main()
