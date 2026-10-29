"""Spec v4 E5.1: representations R0-R3, full-rank multiview CCA, H1 (all 72 view pairs) and H2 (other views ->
held-out view), with gate G0, for one seed (CPU).

  H1  per pair a -> b: ridge and conditional Gaussian on each PCA representation (size and penalty on val), CCA
      (full rank; CCA-256 on R3 codes as the reference): encode a, ridge decoder from a's train encodings to b's
      native edges (penalty per pair on val); copy (FC -> FC). (d, lambda) on the mean val average rank over the 72 pairs.
  H2  per held-out view b, sources = all other views ("other8") and, for FC targets, the other FC views ("other7"):
      CCA-agg (mean of the encodings) and CCA-cat (concatenated) with a ridge decoder; ridge on the concatenated codes
      of each PCA representation; best single CCA source (on val); copy (mean source deviation, FC sources).
  G0  a: loaded edges equal E2's cache; b: R3 ridge at k = 256 (D9 selection) reproduces E4.0c's B1 rows;
      c: full-rank CCA with E4.0c's selection and MAXVAR decoders reproduces E4.0c's B2 full-rank rows.
Writes OUT/results/seed<S>.json (rows, selections, gate) and OUT/h3/seed<S>.npz (CCA encodings + decoder penalties).

    python scripts/experiments/cca_analysis/run_seed.py --seed 0
"""
import argparse
import json
import time

import numpy as np

from e5_common import (ALPHAS, CFG, FC, OUT, PAIRS, REPO_ROOT, SPLITS, VIEWS, avg_rank_cc, block, build_reps, lw_penalty,
                       maxvar, ridge_abs, ridge_path)

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f} s] {msg}", flush=True)


def select_ridge(rep, X, Y, target, sizes, cols):
    """Best (val average rank) over sizes x penalties. X / Y: {split: full-width code arrays}; cols(size) -> (cx, cy)."""
    best = None
    for size in sizes:
        cx, cy = cols(size)
        for a_rel, B in ridge_path(X["train"][:, :cx], Y[:, :cy]).items():
            v = rep.val_rank(target, X["val"][:, :cx] @ B)
            if best is None or v > best[0]:
                best = (v, size, a_rel, B, cx)
    return best


def run(seed):
    reps, sd, splits, g0a = build_reps(seed, log)
    rows, sel = [], {}

    def add(hyp, model, rep, src, tgt, m, setting="pair", **kw):
        rows.append(dict(seed=seed, hyp=hyp, model=model, rep=rep, source=src, target=tgt, setting=setting,
                         block=block(src, tgt) if setting == "pair" else setting, **kw, **m))

    full = reps["full"]

    # ceilings: each view's own test reconstruction per representation and size
    for name in ("R0", "R1", "R2", "R3"):
        rep = reps[name]
        for size in rep.sizes:
            for v in VIEWS:
                add("ceiling", "reconstruction", name, v, v, rep.test_score(v, rep.code(v, "test", size)), size=size)
    log("ceilings")

    # copy (FC -> FC): the source's deviation from its train mean
    for a, b in PAIRS:
        if "SC" not in (a, b):
            add("H1", "copy", "edges", a, b, full.test_score(b, edges=full.mu[b] + full.T[("test", a)] - full.mu[a]))

    # H1: ridge and conditional Gaussian on each PCA representation
    for name in ("R0", "R1", "R2", "R3"):
        rep = reps[name]
        for a, b in PAIRS:
            X = {s: rep.Z[s][a] for s in SPLITS}
            n = rep.ncols
            v, size, a_rel, B, cx = select_ridge(rep, X, rep.Z["train"][b], b, rep.sizes, lambda z: (n(z), n(z)))
            add("H1", "ridge", name, a, b, rep.test_score(b, X["test"][:, :cx] @ B), size=size, alpha_rel=a_rel, val_avg_rank=v)
            best = None
            for size in rep.sizes:
                c = n(size)
                pen, rho = lw_penalty(X["train"][:, :c], rep.Z["train"][b][:, :c])
                B = ridge_abs(X["train"][:, :c], rep.Z["train"][b][:, :c], pen)
                vr = rep.val_rank(b, X["val"][:, :c] @ B)
                if best is None or vr > best[0]:
                    best = (vr, size, rho, B, c)
            vr, size, rho, B, c = best
            add("H1", "cond_gauss", name, a, b, rep.test_score(b, X["test"][:, :c] @ B), size=size, lw_rho=rho, val_avg_rank=vr)
        log(f"H1 {name}")

    # CCA (full rank; CCA-256 on R3 codes): (d, lambda) on the mean val average rank over the 72 pairs, ridge decoders
    cca = {}
    for name, Zsrc in (("CCA", full.Z), ("CCA-256", {s: {v: reps["R3"].Z[s][v][:, :CFG["cca"]["ref_k"]] for v in VIEWS}
                                                     for s in SPLITS})):
        best = None
        for d in CFG["cca"]["d"]:
            if d > min(Zsrc["train"][v].shape[1] for v in VIEWS):
                continue
            for lam in CFG["cca"]["lambda"]:
                A, _, _ = maxvar(Zsrc["train"], d, float(lam))
                enc = {s: {v: Zsrc[s][v] @ A[v] for v in VIEWS} for s in SPLITS}
                per_pair = {}
                for a, b in PAIRS:
                    per_pair[(a, b)] = max((full.val_rank(b, enc["val"][a] @ B), a_rel)
                                           for a_rel, B in ridge_path(enc["train"][a], full.Z["train"][b]).items())
                score_ = float(np.mean([x[0] for x in per_pair.values()]))
                if best is None or score_ > best[0]:
                    best = (score_, d, float(lam), enc, per_pair)
        score_, d, lam, enc, per_pair = best
        cca[name] = dict(d=d, lam=lam, enc=enc, per_pair=per_pair)
        sel[name] = dict(d=d, lambda_rel=lam, val_mean_avg_rank=score_)
        for (a, b), (vr, a_rel) in per_pair.items():
            B = ridge_path(enc["train"][a], full.Z["train"][b], [a_rel])[a_rel]
            add("H1", "cca_ridge", name, a, b, full.test_score(b, enc["test"][a] @ B), size=d, alpha_rel=a_rel, val_avg_rank=vr)
        log(f"H1 {name}: d {d} lambda {lam} (val {score_:.4f})")

    # H2: other views -> held-out view
    h2_alpha = {}
    for b in VIEWS:
        settings = {"other8": [v for v in VIEWS if v != b]}
        if b != "SC":
            settings["other7"] = [v for v in FC if v != b]
        for setting, srcs in settings.items():
            enc = cca["CCA"]["enc"]
            for model, f in (("cca_agg", lambda s: np.mean([enc[s][a] for a in srcs], axis=0)),
                             ("cca_cat", lambda s: np.concatenate([enc[s][a] for a in srcs], axis=1))):
                X = {s: f(s) for s in SPLITS}
                vr, a_rel, B = max(((full.val_rank(b, X["val"] @ B), a_rel, B) for a_rel, B in
                                    ridge_path(X["train"], full.Z["train"][b]).items()), key=lambda t: t[0])
                h2_alpha[f"{model}|{setting}|{b}"] = a_rel
                add("H2", model, "CCA", "+".join(srcs), b, full.test_score(b, X["test"] @ B), setting=setting,
                    size=X["train"].shape[1], alpha_rel=a_rel, val_avg_rank=vr)
            single = {a: cca["CCA"]["per_pair"][(a, b)] for a in srcs}
            a_best = max(single, key=lambda a: single[a][0])
            r = next(r for r in rows if r["hyp"] == "H1" and r["model"] == "cca_ridge" and r["rep"] == "CCA"
                     and r["source"] == a_best and r["target"] == b)
            add("H2", "cca_best_single", "CCA", a_best, b, {k: r[k] for k in ("pearson", "demeaned_pearson", "avg_rank",
                "top1_acc", "mse", "spread_ratio")}, setting=setting, val_avg_rank=single[a_best][0])
            for name in ("R0", "R1", "R2", "R3"):
                rep = reps[name]
                X = {s: np.concatenate([rep.Z[s][a] for a in srcs], axis=1) for s in SPLITS}
                widths = {a: rep.Z["train"][a].shape[1] for a in srcs}
                best = None
                for size in rep.sizes:
                    c = rep.ncols(size)
                    idx = np.concatenate([o + np.arange(c) for o in np.cumsum([0] + [widths[a] for a in srcs[:-1]])])
                    Xs = {s: X[s][:, idx] for s in ("train", "val")}
                    for a_rel, B in ridge_path(Xs["train"], rep.Z["train"][b][:, :c]).items():
                        vr = rep.val_rank(b, Xs["val"] @ B)
                        if best is None or vr > best[0]:
                            best = (vr, size, a_rel, B, idx)
                vr, size, a_rel, B, idx = best
                add("H2", "ridge_cat", name, "+".join(srcs), b, rep.test_score(b, X["test"][:, idx] @ B), setting=setting,
                    size=size, alpha_rel=a_rel, val_avg_rank=vr)
            fc_srcs = [a for a in srcs if a != "SC"]
            if b != "SC":
                dev = np.mean([full.T[("test", a)] - full.mu[a] for a in fc_srcs], axis=0)
                add("H2", "copy", "edges", "+".join(fc_srcs), b, full.test_score(b, edges=full.mu[b] + dev), setting=setting)
        log(f"H2 {b}")

    # G0b: R3 ridge at k = 256 with E4.0c's selection (penalty on val average rank) vs E4.0c's B1 (single-source FC)
    e40c = REPO_ROOT / CFG["e40c_dir"]
    ref = {(r["source"], r["target"]): r for r in json.loads((e40c / "k256" / f"seed{seed}.json").read_text())["rows"]
           if r["model"] == "B1" and r["split"] == "test" and r["source"] in FC}
    rep, g0b = reps["R3"], []
    for (a, b), r in ref.items():
        X = {s: rep.Z[s][a][:, :256] for s in ("train", "val", "test")}
        _, _, _, B, _ = select_ridge(rep, X, rep.Z["train"][b], b, [256], lambda z: (z, z))
        m = rep.test_score(b, X["test"] @ B)
        g0b.append(max(abs(m["demeaned_pearson"] - r["demeaned_pearson"]), abs(m["avg_rank"] - r["avg_rank"])))
    # G0c: full-rank CCA, E4.0c's B2 selection (d grid to 256, mean val average rank over the 56 FC pairs), MAXVAR decoders
    ref = {(r["source"], r["target"]): r for r in json.loads((e40c / "kfull" / f"seed{seed}.json").read_text())["rows"]
           if r["model"] == "B2" and r["split"] == "test" and r["source"] in FC and not r["with_sc"]}
    single = [(a, b) for a in FC for b in FC if a != b]
    best = None
    for d in [16, 32, 64, 128, 256]:
        for lam in CFG["cca"]["lambda"]:
            A, D, _ = maxvar(full.Z["train"], d, float(lam))
            vr = float(np.mean([full.val_rank(b, full.Z["val"][a] @ A[a] @ D[b]) for a, b in single]))
            if best is None or vr > best[0]:
                best = (vr, A, D)
    _, A, D = best
    g0c = []
    for (a, b), r in ref.items():
        m = full.test_score(b, full.Z["test"][a] @ A[a] @ D[b])
        g0c.append(max(abs(m["demeaned_pearson"] - r["demeaned_pearson"]), abs(m["avg_rank"] - r["avg_rank"])))
    # fast scorer vs explicit decoding (one pair per representation)
    from models.eval.metrics import compute_corr_matrix
    fast_diff = 0.0
    for name in ("R1", "R2", "R3"):
        rep = reps[name]
        Zp = rep.Z["val"]["rest"][:, :256]
        fast_diff = max(fast_diff, float(np.abs(rep.val_cc("wm", Zp) - compute_corr_matrix(rep.T[("val", "wm")],
                                                                                          rep.decode("wm", Zp))).max()))
    tol = CFG["gates"]["g0_tol"]
    g0 = dict(g0a_max_abs=g0a, g0b_max_abs=max(g0b), g0b_n=len(g0b), g0c_max_abs=max(g0c), g0c_n=len(g0c),
              fast_scorer_max_abs=fast_diff)
    g0["pass"] = bool(g0a == 0 and max(g0b) < tol and max(g0c) < tol and len(g0b) == 56 == len(g0c) and fast_diff < 1e-8)
    log(f"G0 {'PASS' if g0['pass'] else 'FAIL'} {g0}")

    (OUT / "results").mkdir(parents=True, exist_ok=True)
    (OUT / "h3").mkdir(parents=True, exist_ok=True)
    sel["H1_pca_val"] = {name: float(np.mean([r["val_avg_rank"] for r in rows if r["hyp"] == "H1" and r["rep"] == name
                                              and r["model"] == "ridge"])) for name in ("R0", "R1", "R2", "R3")}
    out = dict(seed=seed, g0=g0, selections=sel, rows=rows, seconds=time.time() - T0)
    (OUT / "results" / f"seed{seed}.json").write_text(json.dumps(out, default=float))
    enc = cca["CCA"]["enc"]
    np.savez(OUT / "h3" / f"seed{seed}.npz",
             **{f"enc__{s}__{v}": enc[s][v] for s in SPLITS for v in VIEWS},
             **{f"split__{s}": splits[s] for s in SPLITS},
             alpha_json=json.dumps(dict(pairs={f"{a}|{b}": x[1] for (a, b), x in cca["CCA"]["per_pair"].items()},
                                        h2=h2_alpha, d=cca["CCA"]["d"], lam=cca["CCA"]["lam"])))
    log(f"wrote results/seed{seed}.json ({len(rows)} rows) and h3/seed{seed}.npz")
    return g0["pass"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    raise SystemExit(0 if run(ap.parse_args().seed) else 1)
