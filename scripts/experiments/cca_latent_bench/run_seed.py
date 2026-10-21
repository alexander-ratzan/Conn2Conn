"""Spec v4 E2: CCA-latent bidirectional benchmark for one seed (closed form, CPU).

1. Latent selection (SC-blind): MAXVAR over the 8 FC conditions for every (k_fc, d, lambda); keep the one with the best
   mean val cross-condition identification (avg rank, 56 ordered pairs). Also records, per configuration, the val
   demeaned r of consensus -> SC ridge (k_sc 256) to pick the secondary SC-tuned latent.
2. FC -> SC: inputs (rest PCA codes; rest encoding; 7-task / 8-condition consensus; concatenated encodings; each task's
   encoding) x readouts (ridge GCV; PLS n), SC target dimension and n chosen on val; scored on decoded SC edges.
3. SC -> FC: SC codes -> {consensus latent, each condition's encoding, each condition's codes (direct)} x readouts,
   decoded to each FC condition; ceilings from the true latent; latent R^2 per dimension and latent identification.

Writes `records_dir/seed<S>.csv` (all scored rows), `seed<S>_latent.csv` (selection grid) and `seed<S>_latent_r2.csv`.

    python scripts/experiments/cca_latent_bench/run_seed.py --seed 0 [--check]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cross_decomposition import PLSRegression

EXP = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in EXP.parents if (p / "main.py").exists())
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments" / "joint_flow_matching"))
from common import SeedData, score  # noqa: E402  (E1's per-seed cache and v3:E0 scorer)
from models.architectures.multiview_cca import maxvar, ridge_gcv  # noqa: E402
from models.eval.metrics import compute_corr_matrix, corr_avg_rank, corr_topn_accuracy  # noqa: E402

CFG = yaml.safe_load((EXP / "config.yml").read_text())
FC, TASKS = CFG["fc_views"], CFG["tasks"]
ALPHAS = [float(a) for a in CFG["readouts"]["ridge_alphas"]]
PLS_N = [int(n) for n in CFG["readouts"]["pls_components"]]


def fast_dr(pred, true, mu):
    a, b = pred - mu, true - mu
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return float(np.mean((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)))


def ident(src, tgt):
    """Rows = targets, columns = candidates; centered on the evaluated set (as the notebook)."""
    cc = compute_corr_matrix(tgt - tgt.mean(0), src - src.mean(0))
    return float(corr_topn_accuracy(cc=cc, topn=1)), float(corr_avg_rank(cc=cc))


def readouts(n_in):
    """(name, setting) pairs: ridge, and PLS with n components <= input dimension."""
    return [("ridge", None)] + [("pls", n) for n in PLS_N if n <= n_in]


def fit_predict(kind, n, Xtr, Ytr, Xs):
    """Fit a readout on train and predict each array in Xs. Returns (list of predictions, setting dict)."""
    xm, ym = Xtr.mean(0), Ytr.mean(0)
    if kind == "ridge":
        B, a_rel = ridge_gcv(Xtr - xm, Ytr - ym, ALPHAS)
        return [ym + (X - xm) @ B for X in Xs], {"alpha_rel": a_rel}
    m = PLSRegression(n_components=n, scale=True).fit(Xtr, Ytr)
    return [m.predict(X) for X in Xs], {"n_components": n}


class Seed:
    def __init__(self, seed):
        self.seed = seed
        self.sd = SeedData(seed, "main")
        self.kmax = self.sd.cs.k_max
        self.Z = {s: {v: self.sd.code(v, s, self.kmax).astype(np.float64) for v in FC + ["SC"]}
                  for s in ("train", "val", "test")}
        self.rows = []

    def add(self, split, metrics, **kw):
        self.rows.append(dict(seed=self.seed, split=split, **kw, **metrics))

    def latent(self, k, d, lam):
        A, D, G = maxvar({v: self.Z["train"][v][:, :k] for v in FC}, d, lam)
        enc = {s: {v: self.Z[s][v][:, :k] @ A[v] for v in FC} for s in ("train", "val", "test")}
        return dict(k=k, d=d, lam=lam, A=A, D=D, enc=enc)

    def cross_id(self, E):
        return float(np.mean([ident(E[a], E[b])[1] for a in FC for b in FC if a != b]))

    # ---- 1. latent selection
    def select(self):
        grid, lat = [], CFG["latent"]
        ysc = {s: self.Z[s]["SC"][:, :256] for s in ("train", "val")}
        for k in lat["k_fc"]:
            for d in [d for d in lat["d"] if d <= k]:
                for lam in [float(x) for x in lat["lambda_rel"]]:
                    L = self.latent(k, d, lam)
                    cons = {s: np.mean([L["enc"][s][v] for v in FC], 0) for s in ("train", "val")}
                    (pv,), _ = fit_predict("ridge", None, cons["train"], ysc["train"], [cons["val"]])
                    grid.append(dict(seed=self.seed, k_fc=k, d=d, lambda_rel=lam,
                                     val_cross_avg_rank=self.cross_id(L["enc"]["val"]),
                                     val_cons_sc_dr=fast_dr(self.sd.decode("SC", pv), self.sd.edges[("val", "SC")],
                                                            self.sd.train_mean["SC"])))
        g = pd.DataFrame(grid)
        best = {"sc_blind": g.loc[g.val_cross_avg_rank.idxmax()], "sc_tuned": g.loc[g.val_cons_sc_dr.idxmax()]}
        for name, r in best.items():
            pca = {v: self.Z["val"][v][:, :int(r.d)] for v in FC}
            g.loc[(g.k_fc == r.k_fc) & (g.d == r.d) & (g.lambda_rel == r.lambda_rel), f"selected_{name}"] = True
            g.loc[(g.k_fc == r.k_fc) & (g.d == r.d) & (g.lambda_rel == r.lambda_rel),
                  f"pooled_pca_d_val_cross_avg_rank"] = self.cross_id(pca)
        self.grid = g
        self.lat = {n: self.latent(int(r.k_fc), int(r.d), float(r.lambda_rel)) for n, r in best.items()}
        for n, L in self.lat.items():
            print(f"seed {self.seed} latent {n}: k_fc {L['k']} d {L['d']} lambda {L['lam']}", flush=True)

    # ---- 2. FC -> SC
    def fc_inputs(self, L):
        e = L["enc"]
        inp = {"rest encoding": {s: e[s]["rest"] for s in e},
               "7-task consensus": {s: np.mean([e[s][v] for v in TASKS], 0) for s in e},
               "8-condition consensus": {s: np.mean([e[s][v] for v in FC], 0) for s in e},
               "concatenated encodings": {s: np.hstack([e[s][v] for v in FC]) for s in e}}
        for t in TASKS:
            inp[f"{t} encoding"] = {s: e[s][t] for s in e}
        return inp

    def fc2sc(self, name, X, latent_tag):
        """Per readout kind: choose (k_sc, PLS n) on val, then score val and test."""
        cands = {}
        for kind, n in readouts(X["train"].shape[1]):
            for ksc in CFG["fc2sc"]["k_sc"]:
                (pv,), hp = fit_predict(kind, n, X["train"], self.Z["train"]["SC"][:, :ksc], [X["val"]])
                v = fast_dr(self.sd.decode("SC", pv), self.sd.edges[("val", "SC")], self.sd.train_mean["SC"])
                cands.setdefault(kind, []).append((v, ksc, n))
        for kind, cs in cands.items():
            _, ksc, n = max(cs, key=lambda c: c[0])
            (pv, pt), hp = fit_predict(kind, n, X["train"], self.Z["train"]["SC"][:, :ksc], [X["val"], X["test"]])
            for split, p in (("val", pv), ("test", pt)):
                self.add(split, score(self.sd.decode("SC", p), self.sd.edges[(split, "SC")], self.sd.train_mean["SC"]),
                         direction="fc2sc", latent=latent_tag, input=name, condition="SC", readout=kind,
                         setting=json.dumps(dict(hp, k_sc=ksc)))

    def run_fc2sc(self):
        for k in CFG["fc2sc"]["pca_k"]:
            self.fc2sc(f"rest PCA codes k={k}", {s: self.Z[s]["rest"][:, :k] for s in self.Z}, "-")
        for tag, L in self.lat.items():
            inputs = self.fc_inputs(L)
            if tag == "sc_tuned":
                inputs = {n: inputs[n] for n in ("rest encoding", "8-condition consensus")}
            for name, X in inputs.items():
                self.fc2sc(name, X, tag)
            print(f"seed {self.seed} fc2sc {tag} done", flush=True)

    # ---- 3. SC -> FC
    def run_sc2fc(self):
        L = self.lat["sc_blind"]
        e, D, k = L["enc"], L["D"], L["k"]
        targets = {"via consensus latent": {s: np.mean([e[s][v] for v in FC], 0) for s in e}}
        for m in FC:
            targets[f"via own encoding:{m}"] = {s: e[s][m] for s in e}
            targets[f"direct codes:{m}"] = {s: self.Z[s][m][:, :k] for s in e}

        def decode_to(route, m, p):
            return self.sd.decode(m, p if route.startswith("direct") else p @ D[m])

        best_cons = None
        for route, Y in targets.items():
            conds = FC if route == "via consensus latent" else [route.split(":")[1]]
            cands = {m: [] for m in conds}
            for kind, n in readouts(min(CFG["sc2fc"]["k_sc"])):
                for ksc in CFG["sc2fc"]["k_sc"]:
                    (pv,), hp = fit_predict(kind, n, self.Z["train"]["SC"][:, :ksc], Y["train"], [self.Z["val"]["SC"][:, :ksc]])
                    for m in conds:
                        v = fast_dr(decode_to(route, m, pv), self.sd.edges[("val", m)], self.sd.train_mean[m])
                        cands[m].append((v, kind, n, ksc, hp))
                    if route == "via consensus latent" and kind == "ridge":   # headline readout for H5
                        mean_v = np.mean([cands[m][-1][0] for m in conds])
                        if best_cons is None or mean_v > best_cons[0]:
                            best_cons = (mean_v, kind, n, ksc)
            for m, kind in [(m, kind) for m in conds for kind in ("ridge", "pls")]:
                v, kind, n, ksc, hp = max((c for c in cands[m] if c[1] == kind), key=lambda c: c[0])
                (pv, pt), _ = fit_predict(kind, n, self.Z["train"]["SC"][:, :ksc], Y["train"],
                                          [self.Z["val"]["SC"][:, :ksc], self.Z["test"]["SC"][:, :ksc]])
                for split, p in (("val", pv), ("test", pt)):
                    self.add(split, score(decode_to(route, m, p), self.sd.edges[(split, m)], self.sd.train_mean[m]),
                             direction="sc2fc", latent="sc_blind", input=route.split(":")[0], condition=m,
                             readout=kind, setting=json.dumps(dict(hp, k_sc=ksc)))
        for m in FC:   # ceilings: decode the true latent
            for split in ("val", "test"):
                cons = np.mean([e[split][v] for v in FC], 0)
                self.add(split, score(self.sd.decode(m, cons @ D[m]), self.sd.edges[(split, m)], self.sd.train_mean[m]),
                         direction="sc2fc", latent="sc_blind", input="ceiling: true consensus latent", condition=m,
                         readout="-", setting="{}")
                self.add(split, score(self.sd.decode(m, e[split][m] @ D[m]), self.sd.edges[(split, m)],
                                      self.sd.train_mean[m]),
                         direction="sc2fc", latent="sc_blind", input="ceiling: true own encoding", condition=m,
                         readout="-", setting="{}")
        # latent R^2 per dimension and latent identification: ridge consensus readout, k_sc by mean val dr over conditions
        _, kind, n, ksc = best_cons
        Y = targets["via consensus latent"]
        (pt,), _ = fit_predict(kind, n, self.Z["train"]["SC"][:, :ksc], Y["train"], [self.Z["test"]["SC"][:, :ksc]])
        true = Y["test"]
        r2 = 1 - np.square(pt - true).sum(0) / np.square(true - true.mean(0)).sum(0)
        self.latent_r2 = pd.DataFrame(dict(seed=self.seed, dim=np.arange(1, len(r2) + 1), test_r2=r2,
                                           readout=kind, n_components=n, k_sc=ksc))
        t1, rk = ident(pt, true)
        self.add("test", dict(top1_acc=t1, avg_rank=rk, latent_r2_mean=float(r2.mean())), direction="sc2fc",
                 latent="sc_blind", input="latent identification: predicted vs true consensus", condition="latent",
                 readout=kind, setting=json.dumps(dict(n_components=n, k_sc=ksc)))

    def check_l1(self):
        """G1(b): rest PCA k=64 -> SC k=64 ridge == E1 L1 (test demeaned r)."""
        Xtr, Xte = self.Z["train"]["rest"][:, :64], self.Z["test"]["rest"][:, :64]
        (pt,), _ = fit_predict("ridge", None, Xtr, self.Z["train"]["SC"][:, :64], [Xte])
        mine = score(self.sd.decode("SC", pt), self.sd.edges[("test", "SC")], self.sd.train_mean["SC"])["demeaned_pearson"]
        e1 = pd.read_csv(REPO_ROOT / CFG["e1_linear"] / f"seed{self.seed}_main.csv")
        ref = e1[(e1.ref == "L1") & (e1.k == 64) & (e1.source == "rest") & (e1.target == "SC") & (e1.split == "test")]
        ref = float(ref.demeaned_pearson.iloc[0])
        ok = abs(mine - ref) < float(CFG["gates"]["l1_match_tol"])
        print(f"{'OK  ' if ok else 'FAIL'} G1b: rest PCA k64 -> SC ridge {mine:.8f} vs E1 L1 {ref:.8f}", flush=True)
        return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--check", action="store_true", help="run the G1 checks (pilot)")
    args = ap.parse_args()
    t0 = time.time()
    S = Seed(args.seed)
    ok = S.check_l1() if args.check else True
    S.select()
    S.run_fc2sc()
    S.run_sc2fc()
    out = REPO_ROOT / CFG["records_dir"]
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(S.rows)
    df.to_csv(out / f"seed{args.seed}.csv", index=False)
    S.grid.to_csv(out / f"seed{args.seed}_latent.csv", index=False)
    S.latent_r2.to_csv(out / f"seed{args.seed}_latent_r2.csv", index=False)
    if args.check:
        num = df.select_dtypes("number").drop(columns=["seed"])
        finite = bool(np.isfinite(num.values[~np.isnan(num.values)]).all()) and not df.demeaned_pearson.dropna().empty
        sel = S.grid[S.grid.get("selected_sc_blind", pd.Series(False, index=S.grid.index)).fillna(False).astype(bool)].iloc[0]
        lat_ok = sel.val_cross_avg_rank > sel.pooled_pca_d_val_cross_avg_rank
        print(f"{'OK  ' if finite else 'FAIL'} G1a: rows finite ({len(df)} rows)")
        print(f"{'OK  ' if lat_ok else 'FAIL'} G1c: latent val cross-condition avg rank {sel.val_cross_avg_rank:.3f} > "
              f"pooled PCA (same d) {sel.pooled_pca_d_val_cross_avg_rank:.3f}")
        print("ALL_OK" if (ok and finite and lat_ok) else "FAILED")
    print(f"seed {args.seed} done in {(time.time() - t0) / 60:.1f} min; wrote {out}/seed{args.seed}*.csv")
    if args.check and not (ok and finite and lat_ok):
        sys.exit(1)          # gate G1 failed: dependent jobs (afterok) do not start


if __name__ == "__main__":
    main()
