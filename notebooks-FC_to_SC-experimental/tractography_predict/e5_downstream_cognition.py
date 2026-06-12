#!/usr/bin/env python3
"""E5 — Downstream cognition prediction from tractography representations.

Mirrors the main notebook's Phase 2 Analysis 2 ("substitution fidelity" +
"lift above bv+demo"). Targets are HCP NIH-Toolbox composites:
  CogTotalComp_Unadj, CogFluidComp_Unadj, CogCrystalComp_Unadj
from HCP1200_UNRESTRICTED.csv.

For each (rep, target, seed):
  rep -> PCA(256) -> BayesianRidge -> cognition score; test Pearson(pred, true).
Plus a "lift above bv+demo" version: residualize cognition on bv+demo (train fit),
then test whether the rep predicts the residual — isolates NON-demographic signal.

Reps:
  bv+demo      — confound floor
  SC           — count connectome (prior work: ~0 above demo)
  r2t          — bundle membership
  r2t_corr     — bundle-similarity
  FC           — reference (carried crystallized signal in prior work)
  SC_r2t       — does r2t add over SC? (per-block PCA, scale-fair)
  r2t->FC->cog — SUBSTITUTION: predict FC from r2t, then cognition from synthetic FC

Outputs:
  e5_downstream_results.csv  (rep, target, seed, pearson_raw, pearson_resid)
  e5_downstream_summary.csv  (medians + lift over bv+demo per rep/target)
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tract_setup import (load_seed_split_with_r2t, source_train_test, source_blocks,
                          target_train_test, pca_pls_predict, block_pca_pls_predict,
                          PCA, BayesianRidge, LinearRegression)
from scipy.stats import pearsonr

THIS_DIR = Path(__file__).resolve().parent
N_SEEDS = 10
K_PCA = 256
COG_TARGETS = ["CogTotalComp_Unadj", "CogFluidComp_Unadj", "CogCrystalComp_Unadj"]

COG_CSV = next(p for p in [
    Path("/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200/HCP1200_UNRESTRICTED.csv"),
    Path("/scratch/ans9868/Conn2Conn/data/HCP1200_UNRESTRICTED.csv"),
] if p.exists())
print(f"Cognition CSV: {COG_CSV}")
cog_df = pd.read_csv(COG_CSV)[["Subject"] + COG_TARGETS]
cog_df["Subject"] = cog_df["Subject"].astype(int)
cog_lookup = cog_df.set_index("Subject")


def cog_vec_for_split(split, target):
    """Return (cog_train, cog_test) aligned to the seed's train/test subject order.
    NaN where cognition is missing for that subject."""
    subj = np.asarray(split["base"].metadata_df["subject"]).astype(int)
    tr, te = split["train_idx"], split["test_idx"]
    def pull(idx):
        ids = subj[idx]
        return np.array([cog_lookup[target].get(s, np.nan) for s in ids], dtype=np.float64)
    return pull(tr), pull(te)


def pca_br_scalar(X_tr, X_te, y_tr, k=K_PCA):
    """PCA(X) -> BayesianRidge -> scalar prediction on test. Drops NaN y rows in train."""
    ok = ~np.isnan(y_tr)
    pca = PCA(n_components=min(k, X_tr.shape[1]), random_state=0).fit(X_tr[ok])
    Z_tr = pca.transform(X_tr[ok])
    Z_te = pca.transform(X_te)
    br = BayesianRidge(max_iter=500).fit(Z_tr, y_tr[ok])
    return br.predict(Z_te)


def test_pearson(pred, true):
    ok = ~np.isnan(true)
    if ok.sum() < 3:
        return float("nan")
    r, _ = pearsonr(pred[ok], true[ok])
    return float(r)


rows = []
for seed in range(N_SEEDS):
    print(f"=== seed {seed} ===", flush=True)
    split = load_seed_split_with_r2t(seed=seed)
    bv_tr, bv_te = split["bv_train"], split["bv_test"]
    dm_tr, dm_te = split["demo_train"], split["demo_test"]
    bvdemo_tr = np.concatenate([bv_tr, dm_tr], axis=1)
    bvdemo_te = np.concatenate([bv_te, dm_te], axis=1)

    # Precompute synthetic FC from r2t (for the substitution arm).
    FC_tr_full, FC_te_full, _ = target_train_test(split, "FC")
    r2t_tr, r2t_te = source_train_test(split, "r2t")
    synth_FC_te = pca_pls_predict(r2t_tr, r2t_te, FC_tr_full)   # (n_test, 64620)
    # For the substitution chain we also need synthetic FC on TRAIN (to fit cog model).
    # Use an internal split-free approx: fit r2t->FC on train, predict train via OOF is
    # overkill; instead fit cog model on REAL FC_train (teacher) and apply to synth FC_test.
    # This measures whether synthetic FC_test lands in the same cognition-predictive
    # subspace as real FC. (Standard substitution-fidelity setup.)

    for target in COG_TARGETS:
        cog_tr, cog_te = cog_vec_for_split(split, target)

        # bv+demo baseline (the confound floor).
        pred_bd = pca_br_scalar(bvdemo_tr, bvdemo_te, cog_tr, k=min(K_PCA, bvdemo_tr.shape[1]))
        r_bd = test_pearson(pred_bd, cog_te)

        # Residualize cognition on bv+demo (train fit) -> isolate non-demographic signal.
        ok = ~np.isnan(cog_tr)
        ols_bd = LinearRegression().fit(bvdemo_tr[ok], cog_tr[ok])
        cog_tr_resid = cog_tr.copy()
        cog_tr_resid[ok] = cog_tr[ok] - ols_bd.predict(bvdemo_tr[ok])
        cog_te_resid = cog_te - ols_bd.predict(bvdemo_te)

        rows.append({"rep": "bv+demo", "target": target, "seed": seed,
                     "pearson_raw": r_bd, "pearson_resid": float("nan")})

        # Standalone reps (shared PCA).
        for rep in ["SC", "r2t", "r2t_corr", "FC"]:
            X_tr, X_te = source_train_test(split, rep)
            pred_raw   = pca_br_scalar(X_tr, X_te, cog_tr)
            pred_resid = pca_br_scalar(X_tr, X_te, cog_tr_resid)
            rows.append({"rep": rep, "target": target, "seed": seed,
                         "pearson_raw":   test_pearson(pred_raw, cog_te),
                         "pearson_resid": test_pearson(pred_resid, cog_te_resid)})

        # SC_r2t combined (per-block PCA, scale-fair).
        btr, bte = source_blocks(split, "SC_r2t")
        # block_pca_pls_predict targets a matrix; for scalar cog we PCA-concat then BR.
        Ztr_parts, Zte_parts = [], []
        for Xtr_b, Xte_b in zip(btr, bte):
            p = PCA(n_components=K_PCA, random_state=0).fit(Xtr_b)
            Ztr_parts.append(p.transform(Xtr_b)); Zte_parts.append(p.transform(Xte_b))
        Ztr = np.concatenate(Ztr_parts, axis=1); Zte = np.concatenate(Zte_parts, axis=1)
        okc = ~np.isnan(cog_tr)
        br_c = BayesianRidge(max_iter=500).fit(Ztr[okc], cog_tr[okc])
        okr = ~np.isnan(cog_tr_resid)
        br_cr = BayesianRidge(max_iter=500).fit(Ztr[okr], cog_tr_resid[okr])
        rows.append({"rep": "SC_r2t", "target": target, "seed": seed,
                     "pearson_raw":   test_pearson(br_c.predict(Zte), cog_te),
                     "pearson_resid": test_pearson(br_cr.predict(Zte), cog_te_resid)})

        # Substitution: cognition model fit on REAL FC_train, applied to synthetic-FC_test.
        pca_fc = PCA(n_components=K_PCA, random_state=0).fit(FC_tr_full[ok])
        Z_fc_tr = pca_fc.transform(FC_tr_full[ok])
        Z_synth_te = pca_fc.transform(synth_FC_te)
        br_sub = BayesianRidge(max_iter=500).fit(Z_fc_tr, cog_tr[ok])
        rows.append({"rep": "r2t->synthFC", "target": target, "seed": seed,
                     "pearson_raw":   test_pearson(br_sub.predict(Z_synth_te), cog_te),
                     "pearson_resid": float("nan")})
        print(f"  {target}: bv+demo={r_bd:.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(THIS_DIR / "e5_downstream_results.csv", index=False)
print(f"\nSaved -> {THIS_DIR / 'e5_downstream_results.csv'}")

# Summary: median pearson per (rep, target) + lift over bv+demo.
summ = (df.groupby(["rep", "target"])[["pearson_raw", "pearson_resid"]]
          .median().reset_index())
# Lift over bv+demo (raw).
bd = summ[summ["rep"] == "bv+demo"].set_index("target")["pearson_raw"]
summ["lift_over_bvdemo_raw"] = summ.apply(
    lambda r: r["pearson_raw"] - bd.get(r["target"], np.nan), axis=1)
summ.to_csv(THIS_DIR / "e5_downstream_summary.csv", index=False)
print("\n=== Median test Pearson by rep x target ===")
for target in COG_TARGETS:
    print(f"\n  {target}:")
    sub = summ[summ["target"] == target].sort_values("pearson_raw", ascending=False)
    for _, r in sub.iterrows():
        print(f"    {r['rep']:14s} raw={r['pearson_raw']:+.3f}  "
              f"resid={r['pearson_resid']:+.3f}  "
              f"lift_over_bvdemo={r['lift_over_bvdemo_raw']:+.3f}")
