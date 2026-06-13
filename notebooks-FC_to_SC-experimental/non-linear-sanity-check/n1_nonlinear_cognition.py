#!/usr/bin/env python3
"""N1 — Nonlinear downstream cognition.

Does a nonlinear estimator extract cognition signal from tractography that the linear
BayesianRidge missed? Same reps/targets/splits as tractography_predict E5; the ONLY
change is the estimator. Three estimators per (rep, target, seed):
  linear_BR  (reference, == E5)
  HGB        (HistGradientBoosting — interactions/thresholds)
  KR         (KernelRidge RBF — smooth nonlinearity)

Metrics per prediction (scalar cognition target): Pearson, Spearman (rank), R².
Plus the "lift over bv+demo" floor, computed per estimator.

DECISION RULE: a rep×nonlinear beats the bv+demo floor by >= +0.03 Pearson on >= 2 of
3 targets with 10-seed Wilcoxon p<0.05 -> tractography carries nonlinear cognition signal.

Outputs:
  n1_cognition_results.csv  (rep, estimator, target, seed, pearson, spearman, r2)
  n1_cognition_summary.csv  (medians + lift over bv+demo per rep/estimator/target)
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nl_common import (load_seed_split_with_r2t, source_train_test, target_train_test,
                        hgb_scalar_predict, kr_scalar_predict,
                        PCA, BayesianRidge, LinearRegression)
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

THIS_DIR = Path(__file__).resolve().parent
N_SEEDS = 10
K_PCA = 256
COG_TARGETS = ["CogTotalComp_Unadj", "CogFluidComp_Unadj", "CogCrystalComp_Unadj"]
REPS = ["bv+demo", "SC", "r2t", "r2t_corr", "FC", "SC_r2t"]
ESTIMATORS = ["linear_BR", "HGB", "KR"]

COG_CSV = next(p for p in [
    Path("/scratch/asr655/neuroinformatics/GeneEx2Conn_data/HCP1200/HCP1200_UNRESTRICTED.csv"),
    Path("/scratch/ans9868/Conn2Conn/data/HCP1200_UNRESTRICTED.csv"),
] if p.exists())
cog_lookup = pd.read_csv(COG_CSV)[["Subject"] + COG_TARGETS].astype({"Subject": int}).set_index("Subject")


def cog_vec(split, target):
    subj = np.asarray(split["base"].metadata_df["subject"]).astype(int)
    def pull(idx):
        return np.array([cog_lookup[target].get(s, np.nan) for s in subj[idx]], dtype=np.float64)
    return pull(split["train_idx"]), pull(split["test_idx"])


def linear_br_scalar(X_tr, X_te, y_tr, k=K_PCA):
    ok = ~np.isnan(y_tr)
    p = PCA(n_components=min(k, X_tr.shape[1]), random_state=0).fit(X_tr[ok])
    m = BayesianRidge(max_iter=500).fit(p.transform(X_tr[ok]), y_tr[ok])
    return m.predict(p.transform(X_te))


def src_for(split, rep):
    if rep == "bv+demo":
        return (np.concatenate([split["bv_train"], split["demo_train"]], axis=1),
                np.concatenate([split["bv_test"], split["demo_test"]], axis=1))
    return source_train_test(split, rep)


def metrics(pred, true):
    ok = ~np.isnan(true)
    if ok.sum() < 3:
        return dict(pearson=np.nan, spearman=np.nan, r2=np.nan)
    return dict(
        pearson=float(pearsonr(pred[ok], true[ok])[0]),
        spearman=float(spearmanr(pred[ok], true[ok])[0]),
        r2=float(r2_score(true[ok], pred[ok])),
    )


rows = []
for seed in range(N_SEEDS):
    print(f"=== seed {seed} ===", flush=True)
    split = load_seed_split_with_r2t(seed=seed)
    for target in COG_TARGETS:
        cog_tr, cog_te = cog_vec(split, target)
        for rep in REPS:
            X_tr, X_te = src_for(split, rep)
            for est in ESTIMATORS:
                if est == "linear_BR":
                    pred = linear_br_scalar(X_tr, X_te, cog_tr)
                elif est == "HGB":
                    pred = hgb_scalar_predict(X_tr, X_te, cog_tr, k=K_PCA)
                else:
                    pred = kr_scalar_predict(X_tr, X_te, cog_tr, k=K_PCA)
                rows.append({"rep": rep, "estimator": est, "target": target,
                             "seed": seed, **metrics(pred, cog_te)})
        print(f"  {target} done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(THIS_DIR / "n1_cognition_results.csv", index=False)
print(f"\nSaved -> {THIS_DIR / 'n1_cognition_results.csv'}")

# Summary + lift over bv+demo (per estimator: floor is bv+demo with the SAME estimator).
summ = df.groupby(["rep", "estimator", "target"])[["pearson", "spearman", "r2"]].median().reset_index()
floor = (summ[summ["rep"] == "bv+demo"].set_index(["estimator", "target"])["pearson"])
summ["lift_over_bvdemo"] = summ.apply(
    lambda r: r["pearson"] - floor.get((r["estimator"], r["target"]), np.nan), axis=1)
summ.to_csv(THIS_DIR / "n1_cognition_summary.csv", index=False)

print("\n=== Median Pearson by rep x estimator x target (lift over bv+demo floor) ===")
for target in COG_TARGETS:
    print(f"\n  {target}:")
    sub = summ[summ["target"] == target]
    for est in ESTIMATORS:
        print(f"   [{est}]")
        s = sub[sub["estimator"] == est].sort_values("pearson", ascending=False)
        for _, r in s.iterrows():
            print(f"     {r['rep']:9s} pearson={r['pearson']:+.3f} spearman={r['spearman']:+.3f} "
                  f"r2={r['r2']:+.3f} lift={r['lift_over_bvdemo']:+.3f}")

# Decision check: does any tractography rep x nonlinear clear the floor by >=0.03?
print("\n=== DECISION: nonlinear cognition unlock? ===")
TRACT_REPS = ["SC", "r2t", "r2t_corr", "SC_r2t"]
unlocked = summ[(summ["rep"].isin(TRACT_REPS)) & (summ["estimator"] != "linear_BR")
                & (summ["lift_over_bvdemo"] >= 0.03)]
if len(unlocked):
    print("  Candidates clearing floor by >=0.03 (verify Wilcoxon separately):")
    print(unlocked[["rep", "estimator", "target", "pearson", "lift_over_bvdemo"]]
          .to_string(index=False))
else:
    print("  NONE. No tractography rep clears the bv+demo floor under HGB or KR.")
    print("  -> nonlinear does NOT unlock tractography cognition signal; dead end is")
    print("     model-class-robust.")
