#!/usr/bin/env python3
"""N2 — Nonlinear connectome reconstruction + asymmetry.

Does a nonlinear estimator (KernelRidge RBF, multi-output) extract cross-modal
reconstruction signal that linear PLS missed? Same reps/directions/splits as
tractography_predict E1+E2; only the estimator changes.

For each rep X in {SC, r2t, r2t_corr}, both directions (FC->X, X->FC), 10 seeds,
two estimators (linear_PLS reference, KR nonlinear), report the FULL 6-metric panel
(demeaned_pearson, pearson, top1_acc, avg_rank, mse, r2) via full_panel_eval — exactly
the same metrics as the linear runs. Asymmetry ratio computed per metric per estimator.

DECISION RULE: KR beats PLS on a direction by >= 0.02 demeaned_pearson AND closes >=
half the count-vs-bundle gap -> bundles have nonlinear reconstruction structure.

Outputs:
  n2_reconstruction_results.csv  (rep, estimator, direction, seed, all 6 metrics)
  n2_reconstruction_summary.csv  (median per rep/estimator/direction + asymmetry ratio)
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nl_common import (load_seed_split_with_r2t, source_train_test, target_train_test,
                        pca_pls_predict, kernelridge_predict, full_panel_eval)

THIS_DIR = Path(__file__).resolve().parent
N_SEEDS = 10
REPS = ["SC", "r2t", "r2t_corr"]
METRICS = ["demeaned_pearson", "pearson", "top1_acc", "avg_rank", "mse", "r2"]

rows = []
for seed in range(N_SEEDS):
    print(f"=== seed {seed} ===", flush=True)
    split = load_seed_split_with_r2t(seed=seed)
    FC_tr, FC_te, FC_mean = target_train_test(split, "FC")
    for rep in REPS:
        X_tr, X_te = source_train_test(split, rep)
        Y_tr, Y_te, Y_mean = target_train_test(split, rep)
        for est, fn in [("linear_PLS", pca_pls_predict), ("KR", kernelridge_predict)]:
            # FC -> X
            pred = fn(FC_tr, FC_te, Y_tr)
            p1 = full_panel_eval(pred, Y_te, Y_mean)
            rows.append({"rep": rep, "estimator": est, "direction": f"FC->{rep}",
                         "seed": seed, **p1})
            # X -> FC
            pred2 = fn(X_tr, X_te, FC_tr)
            p2 = full_panel_eval(pred2, FC_te, FC_mean)
            rows.append({"rep": rep, "estimator": est, "direction": f"{rep}->FC",
                         "seed": seed, **p2})
            print(f"  [{rep:8s}|{est:10s}] FC->{rep} dp={p1['demeaned_pearson']:.4f}  "
                  f"{rep}->FC dp={p2['demeaned_pearson']:.4f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(THIS_DIR / "n2_reconstruction_results.csv", index=False)
print(f"\nSaved -> {THIS_DIR / 'n2_reconstruction_results.csv'}")

# Summary: median each metric per (rep, estimator, direction) + asymmetry ratio
# (FC->X / X->FC for higher-better; flipped for mse; diff for r2).
HIGHER = ["demeaned_pearson", "pearson", "top1_acc", "avg_rank"]
srows = []
for rep in REPS:
    for est in ["linear_PLS", "KR"]:
        fc = df[(df.rep == rep) & (df.estimator == est) & (df.direction == f"FC->{rep}")].sort_values("seed")
        xf = df[(df.rep == rep) & (df.estimator == est) & (df.direction == f"{rep}->FC")].sort_values("seed")
        for m in METRICS:
            a, b = fc[m].values, xf[m].values
            if m in HIGHER:
                comp = np.median(a / np.where(np.abs(b) > 1e-9, b, np.nan))
            elif m == "mse":
                comp = np.median(b / np.where(np.abs(a) > 1e-9, a, np.nan))
            else:
                comp = np.median(a - b)
            srows.append({"rep": rep, "estimator": est, "metric": m,
                          "median_FC_to_X": float(np.median(a)),
                          "median_X_to_FC": float(np.median(b)),
                          "asym_FCwins": float(comp)})
summary = pd.DataFrame(srows)
summary.to_csv(THIS_DIR / "n2_reconstruction_summary.csv", index=False)

print("\n=== Reconstruction: linear_PLS vs KR, FC->X demeaned_pearson (median) ===")
for rep in REPS:
    pls = summary[(summary.rep == rep) & (summary.estimator == "linear_PLS") & (summary.metric == "demeaned_pearson")]["median_FC_to_X"].iloc[0]
    kr  = summary[(summary.rep == rep) & (summary.estimator == "KR") & (summary.metric == "demeaned_pearson")]["median_FC_to_X"].iloc[0]
    print(f"  FC->{rep:8s}  PLS={pls:.4f}  KR={kr:.4f}  Δ(KR-PLS)={kr-pls:+.4f}")
    pls_r = summary[(summary.rep == rep) & (summary.estimator == "linear_PLS") & (summary.metric == "demeaned_pearson")]["median_X_to_FC"].iloc[0]
    kr_r  = summary[(summary.rep == rep) & (summary.estimator == "KR") & (summary.metric == "demeaned_pearson")]["median_X_to_FC"].iloc[0]
    print(f"  {rep:8s}->FC  PLS={pls_r:.4f}  KR={kr_r:.4f}  Δ(KR-PLS)={kr_r-pls_r:+.4f}")

print("\n=== Asymmetry ratio (demeaned_pearson) per estimator ===")
for rep in REPS:
    for est in ["linear_PLS", "KR"]:
        r = summary[(summary.rep == rep) & (summary.estimator == est) & (summary.metric == "demeaned_pearson")]["asym_FCwins"].iloc[0]
        print(f"  {rep:8s} [{est:10s}] FC-wins ratio = {r:.3f}")
