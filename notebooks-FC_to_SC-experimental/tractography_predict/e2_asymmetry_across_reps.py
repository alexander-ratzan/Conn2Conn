#!/usr/bin/env python3
"""E2 — Cross-modal asymmetry across structural representations.

For each rep X in {SC, r2t, r2t_corr}: compute FC->X and X->FC demeaned_pearson
per seed (10 seeds). Compute per-seed asymmetry ratio FC->X / X->FC, report
median + Wilcoxon p vs 1.0.

Note: r2t is not symmetric (it's a 360×66 region-bundle profile), so "FC->r2t"
is well-defined operationally (PCA(FC)->PLS->inverse-PCA(r2t)) even though r2t
isn't a connectivity matrix in the usual sense. Interpretively asymmetric;
that's the point of the test.

Outputs:
  e2_asymmetry_results.csv   (one row per (rep, seed, direction))
  e2_asymmetry_summary.csv   (median ratio + Wilcoxon per rep)
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tract_setup import (load_seed_split_with_r2t, source_train_test,
                          target_train_test, pca_pls_predict, full_panel_eval)

THIS_DIR = Path(__file__).resolve().parent
N_SEEDS = 10
REPS = ["SC", "r2t", "r2t_corr"]

rows = []
for seed in range(N_SEEDS):
    print(f"=== seed {seed} ===", flush=True)
    split = load_seed_split_with_r2t(seed=seed)
    FC_tr, FC_te, FC_mean = target_train_test(split, "FC")
    for rep in REPS:
        X_tr, X_te = source_train_test(split, rep)
        # FC -> X
        print(f"  [{rep:8s}] FC -> {rep}", flush=True)
        Y_tr, Y_te, Y_mean = target_train_test(split, rep)
        pred = pca_pls_predict(FC_tr, FC_te, Y_tr)
        panel = full_panel_eval(pred, Y_te, Y_mean)
        rows.append({"rep": rep, "seed": seed, "direction": f"FC->{rep}", **panel})
        # X -> FC
        print(f"  [{rep:8s}] {rep} -> FC", flush=True)
        pred2 = pca_pls_predict(X_tr, X_te, FC_tr)
        panel2 = full_panel_eval(pred2, FC_te, FC_mean)
        rows.append({"rep": rep, "seed": seed, "direction": f"{rep}->FC", **panel2})
        print(f"    FC->{rep} dp={panel['demeaned_pearson']:.4f}  "
              f"{rep}->FC dp={panel2['demeaned_pearson']:.4f}  "
              f"ratio={panel['demeaned_pearson']/max(panel2['demeaned_pearson'],1e-9):.3f}")

df = pd.DataFrame(rows)
df.to_csv(THIS_DIR / "e2_asymmetry_results.csv", index=False)
print(f"\nSaved -> {THIS_DIR / 'e2_asymmetry_results.csv'}")

# Summary: per-rep median ratio + Wilcoxon vs 1.0.
summary_rows = []
for rep in REPS:
    fc_to_x = df[(df["rep"] == rep) & (df["direction"] == f"FC->{rep}")].sort_values("seed")
    x_to_fc = df[(df["rep"] == rep) & (df["direction"] == f"{rep}->FC")].sort_values("seed")
    ratios = fc_to_x["demeaned_pearson"].values / np.maximum(x_to_fc["demeaned_pearson"].values, 1e-9)
    try:
        _, p_vs_1 = wilcoxon(ratios - 1.0, alternative="greater")
    except ValueError:
        p_vs_1 = float("nan")
    summary_rows.append({
        "rep": rep,
        "n_seeds": len(ratios),
        "median_ratio": float(np.median(ratios)),
        "min_ratio":    float(ratios.min()),
        "max_ratio":    float(ratios.max()),
        "wilcoxon_p_vs_1": float(p_vs_1),
        "median_FC_to_X_dp": float(fc_to_x["demeaned_pearson"].median()),
        "median_X_to_FC_dp": float(x_to_fc["demeaned_pearson"].median()),
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv(THIS_DIR / "e2_asymmetry_summary.csv", index=False)
print(f"\n=== Asymmetry summary across reps ===")
print(summary.to_string(index=False, float_format=lambda x: f"{x:7.4f}"))
