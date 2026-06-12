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

# Per-metric asymmetry, all 6 metrics, with correct directionality so that the
# reported "FC-favoring" comparison is >1 (or >0 for r2) whenever FC->X beats X->FC.
#   higher-better (ratio FC->X / X->FC):  demeaned_pearson, pearson, top1_acc, avg_rank
#   mse (lower-better, ratio X->FC / FC->X): mse
#   r2 (can be negative, use difference FC->X - X->FC): r2
HIGHER_BETTER = ["demeaned_pearson", "pearson", "top1_acc", "avg_rank"]
LOWER_BETTER  = ["mse"]
DIFF_METRICS  = ["r2"]
ALL_METRICS = HIGHER_BETTER + LOWER_BETTER + DIFF_METRICS

summary_rows = []
for rep in REPS:
    fc = df[(df["rep"] == rep) & (df["direction"] == f"FC->{rep}")].sort_values("seed")
    xf = df[(df["rep"] == rep) & (df["direction"] == f"{rep}->FC")].sort_values("seed")
    for metric in ALL_METRICS:
        a = fc[metric].values  # FC->X
        b = xf[metric].values  # X->FC
        if metric in HIGHER_BETTER:
            comp = a / np.where(np.abs(b) > 1e-9, b, np.nan)
            kind, null = "ratio_FCwins", 1.0
        elif metric in LOWER_BETTER:
            comp = b / np.where(np.abs(a) > 1e-9, a, np.nan)  # flipped: >1 = FC wins
            kind, null = "ratio_FCwins", 1.0
        else:  # r2 difference
            comp = a - b
            kind, null = "diff_FCwins", 0.0
        comp = comp[~np.isnan(comp)]
        try:
            _, p = wilcoxon(comp - null, alternative="greater")
        except ValueError:
            p = float("nan")
        summary_rows.append({
            "rep": rep, "metric": metric, "comparison_kind": kind,
            "n_seeds": len(comp),
            "median_FCwins": float(np.median(comp)) if len(comp) else float("nan"),
            "min_FCwins":    float(np.min(comp)) if len(comp) else float("nan"),
            "max_FCwins":    float(np.max(comp)) if len(comp) else float("nan"),
            "wilcoxon_p_FCwins": float(p),
            "median_FC_to_X": float(fc[metric].median()),
            "median_X_to_FC": float(xf[metric].median()),
        })
summary = pd.DataFrame(summary_rows)
summary.to_csv(THIS_DIR / "e2_asymmetry_summary.csv", index=False)
print(f"\n=== Asymmetry across reps — ALL 6 metrics ===")
print("(median_FCwins: ratio>1 or diff>0 means FC->X beats X->FC; p = Wilcoxon one-sided)")
for rep in REPS:
    print(f"\n  {rep}:")
    sub = summary[summary["rep"] == rep]
    for _, r in sub.iterrows():
        star = "*" if (r["wilcoxon_p_FCwins"] < 0.05) else " "
        unit = "x" if r["comparison_kind"] == "ratio_FCwins" else " (diff)"
        print(f"    {r['metric']:18s} {r['median_FCwins']:+7.3f}{unit:7s} "
              f"p={r['wilcoxon_p_FCwins']:.4f}{star}  "
              f"[FC->X={r['median_FC_to_X']:+.4f}  X->FC={r['median_X_to_FC']:+.4f}]")
