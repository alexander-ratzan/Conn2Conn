#!/usr/bin/env python3
"""E1 — Source-representation comparison: predict FC from each of five structural reps.

For each seed (10), for each source rep X, fit PCA(K_PCA=256) -> PLS(K_PLS=64) ->
inverse-PCA(FC) and record the full 6-metric panel.

Reps:
  SC            — current main-model baseline; (n_subj, 64,620)
  r2t           — flattened region-to-tract profile; (n_subj, 23,760)
  r2t_corr      — upper-tri of region-region similarity by tract; (n_subj, 64,620)
  SC_r2t        — concat [SC || r2t]; (n_subj, 88,380)
  kitchen_sink  — concat [r2t || SC || bv || demo]; (n_subj, ~89,400+)

Outputs:
  e1_source_rep_results.csv  (one row per (rep, seed) with full panel)
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tract_setup import (load_seed_split_with_r2t, source_train_test,
                          target_train_test, pca_pls_predict, full_panel_eval)

THIS_DIR = Path(__file__).resolve().parent
N_SEEDS = 10
REPS = ["SC", "r2t", "r2t_corr", "SC_r2t", "kitchen_sink"]
TARGET = "FC"

rows = []
for seed in range(N_SEEDS):
    print(f"=== seed {seed} ===", flush=True)
    split = load_seed_split_with_r2t(seed=seed)
    Y_tr, Y_te, Y_train_mean = target_train_test(split, TARGET)
    for rep in REPS:
        X_tr, X_te = source_train_test(split, rep)
        print(f"  [{rep:13s}] X={X_tr.shape}  fitting ...", flush=True)
        y_pred = pca_pls_predict(X_tr, X_te, Y_tr)
        panel = full_panel_eval(y_pred, Y_te, Y_train_mean)
        rows.append({"rep": rep, "seed": seed, "target": TARGET, **panel})
        print(f"    dp={panel['demeaned_pearson']:.4f}  "
              f"r2={panel.get('r2', float('nan')):.4f}  "
              f"top1={panel.get('top1_acc', float('nan')):.4f}  "
              f"rank={panel.get('avg_rank', float('nan')):.4f}")

df = pd.DataFrame(rows)
df.to_csv(THIS_DIR / "e1_source_rep_results.csv", index=False)
print(f"\nSaved -> {THIS_DIR / 'e1_source_rep_results.csv'}")
print("\n=== median per rep (10 seeds) ===")
print(df.groupby("rep")[["demeaned_pearson", "r2", "top1_acc", "avg_rank"]]
        .median().to_string(float_format=lambda x: f"{x:.4f}"))
