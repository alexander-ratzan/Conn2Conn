#!/usr/bin/env python3
"""E3 — Marginal contribution of r2t over SC for predicting FC.

For each seed (10), fit both:
  - SC -> FC      (baseline; matches main model)
  - [SC || r2t] -> FC   (kitchen sink minus bv/demo)
Record demeaned_pearson per seed for both. Compute per-seed Δ = combined - SC.
Wilcoxon paired test on Δ vs 0.

If median Δ < 0.005, SC is essentially a sufficient statistic for cross-modal
prediction; the bundle-level r2t adds nothing on top of the count-level SC.

If median Δ > 0.02 and Wilcoxon p < 0.05, r2t carries genuinely additional FC-
predictive signal beyond count-SC.

Outputs:
  e3_marginal_results.csv
  e3_marginal_summary.csv  (paired Δ test)
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tract_setup import (load_seed_split_with_r2t, source_blocks,
                          target_train_test, block_pca_pls_predict, full_panel_eval)

THIS_DIR = Path(__file__).resolve().parent
N_SEEDS = 10
# Both arms use the SAME per-block PCA path so SC is represented identically in both:
#   SC      -> one block  [SC]        -> PCA(256) -> PLS
#   SC_r2t  -> two blocks [SC, r2t]   -> PCA(256)+PCA(256) concat -> PLS
# This makes the Δ a clean marginal test of "does r2t's own latent space add FC-
# predictive signal on top of SC's", with no scale-domination artifact.
CONFIGS = ["SC", "SC_r2t"]

rows = []
for seed in range(N_SEEDS):
    print(f"=== seed {seed} ===", flush=True)
    split = load_seed_split_with_r2t(seed=seed)
    FC_tr, FC_te, FC_mean = target_train_test(split, "FC")
    for cfg in CONFIGS:
        btr, bte = source_blocks(split, cfg)
        dims = [b.shape[1] for b in btr]
        print(f"  [{cfg:8s}] blocks={dims} fitting ...", flush=True)
        y_pred = block_pca_pls_predict(btr, bte, FC_tr)
        panel = full_panel_eval(y_pred, FC_te, FC_mean)
        rows.append({"source": cfg, "seed": seed, **panel})
        print(f"    dp={panel['demeaned_pearson']:.4f}")

df = pd.DataFrame(rows)
df.to_csv(THIS_DIR / "e3_marginal_results.csv", index=False)
print(f"\nSaved -> {THIS_DIR / 'e3_marginal_results.csv'}")

# Paired Δ.
sc_only = df[df["source"] == "SC"].sort_values("seed")["demeaned_pearson"].values
combined = df[df["source"] == "SC_r2t"].sort_values("seed")["demeaned_pearson"].values
delta = combined - sc_only
try:
    _, p_two = wilcoxon(delta)
except ValueError:
    p_two = float("nan")
try:
    _, p_greater = wilcoxon(delta, alternative="greater")
except ValueError:
    p_greater = float("nan")
summary = pd.DataFrame([{
    "n_seeds": len(delta),
    "median_dp_SC":      float(np.median(sc_only)),
    "median_dp_SC_r2t":  float(np.median(combined)),
    "median_delta":      float(np.median(delta)),
    "min_delta":         float(delta.min()),
    "max_delta":         float(delta.max()),
    "wilcoxon_p_two_sided": float(p_two),
    "wilcoxon_p_one_sided_greater": float(p_greater),
}])
summary.to_csv(THIS_DIR / "e3_marginal_summary.csv", index=False)
print(f"\n=== Marginal-contribution summary ===")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

if summary["median_delta"].iloc[0] < 0.005:
    verdict = "SC is essentially a sufficient statistic; r2t adds negligible info."
elif summary["median_delta"].iloc[0] >= 0.02 and p_greater < 0.05:
    verdict = "r2t carries genuinely additional FC-predictive signal beyond count-SC."
else:
    verdict = "Modest improvement, mixed evidence; report both numbers."
print(f"\nVerdict: {verdict}")
