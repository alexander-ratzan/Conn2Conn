#!/usr/bin/env python3
"""N3 — Nonlinear marginal contribution of r2t over SC.

Does adding the bundle representation help predict FC under a NONLINEAR estimator,
even though it didn't under linear PLS (E3: Δ≈0)? If r2t's contribution is conjunctive
(interactions with SC that linear can't see), a nonlinear combined model would show a
positive Δ where the linear one didn't.

Per seed, two estimators, two source configs:
  SC      -> FC   (one block)
  SC_r2t  -> FC   (two blocks, per-block PCA)
Estimators: linear (block_pca_pls) vs KR (kernelridge_blocks). Full 6-metric panel.
Paired Δ (SC_r2t - SC) per estimator, Wilcoxon.

DECISION RULE: KR Δ(SC_r2t - SC) >= +0.02 demeaned_pearson, Wilcoxon p<0.05 ->
r2t adds nonlinear FC-predictive signal on top of SC.

Outputs:
  n3_marginal_results.csv  (source, estimator, seed, all 6 metrics)
  n3_marginal_summary.csv  (paired Δ per estimator)
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nl_common import (load_seed_split_with_r2t, source_blocks, target_train_test,
                        block_pca_pls_predict, kernelridge_blocks_predict, full_panel_eval)

THIS_DIR = Path(__file__).resolve().parent
N_SEEDS = 10
CONFIGS = ["SC", "SC_r2t"]
METRICS = ["demeaned_pearson", "pearson", "top1_acc", "avg_rank", "mse", "r2"]

rows = []
for seed in range(N_SEEDS):
    print(f"=== seed {seed} ===", flush=True)
    split = load_seed_split_with_r2t(seed=seed)
    FC_tr, FC_te, FC_mean = target_train_test(split, "FC")
    for cfg in CONFIGS:
        btr, bte = source_blocks(split, cfg)
        for est, fn in [("linear", block_pca_pls_predict), ("KR", kernelridge_blocks_predict)]:
            pred = fn(btr, bte, FC_tr)
            panel = full_panel_eval(pred, FC_te, FC_mean)
            rows.append({"source": cfg, "estimator": est, "seed": seed, **panel})
            print(f"  [{cfg:8s}|{est:7s}] dp={panel['demeaned_pearson']:.4f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(THIS_DIR / "n3_marginal_results.csv", index=False)
print(f"\nSaved -> {THIS_DIR / 'n3_marginal_results.csv'}")

srows = []
for est in ["linear", "KR"]:
    sc = df[(df.source == "SC") & (df.estimator == est)].sort_values("seed")
    cb = df[(df.source == "SC_r2t") & (df.estimator == est)].sort_values("seed")
    for m in METRICS:
        delta = cb[m].values - sc[m].values
        try:
            _, p = wilcoxon(delta, alternative="greater")
        except ValueError:
            p = float("nan")
        srows.append({"estimator": est, "metric": m,
                      "median_SC": float(np.median(sc[m])),
                      "median_SC_r2t": float(np.median(cb[m])),
                      "median_delta": float(np.median(delta)),
                      "wilcoxon_p_greater": float(p)})
summary = pd.DataFrame(srows)
summary.to_csv(THIS_DIR / "n3_marginal_summary.csv", index=False)
print("\n=== Marginal Δ (SC_r2t - SC) per estimator, all metrics ===")
for est in ["linear", "KR"]:
    print(f"\n  [{est}]")
    for _, r in summary[summary.estimator == est].iterrows():
        star = "*" if r["wilcoxon_p_greater"] < 0.05 else " "
        print(f"    {r['metric']:18s} SC={r['median_SC']:+.4f} SC_r2t={r['median_SC_r2t']:+.4f} "
              f"Δ={r['median_delta']:+.4f} p={r['wilcoxon_p_greater']:.3f}{star}")

dp_kr = summary[(summary.estimator == "KR") & (summary.metric == "demeaned_pearson")].iloc[0]
print("\n=== DECISION ===")
if dp_kr["median_delta"] >= 0.02 and dp_kr["wilcoxon_p_greater"] < 0.05:
    print(f"  KR Δ={dp_kr['median_delta']:+.4f} (p={dp_kr['wilcoxon_p_greater']:.3f}) -> r2t adds "
          "nonlinear FC-predictive signal over SC.")
else:
    print(f"  KR Δ={dp_kr['median_delta']:+.4f} (p={dp_kr['wilcoxon_p_greater']:.3f}) -> r2t adds nothing "
          "over SC even nonlinearly; count-SC remains a sufficient statistic.")
