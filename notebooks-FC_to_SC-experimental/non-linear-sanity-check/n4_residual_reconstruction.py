#!/usr/bin/env python3
"""N4-recon — additive residual (boosted) reconstruction.

Architecture A: template = OOF-PLS, NL (KernelRidge) learns the residual above it,
final = template + NL. Tests whether handing the nonlinear model the linear answer
for free lets it find nonlinear structure linear missed.

For each rep X in {SC, r2t, r2t_corr}, both directions (FC->X, X->FC), 10 seeds:
report the FULL 6-metric panel for BOTH `template` (PLS alone) and `final` (PLS+KR),
so the improvement is measured on every benchmark (demeaned_pearson, pearson,
top1_acc, avg_rank, mse, r2).

Outputs:
  n4_recon_results.csv  (rep, direction, variant{template,final}, seed, 6 metrics)
  n4_recon_summary.csv  (median per rep/direction/variant + final-minus-template delta)
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nl_common import (load_seed_split_with_r2t, source_train_test, target_train_test,
                        full_panel_eval)
from _residual import residual_reconstruct

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
        # FC -> X
        final, template = residual_reconstruct(FC_tr, FC_te, Y_tr)
        rows.append({"rep": rep, "direction": f"FC->{rep}", "variant": "template",
                     "seed": seed, **full_panel_eval(template, Y_te, Y_mean)})
        rows.append({"rep": rep, "direction": f"FC->{rep}", "variant": "final",
                     "seed": seed, **full_panel_eval(final, Y_te, Y_mean)})
        # X -> FC
        final2, template2 = residual_reconstruct(X_tr, X_te, FC_tr)
        rows.append({"rep": rep, "direction": f"{rep}->FC", "variant": "template",
                     "seed": seed, **full_panel_eval(template2, FC_te, FC_mean)})
        rows.append({"rep": rep, "direction": f"{rep}->FC", "variant": "final",
                     "seed": seed, **full_panel_eval(final2, FC_te, FC_mean)})
        dp_t = full_panel_eval(template, Y_te, Y_mean)["demeaned_pearson"]
        dp_f = full_panel_eval(final, Y_te, Y_mean)["demeaned_pearson"]
        print(f"  [{rep:8s}] FC->{rep}: template dp={dp_t:.4f} final dp={dp_f:.4f} "
              f"Δ={dp_f-dp_t:+.4f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(THIS_DIR / "n4_recon_results.csv", index=False)
print(f"\nSaved -> {THIS_DIR / 'n4_recon_results.csv'}")

# Summary: per (rep, direction, metric) median template vs final + paired Δ + Wilcoxon.
srows = []
for rep in REPS:
    for direction in [f"FC->{rep}", f"{rep}->FC"]:
        t = df[(df.rep == rep) & (df.direction == direction) & (df.variant == "template")].sort_values("seed")
        f = df[(df.rep == rep) & (df.direction == direction) & (df.variant == "final")].sort_values("seed")
        for m in METRICS:
            # improvement direction: higher-better metrics + r2 use final-template;
            # mse (lower-better) uses template-final so positive = improvement.
            if m == "mse":
                delta = t[m].values - f[m].values
            else:
                delta = f[m].values - t[m].values
            try:
                _, p = wilcoxon(delta, alternative="greater")
            except ValueError:
                p = float("nan")
            srows.append({"rep": rep, "direction": direction, "metric": m,
                          "median_template": float(t[m].median()),
                          "median_final": float(f[m].median()),
                          "median_improvement": float(np.median(delta)),
                          "wilcoxon_p_improve": float(p)})
summary = pd.DataFrame(srows)
summary.to_csv(THIS_DIR / "n4_recon_summary.csv", index=False)

print("\n=== Improvement of final (PLS+KR) over template (PLS), all metrics ===")
for rep in REPS:
    for direction in [f"FC->{rep}", f"{rep}->FC"]:
        print(f"\n  {direction}:")
        sub = summary[(summary.rep == rep) & (summary.direction == direction)]
        for _, r in sub.iterrows():
            star = "*" if r["wilcoxon_p_improve"] < 0.05 else " "
            print(f"    {r['metric']:18s} template={r['median_template']:+.4f} "
                  f"final={r['median_final']:+.4f} Δ={r['median_improvement']:+.4f} "
                  f"p={r['wilcoxon_p_improve']:.3f}{star}")
