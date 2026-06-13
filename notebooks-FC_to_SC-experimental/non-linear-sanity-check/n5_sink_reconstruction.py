#!/usr/bin/env python3
"""N5-recon — multimodal SINK residual-boost for FC reconstruction.

Predict FC from [SC ‖ r2t ‖ bv ‖ demo] (everything except FC). Compares:
  sink_linear    per-block PCA + linear PLS template
  sink_residual  template + KernelRidge residual (OOF)
vs the single-best linear source (SC -> FC) as reference.

Full 6-metric panel. Decision: sink_residual − sink_linear >= 0.02 demeaned_pearson
(p<0.05) -> nonlinear cross-modal reconstruction structure.

Outputs: n5_recon_results.csv, n5_recon_summary.csv
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nl_common import (load_seed_split_with_r2t, source_train_test, target_train_test,
                        pca_pls_predict, full_panel_eval)
from _residual import residual_reconstruct_blocks

THIS_DIR = Path(__file__).resolve().parent
N_SEEDS = 10
METRICS = ["demeaned_pearson", "pearson", "top1_acc", "avg_rank", "mse", "r2"]

rows = []
for seed in range(N_SEEDS):
    print(f"=== seed {seed} ===", flush=True)
    split = load_seed_split_with_r2t(seed=seed)
    FC_tr, FC_te, FC_mean = target_train_test(split, "FC")
    # Reference: SC -> FC (best single linear source).
    sc_tr, sc_te = source_train_test(split, "SC")
    sc_pred = pca_pls_predict(sc_tr, sc_te, FC_tr)
    rows.append({"rep": "SC_linear_ref", "seed": seed, **full_panel_eval(sc_pred, FC_te, FC_mean)})
    # Sink: [SC, r2t, bv, demo] -> FC.
    btr = [split["SC_train"], split["r2t_flat_train"], split["bv_train"], split["demo_train"]]
    bte = [split["SC_test"], split["r2t_flat_test"], split["bv_test"], split["demo_test"]]
    final, template = residual_reconstruct_blocks(btr, bte, FC_tr)
    rows.append({"rep": "sink_linear", "seed": seed, **full_panel_eval(template, FC_te, FC_mean)})
    rows.append({"rep": "sink_residual", "seed": seed, **full_panel_eval(final, FC_te, FC_mean)})
    print(f"  SC_ref dp={full_panel_eval(sc_pred,FC_te,FC_mean)['demeaned_pearson']:.4f} "
          f"sink_lin dp={full_panel_eval(template,FC_te,FC_mean)['demeaned_pearson']:.4f} "
          f"sink_res dp={full_panel_eval(final,FC_te,FC_mean)['demeaned_pearson']:.4f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(THIS_DIR / "n5_recon_results.csv", index=False)
print(f"\nSaved -> {THIS_DIR / 'n5_recon_results.csv'}")

summ = df.groupby("rep")[METRICS].median().reset_index()
summ.to_csv(THIS_DIR / "n5_recon_summary.csv", index=False)
print("\n=== Sink reconstruction (median, all metrics) ===")
print(summ.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Decision: residual vs linear sink on demeaned_pearson.
sl = df[df.rep == "sink_linear"].sort_values("seed")["demeaned_pearson"].values
sr = df[df.rep == "sink_residual"].sort_values("seed")["demeaned_pearson"].values
d = sr - sl
try:
    _, p = wilcoxon(d, alternative="greater")
except ValueError:
    p = float("nan")
print(f"\n=== DECISION ===")
print(f"  sink_residual − sink_linear (demeaned_pearson): median={np.median(d):+.4f} p={p:.3f}")
if np.median(d) >= 0.02 and p < 0.05:
    print("  -> nonlinear cross-modal reconstruction structure exists.")
else:
    print("  -> no nonlinear cross-modal reconstruction gain.")
