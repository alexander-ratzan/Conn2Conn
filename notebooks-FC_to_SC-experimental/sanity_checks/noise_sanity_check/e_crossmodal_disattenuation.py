#!/usr/bin/env python3
"""E — cross-modal disattenuation: SC->FC achieved vs the FC reliability ceiling.

"Of the reproducible FC signal, what fraction does SC actually explain?"
Needs only FC reliability (have it) — the SC-target direction (FC->SC / SC reliability)
is NOT computable (no SC retest) and is deliberately omitted.

  achieved   = SC -> FC  demeaned_pearson (PCA->PLS, median over seeds, test set)
  reference  = bv+demo -> FC  (the demographic floor, same pipeline)
  ceiling    = between-session FC reliability (REST1<->REST2), from A
  fraction   = achieved / ceiling   per metric

Output: outputs/e_crossmodal_disattenuation.csv
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _noise_common import (load_fc_cells, session_means, group_mean, full_panel_eval,
                           pca_pls_predict, load_seed_split, results_dir)

N_SEEDS = 10
METRICS = ["demeaned_pearson", "pearson", "top1_acc", "avg_rank"]  # higher-better subset


def ceiling_glasser():
    """Between-session FC reliability on the Glasser 4-cell cache (matches main pipeline)."""
    sids, cells = load_fc_cells("Glasser")
    rest1, rest2 = session_means(cells)
    p = full_panel_eval(rest1, rest2, group_mean(rest2))
    return {m: float(p.get(m, np.nan)) for m in METRICS}


def achieved(source):
    """median test-set panel for source->FC across seeds, via the main PCA->PLS pipeline."""
    accum = {m: [] for m in METRICS}
    for seed in range(N_SEEDS):
        sp = load_seed_split(seed=seed)
        FC_tr, FC_te = sp["FC_train"], sp["FC_test"]
        if source == "SC":
            X_tr, X_te = sp["SC_train"], sp["SC_test"]
        elif source == "bv+demo":
            X_tr = np.concatenate([sp["bv_train"], sp["demo_train"]], axis=1)
            X_te = np.concatenate([sp["bv_test"], sp["demo_test"]], axis=1)
        else:
            raise ValueError(source)
        # cap components for low-dim inputs (bv+demo is ~26 features, not 256)
        k = min(256, X_tr.shape[1])
        pred = pca_pls_predict(X_tr, X_te, FC_tr, k_src=k, k_pls=min(64, k))
        p = full_panel_eval(pred, FC_te, FC_tr.mean(axis=0))
        for m in METRICS:
            accum[m].append(float(p.get(m, np.nan)))
    return {m: float(np.median(v)) for m, v in accum.items()}


print("[E] computing FC reliability ceiling (Glasser, between-session) ...", flush=True)
ceil = ceiling_glasser()
print(f"    ceiling: {ceil}", flush=True)
print("[E] computing SC->FC achieved (10 seeds) ...", flush=True)
sc = achieved("SC")
print("[E] computing bv+demo->FC reference (10 seeds) ...", flush=True)
bd = achieved("bv+demo")

rows = []
for label, ach in [("SC->FC", sc), ("bv+demo->FC", bd)]:
    for m in METRICS:
        c = ceil[m]
        rows.append({"source": label, "metric": m,
                     "achieved": ach[m], "ceiling": c,
                     "fraction_of_ceiling": (ach[m] / c) if c and not np.isnan(c) else np.nan})
df = pd.DataFrame(rows)
out = results_dir() / "e_crossmodal_disattenuation.csv"
df.to_csv(out, index=False)
print(f"\n[E] saved -> {out}\n")
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\n[E] fraction_of_ceiling = how much of the REPRODUCIBLE FC signal the source captures.")
print("[E] NOTE: FC->SC / SC-reliability is omitted — no SC test-retest in this dataset.")
