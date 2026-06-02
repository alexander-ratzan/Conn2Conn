#!/usr/bin/env python3
"""Method A — PCA(256)→PLS(64)→inverse-PCA. The main model.

Runs FC→SC and SC→FC on all 10 seeds and writes method_a_results.csv. Identical
pipeline to STEP 11 of the main notebook and to the PLS column of Depth 2
Section A — included here so the synthesis script has all three methods loaded
from one place.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

for _cand in [Path.cwd(),
              Path("/scratch/ans9868/Conn2Conn/notebooks-FC_to_SC-experimental/further_exploration"),
              Path("/Users/user/projects/Conn2Conn/notebooks-FC_to_SC-experimental/further_exploration")]:
    if (_cand / "_setup.py").exists():
        sys.path.insert(0, str(_cand))
        break
else:
    raise RuntimeError(f"_setup.py not found from cwd={Path.cwd()}")
from _setup import load_seed_split, pca_pls_predict, full_panel_eval

THIS_DIR = Path(__file__).resolve().parent
N_SEEDS = 10
K_PCA, K_PLS = 256, 64

rows = []
for seed in range(N_SEEDS):
    sp = load_seed_split(seed=seed)
    FC_tr, FC_te = sp["FC_train"], sp["FC_test"]
    SC_tr, SC_te = sp["SC_train"], sp["SC_test"]
    SC_train_mean = SC_tr.mean(axis=0)
    FC_train_mean = FC_tr.mean(axis=0)

    print(f"[seed {seed}] FC->SC ...", flush=True)
    pred_fs = pca_pls_predict(FC_tr, FC_te, SC_tr, k_src=K_PCA, k_tgt=K_PCA, k_pls=K_PLS)
    panel_fs = full_panel_eval(pred_fs, SC_te, SC_train_mean)
    print(f"  demeaned_pearson = {panel_fs['demeaned_pearson']:.4f}")

    print(f"[seed {seed}] SC->FC ...", flush=True)
    pred_sf = pca_pls_predict(SC_tr, SC_te, FC_tr, k_src=K_PCA, k_tgt=K_PCA, k_pls=K_PLS)
    panel_sf = full_panel_eval(pred_sf, FC_te, FC_train_mean)
    print(f"  demeaned_pearson = {panel_sf['demeaned_pearson']:.4f}")

    rows.append({"method": "PCA_PLS_PCA", "jl_variant": "", "seed": seed,
                 "direction": "FC->SC",
                 "demeaned_pearson": panel_fs["demeaned_pearson"],
                 "top1_acc": panel_fs.get("top1_acc", np.nan),
                 "avg_rank": panel_fs.get("avg_rank", np.nan)})
    rows.append({"method": "PCA_PLS_PCA", "jl_variant": "", "seed": seed,
                 "direction": "SC->FC",
                 "demeaned_pearson": panel_sf["demeaned_pearson"],
                 "top1_acc": panel_sf.get("top1_acc", np.nan),
                 "avg_rank": panel_sf.get("avg_rank", np.nan)})

df = pd.DataFrame(rows)
df.to_csv(THIS_DIR / "method_a_results.csv", index=False)
print(f"\nSaved -> {THIS_DIR / 'method_a_results.csv'}")
print("\nSummary (median across seeds):")
print(df.groupby("direction")["demeaned_pearson"].agg(["median", "min", "max"])
      .to_string(float_format=lambda x: f"{x:.4f}"))
