#!/usr/bin/env python3
"""A — FC reliability ceiling in the native metric (the "you can't predict FC better
than it predicts itself" bound).

Treats one real FC measurement as the "prediction" of another and runs full_panel_eval,
so demeaned_pearson / pearson / top1_acc / avg_rank / mse / r2 are directly comparable to
every prediction result. top1_acc here = fingerprinting accuracy (A doubles as F1).

Rows per parcellation:
  within_session_run1 : R1LR  vs R1RL   (minutes apart; + phase-encode distortion)
  within_session_run2 : R2LR  vs R2RL
  between_session      : REST1 vs REST2  (~1 day; the TRAIT ceiling — the right
                         denominator for cross-modal trait prediction)

Output: outputs/a_reliability_ceiling.csv
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _noise_common import (load_fc_cells, session_means, group_mean, full_panel_eval,
                           results_dir, PARCELLATIONS,
                           IDX_R1LR, IDX_R1RL, IDX_R2LR, IDX_R2RL)

METRICS = ["demeaned_pearson", "pearson", "top1_acc", "avg_rank", "mse", "r2"]


def panel(pred, true):
    p = full_panel_eval(pred, true, group_mean(true))
    return {m: float(p.get(m, np.nan)) for m in METRICS}


rows = []
for parc in PARCELLATIONS:
    try:
        sids, cells = load_fc_cells(parc)
    except FileNotFoundError:
        print(f"[A] {parc}: no cell cache (run build_fc_cells.py first); skipping", flush=True)
        continue
    n = cells.shape[0]
    rest1, rest2 = session_means(cells)
    print(f"[A] {parc}: n={n}, edges={cells.shape[2]}", flush=True)

    rows.append({"parc": parc, "n": n, "comparison": "within_session_run1",
                 **panel(cells[:, IDX_R1LR], cells[:, IDX_R1RL])})
    rows.append({"parc": parc, "n": n, "comparison": "within_session_run2",
                 **panel(cells[:, IDX_R2LR], cells[:, IDX_R2RL])})
    rows.append({"parc": parc, "n": n, "comparison": "between_session",
                 **panel(rest1, rest2)})

df = pd.DataFrame(rows)
out = results_dir() / "a_reliability_ceiling.csv"
df.to_csv(out, index=False)
print(f"\n[A] saved -> {out}\n")
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\n[A] note: top1_acc on between_session = whole-connectome fingerprinting accuracy.")
print("[A] note: within_session (LR vs RL) carries a phase-encode distortion confound.")
