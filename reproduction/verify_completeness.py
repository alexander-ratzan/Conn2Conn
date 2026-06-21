#!/usr/bin/env python3
"""Verify completeness — diff produced CSVs against configs/expected_cells.csv.

Hard-fails (exit 1) if ANY expected cell is missing, or its primary metric is non-finite.
This is the guard against the silent failure mode (a swallowed error / dropped target leaves a
blank cell that nobody notices until the final table).
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _grid_common import OUTPUTS_DIR, CONFIGS_DIR  # noqa: E402

KEY = ["task", "parcellation", "seed", "variant", "input_set", "target"]


def primary_metric(task, target):
    if task == "reconstruction":
        return "demeaned_pearson"
    return "balanced_acc" if target == "sex" else "pearson"


def main():
    exp = pd.read_csv(CONFIGS_DIR / "expected_cells.csv")
    recon = OUTPUTS_DIR / "reconstruction.csv"
    down = OUTPUTS_DIR / "downstream.csv"
    parts = []
    if recon.exists():
        parts.append(pd.read_csv(recon))
    if down.exists():
        parts.append(pd.read_csv(down))
    if not parts:
        print("[verify] NO output CSVs found — nothing produced"); sys.exit(1)
    act = pd.concat(parts, ignore_index=True)

    exp_keys = exp[KEY].astype(str).agg("|".join, axis=1)
    act_keys = act[KEY].astype(str).agg("|".join, axis=1)
    missing = exp[~exp_keys.isin(set(act_keys))]

    # present-but-non-finite primary metric
    act_idx = act.set_index(act[KEY].astype(str).agg("|".join, axis=1))
    bad = []
    for _, r in exp[exp_keys.isin(set(act_keys))].iterrows():
        k = "|".join(str(r[c]) for c in KEY)
        pm = primary_metric(r["task"], r["target"])
        rows = act_idx.loc[[k]] if k in act_idx.index else None
        if rows is None or pm not in rows or not np.isfinite(pd.to_numeric(rows[pm], errors="coerce")).any():
            bad.append((k, pm))

    print(f"[verify] expected={len(exp)} produced(unique)={act_keys.nunique()} "
          f"missing={len(missing)} nonfinite_primary={len(bad)}")
    if len(missing):
        print("[verify] MISSING cells (first 25):")
        print(missing[KEY].head(25).to_string(index=False))
        by = missing.groupby(["task", "parcellation"]).size()
        print("[verify] missing by task/parc:\n" + by.to_string())
    if bad:
        print("[verify] NON-FINITE primary metric (first 25):")
        for k, pm in bad[:25]:
            print(f"   {k}  ({pm})")
    if len(missing) or bad:
        print("[verify] *** INCOMPLETE GRID — hard fail ***"); sys.exit(1)
    print("[verify] PASS — every expected cell present with finite primary metric.")


if __name__ == "__main__":
    main()
