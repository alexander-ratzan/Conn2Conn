#!/usr/bin/env python3
"""Finalize F6/F7 — pool per-unit family caches per parcellation and aggregate.

Faithful port of STEP 8.3: pool per-seed pair sims, z-score by unrelated, AUC + bootstrap CI
+ permutation p + BH-FDR (RNG: perm default_rng(42), boot default_rng(43)). Writes one merged
CSV with a `parcellation` column. Includes a regression guard: the Glasser numbers must match
the notebook's aggregate_auc.csv (the canonical single-run result) within tolerance.

    python finalize_fm.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import _fm_common as fm

OUT_PARTS = _HERE / "outputs" / "parts"
OUT_CSV = _HERE / "outputs" / "family_auc.csv"
_NB_AGG = _HERE.parents[1] / (
    "notebooks-FC_to_SC-experimental/model_overviews/results/family_structure_phase2/aggregate_auc.csv")


def pool_parc(parc):
    files = sorted((OUT_PARTS / parc).glob("family_seed*.npz"))
    assert files, f"no family_seed*.npz for {parc} under {OUT_PARTS/parc}"
    pooled = {v: {r: [] for r in fm.RELATIONS} for v in fm.FAM_VARIANTS}
    for f in files:
        data = np.load(f)
        for v in fm.FAM_VARIANTS:
            for r in fm.RELATIONS:
                key = f"{v}__{r}"
                if key in data.files and data[key].size:
                    pooled[v][r].append(data[key])
    pooled = {v: {r: (np.concatenate(lst) if lst else np.array([]))
                  for r, lst in by.items()} for v, by in pooled.items()}
    return pooled, len(files)


def main():
    parcs = [p for p in fm.PARCELLATIONS if (OUT_PARTS / p).exists()]
    assert parcs, f"no parcellation part dirs under {OUT_PARTS}"
    frames = []
    for parc in parcs:
        pooled, n = pool_parc(parc)
        df = fm.aggregate_family(pooled)
        df.insert(0, "parcellation", parc)
        frames.append(df)
        print(f"[{parc}] pooled {n} seeds -> {len(df)} (variant,relation) rows")
    out = pd.concat(frames, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV} ({len(out)} rows)")

    # regression guard: Glasser must match the notebook's canonical single-run aggregate
    if "Glasser" in parcs and _NB_AGG.exists():
        ref = pd.read_csv(_NB_AGG)
        got = out[out.parcellation == "Glasser"]
        m = ref.merge(got, on=["variant", "relation"], suffixes=("_ref", "_got"))
        err = float((m.auc_ref - m.auc_got).abs().max())
        print(f"\n[guard] Glasser vs notebook aggregate_auc.csv: max |AUC err| = {err:.2e}")
        if err < 1e-6:
            print("[guard] PASS — Glasser reproduces the notebook exactly.")
        else:
            print(f"[guard] WARN — Glasser AUC drift {err:.2e} (investigate split/pairing).")

    # headline lines
    def auc(parc, v, r):
        s = out[(out.parcellation == parc) & (out.variant == v) & (out.relation == r)].auc
        return float(s.iloc[0]) if len(s) else float("nan")
    print("\n=== Headlines (sibling AUC) ===")
    for parc in parcs:
        print(f"  [{parc}] F6 pred_SC_resid_bvdemo={auc(parc,'pred_SC_resid_bvdemo','sibling'):.3f}  "
              f"bvdemo_to_SC={auc(parc,'bvdemo_to_SC','sibling'):.3f}  |  "
              f"F7 pred_SC_raw={auc(parc,'pred_SC_raw','sibling'):.3f}  "
              f"combined_pred_SC={auc(parc,'combined_pred_SC','sibling'):.3f}")


if __name__ == "__main__":
    main()
