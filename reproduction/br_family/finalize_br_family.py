#!/usr/bin/env python3
"""Finalize BR-family — pool Glasser per-seed caches, aggregate (AUC + bootstrap + perm + FDR),
write family_auc_br.csv, and print a BR-vs-PLS sibling-AUC comparison.

Reuses family_mechanism/_fm_common aggregation verbatim (estimator-agnostic — it only consumes
pair sims). Sanity guard: the 4 estimator-independent variants (obs_SC/obs_FC/combined_pred_SC/
bvdemo_to_SC) must match the spine PLS family run (same data/pairing).

    python finalize_br_family.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
FAM = _HERE.parent / "family_mechanism"
sys.path.insert(0, str(FAM))
import _fm_common as fm  # noqa: E402

OUT_PARTS = _HERE / "outputs" / "parts"
OUT_CSV = _HERE / "outputs" / "family_auc_br.csv"
PLS_CSV = FAM / "outputs" / "family_auc.csv"   # spine PLS family run (for comparison)
EST_INDEP = ["obs_SC", "obs_FC", "combined_pred_SC", "bvdemo_to_SC"]


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
    parc = "Glasser"
    pooled, n = pool_parc(parc)
    df = fm.aggregate_family(pooled)
    df.insert(0, "parcellation", parc)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[BR-family] pooled {n} Glasser seeds -> {OUT_CSV} ({len(df)} rows)")

    def auc(d, v, r):
        s = d[(d.variant == v) & (d.relation == r)].auc
        return float(s.iloc[0]) if len(s) else float("nan")

    # sanity guard: estimator-independent variants must match the spine PLS run
    if PLS_CSV.exists():
        pls = pd.read_csv(PLS_CSV); pls = pls[pls.parcellation == "Glasser"]
        print("\n[guard] estimator-independent variants (BR vs PLS-spine sibling AUC, must match):")
        worst = 0.0
        for v in EST_INDEP:
            b, p = auc(df, v, "sibling"), auc(pls, v, "sibling")
            worst = max(worst, abs(b - p) if np.isfinite(b) and np.isfinite(p) else 0)
            print(f"    {v:20s} BR={b:.3f}  PLS={p:.3f}  Δ={abs(b-p):.3e}")
        # 1e-4 tolerance: combined_pred_SC uses iterative BayesianRidge on float32 connectomes,
        # so ~1e-6 AUC jitter vs the spine run is float noise, not a wiring difference.
        print(f"[guard] max Δ on estimator-independent variants = {worst:.2e} "
              f"({'PASS' if worst < 1e-4 else 'WARN — investigate'})")

        print("\n=== BR vs PLS — sibling AUC for the IMPUTED variants (the result) ===")
        for v in ["pred_SC_raw", "pred_SC_resid_bvdemo", "pred_FC_raw", "pred_FC_resid_bvdemo"]:
            b, p = auc(df, v, "sibling"), auc(pls, v, "sibling")
            print(f"    {v:22s} BR={b:.3f}   PLS={p:.3f}   Δ={b-p:+.3f}")

    print("\n=== F6/F7 BR headlines (sibling AUC) ===")
    print(f"  F6: pred_SC_resid_bvdemo={auc(df,'pred_SC_resid_bvdemo','sibling'):.3f}  "
          f"bvdemo_to_SC={auc(df,'bvdemo_to_SC','sibling'):.3f}")
    print(f"  F7: pred_SC_raw={auc(df,'pred_SC_raw','sibling'):.3f}  "
          f"combined_pred_SC={auc(df,'combined_pred_SC','sibling'):.3f}")


if __name__ == "__main__":
    main()
