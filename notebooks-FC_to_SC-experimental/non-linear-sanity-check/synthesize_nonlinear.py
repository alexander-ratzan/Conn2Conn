#!/usr/bin/env python3
"""Synthesize N1+N2+N3 into a linear-vs-nonlinear verdict on whether tractography
carries signal that linear models structurally miss.

Outputs: nonlinear_synthesis_output.txt (printed), nonlinear_synthesis.csv
"""
from pathlib import Path
import pandas as pd
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
n1 = pd.read_csv(THIS_DIR / "n1_cognition_summary.csv")
n2 = pd.read_csv(THIS_DIR / "n2_reconstruction_summary.csv")
n3 = pd.read_csv(THIS_DIR / "n3_marginal_summary.csv")

print("=" * 72); print("NONLINEAR SANITY CHECK — SYNTHESIS"); print("=" * 72)

print("\n### N1 cognition: best lift over bv+demo floor, per estimator (tractography reps)")
TRACT = ["SC", "r2t", "r2t_corr", "SC_r2t"]
n1t = n1[n1["rep"].isin(TRACT)]
for est in ["linear_BR", "HGB", "KR"]:
    s = n1t[n1t["estimator"] == est]
    best = s.loc[s["lift_over_bvdemo"].idxmax()] if len(s) else None
    if best is not None:
        print(f"  [{est:9s}] best lift = {best['lift_over_bvdemo']:+.3f} "
              f"({best['rep']} / {best['target']})  "
              f"-> {'CLEARS floor' if best['lift_over_bvdemo']>=0.03 else 'below floor'}")

print("\n### N2 reconstruction: KR vs PLS on FC->X demeaned_pearson")
for rep in ["SC", "r2t", "r2t_corr"]:
    sub = n2[(n2.rep == rep) & (n2.metric == "demeaned_pearson")]
    pls = sub[sub.estimator == "linear_PLS"]["median_FC_to_X"].iloc[0]
    kr  = sub[sub.estimator == "KR"]["median_FC_to_X"].iloc[0]
    print(f"  FC->{rep:8s}  PLS={pls:.4f}  KR={kr:.4f}  Δ={kr-pls:+.4f}  "
          f"-> {'KR better' if kr-pls>=0.02 else 'no nonlinear gain'}")

print("\n### N3 marginal: r2t over SC, linear vs KR (demeaned_pearson Δ)")
for est in ["linear", "KR"]:
    r = n3[(n3.estimator == est) & (n3.metric == "demeaned_pearson")].iloc[0]
    print(f"  [{est:7s}] Δ(SC_r2t-SC)={r['median_delta']:+.4f} p={r['wilcoxon_p_greater']:.3f}")

# Overall verdict.
print("\n" + "=" * 72); print("VERDICT"); print("=" * 72)
cog_unlock = (n1t[(n1t.estimator != "linear_BR")]["lift_over_bvdemo"] >= 0.03).any()
recon_gain = False
for rep in ["r2t", "r2t_corr"]:
    sub = n2[(n2.rep == rep) & (n2.metric == "demeaned_pearson")]
    pls = sub[sub.estimator == "linear_PLS"]["median_FC_to_X"].iloc[0]
    kr  = sub[sub.estimator == "KR"]["median_FC_to_X"].iloc[0]
    if kr - pls >= 0.02:
        recon_gain = True
kr_marg = n3[(n3.estimator == "KR") & (n3.metric == "demeaned_pearson")].iloc[0]
marg_gain = kr_marg["median_delta"] >= 0.02 and kr_marg["wilcoxon_p_greater"] < 0.05

rows = [
    {"test": "N1 cognition nonlinear unlock", "result": bool(cog_unlock)},
    {"test": "N2 reconstruction nonlinear gain", "result": bool(recon_gain)},
    {"test": "N3 marginal nonlinear gain", "result": bool(marg_gain)},
]
pd.DataFrame(rows).to_csv(THIS_DIR / "nonlinear_synthesis.csv", index=False)

if not (cog_unlock or recon_gain or marg_gain):
    print("  NULL across the board: nonlinear (HGB, KernelRidge) does NOT extract any")
    print("  tractography signal that linear models missed — for cognition, FC")
    print("  reconstruction, OR marginal contribution. The tractography dead-end and")
    print("  the FC↔SC asymmetry are MODEL-CLASS ROBUST. Strongest version of the")
    print("  linear finding.")
else:
    print("  SIGNAL FOUND under nonlinear models:")
    if cog_unlock: print("   - N1: a tractography rep clears the cognition floor nonlinearly.")
    if recon_gain: print("   - N2: nonlinear improves bundle->FC reconstruction.")
    if marg_gain:  print("   - N3: r2t adds nonlinear FC-predictive signal over SC.")
    print("  -> tractography carries nonlinear structure linear models miss. Investigate.")
