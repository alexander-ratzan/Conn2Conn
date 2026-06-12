#!/usr/bin/env python3
"""Synthesize E1+E2+E3 results into a single comparison table and a verdict.

E1 (source-rep comparison): which structural rep predicts FC best?
E2 (asymmetry): does the ~1.55x FC<->SC asymmetry hold under r2t and r2t_corr?
E3 (marginal): does r2t add measurable info on top of count-SC?

Outputs:
  tractography_synthesis.csv  (compact summary table)
  tractography_synthesis_output.txt  (printed verdict)
"""
from pathlib import Path
import pandas as pd
import numpy as np

THIS_DIR = Path(__file__).resolve().parent

e1 = pd.read_csv(THIS_DIR / "e1_source_rep_results.csv")
e2 = pd.read_csv(THIS_DIR / "e2_asymmetry_summary.csv")
e3 = pd.read_csv(THIS_DIR / "e3_marginal_summary.csv")


def e2_dp(e2df, rep):
    """median demeaned_pearson FC-wins ratio for a rep (handles long-format summary)."""
    row = e2df[(e2df["rep"] == rep) & (e2df["metric"] == "demeaned_pearson")]
    return row["median_FCwins"].iloc[0] if len(row) else float("nan")

ALL_METRICS = ["demeaned_pearson", "pearson", "top1_acc", "avg_rank", "mse", "r2"]
print("=== E1: source-rep -> FC prediction (median across 10 seeds, ALL metrics) ===")
e1_med = e1.groupby("rep")[ALL_METRICS].median()
print(e1_med.to_string(float_format=lambda x: f"{x:.4f}"))

print("\n=== E2: asymmetry across structural reps (ALL 6 metrics) ===")
# e2 summary is long-format (rep, metric, ...). Pivot median_FCwins for readability.
if "metric" in e2.columns:
    piv = e2.pivot(index="rep", columns="metric", values="median_FCwins")
    piv = piv.reindex(columns=[m for m in ALL_METRICS if m in piv.columns])
    print("median FC-wins (ratio>1 or r2-diff>0 means FC->X beats X->FC):")
    print(piv.to_string(float_format=lambda x: f"{x:+.3f}"))
    pivp = e2.pivot(index="rep", columns="metric", values="wilcoxon_p_FCwins")
    pivp = pivp.reindex(columns=[m for m in ALL_METRICS if m in pivp.columns])
    print("\nWilcoxon p (one-sided, FC wins):")
    print(pivp.to_string(float_format=lambda x: f"{x:.4f}"))
else:
    print(e2.to_string(index=False, float_format=lambda x: f"{x:7.4f}"))

print("\n=== E3: marginal r2t contribution over SC ===")
print(e3.to_string(index=False, float_format=lambda x: f"{x:7.4f}"))

# Compact synthesis CSV.
synth = pd.DataFrame({
    "row": [
        "E1: SC -> FC median dp",
        "E1: r2t -> FC median dp",
        "E1: r2t_corr -> FC median dp",
        "E1: SC_r2t -> FC median dp",
        "E1: kitchen_sink -> FC median dp",
        "E2: FC<->SC median ratio",
        "E2: FC<->r2t median ratio",
        "E2: FC<->r2t_corr median ratio",
        "E3: median Δ (SC_r2t - SC)",
    ],
    "value": [
        float(e1[e1["rep"] == "SC"]["demeaned_pearson"].median()),
        float(e1[e1["rep"] == "r2t"]["demeaned_pearson"].median()),
        float(e1[e1["rep"] == "r2t_corr"]["demeaned_pearson"].median()),
        float(e1[e1["rep"] == "SC_r2t"]["demeaned_pearson"].median()),
        float(e1[e1["rep"] == "kitchen_sink"]["demeaned_pearson"].median()),
        float(e2_dp(e2, "SC")),
        float(e2_dp(e2, "r2t")),
        float(e2_dp(e2, "r2t_corr")),
        float(e3["median_delta"].iloc[0]),
    ],
})
synth.to_csv(THIS_DIR / "tractography_synthesis.csv", index=False)
print(f"\nSaved -> {THIS_DIR / 'tractography_synthesis.csv'}")

print("\n" + "=" * 72)
print("AUTOMATED VERDICT")
print("=" * 72)

sc_e1   = float(e1[e1["rep"] == "SC"]["demeaned_pearson"].median())
r2t_e1  = float(e1[e1["rep"] == "r2t"]["demeaned_pearson"].median())
sc_ratio = float(e2_dp(e2, "SC"))
r2t_ratio = float(e2_dp(e2, "r2t"))
delta_e3 = float(e3["median_delta"].iloc[0])

print(f"\n[E1] SC vs r2t for predicting FC:")
print(f"  SC    median dp = {sc_e1:.4f}")
print(f"  r2t   median dp = {r2t_e1:.4f}")
if r2t_e1 >= sc_e1 - 0.005:
    print("  -> r2t matches or beats SC for FC prediction; the bundle-level data is")
    print("     at least as informative as the count-level SC.")
else:
    print("  -> SC predicts FC better than r2t. Count-level SC retains a real edge")
    print("     over the bundle-level representation for cross-modal prediction.")

print(f"\n[E2] Asymmetry across structural reps:")
print(f"  FC<->SC ratio       = {sc_ratio:.3f}")
print(f"  FC<->r2t ratio      = {r2t_ratio:.3f}")
if abs(r2t_ratio - sc_ratio) <= 0.20:
    print("  -> Asymmetry magnitude is preserved under r2t. The FC->SC > SC->FC effect")
    print("     is NOT a parcellation/count artifact — it persists in the bundle")
    print("     representation derived from the same tractography.")
elif r2t_ratio < sc_ratio - 0.20:
    print("  -> Asymmetry SHRINKS under r2t. A portion of FC↔SC asymmetry was due to")
    print("     parcellation information loss in count-SC.")
else:
    print("  -> Asymmetry GROWS under r2t (unusual). The bundle rep amplifies the")
    print("     directional difference; worth digging into why.")

print(f"\n[E3] Marginal r2t over SC for FC prediction (paired Δ):")
print(f"  median Δ (SC_r2t - SC) = {delta_e3:+.4f}")
if delta_e3 < 0.005:
    print("  -> SC is essentially a sufficient statistic. r2t adds nothing material")
    print("     beyond count-SC for cross-modal prediction.")
elif delta_e3 >= 0.02:
    print("  -> r2t carries genuinely additional FC-predictive signal.")
else:
    print("  -> Modest improvement; below 0.02 dp ceiling.")
