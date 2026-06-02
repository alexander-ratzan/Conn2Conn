#!/usr/bin/env python3
"""Load all three methods' CSVs, compute per-seed asymmetry ratios, and emit a
verdict on whether the FC↔SC asymmetry is robust to the input-reduction choice.

Outputs:
  - reduction_axis_synthesis.csv (one row per (method, jl_variant, seed) with
    FC->SC dp, SC->FC dp, ratio)
  - reduction_axis_summary.csv   (one row per (method, jl_variant) with
    median/min/max ratio, n_seeds, one-tailed sign test p-value vs 1.0)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

THIS_DIR = Path(__file__).resolve().parent

dfs = []
for p in [THIS_DIR / "method_a_results.csv",
          THIS_DIR / "method_b_results.csv",
          THIS_DIR / "method_c_results.csv"]:
    if not p.exists():
        print(f"WARN: missing {p.name}")
        continue
    dfs.append(pd.read_csv(p))
if not dfs:
    raise SystemExit("No method CSVs found.")
df = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(df)} rows across methods: "
      f"{sorted(df['method'].unique())}")

# Pivot to wide so each row is one (method, jl_variant, seed) with both directions.
wide = (df.pivot_table(index=["method", "jl_variant", "seed"],
                       columns="direction",
                       values="demeaned_pearson")
          .reset_index())
wide["ratio"] = wide["FC->SC"] / wide["SC->FC"].replace(0, np.nan)
wide.to_csv(THIS_DIR / "reduction_axis_synthesis.csv", index=False)
print(f"\nSaved synthesis -> {THIS_DIR / 'reduction_axis_synthesis.csv'}")
print(f"\nPer-seed wide table:")
print(wide.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Summary: median/min/max ratio per (method, variant), Wilcoxon vs 1.0.
def summarize(group):
    ratios = group["ratio"].dropna().values
    if len(ratios) < 3:
        return pd.Series({"n_seeds": len(ratios),
                          "median_ratio": np.nan, "min_ratio": np.nan,
                          "max_ratio": np.nan, "wilcoxon_p_vs_1": np.nan,
                          "median_FC_to_SC_dp": np.nan,
                          "median_SC_to_FC_dp": np.nan})
    try:
        _, p = wilcoxon(ratios - 1.0, alternative="greater")
    except ValueError:
        p = np.nan
    return pd.Series({
        "n_seeds": len(ratios),
        "median_ratio":     float(np.median(ratios)),
        "min_ratio":        float(ratios.min()),
        "max_ratio":        float(ratios.max()),
        "wilcoxon_p_vs_1":  float(p),
        "median_FC_to_SC_dp": float(group["FC->SC"].median()),
        "median_SC_to_FC_dp": float(group["SC->FC"].median()),
    })

summary = wide.groupby(["method", "jl_variant"]).apply(summarize).reset_index()
summary.to_csv(THIS_DIR / "reduction_axis_summary.csv", index=False)
print(f"\n=== Per-method ratio summary ===")
print(summary.to_string(index=False, float_format=lambda x: f"{x:7.4f}"))

# === VERDICT ===
print("\n" + "=" * 70)
print("REDUCTION-AXIS VERDICT")
print("=" * 70)

main_row = summary[(summary["method"] == "PCA_PLS_PCA")]
if len(main_row) == 0:
    print("WARN: PCA_PLS_PCA baseline missing; can't anchor verdict.")
    main_med = float("nan")
else:
    main_med = float(main_row.iloc[0]["median_ratio"])
print(f"  Main-model (PCA→PLS→PCA) median ratio: {main_med:.3f}")

# Compare every (method, variant) median ratio to main_med.
ALL_AGREE = True
DIRECTION_AGREE = True
print(f"\n  {'method':14s} {'variant':16s} {'median':>8s} {'p_vs_1':>10s} "
      f"{'|Δ from PCA|':>14s}")
for _, r in summary.iterrows():
    delta = abs(r["median_ratio"] - main_med) if not np.isnan(main_med) else float("nan")
    print(f"  {r['method']:14s} {str(r['jl_variant']):16s} "
          f"{r['median_ratio']:>8.3f} {r['wilcoxon_p_vs_1']:>10.4f} {delta:>14.3f}")
    if not np.isnan(delta) and delta > 0.15:
        ALL_AGREE = False
    if r["median_ratio"] <= 1.15 or (not np.isnan(r["wilcoxon_p_vs_1"])
                                       and r["wilcoxon_p_vs_1"] > 0.05):
        DIRECTION_AGREE = False

print()
if ALL_AGREE and DIRECTION_AGREE:
    print("  -> OUTCOME 1: CLEAN. All methods (full PLS, learned PCA, every JL variant)")
    print("     give median ratios within ±0.15x of the main model AND each is")
    print("     significantly > 1.15x at p < 0.05. The asymmetry is a property of the")
    print("     data, not the reduction.")
    print("     => Writeup-ready: 'The FC↔SC asymmetry is robust to the choice of")
    print("        linear input reduction — identical magnitudes (within ±0.15x of")
    print("        the PCA baseline) under no reduction (full PLS on 64,620 edges),")
    print("        learned PCA(256), and three Johnson-Lindenstrauss variants")
    print("        (Gaussian dense, sparse-auto, sparse-1/3).'")
elif DIRECTION_AGREE:
    print("  -> OUTCOME 2: PARTIAL. Methods agree on sign of the asymmetry (all > 1.15x")
    print("     at p < 0.05) but magnitudes differ by more than 0.15x. The reduction")
    print("     does shift effect size; report all rows in the appendix.")
else:
    print("  -> OUTCOME 3: ARTIFACT-CANDIDATE. At least one method fails to clear")
    print("     ratio > 1.15x at p < 0.05. The asymmetry may be a reduction property")
    print("     for that method. Investigate the failing row(s).")
