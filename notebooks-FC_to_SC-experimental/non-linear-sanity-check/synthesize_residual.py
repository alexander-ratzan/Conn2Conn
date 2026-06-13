#!/usr/bin/env python3
"""Synthesize N4 residual-boost results (reconstruction + cognition) into a verdict:
does handing the nonlinear model the linear prediction for free unlock any signal?
"""
from pathlib import Path
import pandas as pd
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
rec = pd.read_csv(THIS_DIR / "n4_recon_summary.csv")
cog = pd.read_csv(THIS_DIR / "n4_cog_summary.csv")

print("=" * 72); print("RESIDUAL-BOOST (architecture A) — SYNTHESIS"); print("=" * 72)

print("\n### Reconstruction: improvement of final (PLS+KR) over template (PLS)")
print("    (demeaned_pearson; >0 and p<0.05 = nonlinear residual adds signal)")
recon_hit = False
for _, r in rec[rec.metric == "demeaned_pearson"].iterrows():
    star = "*" if r["wilcoxon_p_improve"] < 0.05 else " "
    if r["median_improvement"] >= 0.02 and r["wilcoxon_p_improve"] < 0.05:
        recon_hit = True
    print(f"  {r['direction']:12s} tmpl={r['median_template']:+.4f} "
          f"final={r['median_final']:+.4f} Δ={r['median_improvement']:+.4f} "
          f"p={r['wilcoxon_p_improve']:.3f}{star}")

print("\n### Cognition: tractography lift over bv+demo floor (final = BR+KR)")
cog_hit = False
TRACT = ["SC", "r2t", "r2t_corr", "SC_r2t"]
for target in cog["target"].unique():
    sub = cog[cog.target == target]
    floor = sub[(sub.rep == "bv+demo") & (sub.variant == "final")]["pearson"].iloc[0]
    print(f"  {target} (floor={floor:.3f}):")
    for rep in TRACT + ["FC"]:
        f = sub[(sub.rep == rep) & (sub.variant == "final")]["pearson"].iloc[0]
        lift = f - floor
        if rep in TRACT and lift >= 0.02:
            cog_hit = True
        print(f"    {rep:9s} final={f:+.3f} lift={lift:+.3f}")

print("\n" + "=" * 72); print("VERDICT"); print("=" * 72)
pd.DataFrame([{"recon_unlock": bool(recon_hit), "cognition_unlock": bool(cog_hit)}]).to_csv(
    THIS_DIR / "residual_synthesis.csv", index=False)
if not (recon_hit or cog_hit):
    print("  NULL. Even with the linear prediction handed in as a free template and the")
    print("  nonlinear model tasked ONLY with the residual above it (OOF, no leakage),")
    print("  KernelRidge finds no improvement — not in reconstruction, not in cognition.")
    print("  The tractography dead-end survives the most sensitive nonlinear probe we can")
    print("  construct. Definitive: the signal is not there, linearly or nonlinearly.")
else:
    print("  SIGNAL. Residual-boost unlocked improvement:")
    if recon_hit: print("   - reconstruction: final beats PLS template (nonlinear residual structure).")
    if cog_hit:   print("   - cognition: a tractography rep clears the floor under residual-boost.")
    print("  -> nonlinear structure exists above the linear template. Investigate.")
