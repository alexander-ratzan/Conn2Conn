#!/usr/bin/env python3
"""Pool latent-direct seeds -> scorecard_latent.csv, and diff vs the obj_functions round-trip.

The headline: for matched (objective) on the FC2SC arm, Δ = latent-direct − round-trip on
CogCryst lift and sibling AUC. If ≈0, the inverse-PCA upscale was harmless; if latent-direct is
higher, the round-trip was attenuating the objective (the hypothesis under test).
"""
from pathlib import Path
import sys
import glob
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "family_mechanism"))
import _fm_common as fm   # noqa: E402  (auc_vs_unrelated, RELATIONS — torch-free)

PARTS = HERE / "outputs" / "parts"
OUT = HERE / "outputs"
RT = HERE.parent / "obj_functions" / "outputs" / "scorecard.csv"   # round-trip reference


def main():
    cog = pd.concat([pd.read_csv(f) for f in glob.glob(str(PARTS / "cog_s*.csv"))], ignore_index=True)
    nseed = cog.seed.nunique()

    # pool identity sims across seeds, per (arm, estimator) -> sibling AUC
    pooled = {}
    for f in glob.glob(str(PARTS / "family" / "lat_family_s*.npz")):
        d = np.load(f)
        for k in d.files:
            if "__" not in k:
                continue
            pooled.setdefault(k, []).append(d[k])
    sib = {}
    for k, lst in pooled.items():
        arm, est, rel = k.split("__")
        sib.setdefault((arm, est), {})[rel] = np.concatenate([a for a in lst if a.size]) if any(a.size for a in lst) else np.array([])
    sib_auc = {ae: fm.auc_vs_unrelated(d).get("sibling", np.nan) for ae, d in sib.items()}

    # cognition lift means (over seeds) per (arm, estimator, target)
    g = cog.groupby(["arm", "estimator", "target"]).lift_over_bvdemo.mean().reset_index()
    rows = []
    for (arm, est), grp in cog.groupby(["arm", "estimator"]):
        rec = {"arm": arm, "estimator": est, "sib_AUC": sib_auc.get((arm, est), np.nan)}
        for t in ["CogCryst", "CogTotal", "CogFluid"]:
            v = g[(g.arm == arm) & (g.estimator == est) & (g.target == t)].lift_over_bvdemo
            rec[f"lift_{t}"] = float(v.iloc[0]) if len(v) else np.nan
        rows.append(rec)
    sc = pd.DataFrame(rows).sort_values(["arm", "estimator"])
    OUT.mkdir(parents=True, exist_ok=True)
    sc.to_csv(OUT / "scorecard_latent.csv", index=False)
    print(f"pooled {nseed} seeds -> scorecard_latent.csv ({len(sc)} rows)\n")
    print(sc.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # --- Δ vs round-trip (FC2SC arm vs obj_functions scorecard) ---
    if RT.exists():
        rt = pd.read_csv(RT)
        # obj_functions scorecard cols: estimator, recon_dr/recon, sib_AUC, cog lift (name varies) — match leniently
        rt_cols = {c.lower(): c for c in rt.columns}
        sib_c = next((rt_cols[c] for c in rt_cols if "sib" in c), None)
        cog_c = next((rt_cols[c] for c in rt_cols if "cog" in c or "lift" in c), None)
        print("\n=== Δ (latent-direct FC2SC − round-trip obj_functions) ===")
        lat = sc[sc.arm == "FC2SC"].set_index("estimator")
        rtx = rt.set_index(rt_cols.get("estimator", rt.columns[0]))
        for est in lat.index:
            if est in rtx.index:
                d_sib = lat.loc[est, "sib_AUC"] - (rtx.loc[est, sib_c] if sib_c else np.nan)
                d_cog = lat.loc[est, "lift_CogCryst"] - (rtx.loc[est, cog_c] if cog_c else np.nan)
                print(f"  {est:10s} Δsib_AUC={d_sib:+.3f}  ΔCogCryst_lift={d_cog:+.3f}")
        print("\n(Δ≈0 → round-trip harmless; latent-direct higher → upscale was attenuating the objective)")
    else:
        print(f"\n[note] round-trip reference {RT} not found; skipping Δ.")


if __name__ == "__main__":
    main()
