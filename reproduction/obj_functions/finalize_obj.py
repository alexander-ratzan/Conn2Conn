#!/usr/bin/env python3
"""Finalize Phase 1: aggregate the 3-axis scorecard across seeds (recon / identity / cognition)
and write outputs/scorecard.csv. Identity = pooled sibling AUC per estimator.

    python finalize_obj.py
"""
from pathlib import Path
import sys
import csv
import statistics as st
from collections import defaultdict
import numpy as np

HERE = Path(__file__).resolve().parent
FAM = HERE.parent / "family_mechanism"
sys.path.insert(0, str(FAM))
import _fm_common as fm  # noqa: E402

PARTS = HERE / "outputs" / "parts"
ESTIMATORS = ["BR", "PLS", "obj1a", "obj2c", "obj2c_raw"]


def _load(glob):
    rows = []
    for f in sorted(PARTS.glob(glob)):
        rows += list(csv.DictReader(open(f)))
    return rows


def _mean(rows, pred, key):
    v = [float(r[key]) for r in rows if pred(r) and r.get(key) not in (None, "", "nan")]
    return st.mean(v) if v else float("nan")


def main():
    recon = _load("recon_s*.csv")
    cog = _load("cog_s*.csv")

    # identity: pool family sims across seeds, sibling AUC per estimator
    pooled = {e: {r: [] for r in fm.RELATIONS} for e in ESTIMATORS}
    for f in sorted((PARTS / "family").glob("obj_family_s*.npz")):
        d = np.load(f)
        for e in ESTIMATORS:
            for r in fm.RELATIONS:
                k = f"{e}__{r}"
                if k in d.files and d[k].size:
                    pooled[e][r].append(d[k])
    sib_auc = {}
    for e in ESTIMATORS:
        pe = {r: (np.concatenate(v) if v else np.array([])) for r, v in pooled[e].items()}
        sib_auc[e] = fm.auc_vs_unrelated(pe).get("sibling", float("nan"))

    # scorecard
    out = HERE / "outputs" / "scorecard.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["estimator", "recon_demeaned_r", "avg_rank", "top1_acc",
                    "identity_sib_auc", "cog_lift_predSC_Cryst", "cog_lift_predSCbvd_Cryst",
                    "cog_lift_predSC_Total", "cog_lift_predSC_Fluid"])
        for e in ESTIMATORS:
            rr = lambda key: _mean(recon, lambda r: r["estimator"] == e, key)
            cl = lambda inp, t: _mean(cog, lambda r: r["estimator"] == e and r["input_set"] == inp
                                      and r["target"] == t, "lift_over_bvdemo")
            w.writerow([e, f"{rr('demeaned_pearson'):.4f}", f"{rr('avg_rank'):.4f}",
                        f"{rr('top1_acc'):.4f}", f"{sib_auc[e]:.4f}",
                        f"{cl('pred_SC','CogCryst'):.4f}", f"{cl('pred_SC+bv+demo','CogCryst'):.4f}",
                        f"{cl('pred_SC','CogTotal'):.4f}", f"{cl('pred_SC','CogFluid'):.4f}"])

    # print
    print("\n=== 3-AXIS SCORECARD (Glasser, mean over seeds) ===")
    print(f"{'estimator':11s}{'recon dr':>10s}{'avg_rank':>10s}{'sib_AUC':>9s}"
          f"{'cogCryst':>10s}{'cog+bvd':>9s}{'cogTotal':>10s}{'cogFluid':>10s}")
    for e in ESTIMATORS:
        rr = lambda key: _mean(recon, lambda r: r["estimator"] == e, key)
        cl = lambda inp, t: _mean(cog, lambda r: r["estimator"] == e and r["input_set"] == inp
                                  and r["target"] == t, "lift_over_bvdemo")
        print(f"{e:11s}{rr('demeaned_pearson'):>+10.3f}{rr('avg_rank'):>10.3f}{sib_auc[e]:>9.3f}"
              f"{cl('pred_SC','CogCryst'):>+10.3f}{cl('pred_SC+bv+demo','CogCryst'):>+9.3f}"
              f"{cl('pred_SC','CogTotal'):>+10.3f}{cl('pred_SC','CogFluid'):>+10.3f}")
    print(f"\nwrote {out}")
    print("Diagonal check: obj1a should win sib_AUC; obj2c should win cogCryst; BR wins recon dr.")


if __name__ == "__main__":
    main()
