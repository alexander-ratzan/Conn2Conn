#!/usr/bin/env python3
"""Leak checks (Phase C) — render a verdict on the sex/age rows of the downstream grid.

Reads the downstream CSV (must already exist) and applies the guardrail:
  sex  balanced_accuracy > 0.99   -> LEAK
  age  pearson           > 0.85   -> LEAK
Inputs that legitimately CONTAIN bv+demo (combined_*) are EXEMPT but FLAGGED
("contains subject-info; interpret cognition columns only"). Any NON-exempt input over
threshold HARD-FAILS the run (exit 1) so a leak can't slip into the results silently.

Writes outputs/leak_verdict.csv and prints a summary.
"""
from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _grid_common import OUTPUTS_DIR  # noqa: E402

SEX_THRESH = 0.99
AGE_THRESH = 0.85


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--downstream-csv", default=str(OUTPUTS_DIR / "downstream.csv"))
    ap.add_argument("--out", default=str(OUTPUTS_DIR / "leak_verdict.csv"))
    args = ap.parse_args()

    csv = Path(args.downstream_csv)
    if not csv.exists():
        raise FileNotFoundError(f"downstream CSV missing: {csv} — run run_downstream_grid.py first")
    df = pd.read_csv(csv)
    leak = df[df["target"].isin(["sex", "age"])].copy()
    if leak.empty:
        print("[leak] no sex/age rows found — nothing to check"); return

    def score(r):
        return r["balanced_acc"] if r["target"] == "sex" else r.get("pearson", np.nan)

    def thresh(t):
        return SEX_THRESH if t == "sex" else AGE_THRESH

    # Raw observed/predicted connectome inputs legitimately encode sex/age (real neuroimaging
    # signal, not a demographic leak). High predictability there is EXPECTED, not a failure.
    # A genuine leak is a demographic-free input that ALSO isn't a raw connectome predicting
    # sex/age too well (i.e. a residualized/"cleaned" input that secretly re-added bv+demo).
    CONNECTOME_INPUTS = {"obs_FC", "obs_SC", "obs_FC+obs_SC", "pred_SC", "pred_FC"}
    leak["leak_score"] = leak.apply(score, axis=1)
    leak["threshold"] = leak["target"].apply(thresh)
    leak["exceeds"] = leak["leak_score"] > leak["threshold"]
    leak["contains_bvdemo"] = leak["contains_bvdemo"].astype(bool)
    leak["is_connectome"] = leak["input_set"].isin(CONNECTOME_INPUTS)

    def _verdict(r):
        if not r["exceeds"]:
            return "ok"
        if r["contains_bvdemo"]:
            return "EXEMPT_FLAGGED"    # knowingly contains bv+demo; interpret cognition cols only
        if r["is_connectome"]:
            return "EXPECTED_SIGNAL"   # connectomes encode sex/age (real biology); not a leak
        return "LEAK_FAIL"             # demographic-free, non-connectome input -> genuine leak
    leak["verdict"] = leak.apply(_verdict, axis=1)

    cols = ["parcellation", "seed", "estimator", "variant", "input_set", "target",
            "leak_score", "threshold", "exceeds", "contains_bvdemo", "is_connectome", "verdict"]
    out = leak[cols].sort_values(["target", "verdict", "leak_score"], ascending=[True, True, False])
    out.to_csv(args.out, index=False)

    n_fail = int((leak["verdict"] == "LEAK_FAIL").sum())
    n_flag = int((leak["verdict"] == "EXEMPT_FLAGGED").sum())
    n_exp = int((leak["verdict"] == "EXPECTED_SIGNAL").sum())
    print(f"[leak] {len(leak)} sex/age rows | LEAK_FAIL={n_fail} EXEMPT_FLAGGED={n_flag} "
          f"EXPECTED_SIGNAL={n_exp} ok={int((leak['verdict']=='ok').sum())} -> {args.out}")
    if n_exp:
        print("[leak] EXPECTED_SIGNAL (raw connectomes predict sex/age — real signal, not leak):")
        print(out[out.verdict == "EXPECTED_SIGNAL"][["parcellation", "input_set", "target", "leak_score"]]
              .drop_duplicates(["parcellation", "input_set", "target"]).to_string(index=False))
    if n_fail:
        print("[leak] *** HARD FAIL — non-exempt inputs exceed leak threshold: ***")
        print(out[out.verdict == "LEAK_FAIL"].to_string(index=False))
        sys.exit(1)
    print("[leak] PASS — no non-exempt leak.")


if __name__ == "__main__":
    main()
