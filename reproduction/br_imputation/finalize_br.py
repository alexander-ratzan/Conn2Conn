#!/usr/bin/env python3
"""Finalize the BR-only run: merge per-seed parts -> downstream_br.csv, render the leak verdict,
generate the completeness manifest, and hard-fail on any missing/non-finite spine cell.

Isolated: reads/writes only under reproduction/br_imputation/outputs|configs.
"""
from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
CONFIGS = HERE / "configs"
PARTS = OUTPUTS / "parts"
sys.path.insert(0, str(HERE))
from run_br_unit import (  # noqa: E402
    BR_INPUTS, COG_TARGETS, LEAK_TARGETS, CONTAINS_SUBJECT_INFO, CONNECTOME_ONLY, ESTIMATOR,
)

SEX_THRESH, AGE_THRESH = 0.99, 0.85


def merge_parts():
    parts = sorted(PARTS.glob("down_br_*.csv"))
    if not parts:
        raise FileNotFoundError(f"no per-seed parts in {PARTS} — run run_br_unit.py first")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    out = OUTPUTS / "downstream_br.csv"
    df.to_csv(out, index=False)
    print(f"[finalize] merged {len(parts)} parts -> {out} ({len(df)} rows)")
    return df


def leak_verdict(df):
    leak = df[df["target"].isin(LEAK_TARGETS)].copy()
    if leak.empty:
        print("[finalize] no sex/age rows"); return
    leak["leak_score"] = leak.apply(
        lambda r: r["balanced_acc"] if r["target"] == "sex" else r.get("pearson", np.nan), axis=1)
    leak["threshold"] = leak["target"].apply(lambda t: SEX_THRESH if t == "sex" else AGE_THRESH)
    leak["exceeds"] = leak["leak_score"] > leak["threshold"]

    def verdict(r):
        if not r["exceeds"]:
            return "ok"
        if r["input_set"] in CONTAINS_SUBJECT_INFO:
            return "EXEMPT_FLAGGED"
        if r["input_set"] in CONNECTOME_ONLY:
            return "EXPECTED_SIGNAL"
        return "LEAK_FAIL"
    leak["verdict"] = leak.apply(verdict, axis=1)
    cols = ["parcellation", "seed", "input_set", "target", "leak_score", "threshold",
            "exceeds", "verdict"]
    out = leak[cols].sort_values(["target", "verdict", "leak_score"], ascending=[True, True, False])
    out.to_csv(OUTPUTS / "leak_verdict_br.csv", index=False)
    nfail = int((leak["verdict"] == "LEAK_FAIL").sum())
    from collections import Counter
    print(f"[finalize] leak verdicts: {dict(Counter(leak['verdict']))} -> outputs/leak_verdict_br.csv")
    if nfail:
        print("[finalize] *** LEAK_FAIL ***")
        print(out[out.verdict == "LEAK_FAIL"].to_string(index=False))
        sys.exit(1)


def expected_and_completeness(df, seeds, parc="Glasser"):
    rows = [{"parcellation": parc, "seed": s, "estimator": ESTIMATOR, "input_set": inp, "target": tgt}
            for s in seeds for inp in BR_INPUTS for tgt in COG_TARGETS + LEAK_TARGETS]
    exp = pd.DataFrame(rows)
    CONFIGS.mkdir(parents=True, exist_ok=True)
    exp.to_csv(CONFIGS / "expected_cells_br.csv", index=False)
    key = ["parcellation", "seed", "input_set", "target"]
    got = set(map(tuple, df[key].itertuples(index=False, name=None)))
    want = set(map(tuple, exp[key].itertuples(index=False, name=None)))
    missing = want - got
    print(f"[finalize] completeness: {len(got & want)}/{len(want)} expected cells present")
    if missing:
        print(f"[finalize] *** MISSING {len(missing)} cells, e.g. {sorted(missing)[:5]} ***")
        sys.exit(1)
    # non-finite check on the lead metric
    bad = df[~np.isfinite(pd.to_numeric(df["lift_over_bvdemo"], errors="coerce"))]
    if len(bad):
        print(f"[finalize] *** {len(bad)} non-finite lift_over_bvdemo rows ***")
        sys.exit(1)
    print("[finalize] PASS — complete + finite.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    args = ap.parse_args()
    df = merge_parts()
    leak_verdict(df)
    expected_and_completeness(df, args.seeds)


if __name__ == "__main__":
    main()
