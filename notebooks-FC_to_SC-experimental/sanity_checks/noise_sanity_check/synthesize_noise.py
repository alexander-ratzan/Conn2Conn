#!/usr/bin/env python3
"""Synthesize the FC noise sanity-check suite (A, B, E, F) into one summary + the
headline numbers that sharpen MASTER_FINDINGS F10.
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _noise_common import results_dir

R = results_dir()


def _read(name):
    p = R / name
    return pd.read_csv(p) if p.exists() else None


a = _read("a_reliability_ceiling.csv")
b = _read("b_variance_decomposition.csv")
e = _read("e_crossmodal_disattenuation.csv")
f = _read("f_discriminability.csv")

print("=" * 74); print("FC NOISE SANITY CHECK — SYNTHESIS (HCP-YA)"); print("=" * 74)

if a is not None:
    print("\n### A. FC reliability ceiling (native metric)")
    print(a.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

if b is not None:
    print("\n### B. Variance decomposition (individual-difference fractions)")
    print(b.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

if f is not None:
    print("\n### F. Whole-connectome reliability (fingerprint / discriminability)")
    print(f.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

if e is not None:
    print("\n### E. Cross-modal disattenuation (fraction of reproducible FC captured)")
    print(e.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n" + "=" * 74); print("HEADLINE"); print("=" * 74)
lines = []
if a is not None:
    bs = a[(a.parc == "Glasser") & (a.comparison == "between_session")]
    if len(bs):
        r = bs.iloc[0]
        lines.append(f"FC between-session reliability ceiling (Glasser): demeaned_r="
                     f"{r['demeaned_pearson']:.3f}, fingerprint top1={r['top1_acc']:.3f}, "
                     f"avg_rank={r['avg_rank']:.3f}")
if b is not None:
    g = b[b.parc == "Glasser"]
    if len(g):
        r = g.iloc[0]
        lines.append(f"FC edge variance (Glasser): trait={r['trait_frac_mean']:.0%}, "
                     f"state={r['state_frac_mean']:.0%}, within-session={r['within_sess_frac_mean']:.0%}, "
                     f"noise={r['noise_frac_mean']:.0%}; averaged-connectome reliability G="
                     f"{r['G_mean']:.2f}")
if e is not None:
    sc = e[(e.source == "SC->FC") & (e.metric == "demeaned_pearson")]
    if len(sc):
        r = sc.iloc[0]
        lines.append(f"SC->FC captures {r['fraction_of_ceiling']:.0%} of the reproducible "
                     f"FC signal (demeaned_r {r['achieved']:.3f} of ceiling {r['ceiling']:.3f})")
for ln in lines:
    print(f"  - {ln}")

print("\nInterpretation: this is the physical counterpart to F10's statistical saturation —")
print("it apportions the FC prediction gap into reproducible-signal vs noise/state, in our")
print("native metric. SC noise remains UNMEASURED (no test-retest dMRI in HCP-YA cache).")

pd.DataFrame({"headline": lines}).to_csv(R / "noise_synthesis.csv", index=False)
print(f"\nSaved -> {R / 'noise_synthesis.csv'}")
