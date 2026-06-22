# Resolving the Two Flagged Discrepancies (oracle 0.647 vs 0.38; cognition 0.359 vs 0.129)

An analyst comparing the grid CSVs against the older PDF flagged two worrying gaps. **Both have the
same root cause and neither indicates a problem with the grid:** the analyst read the `pca_pls` rows;
the PDF/report numbers are `bayesian_ridge`. Hold the estimator fixed and everything matches to the
third decimal. Verified empirically below (`python exploration/explore.py` reproduces it).

---

## Discrepancy 1 — the oracle ceiling (the one that worried you most)

**Claim:** "SC→SC oracle is 0.38 here vs 0.647 in the PDF — maybe 0.647 was raw pearson or a
different oracle construction; the disattenuation denominator may be wrong."

**Resolution: it's the same metric (demeaned-r), same pipeline — just a different estimator.**

| SC→SC oracle (demeaned_pearson) | pca_pls | bayesian_ridge | kernel_ridge |
|---|---|---|---|
| **Glasser** | 0.376 | **0.648** | 0.645 |
| **4S456** | 0.386 | 0.622 | 0.618 |

The PDF's **0.647 ≈ bayesian_ridge SC→SC = 0.648 (Glasser)** — a 0.001 match. The 0.38 the analyst
saw is the `pca_pls` row. It is **not** raw pearson: raw pearson for SC→SC is **0.92–0.95** (and r² is
~0.05–0.10). So three candidate values per cell:
- demeaned-r (the PRIMARY metric): pca_pls 0.38 / **BR 0.648** / KR 0.645
- raw pearson (population-mean-dominated, not reportable): ~0.95
- r²: ~0.05–0.10

**Conclusion: the ceiling did not change. The disattenuation denominator is fine — 0.647 was the BR
oracle all along.** The only fix needed is reporting hygiene: the PDF quotes F1 cross-modal with
`pca_pls` but the oracle with `bayesian_ridge`. **Quote one estimator throughout.** Since BR is the
strongest estimator (its oracle is ~50% higher than PLS's, because PLS only reaches ~65% of BR's
ceiling), the cleanest fix is to lead *everything* in BR:

| BR, demeaned-r | Glasser | 4S456 |
|---|---|---|
| FC→SC (cross-modal) | 0.166 | 0.171 |
| SC→FC (cross-modal) | 0.104 | 0.101 |
| FC→FC oracle | 0.673 | 0.632 |
| SC→SC oracle | 0.648 | 0.622 |

The asymmetry **ratio** is estimator-robust (~1.6× in both PLS and BR), so F1 is unaffected; only the
absolute headline numbers move with the estimator.

---

## Discrepancy 2 — cognition baseline dropped (bv+demo 0.359 → 0.129)

**Claim:** "bv+demo CogTotal is 0.129 here vs 0.359 in the PDF; obs_FC CogCryst 0.439 vs 0.434. The
baseline dropped a lot, which widens FC's lift, but perm-p still isn't significant, so variance must
be higher — why did bv+demo cognition drop?"

**Resolution: same estimator confusion. The analyst's numbers are `pca_pls`, exactly:**

| Glasser, pearson | pca_pls | bayesian_ridge (= PDF) |
|---|---|---|
| bv+demo → CogTotal | **0.129** ← analyst | **0.359** ← PDF |
| obs_FC → CogCryst | **0.439** ← analyst | 0.487 |

0.129 and 0.439 are the `pca_pls` Glasser values to the digit. The baseline didn't "drop" — the
analyst is on a different (and worse) estimator.

### Bonus finding: this *validates* the report's use of `bayesian_ridge` for downstream

Digging in revealed why `pca_pls` should never be quoted for the scalar (cognition/age) targets:

1. **Catastrophic r².** pca_pls scalar regression has r² = **−50 (4S456) to −540 (Glasser)** on the
   bv+demo baseline; kernel_ridge is similar (−1 to −7). Only **bayesian_ridge has positive r²
   (+0.115)**. The PLS/KR scalar predictors are wildly overfit/ill-conditioned for low-signal scalar
   targets.

2. **Not numerically reproducible.** The bv+demo input is subject-info — *parcellation-independent* —
   so the baseline should be **bit-identical across parcellations** for the same seed. It is, for
   bayesian_ridge (max |Δ| across seeds = **0.0000**). But for pca_pls, 2 of 10 seeds diverge wildly
   on identical input (seed 3: −0.045 vs +0.355; seed 6: −0.046 vs +0.305). The ill-conditioning is
   so severe that run-to-run numerical noise (different cluster nodes → different BLAS reductions)
   flips the result. The apparent "parcellation difference" is numerical noise, not signal.

**Takeaway:** the report correctly leads downstream with `bayesian_ridge` — it is the *only* stable,
positive-r², parcellation-consistent scalar estimator. pca_pls/kernel_ridge downstream rows exist in
the grid for completeness but **must not be quoted**. (For reconstruction, all three are stable; the
estimator choice there only shifts magnitude, not validity.)

---

## One-line answers

- **Oracle:** nothing changed; 0.647 = the BR oracle (demeaned-r 0.648), the analyst read the PLS row.
  Disattenuation denominator is correct. Just quote one estimator throughout (recommend BR).
- **Cognition baseline:** nothing dropped; 0.129 = the PLS row, the PDF's 0.359 = BR. And PLS/KR
  scalar regression is ill-conditioned (r² ≪ 0, non-reproducible) → only BR is reportable downstream.
- **Action:** add an explicit "estimator = bayesian_ridge" label to every headline cognition/oracle
  number, and a note that PLS/KR downstream rows are diagnostic-only. (This is exactly flag ⑤ in
  `FINDINGS_EXPLORATION.md`.)
