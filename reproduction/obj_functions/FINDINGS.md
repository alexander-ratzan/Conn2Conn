# Objective Functions — Phase 1 Findings (Glasser × 5)

**Result: NEGATIVE (clean).** The cheap closed-form objectives did **not** beat the BR/PLS baselines on
the axis each was built for. The F7 reconstruct↔identify tradeoff did not move. Source:
`outputs/scorecard.csv`. Plan: [`../../dev-notes/objective-functions/PLAN.md`](../../dev-notes/objective-functions/PLAN.md).

## 3-axis scorecard (mean over 5 seeds)

| estimator | recon dr ↑ | sib_AUC ↑ | CogCryst lift ↑ | targets |
|---|---|---|---|---|
| **BR** | **0.163** | 0.771 | 0.037 | reconstruction |
| **PLS** | 0.134 | **0.812** | −0.003 | covariance |
| **obj1a** | 0.162 | 0.752 | 0.018 | *identity* |
| **obj2c** | 0.044 | 0.688 | 0.023 | *cognition (resid)* |
| **obj2c_raw** | 0.133 | 0.685 | **0.047** | *cognition (raw)* |

Validation: BR/PLS reproduce prior runs byte-for-byte per seed (recon dr, CogCryst lift); 125/131…
sibling pairs match the notebook. Differences are purely the objective swap.

## Verdict per objective
- **obj1a (identity) — FAILED.** sib_AUC 0.752 < BR 0.771 ≪ PLS 0.812 — 2nd-from-last on its own axis,
  on both per-seed previews and the pool. **Diagnosis:** the reliability gate (`g_k ∝ r_k`) refuses to
  restore the *low-reliability tail*, which is exactly where the family fingerprint lives → it cannot
  out-identify PLS (which keeps tail amplitude ungated). Re-confirms the F7 mechanism.
- **obj2c (cognition, residualized) — FAILED.** +0.023 CogCryst < BR +0.037, and reconstruction
  collapsed (0.044). Residualizing the target removed the signal; the emphasis reweighting was too
  aggressive.
- **obj2c_raw (cognition, raw) — flicker, NOISE-LEVEL.** +0.047 vs BR +0.037 = +0.010 over 5 seeds with
  ±0.087 per-seed scatter → within noise, and bought at a reconstruction (0.133 vs 0.163) and identity
  (0.685 vs 0.771) cost. Not a real win. (Seed-dependent: helps low-signal seeds where BR fails, e.g.
  seed 2 obj2c_raw +0.070 vs BR −0.012; adds nothing on high-signal seeds, e.g. seed 0 BR +0.215.)

## Takeaway
The reconstruct↔identify frontier is **structural, not a cheap knob.** BR's shrinkage is near-optimal
for reconstruction, PLS's covariance-retention near-optimal for identity, and **post-hoc reshaping of a
BR output (amplitude restore / cognition reweight) does not beat either.** "You can't cheaply optimize
the connectome for a third objective" fits the paper's redirect thesis.

## Status: PARKED (closed as a negative)
We built and correctly ran several well-motivated objective functions — an **identity-maximizing** one
(restore between-participant difference) and **cognition-maximizing** ones (weight/supervise toward the
biomarker) — and **none beat the BR/PLS baselines on the axis it was designed to win.** The
reconstruct↔identify frontier did not move. That's the result: *you can't cheaply re-aim a BR-imputed
connectome at a third objective* — consistent with the structural F7 mechanism (the probe showed BR's
shrinkage is near-optimal and the discriminative tail is directionally noise, corr≈0.04).

This line is closed. See [`../../dev-notes/objective-functions/TODO.md`](../../dev-notes/objective-functions/TODO.md)
for the parked **future TODO** (gradient versions 1B/2B as the untried "fair test", low expectations).

- Caveat: Glasser × 5 only.
- A confirmatory **ungated-obj1a diagnostic** (job 11881048) is finishing; its one-line result will be
  appended below for the record, but it does not change the conclusion.
