# BR-only Run — Findings (Glasser × 10 seeds)

**Run:** BayesianRidge in both estimator slots (BR-impute the connectomes, BR-predict cognition),
18-input set, Glasser, 10 frozen seeds. Source: `outputs/downstream_br.csv` (900 rows, complete,
0 LEAK_FAIL). Compared against the spine grid's **PLS-imputed** `pred_*` (same BR downstream).
Lead metric: `lift_over_bvdemo` (mean ± std over 10 seeds).

## Validation (wiring is correct)
The 6 carried-over **non-imputation** inputs (`obs_FC`, `obs_SC`, `obs_FC+obs_SC`,
`obs_FC+bv+demo`, `obs_SC+bv+demo`, `bv+demo`) reproduce the spine grid **byte-for-byte**
(Δlift = 0.0000, seed 0 checked). Any difference vs spine is therefore attributable to the
imputation-estimator swap (PLS→BR), nothing else.

## Headline 1 — BR imputation uniformly (but modestly) beats PLS downstream
Across **all 4 imputed inputs × all 3 cognition targets = 12/12 comparisons, the BR-impute lift is
higher than PLS-impute** (Δ = +0.006 … +0.055). BR is the better imputer for downstream transfer.
Representative (CogCryst, mean ± std over 10 seeds):

| input | BR-impute | PLS-spine | Δ |
|---|---|---|---|
| `pred_SC` | +0.045 ± 0.087 | −0.010 ± 0.068 | **+0.055** |
| `pred_FC` | −0.101 ± 0.068 | −0.135 ± 0.068 | +0.035 |
| `pred_SC+bv+demo` | +0.090 ± 0.038 | +0.061 ± 0.034 | +0.029 |
| `pred_FC+bv+demo` | +0.005 ± 0.030 | −0.040 ± 0.042 | +0.044 |

## Headline 2 — but F5 holds: the strongest imputer does not rescue transfer
The improvement is **small and mostly not significant** for the bare imputed connectomes:
- `pred_FC` is **still actively harmful** (−0.101 CogCryst, **0/10 seeds** significant). BR de-harms
  it vs PLS but does not make it useful.
- `pred_SC` is **neutral-to-marginal** (+0.045, only **3/10 seeds** significant). The seed-0 pilot
  (+0.215) was a high-signal seed, not the typical case.
- The one **robustly positive** imputed input is `pred_SC+bv+demo` (+0.090, **8/10 sig**) — BR-imputed
  SC + subject-info beats baseline consistently, but modestly.

So the F5 conclusion ("imputed connectomes don't transfer; `pred_FC` harmful, `pred_SC` ~neutral")
**survives the best-foot-forward test.** BR shifts every number in the favorable direction but does
not change the qualitative story.

## Headline 3 — the new combos reconfirm FC-dominance
The expanded inputs show imputed connectomes add **nothing on top of the modality they came from**:
- `obs_FC+pred_SC` (+0.136 ± 0.089) ≈ `obs_FC` (+0.133 ± 0.088) — `pred_SC` is a function of FC, so
  it adds no information beyond FC.
- `obs_FC+pred_SC+bv+demo` (+0.161, 8/10) ≈ `obs_FC+bv+demo` (+0.162, 8/10); `everything` (+0.151) ≈
  same. FC + subject-info is the ceiling; piling on imputed connectomes doesn't move it.
- `obs_SC+pred_FC` (−0.094) ≈ `obs_SC` (−0.101); `pred_FC` adds nothing on top of SC either.
- Every input **without** observed FC sits at ≤ 0 lift. FC remains the cognition ceiling.

## Leak guardrail
360 sex/age rows: **235 ok / 125 EXEMPT_FLAGGED / 0 EXPECTED_SIGNAL / 0 LEAK_FAIL.** All threshold
crossings are demo-containing inputs (demo holds sex/age outright → trivially predicts them, exempt).
No connectome-only input crosses threshold; no genuine leak.

## Caveats
- Glasser only (4S456 deferred). In-sample train imputation kept on purpose (PLAN §6) — not a leak;
  the BR-vs-PLS comparison is apples-to-apples since both use it.
- `pred_SC` lift has high across-seed variance (±0.087) — interpret the mean, not any single seed.

## One-line takeaway
A Bayesian-ridge "SOTA" imputation **uniformly improves** downstream transfer over PLS, but the gain
is small and **F5 stands**: even the best imputer leaves `pred_FC` harmful and `pred_SC` marginal, and
imputed connectomes add nothing on top of the modality they were derived from. FC is still the ceiling.
