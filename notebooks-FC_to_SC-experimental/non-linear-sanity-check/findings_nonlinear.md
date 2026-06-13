# Non-linear sanity check — findings

**Run**: 2026-06-13, 3 experiments in parallel (`:ro` overlay, 4 cpu/24G each, jobs
10757352–54), 10 seeds, Glasser, family-aware splits. Estimator swapped, representation +
splits held fixed vs `tractography_predict/`. Linear baselines byte-identical to the
linear runs (shared `_tract_setup.py`).

## TL;DR

**NULL across the board — the tractography dead-end and the FC↔SC asymmetry are
model-class robust.** Neither HistGradientBoosting nor KernelRidge (RBF) extracts any
tractography signal that linear models missed — not for cognition, not for FC
reconstruction, not for marginal contribution over SC. This is the strongest version of
the linear finding: tractography's failure to predict FC/cognition is not a linear-model
artifact.

Two estimator caveats matter for reading this (below): KernelRidge is the trustworthy
nonlinear probe (it preserves FC's known cognition signal); HGB underperforms at this
sample size even where signal exists.

## N1 — nonlinear cognition

Best lift over the bv+demo floor among tractography reps (SC, r2t, r2t_corr, SC_r2t),
per estimator (median 10 seeds; lift = Pearson − bv+demo floor with the same estimator):

| estimator | best tractography lift | rep / target | clears floor? |
|---|---|---|---|
| linear_BR | −0.079 | SC / Crystal | no |
| HGB | −0.168 | r2t_corr / Fluid | no |
| **KR** | **−0.029** | SC / Crystal | **no** |

**No tractography representation clears the demographic floor under any estimator.** The
closest is KernelRidge on SC/crystallized at −0.029 — still below floor.

### Estimator sanity (FC, CogTotal) — does nonlinear preserve known signal?
FC carries real cognition signal (linear lift +0.078). A trustworthy nonlinear estimator
should keep it:

| estimator | FC pearson | floor | FC lift |
|---|---|---|---|
| linear_BR | 0.451 | 0.372 | **+0.078** ✓ |
| **KR** | 0.338 | 0.282 | **+0.056** ✓ |
| HGB | 0.254 | 0.355 | **−0.101** ✗ |

- **KernelRidge preserves FC's lift** (+0.056) → it's a valid nonlinear probe, and its
  finding of zero tractography signal is meaningful.
- **HGB destroys even FC's signal** (−0.101) → gradient boosting underperforms at
  n≈683 with 256 PCA inputs (overfits / can't model the smooth FC→cognition map). HGB's
  "below floor for tractography" is therefore partly an estimator-weakness artifact, not
  evidence about tractography. We trust the **KR** result.

## N2 — nonlinear reconstruction + asymmetry

KernelRidge vs linear PLS, FC→X demeaned_pearson (median 10 seeds):

| direction | PLS | KR | Δ (KR−PLS) | nonlinear gain? |
|---|---|---|---|---|
| FC→SC | 0.1355 | 0.1334 | −0.0021 | no |
| FC→r2t | 0.1113 | 0.1133 | +0.0020 | no (< 0.02) |
| FC→r2t_corr | 0.0582 | 0.0605 | +0.0023 | no (< 0.02) |

All Δ are within ±0.0023 — nonlinear neither helps nor hurts reconstruction. Counts still
beat bundles under KR (0.133 vs 0.113), same as linear.

### Asymmetry ratio (demeaned_pearson, FC-wins) holds under nonlinear

| rep | linear_PLS | KR |
|---|---|---|
| SC | 1.621× | **1.638×** |
| r2t | 2.264× | **2.267×** |
| r2t_corr | 1.634× | 1.487× |

The FC↔SC asymmetry is essentially **identical** under linear and kernel models (SC
1.62→1.64, r2t 2.26→2.27). The asymmetry is not a linear-pipeline artifact. (Full
6-metric panel per estimator in `n2_reconstruction_summary.csv`.)

## N3 — nonlinear marginal (r2t over SC)

Paired Δ(SC_r2t − SC) → FC, demeaned_pearson (median 10 seeds):

| estimator | Δ | Wilcoxon p (greater) |
|---|---|---|
| linear | −0.0013 | 0.981 |
| KR | −0.0049 | 0.998 |

r2t adds **nothing** over count-SC even under a nonlinear combined model — slightly
negative both ways. Count-SC remains a sufficient statistic; there is no conjunctive
SC×r2t interaction that a kernel picks up.

## Verdict

All three decision rules return null:
- N1 cognition unlock (≥+0.03 over floor): **not met** (KR best −0.029).
- N2 reconstruction gain (≥+0.02 dp): **not met** (max +0.0023).
- N3 marginal gain (≥+0.02 dp, p<0.05): **not met** (KR −0.0049).

**Conclusion**: the tractography findings from `tractography_predict/` are model-class
robust. Switching from linear (PLS/BR) to nonlinear (KernelRidge RBF, the trustworthy
probe here) changes nothing — bundles still predict FC worse than counts, still carry no
cognition signal above demographics, still add nothing over SC; and the FC↔SC asymmetry
is unchanged. The tractography information is not hiding in nonlinear structure.

## Caveats

- **HGB unreliable at this n**: HistGradientBoosting underperformed even on FC (where
  signal demonstrably exists), so its nulls are not informative on their own. The
  load-bearing nonlinear evidence is **KernelRidge**, which preserved FC's signal and
  still found nothing in tractography.
- **KR hyperparameters fixed** (median-heuristic gamma, alpha=1.0), not CV-tuned. Margins
  to the decision thresholds are wide (best cognition lift −0.029 vs +0.03 rule), so light
  tuning won't flip the null. A gamma/alpha sweep is the natural follow-up only if a
  result had landed near threshold (none did).
- HGB-per-component reconstruction not run (256× the fits); KR multi-output was the
  reconstruction probe. Given KR showed no recon gain, HGB-recon is unlikely to differ.
- Single parcellation (Glasser), 10 family-aware seeds.

## Files

- `n1_cognition_results.csv` / `_summary.csv` (rep × estimator × target × seed:
  pearson, spearman, r2, lift)
- `n2_reconstruction_results.csv` / `_summary.csv` (full 6-metric panel per estimator,
  both directions, + asymmetry ratios)
- `n3_marginal_results.csv` / `_summary.csv` (full panel, paired Δ per estimator)
- `nonlinear_synthesis.csv`, `nonlinear_synthesis_output.txt`
- scripts: `n1`–`n3`, `synthesize_nonlinear.py`, `_nl_common.py`, `run_one.sbatch`
- nonlinear predictors live in `../tractography_predict/_tract_setup.py`
  (`kernelridge_predict`, `kernelridge_blocks_predict`, `hgb_scalar_predict`,
  `kr_scalar_predict`)
