# Residual-boost (architecture A) + multimodal sink — findings

**Run**: 2026-06-13, jobs 10759407/408 (N4), 10759825/826 (N5), parallel `:ro`, 10 seeds,
Glasser, family-aware. Design rationale in `DESIGN_residual_learning.md`. Builds on
`findings_nonlinear.md` (plain nonlinear, also null).

## TL;DR

The residual-learning idea — hand the nonlinear model the linear prediction for free
(OOF, no leakage) and make its loss the improvement above that template — was the right,
maximally-sensitive test. Result:

- **Downstream cognition: comprehensively NULL.** Not single-modality (N4cog), not
  cross-modal sink (N5cog), not residual-boosted. FC alone remains the ceiling;
  combining modalities *dilutes* it; the nonlinear cross-modal residual adds nothing.
- **Reconstruction: one real but tiny signal.** N4recon shows the residual-boost beats
  the PLS template by +0.002–0.006 demeaned_pearson, p=0.001 every direction — a genuine
  nonlinear sliver in connectome↔connectome mapping, ~4× below the +0.02 "matters"
  threshold, and it washes out under the cross-modal sink (N5recon).

**Interpretation: the connectome→cognition relationship is linearly saturated to the
precision this dataset supports; connectome↔connectome reconstruction has a negligible
nonlinear component. The tractography dead-end survives the most sensitive nonlinear
probe we can construct.**

## N4 — single-modality residual-boost

### N4cog (cognition): NULL
`final = OOF-BR template + KernelRidge(residual)`. Δ(final − template) was ±0.008,
mixed sign — KR residual adds ~0. Every tractography rep stayed −0.13 to −0.25 below
the bv+demo floor (unchanged from linear E5/N1). FC stayed above floor.

### N4recon (reconstruction): TINY REAL SIGNAL
final vs template (PLS), demeaned_pearson, 10-seed Wilcoxon:

| direction | template | final | Δ | p |
|---|---|---|---|---|
| FC→SC | 0.1355 | 0.1405 | +0.0052 | 0.001 |
| SC→FC | 0.0847 | 0.0866 | +0.0023 | 0.001 |
| FC→r2t | 0.1113 | 0.1169 | +0.0058 | 0.001 |
| r2t→FC | 0.0493 | 0.0526 | +0.0041 | 0.001 |
| FC→r2t_corr | 0.0582 | 0.0623 | +0.0050 | 0.001 |
| r2t_corr→FC | 0.0362 | 0.0402 | +0.0039 | 0.001 |

Every direction improves at p=0.001 — a genuine nonlinear residual the OOF framing
surfaces where plain nonlinear (N2) saw only mixed-sign noise. But the effect is ~+0.005,
practically negligible (counts still beat bundles; gap not closed). Identifiability
metrics flat except r2t_corr (top1 +0.013 p=0.02; avg_rank +0.008 p=0.01).

## N5 — multimodal sink residual-boost (cross-modal interactions)

The one architecture that can see cross-modal conjunctions: all modalities in one feature
space, nonlinear model free to cross them.

### N5cog (cognition): NULL + combining hurts

| target | bv+demo floor | FC alone | sink_linear | sink_residual |
|---|---|---|---|---|
| CogTotal | 0.373 | **0.436** | 0.401 | 0.381 |
| CogFluid | 0.298 | **0.306** | 0.301 | 0.286 |
| CogCrystal | 0.349 | **0.452** | 0.414 | 0.415 |

- `sink_linear [FC‖SC‖r2t‖bv‖demo]` is **−0.05 to −0.08 below FC alone**: adding
  SC/tractography/demographics to FC *dilutes* cognition prediction (extra dims, no new
  signal). FC alone is the best single predictor.
- `sink_residual − sink_linear` = −0.002 to −0.020 (Wilcoxon p≈0.96–0.999 in the wrong
  direction): the nonlinear cross-modal residual does not help — slightly hurts.
- **No FC×tractography interaction exists.** The conjunctive-modulator hypothesis (the
  most plausible way tractography could matter) is empty.

### N5recon (reconstruction): NULL
`[SC‖r2t‖bv‖demo]→FC`: sink_residual − sink_linear = −0.0023 (p=0.935). The tiny N4recon
nonlinear sliver does not survive the multimodal sink.

## Decision rules — all returned null (except the negligible N4recon whisper)

| test | rule | result |
|---|---|---|
| N4cog cognition unlock | tractography clears floor by ≥0.02 | NO (all below floor) |
| N4recon reconstruction gain | final − template ≥ 0.02 dp | NO (+0.005, though p=0.001) |
| N5cog cross-modal unlock | sink_residual − sink_linear ≥ 0.02, p<0.05 | NO (−0.01, p>0.95 wrong way) |
| N5cog complementarity | sink_linear − FC ≥ 0.02 | NO (−0.05 to −0.08, sink worse) |
| N5recon cross-modal recon | sink_residual − sink_linear ≥ 0.02 | NO (−0.002) |

## Why this is the definitive version

A clean null from OOF residual-boost means `y − ŷ_linear` is **noise with respect to the
source** — not "the model couldn't find it," but "there is no learnable structure left."
The linear model already extracted everything predictable at this n. Handing the
nonlinear model the answer for free and tasking it only with the residual is the most
sensitive probe possible; it found nothing for cognition, single-modality OR cross-modal.

The one exception (N4recon +0.005, p=0.001) is the honest nuance: connectome↔connectome
reconstruction (same data type, dense edges) has a real but trivial nonlinear component;
connectome→cognition (crossing into behavior) is dead-flat linear-saturated.

## Caveats

- KernelRidge alpha/gamma fixed (median-heuristic gamma, alpha=1.0); decision margins
  wide vs effect sizes, so tuning won't flip the nulls.
- HGB excluded here (underperformed even on FC at n≈683 in N1); KR is the trustworthy
  probe and the one used throughout N4/N5.
- n≈683 train: the "linear ceiling" is partly a sample-size ceiling. A much larger cohort
  could in principle surface faint nonlinearity; nothing here suggests it would be
  practically meaningful.
- Single parcellation (Glasser), 10 family-aware seeds.

## Files

- `n4_recon_results.csv`/`_summary.csv`, `n4_cog_results.csv`/`_summary.csv`
- `n5_cog_results.csv`/`_summary.csv`, `n5_recon_results.csv`/`_summary.csv`
- `*_output.txt` per experiment; `residual_synthesis.csv`
- scripts: `n4_*`, `n5_*`, `_residual.py` (OOF template + additive-residual helpers,
  single + multimodal-block variants), `run_one.sbatch`
- design: `DESIGN_residual_learning.md`
