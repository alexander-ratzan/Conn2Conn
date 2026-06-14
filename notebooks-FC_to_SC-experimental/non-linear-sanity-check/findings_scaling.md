# N6 — data-scaling learning curve: model-ceiling vs data-limited

**Run**: 2026-06-14, job 10782789, 10 seeds, multimodal sink, train subsampled at
n = 100/200/400/full(~683), test fixed+full at every n. Question: does the nonlinear
gap (residual-boost `final` − linear `template`) grow with n? If yes → signal is
data-limited (bigger cohort indicated). If flat → structural ceiling (more data won't help).

## TL;DR

**MODEL-CEILING / STRUCTURAL.** The nonlinear gap does not grow into positive territory
with n in either task. Cognition is flat-and-slightly-negative at every n; reconstruction
is negative at every n and merely creeps toward zero from below as data grows (an
overfitting-penalty effect, not signal). Meanwhile the *linear* performance climbs with n
in both tasks — the learning curve works, more data helps the linear model, and
nonlinearity adds nothing at any sample size. **A bigger cohort is NOT indicated for
nonlinearity; the linear-saturation conclusion is definitive.**

## Results

### Cognition (CogTotal), nonlinear gap vs n
| n | median linear | median final | **gap** | Wilcoxon p(gap>0) |
|---|---|---|---|---|
| 100 | 0.302 | 0.290 | **−0.0087** | 1.00 |
| 200 | 0.327 | 0.324 | **−0.0152** | 0.99 |
| 400 | 0.394 | 0.380 | **−0.0119** | 0.999 |
| ~683 | 0.391 | 0.375 | **−0.0128** | 0.99 |

gap-vs-n Spearman **rho=+0.05, p=0.77 → flat**. Gap is ≤0 at every n (nonlinear mildly
*hurts*). Linear grows 0.30→0.39 with n. No nonlinear signal at any size.

### Reconstruction (sink→FC), nonlinear gap vs n
| n | median linear | median final | **gap** | Wilcoxon p(gap>0) |
|---|---|---|---|---|
| 100 | 0.0412 | 0.0353 | **−0.0054** | 0.999 |
| 200 | 0.0670 | 0.0655 | **−0.0019** | 0.88 |
| 400 | 0.0940 | 0.0931 | **−0.0015** | 0.99 |
| ~683 | 0.1098 | 0.1097 | **−0.0009** | 0.98 |

gap-vs-n Spearman rho=+0.52, p=0.001 — **but this is NOT data-limited signal.** The gap
is **negative at every n** and just rising *toward zero from below*; Wilcoxon confirms
gap ≤0 everywhere (p(gap>0) = 0.88–1.0). The positive trend is the KernelRidge residual's
**overfitting penalty shrinking as data grows** — it asymptotes *at* zero, not above it.
It never reaches the +0.02 "matters" threshold, never goes positive. Linear grows
0.04→0.11 with n.

(Note: full-train size is 682 for one seed, 683 for nine — a family-split rounding
artifact that splits the last column into two buckets; the single-seed 682 row is noise,
ignore it. Plot: `n6_scaling_curve.png`.)

## Why the auto-verdict was overridden

The synthesizer's first pass flagged reconstruction as "DATA-LIMITED" on `rho>0 & p<0.05`
alone. That's wrong: a negative gap creeping toward zero has a positive trend but is not
emerging signal. Fixed the rule to require `gap@max_n > 0.005` (actually positive and
meaningful), which correctly returns MODEL-CEILING. Lesson: a positive *slope* on a
*negative* gap is an overfitting penalty vanishing, not signal appearing — always check
the gap sign/level, not just the trend.

## Interpretation

- **Cognition is linearly saturated, structurally.** Nonlinearity doesn't help at n=100
  or n=683; the gap doesn't trend up. More subjects make the *linear* predictor better
  (0.30→0.39) but won't unlock nonlinear structure — there is none to unlock at this
  representation/precision.
- **Reconstruction has no positive nonlinear component either**, at any n. The tiny
  nonlinear sliver seen in N4recon (+0.005 on full data, single-modality) does not appear
  in the multimodal sink and does not grow with n here; the residual penalty just
  converges to ~0.
- **The bottleneck is information, not data or model capacity.** We've now shown the
  null is robust to: model class (N1–N3), the residual-boost head-start (N4), cross-modal
  interactions (N5), AND sample size (N6). Four independent ways of asking "is there
  nonlinear/extra signal," four nulls.

## What this closes, and the one honest caveat

Closes the "maybe a bigger model / harder tuning would find it" question: it wouldn't —
the gap doesn't grow with data, so it's not a capacity-or-sample-size ceiling, it's
structural. The defensible headline: **the connectome→cognition relationship (and the
FC↔SC reconstruction) is linearly saturated; FC is the ceiling; richer tractography adds
nothing — across model classes, architectures, and sample sizes up to n≈683.**

Caveat: we tested to n≈683. We cannot rule out that n≈10⁴ (UK Biobank scale) surfaces a
tiny positive gap — but our data shows convergence *to* zero, not growth *above* it, so
there is no positive evidence for it. If one wanted to chase it, the move is a bigger
cohort (and the expectation should be a very small effect), NOT a bigger model on these
data.

## Files
- `n6_scaling_curve.py`, `synthesize_scaling.py`, `_residual.py` (n-aware block helpers)
- `n6_scaling_results.csv`, `n6_scaling_summary.csv`, `scaling_synthesis.csv`
- `n6_scaling_curve.png` (gap-vs-n, both tasks)
- `n6_scaling_output.txt`
