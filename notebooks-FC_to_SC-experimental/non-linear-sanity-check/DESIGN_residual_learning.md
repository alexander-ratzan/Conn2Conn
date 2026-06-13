# Design conversation — residual (boosted) nonlinear learning

Record of the back-and-forth that produced the N4 experiment, plus the todo.

## The idea (user)

The nonlinear models (HGB, KernelRidge) "spend all their time doing the group
average" — i.e. they burn capacity reproducing what's already easy/linear. So: feed
the nonlinear model the linear model's answer as a template, and make its job (its
loss) the *improvement above that template*.

## Refinement (back-and-forth)

**Two versions of "subtract the easy part":**
- Subtracting the **group average** is mostly already handled — HGB inits at the mean and
  boosts; KernelRidge predicts PCA target latents which are already mean-centered. So the
  literal group mean isn't where capacity leaks.
- The real lever is subtracting the **linear model's prediction**. Hand the nonlinear
  model the full linear prediction as a head start; its target becomes the residual
  `y − ŷ_linear`. Then its whole capacity *and its whole regularization budget* go to
  nonlinear structure orthogonal to the linear fit. This is genuinely a **more sensitive
  probe**: KernelRidge with one global alpha shrinks everything toward zero, so when 90%
  of signal is linear the faint nonlinear residual gets shrunk alongside the linear trend
  it's competing with. Remove the linear part first and the nonlinear residual (if any)
  is the dominant signal in what's left.

**Architecture fork (the crux):** what does the NL model see as input?
- **(B) Pure cascade `NL(ŷ_PLS) → output`** — NL sees ONLY the PLS output. Can learn a
  nonlinear *warping* of the linear prediction (calibration, rescaling) but is **blind to
  the source**: any nonlinear signal PLS zeroed out is unrecoverable. The one version that
  structurally cannot find new signal.
- **(A) Additive residual `final = ŷ_PLS + NL(source)`**, NL trained with target
  `y − ŷ_PLS`. This is exactly the user's stated loss ("improvement above the template").
  NL keeps full source access, so it CAN recover nonlinear source structure. If there's no
  signal, NL→0 and final collapses to PLS.
- (C) Augmented `NL([source ‖ ŷ_PLS])` — same power as A with the template handed in as a
  feature; not worth extra complexity over A.

**Decision: do (A).** It matches the user's loss-function description word-for-word and
strictly dominates the pure cascade. (User initially sketched "PLS → NL → output"; we
converged on A = that idea done right, keeping source access.)

**Non-negotiable implementation detail:** train-set residuals must come from
**out-of-fold** linear predictions (5-fold). In-sample PLS predictions have overfit the
train rows → artificially tiny residual → NL trains on noise → fake null. OOF makes the
residual honest.

## Algorithm (A, additive residual)

```
template      = OOF linear prediction on train      (5-fold; honest residual)
resid_train   = y_train − template
NL (KernelRidge) : source → resid_train             ← loss IS the improvement
ŷ_lin_test    = linear(full train) applied to test
final         = ŷ_lin_test + NL(source_test)
compare  final  vs  linear-alone  (vs bv+demo floor for cognition)
```
- Reconstruction: linear base = PLS (in PCA target-latent space); evaluate FULL 6-metric
  panel (demeaned_pearson, pearson, top1_acc, avg_rank, mse, r2) for `final` and
  `template`, both directions, all reps.
- Cognition: linear base = BayesianRidge; metrics = Pearson, Spearman (rank), R²; lift
  over bv+demo floor. NL probe = KernelRidge (trustworthy; HGB underperformed at n≈683).

## Decision rules (fixed before running)

- **Reconstruction unlock**: `final` beats `template` by ≥ 0.02 demeaned_pearson (or
  meaningfully on rank/top1), 10-seed Wilcoxon p<0.05.
- **Cognition unlock**: residual-boosted prediction beats linear-alone by ≥ 0.02 Pearson
  AND a tractography rep clears the bv+demo floor, p<0.05.
- **Null (prior)**: residual is mostly noise; final ≈ template. Then "no nonlinear signal
  even when handed the linear answer for free" — the definitive version of the dead-end.

## TODO

1. [build] `_residual.py` — OOF template + additive-residual reconstruct + cognition helpers.
2. [build] `n4_residual_reconstruction.py` — A on FC↔{SC,r2t,r2t_corr}, full 6-metric
   panel, `final` vs `template` (=PLS) per direction/seed.
3. [build] `n4_residual_cognition.py` — A on cognition (CogTotal/Fluid/Crystal) for
   {SC,r2t,r2t_corr,SC_r2t,FC,bv+demo}; pearson/spearman/r2; lift over floor; linear vs
   residual-boosted.
4. [build] `synthesize_residual.py` — verdict across both, all metrics.
5. [run] parallel via `run_one.sbatch n4recon / n4cog` (:ro overlay, 4 cpu).
6. [report] `findings_residual.md` + sync to git.
