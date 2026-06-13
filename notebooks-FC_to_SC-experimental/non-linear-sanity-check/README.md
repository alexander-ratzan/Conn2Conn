# Non-linear sanity check

**Question**: the `tractography_predict/` results (richer tractography is a dead end;
FC unique downstream; FC↔SC asymmetry robust) were all produced with **linear**
closed-form models (PCA+PLS, BayesianRidge). Does tractography carry signal in a form
that linear models structurally *cannot* see — i.e. nonlinear / conjunctive structure?

This suite holds the representation and splits fixed and swaps **only the estimator**,
so any difference is attributable to model class. Reuses
`../tractography_predict/_tract_setup.py` (via `_nl_common.py`) so the linear baselines
are byte-identical to the linear runs.

## Estimators

| Estimator | Inductive bias |
|---|---|
| linear (PLS / BayesianRidge) | reference (== tractography_predict) |
| HistGradientBoosting | thresholds + interactions (conjunctive structure) |
| KernelRidge (RBF) | smooth local nonlinearity; multi-output (cheap for recon) |

All operate on PCA-reduced (256), standardized source latents. RBF gamma via the
median pairwise-distance heuristic; alpha=1.0 (fixed — see caveat).

## Experiments

| Script | Question | Metrics |
|---|---|---|
| `n1_nonlinear_cognition.py` | Does HGB/KR extract cognition from tractography that linear BR missed? | Pearson, Spearman (rank), R²; lift over bv+demo floor |
| `n2_nonlinear_reconstruction.py` | Does KR beat PLS at FC↔{SC,r2t,r2t_corr} reconstruction + change the asymmetry? | **full 6-metric panel** (demeaned_pearson, pearson, top1_acc, avg_rank, mse, r2) per estimator, both directions |
| `n3_nonlinear_marginal.py` | Does r2t add over SC under a nonlinear combined model (conjunctive)? | full 6-metric panel; paired Δ |
| `synthesize_nonlinear.py` | linear-vs-nonlinear verdict across all three | — |

Run via `run_one.sbatch <n1|n2|n3>` (parallel, `:ro` overlay, 4 cpu / 24G each).

## Decision rules (fixed before running)

- **N1 cognition unlock**: a tractography rep × nonlinear beats the bv+demo floor by
  ≥ +0.03 Pearson on ≥2/3 targets, 10-seed Wilcoxon p<0.05.
- **N2 reconstruction gain**: KR beats PLS by ≥ 0.02 demeaned_pearson AND closes ≥half
  the count-vs-bundle gap.
- **N3 marginal gain**: KR Δ(SC_r2t − SC) ≥ +0.02 demeaned_pearson, Wilcoxon p<0.05.
- **Null (most likely, per the connectome→behavior near-linear-ceiling literature)**:
  nonlinear ≈ linear → dead-end + asymmetry are model-class-robust (stronger finding).

## Caveats

- KernelRidge alpha/gamma are fixed (median-heuristic gamma, alpha=1.0), not CV-tuned.
  The decision margins (+0.03 / +0.02) are wide enough that light tuning is unlikely to
  flip a null, but a hyperparameter sweep is the natural follow-up if anything lands near
  threshold.
- HGB-per-component reconstruction is deferred (256× the fits of multi-output KR). KR is
  the nonlinear reconstruction probe; HGB covers the cheap scalar cognition target. If
  KR-recon shows a gain, escalate to HGB-recon at reduced target rank.
- Single parcellation (Glasser), 10 family-aware seeds, same as the linear runs.

## Outputs

`n{1,2,3}_*_results.csv` + `_summary.csv`, `*_output.txt`, `nonlinear_synthesis.csv`,
and `findings_nonlinear.md` (written after the run).
