# Supplementary Material

This supplement documents the reproducibility and sanity-check discipline behind the main paper.
The purpose is not only to provide extra numbers, but to show how the analysis avoids common silent
failure modes: population-mean-dominated metrics, split drift, missing grid cells, leakage,
dimensionality artifacts, unstable scalar estimators, measurement-noise explanations, and
objective mismatch.

## S1. Data, Parcellations, and Frozen Splits

All analyses used HCP Young Adults subjects with both resting-state FC and diffusion-derived SC,
plus demographics and anatomy-derived brain-volume features. Connectomes were vectorized as the
upper triangle of each parcellation-specific connectivity matrix.

Two parcellations were used:

| Parcellation | Regions | Edges | Notes |
|---|---:|---:|---|
| Glasser | 360 | 64,620 | HCP-MMP1.0 multimodal cortical atlas |
| 4S456Parcels | 456 | 103,740 | 400 cortical Schaefer-style parcels plus 56 subcortical/cerebellar regions |

Splits were frozen once and reused everywhere. The split contract is central to the paper:
reconstruction, imputation artifact generation, downstream prediction, leak checks, and
family-mechanism analyses all load the same `seed{0..9}.json` split files. Splits are
family-aware: related subjects are kept on the same side of the train/test boundary. The same
subject sets are available across Glasser and 4S456Parcels, so the parcellation comparison is
within-cohort rather than confounded by different subjects.

Source artifacts:

- `reproduction/freeze_splits.py`
- `reproduction/splits/seed{0..9}.json`
- `reproduction/_grid_common.py`

## S2. Metrics and Estimators

### S2.1 Why `demeaned_pearson` is primary

Raw connectome correlations are dominated by the group-mean connectome. This makes raw Pearson
large even when individual deviations are not recovered. The primary reconstruction metric is
therefore `demeaned_pearson`: subtract the training-set mean target connectome, then compare each
subject's predicted and observed deviation.

The main text leads with `demeaned_pearson` and uses fingerprinting metrics (`avg_rank`, `top1_acc`)
as secondary checks. Conventional MSE, R2, and raw Pearson are retained in the grid for diagnostic
purposes but should not drive interpretation.

### S2.2 Estimator reporting rule

Every number should be reported with its estimator.

Reconstruction rows include:

- PCA(256)->PLS(64)->inverse-PCA (`pca_pls`)
- BayesianRidge on PCA components (`bayesian_ridge`)
- KernelRidge RBF 3x3 gamma/alpha sweep (`kernel_ridge`)

Downstream scalar rows should be reported with BayesianRidge. The grid includes PCA->PLS and
KernelRidge scalar rows for completeness, but these are diagnostic-only. A resolved discrepancy
showed why: the older report's cognition baseline and oracle numbers matched BayesianRidge exactly,
while an analyst had accidentally read PCA->PLS rows. For scalar targets, PCA->PLS and KernelRidge
produced ill-conditioned behavior, including extremely negative R2 and parcellation-inconsistent
results for parcellation-independent inputs.

Key resolved examples:

| Quantity | PCA->PLS | BayesianRidge |
|---|---:|---:|
| Glasser SC->SC oracle, `demeaned_pearson` | 0.376 | 0.648 |
| Glasser `bv+demo` -> CogTotal, Pearson | 0.129 | 0.359 |

Source artifact:

- `reproduction/exploration/DISCREPANCY_RESOLUTION.md`

## S3. Grid Completeness and Leak Guardrails

The reproduction grid is designed around an explicit completeness contract. Expected cells are
generated from the grid configuration before running, and completed outputs are checked against that
manifest. Missing cells, unexpected non-finite values, and silently absent metrics are treated as
failures rather than tolerated as blank table entries.

The spine grid contains 13,640 cells:

- 2,640 reconstruction cells
- 11,000 downstream cells

The merged output is complete: 13,640/13,640 cells, zero non-finite values in load-bearing metrics,
and zero hard leak failures.

Leak checks separate expected biological signal from implementation leakage. Sex and age prediction
from connectomes can be real biological signal, while sex/age prediction from incorrectly derived
inputs can indicate leakage. The grid records verdict categories:

| Verdict | Meaning |
|---|---|
| `ok` | No threshold violation |
| `EXPECTED_SIGNAL` | Raw connectome predicts sex/age; interpreted as biological signal, not a leak |
| `EXEMPT_FLAGGED` | Input contains `bv+demo`; cognition rows are interpretable but sex/age is expected |
| `LEAK_FAIL` | Non-exempt derived input exceeds leak threshold |

Observed counts:

| Verdict | Count |
|---|---:|
| `ok` | 3344 |
| `EXPECTED_SIGNAL` | 4 |
| `EXEMPT_FLAGGED` | 1052 |
| `LEAK_FAIL` | 0 |

Only the combined observed FC+SC input crossed the sex threshold as expected biological signal.
No demographic-free derived input triggered `LEAK_FAIL`.

Source artifacts:

- `reproduction/gen_expected_cells.py`
- `reproduction/configs/expected_cells.csv`
- `reproduction/verify_completeness.py`
- `reproduction/outputs/leak_verdict.csv`
- `reproduction/reports/reproduction_findings.md`

## S4. Reduction-Axis Robustness

A natural concern is that the FC->SC > SC->FC asymmetry could be produced by the chosen PCA
reduction rather than the data. The reduction-axis check tested five dimensionality strategies over
10 seeds and both directions:

| Method | Median FC->SC dp | Median SC->FC dp | Median ratio | p vs 1 |
|---|---:|---:|---:|---:|
| PCA->PLS->PCA | 0.1355 | 0.0847 | 1.621x | 0.001 |
| Full PLS on raw 64,620 edges | 0.1382 | 0.0762 | 1.813x | 0.001 |
| JL Gaussian dense | 0.0837 | 0.0610 | 1.397x | 0.001 |
| JL sparse auto | 0.0841 | 0.0592 | 1.389x | 0.001 |
| JL sparse 1/3 | 0.0846 | 0.0535 | 1.554x | 0.001 |

All methods preserve the asymmetry. Full PLS, which removes learned PCA reduction entirely, yields
an even stronger ratio. Random Johnson-Lindenstrauss projections lower absolute reconstruction
quality, as expected, but retain the direction. The asymmetry is therefore a data property rather
than a PCA artifact.

Source artifact:

- `notebooks-FC_to_SC-experimental/sanity_checks/preprocessing_check/findings.md`

## S5. Reliability Ceilings and FC Measurement Noise

The main question for SC->FC is whether the weak prediction reflects true cross-modal independence
or merely noisy FC targets. HCP-YA includes repeated FC measurements, allowing an FC-side
reliability analysis.

### S5.1 FC is noisy by edge but reliable as a whole connectome

At the single-edge level, much of the between-subject variance is noise. Yet the whole connectome is
highly identifiable.

| Parcellation | Trait | State/day | Within-session | Noise | G of averaged connectome |
|---|---:|---:|---:|---:|---:|
| Glasser | 0.301 | 0.035 | 0.021 | 0.643 | 0.585 |
| 4S456 | 0.261 | 0.035 | 0.021 | 0.683 | 0.519 |

Whole-connectome fingerprint top1 was 0.927 in Glasser and 0.934 in 4S456, with discriminability
near 1.0. The reconciliation is that individual signal is distributed: edges are noisy, but the
multivariate pattern is stable.

### S5.2 FC reproducibility ceiling

Between-session FC reliability is the valid FC reproducibility ceiling. In Glasser, REST1 vs REST2
reached `demeaned_r=0.491`, fingerprint `top1=0.933`, and `avg_rank=0.992`. The within-session
LR/RL rung was lower because phase-encode distortion differs across directions, so it is not used as
the clean short-interval ceiling.

| Parcellation | Comparison | `demeaned_r` | Pearson | top1 | avg_rank |
|---|---|---:|---:|---:|---:|
| Glasser | within-session LR/RL | 0.364 | 0.710 | 0.780 | 0.968 |
| Glasser | between-session REST1/REST2 | 0.491 | 0.813 | 0.933 | 0.992 |
| 4S456 | within-session LR/RL | 0.337 | 0.670 | 0.780 | 0.969 |
| 4S456 | between-session REST1/REST2 | 0.457 | 0.783 | 0.936 | 0.993 |

### S5.3 SC->FC captures a small fraction of reproducible FC

Using the Glasser between-session ceiling of 0.491, SC->FC captures about 17% of reproducible FC
in `demeaned_pearson` and about 5% of the fingerprinting ceiling.

| Source -> FC | Metric | Achieved | Ceiling | Fraction |
|---|---|---:|---:|---:|
| SC->FC | `demeaned_r` | 0.085 | 0.491 | 0.17 |
| SC->FC | top1 | 0.051 | 0.933 | 0.05 |
| `bv+demo`->FC | `demeaned_r` | 0.098 | 0.491 | 0.20 |

### S5.4 Per-subject reliability does not explain SC->FC

Per-subject FC reliability is heterogeneous, spanning roughly 0 to 0.78. If SC->FC were limited by
FC measurement noise, cleaner subjects should be more predictable from SC. They are not. A
subject's SC->FC quality is uncorrelated with that subject's FC reliability (Pearson `r=0.01`,
`p=0.72`). Filtering to high-reliability subjects raises the ceiling but leaves SC->FC prediction
near 0.08, reducing the fraction captured.

The correct caveat is that this is FC-side evidence. HCP-YA does not provide SC test-retest dMRI, so
a uniform SC noise floor cannot be fully ruled out. The supported conclusion is that the SC->FC gap
is not explained by FC measurement noise or per-subject FC reliability.

Source artifact:

- `notebooks-FC_to_SC-experimental/sanity_checks/noise_sanity_check/findings_noise.md`

## S6. Nonlinear, Residual, Sink, and Scaling Nulls

The nonlinear sanity checks ask whether the linear ceiling is a modeling failure. They use several
orthogonal probes:

| Check | Question | Interpretation |
|---|---|---|
| Model-class nonlinearity | Do KernelRidge/HGB recover hidden reconstruction or cognition signal? | No stable gain over linear baselines |
| Residual boost | If the model is handed the linear solution, can it learn the residual? | No meaningful residual signal |
| Cross-modal sink | Do FC x SC x r2t interactions unlock cognition? | No useful downstream gain |
| Scaling curve | Does a nonlinear gap grow from n=100 to n~683? | Gap is flat rather than expanding |

KernelRidge also proved hyperparameter-insensitive in the reproduction grid: the 3x3 gamma/alpha
sweep had within-seed max-min spreads around 0.004-0.005 in `demeaned_pearson`. The nine variants
therefore behave almost like a single estimator for the present data.

Source artifacts:

- `notebooks-FC_to_SC-experimental/non-linear-sanity-check/findings_nonlinear.md`
- `notebooks-FC_to_SC-experimental/non-linear-sanity-check/findings_residual.md`
- `notebooks-FC_to_SC-experimental/non-linear-sanity-check/findings_scaling.md`
- `notebooks-FC_to_SC-experimental/non-linear-sanity-check/nonlinear_synthesis.csv`
- `notebooks-FC_to_SC-experimental/non-linear-sanity-check/residual_synthesis.csv`
- `notebooks-FC_to_SC-experimental/non-linear-sanity-check/scaling_synthesis.csv`

## S7. Richer Structural Representations

The tractography extension asks whether streamline-count SC is too crude. Named-bundle
tractography (`r2t`) was tested as an alternative structural representation.

The result was negative. r2t features predicted FC worse than count-SC, added no marginal value over
SC in marginal tests, and did not lift cognition above the `bv+demo` floor. This supports the
interpretation that the missing cognition signal is not simply hiding in a richer structural feature
set, at least in this regime.

Source artifacts:

- `notebooks-FC_to_SC-experimental/tractography_predict/e1_source_rep_results.csv`
- `notebooks-FC_to_SC-experimental/tractography_predict/e2_asymmetry_summary.csv`
- `notebooks-FC_to_SC-experimental/tractography_predict/e3_marginal_summary.csv`
- `notebooks-FC_to_SC-experimental/tractography_predict/e5_downstream_summary.csv`

## S8. Family Signal and Objective Tradeoff

The family-mechanism grid ports the exploratory notebook analyses into the same frozen-split,
both-parcellation framework used by the main grid. The ports are validated against notebook outputs
with exact or near-exact regression tests.

Main findings:

| Finding | Result | Interpretation |
|---|---|---|
| Predicted SC carries family signal | `pred_SC_resid_bvdemo` sibling AUC 0.810 Glasser / 0.816 4S456 | FC-predicted SC contains heritable information beyond cheap subject information |
| Baseline is weak | `bv+demo` sibling AUC 0.563 / 0.565 | Family signal is not reducible to the baseline |
| Combined reconstruction objective collapses family ID | `combined_pred_SC` AUC 0.505 / 0.501 | Reconstruction and identification are different objectives |
| Raw predicted SC separates partly | `pred_SC_raw` AUC 0.680 / 0.670 | Some family signal survives before residualization |

This is the constructive counterweight to the cognition result. Predicted connectomes can be useful,
but not generically. The task objective determines which signal survives.

Source artifacts:

- `reproduction/family_mechanism/outputs/family_auc.csv`
- `reproduction/family_mechanism/tests/test_aggregation_matches_notebook.py`
- `reproduction/family_mechanism/tests/test_f8_localization_matches_notebook.py`

## S9. Mechanism Mode, Confounds, and Localization

The mechanism analysis should be framed as exploratory. The strongest version is not "PC3 is the
mechanism"; it is "a property-selected, low-variance, FC-predictable SC mode carries weak family
signal." The component index migrates across parcellations: PC3 in Glasser and PC4 in 4S456.

Important safeguards:

- PC1 is strongly confounded with sex and brain volume (R2 about 0.89-0.93) and should not be
  interpreted as a family mechanism.
- An earlier PC2 signal failed under cross-seed alignment, illustrating why single-seed mechanisms
  are fragile.
- The mechanism mode has small effects: FC-predictability around R2=0.22-0.24 and sibling AUC around
  0.58.
- The spatial localization concentrates in visual and dorsal-attention edges, but the interpretation
  is hedged.

The tractography-reliability check tested whether this localization was simply due to high-strength,
short-distance, more reliable tractography edges. Strength and distance explained about 41% of
`|PC3|` magnitude, so the confound is real. After partialling both, the visual/DAN enrichment
survived:

| Network pair | Raw enrichment | Residualized enrichment |
|---|---:|---:|
| visual-visual | 11.97x | 13.32x |
| dorsal attention-dorsal attention | 7.73x | 5.73x |
| dorsal attention-visual | 4.49x | 4.99x |

Thus reliability-related proxies explain part of the magnitude but do not erase the localization.
The rigorous retest-ICC version remains future work.

Source artifacts:

- `reproduction/family_mechanism/outputs/f8_per_pc.csv`
- `reproduction/family_mechanism/outputs/f8_stability.csv`
- `reproduction/family_mechanism/outputs/f8_pc3_localization.csv`
- `reproduction/family_mechanism/outputs/f8_pc3_enrichment_agg.csv`
- `notebooks-FC_to_SC-experimental/sanity_checks/tract_check/findings.md`

## S10. Operational Reproducibility

The project uses a two-layer design:

1. Exploratory notebooks discovered the findings and mechanism hypotheses.
2. The `reproduction/` suite imports the same shared setup functions, freezes the splits, reruns
   confirmatory grids on both parcellations and 10 seeds, writes CSV mirrors, and verifies
   completeness.

CSV outputs are the source of truth. W&B is a replay layer, not a dependency. This matters because
offline or failed syncs cannot change the analysis.

The operational discipline includes:

- Git-synced compute state.
- Frozen split manifests.
- CSV append with provenance.
- Expected-cell manifest before running.
- Verification after merging.
- Leak verdicts separated from scientific outcomes.
- Notebook-vs-grid regression checks for family and mechanism analyses.

Source artifacts:

- `dev-notes/PROJECT-HANDOFF-2026-06-22.md`
- `planning/reproducibility_and_grid_plan_theory.md`
- `planning/reproducibility_and_grid_plan_runlog.md`
- `reproduction/upload_to_wandb.py`

## S11. Supplementary Tables To Generate

These tables should be generated mechanically from CSVs before submission:

| Table | Source |
|---|---|
| S1 full reconstruction grid | `reproduction/outputs/reconstruction.csv` |
| S2 full downstream grid | `reproduction/outputs/downstream.csv` |
| S3 leak-check verdicts | `reproduction/outputs/leak_verdict.csv` |
| S4 completeness manifest diff | `reproduction/configs/expected_cells.csv` + merged outputs |
| S5 reduction-axis robustness | `sanity_checks/preprocessing_check/reduction_axis_summary.csv` |
| S6 FC reliability and variance decomposition | `sanity_checks/noise_sanity_check/outputs/*.csv` |
| S7 nonlinear null synthesis | `non-linear-sanity-check/*synthesis.csv` |
| S8 r2t tractography synthesis | `tractography_predict/*summary.csv` |
| S9 family AUCs | `reproduction/family_mechanism/outputs/family_auc.csv` |
| S10 F8 mechanism mode | `reproduction/family_mechanism/outputs/f8_*.csv` |

## S12. Supplementary Figures To Generate

These figures should be generated after the text stabilizes:

| Figure | Content |
|---|---|
| S1 | Raw Pearson vs `demeaned_pearson` illustration |
| S2 | Seed-level reconstruction distributions |
| S3 | Downstream lift distributions by seed |
| S4 | Sex/age leak-check scores by input |
| S5 | Reduction-axis robustness ratios |
| S6 | FC reliability histogram and SC->FC achieved-vs-reliability scatter |
| S7 | Nonlinear and scaling null plots |
| S8 | r2t vs count-SC comparison |
| S9 | Family AUC distributions |
| S10 | Mechanism-mode localization and enrichment |
