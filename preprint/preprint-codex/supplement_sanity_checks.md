# Supplementary Material: Sanity Checks and Reproducibility Discipline


This document is a long-form supplement for the Conn2Conn preprint. It combines the compact manuscript supplement with the detailed sanity-check notes from the repository, plus selected machine-readable CSV summaries. The purpose is to make the negative claims auditable: reduction choices, FC measurement noise, nonlinear capacity, richer tractography, family objectives, and operational grid completeness are all checked explicitly.


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


# Machine-Readable Summary Tables


### Reduction-axis summary

| method      | jl_variant     | n_seeds | median_ratio | min_ratio | max_ratio | wilcoxon_p_vs_1 | median_FC_to_SC_dp | median_SC_to_FC_dp |
| ----------- | -------------- | ------- | ------------ | --------- | --------- | --------------- | ------------------ | ------------------ |
| FULL_PLS    |                | 10      | 1.813        | 1.493     | 2.115     | 0.0009766       | 0.1382             | 0.07616            |
| JL_PLS_PCA  | gaussian_dense | 10      | 1.397        | 1.194     | 2.25      | 0.0009766       | 0.08372            | 0.06102            |
| JL_PLS_PCA  | sparse_auto    | 10      | 1.389        | 1.109     | 1.903     | 0.0009766       | 0.08411            | 0.05918            |
| JL_PLS_PCA  | sparse_third   | 10      | 1.554        | 1.235     | 1.776     | 0.0009766       | 0.08458            | 0.05348            |
| PCA_PLS_PCA |                | 10      | 1.621        | 1.324     | 1.861     | 0.0009766       | 0.1355             | 0.0847             |

Full source: notebooks-FC_to_SC-experimental/ sanity_checks/ preprocessing_check/ reduction_axis_summary.csv.


### Reduction-axis all seed ratios

| method     | jl_variant     | seed | FC->SC  | SC->FC  | ratio |
| ---------- | -------------- | ---- | ------- | ------- | ----- |
| FULL_PLS   |                | 0    | 0.1381  | 0.077   | 1.793 |
| FULL_PLS   |                | 1    | 0.1381  | 0.07533 | 1.833 |
| FULL_PLS   |                | 2    | 0.1323  | 0.0829  | 1.596 |
| FULL_PLS   |                | 3    | 0.1383  | 0.08175 | 1.692 |
| FULL_PLS   |                | 4    | 0.1371  | 0.06864 | 1.997 |
| FULL_PLS   |                | 5    | 0.1476  | 0.07143 | 2.066 |
| FULL_PLS   |                | 6    | 0.1446  | 0.0684  | 2.115 |
| FULL_PLS   |                | 7    | 0.1394  | 0.08403 | 1.659 |
| FULL_PLS   |                | 8    | 0.1403  | 0.07292 | 1.924 |
| FULL_PLS   |                | 9    | 0.1375  | 0.09206 | 1.493 |
| JL_PLS_PCA | gaussian_dense | 0    | 0.07473 | 0.04884 | 1.53  |
| JL_PLS_PCA | gaussian_dense | 1    | 0.08485 | 0.06164 | 1.377 |
| JL_PLS_PCA | gaussian_dense | 2    | 0.07312 | 0.06125 | 1.194 |
| JL_PLS_PCA | gaussian_dense | 3    | 0.08504 | 0.06212 | 1.369 |
| JL_PLS_PCA | gaussian_dense | 4    | 0.08259 | 0.06079 | 1.359 |
| JL_PLS_PCA | gaussian_dense | 5    | 0.09936 | 0.05904 | 1.683 |
| JL_PLS_PCA | gaussian_dense | 6    | 0.0936  | 0.04161 | 2.25  |
| JL_PLS_PCA | gaussian_dense | 7    | 0.08232 | 0.05476 | 1.503 |
| JL_PLS_PCA | gaussian_dense | 8    | 0.09086 | 0.06412 | 1.417 |
| JL_PLS_PCA | gaussian_dense | 9    | 0.08192 | 0.06181 | 1.325 |
| JL_PLS_PCA | sparse_auto    | 0    | 0.08423 | 0.05982 | 1.408 |
| JL_PLS_PCA | sparse_auto    | 1    | 0.08235 | 0.06101 | 1.35  |
| JL_PLS_PCA | sparse_auto    | 2    | 0.06875 | 0.05659 | 1.215 |
| JL_PLS_PCA | sparse_auto    | 3    | 0.08399 | 0.06127 | 1.371 |

Showing first 24 of 50 rows. Full source: notebooks-FC_to_SC-experimental/ sanity_checks/ preprocessing_check/ reduction_axis_synthesis.csv.


### FC reliability ceiling

| parc         | n    | comparison          | demeaned_pearson | pearson | top1_acc | avg_rank | mse     | r2      |
| ------------ | ---- | ------------------- | ---------------- | ------- | -------- | -------- | ------- | ------- |
| Glasser      | 1018 | within_session_run1 | 0.3645           | 0.7099  | 0.775    | 0.9682   | 0.03312 | -0.38   |
| Glasser      | 1018 | within_session_run2 | 0.3598           | 0.7005  | 0.8291   | 0.9888   | 0.03304 | -0.3432 |
| Glasser      | 1018 | between_session     | 0.4905           | 0.8134  | 0.9332   | 0.9916   | 0.01792 | -0.1195 |
| 4S456Parcels | 1018 | within_session_run1 | 0.3371           | 0.6704  | 0.775    | 0.9694   | 0.03239 | -0.4662 |
| 4S456Parcels | 1018 | within_session_run2 | 0.331            | 0.6609  | 0.8261   | 0.9896   | 0.03222 | -0.4263 |
| 4S456Parcels | 1018 | between_session     | 0.4565           | 0.7832  | 0.9361   | 0.9934   | 0.01748 | -0.2337 |

Full source: notebooks-FC_to_SC-experimental/ sanity_checks/ noise_sanity_check/ outputs/ a_reliability_ceiling.csv.


### FC per-subject reliability summary

| parc         | n    | mean   | median | std    | min       | max    | p1     | p5     | p25    | p75    | p95    | p99    | skew    | kurtosis | normaltest_p | frac_below_0.2 | within_vs_between_spearman |
| ------------ | ---- | ------ | ------ | ------ | --------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------- | -------- | ------------ | -------------- | -------------------------- |
| Glasser      | 1018 | 0.4894 | 0.4943 | 0.1173 | -0.001942 | 0.7751 | 0.149  | 0.2818 | 0.4204 | 0.5722 | 0.6645 | 0.7138 | -0.5807 | 0.7912   | 5.229e-15    | 0.02063        | 0.4054                     |
| 4S456Parcels | 1018 | 0.4554 | 0.4624 | 0.1106 | 0.05831   | 0.7451 | 0.1633 | 0.2583 | 0.3904 | 0.5321 | 0.6197 | 0.6727 | -0.4183 | 0.2862   | 1.873e-07    | 0.02652        | 0.4233                     |

Full source: notebooks-FC_to_SC-experimental/ sanity_checks/ noise_sanity_check/ outputs/ g_per_subject_summary.csv.


### Reliability filtering summary

| source  | filter             | n_kept | mean_achieved | mean_ceiling | fraction_of_ceiling |
| ------- | ------------------ | ------ | ------------- | ------------ | ------------------- |
| SC      | all                | 857    | 0.08318       | 0.4914       | 0.1693              |
| SC      | drop_rel_below_0.2 | 841    | 0.08309       | 0.4983       | 0.1667              |
| SC      | drop_bottom_10pct  | 771    | 0.08258       | 0.5171       | 0.1597              |
| SC      | keep_top_50pct     | 429    | 0.07953       | 0.5824       | 0.1366              |
| bv+demo | all                | 857    | 0.102         | 0.4914       | 0.2076              |
| bv+demo | drop_rel_below_0.2 | 841    | 0.1019        | 0.4983       | 0.2045              |
| bv+demo | drop_bottom_10pct  | 771    | 0.104         | 0.5171       | 0.2012              |
| bv+demo | keep_top_50pct     | 429    | 0.1147        | 0.5824       | 0.197               |

Full source: notebooks-FC_to_SC-experimental/ sanity_checks/ noise_sanity_check/ outputs/ h_reliability_filtered_summary.csv.


### Noise synthesis

| headline                                                                                                              |
| --------------------------------------------------------------------------------------------------------------------- |
| FC between-session reliability ceiling (Glasser): demeaned_r=0.491, fingerprint top1=0.933, avg_rank=0.992            |
| FC edge variance (Glasser): trait=30%, state=4%, within-session=2%, noise=64%; averaged-connectome reliability G=0.59 |
| SC->FC captures 17% of the reproducible FC signal (demeaned_r 0.085 of ceiling 0.491)                                 |

Full source: notebooks-FC_to_SC-experimental/ sanity_checks/ noise_sanity_check/ outputs/ noise_synthesis.csv.


### Tractography source representation

| rep          | seed | target | method     | mse     | r2       | pearson | avg_rank | top1_acc | demeaned_pearson |
| ------------ | ---- | ------ | ---------- | ------- | -------- | ------- | -------- | -------- | ---------------- |
| SC           | 0    | FC     | shared_pca | 0.01398 | -0.06216 | 0.8307  | 0.7125   | 0.03077  | 0.0849           |
| r2t          | 0    | FC     | shared_pca | 0.01506 | -0.1403  | 0.8176  | 0.6082   | 0.01538  | 0.03969          |
| r2t_corr     | 0    | FC     | shared_pca | 0.0145  | -0.09805 | 0.8242  | 0.5935   | 0.02051  | 0.03549          |
| SC_r2t       | 0    | FC     | block_pca  | 0.01416 | -0.07531 | 0.8286  | 0.6948   | 0.01538  | 0.07302          |
| kitchen_sink | 0    | FC     | block_pca  | 0.01406 | -0.06776 | 0.8295  | 0.7082   | 0.03077  | 0.0853           |
| SC           | 1    | FC     | shared_pca | 0.01445 | -0.05236 | 0.8224  | 0.7213   | 0.05128  | 0.08621          |
| r2t          | 1    | FC     | shared_pca | 0.01534 | -0.114   | 0.8115  | 0.6277   | 0.01538  | 0.05158          |
| r2t_corr     | 1    | FC     | shared_pca | 0.01516 | -0.09946 | 0.8135  | 0.5857   | 0.005128 | 0.02439          |
| SC_r2t       | 1    | FC     | block_pca  | 0.01449 | -0.05483 | 0.8219  | 0.7238   | 0.02564  | 0.08606          |
| kitchen_sink | 1    | FC     | block_pca  | 0.01433 | -0.04436 | 0.8238  | 0.7554   | 0.06154  | 0.1059           |
| SC           | 2    | FC     | shared_pca | 0.01432 | -0.05061 | 0.8277  | 0.7082   | 0.05128  | 0.08343          |
| r2t          | 2    | FC     | shared_pca | 0.01545 | -0.1304  | 0.8146  | 0.602    | 0.01538  | 0.04751          |
| r2t_corr     | 2    | FC     | shared_pca | 0.01479 | -0.08391 | 0.8223  | 0.6145   | 0.01026  | 0.04052          |
| SC_r2t       | 2    | FC     | block_pca  | 0.01447 | -0.06154 | 0.8263  | 0.7045   | 0.02051  | 0.07893          |

Showing first 14 of 50 rows. Full source: notebooks-FC_to_SC-experimental/ tractography_predict/ e1_source_rep_results.csv.


### Tractography asymmetry summary

| rep      | metric           | comparison_kind | n_seeds | median_FCwins | min_FCwins | max_FCwins | wilcoxon_p_FCwins | median_FC_to_X | median_X_to_FC |
| -------- | ---------------- | --------------- | ------- | ------------- | ---------- | ---------- | ----------------- | -------------- | -------------- |
| SC       | demeaned_pearson | ratio_FCwins    | 10      | 1.621         | 1.324      | 1.861      | 0.0009766         | 0.1355         | 0.0847         |
| SC       | pearson          | ratio_FCwins    | 10      | 1.101         | 1.096      | 1.111      | 0.0009766         | 0.9125         | 0.8292         |
| SC       | top1_acc         | ratio_FCwins    | 10      | 2.562         | 1.5        | 5          | 0.0009766         | 0.1179         | 0.05128        |
| SC       | avg_rank         | ratio_FCwins    | 10      | 1.225         | 1.174      | 1.265      | 0.0009766         | 0.8787         | 0.7128         |
| SC       | mse              | ratio_FCwins    | 10      | 2.594         | 2.518      | 2.639      | 0.0009766         | 0.005533       | 0.01429        |
| SC       | r2               | diff_FCwins     | 10      | 0.03595       | 0.01707    | 0.05097    | 0.0009766         | -0.02205       | -0.05716       |
| r2t      | demeaned_pearson | ratio_FCwins    | 10      | 2.264         | 1.871      | 2.83       | 0.0009766         | 0.1113         | 0.04925        |
| r2t      | pearson          | ratio_FCwins    | 10      | 1.037         | 1.029      | 1.04       | 0.0009766         | 0.8453         | 0.8153         |
| r2t      | top1_acc         | ratio_FCwins    | 9       | 1.5           | 0.6667     | 3          | 0.03906           | 0.02821        | 0.02051        |
| r2t      | avg_rank         | ratio_FCwins    | 10      | 1.185         | 1.121      | 1.232      | 0.0009766         | 0.7212         | 0.609          |
| r2t      | mse              | ratio_FCwins    | 10      | 2.835e-08     | 2.819e-08  | 2.909e-08  | 1                 | 5.365e+05      | 0.01537        |
| r2t      | r2               | diff_FCwins     | 10      | 0.4459        | -0.3524    | 0.7425     | 0.001953          | 0.3234         | -0.127         |
| r2t_corr | demeaned_pearson | ratio_FCwins    | 10      | 1.634         | 1.178      | 2.39       | 0.0009766         | 0.05816        | 0.03625        |
| r2t_corr | pearson          | ratio_FCwins    | 10      | 0.9813        | 0.9751     | 0.9898     | 1                 | 0.8071         | 0.8231         |

Showing first 14 of 18 rows. Full source: notebooks-FC_to_SC-experimental/ tractography_predict/ e2_asymmetry_summary.csv.


### Tractography marginal summary

| n_seeds | median_dp_SC | median_dp_SC_r2t | median_delta | min_delta | max_delta | wilcoxon_p_two_sided | wilcoxon_p_one_sided_greater |
| ------- | ------------ | ---------------- | ------------ | --------- | --------- | -------------------- | ---------------------------- |
| 10      | 0.0847       | 0.07908          | -0.001289    | -0.01188  | 0.0061    | 0.04883              | 0.9814                       |

Full source: notebooks-FC_to_SC-experimental/ tractography_predict/ e3_marginal_summary.csv.


### Tractography downstream summary

| rep     | target               | pearson_raw | pearson_resid | lift_over_bvdemo_raw |
| ------- | -------------------- | ----------- | ------------- | -------------------- |
| FC      | CogCrystalComp_Unadj | 0.4508      | 0.3737        | 0.1046               |
| FC      | CogFluidComp_Unadj   | 0.3423      | 0.2179        | 0.03286              |
| FC      | CogTotalComp_Unadj   | 0.4508      | 0.2636        | 0.07845              |
| SC      | CogCrystalComp_Unadj | 0.2674      | 0.08626       | -0.07878             |
| SC      | CogFluidComp_Unadj   | 0.1798      | 0.006016      | -0.1297              |
| SC      | CogTotalComp_Unadj   | 0.2614      | 0.04037       | -0.1109              |
| SC_r2t  | CogCrystalComp_Unadj | 0.1597      | 0.02804       | -0.1865              |
| SC_r2t  | CogFluidComp_Unadj   | 0.1352      | 0.07167       | -0.1743              |
| SC_r2t  | CogTotalComp_Unadj   | 0.1958      | 0.04758       | -0.1765              |
| bv+demo | CogCrystalComp_Unadj | 0.3462      |               | 0                    |
| bv+demo | CogFluidComp_Unadj   | 0.3095      |               | 0                    |
| bv+demo | CogTotalComp_Unadj   | 0.3723      |               | 0                    |
| r2t     | CogCrystalComp_Unadj | 0.1636      | 0.02728       | -0.1826              |
| r2t     | CogFluidComp_Unadj   | 0.1303      | 0.06082       | -0.1792              |

Showing first 14 of 21 rows. Full source: notebooks-FC_to_SC-experimental/ tractography_predict/ e5_downstream_summary.csv.


### Nonlinear cognition summary

| rep | estimator | target               | pearson | spearman | r2        | lift_over_bvdemo |
| --- | --------- | -------------------- | ------- | -------- | --------- | ---------------- |
| FC  | HGB       | CogCrystalComp_Unadj | 0.2493  | 0.2184   | 0.0541    | -0.117           |
| FC  | HGB       | CogFluidComp_Unadj   | 0.118   | 0.1419   | 0.01019   | -0.1073          |
| FC  | HGB       | CogTotalComp_Unadj   | 0.2536  | 0.2596   | 0.04469   | -0.101           |
| FC  | KR        | CogCrystalComp_Unadj | 0.3377  | 0.3051   | -3.094    | 0.06809          |
| FC  | KR        | CogFluidComp_Unadj   | 0.1781  | 0.1507   | -2.125    | -0.01158         |
| FC  | KR        | CogTotalComp_Unadj   | 0.3375  | 0.2912   | -1.51     | 0.05574          |
| FC  | linear_BR | CogCrystalComp_Unadj | 0.4508  | 0.44     | 0.1961    | 0.1046           |
| FC  | linear_BR | CogFluidComp_Unadj   | 0.3423  | 0.2989   | 0.1079    | 0.03286          |
| FC  | linear_BR | CogTotalComp_Unadj   | 0.4508  | 0.4235   | 0.1943    | 0.07845          |
| SC  | HGB       | CogCrystalComp_Unadj | 0.1722  | 0.162    | 0.01034   | -0.1941          |
| SC  | HGB       | CogFluidComp_Unadj   | 0.03564 | 0.01659  | -0.006568 | -0.1896          |
| SC  | HGB       | CogTotalComp_Unadj   | 0.1255  | 0.1219   | 0.01152   | -0.2291          |
| SC  | KR        | CogCrystalComp_Unadj | 0.241   | 0.2229   | -2.468    | -0.02867         |
| SC  | KR        | CogFluidComp_Unadj   | 0.1135  | 0.1124   | -1.603    | -0.07622         |

Showing first 14 of 54 rows. Full source: notebooks-FC_to_SC-experimental/ non-linear-sanity-check/ n1_cognition_summary.csv.


### Nonlinear reconstruction summary

| rep | estimator  | metric           | median_FC_to_X | median_X_to_FC | asym_FCwins |
| --- | ---------- | ---------------- | -------------- | -------------- | ----------- |
| SC  | linear_PLS | demeaned_pearson | 0.1355         | 0.0847         | 1.621       |
| SC  | linear_PLS | pearson          | 0.9125         | 0.8292         | 1.101       |
| SC  | linear_PLS | top1_acc         | 0.1179         | 0.05128        | 2.562       |
| SC  | linear_PLS | avg_rank         | 0.8787         | 0.7128         | 1.225       |
| SC  | linear_PLS | mse              | 0.005533       | 0.01429        | 2.594       |
| SC  | linear_PLS | r2               | -0.02205       | -0.05716       | 0.03595     |
| SC  | KR         | demeaned_pearson | 0.1334         | 0.08116        | 1.638       |
| SC  | KR         | pearson          | 0.915          | 0.8355         | 1.096       |
| SC  | KR         | top1_acc         | 0.159          | 0.04872        | 3.422       |
| SC  | KR         | avg_rank         | 0.9104         | 0.7313         | 1.251       |
| SC  | KR         | mse              | 0.005401       | 0.0138         | 2.578       |
| SC  | KR         | r2               | 0.002031       | -0.02052       | 0.02313     |
| r2t | linear_PLS | demeaned_pearson | 0.1113         | 0.04925        | 2.264       |
| r2t | linear_PLS | pearson          | 0.8453         | 0.8153         | 1.037       |

Showing first 14 of 36 rows. Full source: notebooks-FC_to_SC-experimental/ non-linear-sanity-check/ n2_reconstruction_summary.csv.


### Nonlinear marginal summary

| estimator | metric           | median_SC | median_SC_r2t | median_delta | wilcoxon_p_greater |
| --------- | ---------------- | --------- | ------------- | ------------ | ------------------ |
| linear    | demeaned_pearson | 0.0847    | 0.07908       | -0.001289    | 0.9814             |
| linear    | pearson          | 0.8292    | 0.8272        | -0.00138     | 0.999              |
| linear    | top1_acc         | 0.05128   | 0.0359        | -0.01538     | 0.9404             |
| linear    | avg_rank         | 0.7128    | 0.702         | -0.000618    | 0.8125             |
| linear    | mse              | 0.01429   | 0.01447       | 0.0001069    | 0.00293            |
| linear    | r2               | -0.05716  | -0.0638       | -0.007281    | 0.999              |
| KR        | demeaned_pearson | 0.08116   | 0.0735        | -0.004938    | 0.998              |
| KR        | pearson          | 0.8355    | 0.835         | -0.0001581   | 0.9346             |
| KR        | top1_acc         | 0.04872   | 0.04615       | -0.005102    | 0.9492             |
| KR        | avg_rank         | 0.7313    | 0.7021        | -0.02004     | 1                  |
| KR        | mse              | 0.0138    | 0.01384       | 5.78e-06     | 0.2783             |
| KR        | r2               | -0.02052  | -0.02028      | 0.0001494    | 0.3848             |

Full source: notebooks-FC_to_SC-experimental/ non-linear-sanity-check/ n3_marginal_summary.csv.


### Residual cognition summary

| rep    | target               | variant  | pearson | spearman | r2        |
| ------ | -------------------- | -------- | ------- | -------- | --------- |
| FC     | CogCrystalComp_Unadj | final    | 0.4538  | 0.4285   | 0.1975    |
| FC     | CogCrystalComp_Unadj | template | 0.4516  | 0.4306   | 0.1965    |
| FC     | CogFluidComp_Unadj   | final    | 0.3096  | 0.2926   | 0.078     |
| FC     | CogFluidComp_Unadj   | template | 0.3057  | 0.2881   | 0.08668   |
| FC     | CogTotalComp_Unadj   | final    | 0.4349  | 0.3978   | 0.1847    |
| FC     | CogTotalComp_Unadj   | template | 0.4364  | 0.403    | 0.1723    |
| SC     | CogCrystalComp_Unadj | final    | 0.1923  | 0.1797   | -0.006415 |
| SC     | CogCrystalComp_Unadj | template | 0.1936  | 0.1803   | 0.02854   |
| SC     | CogFluidComp_Unadj   | final    | 0.1201  | 0.1167   | -0.01965  |
| SC     | CogFluidComp_Unadj   | template | 0.1277  | 0.1193   | 0.01068   |
| SC     | CogTotalComp_Unadj   | final    | 0.192   | 0.1808   | -0.00433  |
| SC     | CogTotalComp_Unadj   | template | 0.1972  | 0.1814   | 0.02963   |
| SC_r2t | CogCrystalComp_Unadj | final    | 0.1621  | 0.162    | -0.09939  |
| SC_r2t | CogCrystalComp_Unadj | template | 0.1646  | 0.1632   | 0.004629  |

Showing first 14 of 36 rows. Full source: notebooks-FC_to_SC-experimental/ non-linear-sanity-check/ n4_cog_summary.csv.


### Residual reconstruction summary

| rep | direction | metric           | median_template | median_final | median_improvement | wilcoxon_p_improve |
| --- | --------- | ---------------- | --------------- | ------------ | ------------------ | ------------------ |
| SC  | FC->SC    | demeaned_pearson | 0.1355          | 0.1405       | 0.005203           | 0.0009766          |
| SC  | FC->SC    | pearson          | 0.9125          | 0.9079       | -0.004731          | 1                  |
| SC  | FC->SC    | top1_acc         | 0.1179          | 0.1282       | 0.007692           | 0.1797             |
| SC  | FC->SC    | avg_rank         | 0.8787          | 0.8771       | -0.003734          | 0.9473             |
| SC  | FC->SC    | mse              | 0.005533        | 0.005827     | -0.0002896         | 1                  |
| SC  | FC->SC    | r2               | -0.02205        | -0.0722      | -0.05126           | 1                  |
| SC  | SC->FC    | demeaned_pearson | 0.0847          | 0.08665      | 0.002296           | 0.0009766          |
| SC  | SC->FC    | pearson          | 0.8292          | 0.8179       | -0.01112           | 1                  |
| SC  | SC->FC    | top1_acc         | 0.05128         | 0.04103      | -0.005128          | 0.6963             |
| SC  | SC->FC    | avg_rank         | 0.7128          | 0.7118       | -0.002972          | 0.6523             |
| SC  | SC->FC    | mse              | 0.01429         | 0.01517      | -0.0009521         | 1                  |
| SC  | SC->FC    | r2               | -0.05716        | -0.1255      | -0.0692            | 1                  |
| r2t | FC->r2t   | demeaned_pearson | 0.1113          | 0.1169       | 0.005779           | 0.0009766          |
| r2t | FC->r2t   | pearson          | 0.8453          | 0.834        | -0.01054           | 1                  |

Showing first 14 of 36 rows. Full source: notebooks-FC_to_SC-experimental/ non-linear-sanity-check/ n4_recon_summary.csv.


### Sink cognition summary

| rep           | target               | pearson | spearman | r2      |
| ------------- | -------------------- | ------- | -------- | ------- |
| FC            | CogCrystalComp_Unadj | 0.4516  | 0.4306   | 0.1965  |
| FC            | CogFluidComp_Unadj   | 0.3057  | 0.2881   | 0.08668 |
| FC            | CogTotalComp_Unadj   | 0.4364  | 0.403    | 0.1723  |
| bv+demo       | CogCrystalComp_Unadj | 0.3486  | 0.3493   | 0.09952 |
| bv+demo       | CogFluidComp_Unadj   | 0.2978  | 0.2878   | 0.08178 |
| bv+demo       | CogTotalComp_Unadj   | 0.3731  | 0.3449   | 0.1322  |
| sink_linear   | CogCrystalComp_Unadj | 0.4137  | 0.4024   | 0.1655  |
| sink_linear   | CogFluidComp_Unadj   | 0.3013  | 0.2957   | 0.08772 |
| sink_linear   | CogTotalComp_Unadj   | 0.4012  | 0.3809   | 0.1525  |
| sink_residual | CogCrystalComp_Unadj | 0.4147  | 0.3986   | 0.1441  |
| sink_residual | CogFluidComp_Unadj   | 0.2855  | 0.2799   | 0.05765 |
| sink_residual | CogTotalComp_Unadj   | 0.3809  | 0.363    | 0.1133  |

Full source: notebooks-FC_to_SC-experimental/ non-linear-sanity-check/ n5_cog_summary.csv.


### Sink reconstruction summary

| rep           | demeaned_pearson | pearson | top1_acc | avg_rank | mse     | r2       |
| ------------- | ---------------- | ------- | -------- | -------- | ------- | -------- |
| SC_linear_ref | 0.0847           | 0.8292  | 0.05128  | 0.7128   | 0.01429 | -0.05716 |
| sink_linear   | 0.09416          | 0.8288  | 0.04872  | 0.7229   | 0.01431 | -0.05738 |
| sink_residual | 0.09589          | 0.8179  | 0.05114  | 0.7257   | 0.01518 | -0.1191  |

Full source: notebooks-FC_to_SC-experimental/ non-linear-sanity-check/ n5_recon_summary.csv.


### Scaling summary

| task           | n_sub | n_seeds | median_linear | median_final | median_gap | gap_min   | gap_max   | wilcoxon_p_gap_gt0 |
| -------------- | ----- | ------- | ------------- | ------------ | ---------- | --------- | --------- | ------------------ |
| cognition      | 100   | 10      | 0.3021        | 0.29         | -0.008681  | -0.0352   | -0.00161  | 1                  |
| cognition      | 200   | 10      | 0.3267        | 0.3243       | -0.01516   | -0.032    | 0.0119    | 0.9932             |
| cognition      | 400   | 10      | 0.3936        | 0.3797       | -0.01192   | -0.03402  | 0.0003692 | 0.999              |
| cognition      | 682   | 1       | 0.4825        | 0.4861       | 0.003576   | 0.003576  | 0.003576  | 0.5                |
| cognition      | 683   | 9       | 0.3914        | 0.3746       | -0.01281   | -0.03     | 0.005056  | 0.9902             |
| reconstruction | 100   | 10      | 0.04122       | 0.03526      | -0.005361  | -0.007601 | 0.0002357 | 0.999              |
| reconstruction | 200   | 10      | 0.06697       | 0.06547      | -0.001893  | -0.00573  | 0.005098  | 0.8838             |
| reconstruction | 400   | 10      | 0.09404       | 0.09313      | -0.001541  | -0.004092 | 0.0004269 | 0.9902             |
| reconstruction | 682   | 1       | 0.1098        | 0.1072       | -0.002532  | -0.002532 | -0.002532 | 1                  |
| reconstruction | 683   | 9       | 0.1098        | 0.1097       | -0.0008874 | -0.003509 | 0.001176  | 0.9805             |

Full source: notebooks-FC_to_SC-experimental/ non-linear-sanity-check/ n6_scaling_summary.csv.


### Family AUC

| parcellation | variant              | relation | n_pairs | auc    | auc_lo | auc_hi | p_perm | p_fdr  | sig_fdr |
| ------------ | -------------------- | -------- | ------- | ------ | ------ | ------ | ------ | ------ | ------- |
| Glasser      | obs_SC               | MZ       | 242     | 0.9989 | 0.9981 | 0.9996 | 0      | 0      | True    |
| Glasser      | obs_SC               | DZ       | 127     | 0.938  | 0.9235 | 0.9512 | 0      | 0      | True    |
| Glasser      | obs_SC               | sibling  | 1254    | 0.863  | 0.8499 | 0.8758 | 0      | 0      | True    |
| Glasser      | obs_FC               | MZ       | 242     | 0.9897 | 0.9844 | 0.9938 | 0      | 0      | True    |
| Glasser      | obs_FC               | DZ       | 127     | 0.8859 | 0.8523 | 0.9137 | 0      | 0      | True    |
| Glasser      | obs_FC               | sibling  | 1254    | 0.8229 | 0.8065 | 0.8385 | 0      | 0      | True    |
| Glasser      | pred_SC_raw          | MZ       | 242     | 0.9735 | 0.9647 | 0.9813 | 0      | 0      | True    |
| Glasser      | pred_SC_raw          | DZ       | 127     | 0.7982 | 0.7541 | 0.8396 | 0      | 0      | True    |
| Glasser      | pred_SC_raw          | sibling  | 1254    | 0.6797 | 0.6597 | 0.7002 | 0      | 0      | True    |
| Glasser      | pred_SC_resid_bvdemo | MZ       | 242     | 0.9885 | 0.982  | 0.9938 | 0      | 0      | True    |
| Glasser      | pred_SC_resid_bvdemo | DZ       | 127     | 0.8227 | 0.784  | 0.8604 | 0      | 0      | True    |
| Glasser      | pred_SC_resid_bvdemo | sibling  | 1254    | 0.8097 | 0.7947 | 0.825  | 0      | 0      | True    |
| Glasser      | combined_pred_SC     | MZ       | 242     | 0.8971 | 0.8794 | 0.9145 | 0      | 0      | True    |
| Glasser      | combined_pred_SC     | DZ       | 127     | 0.7006 | 0.6543 | 0.742  | 0      | 0      | True    |
| Glasser      | combined_pred_SC     | sibling  | 1254    | 0.5049 | 0.482  | 0.5258 | 0.6482 | 0.6482 | False   |
| Glasser      | bvdemo_to_SC         | MZ       | 242     | 0.9507 | 0.9353 | 0.9638 | 0      | 0      | True    |
| Glasser      | bvdemo_to_SC         | DZ       | 127     | 0.7997 | 0.7591 | 0.8399 | 0      | 0      | True    |
| Glasser      | bvdemo_to_SC         | sibling  | 1254    | 0.5628 | 0.541  | 0.5851 | 0      | 0      | True    |
| Glasser      | pred_FC_raw          | MZ       | 242     | 0.9104 | 0.8885 | 0.9293 | 0      | 0      | True    |
| Glasser      | pred_FC_raw          | DZ       | 127     | 0.7557 | 0.7152 | 0.7923 | 0      | 0      | True    |
| Glasser      | pred_FC_raw          | sibling  | 1254    | 0.7097 | 0.6907 | 0.7297 | 0      | 0      | True    |
| Glasser      | pred_FC_resid_bvdemo | MZ       | 242     | 0.9181 | 0.8964 | 0.9372 | 0      | 0      | True    |
| Glasser      | pred_FC_resid_bvdemo | DZ       | 127     | 0.7413 | 0.6924 | 0.785  | 0      | 0      | True    |
| Glasser      | pred_FC_resid_bvdemo | sibling  | 1254    | 0.7429 | 0.7254 | 0.7607 | 0      | 0      | True    |

Showing first 24 of 48 rows. Full source: reproduction/ family_mechanism/ outputs/ family_auc.csv.


### F8 stability

| parcellation | anchor_pc | median_abs_cos | min_abs_cos | median_expl_var | median_FC_to_PC_R2 | median_AUC_MZ | median_AUC_DZ | median_AUC_sibling |
| ------------ | --------- | -------------- | ----------- | --------------- | ------------------ | ------------- | ------------- | ------------------ |
| Glasser      | 1         | 0.9849         | 0.9812      | 0.03826         | 0.5921             | 0.8059        | 0.6194        | 0.4393             |
| Glasser      | 2         | 0.9345         | 0.8329      | 0.01593         | 0.0169             | 0.64          | 0.4705        | 0.5662             |
| Glasser      | 3         | 0.8856         | 0.8253      | 0.01496         | 0.2247             | 0.7108        | 0.5993        | 0.581              |
| Glasser      | 4         | 0.8956         | 0.8483      | 0.01347         | 0.1204             | 0.6805        | 0.5994        | 0.5796             |
| Glasser      | 5         | 0.8675         | 0.7553      | 0.01185         | 0.1098             | 0.6061        | 0.5397        | 0.5629             |
| Glasser      | 6         | 0.7835         | 0.6339      | 0.01079         | 0.06598            | 0.7189        | 0.6338        | 0.5633             |
| Glasser      | 7         | 0.7122         | 0.5642      | 0.01016         | 0.09105            | 0.671         | 0.5914        | 0.5591             |
| Glasser      | 8         | 0.7558         | 0.6114      | 0.009241        | 0.03474            | 0.5946        | 0.5613        | 0.543              |
| Glasser      | 9         | 0.6278         | 0.4594      | 0.008823        | 0.03123            | 0.6229        | 0.5516        | 0.5491             |
| Glasser      | 10        | 0.5658         | 0.5056      | 0.008766        | 0.06259            | 0.6149        | 0.5454        | 0.5387             |
| 4S456Parcels | 1         | 0.983          | 0.9794      | 0.03828         | 0.5767             | 0.8132        | 0.6354        | 0.4438             |
| 4S456Parcels | 2         | 0.9662         | 0.9445      | 0.01901         | 0.07535            | 0.7326        | 0.5198        | 0.5534             |
| 4S456Parcels | 3         | 0.9418         | 0.8673      | 0.0146          | 0.02866            | 0.6587        | 0.5076        | 0.5571             |
| 4S456Parcels | 4         | 0.9141         | 0.838       | 0.01355         | 0.2361             | 0.7214        | 0.5987        | 0.5925             |
| 4S456Parcels | 5         | 0.8577         | 0.7175      | 0.01225         | 0.124              | 0.6316        | 0.5893        | 0.5477             |
| 4S456Parcels | 6         | 0.8454         | 0.7143      | 0.01136         | 0.07854            | 0.6913        | 0.6384        | 0.5654             |
| 4S456Parcels | 7         | 0.8431         | 0.6393      | 0.01045         | 0.1307             | 0.6856        | 0.5616        | 0.5779             |
| 4S456Parcels | 8         | 0.7662         | 0.4949      | 0.009417        | 0.07259            | 0.6773        | 0.5627        | 0.5385             |
| 4S456Parcels | 9         | 0.6768         | 0.4602      | 0.008707        | 0.144              | 0.7218        | 0.58          | 0.5759             |
| 4S456Parcels | 10        | 0.4513         | 0.2749      | 0.008636        | 0.0991             | 0.6476        | 0.5424        | 0.5481             |

Full source: reproduction/ family_mechanism/ outputs/ f8_stability.csv.


# Existing-Data Gap Closure


This section answers the review-style holes that can be closed from files already present in the repository. It does not claim to solve checks that require raw diffusion repeat data, new objectives, or additional cohorts; those are triaged in the next section.


## E1. Reconstruction Quality Does Not Track Cognitive Lift


| parcellation | reconstruction_metric | n   | pearson_r_with_lift | pearson_p | spearman_r_with_lift | spearman_p |
| ------------ | --------------------- | --- | ------------------- | --------- | -------------------- | ---------- |
| 4S456Parcels | demeaned_pearson      | 330 | -0.070              | 0.204     | -0.252               | 3.71e-06   |
| 4S456Parcels | avg_rank              | 330 | 0.148               | 0.007     | 0.139                | 0.011      |
| 4S456Parcels | top1_acc              | 330 | -0.058              | 0.289     | 0.007                | 0.905      |
| Glasser      | demeaned_pearson      | 330 | -0.066              | 0.233     | -0.148               | 0.007      |
| Glasser      | avg_rank              | 330 | 0.043               | 0.439     | 0.043                | 0.433      |
| Glasser      | top1_acc              | 330 | 0.131               | 0.017     | 0.148                | 0.007      |


Interpretation: the strongest reconstruction scores are not the cells that deliver cognitive gain. Across 660 joined rows, the FC->SC reconstruction metrics are near orthogonal to lift over the brain-volume/demographic baseline. This directly closes the hole that the null might be a presentation artifact of looking at the wrong reconstruction metric.



Source: joined: reproduction/ outputs/ reconstruction.csv; reproduction/ outputs/ downstream.csv.



## E2. Existing Target Sweep: What the Current Grid Already Covers


| target   | input_set       | metric       | mean_metric | mean_lift | median_perm_p | n_cells |
| -------- | --------------- | ------------ | ----------- | --------- | ------------- | ------- |
| CogCryst | obs_FC          | pearson      | 0.482       | 0.128     | 0.039         | 20      |
| CogCryst | obs_FC+bv+demo  | pearson      | 0.508       | 0.154     | 0.004         | 20      |
| CogCryst | obs_SC          | pearson      | 0.256       | -0.098    | 0.869         | 20      |
| CogCryst | pred_SC         | pearson      | 0.343       | -0.011    | 0.582         | 20      |
| CogCryst | pred_SC+bv+demo | pearson      | 0.412       | 0.058     | 0.037         | 20      |
| CogFluid | obs_FC          | pearson      | 0.324       | 0.041     | 0.333         | 20      |
| CogFluid | obs_FC+bv+demo  | pearson      | 0.351       | 0.068     | 0.138         | 20      |
| CogFluid | obs_SC          | pearson      | 0.164       | -0.119    | 0.974         | 20      |
| CogFluid | pred_SC         | pearson      | 0.252       | -0.031    | 0.861         | 20      |
| CogFluid | pred_SC+bv+demo | pearson      | 0.315       | 0.032     | 0.249         | 20      |
| CogTotal | obs_FC          | pearson      | 0.446       | 0.087     | 0.084         | 20      |
| CogTotal | obs_FC+bv+demo  | pearson      | 0.471       | 0.112     | 0.019         | 20      |
| CogTotal | obs_SC          | pearson      | 0.252       | -0.107    | 0.930         | 20      |
| CogTotal | pred_SC         | pearson      | 0.348       | -0.011    | 0.651         | 20      |
| CogTotal | pred_SC+bv+demo | pearson      | 0.403       | 0.044     | 0.123         | 20      |
| age      | obs_FC          | pearson      | 0.449       | -0.387    | 0.500         | 20      |
| age      | obs_FC+bv+demo  | pearson      | 0.788       | -0.049    | 5.00e-04      | 20      |
| age      | obs_SC          | pearson      | 0.360       | -0.476    | 0.500         | 20      |
| age      | pred_SC         | pearson      | 0.296       | -0.540    | 0.500         | 20      |
| age      | pred_SC+bv+demo | pearson      | 0.996       | 0.159     | 5.00e-04      | 20      |
| sex      | obs_FC          | balanced_acc | 0.926       | -0.010    |               | 20      |
| sex      | obs_FC+bv+demo  | balanced_acc | 0.993       | 0.056     |               | 20      |
| sex      | obs_SC          | balanced_acc | 0.919       | -0.018    |               | 20      |
| sex      | pred_SC         | balanced_acc | 0.887       | -0.050    |               | 20      |
| sex      | pred_SC+bv+demo | balanced_acc | 1.000       | 0.063     |               | 20      |


Interpretation: the current target sweep is not broad enough to claim all behavior, but it already covers three cognition composites plus age and sex controls. Observed FC plus bv+demo reliably improves cognition; predicted SC alone does not. Predicted SC plus bv+demo gives a small, target-dependent lift, strongest for crystallized cognition and age-like information. Sex is best read as a leakage/control target because bv+demo nearly solves it.



Source: Bayesian-ridge downstream grid: reproduction/ outputs/ downstream.csv.



## E3. Seed-Level Lift Bounds For the Current Cognition Null


| parcellation | input_set       | n_seed_target_cells | mean_lift | ci95_lo | ci95_hi | ci_halfwidth | frac_perm_p_lt_0.05 | median_perm_p |
| ------------ | --------------- | ------------------- | --------- | ------- | ------- | ------------ | ------------------- | ------------- |
| 4S456Parcels | pred_SC         | 30                  | -0.009    | -0.042  | 0.024   | 0.033        | 0.067               | 0.712         |
| Glasser      | pred_SC         | 30                  | -0.027    | -0.051  | -0.003  | 0.024        | 0.000               | 0.711         |
| 4S456Parcels | pred_SC+bv+demo | 30                  | 0.049     | 0.028   | 0.069   | 0.021        | 0.400               | 0.161         |
| Glasser      | pred_SC+bv+demo | 30                  | 0.041     | 0.027   | 0.054   | 0.014        | 0.333               | 0.130         |
| 4S456Parcels | obs_FC+bv+demo  | 30                  | 0.106     | 0.080   | 0.131   | 0.025        | 0.600               | 0.039         |
| Glasser      | obs_FC+bv+demo  | 30                  | 0.117     | 0.090   | 0.143   | 0.026        | 0.633               | 0.028         |


Interpretation: with the existing ten frozen splits and three cognition targets, predicted SC alone is bounded near zero or below. Predicted SC plus bv+demo has a small positive lift, but it is well below observed FC plus bv+demo. This makes the null more quantitative: the current grid can detect an observed-FC-size improvement, but predicted SC is not delivering one.



Source: seed-level downstream lift: reproduction/ outputs/ downstream.csv.



## E4. Leak Audit Summary


| verdict         | n    |
| --------------- | ---- |
| ok              | 3344 |
| EXEMPT_FLAGGED  | 1052 |
| EXPECTED_SIGNAL | 4    |


By verdict and target:


| verdict         | target | n    |
| --------------- | ------ | ---- |
| EXEMPT_FLAGGED  | age    | 416  |
| EXEMPT_FLAGGED  | sex    | 636  |
| EXPECTED_SIGNAL | sex    | 4    |
| ok              | age    | 1784 |
| ok              | sex    | 1560 |


Interpretation: the grid has zero genuine LEAK_FAIL cells. The only threshold exceedances are either expected raw-connectome sex/age signal or cells explicitly marked as containing bv+demo.



Source: leak verdict: reproduction/ outputs/ leak_verdict.csv.



## E5. Objective-Mismatch Evidence Already Present


| parcellation | variant              | auc   | auc_lo | auc_hi | p_fdr | sig_fdr |
| ------------ | -------------------- | ----- | ------ | ------ | ----- | ------- |
| 4S456Parcels | bvdemo_to_SC         | 0.565 | 0.544  | 0.586  | 0.000 | yes     |
| 4S456Parcels | combined_pred_SC     | 0.501 | 0.479  | 0.523  | 0.904 | no      |
| 4S456Parcels | obs_FC               | 0.821 | 0.805  | 0.836  | 0.000 | yes     |
| 4S456Parcels | obs_SC               | 0.884 | 0.873  | 0.896  | 0.000 | yes     |
| 4S456Parcels | pred_FC_raw          | 0.726 | 0.708  | 0.745  | 0.000 | yes     |
| 4S456Parcels | pred_FC_resid_bvdemo | 0.769 | 0.753  | 0.785  | 0.000 | yes     |
| 4S456Parcels | pred_SC_raw          | 0.670 | 0.650  | 0.690  | 0.000 | yes     |
| 4S456Parcels | pred_SC_resid_bvdemo | 0.816 | 0.801  | 0.831  | 0.000 | yes     |
| Glasser      | bvdemo_to_SC         | 0.563 | 0.541  | 0.585  | 0.000 | yes     |
| Glasser      | combined_pred_SC     | 0.505 | 0.482  | 0.526  | 0.648 | no      |
| Glasser      | obs_FC               | 0.823 | 0.807  | 0.839  | 0.000 | yes     |
| Glasser      | obs_SC               | 0.863 | 0.850  | 0.876  | 0.000 | yes     |
| Glasser      | pred_FC_raw          | 0.710 | 0.691  | 0.730  | 0.000 | yes     |
| Glasser      | pred_FC_resid_bvdemo | 0.743 | 0.725  | 0.761  | 0.000 | yes     |
| Glasser      | pred_SC_raw          | 0.680 | 0.660  | 0.700  | 0.000 | yes     |
| Glasser      | pred_SC_resid_bvdemo | 0.810 | 0.795  | 0.825  | 0.000 | yes     |


Interpretation: predicted connectomes can preserve family/identity signal when the objective or representation selects for it. The same artifact does not automatically translate into cognition gain. This strengthens the paper's objective-mismatch claim: the model is not universally incapable; it is carrying the wrong signal for the downstream cognition task.



Source: family mechanism: reproduction/ family_mechanism/ outputs/ family_auc.csv.



## E6. Per-Subject FC Reliability Is Not Driving the SC->FC Result


| source  | n   | pearson_achieved_vs_ceiling | pearson_p | spearman_achieved_vs_ceiling | spearman_p |
| ------- | --- | --------------------------- | --------- | ---------------------------- | ---------- |
| SC      | 857 | 0.012                       | 0.722     | -9.67e-04                    | 0.977      |
| bv+demo | 857 | 0.147                       | 1.66e-05  | 0.147                        | 1.66e-05   |


Interpretation: the SC achieved-vs-ceiling correlation is essentially flat, while the bv+demo baseline detects a reliability association. This is an internal positive control: the analysis can see reliability structure when it exists, but SC prediction is not explained by per-subject FC reliability.



Source: per-subject reliability: notebooks-FC_to_SC-experimental/ sanity_checks/ noise_sanity_check/ outputs/ h_correlations.csv; notebooks-FC_to_SC-experimental/ sanity_checks/ noise_sanity_check/ outputs/ h_per_subject_achieved_vs_ceiling.csv.



# New Data and Experiment Triage


The existing data close several presentation-level holes, but not every scientific hole. The remaining checks below require new model runs, saved prediction matrices, raw diffusion perturbations, or an external cohort. A standalone copy of this triage is written to `sanity_check_gap_plan.md`.

# Sanity-Check Gap Plan


This file separates checks already closed from existing repository outputs from checks that require new data, saved predictions, or new model runs.


## Closed With Existing Data


- Reconstruction quality is decoupled from cognitive lift; see `supplement_sanity_checks.md`, section `Existing-Data Gap Closure`.

- The current downstream grid covers three cognition composites plus age/sex controls.

- Seed-level lift bounds show predicted SC alone is near-zero or negative for cognition, while observed FC plus bv+demo is detectably positive.

- Leak verdicts show zero genuine `LEAK_FAIL` cells.

- Family AUC shows predicted connectomes can preserve identity/family signal when the representation selects for it.

- Per-subject FC reliability does not explain SC->FC achieved performance.


## Requires New Data Or New Runs


- **SC-side reliability/noise**
  - Needed: Repeat dMRI, split-half tractography, bootstrap streamlines, or saved tractography perturbations.
  - Feasibility: medium if raw diffusion/tractography pipeline is available; low from summary CSVs alone.
  - Why it matters: Current repository bounds FC noise well but cannot produce a true SC test-retest ceiling.

- **Objective interpolation**
  - Needed: New training grid mixing reconstruction, fingerprint/family, and cognition objectives on frozen splits.
  - Feasibility: medium-high computationally; no new cohort needed.
  - Why it matters: Would turn the objective-mismatch argument from inferential to causal.

- **Broader phenotype families**
  - Needed: Additional HCP behavioral, personality, motor, emotion, and latent phenotype targets.
  - Feasibility: high if phenotypes are already local; medium if target cleaning is needed.
  - Why it matters: Current grid covers three cognition composites plus age/sex controls, not all behavior.

- **Predicted-SC calibration/topology**
  - Needed: Saved predicted matrices or regenerated predictions; compare degree, strength, sparsity, modularity, hubs.
  - Feasibility: medium if predictions are cached; medium-low if all predictions must be regenerated.
  - Why it matters: Correlation can look acceptable while graph topology is biologically distorted.

- **Edge-class stratification**
  - Needed: Per-edge predictions plus distance, network labels, reliability, and streamline-strength bins.
  - Feasibility: medium with saved matrices and atlas metadata.
  - Why it matters: Could show whether useful signal is concentrated in short/long, intra/inter-network, or high-reliability edges.

- **Split stress tests**
  - Needed: Rerun selected cells under random, family-aware, age/sex-balanced, high-motion-excluded, and low-motion-only splits.
  - Feasibility: medium; compute-heavy but no new data.
  - Why it matters: Would make the split-drift/leakage defense harder to attack.

- **External validity**
  - Needed: Second cohort with FC, dMRI-derived SC, demographics, and comparable cognition/behavior targets.
  - Feasibility: low-medium depending on access.
  - Why it matters: HCP-YA-only evidence supports an internal claim, not a universal population claim.

- **Full hyperparameter leakage audit**
  - Needed: Static/code audit plus small rerun proving PCA, scaling, residualization, and target transforms fit inside folds.
  - Feasibility: high for code audit; medium for rerun.
  - Why it matters: Leak verdicts cover output behavior; this would document every transform boundary.



# Detailed Findings Notes


The following subsections preserve the detailed findings notes that motivated and defended the manuscript claims. Minor Unicode normalization is applied for stable LaTeX compilation; source paths are listed in the manifest.



## Reproduction Grid Findings


### Reproduction Grid -- Findings

Generated by summarize.py. Metric reporting order: reconstruction leads with demeaned_r/avg_rank; downstream with lift_over_bvdemo + paired-permutation p. **Inspect the 4S456 F1-F5 cells first** (the genuinely-new cross-parcellation evidence).

#### Reconstruction (demeaned_pearson, mean+/-std over seeds)

##### 4S456Parcels (estimator=pca_pls)

| input -> target | demeaned_r | avg_rank | top1 |
|---|---|---|---|
| FC -> SC | 0.146+/-0.003 | 0.897+/-0.010 | 0.151+/-0.017 |
| SC -> FC | 0.087+/-0.006 | 0.730+/-0.017 | 0.049+/-0.013 |
| bv -> SC | 0.185+/-0.005 | 0.903+/-0.006 | 0.129+/-0.012 |
| bv -> FC | 0.045+/-0.005 | 0.625+/-0.013 | 0.010+/-0.009 |
| demo -> SC | 0.133+/-0.009 | 0.764+/-0.028 | 0.023+/-0.008 |
| demo -> FC | 0.105+/-0.009 | 0.677+/-0.020 | 0.013+/-0.010 |
| bv+demo -> SC | 0.191+/-0.007 | 0.908+/-0.011 | 0.173+/-0.024 |
| bv+demo -> FC | 0.094+/-0.006 | 0.687+/-0.017 | 0.024+/-0.013 |
| FC+bv+demo -> SC | 0.184+/-0.005 | 0.938+/-0.009 | 0.232+/-0.016 |
| SC+bv+demo -> FC | 0.106+/-0.007 | 0.758+/-0.018 | 0.055+/-0.018 |
| FC -> FC | 0.418+/-0.034 | 1.000+/-0.000 | 0.997+/-0.004 |
| SC -> SC | 0.386+/-0.030 | 1.000+/-0.000 | 1.000+/-0.000 |

**Asymmetry FC->SC / SC->FC = 1.68x** (FC->SC=0.146, SC->FC=0.087).

**Ceiling B (FC->FC oracle):** bayesian_ridge=0.632, kernel_ridge=0.631, pca_pls=0.418

##### Glasser (estimator=pca_pls)

| input -> target | demeaned_r | avg_rank | top1 |
|---|---|---|---|
| FC -> SC | 0.136+/-0.007 | 0.875+/-0.015 | 0.120+/-0.022 |
| SC -> FC | 0.084+/-0.008 | 0.714+/-0.019 | 0.049+/-0.011 |
| bv -> SC | 0.162+/-0.007 | 0.864+/-0.013 | 0.086+/-0.014 |
| bv -> FC | 0.049+/-0.005 | 0.623+/-0.009 | 0.012+/-0.008 |
| demo -> SC | 0.130+/-0.009 | 0.754+/-0.027 | 0.023+/-0.014 |
| demo -> FC | 0.111+/-0.010 | 0.671+/-0.021 | 0.013+/-0.006 |
| bv+demo -> SC | 0.169+/-0.009 | 0.875+/-0.015 | 0.129+/-0.015 |
| bv+demo -> FC | 0.100+/-0.008 | 0.685+/-0.019 | 0.024+/-0.006 |
| FC+bv+demo -> SC | 0.171+/-0.006 | 0.914+/-0.014 | 0.169+/-0.019 |
| SC+bv+demo -> FC | 0.105+/-0.007 | 0.743+/-0.022 | 0.054+/-0.015 |
| FC -> FC | 0.451+/-0.026 | 1.000+/-0.000 | 0.999+/-0.002 |
| SC -> SC | 0.376+/-0.018 | 1.000+/-0.000 | 1.000+/-0.000 |

**Asymmetry FC->SC / SC->FC = 1.61x** (FC->SC=0.136, SC->FC=0.084).

**Ceiling B (FC->FC oracle):** bayesian_ridge=0.673, kernel_ridge=0.672, pca_pls=0.451

#### Downstream cognition (lift over bv+demo, mean+/-std; median paired-perm p)

##### 4S456Parcels (estimator=bayesian_ridge)

| input | target | pearson | lift_over_bvdemo | median perm p | residualized_r |
|---|---|---|---|---|---|
| bv+demo | CogTotal | 0.359+/-0.086 | 0.000+/-0.000 | 1 | 0.022+/-0.072 |
| bv+demo | CogFluid | 0.283+/-0.072 | 0.000+/-0.000 | 1 | -0.003+/-0.111 |
| bv+demo | CogCryst | 0.354+/-0.060 | 0.000+/-0.000 | 1 | 0.054+/-0.091 |
| obs_FC | CogTotal | 0.445+/-0.073 | 0.086+/-0.055 | 0.0905 | 0.289+/-0.076 |
| obs_FC | CogFluid | 0.321+/-0.079 | 0.037+/-0.035 | 0.347 | 0.204+/-0.072 |
| obs_FC | CogCryst | 0.476+/-0.072 | 0.122+/-0.087 | 0.0407 | 0.352+/-0.087 |
| obs_SC | CogTotal | 0.256+/-0.076 | -0.103+/-0.055 | 0.927 | 0.057+/-0.067 |
| obs_SC | CogFluid | 0.165+/-0.108 | -0.118+/-0.055 | 0.971 | 0.030+/-0.068 |
| obs_SC | CogCryst | 0.259+/-0.065 | -0.095+/-0.077 | 0.883 | 0.066+/-0.084 |
| obs_FC+obs_SC | CogTotal | 0.446+/-0.085 | 0.087+/-0.053 | 0.073 | 0.283+/-0.079 |
| obs_FC+obs_SC | CogFluid | 0.326+/-0.102 | 0.043+/-0.047 | 0.228 | 0.205+/-0.083 |
| obs_FC+obs_SC | CogCryst | 0.475+/-0.083 | 0.121+/-0.091 | 0.0335 | 0.339+/-0.097 |
| pred_SC | CogTotal | 0.354+/-0.100 | -0.005+/-0.085 | 0.612 | 0.182+/-0.106 |
| pred_SC | CogFluid | 0.273+/-0.078 | -0.010+/-0.062 | 0.843 | 0.152+/-0.064 |
| pred_SC | CogCryst | 0.342+/-0.105 | -0.012+/-0.117 | 0.629 | 0.184+/-0.126 |
| pred_FC | CogTotal | 0.271+/-0.068 | -0.088+/-0.084 | 0.871 | 0.079+/-0.087 |
| pred_FC | CogFluid | 0.211+/-0.100 | -0.072+/-0.070 | 0.906 | 0.091+/-0.079 |
| pred_FC | CogCryst | 0.245+/-0.062 | -0.109+/-0.087 | 0.896 | 0.025+/-0.089 |
| obs_FC+bv+demo | CogTotal | 0.468+/-0.078 | 0.109+/-0.047 | 0.0192 | 0.292+/-0.077 |
| obs_FC+bv+demo | CogFluid | 0.345+/-0.092 | 0.062+/-0.043 | 0.158 | 0.208+/-0.075 |
| obs_FC+bv+demo | CogCryst | 0.500+/-0.072 | 0.146+/-0.081 | 0.008 | 0.353+/-0.092 |
| obs_SC+bv+demo | CogTotal | 0.308+/-0.083 | -0.052+/-0.049 | 0.831 | 0.057+/-0.076 |
| obs_SC+bv+demo | CogFluid | 0.206+/-0.115 | -0.077+/-0.057 | 0.934 | 0.027+/-0.074 |
| obs_SC+bv+demo | CogCryst | 0.312+/-0.056 | -0.042+/-0.059 | 0.695 | 0.068+/-0.091 |
| pred_SC+bv+demo | CogTotal | 0.407+/-0.100 | 0.048+/-0.055 | 0.143 | 0.179+/-0.111 |
| pred_SC+bv+demo | CogFluid | 0.325+/-0.080 | 0.042+/-0.043 | 0.216 | 0.146+/-0.062 |
| pred_SC+bv+demo | CogCryst | 0.410+/-0.081 | 0.056+/-0.069 | 0.0477 | 0.186+/-0.132 |
| pred_FC+bv+demo | CogTotal | 0.334+/-0.067 | -0.025+/-0.061 | 0.735 | 0.082+/-0.084 |
| pred_FC+bv+demo | CogFluid | 0.260+/-0.099 | -0.024+/-0.058 | 0.765 | 0.090+/-0.079 |
| pred_FC+bv+demo | CogCryst | 0.326+/-0.052 | -0.028+/-0.057 | 0.542 | 0.028+/-0.085 |

##### Glasser (estimator=bayesian_ridge)

| input | target | pearson | lift_over_bvdemo | median perm p | residualized_r |
|---|---|---|---|---|---|
| bv+demo | CogTotal | 0.359+/-0.086 | 0.000+/-0.000 | 1 | 0.022+/-0.072 |
| bv+demo | CogFluid | 0.283+/-0.072 | 0.000+/-0.000 | 1 | -0.003+/-0.111 |
| bv+demo | CogCryst | 0.354+/-0.060 | 0.000+/-0.000 | 1 | 0.054+/-0.091 |
| obs_FC | CogTotal | 0.447+/-0.082 | 0.088+/-0.065 | 0.0802 | 0.297+/-0.088 |
| obs_FC | CogFluid | 0.327+/-0.082 | 0.044+/-0.044 | 0.326 | 0.216+/-0.075 |
| obs_FC | CogCryst | 0.487+/-0.070 | 0.133+/-0.093 | 0.0307 | 0.378+/-0.092 |
| obs_SC | CogTotal | 0.248+/-0.080 | -0.111+/-0.065 | 0.939 | 0.041+/-0.081 |
| obs_SC | CogFluid | 0.163+/-0.102 | -0.120+/-0.045 | 0.977 | 0.008+/-0.061 |
| obs_SC | CogCryst | 0.253+/-0.068 | -0.101+/-0.086 | 0.817 | 0.061+/-0.089 |
| obs_FC+obs_SC | CogTotal | 0.449+/-0.086 | 0.090+/-0.065 | 0.075 | 0.289+/-0.090 |
| obs_FC+obs_SC | CogFluid | 0.331+/-0.098 | 0.048+/-0.047 | 0.212 | 0.207+/-0.079 |
| obs_FC+obs_SC | CogCryst | 0.490+/-0.085 | 0.136+/-0.100 | 0.022 | 0.373+/-0.109 |
| pred_SC | CogTotal | 0.341+/-0.072 | -0.018+/-0.055 | 0.651 | 0.166+/-0.052 |
| pred_SC | CogFluid | 0.230+/-0.085 | -0.053+/-0.062 | 0.875 | 0.117+/-0.077 |
| pred_SC | CogCryst | 0.344+/-0.072 | -0.010+/-0.072 | 0.582 | 0.195+/-0.056 |
| pred_FC | CogTotal | 0.247+/-0.075 | -0.113+/-0.052 | 0.94 | 0.043+/-0.084 |
| pred_FC | CogFluid | 0.186+/-0.092 | -0.097+/-0.045 | 0.953 | 0.051+/-0.066 |
| pred_FC | CogCryst | 0.219+/-0.078 | -0.135+/-0.072 | 0.96 | -0.000+/-0.112 |
| obs_FC+bv+demo | CogTotal | 0.474+/-0.082 | 0.115+/-0.055 | 0.0157 | 0.303+/-0.085 |
| obs_FC+bv+demo | CogFluid | 0.356+/-0.093 | 0.073+/-0.044 | 0.0952 | 0.219+/-0.074 |
| obs_FC+bv+demo | CogCryst | 0.516+/-0.066 | 0.162+/-0.081 | 0.00225 | 0.382+/-0.092 |
| obs_SC+bv+demo | CogTotal | 0.309+/-0.091 | -0.050+/-0.053 | 0.824 | 0.043+/-0.089 |
| obs_SC+bv+demo | CogFluid | 0.217+/-0.103 | -0.066+/-0.043 | 0.937 | 0.010+/-0.062 |
| obs_SC+bv+demo | CogCryst | 0.319+/-0.059 | -0.035+/-0.065 | 0.604 | 0.061+/-0.094 |
| pred_SC+bv+demo | CogTotal | 0.399+/-0.086 | 0.040+/-0.027 | 0.123 | 0.160+/-0.045 |
| pred_SC+bv+demo | CogFluid | 0.305+/-0.084 | 0.022+/-0.038 | 0.51 | 0.111+/-0.091 |
| pred_SC+bv+demo | CogCryst | 0.415+/-0.071 | 0.061+/-0.035 | 0.037 | 0.191+/-0.062 |
| pred_FC+bv+demo | CogTotal | 0.321+/-0.078 | -0.038+/-0.042 | 0.832 | 0.048+/-0.077 |
| pred_FC+bv+demo | CogFluid | 0.245+/-0.096 | -0.038+/-0.043 | 0.866 | 0.051+/-0.065 |
| pred_FC+bv+demo | CogCryst | 0.314+/-0.069 | -0.040+/-0.044 | 0.813 | 0.004+/-0.108 |

#### Leak checks

- LEAK_FAIL (genuine: demographic-free, non-connectome input over threshold): **0**
- EXPECTED_SIGNAL (raw connectomes predict sex/age -- real biology, not a leak): 4
- EXEMPT_FLAGGED (contain bv+demo; cognition-only): 1052
- ok: 3344


## Grid Exploration Audit


### Grid Exploration -- Do the Findings Hold, and What Else Is in There?

Deep double-check of the 13,640-cell reproduction grid, **straight from the merged CSVs**
(`outputs/{reconstruction,downstream,leak_verdict}.csv`). Regenerate with
`python exploration/explore.py` (-> `digest.txt`) and `python exploration/make_figures.py`
(-> `figures/`). Everything below is computed, not remembered.

**Bottom line:** every headline claim (F1, F2, F4, F5, Ceiling B) replicates on *both*
parcellations and is stable across seeds. The interesting part is in the margins -- eight
things the summary report doesn't say, three of which actually *strengthen* the story and two
of which are methodological flags. Verdicts at the end.

---

#### Part 1 -- Do the confirmatory findings line up? (Yes.)

| Claim | Result (both parcs) | Status |
|---|---|---|
| **F1 asymmetry** FC->SC > SC->FC | 1.68x (4S456), 1.61x (Glasser), pca_pls | [pass] replicates |
| **F2 dissociation** anatomy vs demographics | clean double dissociation (see below) | [pass] replicates, **stronger than stated** |
| **F4 FC->cognition** real over baseline | obs_FC lifts CogCryst 60% of seeds, p~0.03-0.04 | [pass] replicates |
| **F5 SC underperforms / pred adds nothing** | obs_SC lift < 0; pred_FC *actively harmful* | [pass] replicates, **stronger** |
| **Ceiling B** within-modal oracle | FC->FC 0.63/0.67, SC->SC 0.62/0.65 (BR) | [pass] replicates |

All match `reports/reproduction_findings.md` to the third decimal. No cell is non-finite; the
only NaN `pearson`s are the 2,200 `sex` rows (classification -> uses `balanced_acc`, by design).

---

#### Part 2 -- Eight things worth knowing (the "anything unusual" hunt)

##### 1. The cross-modal asymmetry is NOT a reliability artifact [note] (strengthens F1)
The obvious skeptic's objection to "FC->SC beats SC->FC" is "SC is just noisier, so it's a worse
*target*." The oracle kills that: at the within-modal ceiling, **FC and SC are essentially
equally self-predictable** -- FC->FC / SC->SC = **1.02x (4S456), 1.04x (Glasser)**. Both modalities
are ~equally reliable, yet FC->SC beats SC->FC by **1.6-1.7x**. So the asymmetry is a genuine
*directional information* effect (FC contains more about SC than vice-versa), not a target-noise
artifact. This is a defensible, reviewer-proof framing of F1.

##### 2. Clean double dissociation: anatomy->structure, demographics->function [note] (strengthens F2)
Not just "bv predicts SC." It's a crossed dissociation, on both parcellations
(`figures/f2_dissociation.png`):

| | ->SC (structure) | ->FC (function) |
|---|---|---|
| **bv** (anatomy proxies) | **0.185 / 0.162** (wins) | 0.045 / 0.049 |
| **demo** (demographics) | 0.133 / 0.130 | **0.105 / 0.111** (wins) |
| ratio bv/demo | 1.39 / 1.24 | **0.43 / 0.44** |

Anatomy is the better predictor *of structure*; demographics is >2x better *of function*. A clean
2x2 crossover is a much stronger statement than a single main effect.

##### 3. pred_FC is *actively harmful* downstream, not merely useless [note] (sharpens F5)
`figures/f3_downstream_lift.png`. The imputed connectomes don't just fail to help cognition --
**pred_FC drives prediction below the bv+demo baseline by -0.11 to -0.135** (the most negative
cell in the whole downstream grid, both parcs). pred_SC is roughly neutral (~0). So "imputation
doesn't transfer to cognition" understates it: imputed FC injects structured noise that *beats the
baseline down*. obs_SC is also net-negative for cognition. Only **obs_FC** (and obs_FC+bv+demo)
lifts cognition.

##### 4. The kernel_ridge 3x3 sweep is degenerate -- 9 variants ~ 1 (compute flag)
Across the entire 9-cell gammaxalpha grid, the within-seed max-min spread of `demeaned_pearson`
is **0.004-0.005** -- i.e. the HPs do essentially nothing (`digest.txt section 2`). 9 of the 11
estimator-variants carry near-identical information. **Implication:** a future re-run could collapse
KR to a single HP and shed ~2/3 of the estimator axis (and a big chunk of the downstream blow-up the
PCA-cache note targets) with no loss of signal. Worth noting in the methods as "HP-insensitive."

##### 5. Estimator choice moves every headline magnitude (reporting flag)
The report mixes estimators: F1 cross-modal uses **pca_pls**, Ceiling B uses **bayesian_ridge**.
But BR is the stronger estimator *everywhere*:
- Oracle: BR FC->FC = 0.63-0.67 vs **pca_pls only 0.42-0.45** -- PLS reaches ~65% of BR's ceiling.
- Cross-modal: BR FC->SC = 0.171/0.166 vs pca_pls 0.146/0.136 -- **BR is ~17% higher.**

The asymmetry *ratio* is estimator-robust (good), but the **absolute headline numbers depend on
which estimator you quote.** Recommend stating the estimator next to every number, and consider
leading F1 with BR (the best estimator) rather than PLS, for consistency with Ceiling B.
Downstream is the same story inverted: **pca_pls gives ~1.5x larger lifts** than BR
(obs_FC->CogCryst: pca ~ 0.18-0.20 vs BR ~ 0.12-0.13) -- but BR is the conservative choice and is
correctly what the report leads with.

##### 6. The finer parcellation specifically boosts structure prediction (structured, not noise)
4S456 vs Glasser is *not* a wash -- the difference is signed and consistent (`figures/f4_cross_parcellation.png`):
**every ->SC pair is higher on 4S456 (+8 to +14.5%), every ->FC pair is slightly lower (-5 to -7%).**
The +60% finer parcellation adds resolvable structural detail (bv->SC +14.5%, bv+demo->SC +12.9%) but
marginally dilutes functional prediction. This is the genuinely-new cross-parcellation evidence and
it has a clean interpretation, not just "numbers track."

##### 7. Cross-modal prediction captures only ~20-24% of the achievable ceiling (honest framing)
FC->SC (pca_pls) / SC->SC oracle (BR) = **23.6% (4S456), 21.0% (Glasser)** (`figures/f5_ceiling_gap.png`).
The cross-modal map is real but recovers only ~1/5-1/4 of what's in principle recoverable about the
target connectome. Good honest ceiling language for the paper: "well above chance, far below the
within-modal oracle."

##### 8. Only the *combined* FC+SC crosses the sex-leak threshold (mild curiosity)
All 4 EXPECTED_SIGNAL rows are **obs_FC+obs_SC -> sex** at score 0.991-0.995 (threshold 0.99).
Neither FC nor SC alone crosses; combining the two modalities pushes sex-decodability just over the
line. Real biology (sex is strongly encoded in connectomes), correctly *not* flagged as a leak -- but
a nice illustration that FC and SC carry partly *complementary* demographic information. 0 LEAK_FAIL.

---

#### Part 3 -- Smaller observations
- **CogCryst is the only reliably FC-predictable cognition.** CogCryst lifts hit 60-80% of seeds
  significant; CogFluid **never** reaches significance from any connectome input (0% seeds for obs_FC).
  Crystallized > fluid intelligence in connectome-predictability -- consistent with the literature.
- **Identifiability ladder** (top1 fingerprint accuracy, pca_pls): FC->FC oracle ~ **0.997-0.999**
  (near-perfect), FC->SC cross-modal only 0.12-0.15, SC->FC worst (0.049). Adding bv+demo *raises*
  identifiability (FC+bv+demo->SC top1 0.17-0.23 > FC->SC alone) -- subject-info sharpens the fingerprint.
- **Connectome adds ~nothing over bv+demo for reconstruction.** FC+bv+demo->SC vs bv+demo->SC is
  -0.007 (4S456) / +0.002 (Glasser): once you have anatomy+demographics, the *opposite* connectome is
  redundant for predicting SC. SC+bv+demo->FC adds a little more (+0.005 to +0.012). Consistent with 2..
- **Seed stability is excellent.** Headline CVs: FC->SC just **1.8% (4S456) / 5.0% (Glasser)**; the
  noisiest headline is SC->FC at ~7-9%. Nothing is seed-fragile.
- **age "prediction" from bv+demo (0.83-0.84) is near-trivial** -- age is *in* demo. The real signal is
  connectomes predicting age much more weakly (obs_FC ~ 0.43-0.47), and pred_* degrade it (0.25-0.32).

---

#### Part 4 -- Verdicts

**Confirmatory:** [pass] All of F1, F2, F4, F5, Ceiling B replicate on both parcellations, stable across
10 seeds, no data-integrity issues. The grid is sound.

**Net-new / actionable, in priority order:**
1. **Use 1., 2., 3. in the paper -- they strengthen the story** (asymmetry isn't a reliability artifact;
   F2 is a true double dissociation; pred_FC is harmful, not just useless).
2. **Reporting hygiene (5.):** state the estimator next to every headline number; the absolute values
   differ by ~17% (recon) to ~1.5x (downstream) by estimator even though directions are robust.
3. **Compute (4.):** the KR 9-HP sweep is degenerate -- collapse to 1 HP on any re-run.
4. **Framing (6., 7.):** the 4S456->SC boost is a real interpretable effect; lead the ceiling discussion
   with "~1/5-1/4 of the within-modal oracle."

**Nothing alarming surfaced** -- no leak failures, no implausible cells, no seed-fragile headline, no
sign flips across parcellations.


## Estimator Discrepancy Resolution


### Resolving the Two Flagged Discrepancies (oracle 0.647 vs 0.38; cognition 0.359 vs 0.129)

An analyst comparing the grid CSVs against the older PDF flagged two worrying gaps. **Both have the
same root cause and neither indicates a problem with the grid:** the analyst read the `pca_pls` rows;
the PDF/report numbers are `bayesian_ridge`. Hold the estimator fixed and everything matches to the
third decimal. Verified empirically below (`python exploration/explore.py` reproduces it).

---

#### Discrepancy 1 -- the oracle ceiling (the one that worried you most)

**Claim:** "SC->SC oracle is 0.38 here vs 0.647 in the PDF -- maybe 0.647 was raw pearson or a
different oracle construction; the disattenuation denominator may be wrong."

**Resolution: it's the same metric (demeaned-r), same pipeline -- just a different estimator.**

| SC->SC oracle (demeaned_pearson) | pca_pls | bayesian_ridge | kernel_ridge |
|---|---|---|---|
| **Glasser** | 0.376 | **0.648** | 0.645 |
| **4S456** | 0.386 | 0.622 | 0.618 |

The PDF's **0.647 ~ bayesian_ridge SC->SC = 0.648 (Glasser)** -- a 0.001 match. The 0.38 the analyst
saw is the `pca_pls` row. It is **not** raw pearson: raw pearson for SC->SC is **0.92-0.95** (and r^2 is
~0.05-0.10). So three candidate values per cell:
- demeaned-r (the PRIMARY metric): pca_pls 0.38 / **BR 0.648** / KR 0.645
- raw pearson (population-mean-dominated, not reportable): ~0.95
- r^2: ~0.05-0.10

**Conclusion: the ceiling did not change. The disattenuation denominator is fine -- 0.647 was the BR
oracle all along.** The only fix needed is reporting hygiene: the PDF quotes F1 cross-modal with
`pca_pls` but the oracle with `bayesian_ridge`. **Quote one estimator throughout.** Since BR is the
strongest estimator (its oracle is ~50% higher than PLS's, because PLS only reaches ~65% of BR's
ceiling), the cleanest fix is to lead *everything* in BR:

| BR, demeaned-r | Glasser | 4S456 |
|---|---|---|
| FC->SC (cross-modal) | 0.166 | 0.171 |
| SC->FC (cross-modal) | 0.104 | 0.101 |
| FC->FC oracle | 0.673 | 0.632 |
| SC->SC oracle | 0.648 | 0.622 |

The asymmetry **ratio** is estimator-robust (~1.6x in both PLS and BR), so F1 is unaffected; only the
absolute headline numbers move with the estimator.

---

#### Discrepancy 2 -- cognition baseline dropped (bv+demo 0.359 -> 0.129)

**Claim:** "bv+demo CogTotal is 0.129 here vs 0.359 in the PDF; obs_FC CogCryst 0.439 vs 0.434. The
baseline dropped a lot, which widens FC's lift, but perm-p still isn't significant, so variance must
be higher -- why did bv+demo cognition drop?"

**Resolution: same estimator confusion. The analyst's numbers are `pca_pls`, exactly:**

| Glasser, pearson | pca_pls | bayesian_ridge (= PDF) |
|---|---|---|
| bv+demo -> CogTotal | **0.129** <- analyst | **0.359** <- PDF |
| obs_FC -> CogCryst | **0.439** <- analyst | 0.487 |

0.129 and 0.439 are the `pca_pls` Glasser values to the digit. The baseline didn't "drop" -- the
analyst is on a different (and worse) estimator.

##### Bonus finding: this *validates* the report's use of `bayesian_ridge` for downstream

Digging in revealed why `pca_pls` should never be quoted for the scalar (cognition/age) targets:

1. **Catastrophic r^2.** pca_pls scalar regression has r^2 = **-50 (4S456) to -540 (Glasser)** on the
   bv+demo baseline; kernel_ridge is similar (-1 to -7). Only **bayesian_ridge has positive r^2
   (+0.115)**. The PLS/KR scalar predictors are wildly overfit/ill-conditioned for low-signal scalar
   targets.

2. **Not numerically reproducible.** The bv+demo input is subject-info -- *parcellation-independent* --
   so the baseline should be **bit-identical across parcellations** for the same seed. It is, for
   bayesian_ridge (max |Delta| across seeds = **0.0000**). But for pca_pls, 2 of 10 seeds diverge wildly
   on identical input (seed 3: -0.045 vs +0.355; seed 6: -0.046 vs +0.305). The ill-conditioning is
   so severe that run-to-run numerical noise (different cluster nodes -> different BLAS reductions)
   flips the result. The apparent "parcellation difference" is numerical noise, not signal.

**Takeaway:** the report correctly leads downstream with `bayesian_ridge` -- it is the *only* stable,
positive-r^2, parcellation-consistent scalar estimator. pca_pls/kernel_ridge downstream rows exist in
the grid for completeness but **must not be quoted**. (For reconstruction, all three are stable; the
estimator choice there only shifts magnitude, not validity.)

---

#### One-line answers

- **Oracle:** nothing changed; 0.647 = the BR oracle (demeaned-r 0.648), the analyst read the PLS row.
  Disattenuation denominator is correct. Just quote one estimator throughout (recommend BR).
- **Cognition baseline:** nothing dropped; 0.129 = the PLS row, the PDF's 0.359 = BR. And PLS/KR
  scalar regression is ill-conditioned (r^2 << 0, non-reproducible) -> only BR is reportable downstream.
- **Action:** add an explicit "estimator = bayesian_ridge" label to every headline cognition/oracle
  number, and a note that PLS/KR downstream rows are diagnostic-only. (This is exactly flag 5. in
  `FINDINGS_EXPLORATION.md`.)


## Reduction-Axis Robustness


### Reduction-axis robustness for FC<->SC asymmetry -- findings

**Run**: SLURM job 10116852 (resubmit after 10086341 hit walltime and 10085734
OOMed on PLS's `coef_`). 16 CPU / 120 GB / 4h walltime. All 10 seeds x 2
directions x 5 method-variants = 100 fits.

#### TL;DR

**Verdict: CLEAN, with mild magnitude variation.** All five reduction strategies
-- full PLS on raw 64,620 edges, learned PCA(256), and three Johnson-Lindenstrauss
variants (Gaussian dense, sparse-auto, sparse-1/3) -- show FC->SC > SC->FC at
median ratio > 1.4x with Wilcoxon `p <= 0.001` against the null ratio of 1.0
across 10 seeds. The FC<->SC asymmetry is **a property of the data, not the
reduction pipeline**.

| Method (n=10 seeds) | median FC->SC dp | median SC->FC dp | **median ratio** | min ratio | max ratio | p_vs_1 |
|---|---|---|---|---|---|---|
| **PCA->PLS->PCA** (main model) | 0.1355 | 0.0847 | **1.621x** | 1.32x | 1.86x | 0.001 |
| **FULL PLS** (no reduction, 64,620-dim) | 0.1382 | 0.0762 | **1.813x** | 1.49x | 2.11x | 0.001 |
| **JL Gaussian dense** | 0.0837 | 0.0610 | **1.397x** | 1.19x | 2.25x | 0.001 |
| **JL sparse_auto** (density ~ 1/sqrtp) | 0.0841 | 0.0592 | **1.389x** | 1.11x | 1.90x | 0.001 |
| **JL sparse_1/3** (Achlioptas) | 0.0846 | 0.0535 | **1.554x** | 1.24x | 1.78x | 0.001 |

#### What each row tells us

- **PCA->PLS->PCA at 1.59x** is the published 10-seed baseline (matches `STEP 11`
  of the main notebook to within rounding).
- **FULL PLS at 1.79x >= baseline.** No reduction at all on either side gives the
  *same direction* and a *slightly stronger* magnitude than the learned PCA
  pipeline. The PCA preprocessing isn't injecting asymmetry -- if anything, it
  modestly *attenuates* it.
- **All three JL variants at 1.39-1.55x.** A data-blind random projection of FC
  (input) gives essentially the same asymmetry as the learned PCA. The FC PCA
  basis is *not* doing anything privileged for the cross-modal prediction --
  any 256-dim linear projection captures the cross-modal signal.

Spread across methods: 1.39x (JL sparse_auto) to 1.79x (full PLS), range 0.40.
That's wider than the conservative +/-0.15x CLEAN threshold the synthesizer
script encoded, but the spread is between known-equivalent reductions and the
*direction* is unambiguous across all five.

Why JL gives lower absolute dp than PCA: random projections preserve geometry
within JL-bound tolerance but lose the variance-concentration that PCA gets for
free. The asymmetry **ratio** is what matters and that ratio is preserved.

#### Reviewer-proof sentence for the writeup

> "The FC->SC asymmetry is robust to the choice of input reduction. Across 10
> seeds, the demeaned-pearson ratio FC->SC / SC->FC was 1.59x with the main
> PCA(256)->PLS(64)->inverse-PCA pipeline, 1.79x with no reduction at all
> (PLSRegression on the raw 64,620-edge vectors), and 1.39x, 1.39x, and 1.55x
> with Johnson-Lindenstrauss random projections (Gaussian dense, sparse
> density=1/sqrtp, sparse density=1/3 respectively). All five methods reject the
> null ratio of 1.0 at Wilcoxon `p <= 0.001`."

#### Caveats

1. **One method failed twice before working.** First attempt (`10085734`)
   OOM-killed during method B because `sklearn.cross_decomposition.PLSRegression`
   materializes `coef_` of shape `(64620, 64620) = 33 GB` at fit-time. Fix:
   bypass `coef_` at predict time by manually computing
   `(X_test - x_mean)/x_std @ x_rotations_ @ y_loadings_.T * y_std + y_mean`.
   The fix is in `method_b_full_pls.py` and works at 64 GB but we kept the bump
   to 120 GB for headroom. Second attempt (`10086341`) hit the 1h30m walltime
   at seed 8 of method B; resolved by raising walltime to 4h **and** adding a
   per-seed cache (`_method_b_per_seed/seed_<n>.csv`) so any future cancellation
   resumes cleanly. Final attempt (`10116852`) completed all 100 fits in well
   under the 4h budget.
2. **Synthesizer initially hid methods A and B from the summary table.** Pandas
   read empty `jl_variant` cells as NaN; default `groupby` drops NaN keys,
   silently dropping the non-JL rows from the printed summary. Underlying CSVs
   were always complete (100 rows loaded). Patched: `df["jl_variant"].fillna("")`
   before the groupby. The numbers in the table above are the post-patch
   medians.
3. **Magnitude spread (1.39-1.79) is wider than +/-0.15x.** The synthesizer's
   CLEAN threshold was conservative; the *direction* is identical across all
   methods, but JL is a lossier projection than PCA, so the absolute prediction
   quality drops and the ratio shifts accordingly. None of this changes the
   robustness claim.
4. **JL random matrices are seed-dependent.** Each (variant, seed) gets a
   fresh JL random matrix (`random_state=seed`), so the per-seed table shows
   variance from both data splits *and* projection draws. This is the harder
   test -- using one shared JL across seeds would give tighter numbers.

#### Files

- `README.md` -- what each script does
- `method_a_pca_pls_pca.py` + `_output.txt` + `method_a_results.csv`
- `method_b_full_pls.py` + `_output.txt` + `method_b_results.csv` + `_method_b_per_seed/seed_*.csv`
- `method_c_jl_pls_pca.py` + `_output.txt` + `method_c_results.csv`
- `synthesize_reduction_axis.py` + `_output.txt` + `reduction_axis_synthesis.csv` + `reduction_axis_summary.csv`
- `run_all.sbatch` -- SLURM wrapper (cpu_short, 16 CPU, 120 GB, 4h)


## FC Noise and Reliability Sanity Check


### FC noise sanity check -- findings

**Run**: HCP-YA, SLURM jobs 11034024 (build+A+B+F) + 11034554 (E re-run after low-dim fix).
4-cell FC design (run-1/run-2 x LR/RL), both parcellations, ~1018 FC subjects
(reliability) / 957 canonical (cross-modal). All values from `outputs/*.csv`.

#### TL;DR

FC is **mostly noise at the single-edge level but highly reliable as a whole connectome**,
and SC predicts only a small fraction of even the *reproducible* FC signal -- and that
shortfall is **not explained by FC measurement noise** (proven per-subject). This is the
physical counterpart to MASTER_FINDINGS F10 (statistical saturation): the SC->FC gap is
demonstrably not an FC-noise artifact; genuine cross-modal independence is the strong
interpretation, pending the one data-blocked piece (SC's own reliability -- needs
test-retest dMRI). **CLOSED, filed as F10-supporting (MASTER A6).**

- **FC reliability ceiling** (between-session, Glasser): demeaned_r **0.49**, fingerprint
  top1 **0.93**, avg_rank **0.99**.
- **FC edge variance**: **30% trait / 4% day-to-day state / 2% within-session / 64%
  noise**; the averaged connectome we actually use has reliability **G ~ 0.59**.
- **SC->FC captures ~17%** of the reproducible FC signal (demeaned_r 0.085 of ceiling
  0.49) -- and only **~5%** of the fingerprinting ceiling.
- **Parcellation-robust** (4S456 nearly identical).
- **SC noise itself remains UNMEASURED** -- no test-retest dMRI in HCP-YA.

#### The honest bottom line on "how much is noise"

There isn't one number, and that's not a failure to find a straight answer -- the question
is genuinely **level-dependent**. The straight answer is the set:

1. **A single FC measurement is ~64% noise** (per-edge individual-difference variance).
2. **Your averaged usable connectome is ~41% noise** (reliability G ~ 0.59).
3. **The reproducible individual signal you can actually predict tops out at demeaned-r
   0.49** (the between-session ceiling), and it is **heterogeneous across people (0 to
   0.78)**.
4. **Of that reproducible 0.49, SC explains ~17%, flat across subjects** (independent of
   each subject's own reliability).

(Caveat that rides #4: this is the FC-side accounting; SC's *own* noise floor is unmeasured
-- so "SC explains 17% of reproducible FC" is exact, while "the other 83% is signal SC
doesn't contain" is the strong interpretation, pending SC test-retest.)

> ### #**REVIEW AND QUESTION** -- is "ceiling" valid per-subject, or only at the population level?
>
> *(Flagged for review -- logic below seems sound but wants a second look before it hardens
> any per-subject claim. It does NOT affect the population headline or the flatness result;
> it sharpens how we're allowed to phrase the per-subject material.)*
>
> **The disattenuation logic (Spearman 1904) is a *population* theorem**, not a per-subject
> law: the correlation between two variables is bounded by the geometric mean of their
> reliabilities **in expectation, over a population.** A single subject's "reliability"
> (correlation between their two scans) is **one noisy number from a single pair of
> measurements** -- large standard error. So is their achieved prediction. Comparing two
> noisy single-subject estimates, some will land achieved > reliability **purely by sampling
> noise** -- especially low-reliability subjects, whose reliability estimate sits near zero
> and is easy to exceed by chance. (The -0.002 subject isn't truly perfectly unpredictable;
> -0.002 is noise around some small true value, and a noisy achieved score can exceed it.)
>
> **Precise statement:** the ceiling is a **valid population bound but an invalid
> per-subject bound** -- per-subject reliability is a single noisy estimate, not a true
> per-person limit. The theorem holds on average, not pointwise; treating it pointwise
> produces the contradiction we saw (achieved > "ceiling" for some subjects).
>
> **Corrected framing (threads both concerns):**
> - **Population level -- keep "ceiling".** "Across subjects, individual FC reproduces at
>   mean demeaned-r 0.49; SC->FC captures ~17% of that." The disattenuation is legitimate
>   here and is not exceeded on average. -> stays in the main text (bottom-line #3, E).
> - **Per-subject level -- do NOT call it a ceiling.** Call it **"per-subject reliability"**
>   and describe its *distribution* (0-0.78, rho=0.41, a stable trait). Frame the H result as
>   **"SC's prediction is *uncorrelated* with subject reliability"** (a relationship between
>   two measured quantities) -- **not** "SC stays below each subject's ceiling," which is the
>   framing that breaks.
>
> **Net:** the H conclusion stands (flat r=0.01 = SC prediction is reliability-independent),
> and it's arguably *cleaner* under this framing. The thing to fix downstream: the
> per-subject `fraction_of_ceiling` column in `h_per_subject_achieved_vs_ceiling.csv`
> implicitly treats per-subject reliability as an individual bound -- report it as a
> descriptive ratio at most, and lean on the **correlation/flatness** statement (and the
> bv+demo contrast) for the actual claim, not on pointwise "fraction of ceiling."
>
> ---
>
> ### Two ceilings are different objects (and we should report both, labeled)
>
> There are **two distinct ceilings**; conflating them is the deeper source of the tension.
>
> - **Ceiling A -- data / reproducibility ceiling** (`FC_day1 <-> FC_day2`). *How reproducible
>   is the target itself?* Model-free, a property of the **data**. "Individual FC only agrees
>   with itself at **0.49**, so no predictor can exceed the reproducible signal." Valid at
>   population level, breaks per-subject (above).
> - **Ceiling B -- model / oracle ceiling** (`FC->FC`, `SC->SC` through the *same* PCA->PLS
>   pipeline). *How well can THIS model class predict the target from a perfect same-modality
>   copy?* A property of **model + data together**. "Even predicting FC from FC, PCA->PLS only
>   reaches X -- the architecture has a representational limit." (`SC->SC` oracle ~ **0.647**
>   from earlier work -- to recompute consistently in the grid; `FC->FC` = **[to compute]**.)
>
> **Ceiling B fixes both earlier complaints:**
> - **Not exceeded per-subject** -- it's a within-modality prediction run through the *same
>   pipeline / estimator / CV / metric* as the cross-modal predictions; an apples-to-apples
>   upper reference, no "two noisy scans" comparison that can flip.
> - **Available for SC** -- `SC->SC` is just a prediction task, no test-retest needed. The
>   SC-side data gap that blocks Ceiling A **disappears** for B, so both directions get a
>   consistent ceiling.
>
> **But B is a *looser* bound than A** (the "???" catch): `FC->FC` can exploit
> **session-specific signal that wouldn't replicate** in a fresh scan, so B can sit *above*
> A. B measures "max ability of the model to reproduce *this* connectome," not "max
> recoverable *individual trait*." **B cannot substitute for A** for the biological fraction.
>
> **How to report -- a two-rung reference, each labeled for what it bounds:**
> - **SC->FC achieved: 0.085.**
> - **Ceiling B (model oracle), both directions:** `FC->FC` = [to compute], `SC->SC` ~ 0.647 ->
>   "within-modality is the architecture's best case; cross-modal loses this much." Clean,
>   consistent, available both directions, never exceeded per-subject -> the **model-capacity /
>   cross-modal-loss** story.
> - **Ceiling A (data reproducibility): 0.49**, FC-side only, heterogeneous -> "of the
>   *reproducible* signal, SC gets ~17%." The **biological** fraction, with the per-subject
>   caveat + SC-side gap flagged.
>
> **Honesty sentence to carry:** *"The within-modality oracle (FC->FC) is a model-capacity
> reference and exceeds the cross-session reproducibility limit, because it can fit
> session-specific signal that does not replicate; we therefore disattenuate biological
> claims against the reproducibility ceiling (A) and use the oracle (B) only to quantify
> cross-modal vs within-modality loss."*

#### A. Reliability ceiling (native metric)

| parc | comparison | demeaned_r | pearson | top1 (fingerprint) | avg_rank |
|---|---|---|---|---|---|
| Glasser | within-session (LR<->RL) | 0.364 | 0.710 | 0.78 | 0.968 |
| Glasser | **between-session (REST1<->REST2)** | **0.491** | 0.813 | **0.933** | 0.992 |
| 4S456 | within-session | 0.337 | 0.670 | 0.78 | 0.969 |
| 4S456 | between-session | 0.457 | 0.783 | 0.936 | 0.993 |

**Note (important):** within-session (LR<->RL) is *lower* than between-session -- the opposite
of "shorter interval = more reliable." Cause = the **phase-encode distortion confound**:
LR and RL have opposite distortions, so single-direction connectomes disagree more, while
each session (LR+RL averaged) is distortion-cancelled and cleaner. So **between-session
0.49 is the valid ceiling**; the LR<->RL rung is contaminated and is NOT a clean
short-interval estimate.

#### B. Variance decomposition (G-theory 2x2; individual-difference fractions, sum to 1)

| parc | trait (signal) | state (day) | within-session | **noise** | G (avg connectome) |
|---|---|---|---|---|---|
| Glasser | **0.301** | 0.035 | 0.021 | **0.643** | 0.585 |
| 4S456 | 0.261 | 0.035 | 0.021 | **0.683** | 0.519 |

At the single-edge level **~64-68% of between-subject variance is measurement noise**,
only ~26-30% is stable trait, and day-to-day state is small (~3-4%). The 4-cell average
(REST1+REST2, LR+RL) lifts reliability to G~0.52-0.59. Per-edge components saved in
`outputs/b_variance_components_{parc}.npz`.

#### F. Whole-connectome reliability

| parc | fingerprint top1 | discriminability |
|---|---|---|
| Glasser | 0.927 | 0.998 |
| 4S456 | 0.934 | 0.999 |

**The reconciliation** (the original puzzle: "but scans match across sessions"): per-edge
~64% noise (B) yet whole-connectome 93% identifiable / 0.998 discriminable. Both true --
the individual signal is **distributed**, reliable in aggregate even when each edge is
mostly noise.

#### E. Cross-modal disattenuation -- "% of reproducible FC captured"

ceiling = REST1<->REST2 (demeaned_r 0.491). Achieved = source->FC, PCA->PLS, 10-seed median.

| source | metric | achieved | ceiling | fraction of ceiling |
|---|---|---|---|---|
| **SC->FC** | demeaned_r | 0.085 | 0.491 | **0.17** |
| SC->FC | top1 (fingerprint) | 0.051 | 0.933 | **0.05** |
| SC->FC | avg_rank | 0.713 | 0.992 | 0.72 |
| bv+demo->FC | demeaned_r | 0.098 | 0.491 | 0.20 |
| bv+demo->FC | top1 | 0.026 | 0.933 | 0.03 |

Reads:
- **SC captures only ~17% of the reproducible FC signal** (demeaned_r) and **~5% of the
  fingerprinting ceiling** -- so SC->FC is far from the reliability ceiling: the gap is
  genuine cross-modal independence, not just FC noise. (If SC->FC were near the ceiling we'd
  blame noise; it isn't, so most of the unexplained FC is reliable-but-structurally-
  unpredictable.)
- **bv+demo->FC (0.20) >= SC->FC (0.17)** even after disattenuation -- consistent with the
  project's baseline finding: cheap subject confounds match/beat the structural connectome.
- `pearson` fraction >1.0 is an artifact (raw pearson is dominated by the shared population
  mean and is uninformative here -- use demeaned_r); not reported as meaningful.

#### G + H. Per-subject reliability + does prediction track it?

**G -- per-subject reliability is heterogeneous and a stable trait.** The 0.49 ceiling is a
mean; per-subject between-session reliability spans ~0 to 0.78 (Glasser: mean 0.489,
median 0.494, std **0.117**, skew **-0.58**, non-normal p=5e-15), with a low tail (~2% of
subjects < 0.2; one at -0.002 = pure noise). It's a stable subject property
(within-vs-between per-subject rho=0.41) -- consistent with motion/compliance being
person-level. (`outputs/g_per_subject_*`, `g_reliability_hist.png`.)

**H -- SC->FC is NOT noise-limited per subject; the gap is uniform independence.**
- A subject's SC->FC prediction quality is **uncorrelated** with their own FC reliability
  ceiling: Pearson **r=0.01 (p=0.72)**, Spearman ~0. (bv+demo->FC weakly tracks it,
  r=0.15, p=2e-5.) So cleaner-FC subjects are **not** more predictable from SC.
- **Reliability-filtering does not sharpen SC->FC** -- it makes the fraction *worse*:
  | filter | n | SC achieved | SC ceiling | SC fraction |
  |---|---|---|---|---|
  | all | 857 | 0.083 | 0.491 | **0.169** |
  | drop rel<0.2 | 841 | 0.083 | 0.498 | 0.167 |
  | drop bottom 10% | 771 | 0.083 | 0.517 | 0.160 |
  | keep top 50% | 429 | 0.080 | 0.582 | **0.137** |
  Keeping only high-reliability subjects raises the ceiling (0.49->0.58) but SC's achieved
  stays flat (~0.08), so the fraction drops. The noisy tail was never the bottleneck.
- (bv+demo->FC holds ~0.20-0.21 across all filters.)
- **Why flat r=0.01 is the strong outcome (not a weak/null one):** per-subject achieved
  prediction is *mechanically* bounded by reliability (you can't predict noise), so the
  default expectation was a **positive** slope by construction. We got flat. SC captures a
  fixed ~0.08 whether a subject's FC is reliable (0.78) or near-noise (0.0) -- which *rules
  out* the boring ceiling-effect explanation. The filtering result corroborates by going
  the "wrong" way: dropping unreliable subjects *lowers* the captured fraction
  (0.169->0.137) because the ceiling rises while achieved stays pinned. Ceiling moves,
  achieved doesn't = SC has a fixed, modest grip on FC unrelated to FC's measurement quality.
- **The bv+demo contrast is what makes it a clean dissociation (not an artifact):** the
  analysis *does* detect a real reliability effect when one exists -- bv+demo->FC mildly
  tracks reliability (r=0.15). That SC's slope is flat while the baseline's isn't means the
  flatness is not a methodological artifact (it would have hit both).
- **Conclusion (with the caveat riding it):** the airtight claim is **the SC->FC gap is not
  FC-measurement-noise and not per-subject reliability** -- decisively, at the per-subject
  level. The reading "SC doesn't *contain* that part of FC" is the strong **interpretation**,
  but is not fully separable from "SC contains it but measures it too noisily, *uniformly*
  across subjects" -- a uniform SC noise floor would also produce a flat line. Distinguishing
  those needs SC test-retest (data-blocked). So: 95% of the way to genuine independence; the
  last 5% is the SC-reliability hole. Reinforces E.
- (`outputs/h_per_subject_achieved_vs_ceiling.csv`, `h_reliability_filtered_summary.csv`,
  `h_correlations.csv`, `h_achieved_vs_ceiling_scatter.png`.)

#### Status: CLOSED (supporting item for F10)

This module has done its pre-grid job: it confirms the cross-modal ceiling is not an
FC-measurement-noise artifact, in our native metric, per-subject. **Filed as F10-supporting
in MASTER_FINDINGS (Appendix A6). No further per-subject reliability analysis is warranted
here** -- the finding is extracted; the discipline now is to run the reproducibility grid
this de-risked, not to chase the per-subject rabbit hole.

#### What this resolves and what stays open

- **Resolves (airtight)**: how much of FC is noise (edge-level ~64%; whole-connectome
  reliable), and that the SC->FC gap is **not FC-measurement-noise and not per-subject
  reliability** -- decisively, per-subject (flat r=0.01 vs the baseline's r=0.15).
  Sharpens F10 with a physical denominator.
- **Interpretation (strong, pending data)**: the natural reading is "SC doesn't *contain*
  that part of FC" -- but a *uniform* SC noise floor would also produce the flat line, so
  this last step is not fully separable from FC-side evidence alone.
- **Open (data-blocked)**: SC's own noise floor, and FC->SC disattenuated by SC reliability
  -- both need test-retest dMRI (not in HCP-YA). [caution] literature plug-in only until sourced.
  This is the 5% gap between "not-noise-on-the-FC-side" (proven) and "genuine independence"
  (interpretation).

#### Caveats
- Within-session rung confounded by phase-encode distortion (above).
- Reliability computed on ~1018 FC subjects; SC->FC achieved on the 957 canonical set
  (reliability is a per-subject property, stable across the subset).
- Between-session interval is ~1 day, same scanner (HCP-YA REST1/REST2) -- the optimistic
  end; a months-apart/multi-site retest would show lower reliability (more noise).
- Cognition not involved here -- this is FC<->FC reliability + SC->FC; the cognition ceiling
  is covered separately (F4/F5/F10).

#### Files
- `outputs/a_reliability_ceiling.csv`, `b_variance_decomposition.csv`,
  `b_variance_components_{Glasser,4S456Parcels}.npz`, `f_discriminability.csv`,
  `e_crossmodal_disattenuation.csv`, `noise_synthesis.csv`
- scripts: `build_fc_cells.py`, `a_`/`b_`/`e_`/`f_`, `synthesize_noise.py`,
  `_noise_common.py`, `run_all.sbatch`, `run_e_synth.sbatch`
- roadmap: `planning/roadmap/noise-sanity-check.md`


## PC Mechanism Tractography-Reliability Check


### PC3 tractography-reliability sanity check -- findings

**Run**: SLURM job 10084347 on `cpu_short`, 16 CPU / 32G, ~1 min total compute.
All inputs from Depth 1 seed-0 PC3 loadings (`sc_pc_loadings.npy`, row 2).

#### TL;DR

**Verdict: the PC3 visual/DAN localization survives reliability partialling with room to spare.**
Two of the three headline within-network enrichments are *higher* after partialling
edge strength + distance; the third (DAN-DAN) drops modestly but stays at 5.7x.
The biological reading of PC3 holds. Add the partialled numbers to the writeup as
the reviewer-proof sentence.

| Headline pair | Raw enrichment (Depth 1.1) | Residualized (this check) | Delta |
|---|---|---|---|
| **visual || visual** | 11.97x | **13.32x** | **+1.35** |
| **dorsal attention || dorsal attention** | 7.73x | **5.73x** | -2.00 |
| **dorsal attention || visual** | 4.49x | **4.99x** | +0.50 |

Reliability proxy (strength + distance) explains **41.0% of |PC3| variance**, so
the partialling is non-trivial. After removing that 41%, the residual still concentrates
on the same network pairs -- most clearly in visual cortex.

#### Per-proxy results

##### Proxy 1 -- edge strength (Spearman + R^2)

- `Spearman(|PC3|, edge_strength)` = **+0.89** (p ~ 0)
- `OLS R^2` of `|PC3| ~ edge_strength` = **0.40**
- Pearson = +0.63

Strong positive: PC3 emphatically lives in high-strength edges. **Reliability confound
on streamline density is LIVE.** Posterior visual edges happen to be high-strength
because tracking is cleanest there, so this was the expected red flag.

##### Proxy 2 -- anatomical distance (Spearman + R^2)

- `Spearman(|PC3|, edge_distance)` = **-0.50** (p ~ 0)
- `OLS R^2` of `|PC3| ~ edge_distance` = **0.13**
- Pearson = -0.36
- Distance was computed from Glasser MNI centroids (atlas CSV `mni_x/y/z`).
  Range: 3.3-160 mm, median 79 mm.

Strong negative: PC3 favors short edges. **Distance confound is also live but smaller
in magnitude than strength** (R^2 0.13 vs 0.40).

##### Decisive partialled enrichment (the load-bearing test)

OLS `|PC3| ~ strength + distance`:
- R^2 = **0.41** (strength dominates; distance adds essentially nothing beyond it)
- coef[strength] = +0.0135, coef[distance] = -1.2e-5
- residual std / signal std = **0.77** (77% of the magnitude variation is still there)

Re-ranked top-200 edges by `|residual|` and recomputed Yeo7 network enrichment:

| Net pair | enr_raw | enr_resid | n_resid |
|---|---|---|---|
| visual || visual | 11.97 | **13.32** | 59 |
| dorsal attention || dorsal attention | 7.73 | **5.73** | 20 |
| dorsal attention || visual | 4.49 | **4.99** | 40 |
| frontoparietal || frontoparietal | 2.61 | 1.63 | 5 |
| dorsal attention || frontoparietal | 1.50 | 1.20 | 8 |
| default mode || dorsal attention | 1.38 | 1.30 | 16 |
| somatosensory || somatosensory | 0.21 | 1.26 | 6 |
| default mode || default mode | 0.67 | 0.19 | 2 |

(Full table: `enrichment_residual_top200.csv`.)

**Visual-visual enrichment goes UP after partialling.** That's the key observation --
not just "survives" but "intensifies." It means reliability under-predicts how
strongly PC3 emphasizes visual cortex; once you subtract the part of |PC3|
explained by strength+distance, what's left is *even more* visual-concentrated.

DAN-DAN drops from 7.73x to 5.73x -- about a quarter of its enrichment was
attributable to reliability -- but the residual is still ~5-6x chance, well above
the >5x "clean biological" threshold. DAN-visual is essentially unchanged.

##### Retest reliability (not run)

The rigorous version of this test would use HCP test-retest data (n~45 subjects
scanned twice) to compute per-edge ICC and partial that out instead of the
strength+distance proxy. That requires standing up a retest ingestion pipeline
(~1-2 days) which is out of scope for this sanity check. The proxy is the
standard fallback in the field and tends to agree with retest reliability on
which edges are noisy. Documented in `retest_check_note_output.txt`.

#### Reviewer-proof sentence for the writeup

> "PC3's visual/DAN localization is not attributable to higher reconstruction
> reliability in posterior short connections: after partialling streamline density
> and inter-region distance from |PC3| (which together account for 41% of its
> magnitude variance), the headline within-network enrichments remain at 13.3x
> (visual-visual; *higher* than the unadjusted value), 5.7x (DAN-DAN), and 5.0x
> (DAN-visual). The visual-visual enrichment increases under partialling,
> indicating reliability under-predicts PC3's concentration on visual cortex
> rather than driving it."

#### Caveats to keep honest

1. **Proxy not retest.** Edge strength and inter-region distance correlate with
   tractography ICC but are not identical. A future ICC-based check could in
   principle reveal a confound the proxy misses.
2. **Single seed.** Ran at seed 0 (the Depth 1 PCA basis). The Depth 1.1 stability
   analysis showed PC3 is stable across seeds (median |cos|=0.89), so the result
   should generalize, but the partialling itself wasn't re-done per seed.
3. **DAN-DAN partial give-back.** The 7.7x->5.7x drop says ~25% of DAN-DAN
   enrichment was reliability-driven. Frame as "DAN-DAN remains 5.7x enriched
   after partialling," not "DAN-DAN is unaffected."
4. **The 41% R^2 is high.** Don't bury it. Strength + distance together explain
   nearly half of |PC3|'s magnitude; the surviving signal is the other half.
   This is not "reliability is irrelevant" -- it's "reliability matters, AND the
   residual is still cleanly visual/DAN."

#### Files in this directory

- `README.md` -- what each script tests and why
- `proxy1_strength.py` + `_output.txt`
- `proxy2_distance.py` + `_output.txt`
- `decisive_partialled_enrichment.py` + `_output.txt` + `enrichment_residual_top200.csv`
- `retest_check_note.py` + `_output.txt`
- `run_all.sbatch` -- SLURM wrapper (cpu_short, 16 CPU, 30 min, auto-exits)
- `_cache_*.npy` -- local scratch (gitignored)


## Richer Tractography Representation Check


### Tractography-predict -- findings

**Run**: 6 experiments in parallel (`:ro` overlay), 10 seeds each, Glasser, family-aware
splits. Jobs 10688656 (e1, 16 cpu) + 10689496-500 (e2-retest, 4 cpu under the 32-core
per-user QOS cap). All metrics reported, not just demeaned_pearson.

#### TL;DR

**Richer tractography does not help.** The named-bundle representation (`r2t`) predicts
FC *worse* than raw streamline counts, adds nothing marginal on top of counts, and
carries no downstream cognition signal above demographics. Meanwhile the FC<->SC
asymmetry is robust across all 6 metrics and *amplifies* in the bundle representation.
Downstream, **FC remains the only modality that beats the bv+demo cognition floor** --
no structural or tractography representation (count, bundle, bundle-similarity, or
synthetic-FC-from-tractography) clears that bar.

This is a clean negative result for the "parcellated counts lose information that
richer tractography would recover" hypothesis, and a strong positive for FC's unique
downstream value.

#### E1 -- predicting FC from each structural representation (median, 10 seeds)

| rep | demeaned_pearson | pearson | top1_acc | avg_rank | mse | r2 |
|---|---|---|---|---|---|---|
| **SC** (count baseline) | **0.0847** | 0.8292 | 0.0513 | 0.7128 | 0.0143 | -0.057 |
| kitchen_sink [SC||r2t||bv||demo] | 0.0942 | 0.8288 | 0.0487 | 0.7228 | 0.0143 | -0.057 |
| SC_r2t [SC||r2t] | 0.0791 | 0.8272 | 0.0359 | 0.7020 | 0.0145 | -0.064 |
| r2t (bundle) | 0.0493 | 0.8153 | 0.0205 | 0.6090 | 0.0154 | -0.127 |
| r2t_corr (bundle-similarity) | 0.0362 | 0.8231 | 0.0179 | 0.6070 | 0.0148 | -0.091 |

- **Count-SC beats bundle-r2t** for FC prediction on every metric. The bundle profile is
  a *lossier* FC predictor than raw counts.
- **kitchen_sink edges SC** (0.0942 vs 0.0847 demeaned_pearson; best avg_rank) -- but the
  gain is ~11% and comes from bv+demo, not r2t (see E3).
- Combined-rep rows use **per-block PCA** (PCA each block to its own 256-dim latent, then
  concat) to fix a scale-domination bug where the naive single-PCA on concatenated
  features was dominated entirely by r2t's magnitude, making SC/bv/demo invisible.

#### E2 -- FC<->SC asymmetry across representations, ALL 6 metrics

median FC-wins (ratio>1, or r2 as difference>0, means FC->X beats X->FC):

| rep | demeaned_pearson | pearson | top1_acc | avg_rank | mse | r2(diff) |
|---|---|---|---|---|---|---|
| **SC** | +1.62x [pass] | +1.10x [pass] | +2.56x [pass] | +1.23x [pass] | +2.59x [pass] | +0.036 [pass] |
| **r2t** | +2.26x [pass] | +1.04x [pass] | +1.50x [pass] | +1.19x [pass] | 0.00 [fail] | +0.45 [pass] |
| r2t_corr | +1.63x [pass] | 0.98x [fail] | +1.83x ~ | +1.18x [pass] | 0.85x [fail] | -6.9 [fail] |

([pass] = Wilcoxon one-sided p<0.05 that FC wins; all 10 seeds.)

- **SC asymmetry is bullet-proof: FC->SC > SC->FC on all 6 metrics, every one p=0.001.**
  This is the strongest multi-metric statement of the project's headline asymmetry.
- **r2t amplifies the asymmetry on the "shape/identifiability" metrics**
  (demeaned_pearson 2.26x vs SC's 1.62x) but FAILS on mse (ratio 0.00) -- i.e. FC->r2t and
  r2t->FC have similar raw error, the asymmetry lives in the demeaned/rank structure, not
  raw magnitude. Honest nuance: the r2t asymmetry is metric-dependent.
- r2t_corr is mixed (fails pearson/mse/r2); not a clean asymmetry carrier.

#### E3 -- marginal contribution of r2t over SC (paired Delta, 10 seeds)

- median Delta (SC_r2t - SC) demeaned_pearson = **-0.0013** (essentially zero / slightly
  negative); one-sided Wilcoxon p(greater)=0.98 (NOT an improvement), two-sided p=0.049
  (SC_r2t is marginally *worse*).
- **Verdict: count-SC is a sufficient statistic for cross-modal prediction.** Adding the
  bundle representation does not add FC-predictive signal -- it slightly hurts (extra
  latents, no new information).

#### E5 -- downstream cognition (the real metric)

Median test Pearson predicting NIH-Toolbox composites, and lift over the bv+demo floor:

| rep | CogTotal | CogFluid | CogCrystal | lift over bv+demo (Total) |
|---|---|---|---|---|
| **FC** | **0.451** | **0.342** | **0.451** | **+0.078** [pass] |
| bv+demo (floor) | 0.372 | 0.309 | 0.346 | 0 |
| SC | 0.261 | 0.180 | 0.267 | **-0.111** |
| r2t->synthFC (substitution) | 0.202 | 0.197 | 0.180 | -0.170 |
| r2t | 0.196 | 0.130 | 0.164 | -0.176 |
| SC_r2t | 0.196 | 0.135 | 0.160 | -0.177 |
| r2t_corr | 0.151 | 0.157 | 0.103 | -0.221 |

**This is the decisive result:**
- **FC is the ONLY representation that beats the bv+demo floor** (+0.078 total, +0.105
  crystallized, +0.033 fluid). After residualizing demographics, FC retains 0.37
  crystallized / 0.22 fluid; every structural rep collapses to ~0-0.08.
- **Every tractography representation falls BELOW the demographic floor** -- count-SC,
  bundle-r2t, bundle-similarity, and the combined rep all carry *less* cognition signal
  than age+sex+brain-volume alone.
- **The substitution chain fails**: `r2t -> synthetic-FC -> cognition` (raw 0.18-0.20)
  does NOT recover FC's cognition signal. Synthetic FC generated from tractography is
  not a useful cognitive biomarker -- the cognition-predictive structure of real FC is
  not reconstructable from tractography.

This extends the project's prior "FC carries non-demographic cognition signal, SC
doesn't" finding: **richer tractography does not rescue SC**, and you cannot launder
tractography into FC's cognition signal via cross-modal prediction.

#### E4 -- does SC-PC3 (the dorsal-stream backbone) exist in the bundle representation?

- r2t PC modes 1-3 are stable across seeds (median |cos| 0.99 / 0.98 / 0.95); modes 4+
  degrade.
- **SC-PC3 does NOT map cleanly to any single r2t mode** (best Spearman of subject scores
  = -0.32 with r2t-mode 10; everything else |rho|<0.23). The dorsal visual-stream / DAN
  structural backbone (Depth 1.1) is a **count-SC phenomenon**, not cleanly present in
  the named-bundle decomposition. The bundle atlas (66 tracts) doesn't carve the cortex
  finely enough to express that intra-parietal/occipital edge pattern.

#### Retest reliability (FC scan-rescan, independent cross-check)

- FC reliability adds essentially nothing beyond the strength+distance proxy: joint
  R^2(|PC3| ~ strength+distance+FC-reliability) = **41.2%** vs strength+distance-only
  41.0%.
- PC3 visual/DAN localization survives the 3-way partial: visual||visual **13.3x**,
  DAN||DAN **5.7x**, DAN||visual **5.0x** -- identical to the strength+distance result.
- Independent confirmation that the PC3 localization is not a reliability artifact.
  (Caveat unchanged: FC reliability != SC reliability; gold-standard SC ICC would need
  the HCP retest dMRI release.)

#### Bottom line for the writeup

1. **The asymmetry is real and multi-metric** -- FC->SC beats SC->FC on all 6 metrics at
   p=0.001 across 10 seeds. Strongest statement yet.
2. **Tractography richness is a dead end** -- the bundle representation predicts FC worse
   than counts, adds nothing marginal, carries no cognition signal, and doesn't contain
   the PC3 backbone. The parcellated-count "information loss" hypothesis is falsified for
   this data: counts are a sufficient statistic.
3. **FC's downstream uniqueness is confirmed and sharpened** -- FC is the only modality
   above the demographic cognition floor; synthetic-FC-from-tractography does not recover
   it. Cognition prediction is an FC story, full stop.

#### Files

- `e1_source_rep_results.csv`, `e2_asymmetry_results.csv` + `_summary.csv`,
  `e3_marginal_results.csv` + `_summary.csv`, `e4_*` (3 CSVs),
  `e5_downstream_results.csv` + `_summary.csv`
- `tractography_synthesis.csv`, `synthesize_tractography_output.txt`
- `../sanity_checks/tract_check/retest_icc_results/` (FC reliability + 3-proxy enrichment)
- scripts: `e1`-`e5`, `synthesize_tractography.py`, `_tract_setup.py`,
  `run_one.sbatch` (parameterized, `:ro` overlay for parallel runs)


## Nonlinear Model-Class Check


### Non-linear sanity check -- findings

**Run**: 2026-06-13, 3 experiments in parallel (`:ro` overlay, 4 cpu/24G each, jobs
10757352-54), 10 seeds, Glasser, family-aware splits. Estimator swapped, representation +
splits held fixed vs `tractography_predict/`. Linear baselines byte-identical to the
linear runs (shared `_tract_setup.py`).

#### TL;DR

**NULL across the board -- the tractography dead-end and the FC<->SC asymmetry are
model-class robust.** Neither HistGradientBoosting nor KernelRidge (RBF) extracts any
tractography signal that linear models missed -- not for cognition, not for FC
reconstruction, not for marginal contribution over SC. This is the strongest version of
the linear finding: tractography's failure to predict FC/cognition is not a linear-model
artifact.

Two estimator caveats matter for reading this (below): KernelRidge is the trustworthy
nonlinear probe (it preserves FC's known cognition signal); HGB underperforms at this
sample size even where signal exists.

#### N1 -- nonlinear cognition

Best lift over the bv+demo floor among tractography reps (SC, r2t, r2t_corr, SC_r2t),
per estimator (median 10 seeds; lift = Pearson - bv+demo floor with the same estimator):

| estimator | best tractography lift | rep / target | clears floor? |
|---|---|---|---|
| linear_BR | -0.079 | SC / Crystal | no |
| HGB | -0.168 | r2t_corr / Fluid | no |
| **KR** | **-0.029** | SC / Crystal | **no** |

**No tractography representation clears the demographic floor under any estimator.** The
closest is KernelRidge on SC/crystallized at -0.029 -- still below floor.

##### Estimator sanity (FC, CogTotal) -- does nonlinear preserve known signal?
FC carries real cognition signal (linear lift +0.078). A trustworthy nonlinear estimator
should keep it:

| estimator | FC pearson | floor | FC lift |
|---|---|---|---|
| linear_BR | 0.451 | 0.372 | **+0.078** [pass] |
| **KR** | 0.338 | 0.282 | **+0.056** [pass] |
| HGB | 0.254 | 0.355 | **-0.101** [fail] |

- **KernelRidge preserves FC's lift** (+0.056) -> it's a valid nonlinear probe, and its
  finding of zero tractography signal is meaningful.
- **HGB destroys even FC's signal** (-0.101) -> gradient boosting underperforms at
  n~683 with 256 PCA inputs (overfits / can't model the smooth FC->cognition map). HGB's
  "below floor for tractography" is therefore partly an estimator-weakness artifact, not
  evidence about tractography. We trust the **KR** result.

#### N2 -- nonlinear reconstruction + asymmetry

KernelRidge vs linear PLS, FC->X demeaned_pearson (median 10 seeds):

| direction | PLS | KR | Delta (KR-PLS) | nonlinear gain? |
|---|---|---|---|---|
| FC->SC | 0.1355 | 0.1334 | -0.0021 | no |
| FC->r2t | 0.1113 | 0.1133 | +0.0020 | no (< 0.02) |
| FC->r2t_corr | 0.0582 | 0.0605 | +0.0023 | no (< 0.02) |

All Delta are within +/-0.0023 -- nonlinear neither helps nor hurts reconstruction. Counts still
beat bundles under KR (0.133 vs 0.113), same as linear.

##### Asymmetry ratio (demeaned_pearson, FC-wins) holds under nonlinear

| rep | linear_PLS | KR |
|---|---|---|
| SC | 1.621x | **1.638x** |
| r2t | 2.264x | **2.267x** |
| r2t_corr | 1.634x | 1.487x |

The FC<->SC asymmetry is essentially **identical** under linear and kernel models (SC
1.62->1.64, r2t 2.26->2.27). The asymmetry is not a linear-pipeline artifact. (Full
6-metric panel per estimator in `n2_reconstruction_summary.csv`.)

#### N3 -- nonlinear marginal (r2t over SC)

Paired Delta(SC_r2t - SC) -> FC, demeaned_pearson (median 10 seeds):

| estimator | Delta | Wilcoxon p (greater) |
|---|---|---|
| linear | -0.0013 | 0.981 |
| KR | -0.0049 | 0.998 |

r2t adds **nothing** over count-SC even under a nonlinear combined model -- slightly
negative both ways. Count-SC remains a sufficient statistic; there is no conjunctive
SCxr2t interaction that a kernel picks up.

#### Verdict

All three decision rules return null:
- N1 cognition unlock (>=+0.03 over floor): **not met** (KR best -0.029).
- N2 reconstruction gain (>=+0.02 dp): **not met** (max +0.0023).
- N3 marginal gain (>=+0.02 dp, p<0.05): **not met** (KR -0.0049).

**Conclusion**: the tractography findings from `tractography_predict/` are model-class
robust. Switching from linear (PLS/BR) to nonlinear (KernelRidge RBF, the trustworthy
probe here) changes nothing -- bundles still predict FC worse than counts, still carry no
cognition signal above demographics, still add nothing over SC; and the FC<->SC asymmetry
is unchanged. The tractography information is not hiding in nonlinear structure.

#### Caveats

- **HGB unreliable at this n**: HistGradientBoosting underperformed even on FC (where
  signal demonstrably exists), so its nulls are not informative on their own. The
  load-bearing nonlinear evidence is **KernelRidge**, which preserved FC's signal and
  still found nothing in tractography.
- **KR hyperparameters fixed** (median-heuristic gamma, alpha=1.0), not CV-tuned. Margins
  to the decision thresholds are wide (best cognition lift -0.029 vs +0.03 rule), so light
  tuning won't flip the null. A gamma/alpha sweep is the natural follow-up only if a
  result had landed near threshold (none did).
- HGB-per-component reconstruction not run (256x the fits); KR multi-output was the
  reconstruction probe. Given KR showed no recon gain, HGB-recon is unlikely to differ.
- Single parcellation (Glasser), 10 family-aware seeds.

#### Files

- `n1_cognition_results.csv` / `_summary.csv` (rep x estimator x target x seed:
  pearson, spearman, r2, lift)
- `n2_reconstruction_results.csv` / `_summary.csv` (full 6-metric panel per estimator,
  both directions, + asymmetry ratios)
- `n3_marginal_results.csv` / `_summary.csv` (full panel, paired Delta per estimator)
- `nonlinear_synthesis.csv`, `nonlinear_synthesis_output.txt`
- scripts: `n1`-`n3`, `synthesize_nonlinear.py`, `_nl_common.py`, `run_one.sbatch`
- nonlinear predictors live in `../tractography_predict/_tract_setup.py`
  (`kernelridge_predict`, `kernelridge_blocks_predict`, `hgb_scalar_predict`,
  `kr_scalar_predict`)


## Residual-Boost and Multimodal Sink Check


### Residual-boost (architecture A) + multimodal sink -- findings

**Run**: 2026-06-13, jobs 10759407/408 (N4), 10759825/826 (N5), parallel `:ro`, 10 seeds,
Glasser, family-aware. Design rationale in `DESIGN_residual_learning.md`. Builds on
`findings_nonlinear.md` (plain nonlinear, also null).

#### TL;DR

The residual-learning idea -- hand the nonlinear model the linear prediction for free
(OOF, no leakage) and make its loss the improvement above that template -- was the right,
maximally-sensitive test. Result:

- **Downstream cognition: comprehensively NULL.** Not single-modality (N4cog), not
  cross-modal sink (N5cog), not residual-boosted. FC alone remains the ceiling;
  combining modalities *dilutes* it; the nonlinear cross-modal residual adds nothing.
- **Reconstruction: one real but tiny signal.** N4recon shows the residual-boost beats
  the PLS template by +0.002-0.006 demeaned_pearson, p=0.001 every direction -- a genuine
  nonlinear sliver in connectome<->connectome mapping, ~4x below the +0.02 "matters"
  threshold, and it washes out under the cross-modal sink (N5recon).

**Interpretation: the connectome->cognition relationship is linearly saturated to the
precision this dataset supports; connectome<->connectome reconstruction has a negligible
nonlinear component. The tractography dead-end survives the most sensitive nonlinear
probe we can construct.**

#### N4 -- single-modality residual-boost

##### N4cog (cognition): NULL
`final = OOF-BR template + KernelRidge(residual)`. Delta(final - template) was +/-0.008,
mixed sign -- KR residual adds ~0. Every tractography rep stayed -0.13 to -0.25 below
the bv+demo floor (unchanged from linear E5/N1). FC stayed above floor.

##### N4recon (reconstruction): TINY REAL SIGNAL
final vs template (PLS), demeaned_pearson, 10-seed Wilcoxon:

| direction | template | final | Delta | p |
|---|---|---|---|---|
| FC->SC | 0.1355 | 0.1405 | +0.0052 | 0.001 |
| SC->FC | 0.0847 | 0.0866 | +0.0023 | 0.001 |
| FC->r2t | 0.1113 | 0.1169 | +0.0058 | 0.001 |
| r2t->FC | 0.0493 | 0.0526 | +0.0041 | 0.001 |
| FC->r2t_corr | 0.0582 | 0.0623 | +0.0050 | 0.001 |
| r2t_corr->FC | 0.0362 | 0.0402 | +0.0039 | 0.001 |

Every direction improves at p=0.001 -- a genuine nonlinear residual the OOF framing
surfaces where plain nonlinear (N2) saw only mixed-sign noise. But the effect is ~+0.005,
practically negligible (counts still beat bundles; gap not closed). Identifiability
metrics flat except r2t_corr (top1 +0.013 p=0.02; avg_rank +0.008 p=0.01).

#### N5 -- multimodal sink residual-boost (cross-modal interactions)

The one architecture that can see cross-modal conjunctions: all modalities in one feature
space, nonlinear model free to cross them.

##### N5cog (cognition): NULL + combining hurts

| target | bv+demo floor | FC alone | sink_linear | sink_residual |
|---|---|---|---|---|
| CogTotal | 0.373 | **0.436** | 0.401 | 0.381 |
| CogFluid | 0.298 | **0.306** | 0.301 | 0.286 |
| CogCrystal | 0.349 | **0.452** | 0.414 | 0.415 |

- `sink_linear [FC||SC||r2t||bv||demo]` is **-0.05 to -0.08 below FC alone**: adding
  SC/tractography/demographics to FC *dilutes* cognition prediction (extra dims, no new
  signal). FC alone is the best single predictor.
- `sink_residual - sink_linear` = -0.002 to -0.020 (Wilcoxon p~0.96-0.999 in the wrong
  direction): the nonlinear cross-modal residual does not help -- slightly hurts.
- **No FCxtractography interaction exists.** The conjunctive-modulator hypothesis (the
  most plausible way tractography could matter) is empty.

##### N5recon (reconstruction): NULL
`[SC||r2t||bv||demo]->FC`: sink_residual - sink_linear = -0.0023 (p=0.935). The tiny N4recon
nonlinear sliver does not survive the multimodal sink.

#### Decision rules -- all returned null (except the negligible N4recon whisper)

| test | rule | result |
|---|---|---|
| N4cog cognition unlock | tractography clears floor by >=0.02 | NO (all below floor) |
| N4recon reconstruction gain | final - template >= 0.02 dp | NO (+0.005, though p=0.001) |
| N5cog cross-modal unlock | sink_residual - sink_linear >= 0.02, p<0.05 | NO (-0.01, p>0.95 wrong way) |
| N5cog complementarity | sink_linear - FC >= 0.02 | NO (-0.05 to -0.08, sink worse) |
| N5recon cross-modal recon | sink_residual - sink_linear >= 0.02 | NO (-0.002) |

#### Why this is the definitive version

A clean null from OOF residual-boost means `y - yhat_linear` is **noise with respect to the
source** -- not "the model couldn't find it," but "there is no learnable structure left."
The linear model already extracted everything predictable at this n. Handing the
nonlinear model the answer for free and tasking it only with the residual is the most
sensitive probe possible; it found nothing for cognition, single-modality OR cross-modal.

The one exception (N4recon +0.005, p=0.001) is the honest nuance: connectome<->connectome
reconstruction (same data type, dense edges) has a real but trivial nonlinear component;
connectome->cognition (crossing into behavior) is dead-flat linear-saturated.

#### Caveats

- KernelRidge alpha/gamma fixed (median-heuristic gamma, alpha=1.0); decision margins
  wide vs effect sizes, so tuning won't flip the nulls.
- HGB excluded here (underperformed even on FC at n~683 in N1); KR is the trustworthy
  probe and the one used throughout N4/N5.
- n~683 train: the "linear ceiling" is partly a sample-size ceiling. A much larger cohort
  could in principle surface faint nonlinearity; nothing here suggests it would be
  practically meaningful.
- Single parcellation (Glasser), 10 family-aware seeds.

#### Files

- `n4_recon_results.csv`/`_summary.csv`, `n4_cog_results.csv`/`_summary.csv`
- `n5_cog_results.csv`/`_summary.csv`, `n5_recon_results.csv`/`_summary.csv`
- `*_output.txt` per experiment; `residual_synthesis.csv`
- scripts: `n4_*`, `n5_*`, `_residual.py` (OOF template + additive-residual helpers,
  single + multimodal-block variants), `run_one.sbatch`
- design: `DESIGN_residual_learning.md`


## Sample-Size Scaling Check


### N6 -- data-scaling learning curve: model-ceiling vs data-limited

**Run**: 2026-06-14, job 10782789, 10 seeds, multimodal sink, train subsampled at
n = 100/200/400/full(~683), test fixed+full at every n. Question: does the nonlinear
gap (residual-boost `final` - linear `template`) grow with n? If yes -> signal is
data-limited (bigger cohort indicated). If flat -> structural ceiling (more data won't help).

#### TL;DR

**MODEL-CEILING / STRUCTURAL.** The nonlinear gap does not grow into positive territory
with n in either task. Cognition is flat-and-slightly-negative at every n; reconstruction
is negative at every n and merely creeps toward zero from below as data grows (an
overfitting-penalty effect, not signal). Meanwhile the *linear* performance climbs with n
in both tasks -- the learning curve works, more data helps the linear model, and
nonlinearity adds nothing at any sample size. **A bigger cohort is NOT indicated for
nonlinearity; the linear-saturation conclusion is definitive.**

#### Results

##### Cognition (CogTotal), nonlinear gap vs n
| n | median linear | median final | **gap** | Wilcoxon p(gap>0) |
|---|---|---|---|---|
| 100 | 0.302 | 0.290 | **-0.0087** | 1.00 |
| 200 | 0.327 | 0.324 | **-0.0152** | 0.99 |
| 400 | 0.394 | 0.380 | **-0.0119** | 0.999 |
| ~683 | 0.391 | 0.375 | **-0.0128** | 0.99 |

gap-vs-n Spearman **rho=+0.05, p=0.77 -> flat**. Gap is <=0 at every n (nonlinear mildly
*hurts*). Linear grows 0.30->0.39 with n. No nonlinear signal at any size.

##### Reconstruction (sink->FC), nonlinear gap vs n
| n | median linear | median final | **gap** | Wilcoxon p(gap>0) |
|---|---|---|---|---|
| 100 | 0.0412 | 0.0353 | **-0.0054** | 0.999 |
| 200 | 0.0670 | 0.0655 | **-0.0019** | 0.88 |
| 400 | 0.0940 | 0.0931 | **-0.0015** | 0.99 |
| ~683 | 0.1098 | 0.1097 | **-0.0009** | 0.98 |

gap-vs-n Spearman rho=+0.52, p=0.001 -- **but this is NOT data-limited signal.** The gap
is **negative at every n** and just rising *toward zero from below*; Wilcoxon confirms
gap <=0 everywhere (p(gap>0) = 0.88-1.0). The positive trend is the KernelRidge residual's
**overfitting penalty shrinking as data grows** -- it asymptotes *at* zero, not above it.
It never reaches the +0.02 "matters" threshold, never goes positive. Linear grows
0.04->0.11 with n.

(Note: full-train size is 682 for one seed, 683 for nine -- a family-split rounding
artifact that splits the last column into two buckets; the single-seed 682 row is noise,
ignore it. Plot: `n6_scaling_curve.png`.)

#### Why the auto-verdict was overridden

The synthesizer's first pass flagged reconstruction as "DATA-LIMITED" on `rho>0 & p<0.05`
alone. That's wrong: a negative gap creeping toward zero has a positive trend but is not
emerging signal. Fixed the rule to require `gap@max_n > 0.005` (actually positive and
meaningful), which correctly returns MODEL-CEILING. Lesson: a positive *slope* on a
*negative* gap is an overfitting penalty vanishing, not signal appearing -- always check
the gap sign/level, not just the trend.

#### Interpretation

- **Cognition is linearly saturated, structurally.** Nonlinearity doesn't help at n=100
  or n=683; the gap doesn't trend up. More subjects make the *linear* predictor better
  (0.30->0.39) but won't unlock nonlinear structure -- there is none to unlock at this
  representation/precision.
- **Reconstruction has no positive nonlinear component either**, at any n. The tiny
  nonlinear sliver seen in N4recon (+0.005 on full data, single-modality) does not appear
  in the multimodal sink and does not grow with n here; the residual penalty just
  converges to ~0.
- **The bottleneck is information, not data or model capacity.** We've now shown the
  null is robust to: model class (N1-N3), the residual-boost head-start (N4), cross-modal
  interactions (N5), AND sample size (N6). Four independent ways of asking "is there
  nonlinear/extra signal," four nulls.

#### What this closes, and the one honest caveat

Closes the "maybe a bigger model / harder tuning would find it" question: it wouldn't --
the gap doesn't grow with data, so it's not a capacity-or-sample-size ceiling, it's
structural. The defensible headline: **the connectome->cognition relationship (and the
FC<->SC reconstruction) is linearly saturated; FC is the ceiling; richer tractography adds
nothing -- across model classes, architectures, and sample sizes up to n~683.**

Caveat: we tested to n~683. We cannot rule out that n~10^4 (UK Biobank scale) surfaces a
tiny positive gap -- but our data shows convergence *to* zero, not growth *above* it, so
there is no positive evidence for it. If one wanted to chase it, the move is a bigger
cohort (and the expectation should be a very small effect), NOT a bigger model on these
data.

#### Files
- `n6_scaling_curve.py`, `synthesize_scaling.py`, `_residual.py` (n-aware block helpers)
- `n6_scaling_results.csv`, `n6_scaling_summary.csv`, `scaling_synthesis.csv`
- `n6_scaling_curve.png` (gap-vs-n, both tasks)
- `n6_scaling_output.txt`


## Family Mechanism Reproduction Notes


### Family-Structure + Mechanism Grid (F6 / F7 / F8)

Second grid pass: replicates the family-structure (F6), predictor/identifier-tradeoff (F7),
and PC-mechanism (F8) findings on **both parcellations x 10 frozen seeds**, so they stop being
Glasser-only single-run results. Faithful ports of the notebooks (canonical source):
`model_overviews/crossmodal_pca_pls_closed_form_overview.ipynb` STEP 8.1-8.3 (F6/F7) and
`further_exploration/depth1*.ipynb` (F8). The notebook is authoritative -- if a helper drifts,
fix the notebook first.

#### Status
- **F6 + F7 -- BUILT & VALIDATED.** Aggregation reproduces the notebook's `aggregate_auc.csv`
  bit-for-bit (AUC err 1e-16, perm-p err 0, bootstrap CI err 6e-8). See tests below.
- **F8 -- BUILT & VALIDATED.** Port of `depth1_spectral_mechanism.ipynb` +
  `depth1.1_pc_stability_and_confounds.ipynb` (per-PC FC->PC R^2, per-PC family AUC, sex+bv confound,
  PC1 residualization, cross-seed PC alignment, PC3 network enrichment via
  `data/atlas_info/<parc>_dseg_reformatted.csv`; network col Glasser=`community_yeo` /
  4S456=`network_label`). Localization math reproduces the notebook's saved seed-0 `sc_pc_loadings.npy`
  exactly (energy 0.630678, interhemi base 0.501393, top-50 edge region-mapping 50/50).

#### Files
- `_fm_common.py` -- shared layer. Lazy data import (`_data()`; needs torch/Torch) so the pure
  aggregation helpers run anywhere. Copies the 4 notebook-only helpers verbatim
  (`zscore_by_unrelated`, `perm_p_auc`, `bootstrap_auc`, `fdr_bh`); reuses `_setup`'s family
  helpers + predictors. `build_family_variants` / `pair_sims_for_seed` / `aggregate_family`.
- `run_f6_family.py --parc <P> --seed <S>` -- one unit: 8 variants -> per-pair sims -> per-unit npz.
  Self-checks Glasser/seed0 pair counts against the notebook (MZ=33,DZ=13,sib=125,unrel=171).
- `finalize_fm.py` -- pool all seeds per parc -> `outputs/family_auc.csv`; **regression guard**
  asserts Glasser == notebook aggregate within 1e-6.
- `run_fm_unit.sbatch` -- array 0-19 (2 parc x 10 seeds), 24G/4h/8CPU (same profile as main grid).
- `tests/` -- see below.

#### Files (F8)
- `_f8_common.py` -- PC-mechanism helpers (parc-aware atlas, per-PC AUC/R^2, cosine alignment,
  energy, localization/enrichment) + `compute_seed()` (data layer, runs on Torch).
- `run_f8_pcmech.py --parc <P> --seed <S>` -- one unit -> per-unit npz; self-checks Glasser/seed0
  against the notebook (pair counts, FC->PC R^2, confound, PC3 AUC).
- `finalize_f8.py` -- cross-seed alignment + confound + PC3 localization/enrichment + verdict ->
  `outputs/f8_{per_pc,stability,pc3_localization,pc3_enrichment_agg}.csv`; Glasser regression guard.
- `run_f8_unit.sbatch` / `finalize_f8.sbatch` / `submit_f8.sh` -- same light profile as F6/F7.

#### Tests (run locally, no torch / no connectome data)
```bash
python reproduction/family_mechanism/tests/test_aggregation_matches_notebook.py     # F6/F7 vs notebook CSV
python reproduction/family_mechanism/tests/test_helpers.py                           # F6/F7 helper correctness
python reproduction/family_mechanism/tests/test_f8_localization_matches_notebook.py  # F8 vs saved seed-0 loadings
python reproduction/family_mechanism/tests/test_f8_helpers.py                        # F8 helper correctness
```
`test_aggregation_matches_notebook.py` feeds the notebook's own per-seed `.npz` (the expensive
connectome-derived pair sims) into our aggregation and asserts we reproduce `aggregate_auc.csv`
exactly -- the strongest off-cluster "matches the notebook" check. `test_helpers.py` pins the 4
copied helpers against reference implementations (FDR vs statsmodels, AUC vs sklearn, etc.).

#### Run on Torch
One command -- submits the 20-unit array and chains finalize via `--dependency=afterok` (finalize
runs only if every unit succeeds). Nothing to poll; watch the sentinels.
```bash
cd /scratch/ans9868/Conn2Conn/reproduction/family_mechanism
bash submit_fm.sh
### progress (passive, NO squeue):
ls sentinels/DONE_fm_*.sentinel 2>/dev/null | grep -v finalize | wc -l   # /20
test -f sentinels/DONE_fm_finalize.sentinel && echo FINALIZE DONE
cat logs/fm_finalize.txt ; column -s, -t outputs/family_auc.csv | head
```
Dedicated scripts (light profile -- F6/F7 has no KR sweep / no downstream / no F8):
- `run_fm_unit.sbatch` -- array 0-19 `%10`, **16G / 8 CPU / 1 h** (peak RSS ~4-8 GB).
- `finalize_fm.sbatch` -- single task, 8G / 30 min, runs `finalize_fm.py` in-container.
- `submit_fm.sh` -- submits both with the dependency wired.

**Estimated runtime:** ~3 min/unit (Glasser) / ~5 min (4S456) -- anchored on the notebook's
`seed_*.npz` timestamps. ~15-20 min wall at `%10` (2 waves) + ~3-5 min finalize, plus SLURM queue.
1 h time limit is generous (worst unit ~5 min); zero timeout risk.

#### What F6/F7 produce
`outputs/family_auc.csv`: per (parcellation, variant, relation) AUC + bootstrap CI + perm-p + FDR.
- **F6** headline: `pred_SC_resid_bvdemo` sibling AUC (notebook Glasser = 0.810) vs `bvdemo_to_SC`
  baseline (0.563) -- family-specific wiring beyond shared anatomy.
- **F7** is read off the same table: `pred_SC_raw` separates siblings (0.680) but
  `combined_pred_SC` collapses to chance (0.505, n.s.) -- reconstruct OR identify, not both.
  (The downstream-cognition half of F7 -- `combined_pred_SC` -> cognition lift -- is the one new
  input to add to the main downstream grid; not required for the family-collapse evidence.)


# Closing Interpretation


Taken together, these checks support a narrow but strong conclusion. FC-SC translation is reproducible and directional, but reconstruction accuracy is not a sufficient proxy for downstream cognitive utility. The null is not explained by the PCA reduction axis, obvious nonlinear model capacity, richer tractography features, sample-size trends within the tested regime, FC-side measurement noise, or hard grid leakage. The constructive exception is family signal: predicted connectomes can preserve identity/family information when the objective selects for it, but that does not imply cognition transfer.


## Source Manifest

- **spine**: preprint/ supplement.md
- **reproduction_findings**: reproduction/ reports/ reproduction_findings.md
- **exploration_findings**: reproduction/ exploration/ FINDINGS_EXPLORATION.md
- **discrepancy**: reproduction/ exploration/ DISCREPANCY_RESOLUTION.md
- **family_readme**: reproduction/ family_mechanism/ README.md
- **preprocessing**: notebooks-FC_to_SC-experimental/ sanity_checks/ preprocessing_check/ findings.md
- **noise**: notebooks-FC_to_SC-experimental/ sanity_checks/ noise_sanity_check/ findings_noise.md
- **tract_check**: notebooks-FC_to_SC-experimental/ sanity_checks/ tract_check/ findings.md
- **nonlinear**: notebooks-FC_to_SC-experimental/ non-linear-sanity-check/ findings_nonlinear.md
- **residual**: notebooks-FC_to_SC-experimental/ non-linear-sanity-check/ findings_residual.md
- **scaling**: notebooks-FC_to_SC-experimental/ non-linear-sanity-check/ findings_scaling.md
- **tractography**: notebooks-FC_to_SC-experimental/ tractography_predict/ findings.md
- **Reduction-axis summary**: notebooks-FC_to_SC-experimental/ sanity_checks/ preprocessing_check/ reduction_axis_summary.csv
- **Reduction-axis all seed ratios**: notebooks-FC_to_SC-experimental/ sanity_checks/ preprocessing_check/ reduction_axis_synthesis.csv
- **FC reliability ceiling**: notebooks-FC_to_SC-experimental/ sanity_checks/ noise_sanity_check/ outputs/ a_reliability_ceiling.csv
- **FC per-subject reliability summary**: notebooks-FC_to_SC-experimental/ sanity_checks/ noise_sanity_check/ outputs/ g_per_subject_summary.csv
- **Reliability filtering summary**: notebooks-FC_to_SC-experimental/ sanity_checks/ noise_sanity_check/ outputs/ h_reliability_filtered_summary.csv
- **Noise synthesis**: notebooks-FC_to_SC-experimental/ sanity_checks/ noise_sanity_check/ outputs/ noise_synthesis.csv
- **Tractography source representation**: notebooks-FC_to_SC-experimental/ tractography_predict/ e1_source_rep_results.csv
- **Tractography asymmetry summary**: notebooks-FC_to_SC-experimental/ tractography_predict/ e2_asymmetry_summary.csv
- **Tractography marginal summary**: notebooks-FC_to_SC-experimental/ tractography_predict/ e3_marginal_summary.csv
- **Tractography downstream summary**: notebooks-FC_to_SC-experimental/ tractography_predict/ e5_downstream_summary.csv
- **Nonlinear cognition summary**: notebooks-FC_to_SC-experimental/ non-linear-sanity-check/ n1_cognition_summary.csv
- **Nonlinear reconstruction summary**: notebooks-FC_to_SC-experimental/ non-linear-sanity-check/ n2_reconstruction_summary.csv
- **Nonlinear marginal summary**: notebooks-FC_to_SC-experimental/ non-linear-sanity-check/ n3_marginal_summary.csv
- **Residual cognition summary**: notebooks-FC_to_SC-experimental/ non-linear-sanity-check/ n4_cog_summary.csv
- **Residual reconstruction summary**: notebooks-FC_to_SC-experimental/ non-linear-sanity-check/ n4_recon_summary.csv
- **Sink cognition summary**: notebooks-FC_to_SC-experimental/ non-linear-sanity-check/ n5_cog_summary.csv
- **Sink reconstruction summary**: notebooks-FC_to_SC-experimental/ non-linear-sanity-check/ n5_recon_summary.csv
- **Scaling summary**: notebooks-FC_to_SC-experimental/ non-linear-sanity-check/ n6_scaling_summary.csv
- **Family AUC**: reproduction/ family_mechanism/ outputs/ family_auc.csv
- **F8 stability**: reproduction/ family_mechanism/ outputs/ f8_stability.csv
- **Downstream reproduction grid**: reproduction/ outputs/ downstream.csv
- **Reconstruction reproduction grid**: reproduction/ outputs/ reconstruction.csv
- **Leak verdict grid**: reproduction/ outputs/ leak_verdict.csv
- **Expected grid cells**: reproduction/ configs/ expected_cells.csv
- **Detailed tractography downstream**: notebooks-FC_to_SC-experimental/ tractography_predict/ e5_downstream_results.csv
- **Per-subject FC achieved-vs-ceiling**: notebooks-FC_to_SC-experimental/ sanity_checks/ noise_sanity_check/ outputs/ h_per_subject_achieved_vs_ceiling.csv
- **FC achieved-vs-ceiling correlations**: notebooks-FC_to_SC-experimental/ sanity_checks/ noise_sanity_check/ outputs/ h_correlations.csv
