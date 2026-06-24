# Conn2Conn Preprint Structure

Working title:

**What Cross-Modal Connectome Prediction Does and Does Not Buy**

Alternate title:

**Connectome Translation Is Reproducible but Not Sufficient: Baselines, Ceilings, and Utility in FC-SC Prediction**

## Paper Shape

This should be a methods / redirect paper, not a leaderboard paper. The reader-facing claim is
that FC-SC translation is real and reproducible, but reconstruction accuracy alone is too weak a
standard. The paper should install two rules:

1. Report individual-deviation reconstruction metrics, not population-mean-dominated correlations.
2. For cognition or behavioral utility, beat a cheap `bv+demo` subject-information baseline.

The main paper should be tight. The supplement should carry the defense discipline: frozen splits,
completeness, leak checks, reliability ceilings, reduction-axis checks, nonlinear nulls, tractography
checks, and notebook-vs-grid validation.

## Main Text Sections

| Section | Purpose | Main evidence | Figures | Tables |
|---|---|---|---|---|
| Abstract | State the redirect: translation works, utility does not automatically follow | F1-F7 summary | none | none |
| 1. Introduction | Frame the field's assumption: predict the other connectome, then use it | Prior FC-SC work, baseline gap, HCP-YA motivation | none | none |
| 2. Evaluation Design | Define data, parcellations, frozen splits, metrics, estimators, and `bv+demo` baseline | Grid design, `demeaned_pearson`, BR downstream rule | Fig. 1 | Table 1, Table 2 |
| 3. Cross-Modal Prediction Is Directional | Establish FC->SC > SC->FC, replicated across seeds/parcellations | Reconstruction grid, oracle ceilings, double dissociation | Fig. 2 | optional Table 3 |
| 4. Reconstruction Does Not Imply Cognitive Utility | Show FC helps cognition, SC does not, imputed connectomes do not transfer utility | Downstream grid, lift over `bv+demo`, paired permutation p | Fig. 3 | Table 3 |
| 5. The Ceiling Is Structural in This Regime | Rule out easy escape routes: richer SC, nonlinear models, more n, reliability artifact | F9/F10, Ceiling A/B, KR flatness | Fig. 4 | none or Supp table |
| 6. Predicted Connectomes Carry Family Signal but Objectives Diverge | Show the constructive exception: family/heritable signal survives, but reconstruction and identification conflict | F6/F7/F8 property-selected mode | Fig. 5 | optional Table 4 |
| 7. Discussion | Translate into reporting standard and limitations | all above | none | none |

## Main Figures

| Figure | Working title | Panels | What it must prove | Source artifacts |
|---|---|---|---|---|
| Fig. 1 | Evaluation spine | A: inputs and targets; B: reconstruction vs downstream branches; C: frozen split / both-parcellation grid | The study separates reconstruction from downstream utility and makes `bv+demo` the checkpoint | `reproduction/`, `planning/reproducibility_and_grid_plan_theory.md` |
| Fig. 2 | Directional translation | A: FC->SC vs SC->FC `demeaned_r`; B: ratio by seed/parcellation; C: within-modality oracle FC->FC and SC->SC; D: anatomy/demo double dissociation | FC->SC > SC->FC is real, not a target-reliability artifact; anatomy predicts SC, demographics predict FC | `reproduction/outputs/reconstruction.csv`, `reproduction/reports/reproduction_findings.md` |
| Fig. 3 | Utility checkpoint | A: cognition Pearson by input; B: lift over `bv+demo`; C: paired permutation p markers; D: observed vs imputed connectome contrast | Only observed FC reliably lifts cognition; SC and imputed connectomes do not clear the baseline; `pred_FC` is harmful | `reproduction/outputs/downstream.csv`, `reproduction/reports/reproduction_findings.md` |
| Fig. 4 | Closed escape routes | A: cross-modal vs oracle ceiling fraction; B: estimator robustness / KR HP flatness; C: richer tractography/r2t null; D: nonlinear/residual/scaling nulls; E: FC reliability ceiling | The ceiling is not a modeling artifact, richer structural feature artifact, sample-size artifact, or FC-noise artifact in this regime | `reproduction/exploration/FINDINGS_EXPLORATION.md`, `non-linear-sanity-check/*`, `tractography_predict/*`, `sanity_checks/noise_sanity_check/*` |
| Fig. 5 | Signal changes with objective | A: sibling AUC for baseline/raw/residual/combined predicted connectomes; B: reconstruction/utility vs family identification tradeoff; C: property-selected SC mechanism mode PC3 Glasser / PC4 4S456 | Predicted connectomes are not useless; they carry family signal, but the objective determines what signal survives | `reproduction/family_mechanism/outputs/*.csv` |

## Main Tables

| Table | Working title | Columns | Notes |
|---|---|---|---|
| Table 1 | Dataset and parcellations | cohort, subjects, split, parcellation, nodes, edges, metric | Keep compact; full split details in supplement |
| Table 2 | Task grid and claims | task, input, target, estimator, primary metric, claim | Reader-facing version of the internal grid; no implementation minutiae |
| Table 3 | Headline findings | claim, Glasser, 4S456, estimator, interpretation | One-stop summary for F1-F7 |
| Table 4 optional | Reporting standard | requirement, trap avoided, implementation in this study | Could live in Discussion instead |

## Supplement Structure

The supplement should be detailed, not decorative. It should show that the paper did not merely
run a grid, but tested the ways the grid could be silently wrong.

| Supplement section | Purpose | What to include | Source artifacts |
|---|---|---|---|
| S1. Data, features, and splits | Make the cohort and frozen split contract reproducible | HCP-YA inclusion, atlas details, edge vectorization, family-aware frozen splits, `seed{0..9}.json` | `reproduction/splits/`, `data/atlas_info/` |
| S2. Metrics and estimators | Explain why `demeaned_pearson` leads and why BR leads downstream | raw Pearson caveat, demeaned metric, avg_rank/top1, estimator definitions, BR discrepancy resolution | `dev-notes/PROJECT-HANDOFF-2026-06-22.md`, `reproduction/exploration/DISCREPANCY_RESOLUTION.md` |
| S3. Completeness and leak guardrails | Show every expected cell exists and no hard leak was found | expected-cell manifest, non-finite checks, leak verdict categories, W&B/CSV source-of-truth | `reproduction/configs/expected_cells.csv`, `verify_completeness.py`, `outputs/leak_verdict.csv` |
| S4. Reduction-axis robustness | Defend FC->SC asymmetry against PCA artifacts | full PLS, learned PCA, JL projections, all ratios >1 | `sanity_checks/preprocessing_check/findings.md` |
| S5. Reliability ceilings and noise | Defend against "SC->FC fails because FC is noisy" | Ceiling A vs B, FC test-retest, edge noise vs whole-connectome reliability, per-subject flatness | `sanity_checks/noise_sanity_check/findings_noise.md` |
| S6. Nonlinear and scaling nulls | Defend against "bigger model / more data would fix it" | KernelRidge/HGB, residual boost, sink models, n-scaling curve | `non-linear-sanity-check/*summary.csv`, `findings_*.md` |
| S7. Richer structural representations | Defend against "streamline counts are too crude" | r2t bundle features, source representation, marginal tests, cognition tests | `tractography_predict/*summary.csv` |
| S8. Family mechanism validation | Give depth for F6/F7/F8 | bit-exact notebook validation, AUCs, predictor/identifier tradeoff, PC mode selection by property | `reproduction/family_mechanism/outputs/*.csv` |
| S9. PC localization and tractography reliability | Hedge F8 correctly | PC1 confound, PC3/PC4 migration, visual/DAN enrichment, strength/distance partialling | `sanity_checks/tract_check/findings.md` |
| S10. Operational reproducibility | Document exact run discipline | Torch, SLURM, sentinels, W&B replay, CSV source-of-truth, git commit provenance | `planning/*runlog.md`, `dev-notes/PROJECT-HANDOFF-2026-06-22.md` |

## Supplement Tables

| Table | Content |
|---|---|
| S1 | Full reconstruction grid by parcellation, estimator, input, target |
| S2 | Full downstream grid by parcellation, estimator, input, target |
| S3 | Leak-check verdict counts and threshold definitions |
| S4 | Expected vs observed completeness counts |
| S5 | Reduction-axis robustness results |
| S6 | FC reliability ceiling and variance decomposition |
| S7 | Nonlinear null summary |
| S8 | Tractography/r2t summary |
| S9 | Family AUC summary |
| S10 | F8 per-PC stability, confounds, and property-selected mechanism mode |

## Supplement Figures

| Figure | Content |
|---|---|
| S1 | Metric illustration: raw Pearson vs demeaned Pearson |
| S2 | Seed-level reconstruction distributions |
| S3 | Downstream seed-level lifts |
| S4 | Leak-check sex/age scores |
| S5 | Reduction-axis robustness plot |
| S6 | FC reliability histogram and achieved-vs-reliability scatter |
| S7 | Nonlinear/scaling null plots |
| S8 | r2t/source-representation comparison |
| S9 | Family AUC distributions |
| S10 | PC mechanism localization and enrichment, Glasser vs 4S456 |

## Style Rules For Drafting

- Main text should name estimators next to every number.
- Main text should not overclaim F8. Use "property-selected low-variance SC mode" rather than
  "PC3" unless specifically referring to Glasser.
- Treat sex/age as leak checks, not as core findings.
- Treat `bv+demo` as the reporting-standard baseline, not a nuisance covariate.
- Keep the main paper figure captions interpretive: each caption should state the result.
- Put implementation gates, operational failures, and detailed sanity-check mechanics in the
  supplement, but do not hide their conclusions.
