# Main figures — `preprint-claude`

Publication figures for **"What Cross-Modal Connectome Prediction Does and Does
Not Buy"** ([../manuscript.md](../manuscript.md)), generated directly from the
frozen reproduction-grid CSVs. No numbers are hand-entered: every value is read
from a source artifact at render time.

## How to regenerate

```bash
bash build_pdf.sh                                  # figures + tables + compile PDF
/Users/user/dev-env/bin/python make_all.py        # 5 main figs + 10 supp figs + 3 tables
/Users/user/dev-env/bin/python make_fig3.py       # a single main figure
/Users/user/dev-env/bin/python make_supp.py       # supplement figures only
/Users/user/dev-env/bin/python make_tables.py     # tables only
```

Two compiled documents:
- `conn2conn-preprint.pdf` (13 pages) — the main paper. Figures are embedded inline
  next to the sections they support (no appendix dump).
- `conn2conn-supplement.pdf` (11 pages) — a long, standalone **Sanity Checks &
  Reproducibility** supplement covering every check in `notebooks-FC_to_SC-experimental/`
  (reduction-axis, FC noise/reliability, nonlinear, residual-boost, scaling, tractography,
  PC-mechanism partialling, completeness/leaks). Its prose source is
  `SANITY_CHECKS_SUPPLEMENT.md`. It is built from `conn2conn-preprint.tex` using the
NeurIPS-2026 `preprint` style (`neurips_2026.sty`, taken from
`lucky-fold-paper/arXiv-preprint`). Tables are `\input` from CSV-generated
`tables/tableN_body.tex` (no hand-typed numbers); figures are embedded as vector PDFs.
LaTeX is compiled with TinyTeX (`latexmk -pdf`).

Outputs:
- `figures/figN_*.png|pdf` — five main figures (300 dpi PNG + vector PDF)
- `figures/supp/SN_*.png|pdf` — ten supplement figures
- `tables/tableN_*.png|pdf` and `tables/tables.md` — main tables 1–3

`style.py` holds the shared palette, rcParams, and the CSV loaders.

### Post-hoc robustness analyses (from existing CSVs)

- `make_robustness.py` → `ROBUSTNESS_ANALYSIS.md`, `tables/robustness_*.csv`, figure
  `SX5` — **frequentist** reviewer-hole fixes: BH-FDR + CIs on downstream lifts,
  TOST equivalence + minimum-detectable-effect, same-estimator oracle, imputation-harm
  contrast.
- `make_robustness_bayes.py` → `ROBUSTNESS_ANALYSIS_BAYES.md`, figure `SX6` — the
  **Bayesian** counterpart: hierarchical partial-pooling (PyMC NUTS) for multiplicity,
  ROPE + HDI decisions, JZS Bayes factors, posterior credible intervals. Includes a
  unit-of-analysis caveat reconciling it with the frequentist doc.
- `NEEDED_EXPERIMENTS.md` — what was fixed vs what needs new data/pipeline runs (E1–E9,
  with feasibility ratings).

## Figures, claims, and provenance

| Figure | Manuscript claim it carries | Source artifact(s) |
|---|---|---|
| **Fig 1** `fig1_evaluation_spine` | The study separates reconstruction from utility and makes `bv+demo` the checkpoint (§2). | schematic |
| **Fig 2** `fig2_directional_translation` | FC→SC > SC→FC (1.61× Glasser, 1.68× 4S456), every seed >1; oracle self-prediction ceilings ~equal; anatomy→SC / demographics→FC double dissociation (§3). | `reproduction/outputs/reconstruction.csv` |
| **Fig 3** `fig3_utility_checkpoint` | Only observed FC clears `bv+demo` for cognition; observed SC and **imputed** connectomes fall below it; `pred_FC` is actively harmful (§4). | `reproduction/outputs/downstream.csv` |
| **Fig 4** `fig4_closed_escape_routes` | The ceiling is not a PCA, model-class, structural-detail, sample-size, or FC-noise artifact in this regime (§5). | `sanity_checks/noise_sanity_check/outputs/*`, `sanity_checks/preprocessing_check/reduction_axis_*`, `non-linear-sanity-check/n6_scaling_summary.csv`, `tractography_predict/tractography_synthesis.csv` |
| **Fig 5** `fig5_objective_divergence` | Predicted SC carries sibling/family signal (AUC≈0.81 residualized) but the reconstruction-optimized predictor collapses to chance; a property-selected FC-predictable low-variance SC mode (PC3 Glasser / PC4 4S456) carries it, while PC1 is a sex/volume confound (§6). | `reproduction/family_mechanism/outputs/*` |

## Supplement figures (`figures/supp/`)

| Fig | Content | Source |
|---|---|---|
| S1 | raw vs demeaned pearson (why demeaned leads) | `reconstruction.csv` |
| S2 | seed-level reconstruction distributions | `reconstruction.csv` |
| S3 | seed-level cognition lifts | `downstream.csv` |
| S4 | leak-check sex/age by input | `downstream.csv` |
| S5 | reduction-axis robustness (per-seed ratios) | `reduction_axis_*` |
| S6 | FC reliability histogram + achieved-vs-ceiling | `noise_sanity_check/outputs/*` |
| S7 | nonlinear & scaling nulls | `n6_scaling_summary.csv` |
| S8 | tractography/source-representation comparison | `tractography_synthesis.csv` |
| S9 | family AUC by predictor × relatedness | `family_auc.csv` |
| S10 | PC mechanism: FC-predictability/AUC + network enrichment | `f8_per_pc.csv`, `f8_pc3_enrichment_agg.csv` |

## Tables (`tables/`)

`table1_dataset` (cohort/parcellations/metrics), `table2_grid` (task→claim map),
`table3_headline` (headline numbers, computed live from the CSVs). All three are
also collected in `tables/tables.md` for direct paste into the manuscript.

## Key numbers (verified against CSVs at build time)

- FC→SC vs SC→FC demeaned-r (PCA→PLS): **0.136 / 0.084** Glasser (1.61×),
  **0.146 / 0.087** 4S456 (1.68×).
- Oracle ceilings (BayesianRidge): FC→FC 0.673 / SC→SC 0.648 (Glasser).
- `bv+demo` cognition (BayesianRidge): CogTotal 0.359, CogFluid 0.283,
  CogCryst 0.354.
- Observed FC CogCryst lift: **+0.133** Glasser (p=0.031), **+0.122** 4S456
  (p=0.041). Imputed `pred_FC` CogCryst lift: **−0.135 / −0.109**.
- SC→FC reaches only **17%** of the FC between-session reliability ceiling
  (0.085 of 0.491); per-subject achieved vs reliability r=0.01.
- Sibling AUC: residualized pred-SC ≈ **0.81**, combined pred-SC ≈ **0.50**
  (chance), `bv+demo` ≈ **0.56**.

## Notes

- Reconstruction leads with `demeaned_pearson`; downstream leads with
  BayesianRidge lift over `bv+demo` — matching the drafting stance in
  [../README.md](../README.md).
- Fig 5C is labelled as a *property-selected mode*, not a fixed PC3 result, per
  the F8 style rule.
