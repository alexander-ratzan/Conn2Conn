# preprint-claudev2 — Findings Ledger (F1–F10), deep dive

A finding-by-finding companion to the main preprint. Instead of a narrative paper, this
version walks **each finding in the project ledger one at a time** and shows the evidence
**across both parcellations** (Glasser, 4S456Parcels) **and across all three estimators**
(PCA→PLS, BayesianRidge, KernelRidge). Every finding gets its own rich multi-panel figure.

## Build

```bash
bash build_pdf.sh                              # figures + compile
/Users/user/dev-env/bin/python make_findings.py        # F1-F10 figures
/Users/user/dev-env/bin/python make_findings_extra.py  # F11-F13 figures
```

Output: **`conn2conn-findings.pdf`** (11 pages) + `figures/F1..F13_*.pdf/.png`.

### Pick-and-choose slideshow

```bash
/Users/user/dev-env/bin/python make_slides.py     # ~69 chart variants -> figures/slides/
/Users/user/dev-env/bin/python build_slides.py    # auto-assemble slides.tex
latexmk -pdf slides.tex                            # -> slides.pdf (Beamer, 85 pages)
```

**`slides.pdf`** is a Beamer deck with one section per finding (F1–F13) and **multiple chart
variants per finding** (grouped bars, horizontal bars, lollipop, dumbbell, slope, heatmap,
box+seed-dots, plus bespoke line charts) so you can pick favourites. Individual variant images
live in `figures/slides/F{n}_v_{type}.{pdf,png}`. `build_pdf.sh` builds everything
(findings doc + slideshow) in one go.

## The ledger

| F | Meaning | Status | Figure |
|---|---|---|---|
| F1 | FC→SC > SC→FC asymmetry | replicated both | `F1_directional_asymmetry` |
| F2 | anatomy→SC / demographics→FC dissociation | replicated both | `F2_double_dissociation` |
| F3 | imputed connectomes measurable, utility asymmetric | replicated both (via F5) | `F3_imputation_usable_asymmetric` |
| F4 | observed FC carries cognition signal | replicated both | `F4_obs_fc_cognition` |
| F5 | SC underperforms; imputation no transfer; pred_FC harmful | replicated both | `F5_utility_wall` |
| F6 | predicted connectomes carry family signal | replicated both | `F6_family_signal` |
| F7 | reconstruction vs identification objective tradeoff | replicated both | `F7_objective_tradeoff` |
| F8 | exploratory PC mechanism (PC3 Glasser / PC4 4S456) | partial / exploratory | `F8_pc_mechanism` |
| F9 | richer tractography does not help | Glasser / exploratory | `F9_tractography` |
| F10 | nonlinear / more data do not break the ceiling | Glasser + both-parc reliability | `F10_structural_ceiling` |
| F11 | observed-FC cognition lift is real but multiplicity-fragile | both (this analysis) | `F11_statistical_fragility` |
| F12 | Bayesian corroboration (shrinkage / ROPE / Bayes factors) | both (this analysis) | `F12_bayesian_corroboration` |
| F13 | imputation harm direction-specific; cross-modal loss vs oracle | both (this analysis) | `F13_imputation_and_oracle` |

F1–F10 come from the project ledger (`PROJECT-HANDOFF-2026-06-22.md`); **F11–F13 are extra
findings produced in this analysis** (`make_findings_extra.py`) that stress-test the affirmative
claim (multiplicity + Bayesian) and quantify imputation harm / cross-modal loss.

## Notes

- Source of truth: the frozen reproduction-grid CSVs (`reproduction/outputs/`,
  `reproduction/family_mechanism/outputs/`, `notebooks-FC_to_SC-experimental/...`).
  Figures are generated, not hand-drawn; `style.py` carries the loaders and the
  cross-estimator aggregation (KernelRidge is the median over its 9 HP variants).
- Reconstruction leads with `demeaned_pearson`; downstream leads with BayesianRidge lift over
  `bv+demo`. KernelRidge scalar (cognition) rows are ill-conditioned and shown diagnostically
  only — visible as sign-flips in F5C.
- Ledger source: `dev-notes/PROJECT-HANDOFF-2026-06-22.md` (“The findings (F1–F10)”).
