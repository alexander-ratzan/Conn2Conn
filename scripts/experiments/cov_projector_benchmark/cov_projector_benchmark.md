# Covariate-Projector Benchmark

**Code:** `scripts/experiments/cov_projector_benchmark/` — `config.yml` (conditions, row groups, tables, figures), `run.py` (runner)
**Results snapshot (tracked):** `scripts/experiments/cov_projector_benchmark/records.json` — scraped 2026-09-23 15:47 from W&B `alexander-ratzan-new-york-university/conn2conn`
**Outputs (regenerable, in this folder):** `tables/`, `figures/` (PNG, 300 dpi), `manifest.json` — all tracked
**Status:** active · replaces `scripts/notebooks/results_scrape/scrape_covtype_results.ipynb` (reproduced exactly, then retired)
**W&B / Ray:** best-trial `prod` runs (`best_trial_report` tag); `NodalGNN` from direct `prod` runs; 70 distinct `ray_tune_id`s; per-cell run ids in `records.json`

---

## 1. Question

1. **Covariates:** does conditioning the PCA-PLS map on subject covariates (a residual projector) improve SC→FC
   prediction over the `CrossModal_PCA_PLS_learnable` baseline, and which feature set helps most — demographics
   (age, sex, race/ethnicity), FreeSurfer volumes, all FreeSurfer features, or demographics + all FreeSurfer?
2. **Model families:** how do the linear family and Krakencoder compare with published deep models (Sarwar 2021 MLP,
   Chen 2024 GCN) and the nodal-feature GNN, all on SC input?

## 2. Setup

| | |
|---|---|
| Conditions (`rows`) | Krakencoder, `PCA_PLS_learnable`, `CrossModal_PCA_PLS_CovProjector` × {`demo`, `fs_volumes`, `fs_all`, `fs_all_demo`}, `Sarwar2020MLP`, `Chen2024GCN`, `NodalGNN` — all `SC` input |
| Row groups | `full` (all 9), `projector_focus` (baseline + 4 projector variants), `global` (all models, projector = `demo`) |
| Seeds | 0–9 (shared seeded splits across models) |
| Run selection | duplicates within a cell → keep max `val_demeaned_r` |
| Tune trials per selected run | learnable 30; projector 24 (one cell 72); Sarwar 8–24; Chen 10–16; Krakencoder / NodalGNN 0 (no tuning) |
| Aggregation | test-split metrics, mean ± std over completed seeds (all 90 cells complete) |

## 3. How to run

```bash
# inside kraken_env via `source /ext3/env.sh` (wandb lives in ~/.local; activate_env.sh hides it)
python scripts/experiments/cov_projector_benchmark/run.py              # tables + figures from records.json
python scripts/experiments/cov_projector_benchmark/run.py --rescrape   # refresh records.json from W&B first
```

Add a condition under `rows:` and reference it from a `row_groups:` list, then run with `--rescrape` (the runner
refuses a cache that does not cover the config, and rejects two rows naming the same condition). Group entries can
override labels, e.g. `{row: projector_demo, plot_label: PCA_PLS_projector}`.

## 4. Results (test split, 10 seeds)

| Condition | corr | demeaned corr | avg rank | top-1 |
|---|---|---|---|---|
| Krakencoder | 0.833 ± 0.004 | 0.085 ± 0.005 | **0.749 ± 0.012** | **0.055 ± 0.011** |
| `PCA_PLS_learnable` (baseline) | 0.837 ± 0.005 | 0.092 ± 0.027 | 0.692 ± 0.061 | 0.033 ± 0.020 |
| projector + demographics | 0.835 ± 0.006 | **0.103 ± 0.010** | 0.725 ± 0.022 | 0.037 ± 0.014 |
| projector + fs_volumes | 0.835 ± 0.005 | 0.094 ± 0.011 | 0.719 ± 0.023 | 0.039 ± 0.017 |
| projector + fs_all | 0.835 ± 0.005 | 0.094 ± 0.013 | 0.711 ± 0.026 | 0.038 ± 0.013 |
| projector + demographics + fs_all | 0.835 ± 0.005 | 0.097 ± 0.010 | 0.707 ± 0.020 | 0.046 ± 0.012 |
| MLP (Sarwar, 2021) | 0.820 ± 0.013 | 0.064 ± 0.012 | 0.638 ± 0.023 | 0.030 ± 0.013 |
| GNN (Chen, 2024) | 0.749 ± 0.018 | 0.019 ± 0.005 | 0.569 ± 0.015 | 0.006 ± 0.004 |
| Nodal GNN | 0.770 ± 0.006 | 0.008 ± 0.015 | 0.549 ± 0.009 | 0.011 ± 0.006 |

Tables: [`tables/full_summary.md`](tables/full_summary.md), [`tables/projector_focus_summary.md`](tables/projector_focus_summary.md);
per-condition seed metrics: [`tables/row_seed_metrics.csv`](tables/row_seed_metrics.csv); figures in [`figures/`](figures/).

## 5. Findings

1. **Every covariate projector matches or beats the learnable baseline on demeaned correlation and avg rank, with
   about a third of its seed-to-seed variance** (demeaned-corr std ≈ 0.010–0.013 vs 0.027). Demographics alone is best
   on demeaned correlation (0.103 vs 0.092) and avg rank (0.725 vs 0.692).
2. **FreeSurfer features add nothing over demographics**: `fs_volumes` and `fs_all` are tied (0.094), and adding
   `fs_all` to demographics lowers demeaned correlation (0.097) while giving the best projector top-1 (0.046).
   Differences among projector variants are within about one standard deviation.
3. **Krakencoder remains best on identifiability** (avg rank 0.749, top-1 0.055) despite a lower demeaned correlation.
4. **Deep models underperform the linear family on SC input.** The Sarwar MLP is closest (demeaned 0.064); both graph
   models are near the null (Chen 0.019, Nodal GNN 0.008 vs 0.012 for `CrossModalPCA` SC→FC in the SC-type benchmark),
   and their raw correlation (0.75–0.77) falls below that null (0.818).
5. Raw correlation again does not separate the linear/projector conditions (0.834–0.837).

## 6. Caveats

- **v2:C!1 (regularization not applied):** the `PCA_PLS_learnable`, all four projector, and `Sarwar2020MLP` rows come
  from sweeps that trained with the default regularization (L2 = 1e-4; none for Sarwar); their logged `l1_reg`/`l2_reg`
  were not applied (v1:M5b). The projector-vs-baseline comparison is at equal fixed L2. Resolved by v2:C2 (re-tune).
- Single input source (`SC`); covariate effects on `SC_r2t` / `SC+SC_r2t` are not tested here.
- `NodalGNN` results come from direct `prod` runs with default configs (no tune sweep), so it is less tuned than the
  other learned models.
- 10 older `CrossModal_PCA_PLS_learnable` best-trial runs have neither `source` nor `shuffle_seed` in their W&B config
  and are skipped at scrape time (the notebook skipped them too; both of its tables are reproduced exactly).
- Figure note: in `cov_dl_panels__global.png` the best-value annotation can overlap its error bar (plotting-function
  styling, unchanged from the notebook).

Last updated at: 2026-09-23 EDT
