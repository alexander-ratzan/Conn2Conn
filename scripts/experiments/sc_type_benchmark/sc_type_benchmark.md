# SC-Type Benchmark

**Code:** `scripts/experiments/sc_type_benchmark/` — `config.yml` (grid, table, figures), `run.py` (runner)
**Results snapshot (tracked):** `scripts/experiments/sc_type_benchmark/records.json` — scraped 2026-09-23 14:57 from W&B `alexander-ratzan-new-york-university/conn2conn`
**Outputs (regenerable, in this folder):** `tables/`, `figures/` (PNG, 300 dpi), `manifest.json` — all tracked
**Status:** active · replaces `scripts/notebooks/results_scrape/scrape_SCtype_results.ipynb` (reproduced exactly, then retired)
**W&B / Ray:** best-trial `prod` runs (`best_trial_report` tag); 110 distinct `ray_tune_id`s; per-cell run ids in `records.json`

---

## 1. Question

How much does the structural input type — `SC`, `SC_r2t`, or `SC+SC_r2t` — matter for SC→FC prediction across the
linear model family and Krakencoder? `CrossModalPCA` provides the references: `FC` input = oracle, `SC` input = null.

## 2. Setup

| | |
|---|---|
| Models | `CrossModalPCA`, `CrossModal_PLS_SVD`, `CrossModal_PCA_PLS`, `CrossModal_PCA_PLS_learnable`, `Krakencoder_precomputed` |
| Sources | `SC`, `SC_r2t`, `SC+SC_r2t`, `FC` |
| Seeds | 0–9 (shuffle seeds; every model shares the same seeded splits) |
| Run selection | duplicates within a `(model, source, seed)` cell → keep max `val_demeaned_r` |
| Aggregation | test-split metrics, mean ± std over completed seeds |

## 3. How to run

```bash
# inside kraken_env via `source /ext3/env.sh` (wandb lives in ~/.local; activate_env.sh hides it)
python scripts/experiments/sc_type_benchmark/run.py              # tables + figures from records.json
python scripts/experiments/sc_type_benchmark/run.py --rescrape   # refresh records.json from W&B first
```

To benchmark more models or sources, edit `config.yml` and run with `--rescrape` (the runner refuses to use a cache
that does not cover the config). New figures: add entries to `figures:`; new figure types: add a function to
`scripts.results_utils.FIGURE_TYPES`.

## 4. Results (test split, 10 seeds)

| Model | Source | corr | demeaned corr | avg rank | top-1 |
|---|---|---|---|---|---|
| `CrossModalPCA` (null) | SC | 0.818 ± 0.010 | 0.012 ± 0.008 | 0.522 ± 0.016 | 0.010 ± 0.008 |
| `CrossModalPCA` (oracle) | FC | 0.916 ± 0.006 | 0.691 ± 0.026 | 1.000 | 1.000 |
| `CrossModal_PLS_SVD` | SC | 0.823 ± 0.011 | 0.073 ± 0.014 | 0.653 ± 0.037 | 0.035 ± 0.013 |
| | SC_r2t | 0.817 ± 0.020 | 0.039 ± 0.011 | 0.588 ± 0.025 | 0.024 ± 0.008 |
| `CrossModal_PCA_PLS` | SC | 0.834 ± 0.007 | 0.090 ± 0.010 | 0.704 ± 0.024 | 0.034 ± 0.013 |
| | SC_r2t | 0.834 ± 0.005 | 0.048 ± 0.007 | 0.613 ± 0.018 | 0.016 ± 0.009 |
| | SC+SC_r2t | 0.835 ± 0.005 | 0.079 ± 0.009 | 0.677 ± 0.026 | 0.029 ± 0.013 |
| `CrossModal_PCA_PLS_learnable` | SC | 0.837 ± 0.005 | **0.092 ± 0.027** | 0.692 ± 0.061 | 0.033 ± 0.020 |
| | SC_r2t | 0.835 ± 0.004 | 0.033 ± 0.021 | 0.581 ± 0.048 | 0.012 ± 0.007 |
| | SC+SC_r2t | 0.836 ± 0.004 | 0.083 ± 0.015 | 0.679 ± 0.029 | 0.026 ± 0.011 |
| `Krakencoder_precomputed` | SC | 0.833 ± 0.004 | 0.085 ± 0.005 | **0.749 ± 0.012** | **0.055 ± 0.011** |

Full table (incl. MSE and empty cells): [`tables/summary_table.md`](tables/summary_table.md); figures in [`figures/`](figures/).

## 5. Findings

1. **Raw correlation does not discriminate models** (0.82–0.84 for every SC model vs 0.818 for the null); it is dominated by
   the population-mean FC. Demeaned correlation and identifiability (avg rank, top-1) are the informative metrics.
2. **Among SC-input models, PCA/PLS variants and Krakencoder are close on demeaned correlation** (0.085–0.092);
   the learnable map has the highest mean but ~3× the seed variance of closed-form `PCA_PLS`. Krakencoder leads on
   identifiability.
3. **Input type: SC > SC+SC_r2t > SC_r2t** for demeaned correlation and avg rank in every model with all three;
   adding r2t to SC does not help. `SC_r2t` alone is the weakest structural input.
4. All models remain far from the FC→FC oracle (demeaned corr 0.69).

## 6. Coverage gaps and caveats

- Missing cells (80 / 200): no `SC+SC_r2t` runs for `CrossModalPCA` / `PLS_SVD`; Krakencoder is SC-only; the FC
  oracle exists only for `CrossModalPCA` (by design).
- 15 older best-trial runs have neither `source` nor `shuffle_seed` in their W&B config and are skipped at scrape
  time (the notebook skipped them too; its table is reproduced exactly).
- Krakencoder has two seed-0 runs; the one with higher `val_demeaned_r` (`cbezlmbw`) is used.
- Scraper fixes made while building this (2026-09-23): wandb 0.25 `Api.runs()` lazy-loads empty configs
  (`query_runs` requests full data); runs are attributed to the model they were fetched for, not their first tag
  (seed-0 Krakencoder carries an extra `Glasser` tag).

Last updated at: 2026-09-23 EDT
