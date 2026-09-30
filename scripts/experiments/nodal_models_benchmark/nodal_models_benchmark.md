# Nodal Models Benchmark

**Code:** `scripts/experiments/nodal_models_benchmark/` — `config.yml` (variants, references, rows, tables, figures), `run.py` (runner), `preflight/` (spec v2 E0.6)
**Results snapshots (tracked):** `records.json` (one best-trial run per variant × seed) and `trials.json` (554 tune trials) — scraped 2026-09-30 09:45 from W&B `alexander-ratzan-new-york-university/conn2conn`
**Outputs (regenerable, in this folder):** `tables/`, `figures/` (PNG, 300 dpi), `manifest.json` — all tracked
**Status:** preliminary (spec v2 E0) · NodalGNN pilot pending (E0.5, needs the environment fix v2:C6 / v2:C1) · replaces `scripts/notebooks/results_scrape/nodal_decoder_importance.ipynb` (never run to completion; retired at close-out)
**W&B / Ray:** best-trial `prod` runs (`best_trial_report` tag), NodalMLP tune trials created ≥ 2026-04-27; per-cell run ids in `records.json`

---

## 1. Question

1. How much of SC→FC can **per-region (nodal) models** recover? NodalMLP embeds each region from its SC row and
   decodes edges with a simple "probe" decoder — `dot`, `bilinear`, `linear_beta`, or a connectome-harmonic
   (`spectral`) encoder — compared with NodalMLP's flexible MLP decoder, NodalGNN, `Chen2024GCN`, the null, and the
   linear family.
2. Which hyperparameters matter for the probes? (Recorded as closed for a later revisit from the SMT angle in
   GeneEx2Conn.)

## 2. Setup

| | |
|---|---|
| Task | `SC → FC`, Glasser, MSE-only loss (every logged `loss_signature` is `mse`; older trials predate the field) |
| Variants (scraped) | NodalMLP `dot`, `bilinear`, `linear_beta`, `spectral` — tune trials whose config matches `models/configs/NodalMLP_<variant>.yml`; `mlp_sc_rows` — pre-schema `NodalMLP.yml` runs (`use_sc_row: true`, `use_volume: false`, no `decoder_type`) |
| References (from snapshots) | null `CrossModalPCA` SC and `CrossModal_PCA_PLS_learnable` (`sc_type_benchmark`); `Chen2024GCN` and untuned `NodalGNN` (`cov_projector_benchmark`) |
| Seeds | 0–3 (shared seeded splits); bilinear seed 0 has no best-trial run (its job was killed at 26/32 trials, not re-run) |
| Tuning | Optuna + ASHA, 32 trials per variant × seed (spectral seed 3: 56; bilinear seed 0: 50 across two jobs) |
| Run selection | duplicates within a cell → keep max `val_demeaned_r` (every cell had one candidate) |
| Importance | fANOVA (forest seed 0) per variant on tune trials, **after subtracting each seed's median trial score** (`importance_center: seed`, see Finding 3); `reg` read as `l2_reg` |
| Aggregation | test-split metrics, mean ± std over available seeds |

## 3. How to run

```bash
# inside kraken_env via `source /ext3/env.sh` (wandb/optuna live in the launcher stack; v2:C6)
python scripts/experiments/nodal_models_benchmark/run.py              # tables + figures from records.json + trials.json
python scripts/experiments/nodal_models_benchmark/run.py --rescrape   # refresh both from W&B first
```

A cache-mode run reproduces every table and figure byte for byte (checked 2026-09-30).

## 4. Results (test split, seeds 0–3)

| Condition | n | corr | demeaned corr | mse | avg rank |
|---|---|---|---|---|---|
| `PCA_PLS_learnable` | 4 | 0.834 ± 0.005 | **0.075 ± 0.034** | 0.0136 | **0.665 ± 0.086** |
| Null (`CrossModalPCA` SC) | 4 | 0.824 ± 0.002 | 0.013 ± 0.010 | 0.0144 | 0.518 ± 0.019 |
| GNN (Chen, 2024) | 4 | 0.743 ± 0.024 | 0.018 ± 0.005 | 0.0201 | 0.564 ± 0.008 |
| NodalGNN (untuned default) | 4 | 0.769 ± 0.005 | 0.016 ± 0.008 | 0.0184 | 0.554 ± 0.011 |
| NodalMLP, MLP decoder | 2 | 0.773 ± 0.033 | 0.023 ± 0.003 | 0.0178 | 0.562 ± 0.021 |
| NodalMLP `dot` | 4 | 0.436 ± 0.182 | 0.011 ± 0.013 | 0.343 | 0.535 ± 0.010 |
| NodalMLP `bilinear` | 3 | 0.165 ± 0.212 | 0.013 ± 0.014 | 3.26 | 0.503 ± 0.013 |
| NodalMLP `linear_beta` | 4 | 0.436 ± 0.220 | 0.018 ± 0.010 | 0.0347 | 0.534 ± 0.019 |
| NodalMLP `spectral` | 4 | 0.259 ± 0.032 | 0.014 ± 0.016 | 0.0418 | 0.529 ± 0.008 |

Tables: [`tables/test_summary.md`](tables/test_summary.md), [`tables/trial_summary.md`](tables/trial_summary.md),
[`tables/importance_wide.md`](tables/importance_wide.md); per-seed metrics:
[`tables/row_seed_metrics.csv`](tables/row_seed_metrics.csv); figures in [`figures/`](figures/). The linear reference is
0.075 here (seeds 0–3) vs 0.092 over the 10 seeds of `sc_type_benchmark`.

## 5. Findings (preliminary)

1. **Every nodal model is at the null.** The four probes (demeaned corr 0.011–0.018, avg rank 0.50–0.54) and the
   nodal/graph references (0.016–0.023) sit within about one seed SD of the null (0.013) and far below `PCA_PLS_learnable`
   (0.075). The probes cannot be ranked against each other at n = 3–4.
2. **Probe decoders are badly calibrated in scale.** Raw corr 0.16–0.44 and MSE 2–230× the null (bilinear 3.26 from a
   diverged seed), so they recover neither FC's shared structure nor its subject-specific part.
3. **Validation scores are set by the split, not the hyperparameters.** Tune-trial scores fall into a few bands per
   seed — all 32 spectral trials on seed 1 score the same, and seed 2 scores ≈ 0.042 for every variant
   ([`figures/trial_distribution.png`](figures/trial_distribution.png)). Pooled across seeds, fANOVA credited this split
   variance to whichever parameter happened to co-vary (bilinear `sc_row_norm` 0.56); after within-seed centring the
   importances are near-uniform (largest: bilinear `lr` 0.37; every other value ≤ 0.27). No
   hyperparameter moves these models off the null.
4. **The tuning budget was not justified** (612 trials over 38 array tasks, ~47 GPU-hours, for null-level models).
   This motivated spec v2 D2: pilot first, pack small models, scale on evidence.
5. **Gate for NodalGNN (E0.5):** validation scores are only comparable within a seed, so the seed-0 pilot must reach a
   best val demeaned r ≥ 0.045 (the null's seed-0 val 0.025 + 0.02; `PCA_PLS_learnable` reaches 0.126) before
   seeds 1–3 are run. The probes' seed-0 best trials reach −0.006 to 0.012.

## 6. Caveats

- **Preliminary:** NodalGNN is so far only its untuned default; bilinear has n = 3; NodalMLP MLP decoder has test
  metrics for 2 of its 4 seeds (the April runs for seeds 1–2 logged no test metrics; `tables/test_summary.md` shows
  `(n=4)` because it counts records, not values).
- **v2:C!1 (regularization not applied):** affects the `PCA_PLS_learnable` reference, whose sweeps trained with the
  default L2 = 1e-4 (v1:M5b). NodalMLP applies its own `l2_reg`, so the probe rows are unaffected.
- Best-trial re-runs score lower on val than their tune trial (for example bilinear seed 2: 0.049 in tuning, 0.025 on
  re-fit), consistent with Finding 3: the tune maximum is partly selection noise.
- `mlp_sc_rows` comes from an older schema and date window (created before 2026-04-27), so it is a control, not a
  matched condition.

Last updated at: 2026-09-30 EDT
