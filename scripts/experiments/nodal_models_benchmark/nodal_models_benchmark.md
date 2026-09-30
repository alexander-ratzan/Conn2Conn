# Nodal Models Benchmark

**Code:** `scripts/experiments/nodal_models_benchmark/` — `config.yml` (variants, references, rows, tables, figures), `run.py` (runner), `preflight/` (spec v2 E0.6), `pilot/` (NodalGNN pilot, E0.5)
**Results snapshots (tracked):** `records.json` (one best-trial run per variant × seed) and `trials.json` (554 tune trials) — scraped 2026-09-30 14:58 from W&B `alexander-ratzan-new-york-university/conn2conn`
**Outputs (regenerable, in this folder):** `tables/`, `figures/` (PNG, 300 dpi), `manifest.json` — all tracked
**Status:** closed 2026-09-30 (spec v2 E0) · follow-up recorded in §7 · replaces `scripts/notebooks/results_scrape/nodal_decoder_importance.ipynb` (never run to completion; removed at close-out, in git history)
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
| Task | `SC → FC`, Glasser, **MSE-only** loss. Checked per run from the W&B config (`loss_type`, `loss_terms`, searched `loss_weight_*`): every learned run in the tables and figures and all 554 importance trials trained on plain MSE; `run.py` refuses to render otherwise |
| Variants (scraped) | NodalMLP `dot`, `bilinear`, `linear_beta`, `spectral` — tune trials whose config matches `models/configs/NodalMLP_<variant>.yml`; `mlp_sc_rows` — pre-schema `NodalMLP.yml` runs (`use_sc_row: true`, `use_volume: false`, no `decoder_type`) |
| Linear reference (scraped) | `CrossModal_PCA_PLS_learnable` SC, best MSE-only run per seed (the `sc_type_benchmark` snapshot picks across loss types, and `demeaned_mse` wins 3 of 4 seeds there) |
| References (from snapshots) | null `CrossModalPCA` SC (closed-form, no training loss) from `sc_type_benchmark`; `Chen2024GCN` and untuned `NodalGNN` (both `loss_type: mse`) from `cov_projector_benchmark` |
| Seeds | 0–3 (shared seeded splits); bilinear seed 0 has no best-trial run (its job was killed at 26/32 trials, not re-run) |
| Tuning | Optuna + ASHA, 32 trials per variant × seed (spectral seed 3: 56; bilinear seed 0: 50 across two jobs) |
| Run selection | duplicates within a cell → keep max `val_demeaned_r` (every cell had one candidate) |
| Importance | fANOVA (forest seed 0) per variant on tune trials, **after subtracting each seed's median trial score** (`importance_center: seed`, see Finding 3); `reg` read as `l2_reg` |
| Aggregation | test-split metrics, mean ± std over available seeds |

## 3. How to run

```bash
# inside kraken_env via `source /ext3/env.sh` (the launchers' job environment)
python scripts/experiments/nodal_models_benchmark/run.py              # tables + figures from records.json + trials.json
python scripts/experiments/nodal_models_benchmark/run.py --rescrape   # refresh both from W&B first
```

A cache-mode run reproduces every table and figure byte for byte (checked 2026-09-30).

## 4. Results (test split, seeds 0–3)

| Condition | n | corr | demeaned corr | mse | avg rank |
|---|---|---|---|---|---|
| `PCA_PLS_learnable` (MSE) | 4 | 0.835 ± 0.005 | **0.078 ± 0.035** | 0.0136 | **0.686 ± 0.100** |
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
0.078 here (MSE-only, seeds 0–3) vs 0.092 over the 10 seeds of `sc_type_benchmark` (any loss).

### NodalGNN pilot (E0.5, seed 0, stopped early)

MSE-only tune of `NodalGNN` (`pilot/NodalGNN_mse_pilot.yml`, all nodal features, 4 trials packed per GPU), job
`18887602`, stopped by decision after 2 h 49 min with 10 of 12 trials started (no best-trial report). Per-trial
validation from Ray's progress logs: [`pilot/pilot_trials.csv`](pilot/pilot_trials.csv).

| | seed-0 val demeaned r |
|---|---|
| NodalGNN pilot, best of 10 trials (`20b6613a`, 114 epochs) | 0.011 |
| NodalGNN pilot, other trials | −0.007 to 0.011 |
| NodalMLP probes, best trial per variant | −0.006 to 0.012 |
| Null (`CrossModalPCA` SC) | 0.025 |
| `PCA_PLS_learnable` (MSE) | 0.117 |
| Gate to run seeds 1–3 | ≥ 0.045 |

Training demeaned r stayed at 0.003–0.007, so the model does not fit subject-specific FC even on training subjects.

## 5. Findings

1. **Every nodal model is at the null.** The four probes (demeaned corr 0.011–0.018, avg rank 0.50–0.54) and the
   nodal/graph references (0.016–0.023) sit within about one seed SD of the null (0.013) and far below `PCA_PLS_learnable`
   (0.078). The probes cannot be ranked against each other at n = 3–4.
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
5. **Tuning does not rescue NodalGNN either.** Its seed-0 pilot peaked at val 0.011 — below the null (0.025), level
   with the probes, far from the 0.045 gate — so seeds 1–3 were not run. Tuned `Chen2024GCN` is no better (test 0.017 ±
   0.007 over 10 seeds, best seed 0.026; raw corr 0.754 < the null's 0.824).
6. **The shared limitation is the formulation, not the decoder or the tuning.** Every nodal design here (NodalMLP with
   any decoder, Chen, NodalGNN) predicts full FC edges under MSE. The population-mean FC dominates that error, so the
   models learn it (or a worse copy) and stop: near-zero demeaned r on train as well as test. The linear family
   instead models the population mean explicitly and learns per-subject deviations (PCA/PLS scores), which is where
   all of its demeaned-r advantage lives.

## 6. Takeaways

- **NodalMLP-style models are closed** for SC→FC in this repo: no probe decoder, the MLP decoder, or hyperparameter
  setting moves them off the null.
- **The repo runs end to end after the v1 refactor:** launchers, tune/best-trial cycle, results tooling and runner
  pattern (E0.1–E0.3); one job environment with `torch_geometric` (v2:C6, C1); `reuse_actors=True` with packed trials is
  the efficient setting (E0.6).
- **Process lessons carried forward:** pilot before sweeping and pack small models (v2:D2); compare val scores within a
  seed only; compute importance on within-seed-centred scores; enforce the loss design per run (`run.py`).

## 7. Follow-up (not scheduled)

**Graph / nodal models that learn on top of the mean**, formulated like the PCA-family models: subtract the
train-split population-mean FC and have the GNN predict each subject's deviation (demeaned target) or a small set of
PCA scores / low-rank factors, rather than the full edge vector. This removes the shared pattern that currently
absorbs the MSE and puts the graph model on the same footing as `PCA_PLS_learnable`. An input-matched variant
(`NodalGNN` with `use_r2t: false`; SC-only node inputs) belongs in the same study. Tracked in spec v2 §5 (backlog).

## 6. Caveats

- **Coverage:** bilinear has n = 3; the NodalGNN pilot is seed 0 only (10 of 12 trials, stopped early); the NodalGNN
  row in the tables is its untuned default. NodalMLP MLP decoder has test metrics for 2 of its 4 seeds (the April runs for seeds 1–2 logged no test metrics; `tables/test_summary.md` shows
  `(n=4)` because it counts records, not values).
- **v2:C!1 (regularization not applied):** affects the `PCA_PLS_learnable` reference, whose sweeps trained with the
  default L2 = 1e-4 (v1:M5b). NodalMLP applies its own `l2_reg`, so the probe rows are unaffected.
- Best-trial re-runs score lower on val than their tune trial (for example bilinear seed 2: 0.049 in tuning, 0.025 on
  re-fit), consistent with Finding 3: the tune maximum is partly selection noise.
- **Input mismatch:** default `NodalGNN` node features include the region's tract profile (`r2t`, the `SC_r2t`
  information), so it is not input-matched to the SC-only rows.
- **Chen seed 3** comes from the one `node_feature_type: sc_row` run (`9cdzcuae`, test 0.025) — the
  `cov_projector_benchmark` snapshot keeps the best val run across node-feature types; the other seeds use the paper's
  identity features. It barely moves the mean (seeds 0–3: 0.0184 as tabled vs 0.0179 identity-only).
- `mlp_sc_rows` comes from an older schema and date window (created before 2026-04-27), so it is a control, not a
  matched condition.

Last updated at: 2026-09-30 EDT
