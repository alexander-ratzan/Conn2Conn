# Experiments Index

One row per experiment under `scripts/experiments/`. Each folder holds that experiment's code, config, launchers,
and its tables and PNG figures (all tracked). Truly bulky outputs
(checkpoints, large arrays) go to `results/experiments/<slug>/`. An experiment is documented by a write-up
named after its folder (`<slug>/<slug>.md`), or by its own notebook when that already describes it.

| Experiment | Status | Dates | What it asks | Documentation |
|---|---|---|---|---|
| [`sc_type_benchmark`](sc_type_benchmark/) | active | 2026-09-23 | How much does the structural input type (`SC`, `SC_r2t`, `SC+SC_r2t`) matter for SC→FC prediction across the linear family and Krakencoder? Config-driven; tracked `records.json` snapshot. | [`sc_type_benchmark.md`](sc_type_benchmark/sc_type_benchmark.md) |
| [`cov_projector_benchmark`](cov_projector_benchmark/) | active | 2026-09-23 | Which covariate feature sets help the PCA-PLS residual projector over the learnable baseline, and how do linear models, Krakencoder, and published deep models compare on SC input? Config-driven (labelled condition rows); tracked `records.json` snapshot. | [`cov_projector_benchmark.md`](cov_projector_benchmark/cov_projector_benchmark.md) |
| [`nodal_models_benchmark`](nodal_models_benchmark/) | closed | 2026-09-30 | How much of SC→FC can per-region (nodal) models recover — NodalMLP probe decoders, NodalMLP MLP decoder, NodalGNN, Chen GCN — against the null and the linear family, and which hyperparameters matter? Config-driven; tracked `records.json` + `trials.json` snapshots. | [`nodal_models_benchmark.md`](nodal_models_benchmark/nodal_models_benchmark.md) |
| [`composite_loss`](composite_loss/) | complete, both directions (spec v2 E1 SC → FC; E3 Phase D FC → SC) | 2026-09-30 → 2026-10-02 | Composite-loss protocol: with MSE fixed at 1, how do Var-match / Corr-eye / Demeaned corr-eye / Neighbor dist shape training dynamics and trade test demeaned-r against avg-rank? Grid v3 ([`grid_v3.yml`](composite_loss/grid_v3.yml), 29 combinations; v1 / v2 frozen); instances `linear_backbone` (primary), `pca_pls_learnable` (replicability, effects correlate 0.92 / 0.98), `pca_pls_covprojector`, `krakencoder`, each in `sc2fc/` and `fc2sc/`. Headline: SC → FC, Demeaned corr-eye is the strongest identity term (top-1 ×2, MSE unchanged) and raw Corr-eye is inert until it collapses predictions; FC → SC, no term helps (MSE-only best). | [`composite_loss.md`](composite_loss/composite_loss.md) |
| [`model_benchmark`](model_benchmark/) | E2.2 (MSE-only) complete, both directions; composite benchmark planned (spec v3 C1) | 2026-10-02 → 2026-10-04 | With every model tuned per seed on the same splits under MSE only, how do test Pearson r, demeaned r, avg rank and top-1 compare across model types, SC → FC and FC → SC? 12 models (+ Krakencoder MSE / paper-loss variants, test-retest ceiling), seeds 0–4, one campaign per model; autopilot-driven runs (111.5 GPU-h). Headline: the linear family leads and is tied SC → FC (demeaned r 0.092–0.098), graph / nodal models are at the null; FC → SC is easier, the covariate model's lead there is likely anatomy. | [`model_benchmark.md`](model_benchmark/model_benchmark.md) |
| [`task_fc2sc`](task_fc2sc/) | complete (spec v3 E0) | 2026-10-05 → 2026-10-06 | Which source FC condition (rest, rest session 1, 7 HCP tasks) best predicts SC with PCA-PLS learnable (MSE, FC → SC, Glasser), on a matched 917-subject cohort, seeds 0–4 (13.6 GPU-h)? Headline: rest is best on every individual-level metric (demeaned r 0.148); working memory is the best task (0.108); scan time explains much of the ordering but not all. | [`task_fc2sc.md`](task_fc2sc/task_fc2sc.md) |
| [`linear_backbone/geodesic`](linear_backbone/geodesic/) | exploratory | pre-April 2026 run | How do geodesic FC-distance identifiability metrics behave for the pure linear latent backbone (`CrossModal_linear_backbone`, formerly `LatentAttnMasked` with `residual_mode: none`)? | notebook: [`linear_backbone_geodesic_metrics.ipynb`](linear_backbone/geodesic/linear_backbone_geodesic_metrics.ipynb) |
| [`adel_summer_2026`](adel_summer_2026/) | parked off `main` (branch `adel-temp`) | 2026-05-25 → 2026-06-26 | Is cross-modal connectome prediction directional (FC→SC ≫ SC→FC), linearly saturated, and capped by FC for cognition? Pointer to work that lives on a branch. | [`adel_summer_2026.md`](adel_summer_2026/adel_summer_2026.md) |

## Adding an experiment

1. Create `scripts/experiments/<slug>/` (plain descriptive slug; avoid a `_context` suffix, which `.gitignore` ignores).
2. Notebooks start with the standard `REPO_ROOT` bootstrap; results/benchmark experiments copy
   `sc_type_benchmark/` (model × source grid) or `cov_projector_benchmark/` (labelled condition rows):
   `config.yml` + `run.py` + tracked `records.json`.
3. Write `<slug>/<slug>.md` unless the notebook already documents the question, setup, how to run, W&B ids, results,
   and caveats.
4. Add a row to this index.
