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
| [`composite_loss`](composite_loss/) | grid v2 complete (spec v2 E1) | 2026-09-30 → 2026-10-01 | Composite-loss protocol: with MSE fixed at 1, how do `varmatch` / `correye` / `correye_dm` / `neidist` shape training dynamics and trade test demeaned-r against avg-rank? Grid v2 ([`grid_v2.yml`](composite_loss/grid_v2.yml), 32 combinations; v1 frozen); instances `linear_backbone` (primary) and `pca_pls_learnable` (replicability). Headline: `correye_dm` is the strongest identity term, ahead of `neidist`; raw `correye` is inert until it collapses predictions. | [`composite_loss.md`](composite_loss/composite_loss.md) |
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
