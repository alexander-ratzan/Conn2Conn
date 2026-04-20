# Conn2Conn — Agent Context

This is the fast technical context for AI agents and developers. Use this first for triage, then open code.

For user-facing usage/setup, see `README.md`.

---

## Mission

`Conn2Conn` predicts one connectome modality from another on HCP-derived data (default `SC -> FC`) and compares:
- closed-form baselines (PCA/PLS family)
- conditional Gaussian baseline
- learned variants (learnable map, covariate-conditioned residual projector, VAE, Sarwar MLP, Chen GCN)
- latent attention variant (`LatentAttnMasked`)
- nodal-feature GNN baseline (`NodalGNN`)
- precomputed Krakencoder baseline

The main evaluation axis is performance across `(model, source, shuffle_seed)` with W&B-backed experiment tracking.

---

## Fast Triage

When a task arrives, start in this order:
1. `main.py` for orchestration and experiment mode behavior.
2. `models/configs/<model>.yml` for ground-truth defaults/search space.
3. `models/registry.py` for config resolution and model construction.
4. `models/architectures/` for architecture details.
5. `results/results_scraper.py` for results aggregation logic.

If task is data/splits/covariates, read `data/hcp_dataset.py` immediately after `main.py`.

---

## Canonical Entrypoints

- `main.py`
  - single runs (closed-form vs learned)
  - Ray Tune sweeps (`--use_tune`)
  - best-trial rerun/report (`--report_best_after_tune`)
- `models/configs/*.yml`
  - one YAML per model variant; includes `learned`, `default`, `search_space`
- `models/registry.py`
  - `load_config`, `get_default_config`, `get_search_space`, `build_model`
  - source-dependent PCA dim normalization
- `models/utils.py`
  - shared inference helpers (`predict_from_loader`) and small batch utilities
- `models/architectures/`
  - actual model classes grouped by architecture family
  - shared architecture helpers live in `models/architectures/utils.py`
- `models/train/`
  - training orchestration, Lightning wrapper, losses, composite losses, and training plots
- `models/eval/`
  - evaluator, metrics, PCA analysis, FC distance utilities, visualization/markdown reporting
- `results/results_scraper.py`
  - W&B best-trial scraping + table builders
- `README.md`
  - user-level commands + workflow
- `notebooks/results_scrape/*.ipynb`
  - active experiment-table / plotting notebooks
- `data/data_viz.py`, `data/demeaned_viz.py`
  - reusable visualization helpers used by lightweight notebooks

---

## Core Execution Model (`main.py`)

`Sim.run_single()` dispatches by model type:
- learned model -> `_run_learned_single(...)`
- closed-form model -> `_run_closed_form_single(...)`

Common eval path:
- `_evaluate_model(...)`
- uses `predict_from_loader(...)` for normal models
- uses `predict_split(...)` for precomputed models with `is_precomputed=True`
- forwards optional `eval_kwargs` into `Evaluator.analyze_results(...)`

Tune path:
- `run_tune(...)` builds flat Ray param space
- tune runs are tagged in W&B with `ray_tune_id:{id}` and `tune`

Best-trial path:
- `report_best_tune_trial(...)` reruns best config in prod mode
- best-trial run is tagged with `best_trial_report`

---

## Model Inventory

Closed-form (`learned: false`):
- `CrossModalPCA`
- `CrossModal_PLS_SVD`
- `CrossModal_PCA_PLS`
- `Krakencoder_precomputed` (precomputed artifacts, no training; class `KrakencoderPrecomputed`)

Learned (`learned: true`):
- `CrossModal_PCA_PLS_learnable`
- `CrossModal_PCA_PLS_CovProjector`
- `CrossModalVAE`
- `LatentAttnMasked` (implemented in `models/architectures/latent_attention/latent_attn_masked.py`)
- `MaskedLatentPretrainer` (`models/architectures/latent_attention/masked_latent_pretrainer.py`) — **experimental, in development, not production**. SSL pretrainer for joint SC/FC PCA-latent reconstruction; transfers weights to `LatentAttnMasked` via `export_to_latent_attn_masked(downstream_model)`. Kept isolated: no cross-module changes in `lightning_module.py` / `trainer.py` / `main.py` should be made on its behalf. Dev harness lives in `latent_pretraining_test.ipynb` + `models/eval/pretrain_eval.py`. Known limitation: within-modality masked reconstruction in PC space is structurally degenerate (PCA decorrelates scores); working signal is cross-modal only. Composite-loss integration was scoped out to avoid interfering with the edge-composite path.
- `Sarwar2020MLP` (implemented in `models/architectures/sarwar2020_mlp.py`)
- `Chen2024GCN` (implemented in `models/architectures/graph_based/chen2024_gnn.py`)
- `NodalGNN` (implemented in `models/architectures/graph_based/nodal_gnn.py`)

Closed-form / hybrid special cases present in configs:
- `CrossModal_ConditionalGaussian` (implemented in `models/architectures/latent_attention/conditional_gaussian.py`)

### Special: `Krakencoder_precomputed`

CLI/YAML model ID: `Krakencoder_precomputed`  
Implementation class in `models/architectures/krakencoder_precomputed.py`: `KrakencoderPrecomputed`.

Behavior:
- loads per-seed inference `.mat` predictions
- stores full prediction matrix + FC targets from `base`
- serves split-specific `(preds, targets)` via `predict_split(split)`
- raises if `forward()` is called directly

Input assumptions:
- default artifact dir: `krakencoder/example_data/`
- file pattern: `mydata_kraken_seed{seed}_source_{parc}.{conn_type}.mat`
- supported source keys currently map to `SC` and `FC`

Naming note:
- config file is `models/configs/Krakencoder_precomputed.yml`
- YAML model name is `Krakencoder_precomputed`
- class name is `KrakencoderPrecomputed`
- `build_model()` resolves classes by exact attribute name
- keep this path validated in smoke tests after refactors

---

## Data + Partitioning

`data/hcp_dataset.py` provides:
- `HCP_Base` for loading modalities + covariates
- `HCP_Partition` for family-aware train/val/test partitioning
- support for composite sources like `SC+SC_r2t`

Parcel-level node feature pipeline:
- parcel volume and centroid CSVs are loaded per subject
- feature selectors are controlled by `volume_feature_type` and `centroid_feature_type`
- local `SC_r2t` node-wise tract features are appended to parcel node features
- final node feature layout is:
  - `[volume, centroid_x, centroid_y, centroid_z, sc_r2t_channels...]`

Data-loading modes in `HCP_Base`:
- `manual` (default): load raw source files
- `precomputed`: load cached `.npy` arrays from cache root

Cache defaults:
- cache root: `/scratch/asr655/neuroinformatics/Conn2Conn_data`
- `write_manual_cache=False` by default

Performance note:
- `HCP_Partition` now reuses shared base-level tensors across train/val/test partitions (avoids triple full-dataset tensor copies per split).
- `Sim` enables `expose_node_features=True` only for `NodalGNN`, so other models do not pay for unused batch payloads.

Covariates used by projector variants include demographics and FreeSurfer features; category collapsing for sparse `race_eth` occurs at partition time.

---

## W&B Schema (Important)

Two run types exist:

1. Tune trial runs
- tags include: model name, `tune`, `ray_tune_id:{id}`
- config shape: flat/dot-notation keys

2. Best-trial prod runs (primary reporting target)
- tags include: model name, `prod`, `best_trial_report`, `ray_tune_id:{id}`
- config shape: nested (`data`, `model`, `trainer`) + top-level metadata
- summary includes train/val metrics and `eval_test/*` metrics

3. Direct prod runs (used selectively)
- tags include: model name, `prod`
- used for notebook-driven or no-tune evaluation flows
- scraper can now opt specific models into this fetch path (currently used for `NodalGNN` in cov/deep comparison notebooks)

Do not use W&B `group` as sweep identity. Use `ray_tune_id:{id}`.

---

## Results Scraper (`results/results_scraper.py`)

Status: active development; already functional for best-trial aggregation.

Primary API:
- `fetch_best_trial_runs(model_name)`
- `fetch_direct_prod_runs(model_name)`
- `parse_run_record(run)`
- `build_experiment_records(models, sources, seeds, ...)`
- `records_to_df(records)`
- `build_status_table(records)`
- `build_metric_table(records, metric=...)`
- `enrich_records_with_local(records)`
- `load_local_artifact_df(records)`

Key behaviors:
- handles nested and flat W&B config formats
- fallback for legacy runs via local checkpoint config
- resolves local artifact directories under `results/ray_results/`
- can mix best-trial and direct-prod fetch paths per model via `direct_prod_models`
- includes experiment-specific helpers for SC-type and covariate/deep-model summary tables and plots

### Testing Priorities For Scraper

When editing scraper logic, validate:
1. run-type filtering (`best_trial_report` only for core tables)
2. config extraction order for source/seed
3. duplicate run deduping (best validation metric wins for the requested selector)
4. metric extraction (`eval_test/` prefix stripping)
5. missing-cell fill across full model x source x seed grid
6. local enrichment merge precedence (W&B metrics should win base keys)

Suggested quick smoke workflow:
- build records for 1–2 models, all three sources, seeds `[0,1]`
- check `build_status_table` shape/content
- check `build_metric_table(metric="demeaned_pearson")`
- run `enrich_records_with_local` and verify added non-W&B fields

---

## Evaluation Notes

- `Evaluator._metrics` / `base_metrics` are the default scalar summary surface used for returned `Sim` metrics and `eval_test/*` W&B logging.
- Geodesic FC distance support lives in `models/eval/fc_distance.py`.
  - `affine_invariant` is the metric most faithful to the geometry-aware FC paper and the legacy `GeneEx2Conn/models/metrics/distance_FC.py` implementation.
  - `log_euclidean` is kept as a faster SPD-aware alternative.
- `Evaluator.analyze_results(...)` can optionally append exploratory geodesic summary metrics into `base_metrics` via:
  - `include_geodesic_metrics=True`
  - `geodesic_metric_method=...`
  - `geodesic_metric_demeaned=True|False`
  - related `geodesic_metric_*` kwargs
- `Sim.run_single(..., eval_kwargs={...})` is the intended way to opt into those exploratory metrics from notebooks or scripts without changing the default reporting path.

---

## Config System Notes (`models/registry.py`)

- `FLAT_METADATA_KEYS` are logging/display keys and must not reach model constructors.
- `l1_reg`/`l2_reg` are recombined into `l1_l2_tuple` in `build_model`.
- source-dependent PCA dims are normalized by `resolve_source_dependent_config`.
- YAML `search_space` is converted to Ray Tune objects by `search_space_to_tune`.

Trainer loss config currently supports standard atomic loss names plus `loss_type: composite`.
Composite losses use structured `loss_terms`, for example:

```yaml
trainer:
  loss_type: composite
  loss_terms:
    - {name: mse, weight: 1.0}
    - {name: neidist, weight: 0.25}
  loss_normalize: ema
```

Regularization remains model-owned through `model.get_reg_loss()` and is added separately by the Lightning module.

---

## HPC / Workflow Guardrails

- Prefer targeted reads over recursive scans.
- Avoid heavy artifact trees unless task explicitly needs them:
  - `results/ray_results/`
  - `results/ray_checkpoints/`
  - `wandb/`
  - large notebooks
- Use SLURM scripts for large jobs; avoid long compute on login nodes.

---

## High-Risk Gotchas

1. W&B `group` is not sweep-level identity in this repo.
2. Tune configs are flat; best-trial configs are nested.
3. Multi-source PCA settings may be scalar or dict; resolve before model build.
4. Precomputed model path bypasses DataLoader-based prediction.
5. Krakencoder config/class naming should be verified before relying on automated runs.
6. `Chen2024GCN` requires `torch-geometric` in the runtime environment.
7. `NodalGNN` also requires `torch-geometric`.
8. `precomputed` data_load_mode only works when cache files already exist at the resolved cache root.
9. Active multi-seed SLURM launchers live under `sbatch/<ModelName>/`; older root-level wrappers should not be treated as canonical.
10. Sbatch scripts and input-feature subset configs are still duplicated by experiment variant; a future manifest/launcher layer should move grids out of copied shell/YAML files.

---

## Minimal Task Recipes

Add/modify model:
1. edit or add a class under `models/architectures/`
2. add/update YAML in `models/configs/`
3. ensure `build_model()` can resolve class name
4. run one dev/prod dry run with fixed seed

Add/modify loss behavior:
1. add atomic terms or factories in `models/train/loss.py`
2. route training behavior through `models/train/lightning_module.py`
3. keep metric-only calculations in `models/eval/metrics.py` unless they are part of the differentiable training objective
4. prefer `loss_type: composite` with structured `loss_terms` for weighted multi-term objectives

For `NodalGNN` specifically:
1. node features come from `batch["node_features"]`, not from `cov`
2. `CrossModalLightningModule`, `predict_from_loader`, and loss/eval helpers already know how to pass `node_features`
3. ablations are controlled in config via `use_volume`, `use_spatial`, and `use_r2t`

Update experiment reporting:
1. patch `results/results_scraper.py`
2. run smoke table builds (`records_to_df`, `build_status_table`, `build_metric_table`)
3. verify local enrichment still merges correctly

Debug missing results cell:
1. confirm best-trial run exists in W&B with expected tags
2. inspect parsed source/seed path (nested vs flat fallback)
3. verify model/source/seed is included in requested grid

## Recent Changes
- Analysis/figure schematics and context images now live under `context_packages/schematics/` (for example `SC_results.png`, `cov_dl_results.PNG`, `matrix_gif.jpg`, and `demeaned_plot/`).
- Results-analysis workflow now centers on:
  - `notebooks/results_scrape/scrape_SCtype_results.ipynb`
  - `notebooks/results_scrape/scrape_covtype_results.ipynb`
- Lightweight visualization helpers were moved into:
  - `data/data_viz.py`
  - `data/demeaned_viz.py`
- `notebooks/kraken/track_krakencoder_model.ipynb` can log W&B runs and optionally save local markdown reports under `results/local_results/Krakencoder_precomputed/`.
- Model code now lives under `models/architectures/`, training code under `models/train/`, and evaluation/reporting code under `models/eval/`.
- Backward-compatibility shims for old top-level model/train/eval files are intentionally removed.

Last updated at: 2026-04-17 19:11:00 EDT
