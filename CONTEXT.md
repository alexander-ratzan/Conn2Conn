# Conn2Conn — Agent Context

This is the fast technical context for AI agents and developers. Use this first for triage, then open code.

For user-facing usage/setup, see `README.md`.

---

## Mission

`Conn2Conn` predicts one connectome modality from another on HCP-derived data (default `SC -> FC`) and compares:
linear, latent, graph / nodal and deep models (full list: Model Inventory below; user-facing table: `README.md`
§Models) in both directions (`SC → FC`, `FC → SC`), on two parcellations, and with FC from rest or HCP task
conditions. Spec v4 adds latent / generative models of FC across conditions (multiview CCA latent, conditional flows)
and regional BOLD timeseries as pre-modeling infrastructure for more samples per subject.

The main evaluation axis is performance across `(model, direction, source / FC condition, shuffle_seed)`.
Hyperparameters are tuned on val demeaned r, and test is reported. Current plans: `context_packages/repo_spec_docs/spec_doc_v4.md` (v3 closed 2026-10-21).

---

## Fast Triage

When a task arrives, start in this order:
1. `main.py` for orchestration and experiment mode behavior.
2. `models/configs/<family>/<model>.yml` for ground-truth defaults/search space. Layout: family folders
   (`null_ceiling/`, `linear/`, `latent/`, `graph_nodal/`, `deep/`), `variants/<family>/` for input / covariate /
   decoder variants, `benchmark/<campaign>/` for experiment-generated configs (by path only). `models/registry.py`
   finds a config by name; old flat paths (`models/configs/<name>.yml`) still resolve with a notice.
3. `models/registry.py` for config resolution and model construction.
4. `models/architectures/` for architecture details.
5. `scripts/results_utils/` for results aggregation logic (`records.py` → `tables.py` → `plots.py`).

If task is data/splits/covariates, read `data/hcp_dataset.py` immediately after `main.py`.

---

## Canonical Entrypoints

- `main.py`
  - single runs (closed-form vs learned)
  - Ray Tune sweeps (`--use_tune`)
  - best-trial rerun/report (`--report_best_after_tune`)
- `models/configs/**/*.yml` (family folders + `variants/`; `benchmark/` is experiment-generated)
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
- `scripts/results_utils/`
  - W&B/Ray results scraping → tables → figures (see "Results Utilities" below)
- `scripts/sbatch/<ModelName>/`
  - legacy per-model tune arrays (spec v3 I2 replaces them with one launcher + rosters); `scripts/sbatch/checks/`
    (verification scripts), `scripts/sbatch/data/` (FC cache builder). Campaign launchers live in their experiment folder.
- `README.md`
  - user-level commands + workflow
- `scripts/experiments/` (index: `experiments_index.md`); two patterns:
  - W&B-scraped results experiments (`sc_type_benchmark/`, `covariate_projector/cov_projector_benchmark/`, `nodal_models_benchmark/`):
    `config.yml` + `run.py` + tracked `records.json`
  - SLURM campaigns scraped from task logs: `model_benchmark/` (MSE benchmark, both directions; `build_configs.py`,
    `submit.py`, `autopilot.py`, `run.py`; composite-loss campaign with `MB_CAMPAIGN_CONFIG=config_composite_loss.yml`:
    `composite_scales.py`, `build_composite_configs.py`, `composite_vs_mse.py`, outputs `composite_loss/`),
    `covariate_projector/covariate_ablation/` (spec v3 E1: covariate families, `run_cell.py`, `node_size.py`), `composite_loss/` (`protocol.py`, `report.py`, `compare.py`; instances
    `<model>/{sc2fc,fc2sc}/`; viewer `viewers/cross_model/`), `task_bidirectional_bench/` (FC condition × seed × direction × parcellation; `make_configs.py`, `submit.py`, `run.py`)
- `scripts/data_viewers/` (self-contained HTML viewers; layout and rules under "Viewers" in Repo Layout Conventions)
  - `viewer_kit/` (shared base: `Viewer` page builder, `kit.css`, `kit.js`, `smoke_test.js`), `timeseries_viewer/`
    (regional timeseries, window FC and metrics from `Conn2Conn_data/timeseries`), `multitask_latent_viewer/` (subjects'
    FC encodings in the E1 multitask CCA latent vs pooled PCA codes), `deviation_similarity_viewer/` (per condition pair:
    own-subject vs other-subject correlation of FC deviations, rank; same quantities as `notebooks/EDA/FC_deviation_similarity.py`)
- `scripts/notebooks/model_overviews/*.ipynb`
  - conceptual onboarding notebooks for PCA/PLS, conditional Gaussian, latent-attention, masked pretraining, and nodal baselines
- `scripts/notebooks/model_testing/*.ipynb`
  - per-model smoke-test / dev notebooks
- `scripts/notebooks/{EDA,kraken}/*.ipynb`
  - data exploration; Krakencoder tracking/eval
- `context_packages/`
  - agent/human reference material: `modeling/` (model design notes; v4 reference `modeling/FM+tx_integrate/modeling_directions.md`), `repo_spec_docs/` (refactor specs; `markdown_refs/` handoff notes), `schematics/`, `T1/`
- `data/timeseries.py`, `data/timeseries_metrics.py`
  - regional BOLD timeseries cache reader (`TimeseriesCache`), run-bounded windows (`window_starts`), windowed FC
    (`window_fc`, `shrink`); per-region intrinsic timescale and multiscale sample entropy (see Data + Partitioning)
- `data/data_caching/`
  - cache builders: `build_fc_cache.py` (FC npy caches, combined and per run; exclusions TSV), `build_ts_cache.py`
    (timeseries cache, validated against fc/); sbatch arrays in `scripts/sbatch/data/`
- `models/architectures/multiview_cca.py`, `models/architectures/crossmodal_flow_matching.py`
  - spec v4 library (not registry models): MAXVAR multiview CCA (`maxvar`, `ridge_gcv`); `CodeSpace` (pooled block-PCA
    FC basis with equal-variance block scaling + SC basis), `TokenBackbone`, `integrate` (flow ODE); used by
    `scripts/experiments/{cca_latent_bench,joint_flow_matching}/`
- `data/data_viz.py`, `data/demeaned_viz.py`
  - reusable visualization helpers used by lightweight notebooks; `data_viz` also has edgewise regressions with held-out
    prediction (`predict_edgewise_multifeature_regression`, `evaluate_edgewise_predictions`) and task-FC signatures
    (`compute_task_signatures`, `plot_task_signature_*`)
- `scripts/maintenance/prune_ray_storage.sh`
  - prunes `results/ray_tmp/` sessions to their logs and old `results/ray_checkpoints/` weights (dry run by default)

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
- `CrossModal_ConditionalGaussian` (`models/architectures/crossmodal_conditional_gaussian.py`; conditional mean in PCA space or ridge in edge space, optional covariates)
- `CrossModal_PCA_PLS_TaskResid` (`models/architectures/crossmodal_pca_pls.py`, spec v3 E2.2): sources `FC+FC_<task>…`; per task a ridge (GCV, `fit_rest_residualizer`) predicts task FC from the subject's `k_rest` rest scores, the block is the train PCA (`k_task`) of the residual; PLS to SC PCs. Args `k_rest`, `k_task`, `n_components_pls`, `n_components_pca_target`, `resid_alphas`.
- `Krakencoder_precomputed` (precomputed artifacts, no training; class `KrakencoderPrecomputed`)
- `Krakencoder` (`learned: false` in the Lightning sense: retrained per seed outside `main.py` by the vendored upstream trainer, then served like `Krakencoder_precomputed`; see below)
- `TestRetestPrecomputed` (test-retest oracle, loads session 1/2 cached connectomes; class in `models/architectures/test_retest_precomputed.py`)

Learned (`learned: true`):
- `CrossModal_PCA_PLS_learnable` (flags include `mid_bias` and `zscore_pca_scores`; defaults reproduce the original model)
- `CrossModal_linear_backbone` (`models/architectures/crossmodal_pca_pls.py`) — thin `_learnable` subclass: frozen source PCA encoder → learned affine k×k latent map (`W_mid`, `mid_bias`) → frozen target PCA decoder. Pins the `_learnable` flags; free keys are `n_components_pca_source` (target tied to it), `zscore_pca_scores`, `l1_reg`/`l2_reg`. Formerly `LatentAttnMasked` with `residual_mode: none`. A fast, strong probe for loss/regularization studies.
- `CrossModal_PCA_PLS_CovProjector`
- `CrossModalVAE`
- `LatentAttnMasked` (implemented in `models/architectures/latent_attention/latent_attn_masked.py`); `residual_mode` ∈ {`attention_only`, `pls_residual`, `linear_residual`}. `none` was removed and raises, pointing to `CrossModal_linear_backbone`.
- `MaskedLatentPretrainer` (`models/architectures/latent_attention/masked_latent_pretrainer.py`) — **experimental, in development, not production**. SSL pretrainer for joint SC/FC PCA-latent reconstruction; transfers weights to `LatentAttnMasked` via `export_to_latent_attn_masked(downstream_model)`. Kept isolated: no cross-module changes in `lightning_module.py` / `trainer.py` / `main.py` should be made on its behalf. Dev harness: `scripts/notebooks/model_overviews/masked_attn_pretraining_overview.ipynb`.
- `MaskedMLPPretrainer` (`models/architectures/latent_attention/masked_mlp_pretrainer.py`) — **experimental**. SSL pretrainer variant of the masked-latent path using linear, low-rank-linear, or nonlinear MLP encoders plus configurable SC/FC masking. Configs: `latent/MaskedMLPPretrainer.yml` + `variants/latent/MaskedMLPPretrainer_{linear,nonlinear,mask_grid}.yml`; sbatch launchers live under `scripts/sbatch/MaskedMLPPretrainer/`.
- `Sarwar2020MLP` (implemented in `models/architectures/sarwar2020_mlp.py`)
- `Chen2024GCN` (implemented in `models/architectures/graph_based/chen2024_gnn.py`)
- `NodalGNN` (implemented in `models/architectures/graph_based/nodal_gnn.py`)
- `NodalMLP` (implemented in `models/architectures/graph_based/nodal_mlp.py`) — graph-free node/edge baseline. It can use anatomical parcel features (`volume`, `spatial`, `SC_r2t`), subject SC rows, or spectral SC eigenvectors; decoder variants include `mlp`, `dot`, `bilinear`, `diag_bilinear`, and `linear_beta`. Configs: `graph_nodal/NodalMLP.yml` + `variants/graph_nodal/NodalMLP_{spatial,all_features,dot,bilinear,linear_beta,spectral}.yml`; sbatch launchers live under `scripts/sbatch/NodalMLP/`.

Input / covariate / decoder variants of these models: `models/configs/variants/<family>/` (e.g. `SC_r2t` and
`SC+SC_r2t` sources, CovProjector covariate sets, NodalMLP decoders, MaskedMLP backbones).

### Special: `Krakencoder_precomputed` and `Krakencoder` (retrained)

CLI/YAML model IDs: `Krakencoder_precomputed` (cached March 2026 predictions) and `Krakencoder` (retrained in the repo).
Both map to `KrakencoderPrecomputed` in the package `models/architectures/krakencoder/` (`precomputed.py`; the
package also holds `vendor/` = upstream Krakencoder, unmodified, and `retrain.py`).

Behavior:
- loads the per-seed inference `.mat` for the run's **source** modality and takes `predicted_alltypes[source][target]`,
  so `SC → FC` and `FC → SC` both work (every file holds all input → output types)
- stores the prediction matrix + the matching targets (`fc_` or `sc_upper_triangles`) from `base`
- serves split-specific `(preds, targets)` via `predict_split(split)`; raises if `forward()` is called directly

Prediction files (`krakencoder_prediction_path`):
- `Krakencoder_precomputed` (no `tag`): `krakencoder_experimental/example_data/mydata_kraken_seed{seed}_source_{parc}.{SC|FC}.mat`
  (gitignored local Krakencoder copy = upstream + demeaned-MSE loss; kept as reference)
- `Krakencoder` (`tag` in `models/configs/deep/Krakencoder.yml`): `results/cache/krakencoder/<tag>/seed{seed}/predictions_source_{parc}.{SC|FC}.mat`,
  written by `python -m models.architectures.krakencoder.retrain` (vendored upstream `vendor/`, commit `b57e39c`,
  unmodified; `_vendor_entry.py` pins the import and accepts our non-upstream flavor names). Inputs are built from
  `HCP_Base` (canonical subject order = `metadata_df["subject"]`; `sc_/fc_subject_ids` keep raw load order) with the
  per-seed `trainvaltest_partition_indices`. SLURM: `scripts/sbatch/Krakencoder/train_array_krakencoder_seeds.sh`.
- `checkpoint_eval.py` scores every saved checkpoint in-process (our metrics + loss terms, both directions) →
  `epoch_history.csv`; recipe overrides via `retrain.py --set KEY=VALUE` (e.g. `losstype`, `batch_size`, `random_seed`).
  Loss-weight grid instance: `scripts/experiments/composite_loss/krakencoder/` (`grid_runner.py`, packed fits).

Naming note:
- config file is `models/configs/deep/Krakencoder_precomputed.yml`
- YAML model name is `Krakencoder_precomputed`
- class name is `KrakencoderPrecomputed`
- `build_model()` resolves classes by exact attribute name
- keep this path validated in smoke tests after refactors

---

## Data + Partitioning

`data/hcp_dataset.py` provides:
- `HCP_Base` for loading modalities + covariates
- `HCP_Partition` exposing one split (family-preserving splits come from `dataset_utils.load_metadata` / `generate_train_val_test`)
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

Performance / payload notes:
- `HCP_Partition` reuses shared base-level tensors across train/val/test partitions (avoids triple full-dataset tensor copies per split).
- `Sim` enables `expose_node_features=True` for `NodalGNN` and `NodalMLP`.
- `Sim` enables `expose_sc_matrix=True` for `NodalMLP` so SC-row and spectral encoders can reconstruct dense subject SC matrices from source edges.
- Other models do not pay for unused node-feature or dense-SC batch payloads.

Covariates used by projector variants include demographics and FreeSurfer features; category collapsing for sparse `race_eth` occurs at metadata load (training counts < 10 → `Other`).

FC conditions (spec v2 I1.4–I1.5, v3 E0; user-facing summary in `README.md` → Data Structure and Splits):
- `fc_conditions=[...]` loads task caches (`FC_TASK_CONDITIONS` in `data/dataset_utils.py`: emotion, gambling,
  language, motor, relational, social, wm; combined LR+RL relmats) in either load mode; `fc_condition_matrices`
  adds dense matrices. `expose_fc_sessions` loads rest session 1 / 2.
- Every loaded condition joins the canonical intersection (957 → 917 Glasser / 916 4S456 with all tasks).
- `fc_source_condition` (`rest_S1`, `rest_S2`, or a loaded task) rebinds `fc_upper_triangles` / `fc_matrices`
  **before** the train-split PCA, so it is "FC" for the whole run (source or target).
- These are `data:` keys only (`models/registry.py` `DATA_KEYS`, `main.py` `HCP_BASE_DATA_KEYS`); no CLI flags.
- `FC_<task>` source modalities (spec v3 I5; `FC_CONDITION_MODALITIES` in `data/hcp_dataset.py`): source only, need the task
  in `fc_conditions`, cannot be combined with `fc_source_condition`. `HCP_Base.fc_condition_train_pca[task]` holds the
  train PCA (built only for requested sources); `HCP_Partition` serves the edges; `get_modality_data` maps them.
- Split indices are positions in the post-intersection `metadata_df` (fixed 2026-10-04; positions in the
  pre-intersection list were wrong whenever the intersection dropped subjects).
- Splits are family-preserving for every seed; seed 0 is the `participants.tsv` split (683 / 79 / 195 of 957).

Timeseries and per-run FC (2026-10-26; not wired into `HCP_Base`, read directly):
- `Conn2Conn_data/timeseries/parc-{P}_hemi-both[_task-{T}]/` (`data/timeseries.py`, README in the folder): xcp-d 0.9.1
  parcel means, every subject's runs concatenated (`timeseries.npy` memmap + `offsets.npy` + `runs.tsv`), TR 0.72 s,
  band-pass filtered (~0.01-0.08 Hz), not z-scored, no censoring. Rest N = 1090 (Glasser). `corr(rows)` reproduces
  every fc/ combined matrix to float32 rounding (~3e-8), so FC processing = timeseries processing.
- Per-run FC: `Conn2Conn_data/fc/..._dir-{LR,RL}_run-{1,2}` (rest 4 runs, tasks LR1 / RL1); `build_fc_cache.py --run`.
- Windowed FC and regional metrics are computed on the fly, never stored; windows never span a run boundary.
  Metrics (`data/timeseries_metrics.py`): timescale = TR × ACF area before its first value ≤ 0 (Parkes et al.); multiscale
  sample entropy m = 2, r = 0.3 SD, scales 1-13, MSE area (Wijesinghe et al. 2026 / LOFT). No task EVs or motion files on
  disk: task designs are not regressed out and windows cannot be motion-screened.
- Viewer: `scripts/data_viewers/timeseries_viewer/`.

---

## W&B Schema (Important)

Three run types exist:

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

## Results Utilities (`scripts/results_utils/`)

Status: active development; functional for best-trial aggregation. Notebooks import from the package
level (`from scripts.results_utils import ...`) so files inside the package can be reorganized freely.

Modules (one-way dependencies `plots → tables → records`):
- `records.py` — shared constants (`REPO_ROOT` derived from `__file__`, `RESULTS_ROOT`, `RAY_RESULTS_DIR`,
  `RAY_CHECKPOINTS_DIR`, `LOCAL_RESULTS_DIR`, `WANDB_PROJECT`/`WANDB_ENTITY`), metric/model display vocabulary,
  W&B fetchers (`wandb_api`, `fetch_best_trial_runs`, `fetch_direct_prod_runs`, `count_tune_trials_for_run`),
  `RunRecord`, `parse_run_record`, `build_experiment_records(_covtype)`, `records_to_df`,
  `save_records_cache` / `load_records_cache`, `enrich_records_with_local`, `load_local_artifact_df`
- `tables.py` — `build_status_table`, `build_metric_table`, `build_covtype_status_table`,
  `build_covtype_metric_table`, `build_sc_type_summary_table`, `build_cov_dl_seed_df`, `build_cov_dl_summary_table`
- `plots.py` — `plot_source_metric_bars`, `plot_model_metric_scatter`, `plot_cov_dl_metric_bars`,
  `plot_cov_dl_global_metric_panels`; config-driven helpers `FIGURE_TYPES` (figure-type registry),
  `render_figure(spec, records, **defaults)` (validates spec keys against the plotting function),
  `figure_name`, `save_figure` (PNG at 300 dpi by default; other formats only on request)
- `runner.py` — shared plumbing for config-driven `scripts/experiments/<slug>/run.py`: CLI (`arg_parser`),
  `load_config`, the records-cache contract (`load_cached` exits if the cache is missing or does not cover the
  config; `scrape_metadata`), `write_table`, `write_figures(cfg, out_dir, resolve)`, `write_manifest`
- `local_results.py` — loaders/plots for notebook-written `results/local_results/`
- `loss_grid.py` — composite-loss protocol helpers (`find_stage1_runs`, `consensus_config`, `reference_scales`,
  `combo_loss_trainer`, `run_one`, `collect_runs`); not re-exported
- `optuna_importance.py` — hparam importance from W&B tune trials (not re-exported; needs optuna):
  `python -m scripts.results_utils.optuna_importance --help`
- `__init__.py` — re-exports the notebook-facing API; `reload()` reloads submodules in dependency order for
  notebook iteration. `__all__` excludes submodule names so `import *` never clobbers a `records` variable.

Key behaviors:
- handles nested and flat W&B config formats
- fallback for legacy runs via local tune-trial config (`results/ray_checkpoints/{model}_tune_{id}/{trial}/final/config.json`)
- resolves local best-trial artifact directories under `results/ray_results/{model}/{trial_id}/`
- can mix best-trial and direct-prod fetch paths per model via `direct_prod_models`
- fetches runs through `query_runs` (`lazy=False`): wandb ≥0.2x `Api.runs()` lazy-loads runs with an empty `config`
- attributes each run to the model it was fetched for (`parse_run_record(model_name=...)`), not its first tag
- includes experiment-specific helpers for SC-type and covariate/deep-model summary tables and plots
- reads results from the W&B cloud only (`wandb.Api()`); local W&B run folders are never used

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
- `Evaluator.analyze_results(...)` (defined in the `models/eval/evaluator_viz_markdown.py` mixin) can optionally append exploratory geodesic summary metrics into `base_metrics` via:
  - `include_geodesic_metrics=True`
  - `geodesic_metric_method=...`
  - `geodesic_metric_demeaned=True|False`
  - related `geodesic_metric_*` kwargs
- `Sim.run_single(..., eval_kwargs={...})` is the intended way to opt into those exploratory metrics from notebooks or scripts without changing the default reporting path.

---

## Config System Notes (`models/registry.py`)

- `FLAT_METADATA_KEYS` are logging/display keys and must not reach model constructors.
- Regularization keys: every learned model takes `l1_reg` / `l2_reg` (YAML and search). `build_model` maps them to `l1_l2_tuple`, and they override any `l1_l2_tuple` so a sampled value is never replaced by a default. Which parameters are penalized is model-owned (`get_reg_loss()`). Exception: `Chen2024GCN` keeps the paper's plain-norm penalty driven by `l2_reg` and rejects `l1_reg`.
- `search_space_to_tune` supports `choice`, `grid`, `loguniform`, `uniform`, and raises on anything else (unsupported types used to be dropped silently).
- Keys starting with `loss_weight_` / `loss_kwarg_` are trainer keys (`registry.is_trainer_key`) and never reach model constructors.
- source-dependent PCA dims are normalized by `resolve_source_dependent_config`.
- Name lookup searches `models/configs/` except `benchmark/` and raises on duplicate names; old flat paths resolve
  with a `[config] … moved to …` notice.

Training loss (`models/train/loss.py`): every edge-space model uses `loss_type: composite`. The only other types are the latent
losses `latent_mse` / `latent_weighted_mse` (PCA-score reconstruction). `resolve_loss_config()` is the single entry point: it
fills defaults, applies overrides, validates, and adds `loss_signature`. A config with no loss keys is plain MSE.

```yaml
trainer:
  loss_type: composite
  loss_terms:                       # default when omitted: [mse]
    - {name: mse, weight: 1.0}      # anchor; not searched
    - {name: neidist, weight: 0.0}  # weight 0 drops the term
  loss_normalize: auto              # auto: none for one active term or any fixed scale, ema otherwise
search_space:
  loss_weight_neidist: {type: choice, values: [0.0, 0.25, 0.5, 1.0]}   # flat, Optuna-safe
  loss_kwarg_pairwise_corr__corr_target: {type: uniform, lower: 0.3, upper: 0.9}
```

- Terms (`VALID_TERMS`): `mse`, `varmatch`, `correye`, `correye_dm` (Corr-eye on demeaned predictions; needs the
  target mean like `demeaned_mse`), `neidist` (signed; kwarg `margin`), `demeaned_mse` (needs `base`),
  `pairwise_corr` (Sarwar; kwarg `corr_target`), `kld` (needs the model's `mu`/`logvar`, raises otherwise).
- Fixed reference scales (spec v2 D6): any term takes kwarg `scale` (> 0) and contributes `weight * raw / scale`.
  Scales replace `ema` (combining them raises). The composite-loss protocol (`scripts/results_utils/loss_grid.py`)
  always sets them; a plain multi-term config without scales still falls back to `ema`.
- `loss_monitor_terms`: extra terms computed and logged, never optimized.
- `ema` estimates each term's scale from `|raw|` during `loss_scale_warmup_steps` training steps, then freezes it.
  Scales are logged as `*_loss_ref_*`.
- `loss_signature` (e.g. `mse+0.5*neidist`) is logged on prod, best-trial and Tune-trial runs; group W&B runs by it.
- Per-term metrics: `{train,val}_loss_{raw,term,weighted,ref}_<term>` (Lightning). Tune trials receive
  `train_loss_raw_*` and `val_loss_{raw,weighted,ref}_*`.
- `Sarwar2020MLP` (`[mse, pairwise_corr]`) and `CrossModalVAE` (`[mse, kld]`) use `loss_normalize: none`, so the weight
  *is* the paper weight / β. `weighted_mse(α)` is `[mse: α, demeaned_mse: 1−α]` under `ema` with warmup 1.

Regularization remains model-owned through `model.get_reg_loss()` and is added separately by the Lightning module.

---

## Repo Layout Conventions

- **Library vs scripts vs artifacts.** Core library + entrypoint: `main.py`, `data/`, `models/`. Everything that
  uses the library lives under `scripts/` (tracked). `results/` holds generated artifacts only and is gitignored,
  except the two tracked reports `results/local_results/{Krakencoder_precomputed,test_structured_loss_model}/`.
- **Notebook bootstrap.** Every notebook's first code cell resolves the repo root by walking up to `main.py`,
  so notebooks keep working when moved within the repo:
  ```python
  import sys
  from pathlib import Path

  REPO_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "main.py").exists())
  if str(REPO_ROOT) not in sys.path:
      sys.path.insert(0, str(REPO_ROOT))
  ```
  Write repo paths as `REPO_ROOT / "results/..."`, never cwd-relative or absolute `/scratch/...`. Raises a bare
  `StopIteration` if the kernel cwd is outside the repo.
- **Experiments.** `scripts/experiments/<slug>/` holds an experiment's code, config, launchers, small outputs, and
  its write-up `<slug>.md` (question, setup, how to run, W&B ids, results, caveats; descriptive name, not
  `README.md`). Skip the write-up when the experiment's notebook already documents it. Bulky outputs go to
  `results/cache/experiments/<slug>/`. Standard layout (new experiments; older ones are converted when touched):
  `<slug>.md`, `config.yml`, entry scripts (`run.py`, ...), `launch/` (sbatch launchers), `checks/` (verification
  scripts and audits), `results/` (raw per-seed / per-run records), `tables/`, `figures/` (PNG), `viewers/<name>/`
  (interactive pages, see below), all at the same depth. Reference: `cca_latent_bench/`. `scripts/experiments/experiments_index.md` lists every experiment (status,
  dates, question, documentation link). Avoid folder names ending in `_context` (a global `.gitignore` rule
  ignores `*_context/`).
- **Viewers.** Self-contained HTML pages (no CDN, data embedded), one folder each holding the builder, the template and
  the built page side by side: `scripts/data_viewers/<name>/` for data viewers, `scripts/experiments/<slug>/viewers/<name>/`
  for an experiment's results. `build_<name>.py` collects the data and calls `scripts.data_viewers.viewer_kit.Viewer(__file__).write(meta, sets)`;
  `<name>.template` is the page (kit markers `<!--KIT:HEAD-->`, `<!--KIT:TOP-->`, `<!--KIT:DATA-->`, a `kit-template-note`
  line); the built `<name>[_<variant>].html` and its data cache `*_data.json` are gitignored (only code is tracked).
  `--from-cache` rebuilds a page from the cache, `--smoke` runs its JS under node (cuda image) with a DOM stub. Pages
  are opened in Cursor with "Live Preview: Show Preview" (open the `.html`, never the `.template`). Claude skill:
  `Conn2ConnWorkspace/.claude/skills/viewer/`.
- **Results experiments (config-driven).** Two templates: `scripts/experiments/sc_type_benchmark/` (a model × source
  grid; record-level figure types `source_bars`, `metric_scatter`) and `scripts/experiments/covariate_projector/cov_projector_benchmark/`
  (labelled condition `rows` + ordered `row_groups` with label overrides; seed-DataFrame figure types
  `cov_dl_bars`, `cov_dl_panels`). `config.yml` declares the scrape grid (or rows), tables, and a `figures:` list
  (each entry = a `FIGURE_TYPES` key plus that plotting function's kwargs); `run.py` (thin, on top of
  `scripts.results_utils.runner`) loads the tracked `records.json` snapshot (or `--rescrape` from W&B) and writes
  `tables/`, `figures/` (PNG, 300 dpi), and `manifest.json` into the experiment folder itself (`output_dir`,
  relative to the folder; default `.`), all tracked. Every plotted value is recomputable from the tracked seed-level
  table (`tables/seed_records.csv`, or `tables/row_seed_metrics.csv` for row-based experiments). Rendering is deterministic, so re-running on unchanged data leaves tracked outputs
  byte-identical (only `manifest.json`'s `generated_at` changes). It refuses to run on a missing cache or one that
  does not cover the config.
- **Specs.** Refactors and experiment plans are tracked in `context_packages/repo_spec_docs/`: `spec_doc_v1.md`
  (complete: the `scripts/` build-out and the composite-loss / regularization modeling track), `spec_doc_v2.md`
  (closed 2026-10-04: E0–E3 experiments, environment consolidation, task-FC caches, repo organisation), `spec_doc_v3.md`
  (closed 2026-10-21: E0 task FC ↔ SC, E1 covariate ablation, E2 joint task FC, E3 composite benchmark, E4 task
  signatures), `spec_doc_v4.md` (active: E1 multiview CCA-latent benchmark (done), E2 SC → functional latent (E2.1
  done, E2.2 planned), E3 generative follow-ups (outline: more samples per subject from timeseries windows / runs,
  activity-flow baseline), I1 joint translation infra, I2 multiview CCA module). Format and item IDs (`I` infrastructure, `M` modeling, `E` experiment,
  `C` carryover, `C!` caveat, `D` decision) follow `spec_conventions.md`; each spec's status table is the place to read
  its current state.

## HPC / Workflow Guardrails

- Prefer targeted reads over recursive scans.
- Avoid heavy artifact trees unless task explicitly needs them:
  - `results/ray_results/` (best-trial reports + checkpoints, ~118 GB)
  - `results/ray_checkpoints/` (Ray Tune storage: every trial's params/progress/W&B local copy)
  - `results/ray_tmp/` (per-job Ray session scratch + Ray system logs; a 45-tune campaign left 246 GB here, spec v3 backlog)
  - `results/conn2conn/` (Lightning checkpoints per W&B run id), `results/cache/krakencoder/` (retrain runs, ~13 GB per grid cell)
  - `results/logs/` (SLURM stdout/stderr; forwarded Ray worker output lands here)
  - `wandb/` (local W&B run copies; the W&B cloud is the source of truth)
  - large notebooks
- Use SLURM scripts for large jobs; avoid long compute on login nodes.
- **GPU-usage policy:** jobs averaging < 50% GPU utilisation for > 2 h are cancelled (warning < 70%). Our models reach
  2–35% on any GPU (L40S included), so every Tune task must finish within 2 h: E2.2 packing (0.25 × 4, 10 CPUs) with
  actor reuse on; `autopilot.py` `budget.max_wall_h`; `launch_model.sh` writes `<task log>.gpu.csv`.
- Prune Ray storage with `scripts/maintenance/prune_ray_storage.sh` (keeps every session's `logs/`).
- Kernel environment: `kraken_env` runs inside a Singularity overlay (see `/scratch/asr655/envs/README.md`);
  mount the overlay `:ro` for import-only checks so running jobs are unaffected. Use `source /ext3/env.sh`
  (what the launchers use): since 2026-09-30 (spec v2 C6) every job package, including `torch_geometric`, lives in the
  overlay and `env.sh` sets `PYTHONNOUSERSITE=1` / `PIP_USER=0`, so `~/.local` is never read or written. Install
  packages with a plain `:rw` mount (no `--fakeroot`; it cannot write the overlay on this cluster). `activate_env.sh`
  is a different stack (`pylibs`, torch 2.11) until spec v3 I4. Without the library, `import wandb` silently picks up the
  repo-root `wandb/` run folder.

---

## High-Risk Gotchas

1. W&B `group` is not sweep-level identity in this repo.
2. Tune configs are flat; best-trial configs are nested.
3. Multi-source PCA settings may be scalar or dict; resolve before model build.
4. Precomputed model path bypasses DataLoader-based prediction.
5. Krakencoder config/class naming should be verified before relying on automated runs.
6. `Chen2024GCN` and `NodalGNN` require `torch_geometric` (in the kraken overlay since 2026-09-30, spec v2 C6).
7. Worktree runs write Tune / Ray outputs under the worktree's own `results/`; move keepers to the main checkout and
   clean `ray_tmp/` before `git worktree remove`.
8. `NodalMLP` does not require `torch-geometric`, but configs with `use_sc_row=True` require `batch["sc_matrix"]`; this is wired through `Sim` and the train/eval wrappers.
9. `precomputed` data_load_mode only works when cache files already exist at the resolved cache root.
10. Launchers (`scripts/sbatch/<ModelName>/` and campaign launchers in experiment folders) hardcode an absolute `CONN2CONN_DIR` and absolute log paths, so they are location-independent.
11. Sbatch scripts and input-feature subset configs are still duplicated by experiment variant; spec v3 I2 (one tune launcher, variants as roster overrides) is the planned fix.
12. `models/`, `data/` have no `__init__.py` (namespace packages) and `models/eval/__init__.py`, `models/train/__init__.py` re-export nothing: import from the defining module (e.g. `from models.eval.evaluator import Evaluator`), not the package.
13. W&B project/entity and results paths are duplicated in `main.py` and `scripts/results_utils/records.py`, and
    results paths again in `loss_grid.py`, `optuna_importance.py` and `models/train/trainer.py`
    (cwd-relative `results/lightning_logs`); change them together.
14. W&B public API: always fetch runs via `query_runs` (full data); a plain `api.runs(...)` on wandb 0.25 returns empty configs and every source/seed parse fails silently (runs are skipped with a warning).
15. The regularization term is added unnormalized on top of the loss, so the effective reg strength depends on the loss
    scale: an `ema`-normalized composite starts at about Σweights, a raw `none` loss at the raw MSE scale. Re-tune reg ranges
    when switching normalization or weight schemes.
16. `correye` / `neidist` compare subjects within a batch: their values and meaning depend on `batch_size` (0 for a batch
    of 1). Keep `batch_size` fixed when comparing weights.
17. Queued SLURM jobs read `main.py` / `models/` / `data/` when they **start**, but run SLURM's own copy of the batch
    script and the resources / `--export` values fixed at submission. Users cannot `scontrol hold` or `update` jobs
    here: changing queued jobs means cancel + resubmit. Write `main.py` edits atomically while jobs may start.
18. Judge a Tune task by its log (best-trial report(s) present), not its SLURM state: a post-run cleanup error once
    marked a finished task `FAILED` (cleanup is non-fatal since `f2f4b90`). `run.py` / `autopilot.py` take the latest
    finished log per (model, seed), so move aborted logs aside; smoke tests go under `submit.py --job-prefix`.
19. Ray actor reuse must stay on for packed Tune runs (each worker keeps its HCP_Base / loaders); with it off every
    trial rebuilds the data (~10 min). Smoke tests need more trials than concurrent slots to exercise reuse; an
    intermittent `reset_config` stall is handled as a task failure (one retry).
20. Tune trials log `val_avg_rank` / `val_top1_acc` from `CrossModalLightningModule.on_validation_epoch_end` (GPU;
    `CONN2CONN_VAL_RANK_CPU=1` = old CPU path), not from `ValidationEvalCallback` (prod path only). Lightning per-trial
    checkpoints are off in Tune (`--tune_lightning_checkpoints`); Tune's own checkpoint is unaffected.
21. Sweeps before 2026-09-23 of `CrossModal_PCA_PLS_learnable`, `CrossModal_PCA_PLS_CovProjector` and `Sarwar2020MLP`
    ignored the sampled `l1_reg`/`l2_reg` (default tuple won; spec v1 M5b). Their logged reg values were not applied.

---

## Minimal Task Recipes

Add/modify model:
1. edit or add a class under `models/architectures/`
2. add/update YAML in `models/configs/<family>/` (variants in `variants/<family>/`)
3. ensure `build_model()` can resolve class name
4. run one dev/prod dry run with fixed seed

Add/modify loss behavior:
1. add atomic terms or factories in `models/train/loss.py`
2. route training behavior through `models/train/lightning_module.py`
3. keep metric-only calculations in `models/eval/metrics.py` unless they are part of the differentiable training objective
4. new edge-space terms go in `CompositeLoss` (`VALID_TERMS` + `_compute_raw_term`; add to `MEAN_TERMS` / `MONITOR_TERMS` if it needs the target mean or may be monitored); expose weights to Tune as `loss_weight_<term>`
5. check equivalence and regressions with `resolve_loss_config` + `create_loss_fn` on random tensors before any GPU run

For nodal models:
1. `NodalGNN` node features come from `batch["node_features"]`, not from `cov`.
2. `NodalMLP` may consume `batch["node_features"]`, `batch["sc_matrix"]`, or both depending on `use_volume`, `use_spatial`, `use_r2t`, and `use_sc_row`.
3. `CrossModalLightningModule`, `predict_from_loader`, and loss/eval helpers already know how to pass `node_features` and `sc_matrix`.
4. anatomical ablations are controlled in config via `use_volume`, `use_spatial`, and `use_r2t`; SC-row/spectral ablations use `use_sc_row`, `encoder_type`, `decoder_type`, and `sc_row_norm`.

Update experiment reporting:
1. patch `scripts/results_utils/` (fetch/parse → `records.py`, tables → `tables.py`, figures → `plots.py`); re-export new public names in `__init__.py`
2. run smoke table builds (`records_to_df`, `build_status_table`, `build_metric_table`)
3. verify local enrichment still merges correctly

Add a notebook or experiment:
1. notebooks → `scripts/notebooks/<purpose>/`; side experiments → `scripts/experiments/<slug>/`
2. start the first code cell with the standard bootstrap (see "Repo Layout Conventions"); anchor paths on `REPO_ROOT`
3. for experiments, write `<slug>/<slug>.md` (unless the notebook documents itself) and add a row to `scripts/experiments/experiments_index.md`
4. for a results/benchmark experiment, copy `scripts/experiments/sc_type_benchmark/` (model × source grid) or
   `scripts/experiments/covariate_projector/cov_projector_benchmark/` (labelled condition rows) and edit its `config.yml`

Debug missing results cell:
1. confirm best-trial run exists in W&B with expected tags
2. inspect parsed source/seed path (nested vs flat fallback)
3. verify model/source/seed is included in requested grid

## Recent Changes

2026-10-20 → 2026-10-27 — spec v4 E1, E2.1; timeseries infrastructure; viewers:
- E1 (`scripts/experiments/cca_latent_bench/`): multiview CCA latent of FC over rest + 7 tasks; tasks through the latent
  beat every v3:E0 task FC → SC row, SC → latent → FC beats v3:E0 on all conditions; the latent loses rest-specific
  information. Library: `models/architectures/multiview_cca.py` (I2).
- E2.1 (`scripts/experiments/joint_flow_matching/`): SC → each condition's latent encoding; linear at this n; SC predicts
  where a subject sits, not how variable they are; a shared-covariance Gaussian matches the flow. Library:
  `models/architectures/crossmodal_flow_matching.py` (I1). Bulky outputs in `results/cache/experiments/<slug>/`.
- Timeseries cache + per-run FC caches + `data/timeseries*.py` (see Data + Partitioning).
- Viewers: shared `scripts/data_viewers/viewer_kit/`; one folder per viewer, built pages untracked; experiment layout
  standard (see Repo Layout Conventions); `cca_latent_bench` converted to it. composite_loss instance outputs
  (runs, tables, figures, state.yml) are now tracked.

2026-10-12 → 2026-10-21 — spec v3 E1, E1.5, E3, E4 (v3 closed 2026-10-21; details in `spec_doc_v3.md`):
- E1 covariate ablation (`scripts/experiments/covariate_projector/covariate_ablation/`): `fs_vol` / `fs_area` / `fs_thick` cov sources
  (`data/hcp_dataset.py`), CovProjector `cov_sources: []`, unnormalised Glasser SC cache. FC → SC covariate gain =
  regional size, ~40% of it the `invnodevol` normalisation (C!1). E1.5 `node_size.py`: parcel volumes alone identify
  subjects from SC, normalised or raw (C!2, standing caveat; D4: SC stays a target).
- E4 task-FC signature above rest (`FC_matrix_analysis.ipynb` §4, `data_viz.compute_task_signatures`): every task has a
  reproducible signature that individuals express.
- E3 composite-loss benchmark (`model_benchmark/composite_loss/`, `config_composite_loss.yml`): declared weight grid per
  direction, scales measured per model; composite loss helps every model SC → FC, not FC → SC. Infra: `--report_metrics`
  (one best-trial report per metric; the second reuses the retrain when the same trial wins), `val_avg_rank` /
  `val_top1_acc` in Tune trials, `--tune_lightning_checkpoints` (default off), `launch_model.sh` env-driven
  (`TUNE_REUSE_ACTORS`, `TUNE_LIGHTNING_CHECKPOINTS`) with GPU-use log and non-fatal Ray cleanup, `submit.py
  --job-prefix / --time`, per-model `cpus`, `autopilot` `max_wall_h` and gate-less campaigns.
- `data_viz`: edgewise held-out prediction / evaluation helpers, Glasser-safe atlas reader.
- Storage: `scripts/maintenance/prune_ray_storage.sh`; scratch 4.81 → 3.0 TB (2026-10-20).

2026-10-07 → 2026-10-12 — spec v3 E2 / I5:
- `FC_<task>` source modalities (`data/hcp_dataset.py`, `models/architectures/utils.py`): a task FC as its own source
  block (needs the task in `fc_conditions`; train PCA only when requested). `CrossModal_PCA_PLS_TaskResid` +
  `fit_rest_residualizer` (predictive-residual task blocks).
- `scripts/experiments/joint_taskfc2sc/` (E2): CPU grid runner for closed-form multi-block PCA-PLS (one PLS fit per cell
  truncated to every `n_pls`; Evaluator metrics). Result: task-FC blocks add no FC → SC signal over rest.

2026-10-05 → 2026-10-07 — spec v3 started:
- `HCP_Base(fc_source_condition=...)` (`31de503`): rest sessions or a task become the run's FC; one shared
  `HCP_Base` data-key list for all builders.
- `scripts/experiments/task_bidirectional_bench/` (spec v3 E0; was `task_fc_to_sc`, then `task_fc2sc` until 2026-10-07; old config paths in logs stay valid): 9 FC conditions × 5
  seeds × {FC → SC, SC → FC} × {Glasser, 4S456Parcels}; rest best everywhere, condition order replicates. Steps in
  `config.yml`; `autopilot.py` (status / gate / budget / launch); `launch.sh` deletes each finished task's Ray temp and
  Tune storage (`KEEP_RAY=1` keeps them). Best-trial summaries now record `parcellation`, `fc_source_condition`,
  `n_subjects`. Gotcha: `--parcellation` (CLI, default Glasser) overrides `data.parcellation` in a config.
- `model_benchmark`: Masked MLP FC → SC at full budget via `gate_override` (‡, `8f97342`); restyled figures with
  `panel_grouped.png` / `panel_performance.png`.
- Config family folders merged (`5efe03a`, spec v2 I2.3 / v3 I2).
- Best-trial reports record their direction (`2e9df7f`); eval-report labels follow the target (pFC / eSC, `4f575d3`).

2026-10-04 — spec v2 closed (E1, E2.2, E3, I1.4–I1.5; details in `spec_doc_v2.md` § Closure):
- `scripts/experiments/composite_loss/` (E1 + E3): per-model instances in `<model>/{sc2fc,fc2sc}/` (`--instance
  <model>/<direction>`); grid v3; cross-model
  viewer `viewers/cross_model/cross_model.html` (was `figures/cross_model_interactive.html`) with a direction switch and the SC → FC test-retest ceiling. Fixed
  reference scales are the protocol's default balancing (spec v2 D6).
- `scripts/experiments/model_benchmark/` (E2.2): MSE-only benchmark, both directions. Configs generated into
  `models/configs/benchmark/mse/` by `build_configs.py` (no `data:` block: `--source/--target` set the direction;
  `<Model>_fc2sc.yml` only where needed); roster / packing / gate / budget in `config.yml`; `autopilot.py` (host python:
  status, latent gate, GPU-h budget, launch of missing seeds) and `run.py` (task logs → `mse/<direction>/` records,
  tables, figures). Packing note: tasks at 0.5 GPU × 2 trials were killed by the cluster after ~2 h (GPU-underuse
  policy); 0.25 × 4 runs clean.
- Retrainable Krakencoder (`models/architectures/krakencoder/`, spec v2 E2.1) serves both directions.
- Task FC caches and loaders: `HCP_Base(fc_conditions=[...])`, `load_fc_precomputed(task=)` (spec v2 I1.4–I1.5).

2026-09-30 — E0 closed and environment consolidated (spec v2 E0, C6, C1):
- `scripts/experiments/nodal_models_benchmark/` (closed): every nodal model is at the null on SC→FC; follow-up
  "graph models on top of the mean" in spec v2 §5. `nodal_decoder_importance.ipynb` removed (superseded).
- Job environment: `~/.local` packages + `torch_geometric 2.8.0.post1` moved into the kraken overlay; `/ext3/env.sh`
  disables the user site. Snapshots and scripts: `/scratch/asr655/envs/kraken_env/c6_snapshot_2026-09-30/`.
- `main.py --tune_reuse_actors {true,false}` (default true; keep it with packed trials, spec v2 E0.6).

2026-09-23 — modeling track (spec v1 §8, M1–M8):
- One loss path: `resolve_loss_config` → `create_loss_fn`; `composite` is the only edge-space loss type. `mse`,
  `weighted_mse`, `sarwar_mse_corr` and `vae` were retired and folded into composite terms, bit-exact with the old losses.
- Searchable `loss_weight_*` / `loss_kwarg_*`; `loss_normalize: auto`; `loss_signature` in W&B; per-term losses in Tune.
- `CompositeLoss` EMA scales now track `|raw|` (a signed `neidist` used to inflate about 10⁸×).
- `l1_reg` / `l2_reg` for every learned model; sampled reg values now take effect in sweeps (see gotcha 21).
- `CrossModal_linear_backbone` added; `LatentAttnMasked` `residual_mode: none` removed. Four notebooks migrated.
- Real-data verification array: `scripts/sbatch/checks/verify_modeling_track_array.sh` (reports in `results/logs/`).

2026-09-23 — `scripts/` build-out (details: `context_packages/repo_spec_docs/spec_doc_v1.md`):
- All non-library code now lives under `scripts/`: `results_utils/`, `notebooks/`, `experiments/`, `sbatch/`.
- `results/results_scraper.py` (untracked, 2.1k lines) became the tracked package `scripts/results_utils/`
  (`records` / `tables` / `plots` / `local_results` / `optuna_importance`); `results/` is artifacts-only.
- Every notebook uses the walk-up `REPO_ROOT` bootstrap; stale `models.*` imports left over from the `models/`
  refactor were fixed in 13 notebooks (all 32 pass an import-only check in `kraken_env`).
- `notebooks/quick_experiments/` was retired; its notebook is now
  `scripts/experiments/linear_backbone/geodesic/linear_backbone_geodesic_metrics.ipynb` (self-documenting; moved under
  `linear_backbone/` on 2026-09-30, next to `composite_loss/linear_backbone/`).
- `context_packages/` reorganized: `modeling/` (design notes), `repo_spec_docs/`, `schematics/`, `T1/`; stale copies
  of `main.py` / `hcp_dataset.py` removed. The former `experiment_ledger/` was folded into the experiment folders:
  write-ups now live at `scripts/experiments/<slug>/<slug>.md` (Adel's summer summary →
  `scripts/experiments/adel_summer_2026/adel_summer_2026.md`), indexed by `scripts/experiments/experiments_index.md`.
- Artifact cleanup: `.ipynb_checkpoints/` removed at repo root and under notebooks; dangling `results/wandb/`
  removed; `results/ray_tmp/` sessions before 2026-04-01 pruned.
- `scrape_SCtype_results.ipynb` replaced by the config-driven `scripts/experiments/sc_type_benchmark/` (tracked
  `records.json` snapshot; reproduces the notebook's table exactly); write-up `sc_type_benchmark/sc_type_benchmark.md`.
- Scraper fixes: `query_runs` (wandb 0.25 lazy runs had empty configs), model attribution by fetch target,
  hash-based fallback plot colors for models without an explicit color.
- `scrape_covtype_results.ipynb` replaced by the config-driven `scripts/experiments/covariate_projector/cov_projector_benchmark/`
  (both notebook tables reproduced exactly); shared runner plumbing extracted to `scripts/results_utils/runner.py`;
  `cov_dl_bars` / `cov_dl_panels` added to `FIGURE_TYPES`.

Earlier: model code moved to `models/architectures/`, training to `models/train/`, evaluation to `models/eval/`;
old top-level shims removed (git history has the rest).

Last updated at: 2026-10-27 EDT
