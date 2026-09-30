# Conn2Conn — Agent Context

This is the fast technical context for AI agents and developers. Use this first for triage, then open code.

For user-facing usage/setup, see `README.md`.

---

## Mission

`Conn2Conn` predicts one connectome modality from another on HCP-derived data (default `SC -> FC`) and compares:
- closed-form baselines (PCA/PLS family)
- conditional Gaussian baseline
- learned variants (learnable map, covariate-conditioned residual projector, VAE, Sarwar MLP, Chen GCN)
- linear latent backbone probe (`CrossModal_linear_backbone`)
- latent attention variant (`LatentAttnMasked`)
- experimental masked-latent pretraining variants (`MaskedLatentPretrainer`, `MaskedMLPPretrainer`)
- nodal-feature GNN baseline (`NodalGNN`)
- graph-free nodal/edge MLP baseline (`NodalMLP`)
- precomputed Krakencoder baseline
- test-retest oracle baseline (`TestRetestPrecomputed`)

The main evaluation axis is performance across `(model, source, shuffle_seed)` with W&B-backed experiment tracking.

---

## Fast Triage

When a task arrives, start in this order:
1. `main.py` for orchestration and experiment mode behavior.
2. `models/configs/<model>.yml` for ground-truth defaults/search space.
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
- `scripts/results_utils/`
  - W&B/Ray results scraping → tables → figures (see "Results Utilities" below)
- `scripts/sbatch/<ModelName>/`
  - SLURM launchers (tune arrays, parallel/sequential sweeps); submit with `sbatch scripts/sbatch/<ModelName>/<script>.sh`
- `README.md`
  - user-level commands + workflow
- `scripts/experiments/sc_type_benchmark/`, `scripts/experiments/cov_projector_benchmark/`
  - config-driven results benchmarks (`config.yml` + `run.py` + tracked `records.json`); templates for turning results
    notebooks into scripts (model × source grid vs labelled condition rows)
- `scripts/experiments/nodal_models_benchmark/`
  - closed E0 benchmark of nodal models (NodalMLP probe decoders, NodalGNN, Chen GCN) with scraped tune trials
    (`trials.json`) for within-seed hyperparameter importance; its `run.py` enforces an MSE-only loss per run
- `scripts/notebooks/model_overviews/*.ipynb`
  - conceptual onboarding notebooks for PCA/PLS, conditional Gaussian, latent-attention, masked pretraining, and nodal baselines
- `scripts/notebooks/model_testing/*.ipynb`
  - per-model smoke-test / dev notebooks
- `scripts/notebooks/{EDA,kraken}/*.ipynb`
  - data exploration; Krakencoder tracking/eval
- `scripts/experiments/<name>/`
  - self-contained side experiments (code + small outputs + write-up `<slug>/<slug>.md`); index: `scripts/experiments/experiments_index.md`
- `context_packages/`
  - agent/human reference material: `modeling/` (model design notes), `repo_spec_docs/` (refactor specs), `schematics/`, `T1/`
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
- `TestRetestPrecomputed` (test-retest oracle, loads session 1/2 cached connectomes; class in `models/architectures/test_retest_precomputed.py`)

Learned (`learned: true`):
- `CrossModal_PCA_PLS_learnable` (flags include `mid_bias` and `zscore_pca_scores`; defaults reproduce the original model)
- `CrossModal_linear_backbone` (`models/architectures/crossmodal_pca_pls.py`) — thin `_learnable` subclass: frozen source PCA encoder → learned affine k×k latent map (`W_mid`, `mid_bias`) → frozen target PCA decoder. Pins the `_learnable` flags; free keys are `n_components_pca_source` (target tied to it), `zscore_pca_scores`, `l1_reg`/`l2_reg`. Formerly `LatentAttnMasked` with `residual_mode: none`. A fast, strong probe for loss/regularization studies.
- `CrossModal_PCA_PLS_CovProjector`
- `CrossModalVAE`
- `LatentAttnMasked` (implemented in `models/architectures/latent_attention/latent_attn_masked.py`); `residual_mode` ∈ {`attention_only`, `pls_residual`, `linear_residual`}. `none` was removed and raises, pointing to `CrossModal_linear_backbone`.
- `MaskedLatentPretrainer` (`models/architectures/latent_attention/masked_latent_pretrainer.py`) — **experimental, in development, not production**. SSL pretrainer for joint SC/FC PCA-latent reconstruction; transfers weights to `LatentAttnMasked` via `export_to_latent_attn_masked(downstream_model)`. Kept isolated: no cross-module changes in `lightning_module.py` / `trainer.py` / `main.py` should be made on its behalf. Dev harness: `scripts/notebooks/model_overviews/masked_attn_pretraining_overview.ipynb`.
- `MaskedMLPPretrainer` (`models/architectures/latent_attention/masked_mlp_pretrainer.py`) — **experimental**. SSL pretrainer variant of the masked-latent path using linear, low-rank-linear, or nonlinear MLP encoders plus configurable SC/FC masking. Config surfaces include `MaskedMLPPretrainer.yml`, `MaskedMLPPretrainer_linear.yml`, `MaskedMLPPretrainer_nonlinear.yml`, and `MaskedMLPPretrainer_mask_grid.yml`; sbatch launchers live under `scripts/sbatch/MaskedMLPPretrainer/`.
- `Sarwar2020MLP` (implemented in `models/architectures/sarwar2020_mlp.py`)
- `Chen2024GCN` (implemented in `models/architectures/graph_based/chen2024_gnn.py`)
- `NodalGNN` (implemented in `models/architectures/graph_based/nodal_gnn.py`)
- `NodalMLP` (implemented in `models/architectures/graph_based/nodal_mlp.py`) — graph-free node/edge baseline. It can use anatomical parcel features (`volume`, `spatial`, `SC_r2t`), subject SC rows, or spectral SC eigenvectors; decoder variants include `mlp`, `dot`, `bilinear`, `diag_bilinear`, and `linear_beta`. Configs include `NodalMLP.yml`, `NodalMLP_spatial.yml`, `NodalMLP_all_features.yml`, `NodalMLP_dot.yml`, `NodalMLP_bilinear.yml`, `NodalMLP_linear_beta.yml`, and `NodalMLP_spectral.yml`; sbatch launchers live under `scripts/sbatch/NodalMLP/`.

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

Performance / payload notes:
- `HCP_Partition` reuses shared base-level tensors across train/val/test partitions (avoids triple full-dataset tensor copies per split).
- `Sim` enables `expose_node_features=True` for `NodalGNN` and `NodalMLP`.
- `Sim` enables `expose_sc_matrix=True` for `NodalMLP` so SC-row and spectral encoders can reconstruct dense subject SC matrices from source edges.
- Other models do not pay for unused node-feature or dense-SC batch payloads.

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
- `Evaluator.analyze_results(...)` can optionally append exploratory geodesic summary metrics into `base_metrics` via:
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
- YAML `search_space` is converted to Ray Tune objects by `search_space_to_tune`.

Training loss (`models/train/loss.py`): every edge-space model uses `loss_type: composite`. The only other types are the latent
losses `latent_mse` / `latent_weighted_mse` (PCA-score reconstruction). `resolve_loss_config()` is the single entry point: it
fills defaults, applies overrides, validates, and adds `loss_signature`. A config with no loss keys is plain MSE.

```yaml
trainer:
  loss_type: composite
  loss_terms:                       # default when omitted: [mse]
    - {name: mse, weight: 1.0}      # anchor; not searched
    - {name: neidist, weight: 0.0}  # weight 0 drops the term
  loss_normalize: auto              # auto: none for one active term, ema otherwise; or ema / none
search_space:
  loss_weight_neidist: {type: choice, values: [0.0, 0.25, 0.5, 1.0]}   # flat, Optuna-safe
  loss_kwarg_pairwise_corr__corr_target: {type: uniform, lower: 0.3, upper: 0.9}
```

- Terms: `mse`, `varmatch`, `correye`, `neidist` (signed; kwarg `margin`), `demeaned_mse` (needs `base`),
  `pairwise_corr` (Sarwar; kwarg `corr_target`), `kld` (needs the model's `mu`/`logvar`, raises otherwise).
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
  `results/experiments/<slug>/`. `scripts/experiments/experiments_index.md` lists every experiment (status,
  dates, question, documentation link). Avoid folder names ending in `_context` (a global `.gitignore` rule
  ignores `*_context/`).
- **Results experiments (config-driven).** Two templates: `scripts/experiments/sc_type_benchmark/` (a model × source
  grid; record-level figure types `source_bars`, `metric_scatter`) and `scripts/experiments/cov_projector_benchmark/`
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
  (active: experiments and carried-over items). Format and item IDs (`I` infrastructure, `M` modeling, `E` experiment,
  `C` carryover, `C!` caveat, `D` decision) follow `spec_conventions.md`; each spec's status table is the place to read
  its current state.

## HPC / Workflow Guardrails

- Prefer targeted reads over recursive scans.
- Avoid heavy artifact trees unless task explicitly needs them:
  - `results/ray_results/` (best-trial reports + checkpoints, ~118 GB)
  - `results/ray_checkpoints/` (Ray Tune storage: every trial's params/progress/W&B local copy)
  - `results/ray_tmp/` (per-job Ray session scratch + Ray system logs; sessions before 2026-04-01 pruned, rest kept for debugging)
  - `results/logs/` (SLURM stdout/stderr; forwarded Ray worker output lands here)
  - `wandb/` (local W&B run copies; the W&B cloud is the source of truth)
  - large notebooks
- Use SLURM scripts for large jobs; avoid long compute on login nodes.
- Kernel environment: `kraken_env` runs inside a Singularity overlay (see `/scratch/asr655/envs/README.md`);
  mount the overlay `:ro` for import-only checks so running jobs are unaffected. Use `source /ext3/env.sh`
  (what the launchers use): since 2026-09-30 (spec v2 C6) every job package, including `torch_geometric`, lives in the
  overlay and `env.sh` sets `PYTHONNOUSERSITE=1` / `PIP_USER=0`, so `~/.local` is never read or written. Install
  packages with a plain `:rw` mount (no `--fakeroot`; it cannot write the overlay on this cluster). `activate_env.sh`
  is a different stack (`pylibs`, torch 2.11) until C6.5. Without the library, `import wandb` silently picks up the
  repo-root `wandb/` run folder.

---

## High-Risk Gotchas

1. W&B `group` is not sweep-level identity in this repo.
2. Tune configs are flat; best-trial configs are nested.
3. Multi-source PCA settings may be scalar or dict; resolve before model build.
4. Precomputed model path bypasses DataLoader-based prediction.
5. Krakencoder config/class naming should be verified before relying on automated runs.
6. `Chen2024GCN` requires `torch-geometric` in the runtime environment. As of 2026-09-23 it is **not installed** in the
   `kraken_env` overlay (torch 2.9.0), so this model and `NodalGNN` fail at import until PyG is reinstalled (`:rw` mount).
7. `NodalGNN` also requires `torch-geometric`.
8. `NodalMLP` does not require `torch-geometric`, but configs with `use_sc_row=True` require `batch["sc_matrix"]`; this is wired through `Sim` and the train/eval wrappers.
9. `precomputed` data_load_mode only works when cache files already exist at the resolved cache root.
10. Active multi-seed SLURM launchers live under `scripts/sbatch/<ModelName>/`. They hardcode an absolute `CONN2CONN_DIR` and absolute log paths, so they are location-independent.
11. Sbatch scripts and input-feature subset configs are still duplicated by experiment variant; a future manifest/launcher layer should move grids out of copied shell/YAML files.
12. `models/`, `data/` have no `__init__.py` (namespace packages) and `models/eval/__init__.py`, `models/train/__init__.py` re-export nothing: import from the defining module (e.g. `from models.eval.evaluator import Evaluator`), not the package.
13. W&B project/entity and results paths are duplicated in `main.py` and `scripts/results_utils/records.py`; change both together.
14. W&B public API: always fetch runs via `query_runs` (full data); a plain `api.runs(...)` on wandb 0.25 returns empty configs and every source/seed parse fails silently (runs are skipped with a warning).
15. The regularization term is added unnormalized on top of the loss, so the effective reg strength depends on the loss
    scale: an `ema`-normalized composite starts at about Σweights, a raw `none` loss at the raw MSE scale. Re-tune reg ranges
    when switching normalization or weight schemes.
16. `correye` / `neidist` compare subjects within a batch: their values and meaning depend on `batch_size` (0 for a batch
    of 1). Keep `batch_size` fixed when comparing weights.
17. Sweeps before 2026-09-23 of `CrossModal_PCA_PLS_learnable`, `CrossModal_PCA_PLS_CovProjector` and `Sarwar2020MLP`
    ignored the sampled `l1_reg`/`l2_reg` (default tuple won; spec v1 M5b). Their logged reg values were not applied.

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
4. new edge-space terms go in `CompositeLoss` (`VALID_TERMS` + `_compute_raw_term`); expose weights to Tune as `loss_weight_<term>`
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
   `scripts/experiments/cov_projector_benchmark/` (labelled condition rows) and edit its `config.yml`

Debug missing results cell:
1. confirm best-trial run exists in W&B with expected tags
2. inspect parsed source/seed path (nested vs flat fallback)
3. verify model/source/seed is included in requested grid

## Recent Changes

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
- `l1_reg` / `l2_reg` for every learned model; sampled reg values now take effect in sweeps (see gotcha 17).
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
- `scrape_covtype_results.ipynb` replaced by the config-driven `scripts/experiments/cov_projector_benchmark/`
  (both notebook tables reproduced exactly); shared runner plumbing extracted to `scripts/results_utils/runner.py`;
  `cov_dl_bars` / `cov_dl_panels` added to `FIGURE_TYPES`.

Earlier:
- Analysis/figure schematics and context images now live under `context_packages/schematics/`.
- Notebook organization uses purpose folders under `scripts/notebooks/`: `EDA/`, `kraken/`, `model_overviews/`,
  `model_testing/` (`results_scrape/` emptied and removed 2026-09-30).
- PCA/PLS onboarding notebooks split closed-form models from learnable/covariate-projector models:
  - `scripts/notebooks/model_overviews/crossmodal_pca_pls_closed_form_overview.ipynb`
  - `scripts/notebooks/model_overviews/crossmodal_pca_pls_learnable_overview.ipynb`
- Results-analysis workflow centers on:
  - `scripts/experiments/sc_type_benchmark/` (was `scrape_SCtype_results.ipynb`)
  - `scripts/experiments/cov_projector_benchmark/` (was `scrape_covtype_results.ipynb`)
- Lightweight visualization helpers were moved into:
  - `data/data_viz.py`
  - `data/demeaned_viz.py`
- `scripts/notebooks/kraken/track_krakencoder_model.ipynb` can log W&B runs and optionally save local markdown reports under `results/local_results/Krakencoder_precomputed/`.
- `NodalMLP` now has explicit decoder/encoder variant configs and launchers, including dot, bilinear, linear-beta, and spectral/connectome-harmonic probes.
- `MaskedMLPPretrainer` now has separate linear, nonlinear, and mask-grid config/launcher surfaces.
- Model code now lives under `models/architectures/`, training code under `models/train/`, and evaluation/reporting code under `models/eval/`.
- Backward-compatibility shims for old top-level model/train/eval files are intentionally removed.

Last updated at: 2026-09-30 EDT
