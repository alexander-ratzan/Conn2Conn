# Conn2Conn

Predicts one connectome modality from another on HCP-derived data (default `SC → FC`). Benchmarks a family of cross-modal mapping models — from closed-form PCA/PLS baselines to learned projectors conditioned on subject-level covariates — with systematic hyperparameter tuning, multi-seed evaluation, and W&B experiment tracking.

> For a detailed technical reference (architecture, config system, W&B schema, scraper API), see [CONTEXT.md](CONTEXT.md).

---

## Models

Grouped by **model type** (what the model is) and **learning type** (how it is fit). Every model runs in both
directions (`SC → FC`, `FC → SC`) except `Chen2024GCN`, `NodalGNN` and `NodalMLP` (SC → FC only; FC → SC variants are
spec v3 I3) and `TestRetestPrecomputed` (FC target only). How the models compare, tuned per seed under MSE in both
directions: `scripts/experiments/model_benchmark/model_benchmark.md`.

| Model | Model type | Learning type | Description |
|---|---|---|---|
| `CrossModalPCA` | Null / ceiling | Closed-form | Source PCs mapped index-for-index onto target PCs; no cross-modal fitting (≈ population-mean prediction) |
| `TestRetestPrecomputed` | Null / ceiling | Precomputed (no fitting) | Test-retest oracle: the other session's FC as the prediction (north-star ceiling; `FC` target only, needs `expose_fc_sessions: true`) |
| `CrossModal_PLS_SVD` | Linear decomposition | Closed-form | PLS via SVD decomposition |
| `CrossModal_PCA_PLS` | Linear decomposition | Closed-form | PCA reduction + PLS regression in latent space |
| `CrossModal_ConditionalGaussian` | Linear decomposition | Closed-form | Conditional mean of a joint Gaussian in PCA space (or ridge in edge space), optional covariates |
| `CrossModal_PCA_PLS_learnable` | Linear decomposition | Supervised (gradient) | PCA/PLS-initialized linear map, fine-tuned end-to-end |
| `CrossModal_linear_backbone` | Linear decomposition | Supervised (gradient) | Frozen PCA encoder/decoder around one learned affine latent map; fast probe for loss/regularization studies |
| `CrossModal_PCA_PLS_CovProjector` | Linear decomposition | Supervised (gradient) | PCA+PLS + residual correction conditioned on subject covariates |
| `LatentAttnMasked` | Latent / pretrained | Supervised (gradient) | Latent PCA backbone (zero / PLS / learned linear) plus a masked target-token attention residual |
| `MaskedLatentPretrainer` | Latent / pretrained (experimental) | Self-supervised (masked reconstruction) | Joint source+target latent-token reconstruction; predicts by masking every target token |
| `MaskedMLPPretrainer` | Latent / pretrained (experimental) | Self-supervised (masked reconstruction) | Same masked objective with linear / MLP backbone variants |
| `NodalMLP` | Pairwise nodal | Supervised (gradient) | Per-region embeddings (anatomy and/or subject SC rows) → pairwise edge decoder (MLP or dot / bilinear / linear-β probes) |
| `NodalGNN` | Pairwise nodal | Supervised (gradient) | Subject anatomy node features (volume, centroid, `SC_r2t`) → GCN over the SC graph → pairwise edge decoder |
| `Chen2024GCN` | Deep-learning baseline | Supervised (gradient) | Chen et al. (2024): one-hot nodes → GCN over the SC graph → edge MLP on `[h_i ‖ h_j]` |
| `Sarwar2020MLP` | Deep-learning baseline | Supervised (gradient) | Sarwar et al. (2021): edge-vector MLP trained with MSE + inter-subject correlation penalty |
| `Krakencoder_precomputed` | Deep-learning baseline | Precomputed (trained externally) | Krakencoder predictions loaded per seed (class `KrakencoderPrecomputed`) |
| `Krakencoder` | Deep-learning baseline | Supervised (upstream trainer, retrained per seed) | Krakencoder retrained in the repo from vendored upstream; one fit serves `SC → FC` and `FC → SC` |
| `CrossModalVAE` | Experimental (in development) | Supervised (gradient) | Variational autoencoder cross-modal mapping |

---

## Setup

Jobs and notebooks run in the `kraken_env` Singularity overlay (all packages, including `torch_geometric`, live in the
overlay; `~/.local` is never read). Environment details: `/scratch/asr655/envs/README.md`.

```bash
singularity exec --nv \
  --overlay /scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif /bin/bash
source /ext3/env.sh          # what every launcher uses
wandb login                  # once per host
```

Mount the overlay `:ro` for runs and import checks (running jobs share it); install packages with a plain `:rw` mount
(no `--fakeroot`). `kraken_env.yml` is a reference spec of the env, not the install path. Data paths are set in
`data/hcp_dataset.py` / `data/dataset_utils.py` and need access to the HCP-derived files below.

---

## Data Structure and Splits

### Connectomes and parcellations

| Modality | What it is | Source (under `GeneEx2Conn_data/HCP1200/`) |
|---|---|---|
| `SC` | Structural connectome upper triangle: `sift_invnodevol_radius2_count_connectivity` (default `sc_metric_type`), `log1p` (`sc_apply_log1p: true`) | `HCP1200_DTI/qsirecon/sub-*/anat/*_connectivity.mat` |
| `SC_r2t` | Region-to-tract profiles (66 tracts) turned into a region × region correlation matrix, upper triangle | `HCP1200_DTI/qsirecon/sub-*/anat/tract-to-region/` |
| `FC` | Pearson functional connectome upper triangle (xcp-d 0.9.1); which condition is `FC` is set per run (below) | `HCP1200_fMRI/xcpd-0-9-1/sub-*/func/` |

- Parcellations: `Glasser` (360 nodes, default) and `4S456Parcels` (456 nodes), set with `--parcellation` or
  `data: {parcellation: ...}`. Node order: `data/atlas_info/<parcellation>_dseg_reformatted.csv`.
- Source can be one modality or a `+`-join (`SC+SC_r2t`, `FC`, ...); the target is one modality (`FC` or `SC`).

### FC conditions (rest, sessions, tasks)

FC is available for rest and for the seven HCP task conditions, as npy caches in `Conn2Conn_data/fc/` (see its
`README.md`; builder `data/data_caching/build_fc_cache.py`).

| Condition | Built from | Timepoints per subject |
|---|---|---|
| `rest` (default `FC`) | 4 rest runs concatenated | 4800 |
| `rest_S1`, `rest_S2` | one session: mean of its LR and RL run matrices | 2 × 1200 each |
| `wm`, `language`, `motor`, `social`, `gambling`, `relational`, `emotion` | one LR + one RL run concatenated (xcp-d combined relmat) | 810, 632, 568, 548, 506, 464, 352 |

Task FC is a full-timeseries correlation (task-evoked activity is not regressed out), and conditions differ in scan
length. Choose conditions through the config `data:` block (there are no CLI flags for these):

```yaml
default:
  data:
    fc_conditions: [emotion, gambling, language, motor, relational, social, wm]  # load these task caches
    expose_fc_sessions: true        # load rest_S1 / rest_S2 (also needed by TestRetestPrecomputed)
    fc_source_condition: wm         # rebinds "FC" to this condition before the train-split PCA
```

- `fc_source_condition` (`rest` / unset, `rest_S1`, `rest_S2`, or a loaded task) makes that condition *the* FC for the
  whole run (PCA bases, model, evaluator), so it is the source in `FC → SC` and the target in `SC → FC`.
- `fc_condition_matrices: true` also loads dense task matrices (upper triangles only by default).
- Example: `scripts/experiments/task_fc2sc/` (spec v3 E0).

### Subjects and cohort

`HCP_Base` keeps only subjects present in every loaded source: metadata, SC, FC, FreeSurfer, parcel node features,
plus any requested sessions and task conditions. Rest-only runs use **957 subjects**. Loading all seven tasks drops
subjects missing any task (5–15 per task): **917** (Glasser) / **916** (4S456Parcels).

Metadata comes from `participants.tsv` (in the gitignored `krakencoder_experimental/example_data/HCP-YA_dataset/`)
merged with `HCP1200_RESTRICTED.csv` (family, zygosity, race/ethnicity).

### Train / Val / Test splitting

- `shuffle_seed: 0` is the original `participants.tsv` split: 683 / 79 / 195 of 957.
- Any other seed regenerates a split with the same proportions (`dataset_utils.generate_train_val_test`).
- Every split is **family-preserving**: siblings and twins share a partition (seed 0: 423 families, none split).
- Splits are assigned on the 957-subject table and then restricted to the run's cohort. With all task conditions,
  seed 0 is 659 / 74 / 184.
- `HCP_Base` stores `trainvaltest_partition_indices` / `_ids`; `HCP_Partition(base, "train"|"val"|"test")` is the
  PyTorch dataset for one split.
- Benchmarks use seeds 0–4. Hyperparameters are selected on val demeaned r, and test is reported.

### Data loading modes

- `precomputed`: read the npy caches in `Conn2Conn_data/` (`fc/`, `sc/`, `parcel_node_features/`). This is fast and is
  what the benchmark launchers use (`--data_load_mode precomputed`). Task conditions and sessions are always read from caches.
- `manual` (the `HCP_Base` default): parse the raw per-subject files. It writes caches only with
  `write_manual_cache=True`.

### Per-subject sample schema

Each dataset item returns:
- `x`: model input tensor (single source) or modality dict (multi-source)
- `x_modalities`: dict of all source modality tensors
- `y`: target modality tensor
- `cov`: covariate dict for the requested covariate sources
- `subject_id`: HCP subject identifier
- `node_features` (only `NodalGNN` and feature-enabled `NodalMLP`): `[num_nodes, num_features]`, channels
  `[volume, centroid_xyz, SC_r2t...]`
  - `volume_feature_type` defaults to `volume_mm3`; `centroid_feature_type` defaults to `centroid_mm`.
- `sc_matrix` (only `NodalMLP` with `use_sc_row=True`): the dense subject SC matrix

### Covariates

| Source | Contents | Encoding |
|---|---|---|
| `fs_all` | full FreeSurfer feature set (includes regional volumes) | z-scored |
| `fs_volumes` | 16 whole-brain volume columns | z-scored |
| `age` | age | scalar, z-scored |
| `sex` | sex | one-hot |
| `race_eth` | race × ethnicity | one-hot; categories with < 10 training subjects collapse to `Other` |

All normalization uses training-split statistics. FreeSurfer covariates carry anatomy that relates to the SC target's
`invnodevol` normalization (spec v3 C!1).

### Subject ordering (evaluation)

Identifiability heatmaps can order subjects by `original`, `family` (`Family_ID`), `demographic` (sex × race_eth) or
`age`. Ordering affects plots only, not split membership.

---

## Running Experiments

### Single run (interactive / dev)
```bash
python main.py --mode dev --model CrossModal_PCA_PLS_learnable
```

### Production run with multi-source input
```bash
python main.py --mode prod --model CrossModal_PCA_PLS_learnable --source SC+SC_r2t --target FC --shuffle_seed 0
```

### Hyperparameter tuning (Ray Tune)
```bash
python main.py --mode prod --model CrossModal_PCA_PLS_learnable \
  --use_tune --num_samples 100 --max_concurrent_trials 4
```

### Tune then evaluate best trial
```bash
python main.py --mode prod --model CrossModal_PCA_PLS_learnable \
  --use_tune --num_samples 100 --report_best_after_tune --store_eval_md
```

### Key CLI flags

| Flag | Description |
|---|---|
| `--mode dev\|prod` | `dev` for interactive/notebook use; `prod` for SLURM/batch |
| `--model` | Model name (its default config is found by name anywhere under `models/configs/`) |
| `--config` | YAML/JSON overrides (a file with a `default:` key is read like a model config); the only way to set `data:` keys such as FC conditions |
| `--parcellation` | `Glasser` (default) or `4S456Parcels` |
| `--source` | Input modality, or a `+`-join: `SC`, `SC_r2t`, `FC`, `SC+SC_r2t`, ... |
| `--target` | Output modality, one of `FC` (default) or `SC` |
| `--shuffle_seed` | Train/val/test split seed (benchmarks use 0–4; older arrays 0–9) |
| `--data_load_mode` | `manual` raw loading or `precomputed` cached-array loading |
| `--use_tune` | Enable Ray Tune HPO |
| `--num_samples` | Number of Ray Tune trials (default 10) |
| `--search_alg` | `random` or `optuna` |
| `--max_concurrent_trials`, `--tune_cpus_per_trial`, `--tune_gpus_per_trial` | Trial packing (env `TUNE_*_PER_TRIAL` also works) |
| `--tune_reuse_actors` | `true` (default; keep with packed trials) / `false` |
| `--report_best_after_tune` | Re-run and fully evaluate the best trial after tuning |
| `--save_checkpoint` | Write a checkpoint and report to the run dir |
| `--store_eval_md` | Save a markdown evaluation report when supported by the run/report path |
| `--hemi` | `both` (default), `left`, `right` |

### Loss and regularization

Every learned edge-space model trains on a **composite loss**: MSE plus optional weighted terms (`varmatch`,
`correye`, `correye_dm`, `neidist`, `demeaned_mse`, `pairwise_corr`, `kld`). A config with no loss keys is plain MSE;
every learned model takes `l1_reg` / `l2_reg`. Each run logs `loss_signature` (e.g. `mse+0.5*neidist`) to W&B. Term
weights, fixed reference scales and the normalization rules: `CONTEXT.md` → Config System Notes.

---

## Batch Jobs (SLURM)

- **Experiment campaigns** have their own launchers in their folder. Examples:
  - `scripts/experiments/model_benchmark/{launch_model.sh,submit.py,autopilot.py}`
  - `scripts/experiments/task_fc2sc/{launch.sh,submit.py}`
  - `scripts/experiments/composite_loss/launch_*.sh`
- **Per-model tune arrays** live in `scripts/sbatch/<ModelName>/`. They are legacy launchers: spec v3 I2 replaces them
  with one `launch_tune.sh` plus rosters. Folders: `Chen2024GCN/`, `CrossModal_ConditionalGaussian/`, `CrossModalPCA/`,
  `CrossModal_PCA_PLS/`, `CrossModal_PCA_PLS_CovProjector/`, `CrossModal_PCA_PLS_learnable/`, `CrossModal_PLS_SVD/`,
  `Krakencoder/`, `MaskedLatentPretrainer/`, `MaskedMLPPretrainer/`, `NodalGNN/`, `NodalMLP/`, `Sarwar2020MLP/`.
  ```bash
  sbatch scripts/sbatch/NodalMLP/tune_array_nodalmlp_seeds.sh
  ```
- **Other folders:**
  - `scripts/sbatch/checks/`: loss regression, modeling-track and FC-condition verification, notebook cell runner,
    `watch_jobs.py`.
  - `scripts/sbatch/data/`: FC cache builder array.

Launchers hardcode the repo path and absolute log paths (`results/logs/`), so you can submit them from any directory.

---

## Krakencoder Baseline (retrainable and precomputed)

Krakencoder runs outside the Lightning loop: its upstream code trains and predicts, and the loader class
`KrakencoderPrecomputed` serves the predictions to the evaluator (split slicing, no `forward` pass). Every inference
file holds all input → output types, so both `SC → FC` and `FC → SC` are served from one fit. Not tunable through
Ray (empty `search_space`); a variant is a new retrain tag.

**Retrained (`Krakencoder`):** package `models/architectures/krakencoder/` — vendored upstream (`vendor/`, commit
`b57e39c`, unmodified; see `vendor/VENDOR.md`), retrain wrapper (`retrain.py`), prediction loader (`precomputed.py`); inputs built from `HCP_Base` with the same per-seed splits as every other model.
```bash
# per seed: train + infer + evaluate both directions (EVAL_MODE=prod logs to W&B; dev does not)
sbatch scripts/sbatch/Krakencoder/train_array_krakencoder_seeds.sh                       # seeds 0-9, default recipe
sbatch --array=0 --export=ALL,CONFIG=<variant.yml> scripts/sbatch/Krakencoder/train_array_krakencoder_seeds.sh
# or step by step
python -m models.architectures.krakencoder.retrain --config models/configs/deep/Krakencoder.yml --seed 0
python main.py --mode prod --model Krakencoder --config models/configs/deep/Krakencoder.yml --source FC --target SC --shuffle_seed 0
```
- recipe: the `retrain:` block of `models/configs/deep/Krakencoder.yml` (flavors, loss string, epochs, latent size, dropout);
  defaults reproduce the March 2026 runs; `default.model.tag` names the output folder
- outputs: `results/krakencoder/<tag>/seed{seed}/` (checkpoint, transforms, logs, `manifest.json`,
  `predictions_source_{parc}.{SC|FC}.mat`); shared inputs in `results/krakencoder/_inputs/`
- about 50 min per seed on one GPU for the default recipe (2000 epochs, 4 flavors)
- per-checkpoint curves: `python -m models.architectures.krakencoder.checkpoint_eval --run-dir results/krakencoder/<tag>/seed<S>`
  (our metrics and loss terms at every saved checkpoint, both directions) → `epoch_history.csv`
- loss-weight grid (composite-loss experiment instance): `scripts/experiments/composite_loss/krakencoder/`

**Precomputed (`Krakencoder_precomputed`):** the cached March 2026 predictions in
`krakencoder_experimental/example_data/` (`mydata_kraken_seed{seed}_source_{parc}.{SC|FC}.mat`), produced by the
gitignored local copy `krakencoder_experimental/` (upstream + a demeaned-MSE loss; kept as reference and for
development). Run with `python main.py --mode prod --model Krakencoder_precomputed --source SC --target FC --shuffle_seed 0`.

---

## Results

All runs log to W&B project `conn2conn`. The W&B cloud is the source of truth; local `wandb/` run folders are upload staging copies.

Primary results API: `scripts/results_utils/` (active development surface).
- `records.py` — fetches best-trial prod runs from W&B (optionally direct `prod` runs for models logged outside the best-trial report path), resolves `(model, source, seed)` records including missing cells, caches records, and enriches them with local Ray artifact metrics (`metrics_final.json`)
- `tables.py` — flat DataFrames, model-vs-source pivot tables, and covariate-projector / deep-model comparison tables
- `plots.py` — grouped bar charts, model scatter plots, and covariate/deep-model panels, plus a figure-type registry used by config-driven experiment scripts
- `runner.py` — shared plumbing (CLI, cache checks, table/figure writing, manifest) for the config-driven experiment scripts
- `loss_grid.py` — composite-loss protocol helpers (Stage 1 runs, consensus config, reference scales, grid runs) used by `scripts/experiments/composite_loss/`
- `local_results.py` — loaders / plots for notebook-written `results/local_results/`
- `optuna_importance.py` — hyperparameter importance from W&B tune trials: `python -m scripts.results_utils.optuna_importance --help`

Notebook surface (`scripts/notebooks/`):
- `kraken/track_krakencoder_model.ipynb`, `kraken/kraken_eval.ipynb`
- model onboarding notebooks under `model_overviews/`, including PCA/PLS closed-form vs learnable overviews and latent-attention overviews
- model smoke-test notebooks under `model_testing/`
- exploratory data notebooks under `EDA/`

Config-driven results benchmarks (replace the former `scrape_SCtype_results` / `scrape_covtype_results` notebooks):

```bash
# inside kraken_env via `source /ext3/env.sh`
python scripts/experiments/sc_type_benchmark/run.py              # tables + figures from the tracked records.json
python scripts/experiments/sc_type_benchmark/run.py --rescrape   # refresh records.json from W&B first
python scripts/experiments/cov_projector_benchmark/run.py        # covariate projector vs baselines and deep models
python scripts/experiments/nodal_models_benchmark/run.py         # nodal models (NodalMLP decoders, GNNs) vs null / linear
```

Campaign experiments are scraped from their SLURM task logs, not from W&B. They follow `config.yml` + `submit.py` /
launcher + `run.py`:
- **Cross-model benchmark** (`scripts/experiments/model_benchmark/`, spec v2 E2.2): every model tuned per seed under
  MSE, both directions, one campaign per model (job names `e2_mse_<Model>_<direction>`). A composite-loss version
  needs spec v3 I1.
- **Composite-loss protocol** (`scripts/experiments/composite_loss/`): `protocol.py`, `report.py`, `compare.py`, with
  per-model instances in `<model>/{sc2fc,fc2sc}/`.
- **Task FC** (`scripts/experiments/task_fc2sc/`, spec v3 E0): FC condition × seed for PCA-PLS learnable.

`scripts/experiments/experiments_index.md` lists every experiment with its status and write-up.

```bash
python3 scripts/experiments/model_benchmark/autopilot.py status --direction sc2fc   # host python (needs squeue)
python3 scripts/experiments/model_benchmark/autopilot.py launch --direction sc2fc --dry-run
python scripts/experiments/model_benchmark/run.py --direction sc2fc                 # kraken env: records, tables, figures
```

Edit each experiment's `config.yml` to change models / conditions, seeds, the summary tables, or the list of figures. Outputs are written into the experiment folder and tracked in git: `tables/`, `figures/` (PNG,
300 dpi), and `manifest.json`.

```python
from scripts.results_utils import (
    build_experiment_records,
    records_to_df,
    build_metric_table,
)

records = build_experiment_records(
    models=["CrossModal_PCA_PLS_learnable", "CrossModal_PCA_PLS"],
    sources=["SC", "SC_r2t", "SC+SC_r2t"],
    seeds=[0, 1, 2, 3, 4],
)
df = records_to_df(records)
table = build_metric_table(records, metric="demeaned_pearson")
```

---

## Notebooks and Experiments

- Notebooks live under `scripts/notebooks/<purpose>/`. Each one starts with a small bootstrap that finds the repo root by walking up to `main.py`, so notebooks work from any folder in the repo; write repo paths as `REPO_ROOT / "results/..."`.
- Side experiments live under `scripts/experiments/<name>/` (code, launchers, and their tables/figures). Truly bulky outputs (checkpoints, large arrays) go to `results/experiments/<name>/`.
- W&B-scraped results experiments follow `scripts/experiments/sc_type_benchmark/` (model × source grid) or `scripts/experiments/cov_projector_benchmark/` (labelled condition rows): a `config.yml`, a `run.py`, and a small tracked `records.json` snapshot of the latest W&B scrape. Campaign experiments (`model_benchmark`, `task_fc2sc`) add a `submit.py` / launcher and read their task logs.
- Each experiment is documented in its own folder by a write-up named after it (`<name>/<name>.md`: the question, setup, how to run it, W&B ids, results, and caveats), or by its notebook when that already describes it. `scripts/experiments/experiments_index.md` lists all experiments.

---

## Repo Layout

```
Conn2Conn/
├── main.py                          # Entrypoint: single run, Ray Tune, best-trial reporting
├── kraken_env.yml                   # Reference env spec (the overlay is the real env)
├── CONTEXT.md                       # Technical reference for agents and developers
│
├── data/
│   ├── hcp_dataset.py               # HCP_Base / HCP_Partition: loading, cohort alignment, FC conditions, splits
│   ├── dataset_utils.py             # Raw + precomputed loaders, metadata, family-preserving splits
│   ├── data_utils.py, graph_adapter.py
│   ├── data_viz.py                  # Matrix overviews, FC condition views, gif helpers
│   ├── demeaned_viz.py              # Demeaned FC / prediction visualization helpers
│   ├── data_caching/                # FC cache builder, condition availability, cache exclusions
│   └── atlas_info/                  # Parcellation node tables (Glasser, 4S456Parcels)
├── models/
│   ├── registry.py                  # Config lookup by name, search-space conversion, model construction
│   ├── utils.py                     # Shared prediction / batch covariance helpers
│   ├── architectures/
│   │   ├── crossmodal_pca_pls.py    # CrossModalPCA, PLS_SVD, PCA_PLS (+ learnable, linear_backbone, CovProjector)
│   │   ├── crossmodal_conditional_gaussian.py
│   │   ├── crossmodal_vae.py        # experimental VAE
│   │   ├── sarwar2020_mlp.py
│   │   ├── test_retest_precomputed.py
│   │   ├── krakencoder/             # vendor/ (upstream, unmodified), retrain.py, precomputed.py, checkpoint_eval.py
│   │   ├── latent_attention/        # LatentAttnMasked + masked pretrainers
│   │   └── graph_based/             # Chen GCN, NodalGNN, NodalMLP, graph feature builders
│   ├── configs/                     # {null_ceiling,linear,latent,graph_nodal,deep}/<Model>.yml, variants/<family>/, benchmark/<campaign>/ (generated)
│   ├── train/                       # Training loop, Lightning wrapper, composite losses, training plots
│   └── eval/                        # Evaluator, metrics, FC distance, PCA analysis, reports, plots
├── scripts/                         # Everything that uses the library (tracked)
│   ├── results_utils/               # records / tables / plots / runner / loss_grid / local_results / optuna_importance
│   ├── notebooks/                   # EDA/, kraken/, model_overviews/, model_testing/
│   ├── experiments/                 # One folder per experiment (write-up <name>.md); experiments_index.md
│   └── sbatch/                      # Per-model tune arrays (legacy), checks/, data/
├── results/                         # Generated artifacts only (gitignored)
│   ├── ray_results/                 # Best-trial reports: checkpoint, test_results.md, plots, metrics_final.json
│   ├── ray_checkpoints/             # Ray Tune storage: per-trial params, progress, W&B local copies
│   ├── ray_tmp/                     # Per-job Ray session scratch + Ray system logs
│   ├── conn2conn/                   # Lightning checkpoints per W&B run id (prod / best-trial runs)
│   ├── wandb/                       # Local W&B run staging (the cloud is the source of truth)
│   ├── krakencoder/                 # Retrained Krakencoder runs by tag + shared inputs
│   ├── local_results/, figures/, tables/
│   └── logs/                        # SLURM stdout/stderr
├── checkpoints/                     # Legacy cwd-relative Lightning checkpoints (gitignored)
├── context_packages/                # Reference material: modeling/, repo_spec_docs/ (specs, current v3), schematics/, T1/
└── krakencoder_experimental/        # Local Krakencoder copy + data, cached predictions, participants.tsv (gitignored)
```

Last updated at: 2026-10-06 EDT
