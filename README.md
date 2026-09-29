# Conn2Conn

Predicts one connectome modality from another on HCP-derived data (default `SC → FC`). Benchmarks a family of cross-modal mapping models — from closed-form PCA/PLS baselines to learned projectors conditioned on subject-level covariates — with systematic hyperparameter tuning, multi-seed evaluation, and W&B experiment tracking.

> For a detailed technical reference (architecture, config system, W&B schema, scraper API), see [CONTEXT.md](CONTEXT.md).

---

## Models

| Model | Type | Description |
|---|---|---|
| `CrossModalPCA` | Closed-form | PCA projection from source to target space |
| `CrossModal_PLS_SVD` | Closed-form | PLS via SVD decomposition |
| `CrossModal_PCA_PLS` | Closed-form | PCA whitening + PLS regression |
| `CrossModal_PCA_PLS_learnable` | Learned | PCA/PLS-initialized linear map, fine-tuned end-to-end |
| `CrossModal_PCA_PLS_CovProjector` | Learned | PCA+PLS + residual correction conditioned on subject covariates |
| `CrossModal_ConditionalGaussian` | Closed-form | Conditional Gaussian mapping in latent space, with optional covariates |
| `CrossModalVAE` | Learned | Variational autoencoder cross-modal mapping |
| `LatentAttnMasked` | Learned | Latent PCA backbone with optional masked FC attention residual |
| `MaskedLatentPretrainer` | Experimental | Self-supervised latent-token reconstruction pretrainer for `LatentAttnMasked` |
| `MaskedMLPPretrainer` | Experimental | Masked latent reconstruction pretrainer with linear/MLP backbone variants |
| `Sarwar2020MLP` | Learned | Fully non-linear MLP baseline with correlation-aware loss option |
| `Chen2024GCN` | Learned | Edge-level GCN baseline (`SC` graph message passing, FC edge regression) |
| `NodalGNN` | Learned | SC-conditioned GNN using subject-specific parcel node features (volume, centroid, `SC_r2t`) |
| `NodalMLP` | Learned | Graph-free node/edge baseline over anatomical features and/or subject SC-row features |
| `TestRetestPrecomputed` | Precomputed | Test-retest oracle that loads session 1/2 cached connectomes as predictions |
| `Krakencoder_precomputed` | Closed-form / Precomputed | Precomputed Krakencoder baseline (implemented by class `KrakencoderPrecomputed`) |

---

## Setup

```bash
conda env create -f kraken_env.yml
conda activate base   # kraken_env.yml declares `name: base`
wandb login   # authenticate once with your W&B API key
# required for GNN baselines
pip install torch-geometric
```

Data paths are configured in `data/hcp_dataset.py` and require local access to HCP-derived connectome and covariate files.

---

## Data Structure and Splits

### Modalities

Conn2Conn currently supports three connectome modalities:
- `SC`: structural connectivity upper-triangle vector
- `FC`: functional connectivity upper-triangle vector
- `SC_r2t`: region-to-target structural profiles converted to correlation-connectivity, then vectorized

Source modality can be single or multi-input:
- single source: `--source SC` or `--source SC_r2t`
- multi-source: `--source SC+SC_r2t`

Target modality is currently single-input only (default `FC`).

### Canonical Subject Alignment

`HCP_Base` aligns subjects across all required assets before modeling:
- metadata / partition table
- SC data
- FC data
- FreeSurfer covariates
- parcel-level node features (volume + centroid + appended `SC_r2t` tract-profile channels)

Only subjects present in all required sources are kept.

### Data Loading Modes

`HCP_Base` supports two data-loading modes:
- `manual` (default): original raw-file loading path from HCP-derived files
- `precomputed`: load cached `.npy` arrays from `precompute_cache_root` (fast startup)

Cache behavior:
- default cache root: `/scratch/asr655/neuroinformatics/Conn2Conn_data`
- default `write_manual_cache`: `False` (manual mode does **not** write cache unless explicitly enabled)
- if `data_load_mode='precomputed'` is passed through `Sim`, cache root defaults to the path above and manual cache writing is forced off

`Sim(...)` accepts:
- `data_load_mode`
- `precompute_cache_root`
- `write_manual_cache`

### Train / Val / Test Splitting

- Split identity is loaded from metadata (`train_val_test`) with `shuffle_seed` support.
- `HCP_Base` stores split indices and subject IDs under:
  - `trainvaltest_partition_indices`
  - `trainvaltest_partition_ids`
- `HCP_Partition(base, partition)` exposes one split as a PyTorch dataset (`train`, `val`, or `test`).

### Per-subject Sample Schema

Each dataset item returns:
- `x`: model input tensor (single source) or modality dict (multi-source)
- `x_modalities`: explicit dict of all source modality tensors
- `y`: target modality tensor
- `cov`: covariate dict for requested covariate sources
- `subject_id`: HCP subject identifier

For models that explicitly request parcel node features (currently `NodalGNN` and feature-enabled `NodalMLP` configs), each sample also includes:
- `node_features`: `[num_nodes, num_features]` tensor built from parcel volume, parcel centroid coordinates, and appended local `SC_r2t` node features

For `NodalMLP` configs that use subject SC rows (`use_sc_row=True`), each sample also includes:
- `sc_matrix`: dense subject SC matrix reconstructed from the source upper triangle

Node-feature loading options exposed through `HCP_Base` / `Sim`:
- `volume_feature_type`: default `volume_mm3`, optional normalized volume variant
- `centroid_feature_type`: default `centroid_mm`, optional medoid coordinates

### Covariates

Supported covariate sources:
- `fs_all`: full FreeSurfer feature set
- `fs_volumes`: selected whole-brain volume subset
- `age`: z-scored scalar
- `sex`: one-hot vector
- `race_eth`: one-hot vector

Normalization uses **training-split statistics** to avoid leakage.

### Grouping / Ordering Strategies (Evaluation)

For identifiability heatmaps and related diagnostics (`models/eval/evaluator.py`), subject order can be set to:
- `original`: preserve dataset order
- `family`: group by `Family_ID`
- `demographic`: group by `(sex × race_eth)` category
- `age`: sort from younger to older (z-scored age)

These strategies are visualization/evaluation controls and do not change the underlying train/val/test membership.

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
| `--model` | Model name (must match a YAML in `models/configs/`) |
| `--source` | Input modality: `SC`, `SC_r2t`, or `SC+SC_r2t` |
| `--target` | Output modality (default `FC`) |
| `--shuffle_seed` | Train/val/test split seed (0–4 for multi-seed evaluation) |
| `--data_load_mode` | `manual` raw loading or `precomputed` cached-array loading |
| `--use_tune` | Enable Ray Tune HPO |
| `--num_samples` | Number of Ray Tune trials |
| `--report_best_after_tune` | Re-run and fully evaluate the best trial after tuning |
| `--store_eval_md` | Save a markdown evaluation report when supported by the run/report path |

---

## Batch Jobs (SLURM)

Active multi-seed array scripts live under per-model folders in `scripts/sbatch/`:

```bash
sbatch scripts/sbatch/Sarwar2020MLP/tune_array_sarwar2020_SC_seeds.sh
sbatch scripts/sbatch/Sarwar2020MLP/tune_array_sarwar2020_SCr2t_seeds.sh
sbatch scripts/sbatch/Chen2024GCN/tune_array_chen2024gcn_SC_seeds.sh
sbatch scripts/sbatch/NodalGNN/tune_array_nodalgnn_SC_seeds.sh
sbatch scripts/sbatch/NodalGNN/run_array_nodalgnn_SC_default_seeds.sh
sbatch scripts/sbatch/NodalMLP/tune_array_nodalmlp_seeds.sh
sbatch scripts/sbatch/NodalMLP/tune_array_nodalmlp_spectral_seeds.sh
```

Launchers use an absolute repo path and absolute log paths (`results/logs/`), so they can be submitted from any directory.

Current model folders under `scripts/sbatch/`:
`Chen2024GCN/`, `CrossModal_ConditionalGaussian/`, `CrossModalPCA/`, `CrossModal_PCA_PLS/`,
`CrossModal_PCA_PLS_CovProjector/`, `CrossModal_PCA_PLS_learnable/`, `CrossModal_PLS_SVD/`,
`MaskedLatentPretrainer/`, `MaskedMLPPretrainer/`, `NodalGNN/`, `NodalMLP/`, `Sarwar2020MLP/`.

---

## Krakencoder Baseline (Precomputed)

`Krakencoder_precomputed` (class `KrakencoderPrecomputed`) is a special baseline that loads prediction matrices produced outside the training loop (from Krakencoder inference `.mat` files), then returns split-specific predictions/targets directly.

- default inference location: `krakencoder/example_data/`
- file pattern: `mydata_kraken_seed{seed}_source_{parc}.{source}.mat`
- supported source modalities: `SC`, `FC`
- run with: `python main.py --mode prod --model Krakencoder_precomputed --source SC --shuffle_seed 0`
- not tuneable: the model's search space is intentionally empty
- evaluation path uses precomputed split slicing (no neural `forward` pass)

---

## Results

All runs log to W&B project `conn2conn`. The W&B cloud is the source of truth; local `wandb/` run folders are upload staging copies.

Primary results API: `scripts/results_utils/` (active development surface).
- `records.py` — fetches best-trial prod runs from W&B (optionally direct `prod` runs for models logged outside the best-trial report path, currently `NodalGNN`), resolves `(model, source, seed)` records including missing cells, caches records, and enriches them with local Ray artifact metrics (`metrics_final.json`)
- `tables.py` — flat DataFrames, model-vs-source pivot tables, and covariate-projector / deep-model comparison tables
- `plots.py` — grouped bar charts, model scatter plots, and covariate/deep-model panels, plus a figure-type registry used by config-driven experiment scripts
- `runner.py` — shared plumbing (CLI, cache checks, table/figure writing, manifest) for the config-driven experiment scripts
- `optuna_importance.py` — hyperparameter importance from W&B tune trials: `python -m scripts.results_utils.optuna_importance --help`

Notebook surface (`scripts/notebooks/`):
- `results_scrape/nodal_decoder_importance.ipynb`
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
- Results/benchmark experiments follow `scripts/experiments/sc_type_benchmark/` (model × source grid) or `scripts/experiments/cov_projector_benchmark/` (labelled condition rows): a `config.yml`, a `run.py`, and a small tracked `records.json` snapshot of the latest W&B scrape.
- Each experiment is documented in its own folder by a write-up named after it (`<name>/<name>.md`: the question, setup, how to run it, W&B ids, results, and caveats), or by its notebook when that already describes it. `scripts/experiments/experiments_index.md` lists all experiments.

---

## Repo Layout

```
Conn2Conn/
├── main.py                          # Entrypoint: single run, Ray Tune, best-trial reporting
├── kraken_env.yml                   # Conda environment spec
├── CONTEXT.md                       # Technical reference for agents and developers
│
├── data/
│   ├── hcp_dataset.py               # HCP loading, alignment, cached-precompute path
│   ├── dataset_utils.py             # Precomputed cache loaders
│   ├── data_viz.py                  # Matrix overview plotting / gif helpers
│   └── demeaned_viz.py              # Demeaned FC / prediction visualization helpers
├── models/
│   ├── registry.py                  # Config loading, search-space conversion, model construction
│   ├── utils.py                     # Shared prediction / batch covariance helpers
│   ├── architectures/               # Model definitions grouped by architecture family
│   │   ├── crossmodal_pca_pls.py    # PCA/PLS closed-form, learnable, and cov-projector baselines
│   │   ├── crossmodal_vae.py        # VAE baseline
│   │   ├── krakencoder_precomputed.py
│   │   ├── sarwar2020_mlp.py
│   │   ├── latent_attention/        # Latent attention and conditional Gaussian models
│   │   └── graph_based/             # Chen GCN, NodalGNN, NodalMLP, graph feature builders
│   ├── configs/                     # Per-model YAML (default + search_space)
│   ├── train/                       # Training loop, Lightning wrapper, composite losses, training plots
│   └── eval/                        # Evaluator, metrics, FC distance, PCA analysis, reports, plots
├── scripts/                         # Everything that uses the library (tracked)
│   ├── results_utils/               # W&B/Ray scraping → tables → figures
│   │   ├── records.py               # Paths, W&B constants, fetch/parse → RunRecord, cache, local enrichment
│   │   ├── tables.py                # Status / metric / covtype / SC-type / cov_dl tables
│   │   ├── plots.py                 # Source bars, model scatter, cov_dl plots
│   │   ├── local_results.py         # results/local_results/ loaders + plots
│   │   └── optuna_importance.py     # Hparam importance CLI
│   ├── notebooks/                   # Interactive notebooks
│   │   ├── EDA/
│   │   ├── kraken/
│   │   ├── model_overviews/
│   │   ├── model_testing/
│   │   └── results_scrape/
│   ├── experiments/                 # Self-contained side experiments (one folder each, write-up <name>.md)
│   │   └── experiments_index.md     # One row per experiment
│   └── sbatch/                      # Per-model SLURM scripts and seed arrays
│       ├── Sarwar2020MLP/
│       ├── Chen2024GCN/
│       └── ...
├── results/                         # Generated artifacts only (gitignored)
│   ├── ray_results/                 # Best-trial reports: checkpoint, test_results.md, plots, metrics_final.json
│   ├── ray_checkpoints/             # Ray Tune storage: per-trial params, progress, W&B local copies
│   ├── ray_tmp/                     # Per-job Ray session scratch + Ray system logs
│   ├── local_results/               # Notebook / local evaluation artifacts and markdown reports
│   ├── figures/                     # Notebook-generated figures
│   └── logs/                        # SLURM stdout/stderr
├── context_packages/                # Reference material for humans and agents
│   ├── modeling/                    # Model design notes
│   ├── repo_spec_docs/              # Repo refactor specs
│   ├── schematics/                  # Figure / model schematics
│   └── T1/                          # Example T1 parcellation files
└── krakencoder/                     # Bundled KrakenEncoder codebase
```

Last updated at: 2026-09-23 EDT
