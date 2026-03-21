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
| `CrossModalVAE` | Learned | Variational autoencoder cross-modal mapping |
| `Sarwar2020MLP` | Learned | Fully non-linear MLP baseline with correlation-aware loss option |
| `Chen2024GCN` | Learned | Edge-level GCN baseline (`SC` graph message passing, FC edge regression) |
| `NodalGNN` | Learned | SC-conditioned GNN using subject-specific parcel node features (volume, centroid, `SC_r2t`) |
| `Krakencoder_precomputed` | Closed-form / Precomputed | Precomputed Krakencoder baseline (implemented by class `KrakencoderPrecomputed`) |

---

## Setup

```bash
conda env create -f kraken_env.yml
conda activate kraken
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

For models that explicitly request parcel node features (currently `NodalGNN`), each sample also includes:
- `node_features`: `[num_nodes, num_features]` tensor built from parcel volume, parcel centroid coordinates, and appended local `SC_r2t` node features

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

For identifiability heatmaps and related diagnostics (`models/eval.py`), subject order can be set to:
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
| `--store_eval_md` | Save a markdown evaluation report for the best trial |

---

## Batch Jobs (SLURM)

**Top-level seed array scripts** (run from repo root):

```bash
sbatch tune_array_sarwar2020_SC_seeds.sh
sbatch tune_array_sarwar2020_SCr2t_seeds.sh
sbatch tune_array_chen2024gcn_SC_seeds.sh
sbatch tune_array_nodalgnn_SC_seeds.sh
```

Parallel per-source scripts live in:
- `sbatch/Sarwar2020MLP/`
- `sbatch/Chen2024GCN/`
- `sbatch/NodalGNN/`
- plus existing model folders under `sbatch/`

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

All runs log to W&B project `conn2conn`.

Primary results API: `results/results_scraper.py` (active development surface).
- fetches best-trial prod runs from W&B
- resolves `(model, source, seed)` records (including missing cells)
- builds flat DataFrames and model-vs-source pivot tables
- can enrich records with local Ray artifact metrics (`metrics_final.json`)

Notebook surface: `scrape_results.ipynb` for ad hoc analysis and plotting.

```python
from results.results_scraper import (
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

## Repo Layout

```
Conn2Conn/
├── main.py                          # Entrypoint: single run, Ray Tune, best-trial reporting
├── kraken_env.yml                   # Conda environment spec
├── CONTEXT.md                       # Technical reference for agents and developers
│
├── data/                            # HCP dataset loading and partitioning
├── models/                          # Model definitions, configs, loss, eval, Lightning module
│   └── configs/                     # Per-model YAML (default + search_space)
├── results/
│   ├── results_scraper.py           # W&B results scraper
│   ├── ray_results/                 # Ray Tune trial artifacts
│   ├── ray_checkpoints/             # Best-trial checkpoints
│   └── logs/                        # SLURM stdout/stderr
├── sbatch/                          # Per-model one-off SLURM scripts
│   ├── Sarwar2020MLP/
│   ├── Chen2024GCN/
│   └── ...
├── tune_array_*.sh                  # Optional top-level array wrappers
├── scrape_results.ipynb             # Results analysis notebook
├── test_learnable_model.ipynb       # Learnable model dev notebook
├── test_proj_model.ipynb            # CovProjector dev notebook
├── test_sarwar2020_model.ipynb      # Sarwar baseline dev notebook
├── test_chen2024_model.ipynb        # Chen GCN baseline dev notebook
├── test_nodal_gnn_model.ipynb       # NodalGNN baseline dev notebook
├── notebooks/                       # Exploratory and evaluation notebooks
└── krakencoder/                     # Bundled KrakenEncoder codebase
```
