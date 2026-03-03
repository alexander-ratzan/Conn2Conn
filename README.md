# Conn2Conn

Minimal status README for the current state of the repository. The code is active and some details may change.

## What This Repo Does

Conn2Conn predicts one connectome modality from another, primarily `SC -> FC`, using HCP-derived data.

Current workflow:

- load structural and functional connectivity matrices plus metadata
- vectorize connectomes as upper-triangle edge features
- fit a cross-modal model
- evaluate predictions on train, validation, and test splits

## Current Layout

```text
Conn2Conn/
├── main.py                # main experiment entrypoint
├── data/                  # HCP loading, splits, PCA preprocessing, utilities
├── models/                # model classes, Lightning module, losses, eval, configs
├── notebooks/             # exploratory analysis and evaluation notebooks
├── sbatch/                # SLURM launch scripts
├── results/               # Ray Tune outputs, checkpoints, logs
├── wandb/                 # local W&B run artifacts
└── krakencoder/           # bundled reference package / external comparison material
```

## Main Components

### Data

- `data/hcp_dataset.py`: dataset objects for full HCP state and train/val/test partitions
- `data/dataset_utils.py`: FC/SC loading, metadata merging, family-aware split generation, PCA summaries
- `data/data_utils.py`: connectome matrix helpers and visualization utilities

### Models

Model definitions live in `models/models.py`.

Current model families include:

- PCA projection baseline
- PCA + PLS closed-form model
- direct PLS-SVD closed-form model
- learnable PCA/PLS-initialized linear model
- cross-modal VAE

Training support:

- `models/lightning_module.py`: Lightning wrapper for learned models
- `models/loss.py`: MSE, demeaned MSE, weighted MSE, and VAE losses
- `models/eval.py`: prediction evaluation and reporting
- `models/configs/*.yml`: default configs and Ray Tune search spaces

## Running

Single run:

```bash
python main.py --mode dev --model CrossModal_PCA_PLS_learnable
```

Tune run:

```bash
python main.py --mode prod --model CrossModal_PCA_PLS_learnable --use_tune --num_samples 10
```

Cluster launch scripts are under `sbatch/`.

## Notes

- `main.py` is the real entrypoint.
- The repo currently assumes local access to HCP-derived datasets configured in the data loaders.
- The README is intentionally brief and may lag behind active development.
