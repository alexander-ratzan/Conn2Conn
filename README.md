# Conn2Conn

Predicts one connectome modality from another on HCP-derived data (default `SC -> FC`), with both closed-form and learned cross-modal models.

## Current Functionality

- Modalities: `SC`, `FC`, and `SC_r2t`
- Supports multi-source inputs via `--source` (for example `SC+SC_r2t`), with a single target modality via `--target`
- Family-aware train/val/test partitioning with configurable `--shuffle_seed`
- Evaluation on train/val/test with optional markdown report export

## Models

Implemented in `models/models.py` and configured in `models/configs/*.yml`:

- `CrossModalPCA`
- `CrossModal_PLS_SVD`
- `CrossModal_PCA_PLS` (closed-form PCA+PLS)
- `CrossModal_PCA_PLS_learnable` (trainable PCA/PLS-initialized linear map)
- `CrossModal_PCA_PLS_CovProjector` (covariate residual projector variant)
- `CrossModalVAE`

## Run

Single run:

```bash
python main.py --mode dev --model CrossModal_PCA_PLS_learnable
```

Single run with multi-source input:

```bash
python main.py --mode prod --model CrossModal_PCA_PLS_learnable --source SC+SC_r2t --target FC
```

Tune run:

```bash
python main.py --mode prod --model CrossModal_PCA_PLS_learnable --use_tune --num_samples 10 --max_concurrent_trials 1
```

After tuning, report full best-trial metrics and write eval markdown:

```bash
python main.py --mode prod --model CrossModal_PCA_PLS_learnable --use_tune --num_samples 10 --report_best_after_tune --store_eval_md
```

## Layout

- `main.py`: entrypoint (single run + Ray Tune + best-trial reporting)
- `kraken_env.yml`: environment spec used for local runs
- `data/`: dataset loading, partitioning, preprocessing, atlas metadata
- `models/`: model definitions, Lightning module, losses, eval, configs
- `notebooks/` + top-level `*.ipynb`: exploratory and evaluation notebooks
- `krakencoder/`: bundled KrakenEncoder codebase + examples
- `sbatch/`: SLURM launch scripts grouped by model
- `checkpoints/`: saved model checkpoints
- `results/`: Ray Tune artifacts, logs, reports, and local eval outputs
- `wandb/`: Weights & Biases run logs

Assumes local access to HCP-derived datasets configured in the data loaders.
