# Composite-loss dynamics on the linear backbone

**Status:** planned (spec v2 E1) · **Owner:** agent:modeling · **Config:** [`config.yml`](config.yml)

## Question

With MSE fixed at weight 1, how do `varmatch`, `correye` and `neidist` shape the training dynamics of a simple, strong
linear probe (`CrossModal_linear_backbone`, SC → FC), and how do they trade test `demeaned_pearson` against `avg_rank`?
Which weightings are reasonable before the composite loss is extended to other models?

## Design

- **Model:** `CrossModal_linear_backbone` (frozen PCA encoder / decoder, learned affine k×k latent map), SC → FC,
  Glasser, `batch_size` 128 in every stage. Seeds 0–4. Selection on `val_demeaned_r`; test metrics are reported.
- **Stage 1 (E1.2):** 32-trial MSE-only tune per seed (packed 4 trials per GPU). The other three terms are logged as
  **monitor-only terms** (computed, never optimized), so even MSE-only training records their trajectories.
- **E1.3:** the consensus Stage 1 config is retrained per seed; each term's mean |raw| on training batches gives the
  **fixed reference scales** `c_t = s_t / s_mse` (MSE stays raw, so Stage 1's `l2_reg` stays calibrated).
- **Stage 2 (E1.4):** 16 weight combinations × 5 seeds, `loss_normalize: none`, terms divided by `c_t`
  (no EMA). All four terms are logged in every run.
- **Analysis (E1.5):** trade-off scatter (PNG + clickable HTML with each point's weights), term trajectories, loss
  composition over epochs, validation trajectories, single-term response curves, and gradient cosines between terms
  on the latent map.

## How to run

| Stage | Command | Needs |
|---|---|---|
| E1.2 | `sbatch scripts/experiments/linear_backbone/composite_loss/stage1/tune_stage1_seeds.sh` | E1.1 on `main` |
| E1.3 | runner script (to be added) | Stage 1 done |
| E1.4 | runner array (to be added) | E1.3 scales in `config.yml` |
| E1.5 | `python scripts/experiments/linear_backbone/composite_loss/run.py` (to be added) | Stage 2 done |

Stage 1 tune runs are identified by `ray_tune_id` (recorded in `config.yml`); Stage 2 runs carry the W&B tags
`composite_loss`, `composite_loss:stage2`, `combo:<id>`.

## Compute envelope

Stage 1 ≤ 4 GPU-h, E1.3 ≤ 1 GPU-h, Stage 2 ≤ 8 GPU-h (spec v2 D3). Stop conditions are listed in `config.yml`.

## Caveats

- v2:C!3 — Stage 1 searches `zscore_pca_scores`; with z-scoring on, latent diagnostics are measured in PCA space.
  Edge outputs and the edge-space composite terms are unaffected.

## Results

Pending.
