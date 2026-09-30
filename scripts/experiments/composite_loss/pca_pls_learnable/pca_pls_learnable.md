# Composite-loss dynamics: replicability instance (`CrossModal_PCA_PLS_learnable`)

**Status:** planned (spec v2 E1) · **Owner:** agent:modeling · **Config:** [`config.yml`](config.yml)

## Question

Does the composite-loss protocol reproduce on a second linear-family model? Same question as the primary instance
[`composite_loss/linear_backbone`](../linear_backbone/linear_backbone.md), with
`CrossModal_PCA_PLS_learnable` (PCA/PLS-initialized encoder → mid map → decoder, SC → FC). Comparing the two instances
on identical combination ids tests whether the loss landscape is a property of the terms or of one model.

## Design

Identical protocol: grid [`../grid.yml`](../grid.yml) v1, batch 64 (D4), seeds 0–4,
selection on `val_demeaned_r`, fixed reference scales measured on this model's own Stage 1 solution, monitor-only terms
in every run, the same output schema and stop conditions. Stage 1 searches this model's own 12 keys (MSE-only;
`loss_type` fixed to `composite`, so the latent losses are excluded).

## How to run

| Stage | Command |
|---|---|
| Stage 1 | `sbatch scripts/experiments/composite_loss/pca_pls_learnable/stage1/tune_stage1_seeds.sh` |
| E1.3–E1.5 | shared protocol runner (see the primary instance) |

## Caveats

- v2:C!1 — earlier `CrossModal_PCA_PLS_learnable` sweeps ran at default regularization; this instance's Stage 1 is a
  fresh tune with `l1_reg` / `l2_reg` applied, so its results do not inherit C!1.

## Results

Pending.
