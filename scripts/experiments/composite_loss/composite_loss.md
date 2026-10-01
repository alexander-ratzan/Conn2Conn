# Composite-loss protocol

**Status:** running (spec v2 E1) · **Owner:** agent:modeling · **Grid:** [`grid.yml`](grid.yml) (v1)

## Question

With MSE fixed at weight 1, how do `varmatch`, `correye` and `neidist` shape training dynamics and trade test
`demeaned_pearson` against `avg_rank`? The protocol is run per model on identical combination ids, so the loss
landscape can be compared across configurations within a model and across models.

## Layout

```
composite_loss/
├── composite_loss.md        # this file: protocol + cross-model comparison
├── grid.yml                 # protocol grid (versioned; never edited after release)
├── checks/                  # loss-code checks for the protocol (E1.1)
└── <model>/                 # one folder per model instance
    ├── <model>.md           # instance write-up
    ├── config.yml           # model, Stage 1 config, consensus, reference scales, stop conditions
    └── stage1/              # MSE-only tune config + packed launcher
```

## Protocol v1

- **Stage 1:** MSE-only tune of the model's own search space (24 trials × seeds 0–4, packed), with `varmatch`,
  `correye`, `neidist` logged as monitor-only terms.
- **Consensus + reference scales:** one consensus config (categorical by majority, `lr` / `l2_reg` by geometric
  median; accepted within 1 SE of the per-seed bests); fixed scales `c_t = s_t / s_mse` measured on it.
  If the check fails, `protocol.py rebaseline` (`launch_rebaseline.sh`) retrains each seed's own best config on its
  seed (epochs capped at what ASHA let the trial train) and re-checks against those retrained values, which removes the
  best-of-24 selection bias. If it still fails, the consensus is accepted as a recorded fallback
  (`state.yml` `consensus_check.basis: fallback_accept`, echoed in `tables/consensus_note.txt`) so Stage 2 runs.
- **Stage 2:** the 16 grid combinations (8-cell on/off factorial at w = 0.5 + 8 dose-response points) × 5 seeds,
  Stage 1 hyperparameters fixed, `loss_normalize: none`, all four terms logged every epoch.
- **Fixed across models:** `batch_size` 64 (D4), seeds, selection on `val_demeaned_r`, output schema
  (`seed_records.csv`, `epoch_history.csv`), W&B tags (`<model>`, `loss_grid:v1`, `combo:<id>`).
- **Scope:** the grid maps the landscape with each model's Stage 1 hyperparameters held fixed; magnitude tuning for a
  final model is spec v2 E3.

## Instances

| Instance | Role | Status |
|---|---|---|
| [`linear_backbone`](linear_backbone/linear_backbone.md) | primary | Stage 1 running |
| [`pca_pls_learnable`](pca_pls_learnable/pca_pls_learnable.md) | replicability | Stage 1 queued |

## Cross-model comparison

Pending: concatenates the instances' `seed_records.csv` / `epoch_history.csv` on `combo_id`.
