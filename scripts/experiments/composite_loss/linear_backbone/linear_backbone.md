# Composite-loss dynamics on the linear backbone

**Status:** complete, grid v2 (spec v2 E1) · **Owner:** agent:modeling · **Config:** [`config.yml`](config.yml) ·
**Protocol:** [`../composite_loss.md`](../composite_loss.md)

## Question

With MSE fixed at weight 1, how do `varmatch`, `correye`, `correye_dm` and `neidist` shape the training of a simple,
strong linear probe (`CrossModal_linear_backbone`, SC → FC), and how do they trade test `demeaned_pearson` against
`avg_rank`?

## Design

- **Model:** `CrossModal_linear_backbone` (frozen PCA encoder / decoder, learned affine latent map `W_mid`), SC → FC,
  batch 64 in every stage (D4). Seeds 0–4 = five family-preserving train/val/test splits. Selection on
  `val_demeaned_r`; test metrics reported.
- **Stage 1:** 24-trial MSE-only tune per seed (run once, in v1; reused for v2). Consensus config: 256 PCA
  components, no z-scoring, `l2_reg` 7.2e-5, `l1_reg` 1e-7, `lr` 4.1e-4, 100 epochs.
- **Consensus gate:** missed against the Stage 1 bests (0.0992 vs 0.1046, best-of-24 selection bias). Passed the
  re-check against the retrained per-seed best configs: 0.0992 vs 0.1004, gap 0.0013 < SE 0.0046
  (`tables/consensus_note.txt`).
- **Reference scales** $c_t$ (term ÷ MSE at the consensus fit; spread across seeds in brackets): varmatch 79.3 (0.6%),
  correye 4342 (0.6%), correye_dm 705 (1.6%), neidist 103 (4.9%).
- **Grid v2:** 32 combinations × 5 seeds = 160 runs ([`../grid_v2.yml`](../grid_v2.yml)), about 2.5 GPU-h.

## Results

Effects are **paired by seed** (each combination minus MSE-only on the same split, mean ± SE over 5 seeds), which
removes split-to-split variance. MSE-only: test demeaned r 0.1034, avg_rank 0.782, MSE 0.0135.

| Combination | Δ demeaned r | Δ avg_rank | Δ test MSE |
|---|---|---|---|
| neidist 0.1 | −0.010 ± 0.002 | **+0.050** ± 0.002 | +0.0001 |
| neidist 0.5 | −0.021 ± 0.002 | +0.051 ± 0.005 | +0.0010 |
| neidist 1.0 | −0.025 ± 0.002 | +0.038 ± 0.006 | +0.0018 |
| correye_dm 0.1 | −0.017 ± 0.003 | **+0.092** ± 0.006 | +0.0000 |
| correye_dm 0.5 | −0.038 ± 0.002 | **+0.098** ± 0.010 | +0.0001 |
| correye_dm 1.0 | −0.043 ± 0.002 | +0.092 ± 0.011 | +0.0001 |
| correye_dm 50 | −0.048 ± 0.002 | +0.067 ± 0.012 | +0.0003 |
| correye 1.0 | −0.001 ± 0.000 | +0.001 ± 0.001 | 0 |
| correye 5 | −0.004 ± 0.001 | −0.003 ± 0.003 | +0.0001 |
| correye 10 | −0.040 ± 0.003 | **−0.183** ± 0.009 | +0.0030 |
| correye 50 | −0.087 ± 0.003 | −0.255 ± 0.011 | +0.0281 |
| varmatch 0.5 | −0.003 ± 0.001 | +0.001 ± 0.002 | +0.0001 |
| varmatch 1.0 | −0.040 ± 0.003 | **−0.189** ± 0.009 | +0.0027 |
| all three 0.5 (correye) | −0.022 ± 0.002 | +0.043 ± 0.005 | +0.0013 |
| all three 0.5 (correye_dm) | −0.023 ± 0.002 | +0.050 ± 0.006 | +0.0013 |

Every combination is in `tables/combo_summary.csv`; per-seed values are in `tables/seed_records.csv`.

**Findings**

1. **`correye_dm` is the strongest identifiability term.** Its avg_rank gain is about twice neidist's (+0.09 vs +0.05),
   and it doubles test top-1 accuracy (0.11–0.12 vs 0.06 for MSE-only; neidist 0.08), **without changing test MSE**. The cost is demeaned r:
   −0.017 at w = 0.1, saturating near −0.045 from w ≈ 1. The gain peaks at w ≈ 0.5 and then slowly declines.
2. **`neidist`** trades about 0.01 demeaned r for +0.05 avg_rank at w = 0.1. Larger weights cost more demeaned r and
   gain nothing more. It also overfits: train MSE falls (0.0122 → 0.0103 at w = 1) while test MSE rises (0.0135 → 0.0153). Train
   avg_rank is already 0.999 under MSE-only, so the identifiability gap is a generalization gap.
3. **Raw `correye` is inert up to w ≈ 5, then collapses** like varmatch: at w = 10 its result (0.063 / 0.599) is nearly
   identical to varmatch at w = 1 (0.064 / 0.593). This confirms the v1 gradient reading: raw correye mostly
   says "spread the predictions out", at about a twentieth of varmatch's strength.
4. **Mixtures are dominated by their strongest identity term.** When `neidist` is on, the other terms add little;
   `correye_dm` combined with `neidist` lands near `neidist` alone.
5. **Reproducibility:** the 16 v1 combinations, rerun from the same Stage 1 on the same splits, reproduce v1 to within
   about 0.001. Examples: MSE-only 0.1033 / 0.781 (v1) vs 0.1034 / 0.782; neidist 0.5 0.0822 / 0.832 vs 0.0826 / 0.833;
   varmatch 1.0 0.0639 / 0.592 vs 0.0636 / 0.593.

**Gradient strength on `W_mid`** (MSE-only runs; scaled gradient norm ÷ MSE gradient norm, and cosine with the MSE
gradient): varmatch 0.40 (cos 0.33), correye 0.05 (0.29), **correye_dm 6.3 (0.19)**, neidist 13.1 (0.77).
`correye_dm` is nearly orthogonal to MSE, to varmatch (−0.07) and to raw correye (−0.02). It is a genuinely
different direction, which fits its leaving MSE untouched.

## Figures (`figures/`)

| File | Shows |
|---|---|
| `tradeoff_scatter.png`, `tradeoff_interactive.html` | test demeaned r vs avg_rank per combination (HTML: hover or click for weights and per-seed values) |
| `dose_response.png` | paired Δ vs MSE-only along each term's weight ladder (symlog x) |
| `term_trajectories.png` | validation value of every term over epochs, factorial cells (all minimized) |
| `loss_composition.png` | each active term's share of the weighted training loss |
| `val_trajectories.png` | validation demeaned r over epochs (maximized) |
| `grad_cosine.png` | per-term gradient cosines on `W_mid` over training |

## How to run

| Step | Command |
|---|---|
| Stage 1 | `sbatch scripts/experiments/composite_loss/linear_backbone/stage1/tune_stage1_seeds.sh` |
| Consensus + scales (+ automatic re-check) | `sbatch scripts/experiments/composite_loss/launch_consensus.sh linear_backbone` |
| Grid | `sbatch --dependency=afterok:<consensus> scripts/experiments/composite_loss/launch_grid.sh linear_backbone` |
| Report | `sbatch --dependency=afterok:<grid> scripts/experiments/composite_loss/launch_report.sh linear_backbone` |

Generated: `state.yml`, `runs/`, `tables/`, `figures/`. W&B tags: `CrossModal_linear_backbone`, `loss_grid:v2`,
`composite_loss:<stage>`, `combo:<id>`. The v1 runs remain in W&B under `loss_grid:v1`.

## Caveats

- Test metrics are taken at the last epoch (no early stopping). Validation demeaned r peaks at or near the final epoch
  for every combination, so this does not penalize any term.
- The grid holds the Stage 1 hyperparameters fixed. A composite-loss model might prefer different `lr` / `l2_reg`;
  magnitude tuning is spec v2 E3.
- v2:C!3: Stage 1 searched `zscore_pca_scores` (consensus: off), so latent diagnostics are in PCA space.
