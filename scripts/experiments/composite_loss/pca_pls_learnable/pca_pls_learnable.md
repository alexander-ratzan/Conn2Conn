# Composite-loss dynamics: replicability instance (`CrossModal_PCA_PLS_learnable`)

**Status:** complete, grid v3, both directions (spec v2 E1 SC → FC; E3 Phase D FC → SC) · **Owner:** agent:modeling · **Config:** [`sc2fc/config.yml`](sc2fc/config.yml), [`fc2sc/config.yml`](fc2sc/config.yml) ·
**Protocol:** [`../composite_loss.md`](../composite_loss.md)

## Question

Does the composite-loss landscape found on the linear backbone ([`../linear_backbone`](../linear_backbone/linear_backbone.md))
reproduce on a second linear-family model? `CrossModal_PCA_PLS_learnable` is a PCA encoder → learnable latent map
`W_mid` (initialized from PLS) → PCA decoder, SC → FC. Both instances run identical combination ids on the same five
splits, so each term's effect can be compared across models directly.

## Design

- **Architecture (hand-selected, fixed):** 256 source and 256 target PCA components, PLS rank 16 for the
  initialization, only `W_mid` learnable (256 × 256, about 65k parameters, the same size as the linear backbone's latent
  map), `l1_reg` 0. This replaces the 2026-10-01 tuned consensus (64 / 4 / 64, `lr` 3.8e-5 × 50 epochs), which barely
  trained: train MSE was nearly equal to test MSE, and every composite term moved it 5–10× less than the linear
  backbone. That earlier run's outputs were overwritten; its runs remain in W&B under `loss_grid:v2`.
- **Stage 1:** 16-trial MSE-only Optuna tune per seed over the optimizer only: `lr` 3e-5 to 3e-3, `l2_reg`, `dropout`,
  `max_epochs` 100–250. The epoch and lr floors rule out the barely-trained corner. `max_epochs` takes the median of
  the per-seed bests, not a majority vote.
  Per-seed bests: 0.105, **0.0898**, 0.100, 0.103, 0.111.
- **Stop-threshold deviation:** seed 1 (the hardest split for every model; linear backbone 0.0955) missed the 0.09
  Stage 1 stop by 0.0002. The threshold for this instance was lowered to 0.085 by user decision, and the chain resumed
  from the consensus step on the same Stage 1 results.
- **Consensus:** `lr` 5.4e-4, 150 epochs, `dropout` 0.25, `l2_reg` 1.6e-4. Its step budget (`lr` × epochs = 0.081) is
  about 2× the linear backbone's. The **gate passed directly**: 0.0988 vs the Stage 1 best mean 0.1019, gap 0.0031 <
  SE 0.0035. It is the first instance that did not need the re-check.
- **Reference scales** $c_t$ (spread across seeds): Var-match 77.0 (0.6%), Corr-eye 4124 (0.6%), Demeaned corr-eye 699
  (1.8%), Neighbor dist 152 (2.9%). They are close to the linear backbone's (79.3 / 4342 / 705 / 103).
- **Grid v3:** 29 combinations × 5 seeds = 145 runs ([`../grid_v3.yml`](../grid_v3.yml)), about 2.7 GPU-h; Stage 1
  about 1.3 GPU-h.

## Results

Effects are paired by seed (each combination minus MSE-only on the same split; mean ± SE over 5 seeds). MSE-only:
test demeaned r 0.1023, avg_rank 0.768, top-1 0.048, MSE 0.0135. The linear backbone's MSE-only is 0.1034 / 0.782 /
0.062 / 0.0135.

| Training loss = MSE + | Δ demeaned r | Δ avg_rank | top-1 | linear backbone Δ dm-r / Δ avg_rank |
|---|---|---|---|---|
| Neighbor dist 0.1 | −0.008 ± 0.002 | +0.063 ± 0.002 | 0.075 | −0.010 / +0.050 |
| Neighbor dist 0.5 | −0.013 ± 0.003 | +0.078 ± 0.004 | 0.093 | −0.021 / +0.051 |
| Neighbor dist 1 | −0.016 ± 0.003 | +0.069 ± 0.005 | 0.085 | −0.025 / +0.038 |
| Demeaned corr-eye 0.1 | −0.025 ± 0.003 | **+0.111** ± 0.007 | 0.121 | −0.017 / +0.092 |
| Demeaned corr-eye 0.5 | −0.041 ± 0.002 | +0.103 ± 0.008 | 0.113 | −0.038 / +0.098 |
| Demeaned corr-eye 1 | −0.045 ± 0.002 | +0.098 ± 0.009 | 0.119 | −0.043 / +0.092 |
| Demeaned corr-eye 50 | −0.047 ± 0.003 | +0.080 ± 0.011 | 0.118 | −0.048 / +0.067 |
| Demeaned corr-eye 0.5 + Neighbor dist 0.5 | **−0.021** ± 0.003 | **+0.101** ± 0.007 | 0.103 | −0.023 / +0.058 |
| All three 0.1 | **−0.018** ± 0.003 | **+0.106** ± 0.005 | 0.108 | −0.014 / +0.071 |
| All three 0.5 | −0.019 ± 0.003 | +0.099 ± 0.005 | 0.102 | −0.023 / +0.050 |
| All three 1 | −0.020 ± 0.003 | +0.076 ± 0.006 | 0.099 | −0.029 / +0.030 |
| Var-match 0.5 | −0.002 ± 0.001 | −0.020 ± 0.005 | 0.052 | −0.003 / +0.001 |
| Var-match 1 | −0.024 ± 0.002 | −0.154 ± 0.008 | 0.027 | −0.040 / −0.189 |
| Corr-eye 1 | +0.001 ± 0.001 | +0.001 ± 0.002 | 0.051 | −0.001 / +0.001 |
| Corr-eye 10 | −0.020 ± 0.002 | −0.117 ± 0.006 | 0.030 | −0.040 / −0.183 |
| Corr-eye 50 | −0.059 ± 0.003 | −0.206 ± 0.008 | 0.006 | −0.087 / −0.255 |

Test MSE changes by at most +0.0003 in every Demeaned-corr-eye and Neighbor-dist combination. All values are in
`sc2fc/tables/combo_summary.csv` (`d_*` columns).

**Findings**

1. **The landscape replicates.** Across the 28 non-baseline combinations, the two models' effects correlate at
   **0.92** (Δ demeaned r) and **0.98** (Δ avg_rank). The sign agrees in 86% and 93% of combinations. Every
   qualitative finding of the linear backbone holds:
   - Demeaned corr-eye is the strongest identity term, raising top-1 accuracy to more than twice MSE-only's
     (0.11–0.13 vs 0.048), with MSE unchanged.
   - Neighbor dist trades a little demeaned r for avg_rank.
   - Raw Corr-eye is inert up to w ≈ 2, then collapses predictions like Var-match at 1.
2. **Mixtures help on this model.** Demeaned corr-eye 0.5 + Neighbor dist 0.5 and all three at 0.1 keep nearly all of
   Demeaned corr-eye's avg_rank gain (+0.10 to +0.11) at **half its demeaned-r cost** (−0.018 to −0.021 vs −0.041). On
   the linear backbone the same mixtures land closer to Neighbor dist alone. All three at 0.1 is the best trade-off
   measured on either model.
3. **Neighbor dist is stronger and cheaper here** (+0.078 avg_rank at −0.013 demeaned r, vs +0.051 / −0.021 on the
   linear backbone). It still overfits identity: train MSE falls (0.0129 → 0.0106 at w = 1) while test MSE rises
   slightly (0.0135 → 0.0138).
4. **Gradient strength on `W_mid`** (scaled gradient norm ÷ MSE's, MSE-only runs; cosine with the MSE gradient in
   brackets): Var-match 0.27 (0.44), Corr-eye 0.04 (0.35), Demeaned corr-eye 11.0 (0.14), Neighbor dist 8.7 (0.76).
   The ordering and orders of magnitude match the linear backbone.

## Figures (`sc2fc/figures/`)

| File | Shows |
|---|---|
| `tradeoff_scatter.png`, `tradeoff_interactive.html` | test demeaned r vs avg_rank per combination (HTML: hover or click for weights, top-1 and per-seed values) |
| `dose_response.png` | paired Δ vs MSE-only along each term's weight ladder (symlog x) |
| `term_trajectories.png` | validation value of every term over epochs, factorial cells (all minimized) |
| `loss_composition.png` | each active term's share of the weighted training loss |
| `val_trajectories.png` | validation demeaned r over epochs (maximized) |
| `grad_cosine.png` | per-term gradient cosines on `W_mid` over training |

## How to run

| Step | Command |
|---|---|
| Stage 1 | `sbatch scripts/experiments/composite_loss/pca_pls_learnable/sc2fc/stage1/tune_stage1_seeds.sh` |
| Consensus + scales (+ automatic re-check) | `sbatch --dependency=afterok:<stage1> scripts/experiments/composite_loss/launch_consensus.sh pca_pls_learnable/sc2fc` |
| Grid | `sbatch --dependency=afterok:<consensus> scripts/experiments/composite_loss/launch_grid.sh pca_pls_learnable/sc2fc` |
| Report | `sbatch --dependency=afterok:<grid> scripts/experiments/composite_loss/launch_report.sh pca_pls_learnable` |

Generated: `state.yml`, `sc2fc/runs/`, `sc2fc/tables/`, `sc2fc/figures/`. W&B tags: `CrossModal_PCA_PLS_learnable`, `loss_grid:v3`,
`composite_loss:<stage>`, `combo:<id>`.

## FC → SC (spec v2 E3 Phase D, 2026-10-02)

Same protocol and grid v3 in the reverse direction (`fc2sc/`; Stage 1 stop threshold 0.13 = SC → FC 0.085 × 1.607, the
closed-form PCA/PLS ratio). Stage 1 per-seed bests 0.145–0.170; consensus lr 5.2e-5, dropout 0.30, l2 2.1e-5, 200
epochs; gate passed (gap 0.004, SE 0.005). Compute: Stage 1 1.4, consensus 0.1, grid 2.9 GPU-h.

| test, 5 seeds | MSE-only | Var-match 1 | Demeaned corr-eye 1 | Neighbor dist 1 | all three 1 | best avg_rank cell |
|---|---|---|---|---|---|---|
| SC → FC (Δ dr / Δ rank) | 0.102 / 0.768 | −0.024 / −0.154 | −0.045 / **+0.098** | −0.016 / +0.069 | −0.020 / +0.076 | `vm_cedm_0.5` 0.881 (Δ dr −0.039) |
| FC → SC (Δ dr / Δ rank) | **0.141 / 0.910** | −0.025 / −0.078 | **−0.028 / −0.007** | −0.025 / −0.023 | −0.030 / −0.034 | `ce_0.5` 0.911 (Δ dr −0.000) |

- MSE-only is (within noise) the best cell on both metrics in FC → SC; every identity term lowers both.
- **Schedule caveat:** in `fc2sc/figures/val_trajectories.png` every composite combination peaks at epochs 25–60 and
  declines, while MSE-only plateaus near 120 under the shared 200-epoch consensus, so part of the composite penalty is
  the MSE-tuned schedule. **Search-edge caveat:** all five per-seed best lr are 3.7e-5–6.6e-5 (floor 3e-5) and epochs
  cluster at 200–250 (top 250): the model wants slower, longer training than the range allowed.

## Caveats

- The architecture is hand-selected, not tuned. An MSE-only tune over architecture favoured small PLS ranks and very
  low lr: 2,617 earlier trials, and the 2026-10-01 consensus. Their scores were a little higher, but those models
  barely moved from initialization. This instance gives the loss terms a model that actually trains; its MSE-only
  baseline (0.1023) is close to the linear backbone's.
- The Stage 1 stop threshold was lowered for this instance (see Design).
- Test metrics are taken at the last epoch (no early stopping); the grid holds the Stage 1 hyperparameters fixed.
