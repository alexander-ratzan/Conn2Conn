# Composite-loss dynamics: CovProjector with all covariates (`CrossModal_PCA_PLS_CovProjector`)

**Status:** complete, grid v3 (spec v2 E1.9) · **Owner:** agent:modeling · **Config:** [`config.yml`](config.yml) ·
**Protocol:** [`../composite_loss.md`](../composite_loss.md)

## Question

Do covariates change how a model responds to the composite losses? This instance is the PCA/PLS learnable model of
E1.7 **plus a covariate branch** (FreeSurfer `fs_all` + age, sex, race/ethnicity → projectors → fusion, added to the
latent prediction), SC → FC. Every other setting matches E1.7 except the training length (30 epochs, its own
validation optimum), so the two instances differ in the covariates and each runs at its own budget.

## Design

- **Hand-selected config, no Stage 1** (user 2026-10-01; `fixed_consensus` in `config.yml`, consensus gate skipped):
  - **Backbone:** E1.7's 256 / 16 / 256, `W_mid` learnable. E1.7 optimizer (`lr` 5.4e-4, dropout 0.25, `l2_reg`
    1.6e-4), but **30 epochs**.
  - **Covariate branch:** projectors fs_all 64, age 4, sex 4, race/ethnicity 8; MLP fusion with 64 hidden units.
- **Why 30 epochs** (user 2026-10-02): the 150-epoch MSE-only runs overfit. Their per-seed validation peaks were at
  epochs 20–33, and the mean curve peaked near 28. E1.7 itself peaks near epoch 140, so each model runs at its own
  budget.
- **History:**
  1. Model default (backbone frozen, Stage 1 tune, fallback consensus): MSE-only test demeaned r 0.100, matching the
     March `cov_projector_benchmark` (0.098).
  2. E1.7 backbone at 150 epochs: overfit.

  Both are overwritten; their runs remain in W&B.
- **Reference scales** $c_t$ (spread across seeds): Var-match 77.6 (0.7%), Corr-eye 4279 (0.5%), Demeaned corr-eye
  1141 (3.3%), Neighbor dist 141 (2.6%). Gradient strength on `W_mid` (scaled gradient ÷ MSE's): 0.56 / 0.06 / 2.7 / 9.7.
- **Grid v3:** 29 combinations × 5 seeds = 145 runs, about 1 GPU-h.

## Results

Paired by seed (Δ vs this model's MSE-only on the same split, mean ± SE). The E1.7 columns are the same combination
without covariates.

| Training loss = MSE + | Δ demeaned r | Δ avg_rank | demeaned r | avg_rank | top-1 | E1.7: demeaned r / avg_rank / top-1 |
|---|---|---|---|---|---|---|
| nothing (MSE only) | – | – | **0.110** | 0.685 | 0.039 | 0.102 / 0.768 / 0.048 |
| Demeaned corr-eye 0.1 | **+0.009** ± 0.008 | **+0.084** ± 0.006 | **0.119** | 0.769 | 0.052 | 0.077 / 0.878 / 0.121 |
| Demeaned corr-eye 0.5 | −0.016 ± 0.009 | +0.126 ± 0.005 | 0.094 | 0.811 | 0.079 | 0.061 / 0.871 / 0.113 |
| Demeaned corr-eye 1 | −0.023 ± 0.010 | **+0.134** ± 0.010 | 0.087 | 0.819 | 0.066 | 0.057 / 0.866 / 0.119 |
| Neighbor dist 0.1 | +0.003 ± 0.005 | +0.043 ± 0.007 | 0.112 | 0.728 | 0.051 | 0.094 / 0.831 / 0.075 |
| Neighbor dist 0.5 / 1 | −0.012 / −0.018 | +0.102 / +0.115 | 0.097 / 0.091 | 0.787 / 0.800 | 0.071 / 0.068 | 0.089 / 0.846 / 0.093 (0.5) |
| All three 0.1 | +0.001 ± 0.006 | +0.077 ± 0.006 | 0.111 | 0.762 | 0.055 | 0.085 / 0.874 / 0.108 |
| All three 0.5 / 1 | −0.014 / −0.022 | +0.125 / +0.132 | 0.096 / 0.088 | 0.810 / 0.817 | 0.072 / 0.075 | 0.084 / 0.867 / 0.102 (0.5) |
| Var-match 0.5 + Demeaned corr-eye 0.5 | −0.019 ± 0.008 | +0.107 ± 0.007 | 0.090 | 0.792 | 0.059 | 0.064 / 0.881 / 0.128 |
| Var-match 0.5 / 1 | −0.021 / −0.034 | −0.028 / −0.084 | 0.089 / 0.075 | 0.657 / 0.601 | 0.046 / 0.015 | 0.101 / 0.749 / 0.052 (0.5) |
| Corr-eye 1 / 10 / 50 | −0.003 / −0.097 / −0.094 | −0.014 / −0.032 / −0.074 | 0.106 / 0.013 / 0.015 | 0.671 / 0.653 / 0.611 | 0.049 / 0.021 / 0.010 | |

**Findings**

1. **Covariates trade identifiability for edge-wise fidelity.** Under MSE they give the highest demeaned r of any
   instance (0.110 vs E1.7's 0.102) but lower avg_rank (0.685 vs 0.768). The covariate branch adds a signal shared by
   subjects with similar demographics and anatomy: each prediction moves toward its own target, but also toward the
   predictions of similar subjects.
2. **The identity terms recover avg_rank but not to E1.7's level.** Demeaned corr-eye and Neighbor dist add +0.08 to
   +0.13. The best avg_rank here is 0.819 (Demeaned corr-eye 1) against E1.7's 0.88, and the best top-1 is about 0.08
   against E1.7's 0.13. At the same combination the covariate model keeps higher demeaned r (0.09–0.12 vs 0.06–0.09).
3. **Demeaned corr-eye 0.1 improves both metrics:** demeaned r 0.119, the highest of any cell in any instance, and
   avg_rank +0.084 over its MSE-only fit.
4. **Neighbor dist is about as strong as Demeaned corr-eye here** (+0.10 to +0.12 at 0.5–1), unlike on E1.7. It still
   overfits identity (train MSE 0.0106 vs test 0.0141 at w = 1).
5. **The 150-epoch result was confounded.** Its "most identifiable model" (avg_rank 0.890, top-1 0.140) came from
   identity terms acting over 5× more training steps on an overfit model. At a non-overfit budget the covariate model
   is less identifiable than E1.7.
6. **Replication of directions:** effects correlate with the linear backbone at 0.81 (Δ demeaned r) / 0.80 (Δ
   avg_rank), and with E1.7 at 0.61 / 0.89.

## Figures (`sc2fc/figures/`)

As the other instances: `tradeoff_scatter.png` + `tradeoff_interactive.html`, `dose_response.png`, `term_trajectories.png`,
`loss_composition.png`, `val_trajectories.png`, `grad_cosine.png`. Cross-model: `../figures/cross_model_interactive.html`.

## How to run

No Stage 1: the config is hand-selected (`fixed_consensus`).

| Step | Command |
|---|---|
| Consensus + scales | `sbatch scripts/experiments/composite_loss/launch_consensus.sh pca_pls_covprojector/sc2fc` |
| Grid | `sbatch --dependency=afterok:<consensus> scripts/experiments/composite_loss/launch_grid.sh pca_pls_covprojector/sc2fc` |
| Report (+ cross-model page) | `sbatch --dependency=afterok:<grid> scripts/experiments/composite_loss/launch_report.sh pca_pls_covprojector` |

## Caveats

- **Budget chosen from validation curves at a fixed learning rate.** Epochs 30 is the MSE-only optimum; a full
  optimizer tune (lr × epochs) is not done. The identity-loss cells peak at a similar epoch (Demeaned corr-eye 0.5: 27).
- Test metrics at the last epoch; the grid holds the hyperparameters fixed.
