# Composite-loss dynamics: CovProjector with all covariates (`CrossModal_PCA_PLS_CovProjector`)

**Status:** complete, grid v3 (spec v2 E1.9) · **Owner:** agent:modeling · **Config:** [`config.yml`](config.yml) ·
**Protocol:** [`../composite_loss.md`](../composite_loss.md)

## Question

Do covariates change how a model responds to the composite losses? This instance is the PCA/PLS learnable model of
E1.7 **plus a covariate branch** (FreeSurfer `fs_all` + age, sex, race/ethnicity → projectors → fusion, added to the
latent prediction), SC → FC. Every other setting matches E1.7, so the two instances differ only in the covariates.

## Design

- **Hand-selected config, no Stage 1** (user 2026-10-01; `fixed_consensus` in `config.yml`, consensus gate skipped):
  - **Backbone:** E1.7's 256 / 16 / 256, `W_mid` learnable. E1.7 optimizer: `lr` 5.4e-4, 150 epochs, dropout 0.25,
    `l2_reg` 1.6e-4.
  - **Covariate branch:** projectors fs_all 64, age 4, sex 4, race/ethnicity 8; MLP fusion with 64 hidden units.
- **Why hand-selected:** a first run used the model default (backbone frozen, Stage 1 tune, fallback consensus). Its
  MSE-only test demeaned r (0.100) matched the March `cov_projector_benchmark` tuned bests on the same splits (0.098),
  so tuning was not the limit; the frozen backbone was. That run is overwritten; its runs remain in W&B.
- **Reference scales** $c_t$ (spread across seeds): Var-match 77.9 (0.5%), Corr-eye 4201 (0.5%), Demeaned corr-eye
  1045 (3.9%), Neighbor dist 145 (3.0%). Gradient strength on `W_mid` (scaled gradient ÷ MSE's): 0.39 / 0.05 / 4.2 / 9.3.
- **Grid v3:** 29 combinations × 5 seeds = 145 runs, about 2.5 GPU-h.

## Results

Paired by seed (Δ vs this model's MSE-only on the same split, mean ± SE). MSE-only: test demeaned r 0.098, avg_rank
0.694, top-1 0.054, MSE 0.0135. E1.7 without covariates: 0.102 / 0.768 / 0.048.

| Training loss = MSE + | Δ demeaned r | Δ avg_rank | avg_rank | top-1 |
|---|---|---|---|---|
| Neighbor dist 0.1 / 0.5 / 1 | −0.001 / −0.006 / −0.008 | +0.027 / +0.033 / +0.020 | 0.72 / 0.73 / 0.71 | 0.064 / 0.060 / 0.055 |
| Demeaned corr-eye 0.1 | −0.008 ± 0.006 | **+0.168** ± 0.002 | 0.862 | 0.106 |
| Demeaned corr-eye 0.5 | −0.027 ± 0.007 | **+0.184** ± 0.005 | 0.879 | 0.111 |
| Demeaned corr-eye 1 | −0.031 ± 0.007 | +0.182 ± 0.005 | 0.876 | 0.115 |
| Demeaned corr-eye 50 | −0.048 ± 0.007 | +0.109 ± 0.006 | 0.804 | 0.055 |
| Var-match 0.5 + Demeaned corr-eye 0.5 | −0.024 ± 0.007 | **+0.196** ± 0.004 | **0.890** | **0.140** |
| Demeaned corr-eye 0.5 + Neighbor dist 0.5 | −0.011 ± 0.006 | +0.104 ± 0.004 | 0.798 | 0.081 |
| All three 0.1 | **−0.001** ± 0.005 | **+0.128** ± 0.002 | 0.823 | 0.079 |
| All three 0.5 / 1 | −0.003 / −0.014 | +0.096 / +0.030 | 0.79 / 0.72 | 0.077 / 0.053 |
| Var-match 0.5 / 1 | −0.012 / −0.039 | −0.020 / −0.141 | 0.67 / 0.55 | 0.052 / 0.008 |
| Corr-eye 1 / 10 / 50 | −0.005 / −0.087 / −0.083 | −0.006 / +0.017 / +0.078 | 0.69 / 0.71 / 0.77 | 0.057 / 0.021 / 0.039 |

**Findings**

1. **Covariates hurt MSE-only training at the E1.7 optimizer.** Demeaned r is −0.004 and avg_rank **−0.074** vs E1.7.
   Validation demeaned r peaks at epoch 20 (0.104) and falls to 0.083 by epoch 150, and train avg_rank is 0.99. The
   covariate branch overfits at a budget tuned for the backbone alone.
2. **Demeaned corr-eye more than recovers it.** It adds +0.17 to +0.18 avg_rank (about 1.8× its effect on E1.7) and
   +0.20 when paired with Var-match. Var-match 0.5 + Demeaned corr-eye 0.5 reaches **avg_rank 0.890 and top-1 0.140,
   the most identifiable model of any instance** (linear backbone best 0.884 / 0.130), with test MSE unchanged.
3. **All three terms at 0.1 is nearly free:** +0.128 avg_rank at −0.001 demeaned r.
4. **With a composite loss, covariates now help demeaned r.** Compared with E1.7 at the same combination, Demeaned
   corr-eye 0.1 gives 0.090 vs 0.077, and all three at 0.1 gives 0.097 vs 0.085. The identity term seems to stop the
   covariate branch from collapsing predictions toward each other.
5. **Neighbor dist is weak here** (+0.03 avg_rank) and overfits: train MSE 0.0087 vs test 0.0146 at w = 1.
6. **Replication of directions:** effects correlate with the linear backbone at 0.85 (Δ demeaned r) / 0.54 (Δ
   avg_rank), and with E1.7 at 0.70 / 0.59. The lower avg_rank correlation comes from the much larger Demeaned
   corr-eye gain and the weaker Neighbor dist effect.

## Figures (`figures/`)

As the other instances: `tradeoff_scatter.png` + `tradeoff_interactive.html`, `dose_response.png`, `term_trajectories.png`,
`loss_composition.png`, `val_trajectories.png`, `grad_cosine.png`. Cross-model: `../figures/cross_model_interactive.html`.

## How to run

No Stage 1: the config is hand-selected (`fixed_consensus`).

| Step | Command |
|---|---|
| Consensus + scales | `sbatch scripts/experiments/composite_loss/launch_consensus.sh pca_pls_covprojector` |
| Grid | `sbatch --dependency=afterok:<consensus> scripts/experiments/composite_loss/launch_grid.sh pca_pls_covprojector` |
| Report (+ cross-model page) | `sbatch --dependency=afterok:<grid> scripts/experiments/composite_loss/launch_report.sh pca_pls_covprojector` |

## Caveats

- **Optimizer not tuned for this model.** The E1.7 budget (150 epochs at `lr` 5.4e-4) overfits with covariates
  (validation peak at epoch 20). A shorter budget would likely give a stronger MSE-only baseline and a fairer covariate
  comparison. Open follow-up: a narrow optimizer-only Stage 1 (as E1.7) or a hand-picked 30–50 epoch run.
- Test metrics at the last epoch; the grid holds the hyperparameters fixed.
