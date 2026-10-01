# Composite-loss dynamics: CovProjector with all covariates (`CrossModal_PCA_PLS_CovProjector`)

**Status:** complete, grid v3 (spec v2 E1.9); consensus accepted as a **recorded fallback** (see caveats) ·
**Owner:** agent:modeling · **Config:** [`config.yml`](config.yml) · **Protocol:** [`../composite_loss.md`](../composite_loss.md)

## Question

How does a covariate-conditioned linear model respond to the composite losses? The model adds a learned covariate
branch (FreeSurfer `fs_all` + age, sex, race/ethnicity → projectors → fusion) to a PCA/PLS backbone, SC → FC.

## Design

- **Model default** (`models/configs/CrossModal_PCA_PLS_CovProjector_SC_fs_all_demo.yml`; user 2026-10-01): the
  PCA/PLS backbone is **frozen** (`W_mid` at its PLS fit), so only the covariate projectors and the fusion network learn.
  Every composite term therefore acts through the covariate path.
- **Stage 1:** 24-trial MSE-only tune per seed over PCA / PLS sizes, projector sizes, fusion type, dropout, `lr`,
  epochs (median consensus). Per-seed bests 0.106, 0.108, 0.113, 0.103, 0.122. Stop threshold 0.085, as E1.7.
- **Consensus:** PCA 128 / 128, PLS 8, smallest projectors (fs_all 32), MLP fusion (64 hidden), dropout 0.08,
  `lr` 1.3e-4, 150 epochs.
- **Reference scales** $c_t$ (spread across seeds): Var-match 71.9 (0.7%), Corr-eye 4225 (0.5%), Demeaned corr-eye 1004
  (2.6%), Neighbor dist 141 (2.0%).
- **Gradient diagnostics** on the fusion network's output layer (`cov_fusion_net.4.weight`, since `W_mid` is frozen):
  scaled gradient ÷ MSE's, Var-match 0.95, Corr-eye 0.12, Demeaned corr-eye 3.3, Neighbor dist 9.7.
- **Grid v3:** 29 combinations × 5 seeds = 145 runs, about 2.2 GPU-h; Stage 1 about 1.6 GPU-h.

## Results

Paired by seed (Δ vs MSE-only on the same split, mean ± SE). MSE-only: test demeaned r 0.0998, avg_rank 0.697,
top-1 0.040, MSE 0.0136.

| Training loss = MSE + | Δ demeaned r | Δ avg_rank | top-1 |
|---|---|---|---|
| Neighbor dist 0.1 / 0.5 / 1 | −0.004 / −0.013 / −0.025 | +0.013 / **+0.048** / +0.040 | 0.034 / 0.043 / 0.041 |
| Demeaned corr-eye 0.1 / 0.5 / 1 | −0.002 / −0.009 / −0.018 | +0.022 / **+0.051** / +0.051 | 0.035 / 0.053 / 0.046 |
| Demeaned corr-eye 50 | −0.060 ± 0.005 | **−0.098** ± 0.010 | 0.018 |
| Demeaned corr-eye 0.5 + Neighbor dist 0.5 | −0.017 ± 0.004 | **+0.058** ± 0.006 | 0.050 |
| All three 0.1 | **+0.000** ± 0.003 | +0.036 ± 0.007 | 0.042 |
| All three 0.5 | −0.019 ± 0.004 | +0.052 ± 0.005 | **0.063** |
| Var-match 0.5 / 1 | −0.027 / −0.022 | **−0.068 / −0.085** | 0.028 / 0.009 |
| Var-match 0.5 + Demeaned corr-eye 0.5 | −0.021 ± 0.004 | +0.004 ± 0.005 | 0.022 |
| Corr-eye 1 / 5 / 10 | −0.001 / −0.019 / −0.084 | −0.010 / −0.054 / −0.101 | 0.037 / 0.024 / 0.013 |

**Findings**

1. **Direction replicates, size is smaller.** Demeaned corr-eye and Neighbor dist again buy avg_rank (+0.05 at 0.5,
   about half the linear models' gain) at a small demeaned-r cost. Effects correlate with the linear backbone at 0.82
   (Δ demeaned r) / 0.68 (Δ avg_rank), and with PCA/PLS learnable at 0.66 / 0.73.
2. **The collapse regime starts earlier.** Var-match already hurts at 0.5 (−0.068 avg_rank; the linear models need 1).
   Raw Corr-eye hurts from 5, and Demeaned corr-eye turns harmful at 50, which it never did on the linear models.
   Through a small MLP covariate path, the strong "spread the predictions" gradients over-shoot.
3. **All three terms at 0.1 is free:** +0.036 avg_rank at no demeaned-r cost (+0.000 ± 0.003). The most accurate
   identity model here is all three at 0.5 (top-1 0.063, 1.6× MSE-only).
4. **Var-match + Demeaned corr-eye cancels:** the pair's avg_rank gain (+0.004) is far below Demeaned corr-eye alone
   (+0.051). On the linear models the same pair gained +0.10 to +0.11.

## Figures (`figures/`)

As the other instances: `tradeoff_scatter.png` + `tradeoff_interactive.html`, `dose_response.png`, `term_trajectories.png`,
`loss_composition.png`, `val_trajectories.png`, `grad_cosine.png`. Cross-model: `../figures/cross_model_interactive.html`.

## How to run

| Step | Command |
|---|---|
| Stage 1 | `sbatch scripts/experiments/composite_loss/pca_pls_covprojector/stage1/tune_stage1_seeds.sh` |
| Consensus + scales (+ automatic re-check) | `sbatch --dependency=afterok:<stage1> scripts/experiments/composite_loss/launch_consensus.sh pca_pls_covprojector` |
| Grid | `sbatch --dependency=afterok:<consensus> scripts/experiments/composite_loss/launch_grid.sh pca_pls_covprojector` |
| Report (+ cross-model page) | `sbatch --dependency=afterok:<grid> scripts/experiments/composite_loss/launch_report.sh pca_pls_covprojector` |

## Caveats

- **Consensus fallback.** The consensus mixes each key's majority value across seeds. For this 8-key, interacting search
  that mix trains worse than any seed's own best: consensus mean val 0.080 vs the retrained per-seed bests 0.101 (gap
  0.021 ≈ 2.7 SE; seed 3 0.054). It was accepted as a recorded fallback (`tables/consensus_note.txt`, `state.yml`), so
  Stage 2 ran. Paired effects are valid comparisons against this model's own MSE-only fit, but they describe a
  below-tuned CovProjector, and its absolute position on the cross-model page sits low. **Revisit:** rerun with one
  hand-selected config (seed 4's best, val 0.122, or seed 2's, 0.113), as was done for E1.7.
- **Backbone frozen** (model default): the composite terms act only through the covariate branch, so magnitudes are
  not directly comparable with the instances that learn `W_mid`.
- Test metrics at the last epoch; Stage 1 hyperparameters held fixed across the grid.
