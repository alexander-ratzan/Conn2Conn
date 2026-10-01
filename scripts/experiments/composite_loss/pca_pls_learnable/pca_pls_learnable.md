# Composite-loss dynamics: replicability instance (`CrossModal_PCA_PLS_learnable`)

**Status:** complete, grid v2 (spec v2 E1) · **Owner:** agent:modeling · **Config:** [`config.yml`](config.yml) ·
**Protocol:** [`../composite_loss.md`](../composite_loss.md)

## Question

Does the composite-loss landscape found on the linear backbone ([`../linear_backbone`](../linear_backbone/linear_backbone.md))
reproduce on a second linear-family model? `CrossModal_PCA_PLS_learnable` is a PCA/PLS-initialized encoder → mid map
→ decoder (SC → FC). Both instances run identical combination ids on the same five splits, so a term's effect can be
compared across models directly.

## Design

- **Protocol:** grid v2 ([`../grid_v2.yml`](../grid_v2.yml), 32 combinations × 5 seeds), batch 64 (D4), seeds 0–4,
  selection on `val_demeaned_r`, fixed reference scales measured on this model's own consensus fit.
- **Stage 1:** 32-trial MSE-only Optuna tune per seed over 11 keys. `learn_mid` is fixed to True, because `W_mid` is
  the model's main learnable component and the gradient diagnostics are measured on it.
  Per-seed bests: 0.100, 0.095, 0.120, 0.095, 0.132 (all above the 0.09 stop).
- **Consensus config** (majority vote / geometric median over the five per-seed bests): 64 source and 64 target PCA
  components, **4 PLS components**, only `W_mid` learnable, `dropout` 0.22, `l2_reg` 6.1e-5, **`lr` 3.8e-5, 50 epochs**.
- **Consensus gate:** missed against the Stage 1 bests (0.0983 vs 0.1083). Passed the re-check against the retrained
  per-seed bests: 0.0983 vs 0.1060, gap 0.0077 < SE 0.0092. That margin is thin, and it passes only because the
  seeds disagree widely; see the caveats.
- **Reference scales** $c_t$ (spread across seeds): varmatch 72.5 (0.6%), correye 3981 (0.5%), correye_dm 915 (1.2%),
  neidist 193 (1.8%).
- **Compute:** Stage 1 about 1.7 GPU-h; grid 160 runs, about 1.6 GPU-h.

## Results

Paired by seed (each combination minus MSE-only on the same split, mean ± SE over 5 seeds). MSE-only: test
demeaned r 0.0938, avg_rank 0.699, MSE 0.0135.

| Combination | Δ demeaned r | Δ avg_rank |
|---|---|---|
| neidist 0.1 / 0.5 / 1.0 | −0.0003 / −0.0011 / −0.0015 | +0.005 / +0.009 / +0.010 |
| correye_dm 0.1 / 0.5 / 1.0 | −0.0000 / −0.0021 / −0.0030 | +0.008 / **+0.016** / +0.016 |
| correye_dm 50 | −0.0044 ± 0.0012 | +0.015 ± 0.002 |
| correye 1.0 / 10 / 50 | +0.0001 / −0.0013 / −0.0055 | +0.001 / +0.009 / +0.005 |
| varmatch 0.5 / 1.0 | −0.0001 / −0.0011 | +0.007 / +0.009 |
| all three 0.5 (correye / correye_dm) | −0.0013 / −0.0019 | +0.011 / +0.016 |

Paired SEs: 0.0002–0.0012 for demeaned r and 0.0003–0.0021 for avg_rank. Test MSE changes by at most 0.0001 in every
combination. All values are in `tables/combo_summary.csv` (`d_*` columns).

**Findings**

1. **The direction of the trade-off replicates; its size does not.** Every identity term buys avg_rank at the cost of
   demeaned r, and `correye_dm` is again the strongest (+0.016 avg_rank, about 1.7× neidist), followed by `neidist`.
   But the effects are about 5–10× smaller than on the linear backbone.
2. **No collapse at large weights.** On the linear backbone, varmatch 1.0 and correye ≥ 10 collapse avg_rank
   (−0.18). Here they slightly *raise* it (+0.009).
3. **The likely reason is optimization budget, not the loss terms.** The consensus fit barely moves from its PLS
   initialization. Its step budget is roughly 20× smaller than the linear backbone's (`lr` × epochs = 0.0019 vs 0.041),
   and its latent map is rank 4. Train and test MSE are almost equal (0.0133 vs 0.0135), and train avg_rank is only
   0.82 (linear backbone: 0.999). So the composite terms can only nudge a near-PLS solution.
4. **Gradient strength on `W_mid`** (scaled gradient norm ÷ MSE's, MSE-only runs): varmatch 0.82, correye 0.09,
   correye_dm 7.1, neidist 7.2. The ordering matches the linear backbone (0.40 / 0.05 / 6.3 / 13.1), so the scaling
   behaves the same way. The difference is in how far the optimizer can move.

## Figures (`figures/`)

Same set as the primary instance: `tradeoff_scatter.png` + `tradeoff_interactive.html`, `dose_response.png` (paired Δ,
symlog x), `term_trajectories.png`, `loss_composition.png`, `val_trajectories.png`, `grad_cosine.png`.

## How to run

| Step | Command |
|---|---|
| Stage 1 | `sbatch scripts/experiments/composite_loss/pca_pls_learnable/stage1/tune_stage1_seeds.sh` |
| Consensus + scales (+ automatic re-check) | `sbatch --dependency=afterok:<stage1> scripts/experiments/composite_loss/launch_consensus.sh pca_pls_learnable` |
| Grid | `sbatch --dependency=afterok:<consensus> scripts/experiments/composite_loss/launch_grid.sh pca_pls_learnable` |
| Report | `sbatch --dependency=afterok:<grid> scripts/experiments/composite_loss/launch_report.sh pca_pls_learnable` |

## Caveats

- **The consensus is a weak summary for this model.** The five per-seed best configs disagree on almost every key:
  PLS components 4–32, `lr` 1.0e-5 to 1.1e-3, target PCA 64–256. Four of five seeds found their best trial after
  trial 16 of 32, so Stage 1 may still be undersampled. The majority vote landed on the smallest, slowest
  configuration.
- Because of (3), this instance tests **whether the composite terms move a near-PLS solution**, not whether the loss
  landscape replicates on a well-trained second model. A fairer replication would hold a larger training budget fixed
  (for example seed 0's best: `lr` 1.1e-3, 16 PLS components). That is untested here and belongs to spec v2 E3 or a
  v3 consensus rule.
- v2:C!1 does not apply: this Stage 1 is a fresh tune with `l1_reg` / `l2_reg` applied.
