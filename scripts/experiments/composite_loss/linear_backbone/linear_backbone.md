# Composite-loss dynamics on the linear backbone

**Status:** complete, grid v3, both directions (spec v2 E1 SC → FC; E3 Phase D FC → SC) · **Owner:** agent:modeling · **Config:** [`config.yml`](config.yml) ·
**Protocol:** [`../composite_loss.md`](../composite_loss.md)

## Question

With MSE fixed at weight 1, how do Var-match, Corr-eye, Demeaned corr-eye and Neighbor dist shape the training of a simple,
strong linear probe (`CrossModal_linear_backbone`, SC → FC), and how do they trade test `demeaned_pearson` against
`avg_rank`?

## Design

- **Model:** `CrossModal_linear_backbone` (frozen PCA encoder / decoder, learned affine latent map `W_mid`), SC → FC,
  batch 64 in every stage (D4). Seeds 0–4 = five family-preserving train/val/test splits. Selection on
  `val_demeaned_r`; test metrics reported.
- **Stage 1:** 24-trial MSE-only tune per seed (run once, in v1; reused for v2 and v3). Consensus config: 256 PCA
  components, no z-scoring, `l2_reg` 7.2e-5, `l1_reg` 1e-7, `lr` 4.1e-4, 100 epochs.
- **Consensus gate:** missed against the Stage 1 bests (0.0992 vs 0.1046, best-of-24 selection bias). Passed the
  re-check against the retrained per-seed best configs: 0.0992 vs 0.1004, gap 0.0013 < SE 0.0046
  (`sc2fc/tables/consensus_note.txt`).
- **Reference scales** $c_t$ (term ÷ MSE at the consensus fit; spread across seeds in brackets): Var-match 79.3 (0.6%),
  Corr-eye 4342 (0.6%), Demeaned corr-eye 705 (1.6%), Neighbor dist 103 (4.9%).
- **Grid v3:** 29 combinations × 5 seeds = 145 runs ([`../grid_v3.yml`](../grid_v3.yml)). 143 of them carried over
  from the v2 run on the same Stage 1 and splits; the two new three-term doses (0.1, 1) ran for v3. The five v2-only
  raw-Corr-eye mixtures stay in `sc2fc/runs/` and W&B (`loss_grid:v2`) but are excluded from the v3 tables and figures.

## Results

Effects are **paired by seed** (each combination minus MSE-only on the same split, mean ± SE over 5 seeds), which
removes split-to-split variance. MSE-only: test demeaned r 0.1034, avg_rank 0.782, MSE 0.0135.

| Training loss = MSE + | Δ demeaned r | Δ avg_rank | Δ test MSE | top-1 |
|---|---|---|---|---|
| Neighbor dist 0.1 | −0.010 ± 0.002 | **+0.050** ± 0.002 | +0.0001 | 0.079 |
| Neighbor dist 0.5 | −0.021 ± 0.002 | +0.051 ± 0.005 | +0.0010 | 0.083 |
| Neighbor dist 1 | −0.025 ± 0.002 | +0.038 ± 0.006 | +0.0018 | 0.082 |
| Demeaned corr-eye 0.1 | −0.017 ± 0.002 | **+0.092** ± 0.006 | +0.0000 | 0.111 |
| Demeaned corr-eye 0.5 | −0.038 ± 0.002 | **+0.098** ± 0.010 | +0.0001 | 0.123 |
| Demeaned corr-eye 1 | −0.043 ± 0.002 | +0.092 ± 0.011 | +0.0001 | 0.127 |
| Demeaned corr-eye 50 | −0.047 ± 0.002 | +0.067 ± 0.012 | +0.0003 | 0.115 |
| Var-match 0.5 + Demeaned corr-eye 0.5 | −0.035 ± 0.002 | +0.102 ± 0.010 | +0.0001 | 0.130 |
| Demeaned corr-eye 0.5 + Neighbor dist 0.5 | −0.023 ± 0.002 | +0.058 ± 0.006 | +0.0010 | 0.087 |
| All three 0.1 | −0.014 ± 0.002 | +0.071 ± 0.004 | +0.0001 | 0.089 |
| All three 0.5 | −0.023 ± 0.002 | +0.050 ± 0.006 | +0.0013 | 0.086 |
| All three 1 | −0.029 ± 0.002 | +0.030 ± 0.006 | +0.0022 | 0.079 |
| Corr-eye 1 | −0.000 ± 0.000 | +0.001 ± 0.001 | 0 | 0.061 |
| Corr-eye 5 | −0.004 ± 0.001 | −0.003 ± 0.003 | +0.0001 | 0.063 |
| Corr-eye 10 | −0.040 ± 0.003 | **−0.183** ± 0.009 | +0.0030 | 0.034 |
| Corr-eye 50 | −0.087 ± 0.003 | −0.255 ± 0.010 | +0.0281 | 0.007 |
| Var-match 0.5 | −0.003 ± 0.001 | +0.001 ± 0.002 | +0.0001 | 0.063 |
| Var-match 1 | −0.040 ± 0.003 | **−0.189** ± 0.009 | +0.0027 | 0.036 |

"All three" = Var-match + Demeaned corr-eye + Neighbor dist at equal weights. MSE-only top-1 is 0.062.
Every combination is in `sc2fc/tables/combo_summary.csv`; per-seed values are in `sc2fc/tables/seed_records.csv`.

**Findings**

1. **Demeaned corr-eye is the strongest identifiability term.** Its avg_rank gain is about twice Neighbor dist's
   (+0.09 vs +0.05), and it doubles test top-1 accuracy (0.11–0.13 vs 0.062 for MSE-only; Neighbor dist 0.08),
   **without changing test MSE**. The cost is demeaned r: −0.017 at w = 0.1, saturating near −0.045 from w ≈ 1. The
   gain peaks at w ≈ 0.5 and then slowly declines.
2. **Neighbor dist** trades about 0.01 demeaned r for +0.05 avg_rank at w = 0.1. Larger weights cost more demeaned r
   and gain nothing more. It also overfits: train MSE falls (0.0122 → 0.0103 at w = 1) while test MSE rises
   (0.0135 → 0.0153). Train avg_rank is already 0.999 under MSE-only, so the identifiability gap is a generalization
   gap.
3. **Raw Corr-eye is inert up to w ≈ 5, then collapses** like Var-match: at w = 10 its result (0.063 / 0.599) is
   nearly identical to Var-match at w = 1 (0.064 / 0.593). Raw Corr-eye mostly says "spread the predictions out", at
   about a twentieth of Var-match's strength.
4. **On this model, mixtures with Neighbor dist land near Neighbor dist.** Adding Neighbor dist to Demeaned corr-eye
   cuts both its demeaned-r cost and its avg_rank gain (+0.058). All three at 0.1 sits between the two single terms
   (−0.014 / +0.071). Var-match adds nothing at 0.5. On `pca_pls_learnable` the same mixtures keep most of
   Demeaned corr-eye's gain at half its cost; see the cross-model section of the protocol write-up.
5. **Reproducibility:** the 16 v1 combinations, rerun from the same Stage 1 on the same splits, reproduce v1 to within
   about 0.001. Examples: MSE-only 0.1033 / 0.781 (v1) vs 0.1034 / 0.782; Neighbor dist 0.5 0.0822 / 0.832 vs
   0.0826 / 0.833; Var-match 1 0.0639 / 0.592 vs 0.0636 / 0.593.

**Gradient strength on `W_mid`** (MSE-only runs; scaled gradient norm ÷ MSE gradient norm, and cosine with the MSE
gradient): Var-match 0.40 (cos 0.33), Corr-eye 0.05 (0.29), **Demeaned corr-eye 6.3 (0.19)**, Neighbor dist 13.1 (0.77).
Demeaned corr-eye is nearly orthogonal to MSE, to Var-match (−0.07) and to raw Corr-eye (−0.02). It is a genuinely
different direction, which fits its leaving MSE untouched.

## Figures (`sc2fc/figures/`)

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
| Stage 1 | `sbatch scripts/experiments/composite_loss/linear_backbone/sc2fc/stage1/tune_stage1_seeds.sh` |
| Consensus + scales (+ automatic re-check) | `sbatch scripts/experiments/composite_loss/launch_consensus.sh linear_backbone/sc2fc` |
| Grid | `sbatch --dependency=afterok:<consensus> scripts/experiments/composite_loss/launch_grid.sh linear_backbone/sc2fc` |
| Report | `sbatch --dependency=afterok:<grid> scripts/experiments/composite_loss/launch_report.sh linear_backbone` |

Generated: `state.yml`, `sc2fc/runs/`, `sc2fc/tables/`, `sc2fc/figures/`. W&B tags: `CrossModal_linear_backbone`, `loss_grid:v3` (and `loss_grid:v2` for the carried-over runs),
`composite_loss:<stage>`, `combo:<id>`. The v1 runs remain in W&B under `loss_grid:v1`.

## FC → SC (spec v2 E3 Phase D, 2026-10-02)

Same protocol and grid v3 in the reverse direction (`fc2sc/`, scaffolded from `sc2fc/`; Stage 1 stop threshold 0.14,
the SC → FC 0.09 scaled by the closed-form PCA/PLS ratio 0.143 / 0.089). Stage 1 per-seed bests 0.166–0.191; consensus
256 PCs (4/5 seeds; **the top of the 64/128/256 choices**), z-scored, lr 5.4e-4, 150 epochs; gate passed (gap −0.001,
SE 0.004). Compute: Stage 1 1.6, consensus 0.1, grid 2.9 GPU-h.

| test, 5 seeds | MSE-only | Var-match 1 | Demeaned corr-eye 1 | Neighbor dist 1 | all three 1 | best avg_rank cell |
|---|---|---|---|---|---|---|
| SC → FC (Δ dr / Δ rank) | 0.103 / 0.782 | −0.040 / −0.189 | −0.043 / **+0.092** | −0.025 / +0.038 | −0.029 / +0.030 | `vm_cedm_0.5` 0.884 (Δ dr −0.035) |
| FC → SC (Δ dr / Δ rank) | **0.167 / 0.888** | −0.008 / −0.111 | **−0.091 / −0.003** | −0.038 / −0.034 | −0.039 / −0.040 | `alldm_0.1` 0.897 (Δ dr −0.016) |

- **FC → SC is easier and nearly saturated on identifiability**: MSE-only already reaches avg_rank 0.888 (top-1 0.10).
- **The identity terms no longer buy avg_rank.** Demeaned corr-eye costs twice the demeaned r it costs in SC → FC and
  gives no rank; the best rank gain anywhere is +0.009 (`alldm_0.1`, at −0.016 demeaned r, top-1 +0.04).
- Raw Corr-eye is inert up to 10 and collapses at 20–50, as in SC → FC; Var-match is free only at 0.1.
- Figures: `fc2sc/figures/` (same set as `sc2fc/`). Caveat: the PC count may be limited by the search range (256 = max).

## Caveats

- Test metrics are taken at the last epoch (no early stopping). Validation demeaned r peaks at or near the final epoch
  for every combination, so this does not penalize any term.
- The grid holds the Stage 1 hyperparameters fixed. A composite-loss model might prefer different `lr` / `l2_reg`;
  magnitude tuning is spec v2 E2.3.
- v2:C!3: Stage 1 searched `zscore_pca_scores` (consensus: off), so latent diagnostics are in PCA space.
