# Composite-loss cross-model benchmark (spec v3 E3)

**Status:** done 2026-10-21 (owner agent:infra) · **Campaign:** `e3_comp` ·
**Config:** [`../config_composite_loss.yml`](../config_composite_loss.yml) · **Paired reference:** E2.2 MSE
([`../mse/`](../mse/), [`../model_benchmark.md`](../model_benchmark.md))

## Question

With composite-loss weights tuned per model and seed alongside the other hyperparameters (E2.2 splits and protocol),
how much does each model gain over its own MSE-only E2.2 result, in each direction, and which metrics move?

## Design

- **Roster (D3 a):** PCA-PLS learnable, linear backbone, PCA-PLS + covariates (†, extra inputs), Sarwar MLP; both
  directions, seeds 0–4, paired with E2.2 by seed. Other E2.2 models keep their MSE rows (reference rows in the tables).
- **Loss:** MSE (weight 1) + `varmatch`, `correye_dm`, `neidist` at fixed per-model reference scales
  (`composite_scales.py`: each seed's E2.2 best config, MSE-only, batch 64; `reference_scales.yml`).
- **Weight grid (D3 b, revised 2026-10-20):** one declared set per term and direction, the same for every model:
  SC → FC {0, 0.1, 0.3, 1}, FC → SC {0, 0.1, 0.3, 0.5}. MSE-only is inside the grid.
- **Search:** E2.2's search space per model + the 3 weight keys; trials = E2.2 trials + 24 (PCA-PLS learnable and
  PCA-PLS + covariates 88, linear backbone 72, Sarwar 40); Optuna + ASHA; batch 64 for every run (v2:D4).
- **Selection (D3 d):** two best-trial reports per tune, selected on **val demeaned r** and on **val avg rank**; both
  retrained from scratch and scored on test.
- **Compute:** 40 tasks, 0.25 GPU × 4 trials per task, ~30–45 min each (well inside the cluster's 2 h low-utilisation
  window; mean GPU utilisation 31%); ≈ 25 GPU-h for the campaign.

## Results

Test metrics, mean over seeds 0–4; Δ = E3 − E2.2 on the same seed (± SE, seeds improved / 5). Full tables:
`<direction>/comparison/delta_vs_mse.md`, selected weights `<direction>/comparison/selected_weights.md`, per-selection
benchmark tables and figures `<direction>/by_{demeaned_r,avg_rank}/`.

### SC → FC

| Model | Selection | Demeaned r | Δ | Average rank | Δ | Top-1 | Δ |
|---|---|---|---|---|---|---|---|
| PCA-PLS learnable | val demeaned r | 0.093 | +0.007 ± 0.007 (4/5) | 0.736 | +0.065 ± 0.034 (4/5) | 0.042 | +0.013 (4/5) |
| | val avg rank | 0.074 | −0.012 ± 0.008 (1/5) | 0.856 | **+0.185 ± 0.027 (5/5)** | 0.105 | **+0.076 (5/5)** |
| Linear backbone | val demeaned r | 0.098 | +0.000 ± 0.003 (2/5) | 0.760 | +0.016 ± 0.011 (3/5) | 0.053 | +0.021 (4/5) |
| | val avg rank | 0.075 | −0.024 ± 0.003 (0/5) | 0.876 | **+0.132 ± 0.015 (5/5)** | 0.103 | **+0.070 (5/5)** |
| PCA-PLS + covariates † | val demeaned r | 0.105 | **+0.008 ± 0.002 (5/5)** | 0.743 | +0.047 ± 0.015 (4/5) | 0.056 | +0.017 (4/5) |
| | val avg rank | 0.084 | −0.013 ± 0.005 (1/5) | 0.867 | **+0.171 ± 0.020 (5/5)** | 0.119 | **+0.080 (5/5)** |
| Sarwar MLP | val demeaned r | 0.073 | +0.005 ± 0.005 (4/5) | 0.692 | +0.048 ± 0.027 (4/5) | 0.029 | −0.002 (3/5) |
| | val avg rank | 0.053 | −0.016 ± 0.002 (0/5) | 0.753 | **+0.108 ± 0.022 (5/5)** | 0.049 | +0.019 (4/5) |

### FC → SC

| Model | Selection | Demeaned r | Δ | Average rank | Δ | Top-1 | Δ |
|---|---|---|---|---|---|---|---|
| PCA-PLS learnable | val demeaned r | 0.146 | −0.016 ± 0.006 (1/5) | 0.855 | −0.034 ± 0.023 (2/5) | 0.100 | −0.030 (2/5) |
| | val avg rank | 0.141 | −0.021 ± 0.007 (0/5) | 0.893 | +0.004 ± 0.010 (3/5) | 0.136 | +0.007 (3/5) |
| Linear backbone | val demeaned r | 0.151 | −0.007 ± 0.006 (3/5) | 0.870 | +0.012 ± 0.028 (4/5) | 0.110 | +0.034 (3/5) |
| | val avg rank | 0.124 | −0.033 ± 0.013 (0/5) | 0.893 | +0.034 ± 0.014 (5/5) | 0.139 | +0.063 (5/5) |
| PCA-PLS + covariates † | val demeaned r | 0.206 | −0.015 ± 0.002 (0/5) | 0.935 | −0.039 ± 0.012 (1/5) | 0.342 | −0.106 (0/5) |
| | val avg rank | 0.189 | −0.032 ± 0.006 (0/5) | 0.962 | −0.011 ± 0.006 (1/5) | 0.382 | −0.066 (2/5) |
| Sarwar MLP | val demeaned r | 0.127 | −0.004 ± 0.004 (1/5) | 0.835 | −0.030 ± 0.021 (1/5) | 0.082 | −0.006 (2/5) |
| | val avg rank | 0.077 | −0.054 ± 0.009 (0/5) | 0.882 | +0.018 ± 0.012 (3/5) | 0.131 | +0.043 (3/5) |

Figures: `<direction>/comparison/tradeoff.png` (test demeaned r vs average rank: E2.2 → E3 by demeaned r → E3 by
avg rank, per model).

## Findings

1. **SC → FC: composite loss helps every model at no cost to accuracy.** Selected on val demeaned r, every model keeps
   or slightly improves its demeaned r (+0.000 to +0.008; PCA-PLS + covariates 5/5 seeds) while average rank and top-1
   rise (+0.016 to +0.065 rank). Every selected configuration switches on at least one term.
2. **SC → FC: a large identifiability gain is available for a small accuracy cost.** Selected on val avg rank,
   average rank rises by +0.11 to +0.19 and top-1 by +0.02 to +0.08 in 5/5 seeds for every model, for −0.012 to
   −0.024 demeaned r. The linear models' rank-selected rows (0.86–0.88 avg rank) beat Krakencoder's paper loss
   (0.80) on identifiability.
3. **FC → SC: composite loss does not help.** Selected on demeaned r, every model loses demeaned r (−0.004 to −0.016;
   0–3/5 seeds up) and mostly rank and top-1. Selected on avg rank, only the linear backbone gains identifiability
   (+0.034 rank, +0.063 top-1, 5/5) and only at −0.033 demeaned r. MSE-only was in every grid, so the losses mean
   that validation selection picked composite-weighted trials that generalise slightly worse; for PCA-PLS learnable
   and PCA-PLS + covariates the batch size is identical to E2.2 (64), so this is not a batch effect.
4. **Interpretation of FC → SC rank metrics (C!2):** FC → SC identifiability is dominated by parcel geometry FC
   cannot see (E1.5), so little room is left for a loss term to improve it from FC.

## Caveats

- Batch size is fixed at 64 for every run (D3 c). E2.2 used 64 for PCA-PLS learnable and PCA-PLS + covariates, but
  128 for the linear backbone and 32 for Sarwar, so their Δ also includes a batch-size change.
- † PCA-PLS + covariates sees FreeSurfer anatomy + demographics (C!1); not an input-matched comparison.
- One best trial per selection per seed; selection uses val only, test is scored once per report.
- Grids are declared, informed by v2:E1 (useful weights 0.1–1), not tuned on test.

## Layout and how to run

```
composite_loss/
├── composite_loss.md          # this file
├── grids.yml, reference_scales.yml, reference_scales_parts/   # build_composite_configs.py, composite_scales.py
└── <direction>/
    ├── by_demeaned_r/, by_avg_rank/   # run.py: records.json, tables/, figures/ (E2.2 layout)
    └── comparison/                     # composite_vs_mse.py: delta_vs_mse, selected_weights, tradeoff.png
```

```bash
export MB_CAMPAIGN_CONFIG=config_composite_loss.yml
python scripts/experiments/model_benchmark/composite_scales.py --combine        # after composite_scales.sh arrays
python scripts/experiments/model_benchmark/build_composite_configs.py [--check]
python3 scripts/experiments/model_benchmark/submit.py --direction sc2fc --stage full   # host (sbatch)
python3 scripts/experiments/model_benchmark/autopilot.py launch --direction sc2fc      # host
python scripts/experiments/model_benchmark/run.py --direction sc2fc                    # kraken container
python scripts/experiments/model_benchmark/composite_vs_mse.py
```
