# Composite-loss grid: Krakencoder instance

**Code:** this folder: `config.yml` (cells, anchors, recipe, autonomy gates), `grid_runner.py` (plan / packed run / collect),
`launch_grid.sh`, `report.py`, `checks/` (`term_magnitudes.py`, `pilot_gate.py`). Model code: `models/architectures/krakencoder/`.
**Tables (tracked):** `tables/seed_records.csv`, `combo_summary.csv`, `epoch_history.csv.gz`, `tables/summary.{csv,md}`, `tables/noise.md`, `tables/{grid,pilot,noise}_{seed_records,epoch_history}.csv`, `tables/pilot_gate.json`
**Figures (drawn with the shared `../report.py` functions, as for the E1 instances):** `figures/tradeoff_scatter.png`, `tradeoff_interactive.html`, `term_trajectories.png`, `val_trajectories.png`, `dose_response.png` (SC → FC); the same with suffix `__fc2sc` (FC → SC). Not yet: `grad_cosine.png` (needs a gradient re-score), `loss_composition.png` (deferred). E1-schema tables `tables/seed_records.csv` (both directions), `combo_summary.csv`, `epoch_history.csv.gz` put Krakencoder on `../figures/cross_model_interactive.html` (SC → FC).
**Fits:** `results/krakencoder/cl_kraken_v1_<cell>/seed<S>/` (not tracked). Grid job `19000122` (27 tasks), pilot `18973935`, noise `18973936`.
**Status:** complete 2026-10-02 · spec v2 E2.1 · sibling of the E1 instances (`../composite_loss.md`)

---

## 1. Question

How do the identifiability terms (variance matching, correlation-identity, nearest-neighbour distance) change
Krakencoder's SC → FC and FC → SC predictions when it is retrained in the repo, and does the paper's loss sit on the
trade-off between demeaned correlation and identifiability that E1 found for the linear models?

## 2. Setup

| | |
|---|---|
| Model | Krakencoder, vendored upstream `b57e39c`, paper-default architecture and optimiser (PCA-256, latent 128 unit-norm, one linear layer per encoder/decoder, dropout 0.5); no Stage 1 tune |
| Training data | all 4 flavors (FC, SC × Glasser, 4S456Parcels) jointly, as in the paper (cross-parcellation alignment) |
| Evaluation | Glasser, both directions, our metrics (`compute_basic_regression_metrics`), test split |
| Batch / epochs | 64 (D4; paper 41) / 2000 (pilot plateau rule); checkpoints every 500 (pilot fits every 100) |
| Fixed loss | `mse.w1000 + enceye.w10 + encdist.w10 + latentsimloss.w10000` in every cell |
| Grid | grid v1 (16 cells) + paper default `correye + neidist` (reference) + 4 level-2.0 cells = 21 cells × seeds 0–4 = 105 fits |
| Weights | grid level × anchor (option B): `correye`, `neidist` anchor 1 (paper weight); `var` anchor 9.2 (level 1 gives `var` the share of the MSE term that `correye` has at its paper weight; `checks/term_magnitudes.py`) |
| Term correspondence | `var` = our `varmatch` (same formula); Krakencoder's `correye` acts in a mean-centred PCA space, so it corresponds to our **`correye_dm`**, not plain `correye`; `neidist` = ours |
| Compute | 40.7 GPU-h for the grid (27 packed jobs of 4 fits) + ~4 GPU-h pilot and noise |

## 3. Results (test, mean over 5 seeds; Δ = paired-by-seed difference from `mse_only`, ± SE)

| Cell | SC→FC demeaned r | Δ | SC→FC avg rank | Δ | FC→SC demeaned r | Δ | FC→SC avg rank | Δ |
|---|---|---|---|---|---|---|---|---|
| `mse_only` | 0.0860 | — | 0.683 | — | 0.1341 | — | 0.902 | — |
| `ce_0.5` | 0.0854 | −0.0006 ± 0.0017 | 0.738 | **+0.056** ± 0.003 | 0.1176 | **−0.017** ± 0.001 | 0.900 | −0.002 |
| `ce_1.0` | 0.0799 | −0.006 ± 0.003 | 0.802 | **+0.120** ± 0.005 | 0.1042 | **−0.030** ± 0.001 | 0.897 | −0.004 |
| `ce_2.0` | 0.0723 | −0.014 ± 0.003 | 0.841 | **+0.158** ± 0.004 | 0.0889 | **−0.045** ± 0.001 | 0.891 | −0.011 |
| `kraken_default` (paper) | 0.0801 | −0.006 ± 0.003 | 0.802 | +0.119 ± 0.005 | 0.1048 | −0.029 ± 0.001 | 0.898 | −0.004 |
| `vm_1.0` | 0.0868 | +0.001 ± 0.001 | 0.669 | −0.014 ± 0.002 | 0.1367 | +0.003 ± 0.000 | 0.885 | −0.017 |
| `nd_1.0` | 0.0861 | +0.000 | 0.685 | +0.003 | 0.1334 | −0.001 | 0.902 | 0.000 |
| `all_1.0` | 0.0871 | +0.001 ± 0.001 | 0.727 | +0.044 ± 0.002 | 0.1220 | −0.012 ± 0.001 | 0.894 | −0.008 |

All 21 cells: [`tables/summary.md`](tables/summary.md). Retrain variability ([`tables/noise.md`](tables/noise.md)):
init-seed SD at a fixed split is 0.0001 (SC→FC) / 0.0009 (FC→SC) demeaned r and 0.002 / 0.001 avg rank, 5–25×
smaller than the split-seed SD (0.0025 / 0.0036; 0.014 / 0.012), so paired cell differences above ~0.002 are real.

## 4. Findings

1. **`correye` drives everything, and the trade-off is direction-specific.** In SC → FC it buys a large identifiability
   gain (avg rank +0.12 at the paper weight, +0.16 at 2×, top-1 0.046 → 0.092) for a small demeaned-r cost (−0.006
   at 1×). In FC → SC it costs demeaned r steeply (−0.030 at 1×) and does not raise avg rank, which is already ≈ 0.90.
2. **The paper's loss is effectively `correye` alone.** `kraken_default` matches `ce_1.0` within noise in every metric,
   and `ce_nd_0.5` equals `ce_0.5`: `neidist` has no measurable effect at 0.1–2× (all |Δ| ≤ 0.0016 demeaned r, ≤ 0.006
   avg rank). Its raw value at the solution is tiny and negative (`checks/term_magnitudes.py`); why it is inert here
   (margin handling, scale, or gradient) is not established.
3. **`var` is a mild demeaned-r / identifiability lever in the opposite direction.** It raises demeaned r slightly in
   both directions (+0.001 / +0.003) and lowers avg rank (−0.014 / −0.017 at 1×); combined with `correye` (`vm_ce_0.5`)
   it gives the best SC → FC demeaned r (0.0879, +0.0019) with a moderate rank gain (+0.016).
4. **No single setting is best in both directions.** For SC → FC, the paper default is a reasonable identifiability-
   leaning choice; for FC → SC, MSE-only or `var` dominates it (demeaned r 0.134–0.137 vs 0.105 at equal avg rank).
5. **Batch size matters for `correye`.** The pilot's paper-default fit at batch 64 was 0.017 below the batch-41 parity
   fit in FC → SC val demeaned r (SC → FC unchanged); accepted as a finding, not re-run.
6. **Training dynamics:** FC → SC val demeaned r plateaus by ≈ epoch 1000; SC → FC is flat from 500 for every cell
   except MSE-only and the paper default, whose seed-0 pilot curves oscillate between checkpoints
   ([`figures/val_trajectories.png`](figures/val_trajectories.png); 4 checkpoints per fit; term curves are our edge-space definitions, not the terms Krakencoder optimises in its PCA space, and lack `correye_dm`).

## 5. Caveats

- Weights are Krakencoder-native (its terms act in its PCA-256 space next to `mse.w1000`), so dose levels are anchored
  to the paper, not to E1's scaled-term footing; compare directions and shapes with E1, not numbers.
- One training run serves both directions and all four flavors; a cell's weights apply to all 16 training paths.
- Two grid tasks (9, 24) were killed by SIGTERM during the CPU-only checkpoint scoring (GPU idle, likely the
  under-utilisation policy); their fits were already trained and were scored in CPU-only jobs. Score checkpoints in
  CPU jobs in future grids.
- Upstream input adaptation (`meanfit+meanshift`) is fit on all subjects (it ignores its subject mask); near identity
  (R² 1.000) with our inputs.

Last updated at: 2026-10-02 EDT
