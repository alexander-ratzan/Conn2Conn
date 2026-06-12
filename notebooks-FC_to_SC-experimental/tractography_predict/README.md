# Tractography-predict experiments

**Question**: Does the FC↔SC asymmetry hold (or shrink) when we use richer
tractography-derived representations than the parcellated count-SC matrix?

## What we have (Glasser, 957 subjects)

| Representation | Shape per subj | What it is |
|---|---|---|
| `SC` (main-model baseline) | (360, 360) → 64,620 upper-tri | SIFT2-weighted streamline counts, vol-normalized, log1p |
| `r2t` | (360, 66) → 23,760 flattened | region-to-tract profile across 66 named bundles |
| `r2t_corr` | (360, 360) → 64,620 upper-tri | pairwise region similarity by tract profile (corrcoef of r2t rows) |

All three come from the same SIFT2 tractography pipeline. `r2t` is the
*bundle membership* representation; `r2t_corr` is the *bundle-similarity* version.

## Experiments

| Script | What it tests |
|---|---|
| `e1_source_rep_comparison.py` | Predict FC from each of 5 reps: `SC`, `r2t`, `r2t_corr`, `[SC ‖ r2t]`, `[r2t ‖ SC ‖ bv ‖ demo]`. PCA(256)→PLS(64)→inv-PCA(FC). 10 seeds. |
| `e2_asymmetry_across_reps.py` | For each rep X ∈ {SC, r2t, r2t_corr}: FC→X vs X→FC. Asymmetry ratio per seed, Wilcoxon vs 1.0. |
| `e3_marginal_r2t.py` | Paired Δ between `SC → FC` and `[SC ‖ r2t] → FC` across 10 seeds. |
| `e4_pc3_localization_on_r2t.py` | Fit PCA on r2t per seed; check stability of top 10 modes; for modes 3/4/5 read off top (region, bundle) entries to ask which named tracts dominate. Also project seed-0 SC-PC3 scores onto r2t-PCs (Spearman) to see whether SC-PC3 is the same physical mode as some r2t-PC. |
| `synthesize_tractography.py` | Aggregate E1-E3 into one comparison table + automated verdict. |

All five run under `run_all.sbatch` (cpu_short, 16 CPU, 64 GB, 1.5h walltime).
The sbatch also retries the previously-failed `retest_icc_pipeline.py` (fixed
to pass `expose_fc_sessions` via `config_overrides`).

## Inputs

- `_tract_setup.py` — wraps `further_exploration/_setup.py` and adds `r2t_flat`
  + `r2t_corr_tri` aliases on top of the standard split. Uses `base.sc_r2t_matrices`
  (loaded automatically by `HCP_Base` when `r2t_matrices.npy` is present in the SC cache).

## Outputs (all under this dir)

- `e1_source_rep_results.csv` — per (rep, seed) 6-metric panel
- `e2_asymmetry_results.csv` + `e2_asymmetry_summary.csv`
- `e3_marginal_results.csv` + `e3_marginal_summary.csv`
- `e4_r2t_pc_stability.csv` + `e4_r2t_top_bundles_per_mode.csv` + `e4_sc_pc3_to_r2t_pc_projection.csv`
- `tractography_synthesis.csv` — compact one-row-per-finding table
- `tractography_synthesis_output.txt` — printed verdict
- `findings.md` — written after the job by the driver
