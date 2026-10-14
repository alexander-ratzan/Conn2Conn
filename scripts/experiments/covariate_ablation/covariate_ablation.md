# Covariate Family Ablation (spec v3 E1)

**Code:** `scripts/experiments/covariate_ablation/` — `config.yml`, `conditions.py` (conditions, frozen configs),
`run_cell.py` (train + evaluate one direction × SC metric × seed), `launch.sh` (GPU array), `prep.sh` (CPU: checks +
SC cache), `build_sc_cache.py` (E1.1), `run.py` (pilot gate, tables, figures), `checks/check_e1.py` (E1.0)
**Outputs:** `runs/<sc_metric>/<direction>/<condition>/seed<S>.json`, `results/{tables,figures}/`
**Status:** done 2026-10-14 (owner agent:infra)

## 1. Question

1. How much does each covariate family add to `CrossModal_PCA_PLS_CovProjector` (PCA-PLS + covariates) in
   FC → SC and SC → FC?
2. Is its FC → SC lead (E2.2: test demeaned r 0.221, top-1 0.45 vs 0.162 / 0.13 for PCA-PLS learnable) mostly anatomy
   undoing the SC target's `invnodevol` node-volume normalisation (spec C!1)?

## 2. Setup

| | |
|---|---|
| Model | `CrossModal_PCA_PLS_CovProjector`, each seed's E2.2 best-trial config frozen (no tuning) |
| Families | `demo` (age, sex, race/eth), `fs_vol` (63 FS volume/global columns), `fs_area` (68 DK areas), `fs_thick` (68 DK thicknesses): an exact split of the E2.2 inputs |
| Conditions | `none` (no covariate branch), `full`, `drop_<family>` × 4, `only_<family>` × 4; `e2_ref` (= E2.2 inputs) in the pilot only |
| Projectors | frozen `fs_all` projector width split over FS families by column count (min 2); demographic projectors as frozen |
| Control | FC → SC `none` / `full` with the unnormalised SC target (`sift_radius2_count_connectivity`, log1p) |
| Data | Glasser, seeds 0–4 (E2.2 splits), precomputed caches |
| Reading | `only − none` = gain on its own; `full − drop` = unique gain; paired by seed |

## 3. How to run

```bash
sbatch scripts/experiments/covariate_ablation/prep.sh                       # E1.0 checks + E1.1 SC cache (CPU)
sbatch --job-name=e1_cov_fc2sc_invnodevol --array=0 --time=1:00:00 \
  scripts/experiments/covariate_ablation/launch.sh fc2sc invnodevol "none full e2_ref"   # pilot (and sc2fc)
python scripts/experiments/covariate_ablation/run.py gate                   # pilot gate
sbatch --job-name=e1_cov_<dir>_<metric> --array=0-4 --time=<t> \
  scripts/experiments/covariate_ablation/launch.sh <dir> <metric> all|"none full"
python scripts/experiments/covariate_ablation/run.py                        # tables + figures
```

## 4. Results

112 / 112 runs (110 grid + control, 2 pilot `e2_ref`), 2.2 GPU-h of training. Gates: cache Spearman median 0.990,
identical zero pattern (Pearson 0.68 from log1p of different count scales; user OK 2026-10-14); pilot `full` within
0.01 of E2.2 both directions. Test set, mean ± std over seeds 0–4; deltas paired by seed (n positive / 5).
Full tables: `results/tables/{summary,paired_deltas}.md`.

**FC → SC (invnodevol SC):** `none` 0.141 ± 0.007 demeaned r, top-1 0.11; `full` 0.218 ± 0.009, top-1 0.42.

| Family | only − none: demeaned r | top-1 | full − drop: demeaned r | top-1 |
|---|---|---|---|---|
| demo | +0.011 (5/5) | +0.02 | +0.002 (5/5) | +0.02 (4/5) |
| fs_vol | +0.047 (5/5) | +0.13 | +0.013 (5/5) | +0.07 |
| fs_area | **+0.052** (5/5) | **+0.17** | **+0.018** (5/5) | **+0.10** |
| fs_thick | +0.018 (5/5) | +0.07 | +0.005 (5/5) | +0.03 |
| all (full − none) | +0.077 (5/5) | +0.31 | | |

**SC → FC:** `none` 0.104 ± 0.013, `full` 0.101 ± 0.016 (full − none −0.003, 3/5); every family delta is within
±0.015 with mixed signs; top-1 stays at 0.03–0.04 throughout.

**Normalisation control (FC → SC, full − none):**

| SC target | none | full | Δ demeaned r | Δ top-1 |
|---|---|---|---|---|
| invnodevol (E2.2) | 0.141 | 0.218 | +0.077 ± 0.008 | +0.31 ± 0.05 |
| unnormalised | 0.126 | 0.171 | +0.045 ± 0.008 (5/5) | +0.11 ± 0.03 (5/5) |

Figures: `fig1_family_contributions.png`, `fig2_normalisation_control.png`, `fig3_context.png`.

## 5. Findings

1. **The covariate gain is FC → SC only, and it is regional size.** Cortical area and FS volumes each give about
   two thirds of the full gain on their own; thickness adds a little, demographics almost nothing. Each family's
   unique gain (full − drop) is much smaller than its solo gain, so area and volume largely carry the same
   (region-size) signal. In SC → FC no family helps.
2. **The node-volume normalisation explains part of it, not all.** On unnormalised SC the gain falls by ~40%
   (demeaned r) and ~60% (top-1), but stays clearly positive in every seed (+0.045, top-1 +0.11). So the E2.2 lead is
   partly the model undoing `invnodevol`, and partly anatomy predicting individual streamline counts directly
   (larger regions / interfaces carry more streamlines), which the FC backbone does not see.
3. **For C!1:** either way the extra signal comes from anatomy, not FC (the covariate branch never sees FC). The
   CovProjector stays a non-input-matched (†) reference; an FC-only comparison should use `none`, which is
   ≈ PCA-PLS learnable (0.141 vs 0.162 demeaned r).

## 6. Caveats

- Hyperparameters were tuned (E2.2) for the full E2.2 covariate set on the normalised target; other conditions reuse
  them. Only paired differences are interpreted.
- The unnormalised control reuses hyperparameters tuned on the normalised target; its `none` baseline differs, so
  only the size of the paired gain is compared, not absolute scores.
- E2.2 training is not seeded; a rerun of the same config differs slightly (pilot reproduction tolerance 0.01).
