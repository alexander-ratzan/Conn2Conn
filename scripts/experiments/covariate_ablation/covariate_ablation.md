# Covariate Family Ablation (spec v3 E1)

**Code:** `scripts/experiments/covariate_ablation/` — `config.yml`, `conditions.py` (conditions, frozen configs),
`run_cell.py` (train + evaluate one direction × SC metric × seed), `launch.sh` (GPU array), `prep.sh` (CPU: checks +
SC cache), `build_sc_cache.py` (E1.1), `run.py` (pilot gate, tables, figures), `checks/check_e1.py` (E1.0)
**Outputs:** `runs/<sc_metric>/<direction>/<condition>/seed<S>.json`, `results/{tables,figures}/`
**Status:** in progress (owner agent:infra)

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

_Pending._

## 5. Findings

_Pending._

## 6. Caveats

- Hyperparameters were tuned (E2.2) for the full E2.2 covariate set on the normalised target; other conditions reuse
  them. Only paired differences are interpreted.
- E2.2 training is not seeded; a rerun of the same config differs slightly (pilot reproduction tolerance 0.01).
