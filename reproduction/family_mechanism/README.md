# Family-Structure + Mechanism Grid (F6 / F7 / F8)

Second grid pass: replicates the family-structure (F6), predictor/identifier-tradeoff (F7),
and PC-mechanism (F8) findings on **both parcellations × 10 frozen seeds**, so they stop being
Glasser-only single-run results. Faithful ports of the notebooks (canonical source):
`model_overviews/crossmodal_pca_pls_closed_form_overview.ipynb` STEP 8.1–8.3 (F6/F7) and
`further_exploration/depth1*.ipynb` (F8). The notebook is authoritative — if a helper drifts,
fix the notebook first.

## Status
- **F6 + F7 — BUILT & VALIDATED.** Aggregation reproduces the notebook's `aggregate_auc.csv`
  bit-for-bit (AUC err 1e-16, perm-p err 0, bootstrap CI err 6e-8). See tests below.
- **F8 — staged (next increment).** Port of `depth1_spectral_mechanism.ipynb` +
  `depth1.1_pc_stability_and_confounds.ipynb` (per-PC FC→PC R², per-PC family AUC, confound R²,
  cross-seed PC alignment, network enrichment via `data/atlas_info/<parc>_dseg_reformatted.csv`).

## Files
- `_fm_common.py` — shared layer. Lazy data import (`_data()`; needs torch/Torch) so the pure
  aggregation helpers run anywhere. Copies the 4 notebook-only helpers verbatim
  (`zscore_by_unrelated`, `perm_p_auc`, `bootstrap_auc`, `fdr_bh`); reuses `_setup`'s family
  helpers + predictors. `build_family_variants` / `pair_sims_for_seed` / `aggregate_family`.
- `run_f6_family.py --parc <P> --seed <S>` — one unit: 8 variants → per-pair sims → per-unit npz.
  Self-checks Glasser/seed0 pair counts against the notebook (MZ=33,DZ=13,sib=125,unrel=171).
- `finalize_fm.py` — pool all seeds per parc → `outputs/family_auc.csv`; **regression guard**
  asserts Glasser == notebook aggregate within 1e-6.
- `run_fm_unit.sbatch` — array 0–19 (2 parc × 10 seeds), 24G/4h/8CPU (same profile as main grid).
- `tests/` — see below.

## Tests (run locally, no torch / no connectome data)
```bash
python reproduction/family_mechanism/tests/test_aggregation_matches_notebook.py  # vs real notebook CSV
python reproduction/family_mechanism/tests/test_helpers.py                        # helper correctness
```
`test_aggregation_matches_notebook.py` feeds the notebook's own per-seed `.npz` (the expensive
connectome-derived pair sims) into our aggregation and asserts we reproduce `aggregate_auc.csv`
exactly — the strongest off-cluster "matches the notebook" check. `test_helpers.py` pins the 4
copied helpers against reference implementations (FDR vs statsmodels, AUC vs sklearn, etc.).

## Run on Torch
One command — submits the 20-unit array and chains finalize via `--dependency=afterok` (finalize
runs only if every unit succeeds). Nothing to poll; watch the sentinels.
```bash
cd /scratch/ans9868/Conn2Conn/reproduction/family_mechanism
bash submit_fm.sh
# progress (passive, NO squeue):
ls sentinels/DONE_fm_*.sentinel 2>/dev/null | grep -v finalize | wc -l   # /20
test -f sentinels/DONE_fm_finalize.sentinel && echo FINALIZE DONE
cat logs/fm_finalize.txt ; column -s, -t outputs/family_auc.csv | head
```
Dedicated scripts (light profile — F6/F7 has no KR sweep / no downstream / no F8):
- `run_fm_unit.sbatch` — array 0–19 `%10`, **16G / 8 CPU / 1 h** (peak RSS ~4–8 GB).
- `finalize_fm.sbatch` — single task, 8G / 30 min, runs `finalize_fm.py` in-container.
- `submit_fm.sh` — submits both with the dependency wired.

**Estimated runtime:** ~3 min/unit (Glasser) / ~5 min (4S456) — anchored on the notebook's
`seed_*.npz` timestamps. ~15–20 min wall at `%10` (2 waves) + ~3–5 min finalize, plus SLURM queue.
1 h time limit is generous (worst unit ~5 min); zero timeout risk.

## What F6/F7 produce
`outputs/family_auc.csv`: per (parcellation, variant, relation) AUC + bootstrap CI + perm-p + FDR.
- **F6** headline: `pred_SC_resid_bvdemo` sibling AUC (notebook Glasser = 0.810) vs `bvdemo_to_SC`
  baseline (0.563) — family-specific wiring beyond shared anatomy.
- **F7** is read off the same table: `pred_SC_raw` separates siblings (0.680) but
  `combined_pred_SC` collapses to chance (0.505, n.s.) — reconstruct OR identify, not both.
  (The downstream-cognition half of F7 — `combined_pred_SC` → cognition lift — is the one new
  input to add to the main downstream grid; not required for the family-collapse evidence.)
