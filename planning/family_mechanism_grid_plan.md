# F6 / F7 / F8 Grid Plan — Family Structure, Predictor/Identifier Tradeoff, PC Mechanism

How to extend the reproducibility grid to cover the three findings it currently does **not** touch,
so they stop being "Glasser-only, single-run, unreplicated." Companion to the main grid trio
(`reproducibility_and_grid_plan_{theory,todo,runlog}.md`). This is a **plan + time estimate**, not yet
built.

## Why these aren't in the main grid

The main grid is the confirmatory engine for the *reconstruction spine* (F1, F2, F4, F5, Ceiling B).
F6–F8 are **second-order analyses on the predicted connectomes** — they consume the grid's `pred_*`
handoff artifacts but ask different questions (heritability, identifiability, spectral mechanism).
They were prototyped once, on Glasser, in notebooks
(`notebooks-FC_to_SC-experimental/model_overviews/crossmodal_pca_pls_closed_form_overview.ipynb`
STEP 8–9, and `further_exploration/depth1*.ipynb`). Porting them to the grid = a **second grid pass**
that reuses the frozen splits + saved artifacts on **both** parcellations × 10 seeds.

## What each finding computes (verified against the notebooks)

- **F6 — heritable family signal.** For each connectome variant, build MZ / DZ / sibling / unrelated
  pairs (HCP restricted: `Family_ID`, `ZygosityGT/SR`; age-matched ±3y unrelated controls), compute
  demeaned-cosine pair similarity, then AUC(relation vs unrelated) + bootstrap CI + permutation p +
  FDR. Headline: `pred_SC_resid_bvdemo` sibling AUC ≈ 0.81 vs bv+demo baseline ≈ 0.56.
- **F7 — predictor/identifier tradeoff.** Cross-reference: `combined_pred_SC` (FC+bv+demo→SC, optimized
  for reconstruction) **wins cognition but collapses to chance on sibling AUC (≈0.505)**, while
  `pred_SC_raw` separates siblings (AUC ≈0.68). Reuses F6's family table + the downstream cognition
  table. The *only* new compute is `combined_pred_SC`'s family AUC and its cognition lift.
- **F8 — PC mechanism (exploratory).** PCA the SC train edges; per PC compute (a) FC→PC R² (BR),
  (b) family AUC, (c) confound R² vs sex+brain-volume; plus cross-seed PC stability (cosine-align to
  seed0) and network enrichment of top edges (via the dseg `network_label`). Headline: SC-PC3 is a
  modest (R²≈0.22), barely-heritable (sibling AUC≈0.58), visual/DAN-localized mode. **PC1 is an 89%
  sex+BV confound; PC2 was a single-seed false positive** — so cross-seed replication is the whole point.

## Feasibility — both blockers are clear

- ✅ **`pred_*` artifacts exist** (per parc/seed on Torch from the main grid; gitignored locally).
  `pred_SC_{train,test}.npy`, `pred_FC_{train,test}.npy`, `subject_ids_*`, `recon_per_subject.csv`.
- ✅ **Network labels exist for BOTH parcellations** — `data/atlas_info/{Glasser,4S456Parcels}_dseg_reformatted.csv`
  carry 7- and 17-network labels + MNI coords → F8 enrichment is replicable on 4S456 (removes the
  "Glasser artifact" exposure, which is the main reason to do this).
- ✅ **Family/twin loaders exist** — `data/dataset_utils.py::load_metadata()` +
  `further_exploration/_setup.py` already read the HCP restricted table on Torch.

## Proposed layout (mirrors the main grid)

```text
reproduction/family_mechanism/
  _fm_common.py          # pairs builder, demeaned-cosine sim, AUC+boot+perm, PC helpers, enrichment
  make_extra_artifacts.py# NEW per (parc,seed): combined_pred_SC matrix + residualized variants
  run_f6_family.py       # variants × relations -> family_auc parts
  run_f7_tradeoff.py     # join F6 family-AUC + combined_pred_SC cognition lift
  run_f8_pcmech.py       # per-PC R²/AUC/confound + top-edge lists
  finalize_fm.py         # merge parts -> bootstrap/FDR, cross-seed PC stability, enrichment agg, report
  run_fm_unit.sbatch     # array 0-19 (2 parc × 10 seeds), --mem=24G --time=04:00:00 --cpus-per-task=8
  outputs/, reports/
```

### New artifacts to generate first (the only genuinely-new compute)
- **`combined_pred_SC`** = FC-PCA(256) ⊕ bv ⊕ demo → SC via BR-per-target-PCA-component (BP-1 block
  path). The main grid computed its *metrics* but did **not save the predicted matrix** — F6/F7 need
  the matrix. ~a few min/unit (same cost as one recon cell, plus disk for the .npy).
- **Residualized variants** (`*_resid_bvdemo`): regress bv+demo out of each connectome's edges
  (cheap closed-form). Needed for F6's headline `pred_SC_resid_bvdemo`.

## Grid dimensions

| Pass | per-unit work (×20 units = 2 parc × 10 seeds) |
|---|---|
| extra artifacts | build combined_pred_SC + 4 residualized variants |
| F6 | ~8 variants × {MZ,DZ,sibling} AUC + 1000-boot + 1000-perm |
| F7 | join (no new family compute) + combined_pred_SC → 3 cog targets (BR) |
| F8 | 10 PCs × {FC→PC R² (BR), family AUC, confound R²} + top-edge lists |
| finalize (×1) | merge → FDR/CI, cross-seed PC cosine-alignment, network enrichment, reports |

## Time & resource estimate

Reuses frozen splits + saved `pred_*` (free); same machine profile as the main grid.

| Component | per unit (Glasser) | per unit (4S456) |
|---|---|---|
| extra artifacts (combined_pred_SC + resid) | ~3–5 min | ~5–8 min |
| F6 family (boot+perm dominate) | ~5–8 min | ~8–15 min |
| F7 (join + combined cognition) | ~1–2 min | ~1–2 min |
| F8 PC mechanism (incl. 100k-dim PCA on 4S456) | ~4–8 min | ~8–15 min |
| **per-unit total** | **~15–25 min** | **~25–40 min** |

- **20 units, embarrassingly parallel, well under the 4h CPU cap** (worst unit ≈40 min ≪ 4h).
- **Wall-clock:** at ~10-wide concurrency (the `QOSMaxMemoryPerUser` ceiling at 24G) → 2 waves +
  finalize ≈ **1.5–2 hours wall**. **Total compute ≈ 8–10 unit-hours.**
- **Memory:** same as main grid (~10–15 GB peak; 4S456 higher). `--mem=24G` is right-sized.
- Net: **one afternoon, same provisioning recipe as the main grid** (§7 of the runlog).

## Build effort (coding, separate from run time)

Most logic exists in the notebooks; the work is porting to deterministic grid form + adding 4S456.
Estimate **~1–1.5 days**: `_fm_common.py` (pairs/AUC/PC helpers from STEP 8 + depth1) is the bulk;
`make_extra_artifacts.py` is small (the block predictor already exists in `_tract_setup`); the three
runners are thin; `finalize_fm.py` reuses the merge/verify pattern. The one fiddly piece is the F8
**edge→region-pair→network** mapping, which must respect each parcellation's edge ordering (use the
dseg row order = connectome region order; verify against `recon_per_subject` alignment).

## Risks / watch-items
- **Restricted-data access on Torch** (HCP1200_RESTRICTED.csv on a different scratch path) — confirm
  read access from the grid account before launching; the notebooks already read it, so it should hold.
- **F8 is genuinely small** (R²≈0.22, AUC≈0.58) and already burned one false positive (PC2). Report it
  with cross-seed bootstrap CIs and treat 4S456 **disagreement** as the informative outcome.
- **Decide the family-pair population** (test-set only vs full sample) and freeze it — the notebook
  used per-seed pools; keep it identical across parcellations (BP-2 discipline) for clean replication.
- **F7's `combined_pred_SC` cognition** is a new downstream input not in the main grid — add it here,
  don't retrofit the main grid (keep the spine grid frozen).
```
