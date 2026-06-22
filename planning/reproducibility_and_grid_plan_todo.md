# Reproducibility Grid — ACTIVE STEPS / TODO

Companion to `reproducibility_and_grid_plan_theory.md` (the *why*) and
`reproducibility_and_grid_plan_runlog.md` (the *what-it-cost*: timings, resources, contention).
This is the *what-now*: the concrete build order, file-by-file, with status. Everything is **Python files** under
`reproduction/` at the **project root** (not buried in `notebooks-FC_to_SC-experimental/`).
All compute on Torch via **sbatch** (never login node); sync via **git** (never scp).

## Layout (project root)

```text
Conn2Conn/reproduction/
  _grid_common.py            # shared: capped predict, frozen-split load+assert, flat W&B keys, CSV append
  freeze_splits.py           # build the 10 frozen splits ONCE -> splits/seed{0..9}.json
  run_reconstruction_grid.py # reconstruction runner (parameterized; smoke = bv/bv+demo)
  run_downstream_grid.py     # (later) downstream runner — joins on subject_id
  run_leak_checks.py         # (later)
  verify_completeness.py     # (later) diff actual CSV vs expected_cells.csv
  summarize_*.py             # (later)
  configs/
    grid.yml                 # (later) full claim-driven rows
    expected_cells.csv       # (later) completeness ground truth
  splits/                    # FROZEN seeds — the BP-2 source of truth (version-controlled)
    seed0.json ... seed9.json
    SPLITS_MANIFEST.json
  outputs/                   # CSV mirrors (source of truth) + wandb/ offline dir
  *.sbatch
  README.md
```

## Status legend: ⬜ todo · 🟡 in progress · ✅ done

### Phase A — foundation (THIS PASS)
- ✅ Rename theory doc → `reproducibility_and_grid_plan_theory.md`; create this TODO.
- 🟡 `_grid_common.py` — capped PCA→PLS (low-dim cap), `set_parcellation`, frozen-split
  load+assert (BP-2), flat W&B metric keys, CSV append, `git_commit`/`config_hash`.
- 🟡 `freeze_splits.py` + `freeze_splits.sbatch` — write `splits/seed{0..9}.json`
  {train/val/test subject IDs} from `base.metadata_df`. Parcellation-independent (verified).
  **Must run on Torch, then commit the generated JSON back to the repo** (definite/versioned).
- 🟡 `run_reconstruction_grid.py` (smoke subset) — inputs `{bv, bv+demo}` × targets `{SC, FC}`
  × estimator `pca_pls`, 1 seed, both parcellations. Validates: data load, the cap, the metric
  panel, frozen-split assert, **offline W&B flat logging**, CSV mirror.
- 🟡 `smoke_recon.sbatch` + run on Torch; inspect CSV + `wandb/` offline run.

### Phase B — full reconstruction
- ⬜ Add estimators: `bayesian_ridge` (exists: `br_per_component_predict`, cap k_src),
  `kernel_ridge` (RBF, 3×3 bandwidth×alpha → 9 rows).
- ⬜ Add inputs: `FC`, `SC` (asymmetry), `demo`, oracle `FC→FC`/`SC→SC`,
  `connectome+bv+demo` (**BP-1: per-block scaling via `_blocks_to_latents`**).
- ⬜ Write the **handoff artifacts** per (parc, seed): `pred_{SC,FC}_{train,test}` (fixed
  PCA→PLS, train via in-sample), `recon_per_subject` (6 metrics, per-subject), `split_index`.
- ⬜ Per-cell write-time assertion (sentinel+reason for expected NaN).

### Phase C — downstream + leak
- ⬜ `run_downstream_grid.py` — targets Cog{Total,Fluid,Cryst}+sex/age; inputs
  bv+demo / obs_FC / obs_SC / obs_FC+obs_SC / pred_SC / pred_FC / connectome+bv+demo.
  **Hard-assert** handoff artifacts exist; **join on `subject_id`** (BP-2).
- ⬜ `run_leak_checks.py` — hard-fail sex>0.99 / age>0.85; `combined_pred_*` exempt+flag.

### Phase D — completeness + launch + summarize
- ⬜ `configs/grid.yml` (explicit claim rows) → generate `configs/expected_cells.csv`
  (per-task valid (input,target) pairs; KR 3×3 = 9).
- ⬜ `verify_completeness.py` — diff actual CSV vs expected_cells; hard-fail on any missing.
- ⬜ Smoke test on **4S456** (worst case) → launch full grid via sbatch.
- ⬜ `summarize_*.py` → `reports/reproduction_findings.md`; inspect **4S456 F1–F5 first**.

## Operational invariants (do not violate)
- Splits are FROZEN ONCE, loaded everywhere, asserted (never silently re-derived) — BP-2.
- `connectome+bv+demo` per-block scaled — BP-1.
- W&B OFFLINE, flat keys; **CSV is the source of truth**.
- Low-dim inputs capped: `k_src=min(256, width)`, `k_pls=min(64, k_src)`.
- All compute on sbatch; sync via git bridge; no squeue polling.

## Next Steps (post-grid) — what the grid does and does NOT cover

The reproducibility grid is the confirmatory engine for the **spine only**: F1 (asymmetry),
F2 (dissociation), F3 (imputation utility), F4 (FC cognition), F5 (baseline + SC underperforms),
and **Ceiling B** (FC→FC / SC→SC oracles). Once 20/20 is verified + committed:

1. **Interpretation pass** — confirm the 4S456 F1–F5 cells track Glasser (the genuinely-new
   cross-parcellation evidence); flag any surprise.
2. **Fold grid numbers into `MASTER_FINDINGS.md`** — replace older ad-hoc-run figures with the
   single consistent grid pass (retires the "numbers from different runs" correctness debt).
3. **NOT covered by this grid (separate passes / existing modules):**
   - **F6 / F7 — family structure & heritability of *predicted* connectomes** (MZ/DZ/sibling
     AUC, predictor-vs-identifier tradeoff). The grid *produced* the `pred_*` handoff artifacts
     but did not run the family-pair analysis on them — that's a **new pass that reuses the
     artifacts** (`outputs/artifacts/{parc}/seed{seed}/pred_*`). Likely the next build.
   - **F8 (PC3 mechanism)**, **F9 (tractography r2t)**, **F10 (nonlinear nulls)** — own modules,
     already complete; the grid does not re-run them.
4. **Deferred niceties:** bootstrap CIs on headline numbers (F3 ratios, F4 fractions, F5 lifts);
   reconcile **Ceiling A** (0.49 cross-session reproducibility) vs **Ceiling B** (FC→FC≈0.672,
   SC→SC oracle) in `sanity_checks/noise_sanity_check/findings_noise.md` now that B is computed.

## Ops learnings (NYU Torch — for any future full re-run)
- **Memory was over-asked:** peak RSS ~10 GB (Glasser) / ~14.4 GB (4S456) vs 48 GB requested.
  Fixed runner to **`--mem=24G`** → ~2× concurrency under the per-user memory QOS
  (`QOSMaxMemoryPerUser` had throttled us to 2–4 wide). **8 CPU is correctly sized** (68–90% eff).
- **CPU jobs are capped at 4h cluster-wide** by an NYU submit plugin (5h/8h rejected with
  "CPU job setup is not valid", on both cpu_short and cpu_prem). 4h covers both parcellations.
- **`--exclude` is blocked by NYU policy** — can't dodge a bad node directly; use cancel+retry.
- **4S456 runs ~2–3h vs Glasser <1.5h** (+60% edges) — it's the time-limit-binding parcellation.
- Per-unit CSVs + `verify_completeness` make stragglers cheap to backfill (only re-run the gaps).
- **FUTURE OPTIMIZATION — cache the PCA compression (one per parc·seed·input, not per cell).**
  Every cell currently re-fits PCA from scratch; the same input's PCA is recomputed
  **11× (reconstruction variants)** and up to **55× (downstream: 5 targets × 11 variants)** —
  and the 103,740-dim 4S456 PCA is exactly the slow part. Fix: cache the fitted PCA / latents
  (`Z_train`/`Z_test`, tiny) in an in-process dict per unit, keyed by `(input, k[, NaN-mask])`;
  estimators consume the cached latents and do only their own step. **Bit-identical** if keyed by
  the target NaN-mask (PCA is deterministic, `random_state=0`) → a cached re-run is a pure
  speedup, validatable cell-for-cell vs this grid. Est. **~2–4× off downstream, ~30–50% off total
  wall-time**; would have made 4S456 s6's downstream minutes instead of >1h (no timeout). Do this
  before any future full re-run.

## Decision log
- 2026-06-20: grid lives at project-root `reproduction/`, all Python. Theory/TODO split.
- Splits frozen at `reproduction/splits/` (Option B; A==B verified, identical across parc).
- Smoke = bv + bv+demo (low-dim, simplest) to validate W&B+CSV plumbing first.
- 2026-06-21: full grid = 13,640 cells (2,640 recon + 11,000 downstream); reproduced
  F1/F4/F5/Ceiling-B on both parcellations. Resources right-sized (mem 48G→24G).
