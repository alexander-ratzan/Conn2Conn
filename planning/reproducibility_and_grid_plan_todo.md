# Reproducibility Grid — ACTIVE STEPS / TODO

Companion to `reproducibility_and_grid_plan_theory.md` (the *why*). This is the *what-now*:
the concrete build order, file-by-file, with status. Everything is **Python files** under
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

## Decision log
- 2026-06-20: grid lives at project-root `reproduction/`, all Python. Theory/TODO split.
- Splits frozen at `reproduction/splits/` (Option B; A==B verified, identical across parc).
- Smoke = bv + bv+demo (low-dim, simplest) to validate W&B+CSV plumbing first.
