# reproduction/ — the reproducibility grid (project root, all Python)

Re-derives the Conn2Conn confirmatory claims from scratch in ONE consistent parameterized
pipeline. See `../planning/reproducibility_and_grid_plan_theory.md` (why) and
`../planning/reproducibility_and_grid_plan_todo.md` (active steps / status).

## Run order (all on Torch via sbatch — never login node; sync via git)
1. `sbatch freeze_splits.sbatch` — write the 10 frozen splits (BP-2 source of truth).
   **Then commit `splits/*.json` back to the repo** so they are definite/versioned.
2. `sbatch smoke_recon.sbatch` — smoke test (bv, bv+demo → SC/FC, seed 0, both parc, pca_pls).
   Validates data load, low-dim cap, metric panel, frozen-split assert, offline W&B, CSV.
3. (Phase B+) full reconstruction → downstream → leak → verify_completeness → summarize.

## Files
- `_grid_common.py` — capped estimators, frozen-split load+assert, flat W&B keys, CSV append.
- `freeze_splits.py` — builds `splits/seed{0..9}.json` (cross-verified across parcellations).
- `run_reconstruction_grid.py` — parameterized reconstruction runner (`--smoke` for the subset).
- `splits/` — frozen seeds (version-controlled). `outputs/` — CSV mirrors + `wandb/` offline.

## Invariants
- Splits frozen once, loaded everywhere, asserted (BP-2). `connectome+bv+demo` per-block
  scaled (BP-1, Phase B). W&B OFFLINE + flat keys; **CSV is the source of truth**. Low-dim
  inputs capped `k_src=min(256,width)`, `k_pls=min(64,k_src)`.

## W&B (offline)
No API key on Torch → `WANDB_MODE=offline`. Runs write to `outputs/wandb/`; `wandb sync` later
if a key is attached. The CSVs are authoritative regardless.

## Next steps after the grid (scope boundary)
This grid is the confirmatory engine for the **spine only** — F1–F5 + Ceiling B — on both
parcellations. It does **not** cover: **F6/F7** (family-structure & heritability of *predicted*
connectomes — a separate pass that reuses the `outputs/artifacts/.../pred_*` handoff artifacts),
**F8** (PC3 mechanism), **F9** (tractography), **F10** (nonlinear nulls — own modules, done).
Post-grid: interpret 4S456-vs-Glasser, fold numbers into `MASTER_FINDINGS.md`, then the
F6/F7 heritability pass. Full detail in
`../planning/reproducibility_and_grid_plan_todo.md` (Next Steps + Ops learnings).
