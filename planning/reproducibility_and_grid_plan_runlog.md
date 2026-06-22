# Reproducibility Grid — RUN LOG / RESOURCES / TIMINGS

Third companion to `reproducibility_and_grid_plan_theory.md` (the *why*) and
`reproducibility_and_grid_plan_todo.md` (the *what-now*). This is the **operational record**:
what the grid actually cost to run — wall-clock, memory, CPU, partition limits, node contention,
and every recovery action — so a future re-run (or reviewer) knows exactly what to provision.

All compute on **NYU Torch** (account `torch_pr_60_tandon_priority`), apptainer container with
`:ro` overlay, conda env per `reference_torch_conn2conn_paths`. Sync via git bridge; never scp,
never poll squeue.

---

## 1. What was computed (scope)

| Block | Cells | Shape |
|---|---|---|
| Reconstruction | 2,640 | 2 parc × 10 seeds × 12 input→target pairs × 11 estimator-variants |
| Downstream | 11,000 | 2 parc × 10 seeds × 10 inputs × 5 targets × 11 estimator-variants |
| **Total** | **13,640** | verified 13,640/13,640 present + finite; 0 LEAK_FAIL |

- **11 estimator-variants** = `pca_pls`(1) + `bayesian_ridge`(1) + `kernel_ridge` 3×3 (gamma_mult{0.5,1,2} × alpha{0.1,1,10}) = 9.
- **2 parcellations**: Glasser (360 regions; FC/SC width 64,620 edges), 4S456Parcels (456 regions; ~103,740 edges, +60% over Glasser).
- **10 seeds**, frozen splits (`reproduction/splits/seed{0..9}.json`), parcellation-independent.
- Result HEAD at completion: grid verified at `428ecae`; handoff/W&B tooling at `2f5f01a`.

## 2. Resource footprint (MEASURED, not requested)

| Quantity | Glasser | 4S456Parcels |
|---|---|---|
| **Peak RSS** | ~10 GB | ~14.4 GB |
| **Wall-clock per unit** (1 parc × 1 seed, full recon+downstream+leak) | < 1.5 h | ~2–3 h |
| **CPU efficiency** (8 cores) | 68–90% | 68–90% |

- **Final sbatch config** (`run_unit.sbatch`): `--partition=cpu_short --time=04:00:00 --mem=24G --cpus-per-task=8`, array `0-19` (one task per parc×seed; `SLURM_ARRAY_TASK_ID` → PARC/SEED).
- **Memory was massively over-asked initially (48 GB).** Measured peak ≤ 14.4 GB → cut to **24 GB**, which roughly **doubled concurrency** under the per-user memory QOS. (See §4.)
- **8 CPU is correctly sized** (68–90% efficiency); not worth changing.
- The slow part is the **PCA fit on the 103,740-dim 4S456 inputs** — this is why 4S456 binds the time limit. See the caching optimization in §6.

## 3. Partition / scheduler constraints (NYU Torch — hard limits learned the hard way)

- **CPU jobs are capped at 4h cluster-wide** by an NYU submit plugin. `--time=5h` and `--time=8h`
  were **both rejected** ("CPU job setup is not valid") on *both* `cpu_short` and `cpu_prem`.
  → settled on `cpu_short` / `04:00:00`. 4h covers both parcellations with margin (4S456 ~2–3h).
- **`--exclude=<node>` is blocked by NYU policy** — rejected with "Please report back to hpc@nyu.edu".
  Cannot dodge a known-bad node directly; the only lever is **cancel + resubmit** and hope for a
  different node.
- **`QOSMaxMemoryPerUser`** throttles total concurrent memory per user. At 48 GB/job this capped us
  to 2–4 jobs wide; at 24 GB/job concurrency roughly doubled.
- Account/partition: `torch_pr_60_tandon_priority`, `cpu_short` (used), `cpu_prem` (also 4h-capped).

## 4. Timeline & node-contention log (what actually happened)

The grid is **embarrassingly parallel** (20 independent units). Total useful compute ≈ 20 units ×
~1–3 h, but wall-clock was dominated by **scheduler queueing + a few pathological nodes**, not by
the math. Contention was handled by cancel+retry — never `scancel -u`, only specific job IDs.

- **First full pass:** 15/20 units completed cleanly. **5 incomplete:**
  - 3 stuck on **`cl012`** — a pathologically slow node (~50× slower; cells that take minutes took hours).
  - 2 were **4S456** units that legitimately needed ~2–3 h and tipped a self-imposed 3h watch limit
    (the user's instinct "give those more time" was correct for these two).
- **Recovery whack-a-mole** (cl012 → cs601 → cs619, etc.):
  - **Unit 9** stuck on `cs601` (1 cell in 40 min) → cancel + retry → recovered on `cs603`.
  - **Unit 16** (4S456, seed 6) was losing the 4h race *during the downstream stage* (recon already
    saved) → cancelled, then ran **`run_downstream_only.sbatch 4S456Parcels 6`** to reuse the saved
    recon + handoff artifacts and run only downstream+leak → recovered on `cs606`.
    This **avoided a ~2.5 h recon redo** — the per-unit-CSV + saved-artifact design made it cheap.
  - `--exclude=cl012` attempted twice as a backfill; **silently failed** (policy rejection) before
    it was caught — do not rely on it.
- **Watcher false-exit** once on a stale `finalize` sentinel → cleared stale sentinels, relaunched.
- **Net:** every gap was backfilled by re-running only the missing parc×seed unit (not the whole
  grid), then `verify_completeness.py` confirmed 13,640/13,640.

## 5. Pipeline stages per unit (where the time goes)

Each `run_unit.sbatch` task, for its (parc, seed):
1. **Reconstruction** (`run_reconstruction_grid.py`) — 12 pairs × 11 variants → `outputs/parts/`.
2. **Handoff artifacts** (`make_handoff_artifacts.py`) — `pred_{SC,FC}_{train,test}.npy`,
   `recon_per_subject.csv`, `split_index.json` under `outputs/artifacts/{parc}/seed{seed}/`.
   *(These `.npy` are large and gitignored — regenerated, consumed locally by downstream.)*
3. **Downstream** (`run_downstream_grid.py`) — 10 inputs × 5 targets × 11 variants; the
   **longest stage on 4S456** (this is what timed out unit 16).
4. **Leak checks** (`run_leak_checks.py`) — per-unit verdicts.

Finalize (`finalize.sbatch`, run once after all units): `gen_expected_cells` → `merge_parts` →
`verify_completeness` (hard-fail on any missing) → leak on the **merged** downstream → `summarize`
(writes `reports/reproduction_findings.md`).

**Backfill a single timed-out unit cheaply:** `sbatch run_downstream_only.sbatch <PARC> <SEED>`
reuses saved recon + handoff and runs only downstream+leak (the trick that saved unit 16).

## 6. FUTURE OPTIMIZATION — cache the PCA compression (do this before any full re-run)

The single biggest speedup available, and **bit-identical** if keyed correctly.

- **Problem:** every cell re-fits PCA from scratch. The *same input's* PCA is recomputed **11×**
  (reconstruction variants) and up to **55×** (downstream: 5 targets × 11 variants). The
  103,740-dim 4S456 PCA is exactly the slow part.
- **Fix:** cache the fitted PCA / latents (`Z_train`/`Z_test`, tiny) in an in-process dict per unit,
  keyed by `(input, k[, NaN-mask])`; estimators consume cached latents and do only their own step.
- **Bit-identical** if keyed by the target NaN-mask (PCA deterministic, `random_state=0`) → a cached
  re-run is a pure speedup, validatable cell-for-cell against this grid.
- **Estimated gain:** ~2–4× off downstream, **~30–50% off total wall-time**; would have made
  4S456 seed-6 downstream *minutes* instead of >1h (no timeout).

## 7. Provisioning recipe for a clean re-run (TL;DR)

```bash
# Per-unit array job (20 units = 2 parc × 10 seeds)
sbatch run_unit.sbatch            # cpu_short, --time=04:00:00, --mem=24G, --cpus-per-task=8, array 0-19
# Backfill any straggler (reuses saved recon+handoff):
sbatch run_downstream_only.sbatch <PARC> <SEED>
# After all 20 complete:
sbatch finalize.sbatch            # merge → verify_completeness (hard-fail) → leak → summarize
```

- Budget **~24 GB / 8 CPU / 4 h** per unit; expect **Glasser <1.5 h, 4S456 ~2–3 h**.
- Concurrency is gated by `QOSMaxMemoryPerUser`, not by your array width — keep `--mem` tight.
- Wall-clock is queue- and contention-bound, not compute-bound; the 20 units are independent.
- Watch via **passive file sentinels** (`sentinels/`, `DONE_*`/`ERROR_*`), **never `squeue` polling**.

## 8. Sync / W&B status

- **CSV is the source of truth.** The full grid ran `--no-wandb` (no API key on Torch).
  Outputs kept in git: `outputs/{reconstruction,downstream,leak_verdict}.csv`,
  `configs/expected_cells.csv`, `splits/*.json`, `reports/*.md`. Large `.npy` artifacts gitignored.
- To populate W&B, replay the CSVs **from a machine with a key** (the Mac):
  `wandb login && python upload_to_wandb.py` → one run, 3 interactive Tables + headline scalars.
- All three locations (laptop = origin = Torch) were in sync at completion.
