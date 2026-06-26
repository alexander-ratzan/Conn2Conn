# BR-only Run — TODO / STATUS

Companion to [PLAN.md](PLAN.md). This is the live checklist — keep the status current.
**Overall status:** 🟡 BUILT + SYNCED to Torch; PILOT (Glasser seed 0) SUBMITTED & RUNNING (job 11824527_0).

Legend: ⬜ todo · 🟡 in progress · ✅ done · ⏸️ blocked/waiting · ❌ dropped

_Last updated: 2026-06-26 (module built+committed 3498a0f, synced to torch via targeted checkout, pilot submitted)_

---

## Phase 0 — Plan & alignment
- ✅ Confirm scope with Adel (BR-only downstream; carry over 10 + 8 new = 18; in-sample stays; Glasser only)
- ✅ Write `PLAN.md` (top-to-bottom)
- ✅ Write this `TODO.md`
- ⬜ **GO-AHEAD to start building** ← currently waiting on this

## Phase 1 — Build the module (no runs yet) ✅ (committed 3498a0f)
- ✅ Create `reproduction/br_imputation/` skeleton (`outputs/`, `configs/`, `README.md`, `.gitignore`)
- ✅ `run_br_unit.py`: BR imputation step (`capped_bayesian_ridge` for pred_{SC,FC}_{train(in-sample),test})
- ✅ Save BR artifacts → `outputs/artifacts/Glasser/seed{N}/` (+ `subject_ids_*`, BP-2 join)
- ✅ `BR_INPUTS` registry = the 18 names (PLAN §3)
- ✅ `build_br_input()` — covers the 8 new block names (#11–18); +bv+demo uses single bvdemo block
- ✅ Downstream loop: `bayesian_ridge_scalar` only; lift / perm-p / residualized; per-seed part CSV
- ✅ Leak classification (PLAN §9): CONTAINS_SUBJECT_INFO vs CONNECTOME_ONLY sets in `run_br_unit.py`
- ✅ `finalize_br.py`: merge parts → `downstream_br.csv` + `leak_verdict_br.csv` + `expected_cells_br.csv`
- ✅ `README.md` (what/why + links to PLAN + §6 in-sample note)
- ✅ Self-check: imports from `_grid_common` only; **zero edits to spine files/outputs**; py_compile + bash -n pass

## Phase 1.5 — Ops / HPC execution setup (sbatch on Torch) ✅
> The connectome data lives on Torch scratch, so this CANNOT run locally — it ships to the cluster.
- ✅ **ACCESS resolved:** `ssh torch` is key-based/non-interactive (BatchMode works); driving via a
      local `tmux` session + one-shot `ssh torch '...'`. Connectivity + paths verified.
- ✅ `run_br_unit.sbatch` — `cpu_short`, 8 CPU, **`--mem=16G` / `--time=02:00:00` / `--array=0-9%10`**,
      account `torch_pr_60_tandon_priority`; PARC=Glasser, SEED=IDX.
- ✅ Redirect `HOME` + `XDG_CACHE_HOME` to **`/scratch`** inside the apptainer `bash -lc`.
- ✅ `submit_br.sh` — array + `--dependency=afterok` finalize; prints sentinel watch commands.
- ✅ `finalize_br.sbatch` + sentinels (`DONE_br_*` / `ERROR_br_*`); finalize merges + verifies.
- ✅ Sync via **git** (push origin → `git fetch` + targeted `git checkout origin/adel-temp --
      reproduction/br_imputation` on torch; torch HEAD left at f8d84d1, dirty files untouched). NOT scp.
- ✅ Confirmed torch deps unchanged since f8d84d1 (`_grid_common`/`_setup`/`_tract_setup` identical;
      float64 leak fix already present). Container SIF + overlay ext3 verified.

## Phase 2 — Pilot (Glasser, seed 0)
- ✅ Submitted `sbatch --array=0 run_br_unit.sbatch` → job 11824527_0, running on cs613
- 🟡 Watching `sentinels/DONE_br_0` via background watcher (bowwnc36p) + `say` ping; no squeue polling
- ⏸️ Verify: 18×5 = 90 downstream rows, all finite, 0 LEAK_FAIL
- ⏸️ Eyeball pred_* CogCryst lifts vs spine PLS (pred_FC −0.135, pred_SC −0.010) → note direction
- ⏸️ **Decision gate:** wiring clean? proceed to full fan-out

## Phase 3 — Full (Glasser × 10 seeds)
- ⏸️ `bash submit_br.sh` → array 0–9 + afterok finalize; watch `ls sentinels/DONE_br_*.sentinel | wc -l` → /10
- ⏸️ 900 downstream rows; verify completeness vs `expected_cells_br.csv` (hard-fail on gaps)
- ⏸️ Leak verdict on merged (expect 0 LEAK_FAIL)
- ⏸️ Aggregate mean ± std over seeds; `say` ping on finalize

## Phase 4 — Analysis & writeup
- ⏸️ Compare BR-imputed vs PLS-imputed pred_* lifts (does F5 move?)
- ⏸️ Rank all 18 inputs by lift per cognition target; flag perm-p < 0.05
- ⏸️ Inspect the new combos (13/14 obs+pred; 18 everything) for any signal
- ⏸️ Short findings note in `reproduction/br_imputation/` (+ fold into MASTER_FINDINGS if real)

## Deferred / out of scope
- ❌ 4S456 parcellation (later pass if results warrant)
- ❌ OOF train imputation (orthogonal; in-sample kept on purpose — PLAN §6)
- ❌ kernel_ridge / pca_pls downstream variants (BR-only by decision)

---

## Ops cheat-sheet (Torch — standing rules + this run's config)

**Resourcing (BR-only, Glasser ×10):** `--account=torch_pr_60_tandon_priority --partition=cpu_short
--cpus-per-task=8 --mem=16G --time=02:00:00 --array=0-9%10`.
- Memory: spine Glasser unit peaked ~10 GB doing *more* work (all estimators + KR sweep + full
  downstream). BR-only does PCA(256) on the same 64,620-dim inputs (the memory driver) + cheap
  256-dim per-component BR fits → 16 GB has headroom. 10×16 = 160 GB stays under the
  ~192 GB `QOSMaxMemoryPerUser` ceiling, so `%10` runs all 10 seeds in one wave.
- Time: Glasser spine unit was < 1.5 h doing more; BR-only well under. 2 h = generous, no timeout risk.
- CPU 8 correctly sized (68–90% eff on spine). 4S456 deferred (it's the time/edge-binding parc).

**Never `$HOME` — everything to `/scratch`.** Mirror `run_unit.sbatch`: inside the apptainer
`bash -lc`, `export HOME=$ROOT/reproduction/.../wandb_home XDG_CACHE_HOME=.../xdg` (+ any wandb/conda
cache) under scratch, `mkdir -p` them. Hitting a `$HOME` quota = symptom of not redirecting.

**No polling / event-driven watching (don't spam squeue → admin emails):**
- Marker/sentinel files: `ls sentinels/DONE_br_*.sentinel | wc -l` (expect /10); `ERROR_br_*` on fail.
- Tail latest log: `ls -1t logs/br_*.txt | head -1 | xargs tail -50`.
- Wait without tight loops: `until [ -f sentinels/DONE_br_finalize.sentinel ]; do sleep 5; done`.
- `squeue` only at major milestones (≤ twice the whole run).

**Background + auto-notify:** launch long commands with `run_in_background: true` (harness pings a
`<task-notification>` on completion — no polling); read interim output from the task's output file.

**`say` ping** on pilot done / finalize done so we get an audio callback instead of staring.

**Sync via git bridge, never scp:** push laptop → torch branch, `git merge --ff-only` on torch,
push origin, laptop fetch. Container = apptainer with `:ro` overlay.

**Persistent tmux on Torch** so the session/auth/background procs survive laptop sleep / net drops;
reattach into the exact context. (This is also how an agent reaches Torch without re-auth each time.)

## Open questions / notes
- **ACCESS (blocker, Phase 1.5):** confirm the Torch reach — live tmux session I can drive, or Adel
  runs the ssh/submit. No interactive ssh/MFA autonomously (standing rule).
- BR demeaned-r only slightly > PLS → cognition lifts may barely move; flat = a valid answer.
- Keep `random_state=0` throughout for bit-reproducibility.
- Confirm 6-way `everything` block (#18) builds within Glasser memory (expected fine).
- The handoff's `memory/reference_torch_conn2conn_paths.md` is not on this machine — re-confirm
  exact scratch root (`/scratch/ans9868/Conn2Conn`?), container `.sif`, and overlay path before submit.
