# Repo Spec v3

**Purpose:** v3 work: task-FC prediction in both directions and parcellations (E0), joint multi-condition FC → SC
(E2, infrastructure I5), composite-loss and bidirectional infrastructure for the next benchmark (I1–I3), and a late
covariate ablation (E1). New experiments are added once
their scope is set by the user.
**Status:** active · **Started:** 2026-10-04 · **Predecessor:** [`spec_doc_v2.md`](spec_doc_v2.md) (closed 2026-10-04) ·
**Format:** [`spec_conventions.md`](spec_conventions.md)

**Contents:** [Status](#status) · [1. Conventions](#1-conventions) · [2. Experiments](#2-experiments) ·
[3. Infrastructure](#3-infrastructure) · [4. Carried over from v2](#4-carried-over-from-v2) · [5. Backlog](#5-backlog-not-scheduled) ·
[6. Change log](#6-change-log)

To add an experiment, append a section under §2 (template: [`spec_conventions.md`](spec_conventions.md) §4) and add a
row to the status table.

## Status

| ID | Title | Status | Depends on | Owner |
|---|---|---|---|---|
| E0 | Which task FC best predicts SC, and which is best predicted from SC (both directions, both parcellations) | done 2026-10-07 | — | agent:infra |
| E1 | Covariate family ablation for PCA-PLS + covariates, both directions, + unnormalised-SC control | done 2026-10-14 | D2 | agent:infra |
| E2 | Joint task-FC → SC: do the 7 task-FC blocks add to rest FC? (E2.1 raw, E2.2 predictive residual) | done 2026-10-12 | I5 | agent:infra |
| E3 | Composite-loss cross-model benchmark: v2:E2.2 rerun with composite-loss weights tuned per model and seed | planned | I1, D3 | — |
| E4 | Task-FC signature above rest: subject-level task − rest residual connectomes, train / val / test means (notebook) | planned | — | — |
| I1 | Composite-loss benchmark infrastructure: composite configs per model, reference scales, extra val metrics | planned | D3 | — |
| I2 | Config layout: family folders (done), one tune launcher, variants as roster overrides | done (layout, `5efe03a`); launcher dropped 2026-10-15 | — | — |
| I3 | Bidirectional graph / nodal models: FC → SC variants with a tuned FC edge threshold | superseded 2026-10-15 (→ v4 nodal-models item, not as a bidirectionality fix) | — | — |
| I4 | Environment alignment (Jupyter kernel image) and v2 C6 archive deletion | planned | — | user |
| I5 | Joint task-FC infrastructure: FC conditions as source modalities, CPU grid runner, predictive-residual PCA-PLS, E2 figures | done 2026-10-07 (`01adae6`) | — | agent:infra |
| C!1 | PCA-PLS + covariates FC → SC gain is anatomy (regional size), not FC; ~40% of it is the `invnodevol` normalisation | resolved 2026-10-14 (E1) | — | — |
| D1 | E0.2–E0.4 compute envelope and autonomous execution | decided | — | user |
| D2 | E1 compute envelope and autonomous execution | decided | — | user |
| D3 | E3 roster, loss-weight grid, batch size, selection metric(s) and compute envelope | pending | — | user |

Carried-over items C1–C8 are all closed or moved; see §4.

---

## 1. Conventions

Inherited from v2 §2 unchanged (experiment homes, reported metrics, seeds 0–4 for staged work, fixed reference scales,
PNG figures, `sbatch` compute, one campaign per benchmark row). Decisions **v2:D1–D6 stay in force** (one job
environment; tuning budget scales with evidence; E1 envelope; composite protocol at batch 64; Demeaned corr-eye in
mixtures; fixed reference scales).

### Decisions (`D`)

| ID | Decision | Rationale |
|---|---|---|
| D1 | **E0.2–E0.4 run autonomously (user 2026-10-06):** cap 55 GPU-h for the three steps, cut by 20% (to 44) if queue waits exceed 2 h. Pilots (rest + wm, seed 0) for E0.2 and E0.3; E0.4 starts without its own pilot once both pass. Gate per pilot task: 16 trials, finite metrics, the log's config / direction / parcellation match, rest val demeaned r > 0.05; E0.2 rest seed-0 test demeaned r within 0.02 of v2:E2.2 SC → FC PCA-PLS learnable seed 0 (0.071). Per-step time limit = pilot wall time × 1.5. Stop and report on a gate failure, > 20% task errors after one retry each, a log mismatch, a budget overrun, two GPU-underuse kills of one task, or a Ray hang `watch_jobs.py` cannot clear; relaxing a threshold needs the user. No split-half reliability ceiling (user). | Lets E0 finish without per-stage approval, with explicit exits (as v2:D3). |
| D2 | **E1 runs autonomously (user 2026-10-12)** through E1.0–E1.4 (see E1) with a 6 GPU-h cap. Gates: E1.1 cache gate, E1.2 pilot gate. Stop and report on a gate failure; on > 20% task errors after one retry each; on a budget overrun; when a log's config / direction / SC metric / covariate set does not match its condition; on a code change outside the files listed in E1 *Changes*; or when another agent's queued or running job imports `models/` or `data/` at the moment E1.0 needs to edit them (rule: jobs read live files). Per-task time limit = pilot wall time × 1.5. Relaxing a threshold needs the user. Commits stay local (no push). | Small, fully specified ablation with frozen hyperparameters; nothing is tuned, so the exits are mechanical. |
| D3 | **Pending (user):** E3's (a) roster: the learned edge-space models whose composite terms are meaningful at batch 64 (proposed: PCA-PLS learnable, linear backbone, PCA-PLS + covariates †, Sarwar, Nodal MLP; Chen GCN and Nodal GNN train at batch 4 / 8 and are at the null, so they keep their E2.2 MSE rows); (b) the weight grid (proposed `varmatch` {0, 0.1, 0.5}, `correye_dm` {0, 0.1, 0.5, 1}, `neidist` {0, 0.1, 0.5}; MSE-only inside the grid); (c) batch fixed at 64 for every composite run (v2:D4); (d) selection: val demeaned r (primary, as every benchmark) plus a secondary val-avg-rank best-trial report per seed from the same tune; (e) compute cap (estimate ≈ 60–80 GPU-h) and whether it runs autonomously. | Composite terms trade demeaned r for rank / top-1 (v2:E1); selecting on demeaned r alone is expected to land near MSE-only, so the secondary selection decides whether E3 can show the trade-off. |

**Parcellation:** experiments run on Glasser first; the 4S456Parcels replication uses the same configs with
`parcellation: 4S456Parcels` (caches exist).

## 2. Experiments

### E0 — Task FC ↔ SC by FC condition   (slug: `task_bidirectional_bench`, `task_fc2sc` until 2026-10-07) · status: done 2026-10-07 · owner: agent:infra

- **Question:** which FC condition (rest, `rest_S1`, or one of the seven HCP tasks) best predicts SC (FC → SC), and
  which is best predicted from SC (SC → FC)? Does the ordering hold in 4S456Parcels?
- **Fixed design (all sub-steps):** `CrossModal_PCA_PLS_learnable`, MSE-only, v2:E2.2 protocol for the direction;
  16-trial tunes; seeds 0–4; 9 conditions (`rest`, `rest_S1`, `emotion`, `gambling`, `language`, `motor`,
  `relational`, `social`, `wm`); matched cohort (intersection over all conditions, 917 Glasser / 916 4S456 subjects),
  so differences are paired by seed and subject. The condition is set by `HCP_Base(fc_source_condition=...)`, which
  rebinds the FC arrays before the train-split PCA; in SC → FC the rebound FC is the target. Outputs: per-condition ×
  seed tables with paired Δ vs rest, four metric bar charts, the scan-time figure, write-up section.
- **Caveat:** condition is confounded with scan length (rest ≈ 4 × 14.4 min vs tasks ≈ 2 × 2–5 min).

| Step | Direction | Parcellation | Status | Result |
|---|---|---|---|---|
| E0.1 | FC → SC | Glasser | done 2026-10-06 (merged, `4f00c4a`; 13.6 GPU-h) | Rest best (demeaned r 0.148); WM best task (0.108, Δ −0.041); other tasks 0.046–0.072; `rest_S1` −0.020. Scan time explains much of the order (tasks: Pearson 0.73 on log scan time). [`task_bidirectional_bench.md`](../../scripts/experiments/task_bidirectional_bench/task_bidirectional_bench.md) |
| E0.2 | SC → FC | Glasser | done 2026-10-07 (14.8 GPU-h) | Rest best (0.085); language best task (0.037), WM 0.030; tasks 0.013–0.037, near the SC → FC null |
| E0.3 | FC → SC | 4S456Parcels | done 2026-10-07 (20.8 GPU-h) | Rest best (0.138); WM best task (0.101); order as Glasser (ρ 0.88) |
| E0.4 | SC → FC | 4S456Parcels | done 2026-10-07 (18.7 GPU-h) | Rest best (0.055, noisy: one seed 0.002); WM best task (0.039); order identical to E0.3 |

- **Changes for E0.2–E0.4:** `steps:` (parcellation × direction) in `task_bidirectional_bench/config.yml`; configs
  `configs/<parcellation>/<condition>.yml` (direction set by the launcher); job names `e0_taskfc_<parc>_<dir>_<cond>`;
  results `results/<parcellation>/<direction>/{tables,figures}` + `results/compare/` (E0.1 moved to `Glasser/fc2sc`,
  re-render checked); `autopilot.py` (status / gate / launch / budget); per-task Ray cleanup in `launch.sh`. SC → FC
  uses the same E2.2 PCA-PLS learnable config (direction-free).
- **Accept:** each step has 9 × 5 results on its matched cohort; the write-up compares the condition order across
  directions (rank correlation of the condition means) and parcellations. **Met.**
- **Result (E0, all steps):**
  - Rest is best and rest S1 second in all four cells; WM is the best task in 3 / 4 cells.
  - The demeaned-r condition order replicates across direction and parcellation (Spearman 0.77–1.00).
  - Scan time tracks the order everywhere (Pearson on log scan time 0.93–0.97 over all conditions).
  - SC → FC is harder, and tasks there sit near the null. In that direction the condition is the target, so target
    reliability is confounded with predictability.
  - Write-up §7 and `results/compare/`. 54.3 GPU-h for E0.2–E0.4, no retries. Queue waits reached about 14 h,
    triggering the D1 cut to 44 GPU-h after every task had been submitted; total use is above the cut value and
    below the 55 cap.
- **Budget / autonomy:** D1 (cap 55 GPU-h; estimate 14 Glasser + 16–18 per 4S456 step).
- **Optional (not scheduled):** length-matched rest control (rest truncated to a task's volume count).

### E1 — Covariate family ablation for PCA-PLS + covariates   (slug: `covariate_ablation`) · status: done 2026-10-14 · owner: agent:infra

- **In plain terms:** in E2.2, adding subject covariates (demographics + FreeSurfer anatomy) to PCA-PLS made FC → SC
  predictions clearly more accurate and far more identifiable, while SC → FC barely changed. The SC target is divided
  by node volume, so a model told each subject's regional volumes / areas may just be undoing that division, a
  trivial gain rather than an FC–covariate link. E1 keeps the tuned model fixed, switches covariate families on and
  off to see which one carries the gain, and repeats `none` vs `full` on unnormalised SC: if the gain mostly vanishes
  there, it was the normalisation.
- **Question:**
  1. How much does each covariate family add to `CrossModal_PCA_PLS_CovProjector` (test demeaned r, top-1, avg rank)
     in FC → SC and SC → FC?
  2. Is the FC → SC lead (v2:E2.2 test demeaned r 0.221, top-1 0.45 vs 0.162 / 0.13 for PCA-PLS learnable) mostly
     anatomy undoing the SC target's `invnodevol` node-volume normalisation (C!1)?
- **Covariate families** (an exact split of the E2.2 inputs `fs_all` + age, sex, race/eth):

  | Key | Contents | Columns |
  |---|---|---|
  | `demo` | age, sex, race/eth (existing `age`, `sex`, `race_eth` sources as one group) | — |
  | `fs_vol` | every `FS_*` column not `*_Area` / `*_Thck`: global + subcortical volumes, eTIV ratios, defect holes | 63 |
  | `fs_area` | DK cortical surface areas (`*_Area`) | 68 |
  | `fs_thick` | DK cortical thicknesses (`*_Thck`) | 68 |

- **Design:**
  - **Model and hyperparameters:** CovProjector with each seed's E2.2 best-trial config frozen (from
    `model_benchmark/mse/<direction>/records.json`); no tuning, so paired differences come from the covariates only.
    The frozen `fs_all` projector `out_dim` is split over `fs_vol` / `fs_area` / `fs_thick` in proportion to column
    count (min 2); `demo` projectors keep their frozen sizes.
  - **Conditions (10):** `none` (`cov_sources: []`, z_cov = 0, same backbone), `full`, `drop_<family>` × 4,
    `only_<family>` × 4. Both directions, seeds 0–4, Glasser, normalised SC (`sift_invnodevol_radius2_count`): 100 runs.
  - **Unnormalised-SC control (FC → SC):** `none` and `full` with target `sift_radius2_count_connectivity` (log1p,
    same field in the raw qsirecon `.mat`, checked 2026-10-12), seeds 0–4: 10 runs. Same frozen hyperparameters
    (caveat: tuned on the normalised target; only the `full − none` difference is interpreted).
  - **Reading:** `only − none` = what a family gives on its own; `full − drop` = what only that family gives; a large
    gap between the two = shared signal. C!1 is supported if the `full − none` gain largely disappears on unnormalised
    SC and `fs_vol` (or `fs_area`) carries most of the normalised FC → SC gain.
- **Steps:**

  | Step | What | Compute |
  |---|---|---|
  | E1.0 | Infra: families, `cov_sources: []`, runner, checks | CPU |
  | E1.1 | Build the unnormalised Glasser SC cache | 1 CPU job |
  | E1.2 | Pilot: seed 0, `none` + `full`, both directions | 4 runs |
  | E1.3 | Remaining runs (normalised grid + unnormalised control) | 106 runs |
  | E1.4 | Tables, figures, write-up, spec / C!1 update | CPU |

- **Changes:**
  - `data/hcp_dataset.py`: `fs_vol`, `fs_area`, `fs_thick` cov sources (column selectors on `fs_all`, train-split
    z-score as now); `demo` expands to `age`, `sex`, `race_eth`.
  - `models/architectures/crossmodal_pca_pls.py`: CovProjector accepts `cov_sources: []` (z_cov = 0).
  - SC cache `Conn2Conn_data/sc/parc-Glasser_hemi-both_metric-sift_radius2_count_connectivity_log1p-1/` via the
    existing `load_sc(write_cache=True)` path in a CPU sbatch job; copy the `system.nfs4_acl` from the sibling
    invnodevol cache onto new files.
  - `scripts/experiments/covariate_ablation/`: `config.yml` (families, conditions, directions, seeds, SC metrics,
    frozen-config source), `launch.sh` (sbatch array over condition × direction × seed × SC metric, packed per GPU),
    `run.py` (tables + figures), `checks/`, write-up `covariate_ablation.md`.
- **Figures (3):**
  1. Per-family effect: `only − none` and `full − drop` per family, demeaned r and top-1, FC → SC and SC → FC side by
     side, seed dots.
  2. Normalisation control: `full − none` gain on normalised vs unnormalised SC (FC → SC), demeaned r and top-1.
  3. Context: `none`, `full`, PCA-PLS learnable (E2.2 rows, reused) per direction.
- **Accept:**
  - checks: the three FS families are disjoint and their union is the 199 `fs_all` columns; `full` with the new
    families' code path and `cov_sources: [fs_all, age, sex, race_eth]` reproduces the E2.2 seed-0 run's test metrics
    (same config) to numerical tolerance; `none` gives z_cov ≡ 0; all z-scoring uses the train split only;
  - 110 / 110 runs with finite metrics; tables with paired (by seed) deltas, mean ± std over seeds;
  - write-up answers both questions; C!1 set to `resolved` (or reworded) with the finding.
- **Gates:** E1.1: subject count ≥ the invnodevol cache's modeling cohort, finite and symmetric matrices, identical
  zero pattern, per-subject Spearman with the invnodevol SC > 0.95 but < 1 (user 2026-10-14: Pearson is ~0.68 only
  because log1p compresses the two count scales differently; Spearman 0.99). E1.2: `full` seed-0 test demeaned r within 0.01 of E2.2
  (FC → SC 0.224, SC → FC 0.095); `none` finite with val demeaned r > 0.05 (FC → SC).
- **Budget / autonomy:** D2. 110 single trainings with frozen configs; estimate 2–4 GPU-h, cap 6.
- **Progress:** E1.0 done (9e7e75e). E1.1 cache built (1063 subjects), gate passed on Spearman. E1.2 pilot passed
  (seed 0 test demeaned r, `none` / `full` / E2.2 inputs: FC → SC 0.144 / 0.223 / 0.227, top-1 0.14 / 0.37 / 0.43;
  SC → FC 0.090 / 0.094 / 0.096). E1.3 / E1.4 done 2026-10-14: 112 / 112 runs, 2.2 GPU-h.
- **Result:** [`covariate_ablation.md`](../../scripts/experiments/covariate_ablation/covariate_ablation.md).
  - FC → SC: covariates add +0.077 demeaned r and +0.31 top-1 (5/5 seeds). Cortical area (+0.052 alone) and FS
    volumes (+0.047) each carry about two thirds of it and overlap heavily (unique gains +0.018 / +0.013); thickness
    +0.018, demographics +0.011 alone.
  - SC → FC: no family helps (all deltas within ±0.015, mixed signs; full − none −0.003).
  - Unnormalised SC: the gain shrinks to +0.045 demeaned r and +0.11 top-1 (−40% / −60%) but stays positive in every
    seed. The E2.2 lead is partly the model undoing `invnodevol` and partly anatomy predicting streamline counts
    directly; neither is FC. CovProjector stays a † reference; its `none` condition (0.141) is the FC-only analogue.

### E2 — Joint task-FC → SC   (slug: `joint_taskfc2sc`) · status: done 2026-10-12 · owner: agent:infra

- **Question:** for FC → SC, do the seven task-FC blocks add predictive signal to rest FC, and does that hold for the
  task-specific, individualized part of task FC (the part of a subject's task FC that their own rest FC cannot
  predict)?
- **Model:** closed-form multi-block PCA-PLS. Each condition (rest + 7 tasks = 8 blocks) gets its own PCA fit on the
  train split; the per-block scores are concatenated (each PC column standardized, `PLSRegression(scale=True)` as in
  v2:E2.2) and PLS maps them to the SC PCs, decoded through the SC PCA. PLS never inverts XᵀX and `n_pls` is its
  regularizer, so D > N (up to 512 + 7 × 64 = 960 inputs vs ≈ 659 training subjects) needs no other model.
- **Steps:**

| Step | Task blocks | Status |
|---|---|---|
| E2.1 | **raw**: PCA of each task's FC (the naive concatenation; early task PCs largely repeat rest) | done 2026-10-08 |
| E2.2 | **predictive residual**: per task, a ridge fit (train only) predicts each subject's task FC from their own `k_rest` rest PC scores; the task block is the PCA of the prediction error, so it holds only what rest cannot predict | done 2026-10-08 |

  Simple subtraction (task FC − own rest FC) was dropped: per edge the task-on-rest slope is ≈ 0.25–0.3, not 1, so it
  over-subtracts and injects a sign-flipped copy of rest (rest explains 5–15 % of the difference's variance vs ≈ 2 % of
  raw task FC; measured 2026-10-07 on WM / language / emotion, Glasser).
- **Fixed design:** FC → SC, Glasser; E0's matched 917-subject cohort and per-seed splits (paired with E0); seeds 0–4;
  selection on val demeaned r, test reported; SC log1p `invnodevol` target as everywhere.
- **Grid (per seed):**

| Axis | Values | Why |
|---|---|---|
| `k_rest` | 64, 128, 256, 512, **640** (added by the edge rule) | v2:E2.2 rest-only val demeaned r: 32 → 0.128, 64 → 0.140, 128 → 0.150, 256 → 0.158, 512 → 0.159 (32 dropped; 512 = plateau edge) |
| `k_task` (shared by all 7 tasks) | **0**, 8, 16, 32, 64 | 0 = rest-only ablation (identical in E2.1 and E2.2) |
| `k_sc` | **8** (added by the edge rule), 16, 32, 64, 128, 256 | the SC test-reconstruction ceiling keeps rising to 256 (test R² 0.33 → 0.42 and demeaned-r ceiling 0.57 → 0.65 from 128 to 256); rest-only prefers 32 (v2:E2.2 val 0.160 at 32 → 0.138 at 256), so a shift of the optimum when tasks are added is itself a result |
| `n_pls` | 4, 8, 16, 32, 64, 128 | one `n_pls = 128` fit truncated to all six (NIPALS is sequential; exact to 1e-15, `checks/bench_pls_grid.py`) |

  E2.1: 4 × 5 × 5 = 100 PLS fits per seed (600 configurations); E2.2 adds 80. Each `k_task` slice has 120
  configurations: the comparison is between per-slice val-selected configs, never best-of-all vs best-of-rest-only.
  Edge rule: a winner at `k_rest = 512`, `k_task = 64`, `k_sc = 256` or `n_pls = 128` extends that axis by one step.
- **Outputs** (`scripts/experiments/joint_taskfc2sc/results/Glasser/{raw,predictive_resid,compare}/{tables,figures}`):
  1. **Ablation curve (headline):** test demeaned r, avg rank, top-1 (2 × 2) vs `k_task`; each point = the slice's
     val-selected config, mean ± SE over seeds; raw and residual lines; paired Δ vs `k_task = 0` below.
  2. `k_rest` × `k_task` heatmaps (val and test demeaned r, maximized over `k_sc`, `n_pls`), one per version.
  3. **Weave (parallel coordinates):** one line per configuration (seed mean) through `k_rest → k_task → total D →
     k_sc → n_pls → val r → test r`, coloured by val r; slice winners highlighted; D > n_train marked. PNG (optional
     interactive HTML).
  4. Dimensionality curve: test r vs total input dimension (log), coloured by `k_task`, line at n_train.
  5. Per-seed paired Δ (best with tasks − best rest-only) per metric and version.
  6. Block contribution: block VIP of each winner's PLS (share of explained SC variance per condition block; seeds as
     dots).
  7. Selected `k_sc` by `k_task` (does the target-dimension optimum move up when tasks are added?).
- **Accept:** both versions complete on 5 seeds with every grid cell scored; figures 1–7 and the write-up
  `joint_taskfc2sc.md`; `k_task = 0` slice reproduces a plain `main.py` `CrossModal_PCA_PLS --source FC` run on the
  same cohort and config to numerical precision. **Met** (I5 checks `19358187`; convergence `19457342`).
- **Result:** [`joint_taskfc2sc.md`](../../scripts/experiments/joint_taskfc2sc/joint_taskfc2sc.md).
  - **No gain from task FC:** rest only is best (test demeaned r 0.149). Every with-task slice is lower, raw −0.005 to
    −0.015 and predictive residual −0.007 to −0.018, negative in all 5 seeds × 4 slices of both versions.
  - **Not a selection artifact:** choosing configs on test (diagnostic bound) still leaves every with-task slice
    (≤ 0.146) below val-selected rest only. The val → test gap grows only slightly with tasks (0.016–0.018 vs 0.013).
  - `k_task = 0` is the best column at every `k_rest`; adding tasks doesn't move the SC-PC optimum (16).
  - PLS gives task blocks ≈ 2× their column share of explained variance, and that weight doesn't generalize.
  - Edge rule applied: `k_rest` 640, `k_sc` 8 added (winners at 640 ≈ the PC maximum).
  - ≈ 6 CPU-node-hours (grid `19365975`, extension `19454464`).
- **Budget:** CPU only (no GPU). Measured ≈ 5–12 s per PLS fit incl. the six truncations and edge metrics
  (`bench_pls_grid.sh`, job `19348816`) → ≈ 20 min per seed per version, ≈ 3–4 CPU-node-hours for E2. One CPU array
  (seed × version). Launch after the I5 checks pass and the user's go.
- **Caveat:** task blocks differ in scan length, so a task's added value is partly data quantity.
- **Follow-ups (not scheduled), in priority order:**
  1. Block-weighted regression (group ridge per block, or PLS block scaling) so the fit can switch noisy task blocks
     off. Per-column standardization gives every task PC rest-level weight; this is the remaining linear route.
  2. Regional Δ maps (per-region demeaned r of predicted SC rows, (rest + task) − rest, seed-paired, FDR) for the
     WM / MTL hypothesis.
  3. Add-one / drop-one per task.
  4. More stable selection (k-fold CV on train + val, consensus): low value, since the test-oracle bound rules out a
     selection fix.
  5. Learnable multi-block model and composite losses; 4S456Parcels replication: deprioritized given no linear
     gain.

### E3 — Composite-loss cross-model benchmark   (slug: `model_benchmark/composite`) · status: planned · owner: —

- **Question:** with composite-loss weights tuned per model and seed alongside the other hyperparameters (on the
  E2.2 splits and protocol), how much does each model gain over its MSE-only E2.2 result, in each direction? Which
  metrics move (demeaned r vs average rank / top-1)?
- **Design:** v2:E2.2 unchanged except the loss: seeds 0–4, the default 957-subject cohort and per-seed splits (paired
  with E2.2), per-seed Optuna tune, MSE + `varmatch` / `correye_dm` / `neidist` at fixed reference scales with
  weights searched (I1). MSE-only is inside the grid, so a model can only lose to E2.2 through search noise. Batch 64
  for every composite run (D3 c). Roster per D3 (a). Excluded models keep their E2.2 MSE rows, marked:
  - closed-form models (no training loss: PCA null, PLS-SVD, PCA-PLS, Conditional Gaussian);
  - Masked MLP pretrainer (native latent objective);
  - Krakencoder (its own composite grid, v2:E2.1; reused as reference rows);
  - Chen GCN and Nodal GNN if D3 (a) keeps them out.
- **Selection:** val demeaned r (primary, every benchmark's rule). Secondary (D3 d): a second best-trial report per
  seed selected on val average rank from the same tune, reported as a separate row set. Neither selects on test.
- **Outputs** (`scripts/experiments/model_benchmark/composite/{sc2fc,fc2sc}/`, tracked like `mse/`):
  - records, tables and figures in the E2.2 layout (bars per metric, grouped / performance panels, scatter);
  - per model × seed paired Δ vs E2.2 MSE (same splits);
  - the selected weights per model × seed (how often each term is switched on);
  - for the secondary selection, the demeaned r ↔ average rank trade-off per model.
- **Accept:** every roster model × direction × seed has a verified best-trial report (both selections); tables and
  figures; write-up section in `model_benchmark.md` (or `composite.md`); E2.2 rows unchanged.
- **Budget:** D3 (e). Estimate ≈ 60–80 GPU-h. E2.2's learned models took most of its 111.5 GPU-h; the composite
  terms add little per step; the trial counts rise to the 64 cap for most models.
- **Then:** the final reoptimized benchmark (the composite winners as the model set) is a v4 item.

### E4 — Task-FC signature above rest   (slug: notebook `scripts/notebooks/EDA/FC_matrix_analysis.ipynb`) · status: planned · owner: —

- **Question:** does each task have a consistent FC signature above rest? Is it reproducible across independent
  groups of subjects, and how consistently do individual subjects express it? This bears on task-FC data quality
  and on why E2's task blocks added nothing.
- **Design:**
  - Matched 917-subject Glasser cohort, seed-0 train / val / test labels (659 / 74 / 184).
  - Per subject and task, the residual connectome `D_i^(c) = F_i^(c) − F_i^(rest)`: their own task FC minus their own
    rest FC.
  - The *mean* signature is the same under plain or fitted (predictive) subtraction, since it equals μ_c − μ_rest. The
    E2 over-subtraction issue only affects individual-level analyses.
- **Outputs (appended cells; the notebook's existing state and the user's uncommitted edits stay untouched):**
  1. Per task, the train, val and test mean residual connectomes side by side, on a shared symmetric colour scale.
  2. **Reproducibility:** correlation of the mean signature between splits (train↔val, train↔test, val↔test) per task.
     The null is the same statistic for `rest_S1 − rest_S2` (no task), which shows the reproducibility of a pure
     session difference.
  3. **Strength:** per-edge effect size (mean / SD of `D` across train subjects), with the share of edges with |d| > 0.5.
  4. **Individual consistency:** per subject, correlation of their `D` with the train-mean signature (val / test
     subjects scored against the train mean), as distributions per task.
- **Code:** reusable functions in `data/data_viz.py` next to the existing condition views (compute + plot); the
  notebook cells call them. Headless execution through `scripts/sbatch/checks/run_notebook_cells.sh` (CPU).
- **Accept:** the four outputs for all 7 tasks plus the session null; a short summary cell; figures saved under
  `results/figures/fc_conditions/`.
- **Budget:** CPU, under 1 node-hour.

## 3. Infrastructure

### I1 — Composite-loss benchmark infrastructure (for E3) · status: planned · owner: —

- **Goal:** everything E3 needs to rerun the v2:E2.2 benchmark with composite losses through the existing campaign
  machinery (`model_benchmark/{build_configs,submit,autopilot,run}.py`, `launch_model.sh`), no new training code.
  Replaces the earlier 4-weightings-as-one-categorical design: the weights are searched as the existing flat
  `loss_weight_<term>` keys.
- **Changes:**
  1. **Composite configs:** `build_configs.py --campaign composite` writes `models/configs/benchmark/composite/<Model>[_fc2sc].yml`
     from each roster model's E2.2 spec: same search space and trial rule (`clamp(8 × free keys, 16, 64)` with the 3
     weight keys added), plus `loss_terms` = `mse` (weight 1, raw) + `varmatch`, `correye_dm`, `neidist` with fixed
     `scale` kwargs (v2:D6, `loss_normalize: auto` → `none`), `batch_size: 64` (v2:D4), and the D3 weight grid as
     `loss_weight_<term>` choices. Raw `correye` excluded (v2:D5).
  2. **Reference scales per model × direction:** reuse `composite_loss/<model>/<direction>/state.yml` for the linear
     backbone, PCA-PLS learnable and PCA-PLS + covariates. For the other roster models (Sarwar, Nodal MLP), one MSE-only
     run per seed at their E2.2 seed-best config with `loss_monitor_terms: [varmatch, correye_dm, neidist]`, then
     `loss_grid.reference_scales` (c_t = s_t / s_mse), written to `model_benchmark/composite/reference_scales.yml`.
  3. **Selection extras (`main.py`, additive):** each Tune trial also reports `val_avg_rank` and `val_top1_acc`; a
     `--select_metric` option for the best-trial report, so `launch_model.sh` can emit a second report selected on
     val avg rank from the same tune (D3 d).
  4. **Campaign config:** a `composite` section in `model_benchmark/config.yml` (or `config_composite.yml`): campaign
     `e3_comp`, `results_dir: composite`, roster per D3, packing / times from E2.2, budget cap per D3, MSE rows of
     excluded models reused from `mse/`.
- **Accept:**
  - MSE configs unchanged (`build_configs.py --check`; loss regression passes);
  - a composite config with all weights 0 reproduces the E2.2 seed-0 trial of the same config;
  - one smoke tune per direction samples non-zero weights, logs `loss_signature`, `val_avg_rank`, `val_top1_acc`,
    and writes both best-trial reports;
  - the reference scales of the two new models are within the per-seed spread of the existing instances' values
    for the same terms (sanity, not a gate).
- **Budget:** reference-scale runs ≈ 1–2 GPU-h; smoke tunes ≈ 1 GPU-h.

### I2 — Config layout and one tune launcher · status: done (layout); launcher dropped 2026-10-15 · owner: —

- **Done:** v2:I2.3 merged 2026-10-05 (`5efe03a`): family folders `models/configs/{null_ceiling,linear,latent,graph_nodal,deep}/`,
  `variants/<family>/`, name-based lookup.
- **Dropped (user 2026-10-15):** the generic `launch_tune.sh` + roster `submit.py`. Every campaign since (v2:E2.2,
  E0, E1, E2) runs from its experiment folder (`config.yml` + `submit.py` / `autopilot.py` + `launch.sh`), which has
  replaced it; the per-model `scripts/sbatch/<Model>/` arrays stay as legacy.

### I3 — Bidirectional graph / nodal models · status: superseded 2026-10-15 (→ v4) · owner: —

- Moved to v4 (user 2026-10-15) as part of a nodal-models item, not as a bidirectionality fix: the FC → SC variants
  (FC graph at a tuned threshold, no SC-derived node inputs, `target == "SC"` guard) join the backlog's "graph / nodal
  models on top of the mean". Chen GCN, Nodal GNN and Nodal MLP stay SC → FC only in v3.

### I4 — Environment alignment · status: planned · owner: user

- Jobs have run clean on `/ext3/env.sh` + the cuda12.8.1 (Ubuntu 24.04) image since 2026-09-30, E0–E2 included.
- **What is left, and why it matters:**
  1. The Jupyter kernel wrapper (`~/.local/share/jupyter/kernels/kraken_env/python`) still starts the cuda12.3.2
     (Ubuntu 22.04) image. Notebooks therefore run the same overlay packages on a different base image (CUDA runtime,
     glibc) than the jobs. No failure seen so far, but a notebook result is not strictly the job environment.
     `/scratch/asr655/envs/README.md` still lists 12.3.2 for `kraken_env`.
  2. `/scratch/asr655/envs/archive/2026-09-30_c6/` (14 GB: the pre-C6 overlay backup, installer, one log). Its
     MANIFEST's delete condition (post-C6 jobs run clean) has been met for two weeks. Deleting it is irreversible, so
     it is the user's call.
  3. `envs/activate_env.sh` is a third stack (`pylibs`, torch 2.11), used only interactively.
- **Necessity:** low; hygiene plus 14 GB. Nothing in v3 is blocked by it.
- **Proposed:** point the kernel wrapper and the envs README at 12.8.1, delete the archive, and document
  `activate_env.sh` as interactive-only (or retire it). This takes about 10 minutes once approved.

### I5 — Joint task-FC infrastructure (for E2) · status: done 2026-10-07 (`01adae6`) · owner: agent:infra

- **Goal:** run E2 from library code (no forked model math), fast enough for exhaustive grids on CPU.
- **Changes:**
  1. **FC conditions as source modalities** (`data/hcp_dataset.py`, `models/architectures/utils.py`): `FC_<task>` for
     each `FC_TASK_CONDITIONS` entry is a valid modality (source only; needs the task in `fc_conditions`).
     `HCP_Base` fits a train-split PCA for a condition only when it is a source; `HCP_Partition` serves it;
     `get_modality_data` maps it. Raw E2.1 is then the existing multi-source `CrossModal_PCA_PLS` with
     `source=FC+FC_emotion+…` and a per-block `n_components_pca` dict.
  2. **Grid runner** (`scripts/experiments/joint_taskfc2sc/`): `config.yml` (grid, versions, seeds), `grid.py` (one
     `HCP_Base` per seed; builds each block's PCA once at its max k and slices; fits one `n_pls = 128` PLS per
     (`k_rest`, `k_task`, `k_sc`) cell and scores all six truncations on val and test; writes every cell to
     `runs/<version>/seed<S>.csv`), `launch_grid.sh` (CPU array), `run.py` (tables + figures 1–7), `checks/`,
     write-up `joint_taskfc2sc.md`.
  3. **Predictive-residual model** `CrossModal_PCA_PLS_TaskResid` (subclass in `models/architectures/crossmodal_pca_pls.py`;
     config `models/configs/linear/CrossModal_PCA_PLS_TaskResid.yml`): per task, ridge of task edges on the train
     `k_rest` rest scores (penalty by generalized cross-validation on train), residual PCA, same PLS; val / test are
     residualized with the train fit.
- **Accept:**
  - a single `FC` source gives bit-identical outputs to `main` before I5;
  - the grid runner's numbers for one cell equal `main.py` (`CrossModal_PCA_PLS`, same cohort and config) to
    numerical precision;
  - TaskResid with an infinite ridge penalty reproduces raw; its train residual blocks have ≈ 0 correlation with
    the rest scores; nothing is fit on val / test;
  - truncated PLS equals separate fits (done: `checks/bench_pls_grid.py`, max rel. diff 7e-16).

## 4. Carried over from v2

| ID | Item | Status |
|---|---|---|
| C1 | Composite-loss benchmark with tuned weights (v2:E2.3) | superseded (by I1; the benchmark run itself is a future E) |
| C2 | Covariate ablation | superseded (by E1) |
| C3 | Config layout follow-ups (v2:I2.3–I2.5) | superseded (by I2) |
| C4 | Subject × subject connectome-similarity views | done (covered by the FC matrix EDA, v2:I1.5) |
| C5 | Cross-condition fingerprinting, network-ordered view | dropped from v3 (follow-up stays with v2:I1.5) |
| C6 | FC → SC variants for Chen GCN, Nodal GNN, Nodal MLP | superseded (by I3) |
| C7 | Environment alignment, archive deletion | superseded (by I4) |
| C8 | Masked MLP FC → SC at full budget despite failed gate | done 2026-10-05 (`8f97342`): demeaned r 0.122, −0.040 vs PCA-PLS learnable; [`model_benchmark.md`](../../scripts/experiments/model_benchmark/model_benchmark.md) |

### Caveats (`C!`)

| ID | Caveat | Affected artifacts | Resolved by |
|---|---|---|---|
| C!1 | `CrossModal_PCA_PLS_CovProjector` sees FreeSurfer `fs_all` + demographics; its FC → SC lead comes from regional size (cortical area, FS volumes), not FC; E1: ~40% of the demeaned-r gain (~60% of top-1) is undoing the SC target's `invnodevol` normalisation, the rest is anatomy predicting streamline counts directly. Reported as not input-matched (†), never a paired reference. | `model_benchmark/mse/fc2sc`, `composite_loss/pca_pls_covprojector/fc2sc`; to a lesser degree its SC → FC rows | E1 |

## 5. Backlog (not scheduled)

- **Graph / nodal models on top of the mean** (v2:E0): GNNs on the deviation from the train-split mean FC, or on PCA
  scores / low-rank factors, including an SC-only `NodalGNN`. Pilot per v2:D2 against PCA-PLS learnable. Absorbs I3
  (FC → SC variants: FC graph at a tuned threshold, no SC-derived node inputs); a v4 nodal-models item.
- **Editable install** (`pyproject.toml` + `pip install -e .`) and `conn2conn/` namespacing.
- **Artifact cleanup / storage** (audit 2026-10-06; repo ≈ 2.3 TB, user wants to revisit it):
  - `results/ray_tmp` 768 GB and `results/ray_checkpoints` 699 GB, mostly sessions and trial storage from Feb–Apr and
    Sep–Oct; `results/krakencoder` 286 GB (≈ 13 GB per grid cell, mostly two 1.1 GB prediction `.mat` per seed);
    `results/conn2conn` 272 GB (Lightning checkpoints per W&B run id, 1,623 dirs); `results/ray_results` 142 GB (Sarwar
    100 GB); `krakencoder_experimental/` 87 GB; legacy cwd-relative `checkpoints/` 69 GB (669 `.ckpt`, Mar–Sep).
  - `.git` 4.2 GB, of which ≈ 3.7 GB is loose objects / garbage (packed history 286 MB): a `git gc` would reclaim it.
  - One stray tracked file: `results/local_results/Krakencoder_precomputed/krakencoder_seed0/.ipynb_checkpoints/eval_test-checkpoint.md`.
  - Prevention already in place: `task_bidirectional_bench/launch.sh` deletes each finished task's Ray temp + Tune
    storage (≈ 7–8 GB per task). Still open: the same for the other launchers, `keep_checkpoints_num`, no per-trial
    checkpoints when only the best trial is re-evaluated, a `results/logs/` retention policy.
- **Shared constants** between `main.py` and `scripts/results_utils/records.py`.
- **Untracked reference code** in `context_packages/modeling/*_context/`.
- **Latent-space terms inside composite** (`latent_mse` / `latent_weighted_mse` as mixable terms).
- **EMA diagnostics:** `*_loss_ref_*` after warmup; batch-size sensitivity of `correye` / `neidist` (64 vs 128).

## 6. Change log

| Date | Change |
|---|---|
| 2026-10-04 | v3 created at v2 closure: carried items, caveats and backlog moved; v2:D1–D6 in force. |
| 2026-10-05 | E0 specified (task FC → SC, PCA-PLS learnable, matched 917-subject cohort); 16-trial tunes, `rest_S1` bar, no null bar. Carry-over pruned and renumbered C1–C8, C!1. |
| 2026-10-05 | C3: I2.3 merged (`5efe03a`). C8 done: Masked MLP FC → SC 0.122 demeaned r; E2.2 figures restyled. |
| 2026-10-06 | E0 FC → SC Glasser done and merged (`4f00c4a`): rest best, WM best task. |
| 2026-10-06 | Restructured (user): E0 extended to both directions × both parcellations (E0.1–E0.4); C1 → I1 (composite-loss search infra, 4 weightings per direction); C3 → I2; C6 → I3 (tuned FC edge threshold); C7 → I4; C2 → E1 (late stage, resolves C!1); C4 done; C5 dropped (v2:I1.5 follow-up); C8 closed. |
| 2026-10-06 | D1: E0.2–E0.4 approved for autonomous execution; E0 layout and plan set; owner agent:infra. |
| 2026-10-07 | E0 done: E0.2–E0.4 (54.3 GPU-h, no retries); rest best everywhere, condition order replicates across direction and parcellation; SC → FC tasks near the null. |
| 2026-10-07 | E2 (joint task-FC → SC, raw vs predictive residual) and I5 specified; grid sized from v2:E2.2 trials and the SC reconstruction ceiling; PLS truncation benchmarked (CPU, ≈ 5–12 s per fit). |
| 2026-10-12 | E2 done: task-FC blocks add no FC → SC signal over rest (raw or predictive residual; −0.005 to −0.018 demeaned r, all seeds); test-oracle bound rules out a selection artifact; edge extension `k_rest` 640 / `k_sc` 8; convergence check OK. I5 done. |
| 2026-10-14 | E1 done: E1.1 cache gate moved to Spearman + zero pattern (user). FC → SC covariate gain (+0.077 demeaned r, +0.31 top-1) is regional size (area ≈ volumes, overlapping); none in SC → FC; ~40% of it disappears on unnormalised SC, the rest stays. C!1 resolved (reworded). |
| 2026-10-15 | Stock-take (user): I2 launcher dropped (layout done); I3 → v4 as a nodal-models item; I4 necessity written out; I1 re-scoped to composite-benchmark infrastructure; E3 (composite-loss cross-model benchmark) and E4 (task-FC signature above rest, notebook) added; D3 pending. |
| 2026-10-12 | E1 specified (user): 4 covariate families (`demo`, `fs_vol`, `fs_area`, `fs_thick`), 10 conditions × 2 directions × 5 seeds with frozen E2.2 configs, plus an unnormalised-SC (`sift_radius2_count`) FC → SC control; 3 figures. D2 (autonomous execution, 6 GPU-h cap) proposed. |

Last updated at: 2026-10-15 EDT
