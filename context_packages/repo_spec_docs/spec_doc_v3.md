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
| E1 | Covariate ablation for PCA-PLS + covariates (late stage) | planned | — | — |
| E2 | Joint task-FC → SC: do the 7 task-FC blocks add to rest FC? (E2.1 raw, E2.2 predictive residual) | planned | I5 | agent:infra |
| I1 | Composite-loss search infrastructure: 4 candidate weightings per direction as a tuned hyperparameter | planned | — | — |
| I2 | Config layout: family folders (done), one tune launcher, variants as roster overrides | in progress | — | agent:modeling |
| I3 | Bidirectional graph / nodal models: FC → SC variants with a tuned FC edge threshold | planned | — | — |
| I4 | Environment alignment (Jupyter kernel image) and v2 C6 archive deletion | planned | — | user |
| I5 | Joint task-FC infrastructure: FC conditions as source modalities, CPU grid runner, predictive-residual PCA-PLS, E2 figures | in progress | — | agent:infra |
| C!1 | PCA-PLS + covariates FC → SC gain is likely anatomy, not FC | open | resolved by E1 | — |
| D1 | E0.2–E0.4 compute envelope and autonomous execution | decided | — | user |

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

### E1 — Covariate ablation for PCA-PLS + covariates · status: planned (late stage, not pressing) · owner: —

- **Question:** is `CrossModal_PCA_PLS_CovProjector`'s FC → SC lead (v2:E2.2 test demeaned r 0.221, top-1 0.45 vs
  0.162 / 0.13 for PCA-PLS learnable) due to FreeSurfer regional volumes predicting the SC target's `invnodevol`
  normalisation (C!1)?
- **Design:** three covariate sets — (a) full (`fs_all` + age, sex, race/eth; existing E2.2 rows), (b) `fs_all`
  without regional volumes + demographics, (c) demographics only. Both directions, E2.2 protocol, seeds 0–4, Glasser.
- **Changes:** two variants under `models/configs/variants/linear/` (or I2 roster overrides once available).
- **Accept:** paired Δ vs PCA-PLS learnable per covariate set and direction; C!1 set to `resolved` with the finding.
- **Budget:** ≈ 1–2 GPU-h.

### E2 — Joint task-FC → SC   (slug: `joint_taskfc2sc`) · status: planned · owner: agent:infra

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
| E2.1 | **raw**: PCA of each task's FC (the naive concatenation; early task PCs largely repeat rest) | planned |
| E2.2 | **predictive residual**: per task, a ridge fit (train only) predicts each subject's task FC from their own `k_rest` rest PC scores; the task block is the PCA of the prediction error, so it holds only what rest cannot predict | planned |

  Simple subtraction (task FC − own rest FC) was dropped: per edge the task-on-rest slope is ≈ 0.25–0.3, not 1, so it
  over-subtracts and injects a sign-flipped copy of rest (rest explains 5–15 % of the difference's variance vs ≈ 2 % of
  raw task FC; measured 2026-10-07 on WM / language / emotion, Glasser).
- **Fixed design:** FC → SC, Glasser; E0's matched 917-subject cohort and per-seed splits (paired with E0); seeds 0–4;
  selection on val demeaned r, test reported; SC log1p `invnodevol` target as everywhere.
- **Grid (per seed):**

| Axis | Values | Why |
|---|---|---|
| `k_rest` | 64, 128, 256, 512 | v2:E2.2 rest-only val demeaned r: 32 → 0.128, 64 → 0.140, 128 → 0.150, 256 → 0.158, 512 → 0.159 (32 dropped; 512 = plateau edge) |
| `k_task` (shared by all 7 tasks) | **0**, 8, 16, 32, 64 | 0 = rest-only ablation (identical in E2.1 and E2.2) |
| `k_sc` | 16, 32, 64, 128, 256 | the SC test-reconstruction ceiling keeps rising to 256 (test R² 0.33 → 0.42 and demeaned-r ceiling 0.57 → 0.65 from 128 to 256); rest-only prefers 32 (v2:E2.2 val 0.160 at 32 → 0.138 at 256), so a shift of the optimum when tasks are added is itself a result |
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
  same cohort and config to numerical precision.
- **Budget:** CPU only (no GPU). Measured ≈ 5–12 s per PLS fit incl. the six truncations and edge metrics
  (`bench_pls_grid.sh`, job `19348816`) → ≈ 20 min per seed per version, ≈ 3–4 CPU-node-hours for E2. One CPU array
  (seed × version). Launch after the I5 checks pass and the user's go.
- **Caveat:** task blocks differ in scan length, so a task's added value is partly data quantity.
- **Follow-ups (not scheduled):** add-one / drop-one per task at the winner; regional Δ maps (per-region demeaned r of
  predicted SC rows, (rest + task) − rest, seed-paired, FDR across regions) for the MTL / WM hypothesis; block scaling
  instead of per-column standardization at each winner; learnable multi-block `CrossModal_PCA_PLS_learnable`
  initialized from the winner, then composite losses (I1); 4S456Parcels replication.

## 3. Infrastructure

### I1 — Composite-loss search infrastructure · status: planned · owner: —

- **Goal:** the next cross-model benchmark (a future E) can tune composite-loss weightings alongside the other
  hyperparameters, without new code. Running that benchmark is not part of I1.
- **Changes:**
  1. **Evaluate** the existing composite-loss results (`composite_loss/<model>/{sc2fc,fc2sc}/tables/combo_summary.csv`
     for linear backbone, PCA-PLS learnable, PCA-PLS + covariates, and Krakencoder) and choose **4 weightings per
     direction** spanning the demeaned r ↔ average rank / top-1 trade-off (`mse_only` always included; terms from
     Demeaned corr-eye, Neighbor dist, Var-match per v2:D5). Record the choice and reasoning in `composite_loss.md`.
  2. **Search support:** the 4 weightings become one categorical hyperparameter (e.g. `loss_combo`) in the Tune search
     space, resolved to `loss_terms` before training, so they are tuned with `lr` / `l2_reg` / epochs
     on val demeaned r, as in every benchmark (`model_benchmark/config.yml` `selection`).
  3. **Configs:** `models/configs/benchmark/composite/` per model and direction (generated by `build_configs.py`), first
     for the linear backbone, PCA-PLS learnable and Conditional Gaussian.
- **Accept:** a one-seed smoke tune per direction samples all 4 combos and logs the chosen combo in W&B and the
  best-trial report; MSE-only configs are unchanged (loss regression passes).
- **Notes for I1:** v2 E3 Phase D flags (widen linear backbone PCs past 256, PCA-PLS learnable `lr` floor near
  3e-5); v2 Phase D found no composite term helps FC → SC, so FC → SC may keep fewer than 4.

### I2 — Config layout and one tune launcher · status: in progress · owner: agent:modeling

- **Goal:** one generic launcher and roster-driven variants, so experiments stop adding per-model sbatch scripts.
- **Done:** v2:I2.3 merged 2026-10-05 (`5efe03a`): family folders `models/configs/{null_ceiling,linear,latent,graph_nodal,deep}/`,
  `variants/<family>/`, name-based lookup; checks pass on `main`.
- **Changes (remaining):** `scripts/sbatch/launch_tune.sh` + roster `submit.py`; per-model sbatch folders move to
  `scripts/sbatch/legacy/`; source / covariate variants as `data:` / `model:` roster overrides.
- **Accept:** one E2.2 model reproduces its seed-0 tune via the new launcher; `build_configs.py --check` passes.

### I3 — Bidirectional graph / nodal models · status: planned · owner: —

- **Goal:** every benchmark model runs in both directions. Chen GCN, Nodal GNN and Nodal MLP currently have no
  FC → SC variant (v2:E2.0) and are excluded from the FC → SC benchmark.
- **Changes:** for FC → SC, message passing on FC thresholded at `fc_graph_threshold`, a tuned hyperparameter
  (tuned on val demeaned r like the other hyperparameters); no SC-derived node inputs (`use_r2t: false`); Nodal MLP reads FC rows; a
  guard refusing `target == "SC"` with SC-derived inputs.
- **Accept:** small contained check: FC → SC seed-0 tune per model on the E2.2 protocol, results in the E2.2 FC → SC
  tables (above or at the null is fine; SC → FC they are at the null). Guard covered by a unit check.
- **Budget:** ≈ 3 GPU-h (3 models × one seed).

### I4 — Environment alignment · status: planned · owner: user

- Jobs have run clean on `/ext3/env.sh` + the cuda12.8.1 image since 2026-09-30. Remaining: point the Jupyter kernel
  wrapper (`~/.local/share/jupyter/kernels/kraken_env/python`, still cuda12.3.2) and `activate_env.sh` at the same
  image, then delete `/scratch/asr655/envs/archive/2026-09-30_c6/`.

### I5 — Joint task-FC infrastructure (for E2) · status: in progress · owner: agent:infra

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
| C!1 | `CrossModal_PCA_PLS_CovProjector` sees FreeSurfer `fs_all` + demographics; its FC → SC lead most likely comes from regional volumes predicting the SC target's `invnodevol` normalisation. Reported as not input-matched (†), never a paired reference. | `model_benchmark/mse/fc2sc`, `composite_loss/pca_pls_covprojector/fc2sc`; to a lesser degree its SC → FC rows | E1 |

## 5. Backlog (not scheduled)

- **Graph / nodal models on top of the mean** (v2:E0): GNNs on the deviation from the train-split mean FC, or on PCA
  scores / low-rank factors, including an SC-only `NodalGNN`. Pilot per v2:D2 against PCA-PLS learnable. After I3.
- **Editable install** (`pyproject.toml` + `pip install -e .`) and `conn2conn/` namespacing.
- **Artifact cleanup:** `results/ray_results/` (118 GB), a `results/logs/` retention policy, and Tune checkpoint /
  Ray temp retention (E0.1 left 246 GB `ray_tmp` + 96 GB `ray_checkpoints`, deleted 2026-10-06): `keep_checkpoints_num`,
  no per-trial checkpoints when only the best trial is re-evaluated, Ray temp cleanup per job.
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

Last updated at: 2026-10-07 EDT
