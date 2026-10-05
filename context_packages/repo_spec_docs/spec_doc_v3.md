# Repo Spec v3

**Purpose:** carry v2's open work forward. New experiments are added here once their scope is set by the user.
**Status:** active · **Started:** 2026-10-04 · **Predecessor:** [`spec_doc_v2.md`](spec_doc_v2.md) (closed 2026-10-04) ·
**Format:** [`spec_conventions.md`](spec_conventions.md)

**Contents:** [Status](#status) · [1. Purpose](#1-purpose) · [2. Conventions](#2-conventions) ·
[3. Carried over from v2](#3-carried-over-from-v2) · [4. Experiments](#4-experiments) · [5. Infrastructure](#5-infrastructure) ·
[6. Backlog](#6-backlog-not-scheduled) · [7. Change log](#7-change-log)

To add an experiment, append a section under §4 (template: v2 §4.0) and add a row to the status table.

## Status

| ID | Title | Status | Depends on | Owner |
|---|---|---|---|---|
| E0 | Which task FC best predicts SC: PCA-PLS learnable, MSE-only, FC → SC per condition (`task_fc_to_sc`) | planned (spec 2026-10-05; decisions pending) | — | agent:infra |
| C1 | Composite-loss benchmark with tuned weights, both directions; then the final reoptimized benchmark (v2:E2.3) | planned (design pending) | v2:E1, v2:E2.2 | — |
| C2 | Covariate ablation for PCA-PLS + covariates (v2:E2.2 / E3 Phase D) | planned (proposed 2026-10-04, not approved) | — | — |
| C3 | Merge I2.3 config family folders (branch `i2-config-layout`); then one tune launcher and variants as overrides (v2:I2.4–I2.5) | in progress (merge done 2026-10-05 `5efe03a`; launcher + overrides planned) | — | agent:modeling |
| C4 | Subject × subject connectome-similarity views (v2:I1.2) | planned | — | — |
| C5 | Task-FC follow-ups: cross-condition fingerprinting, network-ordered view (v2:I1.5) | planned | — | — |
| C6 | FC → SC reverse variants for Chen GCN, Nodal GNN, Nodal MLP (v2:E2.0) | planned | — | — |
| C7 | Environment alignment and C6 archive deletion (v2:C6.5) | planned | user | user |
| C8 | E2.2 addendum: Masked MLP pretrainer FC → SC at full budget despite its failed gate (user 2026-10-05) | in progress (job `19248105`) | — | agent:modeling |
| C!1 | PCA-PLS + covariates FC → SC gain is likely anatomy, not FC | open | resolved by C2 | — |

---

## 1. Purpose

v2 closed with E1 (composite-loss dynamics), E2.2 (MSE-only benchmark, both directions) and E3 (FC → SC replication of
E1 and E2.2) done. v3 holds what remains: the composite-loss benchmark with tuned weights, the covariate ablation,
the repo-organisation follow-ups, and the timeseries / task-FC work. New scope is added by the user.

## 2. Conventions

Inherited from v2 §2 unchanged (experiment homes, reported metrics, seeds 0–4 for staged work, fixed reference scales,
PNG figures, `sbatch` compute, one campaign per benchmark row). Decisions **v2:D1–D6 stay in force** (one job
environment; tuning budget scales with evidence; E1 envelope; composite protocol at batch 64; Demeaned corr-eye in
mixtures; fixed reference scales).

## 3. Carried over from v2

### Open work (`C`)

| ID | Item | Next action |
|---|---|---|
| C1 | **Composite-loss benchmark (tuned weights), both directions** (v2:E2.3 + E3's composite FC → SC). Optuna over scaled weights (Demeaned corr-eye, Neighbor dist, Var-match; raw Corr-eye excluded, v2:D5) jointly with `lr` / `l2_reg` / epochs, seeds 0–4, batch 64 (v2:D4), results in `model_benchmark/composite/<direction>/`. Candidates from v2:E2.2: SC → FC linear backbone, Conditional Gaussian, PCA-PLS learnable (Krakencoder as the deep reference); FC → SC PCA-PLS learnable, linear backbone. Then the final reoptimized benchmark (`model_benchmark/final/`). | Design with the user: model set, weight ranges (v2 E1 gradient-strength table), **selection metric** (val demeaned r, val avg_rank, or a combination; no weighting maximizes both). Also from v2 E3 Phase D: tune epochs with the weights and widen `linear_backbone` PCs (consensus hit 256) and `pca_pls_learnable` lr (near 3e-5). FC → SC note: v2 Phase D found no composite term helps FC → SC. Open: sources beyond SC (`SC_r2t`, `SC+SC_r2t`). |
| C2 | **Covariate ablation** for `CrossModal_PCA_PLS_CovProjector`: (a) `fs_all` without regional volumes + demographics, (b) demographics only; both directions, E2.2 protocol, ~1–2 GPU-h. | Needs approval. Resolves C!1. |
| C3 | **v2:I2.3 merged 2026-10-05** (`5efe03a`; branch and worktree removed): config family folders `models/configs/{null_ceiling,linear,latent,graph_nodal,deep}/`, `variants/<family>/`, name-based lookup with legacy fallback, ConditionalGaussian in `models/architectures/crossmodal_conditional_gaussian.py`. Checks on `main`: loss regression 45/45 vs `5efe03a^1`, E1.1 checks, `build_configs.py --check`, benchmark checks; benchmark configs semantically unchanged (13/13). | **Next:** a generic `scripts/sbatch/launch_tune.sh` + roster `submit.py` (per-model sbatch scripts → `scripts/sbatch/legacy/`), and source / covariate variants as `data:` / `model:` overrides in rosters. |
| C4 | Subject × subject connectome-similarity matrices, raw and demeaned, with interactive per-subject views (self-contained HTML). | — |
| C5 | Cross-condition fingerprinting (top-1, differential identifiability) and a network-ordered condition view. | User follow-ups from the v2:I1.5 notebook review. |
| C6 | FC → SC variants: message passing on FC thresholded at τ (`fc_graph_threshold`, default 0.5) for Chen GCN / Nodal GNN; no SC-derived node inputs (`use_r2t: false`); Nodal MLP reads FC rows; guard refusing `target == "SC"` with SC-derived inputs (v2:E2.0). | Low priority: all three are at the null SC → FC (v2:E2.2). |
| C7 | Align `activate_env.sh` and Jupyter kernels with `/ext3/env.sh`; delete `/scratch/asr655/envs/archive/2026-09-30_c6/` once jobs run clean (they have since 2026-09-30). | User. |
| C8 | **Masked MLP pretrainer FC → SC, all 5 seeds at full budget** (56 trials, 0.25 × 4 packing; ~6.3 GPU-h). It failed the v2:E2.2 latent gate (val 0.132 < 0.169) but is a competitive nonlinear model, so it is reported anyway (`gate_override` in `model_benchmark/config.yml`; marked ‡ in tables and figures; the gate result stays recorded). | When `19248105` finishes: `run.py --direction fc2sc`, update `model_benchmark.md`. **Jobs read `main.py`, `data/` and `models/` live: no edits there (e.g. E0's `HCP_Base` change) until the array is done.** |

### Caveats (`C!`)

| ID | Caveat | Affected artifacts | Resolved by |
|---|---|---|---|
| C!1 | `CrossModal_PCA_PLS_CovProjector` sees FreeSurfer `fs_all` + demographics. FC → SC its lead (v2:E2.2 test demeaned r 0.221, top-1 0.45 vs 0.162 / 0.13 for PCA-PLS learnable; v2 E3 Phase D top-1 0.48) most likely comes from regional volumes predicting the SC target's `invnodevol` normalisation. Reported as not input-matched (†), never a paired reference. | `model_benchmark/mse/fc2sc`, `composite_loss/pca_pls_covprojector/fc2sc`; to a lesser degree its SC → FC rows | C2 |

## 4. Experiments

### E0 — Which task FC best predicts SC   (slug: `task_fc_to_sc`) · status: planned · owner: agent:infra

Entry experiment of v3 (v2:I1.3). Single model, single loss; the only variable is which FC condition is the
source.

- **Question:** for FC → SC, which FC condition (rest or one of the seven HCP tasks) gives the best SC reconstruction
  on Pearson r, demeaned r, average rank and top-1?
- **Design:**
  - **Model / loss:** `CrossModal_PCA_PLS_learnable`, MSE-only, v2:E2.2 FC → SC protocol and config; Glasser.
  - **Conditions (source):** `rest` (reference) + `emotion`, `gambling`, `language`, `motor`, `relational`,
    `social`, `wm` (combined LR+RL relmats, v2:I1.4 caches). Target: SC (default metric, log1p).
  - **Matched subjects and splits:** every run loads all eight conditions (`HCP_Base(fc_conditions=[...all 7 tasks])`),
    so the cohort is the intersection — **917 of 957** subjects (missing per task: 5–15) — and identical across
    conditions. Splits are the per-seed `train_val_test` labels restricted to that cohort (seed 0: 659 / 74 / 184 vs
    683 / 79 / 195 on the full cohort), seeds 0–4. Rest is re-run on this cohort, so rest-vs-task differences are
    paired by seed and subject.
- **Code change (small):**
  - `HCP_Base(fc_source_condition=None)`: when set to a task, the FC arrays (`fc_upper_triangles` / `fc_matrices`)
    are rebound to that condition **before** the train-split PCA, so source `FC` *is* the task FC everywhere
    downstream (PCA bases, models, evaluator). Requires the condition in `fc_conditions`. `rest` = current behaviour.
  - Add `fc_conditions`, `fc_source_condition` to `DATA_KEYS` and to `main.py`'s data allow-list; condition logged in
    the W&B config and run name.
  - Check: with `fc_source_condition=rest` and all conditions loaded, the subset run reproduces the plain
    `HCP_Base` arrays restricted to the 917 subjects (bit-identical).
- **Outputs** (`scripts/experiments/task_fc_to_sc/`): tables per condition × seed with paired Δ vs rest; four bar
  charts (one per metric, one bar per condition, mean ± SE) as in `model_benchmark`; write-up.
- **Compute:** full E2.2 tune = 0.75 GPU-h per seed → 8 conditions × 5 seeds ≈ 30 GPU-h (packed, group GPU quota).
  Cheaper option: 16-trial tunes (≈ ½) or fixed hyperparameters from v2:E2.2's FC → SC consensus (≈ 3–4 GPU-h).
- **Caveat:** scan length differs by condition (rest ≈ 4 × 14.4 min vs tasks ≈ 2 × 2–5 min), so condition is
  confounded with data quantity.
- **Decisions pending (user):** tuning budget (full / 16-trial / fixed); include a closed-form null per condition
  (`CrossModalPCA`, CPU, minutes) as a reference line; also run SC → task FC (reverse) later.

## 5. Infrastructure

None scheduled yet beyond C3 and C7.

## 6. Backlog (not scheduled)

Moved from v2 §6 (originally v1 §6, v1 §8.6, the unrun parts of v1 M10, and E0 / E1 follow-ups):
- **Graph / nodal models on top of the mean** (from v2:E0): train GNNs on the subject's deviation from the train-split
  mean FC, or on PCA scores / low-rank factors, instead of on full FC edges; include an SC-only (`use_r2t: false`)
  `NodalGNN`. Pilot per v2:D2 against `PCA_PLS_learnable` on the same seeds. v2:E2.2 confirms the full-edge models
  are at the null.
- **Editable install** (`pyproject.toml` + `pip install -e .`; the overlay mounts `:rw` without `--fakeroot`) and
  `conn2conn/` namespacing.
- **Artifact cleanup:** `results/ray_results/` (118 GB) and a `results/logs/` retention policy.
- **Shared constants** between `main.py` and `scripts/results_utils/records.py`.
- **Untracked reference code** in `context_packages/modeling/*_context/`.
- **Latent-space terms inside composite** (`latent_mse` / `latent_weighted_mse` as mixable terms).
- **EMA diagnostics:** `*_loss_ref_*` behavior after warmup, and batch-size sensitivity of `correye` / `neidist` (64 vs 128).

## 7. Change log

| Date | Change |
|---|---|
| 2026-10-04 | v3 created at v2 closure: carried items C1–C12, caveats C!1–C!3, backlog moved; v2:D1–D6 in force. |
| 2026-10-05 | E0 specified (task FC → SC with PCA-PLS learnable, matched 917-subject cohort); C7 promoted to E0. |
| 2026-10-05 | C3: I2.3 merged (`5efe03a`), checks pass on `main`. C8 added: Masked MLP FC → SC full run (gate override), job `19248105`. |
| 2026-10-05 | Carry-over pruned to essentials (user): C4/C5 folded into C3; C7 → E0; C10, C11 and caveats on old ema runs / z-scored latents dropped; renumbered C1–C7, C!1. |

Last updated at: 2026-10-05 EDT
