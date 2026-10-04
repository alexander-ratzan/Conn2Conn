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
| C1 | Composite-loss benchmark with tuned weights, both directions; then the final reoptimized benchmark (v2:E2.3) | planned (design pending) | v2:E1, v2:E2.2 | — |
| C2 | Covariate ablation for PCA-PLS + covariates (v2:E2.2 / E3 Phase D) | planned (proposed 2026-10-04, not approved) | — | — |
| C3 | Merge I2.3 config family folders (branch `i2-config-layout`) | planned (no-jobs window) | — | agent:modeling |
| C4 | One tune launcher (v2:I2.4) | planned | C3 | — |
| C5 | Variants as overrides (v2:I2.5) | planned | C3 | — |
| C6 | Subject × subject connectome-similarity views (v2:I1.2) | planned | — | — |
| C7 | Which behavioral FC best predicts SC (v2:I1.3) | planned | C1 | — |
| C8 | Task-FC follow-ups: cross-condition fingerprinting, network-ordered view (v2:I1.5) | planned | — | — |
| C9 | FC → SC reverse variants for Chen GCN, Nodal GNN, Nodal MLP (v2:E2.0) | planned | — | — |
| C10 | `composite_loss/checks/check_protocol.py` reads a pre-E3.0 path | planned | — | — |
| C11 | `latent_masked_test` notebook fixes (v2:C4) | planned | — | — |
| C12 | Environment alignment and C6 archive deletion (v2:C6.5) | planned | user | user |
| C!1 | v1:M1b — `ema` runs with `neidist` ≤ 0 during warmup (v2:C!2) | open | — | — |
| C!2 | `CrossModal_linear_backbone` z-scored latents are PCA-space (v2:C!3) | open | — | — |
| C!3 | PCA-PLS + covariates FC → SC gain is likely anatomy, not FC | open | resolved by C2 | — |

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
| C2 | **Covariate ablation** for `CrossModal_PCA_PLS_CovProjector`: (a) `fs_all` without regional volumes + demographics, (b) demographics only; both directions, E2.2 protocol, ~1–2 GPU-h. | Needs approval. Resolves C!3. |
| C3 | **Merge v2:I2.3** (`7501e94`, worktree `../Conn2Conn_wt_i2`): config family folders `models/configs/{null_ceiling,linear,latent,graph_nodal,deep}/`, `variants/<family>/`, name-based lookup with legacy fallback, ConditionalGaussian moved to `models/architectures/crossmodal_conditional_gaussian.py`. | Merge only when no job is queued or running; re-run loss regression (45/45), E1.1 checks, `build_configs.py --check`, benchmark checks on `main`; remove the worktree. |
| C4 | Generic `scripts/sbatch/launch_tune.sh` + roster `submit.py` (the `model_benchmark` pattern); per-model sbatch scripts → `scripts/sbatch/legacy/`. | After C3; check no agent still submits the old scripts (Krakencoder's launcher stays). |
| C5 | Source / covariate variants (`_SC_r2t`, `_SC+SC_r2t`, `_demo`, `_fs_*`) as `data:` / `model:` overrides in rosters. | After C3. |
| C6 | Subject × subject connectome-similarity matrices, raw and demeaned, with interactive per-subject views (self-contained HTML). | — |
| C7 | With the best model, test which behavioral (task) FC best predicts SC; starting point for timeseries modeling. | After C1 picks the model; task-FC loaders exist (v2:I1.5). |
| C8 | Cross-condition fingerprinting (top-1, differential identifiability) and a network-ordered condition view. | User follow-ups from the v2:I1.5 notebook review. |
| C9 | FC → SC variants: message passing on FC thresholded at τ (`fc_graph_threshold`, default 0.5) for Chen GCN / Nodal GNN; no SC-derived node inputs (`use_r2t: false`); Nodal MLP reads FC rows; guard refusing `target == "SC"` with SC-derived inputs (v2:E2.0). | Low priority: all three are at the null SC → FC (v2:E2.2). |
| C10 | `check_protocol.py` still reads `composite_loss/linear_backbone/config.yml` (moved to `<model>/sc2fc/` in v2 E3.0). | Point it at the direction folders. |
| C11 | `scripts/notebooks/model_testing/latent_masked_test.ipynb`: cell 6 reads `residual_linear.weight` (absent in `attention_only`); cell 4 sets `l2_reg` twice. | Fix when next used. |
| C12 | Align `activate_env.sh` and Jupyter kernels with `/ext3/env.sh`; delete `/scratch/asr655/envs/archive/2026-09-30_c6/` once jobs run clean (they have since 2026-09-30). | User. |

### Caveats (`C!`)

| ID | Caveat | Affected artifacts | Resolved by |
|---|---|---|---|
| C!1 | v1:M1b (was v2:C!2) — any `ema` run whose `neidist` reached ≤ 0 during warmup had that term inflated ~10⁸-fold (includes the old `LatentAttnMasked` default composite). Which past runs were hit is not determined. | past `ema` composite runs, mainly `LatentAttnMasked` | re-run or audit if those results are reused |
| C!2 | (was v2:C!3) `CrossModal_linear_backbone(zscore_pca_scores=True)` returns PCA-space latents from `predict_target_latents`. Edge outputs are unchanged. | latent losses / diagnostics under z-scoring | informational |
| C!3 | `CrossModal_PCA_PLS_CovProjector` sees FreeSurfer `fs_all` + demographics. FC → SC its lead (v2:E2.2 test demeaned r 0.221, top-1 0.45 vs 0.162 / 0.13 for PCA-PLS learnable; v2 E3 Phase D top-1 0.48) most likely comes from regional volumes predicting the SC target's `invnodevol` normalisation. Reported as not input-matched (†), never a paired reference. | `model_benchmark/mse/fc2sc`, `composite_loss/pca_pls_covprojector/fc2sc`; to a lesser degree its SC → FC rows | C2 |

## 4. Experiments

None scheduled yet beyond C1 / C2.

## 5. Infrastructure

None scheduled yet beyond C3–C5, C10, C12.

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

Last updated at: 2026-10-04 EDT
