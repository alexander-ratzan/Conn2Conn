# Repo Spec v2 — Experiments

**Purpose:** run structured experiments on the v1 composite-loss / regularization machinery, clear the items carried over
from v1, and hold the repo backlog.
**Status:** closed 2026-10-04 · **Started:** 2026-09-29 · **Predecessor:** [`spec_doc_v1.md`](spec_doc_v1.md) (closed 2026-09-29) ·
**Successor:** [`spec_doc_v3.md`](spec_doc_v3.md) (open items carried there; see [Closure](#closure)) ·
**Format:** [`spec_conventions.md`](spec_conventions.md)

**Contents:** [Closure](#closure) · [Status](#status) · [1. Purpose](#1-purpose) · [2. Conventions](#2-conventions) ·
[3. Carried over from v1](#3-carried-over-from-v1) · [4. Experiments](#4-experiments) · [5. Infrastructure](#5-infrastructure) ·
[6. Backlog](#6-backlog-not-scheduled) · [7. Change log](#7-change-log)

This spec is closed: only typo fixes and successor pointers change. New work goes into [`spec_doc_v3.md`](spec_doc_v3.md).

## Closure

Closed 2026-10-04. Outcomes:
- **E1** (composite-loss dynamics, SC → FC) and its FC → SC replication (**E3 Phase D**): identity terms trade
  demeaned r for avg_rank SC → FC; the trade-off does not carry over FC → SC, where MSE-only is best.
- **E2.2** (MSE-only benchmark, both directions, 111.5 GPU-h): the linear family leads and is statistically tied SC → FC
  (demeaned r 0.092–0.098); graph / nodal models are at the null; FC → SC is easier for every model, led by PCA-PLS
  learnable and the linear backbone among input-matched models. Write-up `scripts/experiments/model_benchmark/model_benchmark.md`.
- **Resolved:** C2, C!1, C!4 (E2.2 reruns), plus C1, C3, C6.1–C6.4, C7 earlier.

Carried to v3 (IDs there):

| v2 item | Carried as | What remains |
|---|---|---|
| E2.3 (+ E3's composite FC → SC) | v3:C1 | composite-loss benchmark with tuned weights, both directions, then the final reoptimized benchmark |
| E2.2 / E3 Phase D caveat | v3:C2, v3:C!3 | covariate ablation for PCA-PLS + covariates (no-volume `fs_all`; demographics only) |
| I2.3 | v3:C3 | merge branch `i2-config-layout` (`7501e94`) in a no-jobs window |
| I2.4, I2.5 | v3:C4, v3:C5 | generic tune launcher; variants as overrides |
| I1.2, I1.3, I1.5 follow-ups | v3:C6, v3:C7, v3:C8 | similarity views; behavioral FC → SC; cross-condition fingerprinting + network view |
| E2.0 reverse variants | v3:C9 | FC → SC variants of Chen GCN, Nodal GNN, Nodal MLP |
| I2 finding | v3:C10 | `composite_loss/checks/check_protocol.py` reads a pre-E3.0 path |
| C4 | v3:C11 | `latent_masked_test` notebook fixes |
| C6.5 | v3:C12 | align `activate_env.sh` / Jupyter kernels with `/ext3/env.sh`; delete the C6 archive (user) |
| C!2, C!3 | v3:C!1, v3:C!2 | caveats unchanged |
| D1–D6 | in force | v3 cites them as v2:D1–D6 |
| §6 Backlog | v3 §6 | moved unchanged |

## Status

| ID | Title | Status | Depends on | Owner |
|---|---|---|---|---|
| E0 | Nodal models benchmark and architecture check (`nodal_models_benchmark`) | closed 2026-09-30 | — | agent:infra |
| E1 | Composite-loss dynamics and trade-off across models, SC → FC (`composite_loss`, grid v3) | done 2026-10-02 (E1.6–E1.10; conclusions → E2.3, now v3:C1) | D3, D4, D5, D6 | agent:modeling |
| E2 | Cross-model benchmark (`model_benchmark`) | done 2026-10-04 (E2.0–E2.2; E2.3 → v3:C1) | C2 | agent:modeling (E2.2) |
| E3 | Replicate E1 and E2 for FC → SC | done 2026-10-04 (E3.0, Phase D = E1 FC → SC; E2.2 FC → SC; composite FC → SC → v3:C1) | E1, E2, E2.0 | agent:infra, agent:modeling (E2.2 FC → SC) |
| I1 | HCP1200 timeseries and connectome-similarity views | closed (I1.1, I1.4, I1.5 done; I1.2, I1.3, I1.5 follow-ups → v3:C6–C8) | — | agent:infra (I1.1, I1.4, I1.5) |
| I2 | Repo organisation: experiment folders, config layout, launchers | closed (I2.1–I2.2 done; I2.3 built, merge → v3:C3; I2.4–I2.5 → v3:C4–C5; I2.6 deferred) | — | agent:modeling |
| C1 | `torch_geometric` missing from `kraken_env` | done 2026-09-30 (via C6) | — | agent:infra |
| C2 | Re-tune the M5b-affected sweeps | done 2026-10-04 (E2.2 reruns) | E2.2 | agent:modeling |
| C3 | Confirm `loss_signature` in Tune-trial W&B configs | done | — | agent:infra |
| C4 | `latent_masked_test` notebook fixes | carried → v3:C11 | — | — |
| C6 | Environment: two `kraken_env` stacks; jobs import from `~/.local` | done 2026-09-30 (C6.1–C6.4); C6.5 + archive deletion → v3:C12 | C6.5: user | agent:infra |
| C7 | Merge the batch-size fix + `extra_callbacks` hook + Ray CPU cap | done 2026-09-30 (`cfc1c32`) | — | agent:modeling |
| C!1 | v1:M5b — sampled L1/L2 not applied in past sweeps | resolved 2026-10-04 (C2 / E2.2) | C2 | — |
| C!2 | v1:M1b — `ema` runs with `neidist` ≤ 0 during warmup | open → v3:C!1 | — | — |
| C!3 | `CrossModal_linear_backbone` z-scored latents are PCA-space | open → v3:C!2 | — | — |
| C!4 | Single runs trained at batch 128 regardless of the tuned `batch_size` | resolved 2026-10-04 (C7 fix; E2.2 reruns) | C7, E2.2 | — |
| D1 | One job environment for all experiments and runs | decided | — | user |
| D2 | Tuning budget: pilot first, pack small models, scale on evidence | decided | — | user |
| D3 | E1 compute envelope and autonomous execution | decided | — | user |
| D4 | Composite-loss protocol batch size 64 | decided | — | user |
| D5 | Demeaned corr-eye is the corr-eye variant in composite-loss mixtures | decided | — | user |
| D6 | Fixed reference scales are the default way to balance composite-loss terms | decided | — | user |

---

## 1. Purpose

v1 gave every learned edge-space model one composite-loss path, searchable loss weights, unified `l1_reg`/`l2_reg`, and the
`CrossModal_linear_backbone` probe. v2 uses that machinery for structured experiments. It also records the carried-over
items that gate them and the backlog of repo work that is not yet scheduled.

## 2. Conventions

Document format: [`spec_conventions.md`](spec_conventions.md). Repo conventions are inherited from v1 (§3 and §8.2 there);
the table below lists only what is specific to v2.

| Topic | Convention |
|---|---|
| Experiment home | `scripts/experiments/<slug>/`: `<slug>.md` (question, design, how to run, W&B ids, results, caveats), `config.yml`, `run.py`, `tables/`, `figures/`, `manifest.json`, following the `sc_type_benchmark` / `cov_projector_benchmark` runner pattern. Add a row to `scripts/experiments/experiments_index.md`. |
| Reported metrics | Test-split `demeaned_pearson` and `avg_rank` (the `metric_scatter` axes), plus `pearson`, `mse`, `top1_acc` in tables. Selection is always on `val_demeaned_r`; never select on test. |
| Seeds | `shuffle_seed` 0–9 for benchmark-grade results; smaller seed sets are allowed for staged studies and stated per experiment. E1, E2 and E3 use seeds 0–4 (five family-preserving splits; seed 0 = the original split). |
| Loss normalization | **Fixed reference scales** balance composite terms: per-term constants measured on the model's MSE-only fit, `loss_normalize: none`, no EMA (E1.1; D6). `auto`/`ema` stay available. |
| Figures | Canonical figures are PNG, 300 dpi, per the scientific-figure-making skill, tracked. An experiment may add a **self-contained interactive HTML** (inline SVG + small script, no CDN or package dependency; `plotly` is not in `kraken_env`), tracked next to the PNG. |
| Compute | Training and sweeps go through `sbatch` (per-experiment launchers; generic launcher in I2.4). Each compute stage is approved before submission. The local L40S node is used only for short checks, and only inside a compute allocation, never on a login node. |
| W&B / run selection | Runs tagged with the experiment slug and grouped by `loss_signature`; staged runs carry the stage (`<slug>:stage1`). Benchmark results are selected from one campaign only (E2.2: job names `e2_mse_<Model>_<direction>` and their task logs), never the best over older sweeps. |

## 3. Carried over from v1

### Open work (`C`)

| ID | Item | Blocks | Next action |
|---|---|---|---|
| C1 | `torch_geometric` missing from the `kraken_env` overlay (`Chen2024GCN` / `NodalGNN` could not import) | — | **Done 2026-09-30** via C6 (`torch_geometric 2.8.0.post1` in the overlay; `dev_runs` `18880103`). |
| C2 | Re-tune the sweeps affected by v1:M5b (`CrossModal_PCA_PLS_learnable`, `CrossModal_PCA_PLS_CovProjector`, `Sarwar2020MLP`) | E2 | **Done 2026-10-04:** re-tuned per seed in E2.2 (both directions); resolves C!1. Old benchmark rows stay valid only as default-regularization results. |
| C3 | Confirm Tune-trial W&B configs carry `loss_signature` (v1 8.7 check failed) | — | **Done 2026-09-29.** False negative: wandb 0.25 offline runs keep the config inside `run-*.wandb`, which contains `loss_signature`. `verify_modeling_track.py` should read `run-*.wandb`. |
| C4 | `scripts/notebooks/model_testing/latent_masked_test.ipynb`: cell 6 reads `residual_linear.weight` (absent in `attention_only`); cell 4 sets `l2_reg` twice | — | Fix when that notebook is next used. |
| C6 | **Environment divergence** (found 2026-09-29): two `kraken_env` stacks, and jobs imported 38 packages from `~/.local`. Root cause: a non-writable overlay (root-owned skeleton dirs; `--fakeroot` unusable without subuid), so pip fell back to `~/.local`. | E0.5, E2, every run | **C6.1–C6.4 done 2026-09-30:** ownership fixed offline; `~/.local` packages + `torch_geometric` installed into the overlay; `/ext3/env.sh` sets `PYTHONNOUSERSITE=1`, `PIP_USER=0`; verified by `dev_runs` 11/11 (`18880103`). Snapshot `/scratch/asr655/envs/kraken_env/c6_snapshot_2026-09-30/`; backup in `/scratch/asr655/envs/archive/2026-09-30_c6/` (delete once jobs run clean). **Open: C6.5 (user)** — align `activate_env.sh` and Jupyter kernels with `/ext3/env.sh`. `~/.local` is kept as is (other projects may import from it). |
| C7 | Batch-size fix (C!4), `extra_callbacks` hook, Ray sized to the SLURM CPU allocation | — | **Done 2026-09-30** (`cfc1c32`; regression 45/45). |

### Caveats (`C!`)

| ID | Caveat | Affected artifacts | Resolved by |
|---|---|---|---|
| C!1 | v1:M5b — every sweep before 2026-09-23 of `CrossModal_PCA_PLS_learnable`, `CrossModal_PCA_PLS_CovProjector` and `Sarwar2020MLP` trained with the YAML default regularization (L2 = 1e-4 for the PCA/PLS models, none for Sarwar); W&B logged the sampled `l1_reg`/`l2_reg`, which were not applied. Results are valid as default-regularization results. | `sc_type_benchmark` (`PCA_PLS_learnable` rows); `cov_projector_benchmark` (`PCA_PLS_learnable`, all projector rows, `Sarwar2020MLP`) | C2 |
| C!2 | v1:M1b — any `ema` run whose `neidist` reached ≤ 0 during warmup had that term inflated ~10⁸-fold (includes the old `LatentAttnMasked` default composite). Which past runs were hit is not determined. | past `ema` composite runs, mainly `LatentAttnMasked` | re-run or audit if those results are reused |
| C!3 | `CrossModal_linear_backbone(zscore_pca_scores=True)` returns PCA-space latents from `predict_target_latents` (`LatentAttnMasked` returned z-space). Edge outputs are unchanged. | latent losses and latent diagnostics under z-scoring | informational; stays open while z-scored latents are in use |
| C!4 | `main()` builds `Sim` without a `batch_size` (so 128), and `_run_learned_single` reused those loaders unless covariate sources changed. Every `--report_best_after_tune` rerun and direct prod run therefore trained at **batch 128 whatever the config or best trial specified**; Tune trials themselves used the right batch size. Found 2026-09-30 by code reading. | best-trial test metrics of models tuned at batch ≠ 128: E0 `nodal_models_benchmark` (NodalMLP 8–64, NodalGNN 8), `sc_type_benchmark` / `cov_projector_benchmark` rows for `CrossModal_PCA_PLS_learnable` (64), the projector (64), `Sarwar2020MLP` (32), `Chen2024GCN` (4); E1 Stage 1 best-trial reports (not used for E1 results: E1.3/E1.4 build `Sim(batch_size=64)` explicitly) | C7 (fix), then re-runs within E2 / C2 |

### Decisions (`D`)

| ID | Decision | Rationale |
|---|---|---|
| D1 | **One job environment for all experiments and runs:** the launchers' stack (`/ext3/env.sh` in `kraken_env`), consolidated into the overlay with user site-packages disabled (C6). Notebooks and interactive sessions use the same stack. | Every recorded result came from the launcher stack; one stack makes interactive checks reproduce in jobs. |
| D2 | **Tuning budget scales with evidence.** A model without established signal starts with a pilot (1–2 seeds × 8–12 trials); full sweeps (32 trials, more seeds) only if the pilot's best val beats the null by a stated margin. Small models are packed onto the GPU (fractional `TUNE_GPUS_PER_TRIAL`, several trials at once). | NodalMLP probes took 38 array tasks / 612 trials / ~47 GPU-h for test demeaned r ≈ the null; one-trial-per-GPU jobs are killed for underutilization. |
| D3 | **E1 runs autonomously within a fixed envelope, per instance:** Stage 1 ≤ 4 GPU-h (packed), E1.3 ≤ 1 GPU-h, Stage 2 ≤ 10 GPU-h (8 until 2026-10-01); defaults approved (seeds 0–4, the current grid version, test metrics with selection on val, gradient-cosine panel). The agent stops and reports on any stop condition in the instance `config.yml` (weak Stage 1, wrong Stage 1 trial count, failed / non-finite / mismatched runs, budget overrun). A consensus-gate miss runs the re-check instead of stopping (E1.2). Relaxing a stop threshold needs the user and is recorded in the instance write-up. | Lets E1 proceed without per-stage approval, with explicit exits. |
| D4 | **The composite-loss protocol trains at `batch_size` 64 in both stages, for every model.** Models that cannot train at 64 skip the batch-dependent terms (`correye`, `neidist`) rather than change the batch. The envelope in D3 applies per instance. | `correye` / `neidist` compare subjects within a batch, so cross-model comparisons need one batch size; 64 fits the linear family and most learned models in the packed setup. |
| D5 | **Demeaned corr-eye (`correye_dm`) is the corr-eye variant in composite-loss mixtures** (grid v3; user 2026-10-01). Raw `correye` stays only as a single-term reference and is never combined with `correye_dm`. | Raw `correye` is inert until its gradient reaches Var-match's strength, then collapses predictions the same way; `correye_dm` acts on subject-specific deviations (E1 gradient-strength analysis, both instances). |
| D6 | **Fixed reference scales are the default way to balance composite-loss terms** in experiments and benchmarks (user 2026-10-02, closing E1): each term divided by $c_t$ measured on the model's own MSE-only fit, `loss_normalize: none`. The code default (`loss_normalize: auto`, which is plain MSE for a single term) is unchanged. Weights are then fractions of the MSE value, **not** of its gradient (E1: gradient ratios differ 100×+ across terms), so weight ranges are set per term. | Stable across seeds (cv ≤ 4% in every instance), reproducible, and comparable across models; EMA scales drift during training and were the source of v1:M1b. |

## 4. Experiments

Status and owners are in the [status table](#status).

### 4.0 Template

```
### E<n> — <title>   (slug: <slug>) · status: <status> · owner: <owner>
- Question:
- Design: (model(s), sources, seeds, stages, selection)
- Steps: E<n>.1 … each with Changes / Accept
- Outputs: tables, figures, experiment doc
- Budget: GPU-h per stage
- Depends on: carried items / other experiments
- Decisions: (with defaults if not yet confirmed)
```

### E0 — Nodal models benchmark and architecture check   (slug: `nodal_models_benchmark`) · status: closed 2026-09-30 · owner: agent:infra

- **Question:** how much of SC → FC can per-region (nodal) models recover (NodalMLP with probe edge decoders, NodalGNN,
  `Chen2024GCN`) against the null and the linear family? It also served as the last check of the v1 architecture.
- **Result** (write-up `scripts/experiments/nodal_models_benchmark/nodal_models_benchmark.md`): every nodal model is at
  the null (test demeaned r 0.011–0.023; null 0.013; MSE-only `PCA_PLS_learnable` 0.078 on the same seeds). Tuning does
  not help, and train demeaned r is also ≈ 0: the full-edge MSE formulation lets the population mean absorb the loss.
  The repo was confirmed working after the v1 refactor (launchers 59/59; full launcher cycle; deterministic runner).
- **Kept from E0:** `--tune_reuse_actors` (default true; keep reuse with packing, E0.6 `18871767`); fANOVA importance
  in `optuna_importance` (used for E2.2's NodalMLP narrowing); the "learn on top of the mean" follow-up (§6).

### E1 — Composite-loss dynamics and trade-off across models   (slug: `composite_loss`) · status: done 2026-10-02 · owner: agent:modeling

Experiment home `scripts/experiments/composite_loss/`: protocol write-up `composite_loss.md` (protocol, loss math,
gradient-strength analysis, cross-model comparison), grids `grid.yml` (v1), `grid_v2.yml`, `grid_v3.yml` (current;
released grids are frozen), `checks/`, and one folder per model instance (`<model>/`: `<model>.md`, `config.yml`,
`stage1/`; generated `state.yml`, `runs/`, `tables/`, `figures/`).

- **Question:** with MSE fixed at weight 1, how do Var-match, Corr-eye, Demeaned corr-eye and Neighbor dist shape
  training dynamics and trade test `demeaned_pearson` against `avg_rank`, and does that landscape hold across model
  types? (SC → FC; FC → SC is E3.)
- **Protocol (v3), per model instance:**
  1. **Stage 1:** MSE-only Optuna tune per seed (seeds 0–4, packed), all terms logged as monitor-only terms.
  2. **Consensus + reference scales:** one consensus config, trained on seeds 0–4. Categorical keys are chosen by
     majority, loguniform keys by geometric median, and uniform keys and ordered `consensus: median` choices by median.
     Fixed scales $c_t = s_t / s_{\text{mse}}$ are measured on these fits.
  3. **Grid v3** (29 combinations × 5 seeds), with Stage 1 hyperparameters held fixed and `loss_normalize: none`:
     - MSE-only;
     - 22 single-term points: Var-match and Neighbor dist at 0.1 / 0.5 / 1; Demeaned corr-eye and Corr-eye at 0.1–50;
     - a 2 × 2 × 2 factorial at 0.5 over Var-match × Demeaned corr-eye × Neighbor dist;
     - three-term doses at 0.1 and 1.

     Corr-eye and Demeaned corr-eye never appear in one combination (D5).
  4. **Report:** paired-by-seed effects vs MSE-only; trade-off scatter and interactive HTML (top-1 in the panel); dose
     response; term trajectories; loss composition; gradient cosines.
- **Budget / autonomy:** D3, per instance; batch 64 (D4).

#### E1.1–E1.5 — Protocol machinery · done
- **E1.1** fixed per-term `scale` and monitor-only terms in `CompositeLoss`: `75b7c11`; regression 45/45, checks
  23/23 (`scripts/sbatch/checks/loss_regression.py`, `composite_loss/checks/check_e1_loss.py`).
- **E1.2–E1.5** Stage 1 / consensus / grid / report runner (`scripts/results_utils/loss_grid.py`,
  `composite_loss/protocol.py`, `report.py`, launchers, `checks/check_protocol.py`): `a8362bc` (rebaseline re-check),
  `4ed6c44` (`correye_dm` term, grid v2, in-job re-check, Stage 1 trial-count guard), `386573a` (grid v3, median
  consensus, grid filter), `09ff5ea` (readable labels).
- **Consensus gate (E1.2):** accept if the consensus mean val is within 1 SE of the mean of the per-seed Stage 1 bests.
  On a miss, the same job retrains each seed's own best config (epochs capped at what ASHA trained), which removes the
  best-of-N bias, and re-checks against those. If it still misses, it is accepted with a recorded note
  (`consensus_check.basis`).

#### E1.6 — Instance `linear_backbone` · done 2026-10-01
`CrossModal_linear_backbone`, Stage 1 24 trials (v1). Gate passed on the re-check. Grid v3: 145 runs (143 from v2 on the
same Stage 1 and splits). Findings:
- **Demeaned corr-eye 0.1:** Δ avg_rank +0.092, Δ demeaned r −0.017, top-1 ×2, test MSE unchanged.
- **Neighbor dist 0.1:** +0.050 / −0.010.
- **Raw Corr-eye:** inert up to w ≈ 5, collapses at w ≥ 10.
- The v1 combinations reproduce v1 to within about 0.001.

Write-up `composite_loss/linear_backbone/linear_backbone.md`; `876f50a`.

#### E1.7 — Instance `pca_pls_learnable` · done 2026-10-01
`CrossModal_PCA_PLS_learnable`. Architecture hand-selected and fixed (user): 256 / 16 / 256, only `W_mid` learnable.
Stage 1 tunes the optimizer only: 16 trials over `lr` 3e-5 to 3e-3, `l2_reg`, `dropout`, epochs 100–250.
- **Stop-threshold deviation (user):** 0.09 → 0.085, because seed 1 reached 0.0898.
- **Gate:** passed directly. Consensus `lr` 5.4e-4, 150 epochs.
- **Replication:** effects correlate with the linear backbone at 0.92 (Δ demeaned r) and 0.98 (Δ avg_rank). All three
  terms at 0.1 give +0.106 / −0.018, the best trade-off on either model.
- **Superseded run:** the earlier tuned 64 / 4 / 64 consensus (`lr` 3.8e-5 × 50 epochs) barely trained and is
  overwritten (W&B `loss_grid:v2`).

Write-up `composite_loss/pca_pls_learnable/pca_pls_learnable.md`; `876f50a`.

#### E1.8 — Instance `krakencoder` · done 2026-10-02 (run under E2.1)
Krakencoder's own loss grid (`composite_loss/krakencoder/`, write-up `krakencoder.md`): 21 cells × seeds 0–4, paper
architecture, Krakencoder-native weights (not E1's scaled footing; compare directions, not numbers). Its native
`correye` acts in a mean-centred PCA space, so it corresponds to our Demeaned corr-eye (D5).
- **SC → FC:** `correye` raises avg_rank +0.12 at the paper weight (+0.16 at 2×; top-1 0.046 → 0.092) for −0.006
  demeaned r.
- **Neighbor dist is inert** at 0.1–2×, so the paper loss behaves as `correye` alone. Var-match slightly raises
  demeaned r and lowers avg_rank.
- **FC → SC** (for E3): `correye` costs demeaned r steeply (−0.030 at 1×) with no avg_rank gain.

#### E1.9 — Instance `pca_pls_covprojector` (all covariates) · done 2026-10-02 · owner: agent:modeling
`CrossModal_PCA_PLS_CovProjector` = the E1.7 model plus a covariate branch: FreeSurfer `fs_all` + age, sex,
race/ethnicity, mid-size projectors, MLP fusion. Hand-selected config (`fixed_consensus`, no Stage 1; user) at its own
validation-chosen budget of 30 epochs (E1.7's 150 overfit with covariates). Earlier runs are overwritten (W&B): the
frozen-backbone model default matched the March benchmark, and the 150-epoch run was confounded by overfitting.
- **Covariates trade identifiability for fidelity:** under MSE, demeaned r 0.110 (highest of any instance) but
  avg_rank 0.685 (E1.7 0.768).
- **Identity terms add +0.08 to +0.13 avg_rank,** short of E1.7's ceiling (best 0.819 / top-1 0.08 vs 0.88 / 0.13).
- **Demeaned corr-eye 0.1 improves both metrics:** demeaned r 0.119 (best cell of any instance), avg_rank +0.084.

Write-up `composite_loss/pca_pls_covprojector/pca_pls_covprojector.md`.

#### E1.10 — Cross-model interactive comparison · done 2026-10-02 · owner: agent:modeling
`composite_loss/compare.py` → `figures/cross_model_interactive.html` + `tables/cross_model_summary.csv`, rebuilt by
every instance report (`b11fcb3`, zoom toggle `0b6a768`): test demeaned r vs avg_rank on fixed axes, per-model toggles,
hover / click panel. Ceiling from `ceiling/test_retest.py`: demeaned r 0.49, avg_rank 0.987, top-1 0.93 (195 subjects
with both sessions per split).

#### E1 conclusions (closed 2026-10-02; carried into E2.3 → v3:C1)
- **The trade-off is real and shared across models.** Identity terms raise avg_rank / top-1 and cost demeaned r; test
  MSE barely moves. Effects correlate across the linear-family instances at 0.61–0.92 (Δ demeaned r) and 0.80–0.98
  (Δ avg_rank); Krakencoder shows the same direction for its native (de-meaned) corr-eye.
- **Term set for E2.3:**
  - Demeaned corr-eye (D5) is the strongest and most consistent identity term: top-1 about ×2–3, MSE unchanged.
  - Neighbor dist is strong on our models but inert in Krakencoder.
  - Var-match and raw Corr-eye are references only: both collapse predictions at high weight.
- **Weights:** scale by fixed reference scales (D6), but set each term's range from its gradient strength (scaled
  gradient ÷ MSE's: Demeaned corr-eye 3–11, Neighbor dist 7–13, Var-match 0.3–0.6, raw Corr-eye ≈ 0.05). Useful
  ranges are about 0.1–1 for Demeaned corr-eye and Neighbor dist.
- **Cheapest good settings:**
  - Demeaned corr-eye 0.1, or all three terms at 0.1, keeps most of the avg_rank gain at little or no demeaned-r cost.
  - On the covariate model, Demeaned corr-eye 0.1 improves both metrics (demeaned r 0.119, the best cell overall).
- **Selection metric is a decision for E2.3:** no single weighting maximizes both demeaned r and avg_rank. E2.3 must
  choose val demeaned r, val avg_rank, or a combination per model.
- **Each model at its own validated budget:** E1.9's 150-epoch run (E1.7's budget) overfit and inflated the identity
  gains. Composite-loss comparisons need each model's own MSE-validated training length.
- **Covariates** (E1.9) raise demeaned r and lower identifiability vs the same model without them.
- **Far from the ceiling:** test-retest is demeaned r 0.49 / avg_rank 0.987 / top-1 0.93; the best cells reach about
  0.12 / 0.88 / 0.13.

### E2 — Cross-model benchmark   (slug: `model_benchmark`) · status: done 2026-10-04 (E2.3 → v3:C1) · owner: agent:modeling (E2.2)

- **Question:** on equal footing, how are test Pearson r, demeaned r, avg_rank and top-1 distributed across model
  types, first MSE-only (E2.2), then with composite losses tuned where possible (E2.3, carried to v3)? Both
  directions (the FC → SC half is E3's E2 replication).
- **Design:** each model tuned per seed (Optuna or full grid over its benchmark config, selection on val demeaned r,
  best-trial report), seeds 0–4, one campaign per model; models grouped by type as in the README "Models" table.
- **Depends on:** C2 / C!1 and C!4 (resolved by the E2.2 reruns); E1 for E2.3.
- **Not decided in v2:** sources beyond SC (`SC_r2t`, `SC+SC_r2t`); open for v3:C1.

#### E2.0 — Direction audit (`SC → FC` vs `FC → SC`) · done 2026-09-30
The shared pipeline is direction-agnostic: dataset `x` / `y`, loss, evaluator (has a `target == "SC"` branch) and the
PCA helpers (`get_modality_data`) follow `--source` / `--target`; `HCP_Base(source="FC", target="SC")` builds.
Exceptions are batch extras that are **always SC / anatomy regardless of direction**: `sc_matrix` (NodalMLP) and
`node_features` (volume, centroid, `SC_r2t`; NodalMLP, NodalGNN). Target ranges: FC −0.81 … 0.96; SC (log1p)
0 … 3.59, 31 % zeros. FC → SC is now exercised by Krakencoder (E2.1) and E3 Phase D.

| Model | SC → FC | FC → SC | Why / what is needed |
|---|---|---|---|
| Linear decomposition (6), `CrossModalPCA`, `CrossModalVAE` | ✓ | ✓ generic | built from `get_modality_data` source/target means, loadings, scores |
| `LatentAttnMasked`, `MaskedLatentPretrainer`, `MaskedMLPPretrainer` | ✓ | ✓ generic | `sc_*` / `fc_*` names are legacy labels for the source / target roles |
| `Sarwar2020MLP` | ✓ | ⚠ config | default `output_tanh: true` bounds outputs to [−1, 1] but SC reaches 3.59: set `output_tanh: false` for `FC → SC` |
| `Krakencoder`, `Krakencoder_precomputed` | ✓ | ✓ | the loader selects the target's output key and targets (E2.1) |
| `TestRetestPrecomputed` | ✓ | ✗ | FC-only (no SC retest sessions in the dataset); see ceiling note below |
| `NodalMLP` | ✓ | ✗ reverse variant | SC-row input is always the subject's SC — in `FC → SC` that is the target (leak) |
| `NodalGNN` | ✓ | ✗ reverse variant | message passing uses the source edges as weights (negative FC breaks GCN degree normalization); `r2t` node features are SC-derived (leak) |
| `Chen2024GCN` | ✓ | ✗ reverse variant | same negative-weight problem when the source graph is FC |

**Reverse variants (`FC → SC`) for the graph / nodal models** (requirement, not built; carried as v3:C9):
- Message-passing graph from the source FC, **thresholded**: keep edges with FC > τ, weights = FC; τ is a model hparam
  (e.g. `fc_graph_threshold`), default **0.5**, searchable. Applies to `Chen2024GCN` and `NodalGNN`.
- No SC-derived node inputs: `NodalGNN` with `use_r2t: false` (volume / centroid only, or identity); `NodalMLP` reads
  rows of the **source** matrix (FC rows) instead of `sc_matrix`, same thresholding option.
- A guard in each model: refuse `target == "SC"` when any SC-derived input is enabled.

**`FC → SC` ceiling:** none in the data (no SC retest); a literature value (dMRI structural-connectome test-retest
reliability) is cited as a reference line.

#### E2.1 — Retrainable Krakencoder baseline · done 2026-10-02
Krakencoder becomes a refittable benchmark model instead of cached predictions.
- **Code:** `models/architectures/krakencoder/`: upstream vendored unmodified at `b57e39c` (`vendor/`, `VENDOR.md`);
  `retrain.py` builds inputs from `HCP_Base` (same inputs and splits as March), trains, infers both source flavors and
  writes `results/krakencoder/<tag>/seed{S}/`; model `Krakencoder` (config `Krakencoder.yml`, recipe in `retrain:`)
  serves both directions through `main.py`; `checkpoint_eval.py` scores every checkpoint with our metric and loss
  definitions. Launcher `scripts/sbatch/Krakencoder/train_array_krakencoder_seeds.sh`. The old local copy is
  `krakencoder_experimental/` (gitignored; still holds `participants.tsv` used by `data/dataset_utils.py`).
- **Parity** (seed 0, default recipe): retrained vs March cached test demeaned r 0.083 vs 0.083 (SC → FC) and 0.121 vs
  0.120 (FC → SC); per-subject r between prediction sets 0.999 / 1.000.
- **Loss grid** (= E1.8; `composite_loss/krakencoder/`, write-up `krakencoder.md`): 21 cells × seeds 0–4, 40.7 GPU-h,
  Krakencoder-native weights, batch 64. Its `correye` drives the trade-off; the paper loss ≈ `correye` alone. The
  batch-64 FC → SC parity gap (−0.017) was accepted by the user. Init-seed noise is 5–25× below split-seed noise.
- **Pattern for other external baselines:** vendor upstream unmodified, adapt only in a wrapper, build inputs from
  `HCP_Base`, serve predictions through a loader, gate on parity.

#### E2.2 — MSE-only benchmark, both directions · done 2026-10-04 · owner: agent:modeling
Write-up `scripts/experiments/model_benchmark/model_benchmark.md` (design, per-model budgets, results, caveats);
results `model_benchmark/mse/{sc2fc,fc2sc}/` (tracked records, tables, figures).
- **Protocol:** seeds 0–4, each model tuned per seed and selected on val demeaned r, MSE only (two native-objective
  rows labelled: Masked MLP `latent_mse`, Krakencoder); trials `clamp(8 × free keys, 16, 64)` or the full grid
  (closed-form) or the audit-narrowed search (Sarwar, Chen, Nodal GNN, Nodal MLP, Masked MLP); each model at its own
  batch size and budget; one campaign per model × direction (job names `e2_mse_<Model>_<direction>`).
- **Roster:** null, PLS-SVD, PCA-PLS, Conditional Gaussian (first benchmark), PCA-PLS learnable, linear backbone,
  PCA-PLS + covariates (`fs_all` + demographics), Masked MLP pretrainer (nonlinear; pilot-gated, D2), Sarwar, Chen,
  Nodal GNN, Nodal MLP; reused Krakencoder (E2.1; MSE and paper-loss variants) and the test-retest ceiling (E1.10).
  FC → SC drops Chen / Nodal GNN / Nodal MLP (no reverse variants, E2.0). Excluded: `LatentAttnMasked`,
  `MaskedLatentPretrainer`, `CrossModalVAE`.
- **Latent gate:** passed SC → FC (0.088 vs 0.096 − 0.01); **failed FC → SC** (0.132 vs 0.179 − 0.01): negative result.
- **Execution:** pilot → budgets → full runs via `model_benchmark/autopilot.py` under the 130 GPU-h cap; **111.5
  GPU-h** (SC → FC 80.4, FC → SC 31.1). Four tasks killed by the cluster at 0.5 GPU × 2 trials (GPU-underuse policy)
  were rerun at 0.25 × 4. Code `da4ab95` (build), `8626b23` / `350a9b9` (autopilot, Krakencoder variants, paired table,
  scatter), `1e53f20` (gate-failed exclusion, † marking, ceiling-bar panel).
- **Result (test demeaned r / avg rank):**
  - **SC → FC:** linear backbone 0.098 / 0.744, PCA-PLS + covariates 0.097 / 0.696, Conditional Gaussian 0.093 /
    0.744, PCA-PLS 0.092 / 0.710 (all within 1 SE); Masked MLP 0.090; Krakencoder MSE 0.086 / 0.683, paper loss
    0.080 / **0.802** (top-1 0.067); Sarwar 0.068; Chen, Nodal MLP, Nodal GNN 0.022–0.011 ≈ null 0.011; ceiling
    0.49 / 0.987.
  - **FC → SC:** PCA-PLS + covariates † 0.221 / 0.973 (top-1 0.45; not input-matched, v3:C!3); PCA-PLS learnable
    0.162 / 0.889; linear backbone 0.158; PCA-PLS 0.147; Krakencoder MSE 0.134 / 0.902; Conditional Gaussian 0.132;
    Sarwar 0.131; PLS-SVD 0.127; Krakencoder paper 0.105; null 0.007.
- **To the composite benchmark (v3:C1):** SC → FC linear backbone, Conditional Gaussian, PCA-PLS learnable
  (Krakencoder as the deep reference); FC → SC PCA-PLS learnable, linear backbone.

#### E2.3 — Composite-loss benchmark (tuned weights) · carried → v3:C1
Absorbs the former E3 outline (composite-loss magnitude tuning, never started).
- **Design:** the E2 roster where the model trains at batch 64 (D4). Composite weights are tuned per model: Optuna over
  the scaled weights (`loss_weight_*` with each model's fixed scales, measured as in E1.2) jointly with `lr` /
  `l2_reg`. Seeds 0–4.
- **E1 informs:** the term set (Var-match, Demeaned corr-eye, Neighbor dist; raw Corr-eye excluded, D5), the weight
  ranges (value-matched weights are not gradient-matched; E1 gradient-strength table), and the selection metric. The
  trade-off makes the choice of selection metric (demeaned r vs avg_rank, or a combination) a decision to make here.
- **Depends on:** E1, E2.2.

### E3 — FC → SC: replicate E1 and E2   (slug: `composite_loss/<model>/fc2sc` for E1) · status: done 2026-10-04 · owner: agent:infra

- **Question:** do the E1 loss landscape and the E2 model comparison hold in the reverse direction (FC → SC)?
- **Design:** the E1 protocol and the E2.2 / E2.3 benchmark with `--source FC --target SC`, same seeds and grid.
  Models follow the E2.0 direction audit: generic models as is; `Sarwar2020MLP` with `output_tanh: false`;
  Krakencoder's loader already serves both directions (E2.1); graph and nodal models need the reverse variants
  (not built).
- **Ceiling:** none in the data (no SC retest). A literature value is used as a reference line (E2.0).
- **E3.0 — Bidirectional layout and tooling** · done 2026-10-02 (owner agent:infra). `composite_loss/<model>/<direction>/`
  (`sc2fc` / `fc2sc`), each a protocol instance (`--instance <model>/<direction>`); model write-up at `<model>/<model>.md`.
  - Phase A: Krakencoder as the template (one fit, direction folders hold `tables/` + `figures/` views).
  - Phase B: `loss_grid` direction from source/target (checked against the folder; in W&B tags `direction:<d>`, run
    names and records); `compare.py` direction switch with per-direction axes and ceilings (`ceiling/*.json`);
    `protocol.py scaffold --instance <model>/sc2fc --to fc2sc` (hand-written files, source/target, paths, job names).
  - Phase C: the three E1 instances moved to `<model>/sc2fc/` (480 run records' `instance` rewritten); every table
    (minus `instance` / `direction`), figure (21/21 byte-identical) and the cross-model summary re-render identically.
  - **Phase D — E1 protocol FC → SC · done 2026-10-02** (agent:infra; 12.0 GPU-h, inside D3 for every instance).
    Write-ups: `composite_loss/composite_loss.md` § FC → SC and the `fc2sc` sections of each model write-up.
    - Calibration: closed-form `CrossModal_PCA_PLS` val demeaned r 0.089 (SC → FC) vs 0.143 (FC → SC), null 0.009 /
      0.008 (`checks/direction_baselines.json`); Stage 1 stop thresholds × 1.607, rounded down: 0.14 / 0.13.
    - Runs: `linear_backbone/fc2sc` and `pca_pls_learnable/fc2sc` full protocol (Stage 1 → consensus → grid 145 →
      report); `pca_pls_covprojector/fc2sc` with a Stage 1 **pilot** (user 2026-10-02: 4 optimiser keys, 12 trials ×
      5 seeds, epochs 20–120; pilot gate passed) instead of SC → FC's hand-selected config.
    - **Result:** FC → SC is easier (MSE-only demeaned r 0.14–0.21, avg_rank 0.89–0.98) and the SC → FC trade-off does
      not carry over: Demeaned corr-eye 1 costs −0.03 to −0.09 demeaned r and *lowers* avg_rank in every model
      (Krakencoder too); MSE-only is the FC → SC choice. CovProjector (top-1 0.48) most likely exploits `fs_all`
      volumes predicting the `invnodevol` SC normalisation — anatomy, not FC; needs a no-volume ablation.
    - **Open flags (not re-run):** `linear_backbone` consensus 256 PCs = top of its range; `pca_pls_learnable` lr near
      the 3e-5 floor with epochs near the top; composite runs reuse the MSE-tuned schedule (early peak then decline in
      `pca_pls_learnable`). Carried into v3:C1 (tune epochs with the weights, widen these ranges).
    - FC → SC ceiling: skipped (user 2026-10-02; no SC test-retest in the data).
- **E2 FC → SC** · done 2026-10-04 with E2.2 (agent:modeling; roster entries only, no new code): 9 models + Krakencoder,
  31.1 GPU-h; results in E2.2. Confirms Phase D: FC → SC is easier for every model and the covariate model's lead is
  anatomy-driven (v3:C!3). The composite FC → SC benchmark is part of v3:C1.
- **Depends on:** E1, E2, E2.0.

## 5. Infrastructure

### I1 — HCP1200 timeseries and connectome-similarity views   · status: closed (I1.2, I1.3, I1.5 follow-ups → v3:C6–C8) · owner: —

- **I1.1 — Move the HCP1200 timeseries into the data folders** · done 2026-10-01 (agent:infra). Add-only merge into
  `HCP1200/HCP1200_fMRI/xcpd-0-9-1` (job `18978259`, verify passed: 569,808 files moved, 0 errors; never overwrote
  existing files); staging cleaned, the `.tar` backup kept. Procedure: `context_packages/HCP1200_xcpd_transfer_merge_handoff.md`.
- **I1.2 — Subject × subject connectome-similarity matrices** · carried → v3:C6 · owner: —
  - A correlation matrix over all subjects' connectome comparisons (diagonal = same subject), in raw and **demeaned**
    form (training-set mean subtracted, as in Demeaned corr-eye).
  - Interactive per-subject matrix views of the full and the demeaned connectome.
  - Self-contained HTML per the figure convention (§2).
- **I1.3 — Which behavioral FC best predicts SC** · carried → v3:C7 · owner: —
  - With the best E2 model, test which behavioral FC (from the I1.1 timeseries) best predicts SC.
  - This is the starting point for timeseries modeling, with room for spatial analyses.
  - **Depends on:** I1.1, E2, E3 (the FC → SC path).
- **I1.4 — Task FC caches** · done 2026-10-02 (agent:infra).
  - 14 new caches `Conn2Conn_data/fc/parc-{4S456Parcels,Glasser}_hemi-both_task-{emotion,gambling,language,motor,relational,social,wm}/`,
    from the xcp-d combined-run (LR+RL) relmat; same three `.npy` files as rest + `manifest.json`; `fc/catalog.tsv`,
    `availability.tsv`, `README.md`. Per-run (`dir-LR` / `dir-RL`) relmats stay in the xcp-d tree only (user).
  - Builder `data/data_caching/build_fc_cache.py` (array `scripts/sbatch/data/build_fc_cache_array.sh`, job `19047450`,
    commit `0131ad1`; moved `cc78287`). Rest rebuilt bit-identical to the existing caches; every task cache matched
    25 re-read source TSVs exactly.
  - Five source files were damaged before the transfer (the `.tar` holds the same bytes). The only one in a cache,
    sub-118831 emotion 4S456 (truncated), is excluded via `data/data_caching/fc_cache_exclusions.tsv`; sub-120111 rest
    4S456 combined timeseries is truncated and matters for timeseries work.
  - New entries under `Conn2Conn_data/` and the HCP1200 tree inherit only a non-owner NFSv4 ACE and come out mode 000;
    the builder copies the existing caches' ACLs.
- **I1.5 — Condition loaders and FC EDA notebook** · done 2026-10-04, merged `5f5d238` (branch `fc-conditions`
  deleted); notebook in user review · owner: agent:infra
  - **Loaders:** `load_fc_precomputed(task=, load_matrices=)`; `HCP_Base(fc_conditions=[...])` joins the task caches to
    the canonical subject set (917 Glasser / 916 4S456 with all conditions; Glasser split 659/74/184);
    `HCP_Base.subject_order()` is shared with `Evaluator` (same orderings).
  - **Fix:** `trainvaltest_partition_indices` were positions in the pre-intersection subject list. Existing runs are
    unaffected (all 957 metadata subjects have every current modality); requiring the tasks drops 40 and would have
    shifted 806 indices. Indices now come from the canonical `metadata_df`, with an order check.
  - **Views** (`data/data_viz.py`, notebook `scripts/notebooks/EDA/FC_matrix_analysis.ipynb`): condition connectome
    grid (population / Fisher-z / subject, raw or minus the condition's train mean); condition × condition edge r,
    Euclidean and affine-invariant geodesic distance (λ = 0.1 shrinkage at subject level: task scans have fewer TRs
    than parcels); subject × condition correlation map, raw and demeaned, with within- vs between-subject summaries,
    family / demographic / age ordering, per partition. Static PNG views; the interactive HTML of I1.2 is not built.
  - **Checks:** `scripts/sbatch/checks/verify_fc_conditions.{py,sh}` pass on both parcellations (default `HCP_Base`
    identical to `main`; condition rows = cache rows; partition indices; orderings; geodesic vs reference; brute-force
    within/between means). The notebook ran end to end headless (`run_notebook_cells.sh`, job `19069236`).
  - **First numbers (Glasser val, demeaned):** within-subject cross-condition r 0.16 vs between-subject 0.00; task–task
    0.06–0.16 vs rest S1–S2 0.50; emotion (352 TRs) is farthest from every condition.
  - **Merged** on user request (2026-10-04, no E1/E2 job queued); `verify_fc_conditions` re-run on `main` against
    `350a9b9`: passes on both parcellations (job `19178535`). Figures: `results/figures/fc_conditions{,_checks}/`.
  - **Follow-ups (user; carried → v3:C8):** cross-condition fingerprinting (top-1, differential identifiability); network-ordered view.

### I2 — Repo organisation: experiment folders, config layout, launchers   · status: closed (open parts → v3:C3–C5) · owner: agent:modeling

**Goal:** a layout that scales to more models, both directions and the E2.3 / final benchmarks without copied files.
Started from user review 2026-10-02.

- **I2.1 — One cross-model benchmark folder** · done (`43c9174`, `4e93dd7`).
  - `multimodel_scfc/audit/` moved into `model_benchmark/audit/`; `multimodel_scfc/` removed.
  - Results land in `model_benchmark/<campaign>/<direction>/` (`mse/` for E2.2; E2.3 adds `composite/`; the final
    benchmark `final/`).
- **I2.2 — ConditionalGaussian search** · done (`43c9174`, `4e93dd7`). The benchmark config fixes `fit_domain: pca`; the model
  rejects `raw_edges` with shrinkage estimators, and 168 of the pilot's trials errored.
- **I2.3 — Config family folders** · built on branch `i2-config-layout` (`7501e94`; worktree `../Conn2Conn_wt_i2`); merge carried → v3:C3.
  - **Layout:** `models/configs/{null_ceiling,linear,latent,graph_nodal,deep}/<Model>.yml`;
    `variants/<family>/<Model>_<variant>.yml`; `benchmark/<campaign>/` reached by path only.
  - **Lookup:** `models/registry.py` finds a config by name. Old flat paths (`models/configs/<name>.yml`, used by the
    legacy launchers and other experiments' configs) fall back to the new location with a notice.
  - **Model file:** `CrossModal_ConditionalGaussian` moves from `latent_attention/` to
    `models/architectures/crossmodal_conditional_gaussian.py` (linear family).
  - **Checks:** loss regression 45/45 bit-identical; E1.1 checks pass; benchmark configs regenerate identically.
  - **Merge rule:** merge only when no job is queued or running (it touches `models/`).
- **I2.4 — One tune launcher** · carried → v3:C4.
  - A generic `scripts/sbatch/launch_tune.sh` + roster `submit.py` (the `model_benchmark` pattern, generalised).
  - The ~60 per-model sbatch scripts move to `scripts/sbatch/legacy/`, after checking no other agent still submits
    them (Krakencoder's launcher stays).
- **I2.5 — Variants as overrides** · carried → v3:C5. Source and covariate variants (`_SC_r2t`, `_SC+SC_r2t`, `_demo`,
  `_fs_*`) become short `data:` / `model:` overrides in experiment rosters instead of near-copy files.
- **I2.6 — Hydra** · deferred (user 2026-10-02): too large a refactor for the benefit; I2.3–I2.5 cover the need.
- **Found while testing:** `composite_loss/checks/check_protocol.py` fails on `main`: it still reads
  `composite_loss/linear_backbone/config.yml`, moved to `<model>/sc2fc/` in E3.0 Phase C. Carried → v3:C10.

## 6. Backlog (not scheduled)

Moved unchanged to [v3 §6](spec_doc_v3.md#6-backlog-not-scheduled) at closure (v1 §6 / §8.6 items, unrun v1 M10
parts, E0 / E1 follow-ups).

## 7. Change log

| Date | Change |
|---|---|
| 2026-09-29 | v2 created: conventions, carried items C1–C5 from v1, E1 (planned: staged composite-loss trade-off with fixed reference scales), E2 (outline: cross-model benchmark), backlog moved from v1. |
| 2026-09-29 | Reformatted to [`spec_conventions.md`](spec_conventions.md): purpose, contents, status table with owners. Carried items split into open work (C1–C4) and caveats: C!1 (v1:M5b), C!2 (v1:M1b), C!3 (was C5). C3 done: the 8.7 failure was a false negative (signature present in the offline run logs). |
| 2026-09-29 | E0 added (planned): NodalMLP probe-decoder close-out as the last architecture check before E1; moved from a proposed v1 section, since v1 is closed. |
| 2026-09-29 | E0.1 done (59/59 launchers pass `bash -n` and `sbatch --test-only`; `main.py` loads in the launcher image). E0.2 submitted; bilinear stalled on a Ray actor-reuse error and was resubmitted. C6 (environment divergence, `~/.local` leak) and D1 (one job environment) added; C1 now installs through C6; E0.5 (graph-model expansion) planned. |
| 2026-09-30 | E0 renamed `nodal_models_benchmark` with NodalGNN folded in; results marked preliminary (probes at the null, far below the linear family). E0.2 done (spectral complete; bilinear seed 0 not re-run). E0.6 `reuse_actors` preflight added; E0.5 is now a gated NodalGNN pilot. D2 (tuning budget) added. |
| 2026-09-30 | E0.3 done (byte-identical re-render, C3 re-confirmed; importance on within-seed-centred scores). E0.6 done: keep `reuse_actors=True` with packing. C6.1 snapshot and pre-C6 baseline recorded. |
| 2026-09-30 | C6 done: root cause was a non-writable overlay (root-owned skeleton dirs; `--fakeroot` unusable without subuid), fixed by an offline ownership change; `~/.local` packages + PyG consolidated into the overlay; `env.sh` closes the leak. C1 closed. Backup archived. E0.5 NodalGNN pilot submitted (`18887602`). |
| 2026-09-30 | E0 MSE-only enforced: GNN and NodalMLP runs were already `mse`; the linear reference is now scraped MSE-only (the sc_type snapshot's winners were `demeaned_mse` on 3 of 4 seeds); `run.py` rejects non-MSE runs. |
| 2026-09-30 | Status table synced (E0, C1, C6); E0 design records the MSE-only linear reference and the Chen vs NodalGNN difference, incl. NodalGNN's `r2t` input caveat; E0.5 files and 8 h budget. |
| 2026-09-30 | **E0 closed.** NodalGNN pilot stopped by decision (seed-0 best val 0.011 < null); results, takeaways and the "learn on top of the mean" follow-up recorded (§5); notebook retired. |
| 2026-09-30 | E2: both directions in scope; E2.0 direction audit recorded (generic vs needs config/loader vs reverse variants), reverse-variant requirement (FC graph threshold τ, default 0.5), `FC → SC` ceiling from literature, Krakencoder `FC → SC` file/key reference. README model table regrouped by model type × learning type. |
| 2026-10-01 | E2.1: retrainable Krakencoder (vendored upstream `b57e39c`, wrapper, `Krakencoder` model, launcher); loader serves both directions; local copy renamed `krakencoder_experimental/`; smoke check passed, parity run started. |
| 2026-10-01 | E2.1 restructured: vendored upstream, retrain wrapper and loader moved into one package `models/architectures/krakencoder/` (no root `third_party/`, no `scripts/krakencoder/`); launcher stays in `scripts/sbatch/Krakencoder/`. |
| 2026-10-01 | E2.1: near parity with the March fit (seed 0); epoch pilot at batch 41; `checkpoint_eval.py`; Krakencoder loss-grid instance (17 cells incl. paper default, packed runner) built and smoke-checked, not launched. |
| 2026-10-02 | E2.1 Krakencoder loss grid complete (21 cells × 5 seeds, 40.7 GPU-h): correye-driven, direction-specific trade-off; paper loss ≈ correye alone; neidist inert. |
| 2026-09-30 | E1 prereqs done: slug → `composite_loss/linear_backbone`; E1.1 (fixed scales + monitor-only terms) built and verified on local branch `e1-loss-scale-monitor`, merges when E0 closes; Stage 1 config and packed launcher added; E1.3/E1.4 runner design and dynamics figures specified; D3 (compute envelope, autonomous execution). |
| 2026-09-30 | E1.1 merged (`75b7c11`) after E0 closed; regression 45/45 and E1.1 checks 23/23 on `main`; checks tracked as `scripts/sbatch/checks/loss_regression.py` and `composite_loss/checks/check_e1_loss.py`. |
| 2026-09-30 | E1 becomes composite-loss protocol v1: shared versioned grid `composite_loss/grid.yml` (8-cell factorial at w = 0.5 + 8 dose points), batch 64 (D4), Stage 1 at 24 trials; replicability instance `composite_loss/pca_pls_learnable` added; E3 (magnitude tuning) outlined. |
| 2026-09-30 | E1 restructured: one experiment folder `scripts/experiments/composite_loss/` (protocol write-up, `grid.yml`, `checks/`, instances `linear_backbone/`, `pca_pls_learnable/`); Stage 1 resubmitted on the new paths: linear seeds 0–2 `18899811`, 3–4 `18899801`; learnable 0–4 `18899802`. The first linear submission (`18899142`, seeds 0–2) failed when the move ran before its search-space read (`FileNotFoundError`, ~11 GPU-min lost): jobs re-read the config after Ray start-up. |
| 2026-09-30 | E1.3–E1.5 code ready (`loss_grid.py`, `protocol.py`, `report.py`, launchers, `check_protocol.py`). C!4 (single runs ignored the tuned batch size) found; fix + `extra_callbacks` on branch `e1-callbacks-batchsize` (C7, merges after Stage 1). Trial-index parsing fixed in the `multimodel_scfc` audit script. |
| 2026-09-30 | C7 merged (`cfc1c32`, regression 45/45): batch-size fix (C!4), `extra_callbacks`, and **Ray sized to the SLURM CPU allocation** — five Stage 1 tasks had hung because Ray pre-started 128 workers (all node cores) that never registered; cancelled. E1 order: linear backbone first (seeds 1/2/4 resubmitted `18902227`), learnable paused at seeds 0–2 as the reproducibility target. |
| 2026-09-30 | E1 linear backbone runs autonomously as a SLURM `afterok` chain (Stage 1 → consensus (stage1 summary first) → grid → CPU report; `--kill-on-invalid-dep=yes`, so a D3 stop cancels the rest) with `scripts/sbatch/checks/watch_jobs.py` watching for Ray hangs / silent logs. Real-data Stage 1 check on seeds 0–3: best val 0.096–0.108 (≥ 0.09). |
| 2026-10-01 | E1 aligned to the user's plan: protocol v3 (grid v3, 29 combinations; D5 Demeaned corr-eye in mixtures; consensus re-check; Stage 2 cap 10 GPU-h in D3); instances E1.6 linear backbone and E1.7 PCA/PLS learnable done (replicate: effects correlate 0.92 / 0.98); E1.8 Krakencoder (= E2.1 loss grid), E1.9 CovProjector (all covariates), E1.10 cross-model HTML with the test-retest ceiling added. E2 narrowed to SC → FC with E2.2 (MSE-only) and E2.3 (composite-loss tuned; absorbs the former E3 outline). E3 redefined as FC → SC replication of E1 and E2 (the former E3 outline was never started; its content moved to E2.3). I1 added (HCP1200 timeseries merge, connectome-similarity views, behavioral FC → SC). |
| 2026-10-01 | E1.9 CovProjector (all covariates) done on a recorded fallback consensus (rerun with a hand-selected config open); E1.10 cross-model page built with the test-retest ceiling (demeaned r 0.49, avg_rank 0.987). |
| 2026-10-02 | E1.9 rerun on the E1.7 backbone + covariates (hand-selected, `fixed_consensus`), replacing the frozen-backbone fallback run. The 150-epoch run overfit (its avg_rank 0.890 was confounded); rerun at the validation-chosen 30 epochs: covariates raise demeaned r (0.110 MSE-only; 0.119 with Demeaned corr-eye 0.1) but lower identifiability vs E1.7. |
| 2026-10-02 | **E1 closed.** E1.8 Krakencoder done (via E2.1); E1.10 done with four instances + ceiling; E1 conclusions recorded and carried into E2.3; D6 added (fixed reference scales are the default composite-term balancing; resolves the backlog item). |
| 2026-10-02 | E3.0: bidirectional composite-loss layout `<model>/{sc2fc,fc2sc}`, direction-aware tooling (loss_grid, compare.py switch, protocol scaffold); E1 instances migrated to `sc2fc/` with identical re-render. |
| 2026-10-02 | E3 Phase D started: fc2sc scaffolds, FC → SC thresholds (×1.607), two chains submitted; CovProjector held; E2.2 ∥ Phase D rules. |
| 2026-10-02 | E2.2 specified: roster by class (latent pick `MaskedMLPPretrainer`, nonlinear variant, pilot-gated; `LatentAttnMasked`, `MaskedLatentPretrainer`, `CrossModalVAE` excluded), MSE-only protocol with budget rule and the audit's narrowed searches, one tagged campaign per model, reuse / rerun list, steps E2.2.1–E2.2.4. |
| 2026-10-02 | E2.2.1 built: benchmark configs in `models/configs/benchmark/mse/`, roster / submit / launcher / runner / checks in `scripts/experiments/model_benchmark/`; latent entry switched to the nonlinear `MaskedMLPPretrainer`. |
| 2026-10-02 | I2 added (repo organisation): I2.1 benchmark folder consolidation and campaign results level, I2.2 ConditionalGaussian search fix (both done); I2.3 config family folders + name lookup built on branch `i2-config-layout`, merge when no job runs; I2.4–I2.5 planned; I2.6 (Hydra) deferred. |
| 2026-10-02 | E2.2 flight plan recorded (infrastructure → validate both directions → compute budgets → full runs → report → E2.3). SC → FC pilot launched; latent gate passed. |
| 2026-10-02 | Spec audit: status table synced (E2, E3, I1, C2, C!4); closed items shrunk to outcomes (C1, C3, C6 + C6 plan, C7, E0, E1.10, E2.1); E2 head and E2.0 updated (Krakencoder both directions, FC → SC exercised); E2.2 campaign wording, trial budgets, gate result and steps updated; I1.1 done; backlog items covered by I2 / D6 removed. |
| 2026-10-02 | I1.4 done (task FC caches, bit-identical rest rebuild, damaged-source exclusions, ACL handling); I1.5 built on branch `fc-conditions` (condition loaders, partition-index fix, condition EDA views + notebook; checks pass), awaiting user review. |
| 2026-10-02 | E3 Phase D done: E1 protocol FC → SC for linear_backbone, pca_pls_learnable, pca_pls_covprojector (12.0 GPU-h); trade-off does not carry over (MSE-only best); CovProjector anatomy caveat; search-edge flags to E2.3. |
| 2026-10-04 | I1.5 merged into `main` (`5f5d238`); validation re-run on `main` passes on both parcellations; worktree removed. |
| 2026-10-04 | E2.2 done, both directions (111.5 GPU-h): linear family leads SC → FC (tied), graph / nodal at the null; FC → SC easier, covariate model's lead anatomy-driven; latent gate failed FC → SC. E2, E3 done; C2, C!1, C!4 resolved. Write-up `model_benchmark/model_benchmark.md`. |
| 2026-10-04 | **v2 closed.** Open items carried to [`spec_doc_v3.md`](spec_doc_v3.md) (Closure table); I1 and I2 closed with their open parts carried; backlog moved. |

Last updated at: 2026-10-04 EDT
