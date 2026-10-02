# Repo Spec v2 — Experiments

**Purpose:** run structured experiments on the v1 composite-loss / regularization machinery, clear the items carried over
from v1, and hold the repo backlog.
**Status:** active · **Started:** 2026-09-29 · **Predecessor:** [`spec_doc_v1.md`](spec_doc_v1.md) (closed 2026-09-29) ·
**Format:** [`spec_conventions.md`](spec_conventions.md)

**Contents:** [Status](#status) · [1. Purpose](#1-purpose) · [2. Conventions](#2-conventions) ·
[3. Carried over from v1](#3-carried-over-from-v1) · [4. Experiments](#4-experiments) · [5. Infrastructure](#5-infrastructure) ·
[6. Backlog](#6-backlog-not-scheduled) · [7. Change log](#7-change-log)

To add an experiment, append a section under §4 using the template in §4.0 and add a row to the status table.

## Status

| ID | Title | Status | Depends on | Owner |
|---|---|---|---|---|
| E0 | Nodal models benchmark and architecture check (`nodal_models_benchmark`) | closed 2026-09-30 | — | agent:infra |
| E1 | Composite-loss dynamics and trade-off across models, SC → FC (`composite_loss`, grid v3) | done 2026-10-02 (E1.6–E1.10; conclusions → E2.3) | D3, D4, D5, D6 | agent:modeling |
| E2 | Cross-model benchmark (`model_benchmark`) | in progress (E2.0, E2.1 done; E2.2 built, SC → FC pilot running; E2.3 outline) | C2 | agent:modeling (E2.2) |
| E3 | Replicate E1 and E2 for FC → SC | in progress (E3.0 done; Phase D FC → SC runs) | E1, E2, E2.0 | agent:infra |
| I1 | HCP1200 timeseries and connectome-similarity views | in progress (I1.1 done; I1.2–I1.3 planned) | — | — |
| I2 | Repo organisation: experiment folders, config layout, launchers | in progress (I2.1–I2.2 done; I2.3 on branch `i2-config-layout`, merges when no job runs) | — | agent:modeling |
| C1 | `torch_geometric` missing from the `kraken_env` overlay (`Chen2024GCN` / `NodalGNN` could not import) | — | **Done 2026-09-30** via C6 (`torch_geometric 2.8.0.post1` in the overlay; `dev_runs` `18880103`). |
| C2 | Re-tune the M5b-affected sweeps | in progress (E2.2 reruns) | E2.2 | agent:modeling |
| C3 | Confirm `loss_signature` in Tune-trial W&B configs | done | — | agent:infra |
| C4 | `latent_masked_test` notebook fixes | planned | — | — |
| C6 | Environment: two `kraken_env` stacks; jobs import from `~/.local` | done 2026-09-30 (C6.1–C6.4); C6.5 open; archive awaiting deletion | C6.5: user | agent:infra |
| C7 | Merge the batch-size fix + `extra_callbacks` hook + Ray CPU cap | done 2026-09-30 (`cfc1c32`) | — | agent:modeling |
| C!1 | v1:M5b — sampled L1/L2 not applied in past sweeps | open | resolved by C2 | — |
| C!2 | v1:M1b — `ema` runs with `neidist` ≤ 0 during warmup | open | — | — |
| C!3 | `CrossModal_linear_backbone` z-scored latents are PCA-space | open | — | — |
| C!4 | Single runs trained at batch 128 regardless of the tuned `batch_size` | open | fixed by C7; re-runs in E2.2 | — |
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
| C1 | `torch_geometric` missing from the `kraken_env` overlay (torch 2.9.0); `Chen2024GCN` / `NodalGNN` cannot import, and their launchers fail. It was importable when those models ran (Mar/Apr 2026; both import it unconditionally) and has since disappeared; the overlay never had it. | E0.5, E2 (those two models) | **Done 2026-09-30** via C6: `torch_geometric 2.8.0.post1` (+ `xxhash`) in the overlay; post-C6 `dev_runs` `18880103` trains `Chen2024GCN` and `NodalGNN`. |
| C2 | Re-tune the sweeps affected by v1:M5b (`CrossModal_PCA_PLS_learnable`, `CrossModal_PCA_PLS_CovProjector`, `Sarwar2020MLP`) | E2 | Re-tune within E2; resolves C!1. |
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

#### E1 conclusions (closed 2026-10-02; carried into E2.3)
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

### E2 — Cross-model benchmark   (slug: `model_benchmark`) · status: in progress · owner: agent:modeling (E2.2)

- **Question:** on equal footing, how are test Pearson r, demeaned r, avg_rank and top-1 distributed across model
  types, first MSE-only (E2.2), then with composite losses tuned where possible (E2.3)? SC → FC here; FC → SC is E3.
- **Design:** each model tuned per seed (Optuna or full grid over its benchmark config, selection on val demeaned r,
  best-trial report), seeds 0–4, one campaign per model; models grouped by type as in the README "Models" table.
- **Depends on:** C2 / C!1 and C!4 (resolved by the E2.2 reruns); E1 for E2.3.
- **Open decision:** sources beyond SC (`SC_r2t`, `SC+SC_r2t`).

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

**Reverse variants (`FC → SC`) for the graph / nodal models** (requirement, not built):
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

#### E2.2 — MSE-only benchmark, SC → FC · in progress (build) · owner: agent:modeling
**Question:** with every model tuned on the same splits under MSE only, how are test demeaned r, avg_rank and top-1
distributed across model types? E2.2 picks the models that go on to E2.3 (composite loss, tuned weights). The best of
those are then reoptimized for the final benchmark, and E3 repeats it for FC → SC.

**Roster** (from the 2026-09-30 audit, updated 2026-10-02):

| Class | Model | Entry |
|---|---|---|
| Null / ceiling | `CrossModalPCA` (null); `TestRetestPrecomputed` (ceiling) | full grid / reuse E1.10 |
| Linear, closed-form | `CrossModal_PLS_SVD`, `CrossModal_PCA_PLS`, `CrossModal_ConditionalGaussian` | full grid or budget rule |
| Linear, learned | `CrossModal_PCA_PLS_learnable`, `CrossModal_linear_backbone`, `CrossModal_PCA_PLS_CovProjector` (all covariates) | budget rule |
| Deep-learning baselines | `Sarwar2020MLP`, `Chen2024GCN` (narrowed searches); `Krakencoder` | narrowed / reuse E2.1 |
| Pairwise nodal | `NodalMLP`, `NodalGNN` (at the null in E0; narrowed reruns, so every row is from one campaign) | narrowed |
| Latent / pretrained | `MaskedMLPPretrainer`, **nonlinear** variant (PReLU MLP encoder, GELU MLP readout) | **pilot first (D2)** |
| Excluded | `LatentAttnMasked` (never tuned; attention adds nothing over its linear backbone in the dev runs; C!2), `MaskedLatentPretrainer` (test 0.068, below the linear family), `CrossModalVAE` (in development) | — |

- **Latent pick.** `MaskedMLPPretrainer`, the **nonlinear** variant (user 2026-10-02: the entry must have some
  nonlinearity). It is the only tuned latent model with held-out results.
  - **Linear variant** (fully linear, low-rank): seed 0, test 0.089 / 0.692.
  - **Nonlinear variant:** tuned, val 0.107 and test 0.081 / 0.691; it overfits (train demeaned r 0.30). Its mask
    grid (k = 128) reached val 0.113 at SC mask 0 / FC mask 0.05, which equals the linear variant's val (0.115).
  - Evidence in `results/logs/tune_model_parallel_maskedmlp_*` (2026-04-27).
  - **Narrowed search** (from `MaskedMLPPretrainer_nonlinear.yml`):
    - fixed: k = 128, `nonlinear: true`, `readout_type: mlp`;
    - searched: SC mask {0, 0.1, 0.2}, FC mask {0.05, 0.1, 0.2}, hidden {128, 256}, dropout and `l2_reg` (against
      overfitting), `lr`, epochs.
  - **Objective:** its native masked latent reconstruction loss (`latent_mse`), labelled as such, as Krakencoder
    enters with its own fixed losses.
  - **Gate (D2):** seeds 0–1, 12 trials; enters if its mean val ≥ the MSE-only `_learnable` mean val on the same seeds
    − 0.01. **Passed 2026-10-02:** 0.088 vs 0.096 (test 0.088 / 0.109 vs `_learnable` 0.071 / 0.092).

**Protocol:**
- Seeds 0–4, SC → FC, selection on val demeaned r, test metrics reported.
- **MSE only:** loss-weight and EMA keys removed from every search; Sarwar's correlation term off.
- **Each model keeps its own batch size and training budget.** D4's batch 64 is a composite-loss rule.
- **Trials:** `clamp(8 × free keys, 16, 64)` with ASHA (`_learnable` 64, `linear_backbone` 48, `ConditionalGaussian` 32
  with `fit_domain: pca`, I2.2); `CrossModalPCA`, `PLS_SVD`, `PCA_PLS` take their full grid.
- **Narrowed searches** (audit, 3,592 past trials: extra trials bought less than seed noise):
  - **Chen:** identity nodes, 2 layers, 500 epochs; `conv_dim` {128, 256}, `dnn_dim` {32, 64}, `lr`, `l2_reg`; 12 trials.
  - **NodalGNN:** 2 layers, decoder 32, 500 epochs; `hidden_dim` {32, 96}, `lr`, `l2_reg`; 10 trials.
  - **Sarwar:** leaky_relu, 300 epochs, plain MSE; layers {3, 5}, hidden {512, 1024}, dropout, `lr`, `l2_reg`; 16 trials.
  - **NodalMLP:** about 3 keys from E0's importance table; about 12 trials.
- **One campaign per model:** results come only from this campaign's task logs (job names
  `e2_mse_<Model>_<direction>`), never the best over older sweeps (which carry C!1 / C!4 and differ in budget).

**What exists and what reruns:**
- **Reuse:** Krakencoder (E2.1 `mse_only` and paper default, seeds 0–4) and the test-retest ceiling (E1.10).
- **Rerun everything else.**
  - `_learnable`, CovProjector, Sarwar, Chen and NodalGNN: their March 2026 benchmark rows carry C!1 / C!4, and
    Sarwar's used its correlation loss.
  - The closed-form March rows were selected best-over-sweeps; they are cheap to redo.
  - `ConditionalGaussian` has never been benchmarked.
  - E1's MSE-only fits are consensus configs, not per-seed tuning.
- **This resolves C2** (re-tune the M5b-affected sweeps) and the C!4 re-runs for these models.

**Flight plan (pre-E3 MSE benchmark, both directions; user 2026-10-02):**

| Stage | What | Gate / output |
|---|---|---|
| 0. Infrastructure | Merge I2.3 (config family folders) in a no-jobs window; I2.4 generic launcher (new files) | regression 45/45 + benchmark checks on `main`; benchmark configs regenerate identically |
| 1. Validate models | SC → FC pilot (1 seed per model + latent gate on seeds 0–1); FC → SC pilot (direction-valid roster: Sarwar `_fc2sc`, graph / nodal excluded until reverse variants exist; Krakencoder reused) | every model finishes with a parsed best-trial summary; errored trials explained or fixed; latent gate rechecked for FC → SC |
| 2. Compute budgets | per model × direction wall time and trials per hour from the pilot logs → SLURM time and packing in `config.yml`; total GPU-h | budget table approved by the user |
| 3. Full runs | seeds 0–4, full budgets, `mse/sc2fc` and `mse/fc2sc` campaigns under the shared GPU cap; watcher on; failed seeds resubmitted | 5 seeds per roster model per direction |
| 4. Report | `run.py` per direction → records, tables, the four metric bar charts; write-up + spec results | top models per direction → E2.3 (composite, tuned weights) → final reoptimized benchmark |

The FC → SC half of stage 3 is E3's "replicate E2 for FC → SC" (roster entries only, no new code).

**Steps:**
- **E2.2.1 Build** · done 2026-10-02 (`da4ab95`). Configs `models/configs/benchmark/mse/` (generated by
  `model_benchmark/build_configs.py`; direction set by the launcher; `_fc2sc` only for Sarwar); roster `config.yml`,
  `submit.py`, `launch_model.sh`, `run.py` (logs → records → tables + per-metric bars grouped by model type), checks.
- **E2.2.2 Pilot** · SC → FC running (jobs `19061419`–`19061431`); latent gate passed; FC → SC pilot next.
- **E2.2.3 Full runs** and **E2.2.4 Report:** stages 3–4 of the flight plan.

- **Accept:**
  - every row is one campaign × seeds 0–4, verified MSE-only (or native objective, labelled) from run configs;
  - deterministic rendering from tracked records;
  - write-up `scripts/experiments/model_benchmark/model_benchmark.md`.
- **Budget:** about 10–20 GPU-h for the cheap set plus about 30–40 GPU-h for the narrowed expensive set (5 seeds),
  plus pilots. Each stage is approved before launch.

- **Parallel with E3 Phase D (user 2026-10-02):** E2.2 (agent:modeling) and E3 Phase D (agent:infra) run at the
  same time. Rules for both:
  - **`models/` is frozen while either has jobs queued or running** (jobs import it live): no edits to model code or to
    any `models/configs/*.yml` a queued job reads. Config changes go into **new** files (e.g. `Sarwar2020MLP_fc2sc.yml`).
  - **Separate folders:** E2.2 lives in its own experiment folder and does not touch `scripts/experiments/composite_loss/`
    or `scripts/results_utils/loss_grid.py`; Phase D only adds `composite_loss/<model>/fc2sc/` and `ceiling/` files.
  - **Spec:** each edits only its own section (E2.2 / E3) plus its change-log row.
  - **GPUs:** both share the per-user QOS cap; Phase D uses up to ~10 concurrent GPUs while its Stage 1 and grids run
    (two chains), E2.2 the rest. Pack small models (D2) on both sides.
  - **Direction-aware from the start:** build the E2.2 runner on the `sc2fc` / `fc2sc` folder convention (E3.0), so
    E3's FC → SC replication of E2 is configuration, not new code.

#### E2.3 — Composite-loss benchmark (tuned weights) · outline
Absorbs the former E3 outline (composite-loss magnitude tuning, never started).
- **Design:** the E2 roster where the model trains at batch 64 (D4). Composite weights are tuned per model: Optuna over
  the scaled weights (`loss_weight_*` with each model's fixed scales, measured as in E1.2) jointly with `lr` /
  `l2_reg`. Seeds 0–4.
- **E1 informs:** the term set (Var-match, Demeaned corr-eye, Neighbor dist; raw Corr-eye excluded, D5), the weight
  ranges (value-matched weights are not gradient-matched; E1 gradient-strength table), and the selection metric. The
  trade-off makes the choice of selection metric (demeaned r vs avg_rank, or a combination) a decision to make here.
- **Depends on:** E1, E2.2.

### E3 — FC → SC: replicate E1 and E2   (slug: tbd) · status: outline · owner: —

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
  - **Phase D (FC → SC runs; approved 2026-10-02, autonomous within D3 per instance):**
    - Scaffolded `linear_backbone/fc2sc`, `pca_pls_learnable/fc2sc`, `pca_pls_covprojector/fc2sc`.
    - Calibration (`checks/direction_baselines.json`, val demeaned r, seeds 0–4): null 0.009 (SC → FC) / 0.008
      (FC → SC); closed-form `CrossModal_PCA_PLS` 0.089 / 0.143. Stage 1 stop thresholds transferred by that ratio
      (1.607), rounded down: 0.14 (linear_backbone, was 0.09), 0.13 (PCA/PLS models, was 0.085).
    - Chains submitted (Stage 1 → consensus → grid → report, `afterok`): linear_backbone `19053917`→`19053918`→
      `19053919`→`19053920`; pca_pls_learnable `19053921`→`19053922`→`19053923`→`19053924`.
    - `pca_pls_covprojector/fc2sc` held: its SC → FC config was hand-selected (E1.7 backbone + covariates, 30 epochs
      from the covariate val curves); the FC → SC backbone comes from `pca_pls_learnable/fc2sc`'s consensus, then the
      epoch cap needs the same check — proposed to the user when that consensus exists.
    - FC → SC ceiling: literature value as `ceiling/<name>.json` with `direction: FC->SC` (source to pick).
    - Parallel with E2.2: rules in E2.2.
- **Depends on:** E1, E2, E2.0.

## 5. Infrastructure

### I1 — HCP1200 timeseries and connectome-similarity views   · status: in progress · owner: —

- **I1.1 — Move the HCP1200 timeseries into the data folders** · done 2026-10-01 (agent:infra). Add-only merge into
  `HCP1200/HCP1200_fMRI/xcpd-0-9-1` (job `18978259`, verify passed: 569,808 files moved, 0 errors; never overwrote
  existing files); staging cleaned, the `.tar` backup kept. Procedure: `context_packages/HCP1200_xcpd_transfer_merge_handoff.md`.
- **I1.2 — Subject × subject connectome-similarity matrices** · planned · owner: —
  - A correlation matrix over all subjects' connectome comparisons (diagonal = same subject), in raw and **demeaned**
    form (training-set mean subtracted, as in Demeaned corr-eye).
  - Interactive per-subject matrix views of the full and the demeaned connectome.
  - Self-contained HTML per the figure convention (§2).
- **I1.3 — Which behavioral FC best predicts SC** · planned · owner: —
  - With the best E2 model, test which behavioral FC (from the I1.1 timeseries) best predicts SC.
  - This is the starting point for timeseries modeling, with room for spatial analyses.
  - **Depends on:** I1.1, E2, E3 (the FC → SC path).

### I2 — Repo organisation: experiment folders, config layout, launchers   · status: in progress · owner: agent:modeling

**Goal:** a layout that scales to more models, both directions and the E2.3 / final benchmarks without copied files.
Started from user review 2026-10-02.

- **I2.1 — One cross-model benchmark folder** · done (`43c9174`, `4e93dd7`).
  - `multimodel_scfc/audit/` moved into `model_benchmark/audit/`; `multimodel_scfc/` removed.
  - Results land in `model_benchmark/<campaign>/<direction>/` (`mse/` for E2.2; E2.3 adds `composite/`; the final
    benchmark `final/`).
- **I2.2 — ConditionalGaussian search** · done (`43c9174`, `4e93dd7`). The benchmark config fixes `fit_domain: pca`; the model
  rejects `raw_edges` with shrinkage estimators, and 168 of the pilot's trials errored.
- **I2.3 — Config family folders** · built on branch `i2-config-layout` (`7501e94`, rebased on `main`; worktree `../Conn2Conn_wt_i2`).
  - **Layout:** `models/configs/{null_ceiling,linear,latent,graph_nodal,deep}/<Model>.yml`;
    `variants/<family>/<Model>_<variant>.yml`; `benchmark/<campaign>/` reached by path only.
  - **Lookup:** `models/registry.py` finds a config by name. Old flat paths (`models/configs/<name>.yml`, used by the
    legacy launchers and other experiments' configs) fall back to the new location with a notice.
  - **Model file:** `CrossModal_ConditionalGaussian` moves from `latent_attention/` to
    `models/architectures/crossmodal_conditional_gaussian.py` (linear family).
  - **Checks:** loss regression 45/45 bit-identical; E1.1 checks pass; benchmark configs regenerate identically.
  - **Merge rule:** merge only when no job is queued or running (it touches `models/`).
- **I2.4 — One tune launcher** · planned.
  - A generic `scripts/sbatch/launch_tune.sh` + roster `submit.py` (the `model_benchmark` pattern, generalised).
  - The ~60 per-model sbatch scripts move to `scripts/sbatch/legacy/`, after checking no other agent still submits
    them (Krakencoder's launcher stays).
- **I2.5 — Variants as overrides** · planned. Source and covariate variants (`_SC_r2t`, `_SC+SC_r2t`, `_demo`,
  `_fs_*`) become short `data:` / `model:` overrides in experiment rosters instead of near-copy files.
- **I2.6 — Hydra** · deferred (user 2026-10-02): too large a refactor for the benefit; I2.3–I2.5 cover the need.
- **Found while testing:** `composite_loss/checks/check_protocol.py` fails on `main`: it still reads
  `composite_loss/linear_backbone/config.yml`, moved to `<model>/sc2fc/` in E3.0 Phase C. Left to the E3 owner.

## 6. Backlog (not scheduled)

From v1 §6, v1 §8.6, the unrun parts of v1 M10, and E0/E1 follow-ups:
- **Graph / nodal models on top of the mean** (from E0): train GNNs on the subject's deviation from the train-split
  mean FC, or on PCA scores / low-rank factors, as the PCA family does, instead of on full FC edges; include an
  SC-only (`use_r2t: false`) `NodalGNN`. Pilot per D2 against `PCA_PLS_learnable` on the same seeds.
- **Editable install** (`pyproject.toml` + `pip install -e .`; the overlay now mounts `:rw` without `--fakeroot`, C6) and `conn2conn/` namespacing.
- **Artifact cleanup:** `results/ray_results/` (118 GB) and a `results/logs/` retention policy.
- **Shared constants** between `main.py` and `scripts/results_utils/records.py`.
- **Untracked reference code** in `context_packages/modeling/*_context/`.
- **Latent-space terms inside composite** (`latent_mse` / `latent_weighted_mse` as mixable terms).
- **EMA diagnostics:** `*_loss_ref_*` behavior after warmup, and batch-size sensitivity of `correye` / `neidist` (64 vs 128).

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

Last updated at: 2026-10-02 EDT
