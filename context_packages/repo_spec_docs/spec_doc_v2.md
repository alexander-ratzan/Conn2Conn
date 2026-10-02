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
| E2 | Cross-model benchmark (`model_benchmark`, working name) | in progress (E2.0, E2.1 done; E2.2 MSE-only benchmark: spec written, build in progress) | C2 | agent:modeling (E2.2) |
| E3 | Replicate E1 and E2 for FC → SC | outline (E3.0 bidirectional layout + tooling done 2026-10-02) | E1, E2, E2.0 | — |
| I1 | HCP1200 timeseries and connectome-similarity views | in progress (I1.1) | — | agent:infra (I1.1) |
| C1 | `torch_geometric` missing from `kraken_env` | done 2026-09-30 (via C6) | — | agent:infra |
| C2 | Re-tune the M5b-affected sweeps | planned (within E2) | E2 | — |
| C3 | Confirm `loss_signature` in Tune-trial W&B configs | done | — | agent:infra |
| C4 | `latent_masked_test` notebook fixes | planned | — | — |
| C6 | Environment: two `kraken_env` stacks; jobs import from `~/.local` | done 2026-09-30 (C6.1–C6.4); C6.5 open; archive awaiting deletion | C6.5: user | agent:infra |
| C7 | Merge the batch-size fix + `extra_callbacks` hook + Ray CPU cap | done 2026-09-30 (`cfc1c32`) | — | agent:modeling |
| C!1 | v1:M5b — sampled L1/L2 not applied in past sweeps | open | resolved by C2 | — |
| C!2 | v1:M1b — `ema` runs with `neidist` ≤ 0 during warmup | open | — | — |
| C!3 | `CrossModal_linear_backbone` z-scored latents are PCA-space | open | — | — |
| C!4 | Single runs trained at batch 128 regardless of the tuned `batch_size` | open | C7 (fix) + re-runs | — |
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
| Compute | Training and sweeps go through `sbatch` (array templates in `scripts/sbatch/`). Each compute stage is approved before submission. The local L40S node is used only for short checks, and only inside a compute allocation, never on a login node. |
| W&B | Runs tagged with the experiment slug; group by `loss_signature`. Staged-experiment runs carry the stage in their tags (`<slug>:stage1`). |

## 3. Carried over from v1

### Open work (`C`)

| ID | Item | Blocks | Next action |
|---|---|---|---|
| C1 | `torch_geometric` missing from the `kraken_env` overlay (torch 2.9.0); `Chen2024GCN` / `NodalGNN` cannot import, and their launchers fail. It was importable when those models ran (Mar/Apr 2026; both import it unconditionally) and has since disappeared; the overlay never had it. | E0.5, E2 (those two models) | **Done 2026-09-30** via C6: `torch_geometric 2.8.0.post1` (+ `xxhash`) in the overlay; post-C6 `dev_runs` `18880103` trains `Chen2024GCN` and `NodalGNN`. |
| C2 | Re-tune the sweeps affected by v1:M5b (`CrossModal_PCA_PLS_learnable`, `CrossModal_PCA_PLS_CovProjector`, `Sarwar2020MLP`) | E2 | Re-tune within E2; resolves C!1. |
| C3 | Confirm Tune-trial W&B configs carry `loss_signature` (v1 8.7 check failed) | — | **Done 2026-09-29.** The failure was a false negative: wandb 0.25 offline runs write no `files/config.yaml`; the config is inside `run-*.wandb`. Both trials of the 8.7 tune run (`results/ray_checkpoints/CrossModal_linear_backbone_tune_1790207949/*/wandb/offline-run-*/run-*.wandb`) contain `loss_signature` = `mse+0.25*correye+0.5*neidist`. The check in `scripts/sbatch/checks/verify_modeling_track.py` should read `run-*.wandb` (or an online run) instead. E1.2's online sweep can re-confirm at no cost. |
| C4 | `scripts/notebooks/model_testing/latent_masked_test.ipynb`: cell 6 reads `residual_linear.weight` (absent in `attention_only`); cell 4 sets `l2_reg` twice | — | Fix when that notebook is next used. |
| C6 | **Environment divergence** (found 2026-09-29). (1) Two stacks share the overlay: `source /ext3/env.sh` (all 59 launchers) gives torch 2.9.0+cu128 from the overlay **plus `~/.local`**; `source /scratch/asr655/envs/activate_env.sh kraken_env` gives torch 2.11.0+cu130 from `/scratch/asr655/envs/kraken_env/pylibs` (5.1 GB, installed 2026-04-08) and hides `~/.local`. Interactive sessions and jobs can therefore run different torch / Ray / Lightning. (2) Jobs import ray 2.54.1, lightning 2.6.1, wandb 0.25.1, optuna 4.8.0, torchmetrics, pyarrow from `~/.local` (38 packages, 520 MB, 14k files, installed 2026-04-08). Home is at 25.4k / 30k files; `~/.local` is the only large home dir not symlinked to `/scratch`. **Root cause (confirmed 2026-09-30):** the overlay could not be written on this cluster. The image's skeleton dirs (`/upper`, `/upper/ext3`, `/work`) were owned by uid 0 while its contents are user-owned, so a plain `:rw` mount fails; the user has no `/etc/subuid` entry, so `--fakeroot` falls back to a root-mapped namespace that cannot write either. pip then says site-packages "is not writeable" and silently installs into `~/.local`, which `/ext3/env.sh` leaves on every job's `sys.path`. | E0.5, E2, every run | C6.1–C6.4 done 2026-09-30; C6.5 (user); archive cleanup pending. See **C6 plan** below. |
| C7 | Branch `e1-callbacks-batchsize` (`d4368d7`, worktree `../Conn2Conn_wt_e1b`): `Sim._build_runtime_data_context` rebuilds the loaders at the run's `trainer.batch_size` when it differs from `Sim`'s (fixes C!4); `train_model` / `_run_learned_single` gain `extra_callbacks` (E1.4 gradient cosines). Checked: loss regression 45/45 vs `main`; loaders rebuilt only on a batch mismatch; callbacks reach the Trainer. | E1.4 gradient cosines; C!4 | Merge once no queued or running job imports `main.py` / `models/` (E1 Stage 1 done); rerun `loss_regression.py`; delete worktree and branch. |

### Caveats (`C!`)

| ID | Caveat | Affected artifacts | Resolved by |
|---|---|---|---|
| C!1 | v1:M5b — every sweep before 2026-09-23 of `CrossModal_PCA_PLS_learnable`, `CrossModal_PCA_PLS_CovProjector` and `Sarwar2020MLP` trained with the YAML default regularization (L2 = 1e-4 for the PCA/PLS models, none for Sarwar); W&B logged the sampled `l1_reg`/`l2_reg`, which were not applied. Results are valid as default-regularization results. | `sc_type_benchmark` (`PCA_PLS_learnable` rows); `cov_projector_benchmark` (`PCA_PLS_learnable`, all projector rows, `Sarwar2020MLP`) | C2 |
| C!2 | v1:M1b — any `ema` run whose `neidist` reached ≤ 0 during warmup had that term inflated ~10⁸-fold (includes the old `LatentAttnMasked` default composite). Which past runs were hit is not determined. | past `ema` composite runs, mainly `LatentAttnMasked` | re-run or audit if those results are reused |
| C!3 | `CrossModal_linear_backbone(zscore_pca_scores=True)` returns PCA-space latents from `predict_target_latents` (`LatentAttnMasked` returned z-space). Edge outputs are unchanged. | latent losses and latent diagnostics under z-scoring | informational; stays open while z-scored latents are in use |
| C!4 | `main()` builds `Sim` without a `batch_size` (so 128), and `_run_learned_single` reused those loaders unless covariate sources changed. Every `--report_best_after_tune` rerun and direct prod run therefore trained at **batch 128 whatever the config or best trial specified**; Tune trials themselves used the right batch size. Found 2026-09-30 by code reading. | best-trial test metrics of models tuned at batch ≠ 128: E0 `nodal_models_benchmark` (NodalMLP 8–64, NodalGNN 8), `sc_type_benchmark` / `cov_projector_benchmark` rows for `CrossModal_PCA_PLS_learnable` (64), the projector (64), `Sarwar2020MLP` (32), `Chen2024GCN` (4); E1 Stage 1 best-trial reports (not used for E1 results: E1.3/E1.4 build `Sim(batch_size=64)` explicitly) | C7 (fix), then re-runs within E2 / C2 |

#### C6 plan — one job environment in the existing overlay (no replicate overlay)

- **Preconditions:** E0.2 finished; no queued or running job and no Jupyter / OOD session has
  `overlay-15GB-500K.ext3` mounted (an `:rw` mount needs it exclusively; the file's mtime changed on 2026-09-29, so
  something mounted it writable that day); the other agent idle.
- **C6.1 Snapshot** · done 2026-09-30: `pip freeze` of the launcher stack (overlay + `~/.local`) saved as the reference "job version";
  `~/.local` file list + checksums; temporary backup `cp --sparse=always` of the overlay (deleted after C6.4).
  Snapshot and backup (`overlay-15GB-500K.ext3.c6_backup`, `cmp`-verified) taken 2026-09-30:
  `/scratch/asr655/envs/kraken_env/c6_snapshot_2026-09-30/` — launcher
  freeze 153 = overlay 115 + user-site 38; `~/.local` md5s of 14,134 files; original `env.sh`. Pre-C6 baseline
  `verify_modeling_track` dev_runs (`18871223`): 9/9 models pass (matching the 2026-09-23 report within GPU noise),
  Chen2024GCN and NodalGNN skipped (no `torch_geometric`).
- **C6.2 Consolidate into the overlay** · done 2026-09-30. `--fakeroot` could not write (see C6 root cause), so first
  `c6_fix_ownership.sh` gave the three skeleton dirs to the user (offline `debugfs` edit, 6 inode fields; `e2fsck`
  clean); the overlay then mounts `:rw` without `--fakeroot`. `c6_install.sh` (plain `:rw`, launchers' image
  `cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif`), rehearsed end-to-end on a throwaway copy first: install the exact `~/.local` versions (`--no-deps`, from the snapshot) and
  `torch_geometric==2.8.0.post1` into `/ext3/miniforge3` site-packages, existing overlay packages pinned by a constraints
  file. Guards: `PYTHONNOUSERSITE=1`, `PIP_USER=0`, `PIP_CACHE_DIR` on `/scratch`, `unset PYTHONPATH`. Then `pip check`.
  Result: 38 packages + `torch_geometric 2.8.0.post1` + `xxhash 4.0.1` (the only new dependency); `pip check` clean.
  Leftovers in `/ext3`: three empty root-owned vim swap files from 2026-04-06 (`.env.sh.swp`, `.env.sh.swx`,
  `env_sh.swp`, an earlier failed edit of `env.sh`); harmless, removable with `debugfs -w -R "rm …"`.
- **C6.3 Close the leak** · done 2026-09-30: append to the overlay's `/ext3/env.sh`: `export PYTHONNOUSERSITE=1` and `export PIP_USER=0`
  (all launchers pick it up; none edited). Also drop its stray `PYTHONPATH=<bin dirs>` line. Optional safety net for
  every environment: `~/.config/pip/pip.conf` with `[install] user = false`.
- **C6.4 Verify** · done 2026-09-30: the launcher stack's `pip freeze` equals the C6.1 snapshot plus exactly the PyG packages; `~/.local`
  absent from `sys.path`; `~/.local` file list unchanged; `main.py --help`; `verify_modeling_track` `dev_runs` trains
  every model family including `Chen2024GCN` / `NodalGNN` (closes C1). Then delete the backup copy.
  2026-09-30: freeze = snapshot + `torch-geometric`, `xxhash`; user site disabled; every package loads from
  `/ext3/miniforge3`; pip's user-install fallback gone; `~/.local/lib` and `~/.local/bin` byte-identical (only
  Claude / Cursor state under `~/.local/state` changed); PyG `GCNConv` imports. Post-C6 `dev_runs` (`18880103`): 11/11 pass — the 9 baseline models match `18871223`
  within GPU noise, `Chen2024GCN` and `NodalGNN` now train (closes C1). The backup was **moved, not deleted**, to
  `/scratch/asr655/envs/archive/2026-09-30_c6/` (with the Miniforge installer and a stray log; see its `MANIFEST.md`);
  delete that folder once post-C6 jobs have run clean for a while.
- **C6.5 Align interactive use (user):** point `activate_env.sh` at the same setup as `/ext3/env.sh`, or retire `pylibs`;
  give Jupyter kernel specs (`~/.local/share/jupyter/kernels`) the same two variables.
- **Not in scope yet (user, 2026-09-30: hold `~/.local` as is):** removing packages from `~/.local`. `vformer_env` / `main_env` may still import from it; check
  those projects first.

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

Last check of the v1 architecture (moved launchers, results tooling, runner pattern) on a small real experiment before
E1. Closes out the never-run `scripts/notebooks/results_scrape/nodal_decoder_importance.ipynb`; NodalGNN folded in
2026-09-30 (it was E0.5 "expansion").

- **Question:** how much of SC→FC can per-region (nodal) models recover: NodalMLP with simple probe edge decoders
  (`dot`, `bilinear`, `linear_beta`, the connectome-harmonic `spectral` encoder) and NodalGNN, against the null, the
  linear family and `Chen2024GCN`? Which hyperparameters matter? Recorded as closed for a later SMT revisit in GeneEx2Conn.
- **Design:** `SC → FC`, MSE-only loss, seeds 0–3, selection on `val_demeaned_r`, one best-trial run per condition × seed.
  NodalMLP variants: tune trials created ≥ 2026-04-27 whose config matches the variant YAML's options (`reg` treated as
  `l2_reg`, equal per v1 8.7). NodalMLP with the MLP decoder (pre-schema `NodalMLP.yml` runs) as the flexible-decoder
  control. Linear reference scraped here as the best **MSE-only** `CrossModal_PCA_PLS_learnable` SC run per seed (the
  `sc_type_benchmark` snapshot picks across losses: `demeaned_mse` wins 3 of 4 seeds). From the tracked snapshots: null
  `CrossModalPCA` SC (closed-form), `Chen2024GCN` (tuned) and untuned `NodalGNN` (both `loss_type: mse`). `run.py`
  refuses to render if any learned run or importance trial is not plain MSE. Tuning budget follows D2.
- **The two GNNs:** both pass messages over the subject's SC and decode each edge from its two node embeddings. They
  differ in node input: `Chen2024GCN` starts from one-hot region identity (so its first layer is a linear map of each
  region's SC row plus a learned per-region table), while `NodalGNN` starts from per-subject anatomy (volume, centroid,
  the region's tract profile `r2t`) with a node MLP, residual + LayerNorm GCN layers, a richer edge decoder
  (`[h_i, h_j, |h_i−h_j|, h_i·h_j]`) and dropout / edge dropout / ridge on all weights. **Input caveat:** `r2t` carries
  `SC_r2t` information, so default `NodalGNN` is not input-matched to the SC-only rows; an SC-only ablation
  (`use_r2t: false`, existing `tune_array_nodalgnn_ablation_SC_seeds.sh`) only if the E0.5 gate passes.
- **Result (closed 2026-09-30; write-up `nodal_models_benchmark.md`):** every nodal model is at the null on SC→FC
  (test demeaned r: probes 0.011–0.018, NodalMLP MLP decoder / NodalGNN default / Chen 0.016–0.023; null 0.013 / avg
  rank 0.518; MSE-only `PCA_PLS_learnable` 0.078 / 0.686 on the same seeds; all rows verified MSE-only per run). Tuning
  does not help: val scores are set by the split, within-seed importance is near-uniform, and the NodalGNN pilot peaked
  at seed-0 val 0.011 (null 0.025). Train demeaned r is also ≈ 0: the full-edge MSE formulation lets the population
  mean absorb the loss. NodalMLP-style models are closed; the repo is confirmed working after the v1 refactor.
  Follow-up (learn on top of the mean, PCA-like) is in §5.
- **Steps:**
  - **E0.1 — Launcher and entrypoint checks** · done 2026-09-29. `bash -n` 59/59; `sbatch --test-only` 59/59 (no jobs
    queued); `main.py --help` exits 0 in the launcher image (torch 2.9.0+cu128, ray 2.54.1, lightning 2.6.1, wandb
    0.25.1, optuna 4.8.0; `torch_geometric` absent, C1).
  - **E0.2 — Fill the two missing seeds** · done 2026-09-29 (partial). Spectral seed 3 (`18814360`, `04ae6ee`): 32 trials,
    best-trial report, W&B run — the full launcher cycle works from `scripts/sbatch/`. Bilinear seed 0: `18814359`
    stalled after a Ray actor-reuse error (`Trainable runner reuse requires reset_config()`); resubmitted as
    `18816010`, which ran 26/32 trials without that error and was then killed by a signal, most likely the cluster's
    GPU-underutilization policy (1 small trial per whole GPU). Not re-run (D2).
  - **E0.3 — Experiment build** · done 2026-09-30. `scripts/experiments/nodal_models_benchmark/`: `config.yml`, `run.py`,
    tracked `records.json` (best trials) + `trials.json` (tune trials); tables `trial_summary`, `importance`
    (fANOVA, seeded), `test_summary`, `row_seed_metrics`, `seed_records`; PNGs: test bars, metric panels, importance
    heatmap, trial-score distributions (new `FIGURE_TYPES`: `importance_heatmap`, `trial_distribution`).
    `optuna_importance` gains `rows_to_frozen_trials` (cached rows, key aliases) and a `seed` for `get_importance`.
    Accept: deterministic rendering from the snapshots; C3 re-confirmed from online trial configs (`loss_signature`).
    Done 2026-09-30: 23 scraped best-trial records (19 NodalMLP + 4 MSE-only linear) + 12 snapshot references,
    554 tune trials; cache-mode re-render byte-identical; loss verified per run from W&B configs — every learned row
    and all 554 importance trials are plain MSE.
  - **E0.6 — `reuse_actors` preflight** · done 2026-09-30. `main.py` gains `--tune_reuse_actors {true,false}` (default true;
    `4da4cd3`). `preflight/preflight_reuse_actors.sh`: a 12-trial NodalMLP bilinear tune (max_epochs 40), packed
    4 trials per GPU, with reuse on and off, W&B offline; logs trial errors, `HCP_Base` builds and one timed build,
    trial/job wall time, GPU utilization. Decision rule: keep reuse on if it runs packed without the reuse error;
    otherwise weigh the per-trial rebuild cost of reuse off before changing the default.
    **Result (`18871767`):** reuse on — 12/12 trials, 4 `HCP_Base` builds (one per packed actor), no reuse error,
    509 s, GPU 91–100% once training starts; reuse off — 12/12 trials, 12 builds, 833 s (+64%; different node, one
    build 43 s vs 22 s), idle GPU between trial waves. **Keep `reuse_actors=True` with packing** (the earlier reuse
    error was intermittent and did not recur packed). Four idle minutes at start-up (Ray start + per-actor builds) are
    the remaining underutilization risk for very short tunes.
  - **E0.5 — NodalGNN pilot** · stopped 2026-09-30 by decision (gate clearly not reachable). `18887602`: 10 of 12
    trials in 2 h 49 min, best seed-0 val demeaned r 0.011 (< null 0.025 < gate 0.045); per-trial results in
    `pilot/pilot_trials.csv`. Seeds 1–3 and the SC-only ablation not run. Setup: `nodal_models_benchmark/pilot/`:
    `NodalGNN_mse_pilot.yml` (`NodalGNN.yml` without the loss-weight / loss-scale searches, so MSE-only; `max_epochs`
    ≤ 750) and `tune_nodalgnn_pilot.sh` (array index = seed); seed 0 × 12 trials, 4 packed per GPU, reuse on (E0.6). **Gate:** extend to seeds 1–3
    only if the seed-0 best val demeaned r ≥ 0.045 (the null's seed-0 val 0.025 + 0.02 — val is only comparable
    within a seed); otherwise record NodalGNN as a negative result. Budget: 1 job, 8 h limit (old 750-epoch default runs took ~47 min
    each on a whole GPU; 3 packed waves expected ~3–4 h); +3 jobs (`--array=1-3`) if gated in.
  - **E0.4 — Close-out** · done 2026-09-30. Write-up closed with findings, takeaways and the follow-up; index row
    closed; `scripts/notebooks/results_scrape/nodal_decoder_importance.ipynb` removed (superseded; in git history).
    NodalGNN pilot rows not added to the tables (no best-trial report; the pilot is reported from its trial logs).
- **Depends on:** —

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

#### E1.10 — Cross-model interactive comparison · done 2026-10-02 (four instances + ceiling) · owner: agent:modeling
- **Changes:** one self-contained HTML tool in `composite_loss/` built from every instance's `tables/`. It shows test
  demeaned r vs avg_rank on **fixed axes** shared by all models, with per-model toggles and the same hover / click
  panel as the instance plots.
- **Ceiling:** the test-retest point (`TestRetestPrecomputed`, SC → FC) is drawn as the reference ceiling for demeaned
  r and avg_rank.
- **Accept:** rebuilds deterministically from the instance tables; adding an instance needs no code change.
- **Result:** `composite_loss/compare.py` → `figures/cross_model_interactive.html` + `tables/cross_model_summary.csv`,
  rebuilt by every instance report (`b11fcb3`, zoom toggle `0b6a768`). Ceiling `ceiling/test_retest.py` (CPU): test
  demeaned r 0.49, avg_rank 0.987, top-1 0.93 (195 subjects with both sessions per split), about 5× the best model's
  demeaned r.

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

### E2 — Cross-model benchmark, SC → FC   (slug: `model_benchmark`, working name) · status: outline · owner: —

- **Question:** on equal footing, how are test `demeaned_pearson`, `avg_rank` and the other standard metrics
  distributed across model types for all stable models, first MSE-only (E2.2) and then with composite losses tuned where
  possible (E2.3)? FC → SC is E3.
- **Roster:** every stable registered model, grouped by model type (null / ceiling, linear decomposition, latent /
  pretrained, pairwise nodal, deep-learning baseline, experimental) and learning type (closed-form, supervised,
  self-supervised, precomputed); the classification lives in the README "Models" table. `CrossModalVAE` is
  experimental (in development).
- **Design (outline):**
  - Standard CV-style tuning per model: Optuna over each model's YAML `search_space`, selection on `val_demeaned_r`,
    best-trial report. Seeds 0–4. This is not the staged design of E1.
  - Trial budget scales with each model's search space (D2).
  - Closed-form and precomputed baselines are included without tuning.
- **Outputs:** per-metric distributions across model types (per-seed points), bar charts, and the `demeaned_pearson`
  vs `avg_rank` scatter (the `sc_type_benchmark` style), through the runner pattern.
- **Depends on:** C2 (re-tune the M5b-affected models; resolves C!1); C!4 re-runs (best-trial reports now train at the
  tuned batch size, C7); E1 for E2.3.
- **Open decisions:** whether E0's degenerate nodal models enter (narrowed re-runs) or stay as E0 results; sources
  beyond SC (`SC_r2t`, `SC+SC_r2t`); which `CovProjector` / `NodalMLP` variants count as separate entries.

#### E2.0 — Direction audit (`SC → FC` vs `FC → SC`) · done 2026-09-30
The shared pipeline is direction-agnostic: dataset `x` / `y`, loss, evaluator (has a `target == "SC"` branch) and the
PCA helpers (`get_modality_data`) follow `--source` / `--target`; `HCP_Base(source="FC", target="SC")` builds.
Exceptions are batch extras that are **always SC / anatomy regardless of direction**: `sc_matrix` (NodalMLP) and
`node_features` (volume, centroid, `SC_r2t`; NodalMLP, NodalGNN). No `FC → SC` run exists on `main` yet (the only FC
launchers are `CrossModalPCA` FC→FC), so "generic" means the code path, not a tested result. Target ranges: FC
−0.81 … 0.96; SC (log1p) 0 … 3.59, 31 % zeros.

| Model | SC → FC | FC → SC | Why / what is needed |
|---|---|---|---|
| Linear decomposition (6), `CrossModalPCA`, `CrossModalVAE` | ✓ | ✓ generic | built from `get_modality_data` source/target means, loadings, scores |
| `LatentAttnMasked`, `MaskedLatentPretrainer`, `MaskedMLPPretrainer` | ✓ | ✓ generic | `sc_*` / `fc_*` names are legacy labels for the source / target roles |
| `Sarwar2020MLP` | ✓ | ⚠ config | default `output_tanh: true` bounds outputs to [−1, 1] but SC reaches 3.59: set `output_tanh: false` for `FC → SC` |
| `Krakencoder_precomputed` | ✓ | ⚠ loader | predictions exist (below); loader hard-codes the `FCcorr` output key and `fc_upper_triangles` targets — select both by `base.target` |
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

**`FC → SC` ceiling:** none in the data (no SC retest). It is assumed to be very high; take a value from the
literature (dMRI structural-connectome test-retest reliability, HCP-YA retest or comparable) and cite it as a reference
line rather than a computed row.

**Krakencoder `FC → SC` results (for memory):** `krakencoder_experimental/example_data/mydata_kraken_seed{seed}_source_{parc}.FC.mat` (folder renamed from `krakencoder/` on 2026-10-01; retrained runs: E2.1)
(seeds 0–9 present for Glasser and 4S456Parcels), key `predicted_alltypes["FCcorr_{parc}_hpf"]["SCifod2act_{parc}_volnorm"]`
(every file holds all four input → output types; the SC-source files are `…_source_{parc}.SC.mat` with outer key
`SCifod2act_{parc}_volnorm`). Checked 2026-09-30 (Glasser, seed 0): 957 subjects in `HCP_Base` order (per-subject r
with our SC 0.915), already on our SC scale (same mean 0.0533; linear fit slope 1.006, intercept 0.000), test demeaned
r ≈ 0.116. The `mydata_kraken_demeaned*` files are other variants, not the per-seed benchmark files.

#### E2.1 — Retrainable Krakencoder baseline · built 2026-10-01; near parity; loss grid complete 2026-10-02
Krakencoder becomes a refittable benchmark model instead of only cached predictions (Option A: tracked wrapper around
the upstream trainer; a native adapter of `krakencoder.model.Krakencoder` into our Lightning loop is a later option).
- **Code:** one package, `models/architectures/krakencoder/`: vendored upstream `vendor/` at `b57e39c` (unmodified; byte-identical to the overlay's
  `krakencoder 1.0.0`; `VENDOR.md`). The old local copy is renamed `krakencoder_experimental/` (gitignored; upstream +
  demeaned-MSE loss, debug prints, `accept_unknowns=True`) and kept as reference / development copy; it still holds
  `participants.tsv` used by `data/dataset_utils.py`. `_vendor_entry.py` pins the vendored import
  and accepts our non-upstream flavor names (the one behavioural setting the local copy changed);
  `retrain.py` (`python -m models.architectures.krakencoder.retrain`) builds inputs from `HCP_Base` (identical to the March inputs and splits),
  trains, infers both source flavors and writes `results/krakencoder/<tag>/seed{S}/`; model `Krakencoder`
  (`models/configs/Krakencoder.yml`, recipe in `retrain:`, output folder = `tag`; loader `precomputed.py`) serves both
  directions through `main.py`; launcher `scripts/sbatch/Krakencoder/train_array_krakencoder_seeds.sh` (train + evaluate SC → FC and FC → SC).
- **Both directions:** one joint fit predicts every input → output type; the loader now selects the target's output key
  and targets (`Krakencoder_precomputed` gains `FC → SC` too). Checked on the cached seed-0 predictions: `FC → SC`
  test demeaned r 0.116, avg rank 0.876, top-1 0.123.
- **Smoke check (2026-10-01):** tag `smoke`, Glasser only, 20 epochs: inputs → train → infer → evaluate in both
  directions completes (near-chance metrics, as expected); resumable stages verified (requeue skips finished stages).
- **Parity (2026-10-01, seed 0, default recipe, tag `kraken_default`): near parity.** Test, retrained vs March cached:
  SC → FC demeaned r 0.083 vs 0.083, avg rank 0.737 vs 0.734, top-1 0.072 vs 0.067; FC → SC 0.121 vs 0.120, 0.883 vs
  0.879, 0.144 vs 0.149; per-subject r between the two prediction sets 0.999 (SC → FC) and 1.000 (FC → SC). Not
  bit-identical: GPU / shuffle nondeterminism and a different code copy (`krakencoder_experimental/`, its extra loss code
  inactive for this loss string); the init-seed noise check below sizes ordinary retrain variability.
- **Epoch pilot (same run, val, batch 41):** SC → FC val demeaned r is flat from epoch 250 (0.063 → 0.066); FC → SC
  rises to ≈ epoch 1000–1250 (0.099 at 250, 0.117 at 1000, 0.119 at 1250–2000). At batch 64 an epoch has ≈ 1.6×
  fewer updates, so the grid's epoch count is set from the batch-64 pilot (dense checkpoints), not from this run.
- **Checkpoint evaluator (`checkpoint_eval.py`):** scores every saved checkpoint in-process (vendored model + transforms)
  with `compute_basic_regression_metrics` (the benchmark tables' definitions) and the `models/train/loss.py` term
  functions in edge space (batches of 64) → `epoch_history.csv` per fit, both directions on Glasser. The last checkpoint
  reproduces run_model's predictions exactly (per-subject r 1.000000). Upstream records only total loss per path.
  Caveat: upstream `generate_adapt_transformer` resets its subject-mask arguments, so input adaptation is fit on all
  subjects (run_model, March runs and here alike); with our inputs it is near identity (fit R² 1.000).
- **Loss-grid instance (`scripts/experiments/composite_loss/krakencoder/`, a sibling of the E1 instances; = E1.8):**
  grid v1 (16 cells) + Krakencoder's paper-default loss (`correye + neidist`, weight 1) as a reference cell + 4 extension
  cells at level 2.0 = 21 cells (105 fits, 27 jobs); weights = grid level × anchor (option B, user 2026-10-01): `correye`,
  `neidist` anchor 1 (paper weight), `var` anchor 9.2 (its level-1 share of the MSE term = `correye`'s at the paper
  weight: 0.0217 vs 0.199 of 1000·MSE at the paper-default solution, `checks/term_magnitudes.py`);
  E1 terms map to Krakencoder's (`varmatch → var`, same formula; `correye`, `neidist`; Krakencoder's native `correye`
  acts in its mean-centred PCA space, so it is closest to our `correye_dm`, not plain `correye`; grid stays v1); fixed in every cell:
  `mse.w1000 + enceye.w10 + encdist.w10 + latentsimloss.w10000`; weights are Krakencoder-native (its terms act in its
  PCA-256 space), not E1's scaled-term footing. Paper-default architecture / optimiser (no Stage 1); batch 64 (D4);
  trained on all 4 flavors, evaluated on Glasser in both directions. Sets: `pilot` (mse_only, ce_0.5, nd_0.5,
  kraken_default × seed 0), `grid` (17 cells × seeds 0–4 = 85 fits, 22 jobs of 4), `noise` (kraken_default, seed 0,
  init seeds 1–2). `grid_runner.py` plans, runs fits packed 4 per GPU (`launch_grid.sh`), scores them and collects
  E1-schema tables (`seed_records`, `epoch_history` + `direction`, `random_seed`). Smoke check (20 epochs, packed 4):
  passed end to end. Pilot (`18973935`) and noise check (`18973936`) submitted 2026-10-01; then autonomous per the
  instance `config.yml` `autonomy:` block (pilot gate, epochs rule, 40 GPU-h grid budget, stop conditions). **Budget:** ≈ 48 min per fit unpacked at 2000 epochs / batch 41; the pilot measures packed
  throughput and the batch-64 plateau, then the grid's `epochs` is fixed in `config.yml` before launch.
- **Loss-grid result (2026-10-02; write-up `scripts/experiments/composite_loss/krakencoder/krakencoder.md`):** pilot
  gate stopped autonomy once (batch-64 FC → SC gap −0.017 vs parity; budget 54 GPU-h) — user accepted the gap and set
  checkpoints every 500; grid `19000122` used 40.7 GPU-h, 105/105 fits (tasks 9, 24 SIGTERM'd during CPU scoring,
  re-scored in CPU jobs). `correye` (≈ our `correye_dm`) drives everything: SC → FC avg rank +0.12 at the paper weight
  for −0.006 demeaned r; FC → SC −0.030 demeaned r with no rank gain. The paper loss ≈ `correye` alone (`neidist`
  inert at 0.1–2×). `var` nudges demeaned r up and avg rank down. Init-seed noise is 5–25× below split-seed noise.
- **Variants:** a copy of `Krakencoder.yml` with a new `tag` and `retrain:` block (e.g. MSE-only `losstype`, Glasser-only
  `parcellations` for equal-data comparison, E1/E3-informed weights). Not Tune-searchable (each fit is a full run).
- **Pattern for other external baselines:** vendor upstream unmodified; adapt only in a wrapper; build inputs from
  `HCP_Base` and its splits; serve predictions through a loader so evaluation is shared; gate on parity with published or
  cached results. `Sarwar2020MLP` and `Chen2024GCN` are already native reimplementations trained in our loop.

#### E2.2 — MSE-only benchmark, SC → FC · in progress (build) · owner: agent:modeling
**Question:** with every model tuned on the same splits under MSE only, how are test demeaned r, avg_rank and top-1
distributed across model types? E2.2 picks the models that go on to E2.3 (composite loss, tuned weights). The best of
those are then reoptimized for the final benchmark, and E3 repeats it for FC → SC.

**Roster** (from the 2026-09-30 audit, updated 2026-10-02):

| Class | Model | Entry |
|---|---|---|
| Null / ceiling | `CrossModalPCA` (null); `TestRetestPrecomputed` (ceiling) | grid / reuse E1.10 |
| Linear, closed-form | `CrossModal_PLS_SVD`, `CrossModal_PCA_PLS`, `CrossModal_ConditionalGaussian` | full grid or budget rule |
| Linear, learned | `CrossModal_PCA_PLS_learnable`, `CrossModal_linear_backbone`, `CrossModal_PCA_PLS_CovProjector` (all covariates) | budget rule |
| Deep-learning baselines | `Sarwar2020MLP`, `Chen2024GCN` (narrowed searches); `Krakencoder` | narrowed / reuse E2.1 |
| Pairwise nodal | `NodalMLP`, `NodalGNN` (at the null in E0; narrowed reruns, so every row is from one campaign) | narrowed |
| Latent / pretrained | `MaskedMLPPretrainer`, linear / low-rank variant | **pilot first (D2)** |
| Excluded | `LatentAttnMasked` (never tuned; attention adds nothing over its linear backbone in the dev runs; C!2), `MaskedLatentPretrainer` (test 0.068, below the linear family), `CrossModalVAE` (in development) | — |

- **Latent pick.** `MaskedMLPPretrainer` (linear variant) is the only tuned latent model with a held-out result:
  2026-04-27, one seed, test demeaned r 0.089 / avg_rank 0.692 (`results/logs/tune_model_parallel_maskedmlp_*`).
  - **Objective:** it trains on its own masked latent reconstruction loss (`latent_mse`), not edge MSE. It enters with
    its native objective, as Krakencoder enters with its own fixed losses, and is labelled as such.
  - **Gate (D2):** seeds 0–1, about 12 trials. It enters the full run if its best val demeaned r ≥ the MSE-only
    `_learnable` val on the same seeds minus 0.01.

**Protocol:**
- Seeds 0–4, SC → FC, selection on val demeaned r, test metrics reported.
- **MSE only:** loss-weight and EMA keys removed from every search; Sarwar's correlation term off.
- **Each model keeps its own batch size and training budget.** D4's batch 64 is a composite-loss rule.
- **Trials:** `clamp(8 × free keys, 16, 64)` with ASHA; closed-form models take their full grid when it is smaller.
  For example, `_learnable` 64, `linear_backbone` 48, `ConditionalGaussian` 40, `PCA_PLS` 24 (150-cell grid).
- **Narrowed searches** (audit, 3,592 past trials: extra trials bought less than seed noise):
  - **Chen:** identity nodes, 2 layers, 500 epochs; `conv_dim` {128, 256}, `dnn_dim` {32, 64}, `lr`, `l2_reg`; 12 trials.
  - **NodalGNN:** 2 layers, decoder 32, 500 epochs; `hidden_dim` {32, 96}, `lr`, `l2_reg`; 10 trials.
  - **Sarwar:** leaky_relu, 300 epochs, plain MSE; layers {3, 5}, hidden {512, 1024}, dropout, `lr`, `l2_reg`; 16 trials.
  - **NodalMLP:** about 3 keys from E0's importance table; about 12 trials.
- **One tagged campaign per model:** results come only from runs tagged with the E2.2 campaign, never the best over
  older sweeps. Older runs carry C!1 / C!4 and differ in budget.

**What exists and what reruns:**
- **Reuse:** Krakencoder (E2.1 `mse_only` and paper default, seeds 0–4) and the test-retest ceiling (E1.10).
- **Rerun everything else.**
  - `_learnable`, CovProjector, Sarwar, Chen and NodalGNN: their March 2026 benchmark rows carry C!1 / C!4, and
    Sarwar's used its correlation loss.
  - The closed-form March rows were selected best-over-sweeps; they are cheap to redo.
  - `ConditionalGaussian` has never been benchmarked.
  - E1's MSE-only fits are consensus configs, not per-seed tuning.
- **This resolves C2** (re-tune the M5b-affected sweeps) and the C!4 re-runs for these models.

**Steps:**
- **E2.2.1 Build (no GPU).**
  - Experiment folder `scripts/experiments/model_benchmark/`: MSE-pinned, budget-scaled config per model, launchers
    and runner.
  - Campaign tag: a `--wandb_tags` flag in `main.py`, and run selection scoped to the tag. This waits until no running
    job reads `main.py`.
  - Checks: configs resolve to plain MSE; search spaces round-trip; `sbatch --test-only`.
- **E2.2.2 Pilot:** one seed per model (wall time, ASHA convergence), plus the `MaskedMLPPretrainer` gate.
- **E2.2.3 Full run:** seeds 0–4, staged by cost: cheap set first, then the narrowed expensive set.
- **E2.2.4 Report:** per-metric distributions by model type with per-seed points; the demeaned r vs avg_rank
  scatter; paired per-seed differences against the best linear model; the test-retest line.

- **Accept:**
  - every row is one tagged campaign × seeds 0–4, verified MSE-only (or native objective, labelled) from run configs;
  - deterministic rendering from tracked records;
  - write-up `scripts/experiments/model_benchmark/model_benchmark.md`.
- **Budget:** about 10–20 GPU-h for the cheap set plus about 30–40 GPU-h for the narrowed expensive set (5 seeds),
  plus pilots. Each stage is approved before launch.

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
  - **Next — Phase D (needs compute approval):** scaffold `fc2sc` for each E1 model, recalibrate the Stage 1 stop
    threshold and budgets for FC → SC, then Stage 1 + consensus + pilot (D2) per model before any grid. Krakencoder's
    FC → SC is already complete. FC → SC ceiling: literature value as `ceiling/<name>.json` with `direction: FC->SC`.
- **Depends on:** E1, E2, E2.0.

## 5. Infrastructure

### I1 — HCP1200 timeseries and connectome-similarity views   · status: in progress · owner: agent:infra (I1.1)

- **I1.1 — Move the HCP1200 timeseries into the data folders** from the transferred `.tar` · in progress ·
  owner: agent:infra. This is an add-only merge into the partially populated destination: never overwrite existing
  files. Procedure and safety rules: `context_packages/HCP1200_xcpd_transfer_merge_handoff.md`.
- **I1.2 — Subject × subject connectome-similarity matrices** · planned · owner: —
  - A correlation matrix over all subjects' connectome comparisons (diagonal = same subject), in raw and **demeaned**
    form (training-set mean subtracted, as in Demeaned corr-eye).
  - Interactive per-subject matrix views of the full and the demeaned connectome.
  - Self-contained HTML per the figure convention (§2).
- **I1.3 — Which behavioral FC best predicts SC** · planned · owner: —
  - With the best E2 model, test which behavioral FC (from the I1.1 timeseries) best predicts SC.
  - This is the starting point for timeseries modeling, with room for spatial analyses.
  - **Depends on:** I1.1, E2, E3 (the FC → SC path).

## 6. Backlog (not scheduled)

From v1 §6, v1 §8.6, the unrun parts of v1 M10, and E0/E1 follow-ups:
- **Graph / nodal models on top of the mean** (from E0): train GNNs on the subject's deviation from the train-split
  mean FC, or on PCA scores / low-rank factors, as the PCA family does, instead of on full FC edges; include an
  SC-only (`use_r2t: false`) `NodalGNN`. Pilot per D2 against `PCA_PLS_learnable` on the same seeds.
- **Editable install** (`pyproject.toml` + `pip install -e .`; the overlay now mounts `:rw` without `--fakeroot`, C6) and `conn2conn/` namespacing.
- **Artifact cleanup:** `results/ray_results/` (118 GB) and a `results/logs/` retention policy.
- **Launcher/config manifest layer** to replace copied per-variant sbatch scripts and YAMLs (8 `CovProjector`, 7 `NodalMLP`).
- **Shared constants** between `main.py` and `scripts/results_utils/records.py`.
- **Untracked reference code** in `context_packages/modeling/*_context/`.
- **Latent-space terms inside composite** (`latent_mse` / `latent_weighted_mse` as mixable terms).
- **EMA diagnostics:** `*_loss_ref_*` behavior after warmup, and batch-size sensitivity of `correye` / `neidist` (64 vs 128).
- ~~**Default normalization:**~~ decided by D6 (2026-10-02): fixed reference scales for composite-loss runs; code default unchanged.

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
| 2026-10-02 | E2.2 specified: roster by class (latent pick `MaskedMLPPretrainer` linear, pilot-gated; `LatentAttnMasked`, `MaskedLatentPretrainer`, `CrossModalVAE` excluded), MSE-only protocol with budget rule and the audit's narrowed searches, one tagged campaign per model, reuse / rerun list, steps E2.2.1–E2.2.4. |

Last updated at: 2026-10-02 EDT
