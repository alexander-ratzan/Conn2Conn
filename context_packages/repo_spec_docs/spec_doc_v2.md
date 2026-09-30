# Repo Spec v2 — Experiments

**Purpose:** run structured experiments on the v1 composite-loss / regularization machinery, clear the items carried over
from v1, and hold the repo backlog.
**Status:** active · **Started:** 2026-09-29 · **Predecessor:** [`spec_doc_v1.md`](spec_doc_v1.md) (closed 2026-09-29) ·
**Format:** [`spec_conventions.md`](spec_conventions.md)

**Contents:** [Status](#status) · [1. Purpose](#1-purpose) · [2. Conventions](#2-conventions) ·
[3. Carried over from v1](#3-carried-over-from-v1) · [4. Experiments](#4-experiments) · [5. Backlog](#5-backlog-not-scheduled) ·
[6. Change log](#6-change-log)

To add an experiment, append a section under §4 using the template in §4.0 and add a row to the status table.

## Status

| ID | Title | Status | Depends on | Owner |
|---|---|---|---|---|
| E0 | Nodal models benchmark and architecture check (`nodal_models_benchmark`) | closed 2026-09-30 | — | agent:infra |
| E1 | Composite-loss protocol v1: linear backbone (`composite_loss/linear_backbone`) first; `CrossModal_PCA_PLS_learnable` (`composite_loss/pca_pls_learnable`) paused as the reproducibility target | in progress (linear Stage 1: seeds 0, 3 done, 1/2/4 running `18902227`; learnable Stage 1 paused at seeds 0–2) | D3, D4 | agent:modeling |
| E2 | Cross-model benchmark (`model_benchmark`, working name) | outline (E2.0 direction audit done) | C2 | — |
| E3 | Composite-loss magnitude tuning for final models (follow-up to E1) | outline | E1 | — |
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
| Seeds | `shuffle_seed` 0–9 for benchmark-grade results; smaller seed sets are allowed for staged studies and stated per experiment. |
| Loss normalization | **Fixed reference scales** are the go-forward way to balance composite terms: per-term constants, `loss_normalize: none`, no EMA (E1.1). `auto`/`ema` stay available; whether fixed scales become the repo default is decided after E1 (§5). |
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
| D3 | **E1 runs autonomously within a fixed envelope:** Stage 1 ≤ 4 GPU-h (packed), E1.3 ≤ 1 GPU-h, Stage 2 ≤ 8 GPU-h; defaults approved (seeds 0–4, the 16-point grid, test metrics with selection on val, gradient-cosine panel). The agent stops and reports on any stop condition in the E1 `config.yml` (weak Stage 1, consensus miss, non-finite or mismatched runs, budget overrun). | Lets E1 proceed without per-stage approval once E0 closes, with explicit exits. |
| D4 | **The composite-loss protocol trains at `batch_size` 64 in both stages, for every model.** Models that cannot train at 64 skip the batch-dependent terms (`correye`, `neidist`) rather than change the batch. The envelope in D3 applies per instance. | `correye` / `neidist` compare subjects within a batch, so cross-model comparisons need one batch size; 64 fits the linear family and most learned models in the packed setup. |

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

### E1 — Composite-loss dynamics and trade-off (protocol v1)   (slug: `composite_loss`) · status: in progress · owner: agent:modeling

Experiment home: `scripts/experiments/composite_loss/`: protocol write-up `composite_loss.md`, grid `grid.yml`,
`checks/`, and one folder per model instance (`<model>/`: `<model>.md`, `config.yml`, `stage1/`).

**Protocol and instances.** E1 defines a reusable protocol (v1) and runs it on two instances:
- **Protocol:** the shared grid `scripts/experiments/composite_loss/grid.yml` (versioned; never edited after release),
  batch 64 (D4), seeds 0–4, fixed reference scales measured per model, monitor-only terms in every run, one output
  schema (`seed_records.csv`, `epoch_history.csv`) and W&B tags (`<model>`, `loss_grid:v1`, `combo:<id>`), so later
  instances concatenate for cross-model comparison. Shared code (consensus selection, scale measurement, grid runner,
  figures) lives in `scripts/results_utils/loss_grid.py`; each instance is a thin folder
  `scripts/experiments/composite_loss/<model>/` (`config.yml`, `stage1/`, write-up).
- **Instances:** `composite_loss/linear_backbone` (primary) and `composite_loss/pca_pls_learnable` (replicability:
  does the landscape reproduce on a second linear-family model?). Degenerate models (E0: NodalMLP, NodalGNN) are not
  instances. Budget and autonomy: D3, per instance.
- **Scope:** each combination keeps its model's Stage 1 hyperparameters, so the grid maps the loss landscape rather
  than retuning per loss mix; magnitude tuning for final models is E3.

- **Question:** with MSE fixed at weight 1, how do `varmatch`, `correye` and `neidist` shape the training dynamics of a
  simple, strong linear probe, and how do they trade test `demeaned_pearson` against `avg_rank` across about 16 weight
  combinations? Which weightings are reasonable before the composite loss is extended to other models?
- **Design:**
  - Model `CrossModal_linear_backbone` (and the replicability instance), `SC → FC` (Glasser), `batch_size` 64 (D4).
  - Stage 1 learns the structure and regularization under MSE only. Stage 2 runs a fixed weight grid with the Stage 1
    hyperparameters held fixed, under **fixed reference scales (no EMA)**.
  - **All four terms are logged in every run**: inactive terms as monitor-only terms (E1.1), so dynamics are
    comparable between MSE-only and weighted runs.
  - Seeds 0–4 in both stages.

**Fixed reference scales (definition).** MSE stays raw (scale 1), so Stage 1's `l2_reg` stays calibrated against the loss.
Every other term is rescaled to MSE's magnitude at the Stage 1 reference model:
`term_t / c_t` with `c_t = s_t / s_mse`, where `s_t` is the mean `|raw_t|` over training batches (batch 64, eval mode) of
the Stage 1 model, averaged over seeds. A weight `w_t` then means "w × MSE's size at a good MSE-only solution". The
constants `c_t` are recorded in the experiment `config.yml` and are identical across all Stage 2 runs.

#### E1.1 — Fixed per-term scale and monitor-only terms in `CompositeLoss`  · done 2026-09-30
- **Changes:**
  - Term kwarg `scale` (> 0, default 1): the term contributes `weight · raw / scale`; `*_loss_raw_*` stays unscaled.
    Tune-searchable as `loss_kwarg_<term>__scale`. Fixed scales replace EMA: `auto` with any scale resolves to `none`,
    and `ema` with a scale is rejected.
  - New trainer key `loss_monitor_terms`: extra terms computed each step under `no_grad` and logged as
    `*_loss_raw_*` (Lightning and Tune) without entering the loss. Active terms are dropped from the monitor list.
- **Accept (met, `kraken_env`):** regression 45/45 existing configs bit-identical to `main`; `weight · raw / scale`
  exact on random tensors, gradients included; `scale: 1` bit-identical to no scale under `none` and `ema`; monitors
  leave training bit-identical under `auto` / `ema` / `none`; monitored values equal the term functions; a CPU
  Lightning fit logs every monitor under the names Tune reports; the signature ignores scales and monitors.
- **Result:** `4e3d886`, merged into `main` as `75b7c11` after E0 closed; worktree and branch deleted. Re-checked on
  `main`: `scripts/sbatch/checks/loss_regression.py --old-ref 626f37d` → 45/45 configs bit-identical;
  `composite_loss/checks/check_e1_loss.py` → 23/23. Both checks are tracked (node-local `/tmp` is not persistent).

#### E1.2 — Stage 1: MSE-only tune
- **Changes (done):** `stage1/CrossModal_linear_backbone_mse.yml` (MSE-only search over `n_components_pca_source`,
  `zscore_pca_scores`, `l2_reg`, `l1_reg`, `lr`, `max_epochs`; monitors `varmatch`, `correye`, `neidist`) and
  `stage1/tune_stage1_seeds.sh` (`--array=0-4`, 24 Optuna trials, packed 4 per GPU with actor reuse per E0.6,
  `--report_best_after_tune`). Checked: config resolves to plain MSE with monitors; model keys match the constructor;
  search space round-trips; `bash -n` and `sbatch --test-only` pass.
- **Selection rule:** take each seed's best trial; choose categorical keys by majority and `lr` / `l2_reg` by geometric
  median; then run that single consensus config on seeds 0–4 (E1.3). Accept it if its mean `val_demeaned_r` is within
  one standard error of the mean of the per-seed bests; otherwise use the per-seed best config with the highest mean.
- **Accept:** `ray_tune_id`s and the consensus config recorded in `config.yml`; C3 re-confirmed on these trial runs.
- **Budget:** ≤ 4 GPU-h per instance (D3); about 1.7 GPU-h expected for the linear backbone.
- **Replicability instance:** same launcher pattern; Stage 1 searches `CrossModal_PCA_PLS_learnable`'s own 12 keys
  (MSE-only, `loss_type` fixed to `composite`).

#### E1.3–E1.5 implementation (2026-09-30)
- **Shared code** `scripts/results_utils/loss_grid.py`: Stage 1 discovery (task logs → `ray_tune_id` per seed) and trial
  collection, per-seed summary + stop check, consensus rule and 1-SE acceptance, reference-scale measurement (full
  training batches of 64, eval mode, fixed order), grid-combination loss configs, `run_one` (one model × seed in its own
  process; `Sim(batch_size=64)`; W&B tags), `TermGradCosine` callback, per-run outputs.
- **CLI** `scripts/experiments/composite_loss/protocol.py {stage1,consensus,grid,report} --instance <name>`; exit
  code 2 = a D3 stop condition. Generated values go to `<instance>/state.yml` (config.yml stays hand-written).
  Launchers `launch_consensus.sh <instance>` (1 GPU, 5 seeds in parallel) and `launch_grid.sh <instance>` (4 array
  tasks × 20 runs, 5 in parallel per GPU; packing per D2). Report: `report.py` (tables + 6 PNGs + interactive HTML).
- **Stop behaviour:** a consensus miss stops the run (D3) rather than falling back to another config.
- **Checks** `composite_loss/checks/check_protocol.py` (CPU, synthetic inputs): Stage 1 discovery ignores failed
  attempts; consensus math; all 16 combinations resolve to the expected signature / scales / monitors; scale
  measurement matches a manual computation; gradient cosines match a manual computation; report writes every output
  and rebuilds byte-identically. Real-data training is first exercised by the E1.3 consensus step itself.

#### E1.3 — Consensus runs and reference scales  · code ready
- **Changes:** one runner script in the experiment folder trains the consensus config per seed through
  `Sim._run_learned_single` (monitors on), computes `s_t` for `mse`, `varmatch`, `correye`, `neidist` over training
  batches, writes `c_t` (mean and across-seed spread) into `config.yml`, and saves each run's epoch history.
- **Accept:** `c_t` recorded with spread; `neidist`'s sign at the reference model noted (its scaled term is signed).
- **Budget:** ≤ 1 GPU-h.

#### E1.4 — Stage 2: weight grid  · code ready (gradient cosines need C7)
- **Grid (protocol v1, `composite_loss/grid.yml`; 16 combinations; weights on the scaled terms, MSE = 1, never ablated):**

  | Block | Combinations | Count | Answers |
  |---|---|---|---|
  | factorial ablation at w = 0.5 | every on/off subset of `varmatch`, `correye`, `neidist` (incl. MSE only and all three) | 8 | main effects and interactions; pairs are leave-one-out ablations of all three |
  | dose response | each single term and all three at 0.1 and 1.0 | 8 | how effects scale with weight (3 levels with the w = 0.5 cells) |

- **Changes:** a runner array in the experiment folder, one combination per task looping seeds 0–4, calling
  `Sim._run_learned_single(wandb_tags=[<model>, loss_grid:v1, composite_loss:stage2, combo:<id>])`. `main.py` needs no change.
  Stage 1 consensus config, `loss_normalize: none`, the E1.3 scales, inactive terms as monitors. Each run's per-epoch
  history (raw / weighted / monitor terms, reg, val metrics) goes to `tables/epoch_history.csv`.
- **Accept:** 80 runs complete; every run's `loss_signature` matches its combination; MSE-only reproduces the E1.3
  consensus metrics per seed.
- **Budget:** ≤ 8 GPU-h.

#### E1.5 — Analysis and figures  · code ready
- **Changes:** `run.py` following the runner pattern: `tables/seed_records.csv` (combination × seed),
  `tables/combo_summary.csv` (mean, SE), `tables/epoch_history.csv`. Figures:
  - Trade-off scatter: test `demeaned_pearson` vs `avg_rank`, one point per combination (mean ± SE), MSE-only
    highlighted, `sc_type_benchmark` style; plus `figures/tradeoff_interactive.html` (self-contained; hover and click
    show the four weights, `loss_signature`, mean ± SE and per-seed values).
  - Dynamics: each term's raw trajectory over epochs per combination; loss composition (each term's share of the total)
    over epochs; val demeaned-r / avg-rank over epochs.
  - Single-term response curves (weight → each test metric); gradient cosine between terms on the latent map `W`.
- **Accept:** rendering is deterministic from the tracked tables; each instance write-up (`<model>.md`) and the
  protocol write-up `composite_loss.md` record the question, design,
  how to run, W&B ids, results, observations and caveats (cites v2:C!3).

- **Depends on:** D3.

### E2 — Cross-model benchmark   (slug: `model_benchmark`, working name) · status: outline · owner: —

- **Question:** on equal footing, how do all tunable models compare on test `demeaned_pearson`, `avg_rank` and the other
  standard metrics, in **both directions** (`SC → FC` and `FC → SC`)?
- **Roster:** every registered model, grouped by model type (null / ceiling, linear decomposition, latent / pretrained,
  pairwise nodal, deep-learning baseline, experimental) and learning type (closed-form, supervised, self-supervised,
  precomputed); the classification lives in the README "Models" table. `CrossModalVAE` is experimental (in development).
- **Design (outline):** standard CV-style tuning per model (Optuna over each model's YAML `search_space`, selection on
  `val_demeaned_r`, best-trial report), seeds 0–9. This is **not** the staged design of E1. Closed-form and precomputed
  baselines are included without tuning.
- **Outputs:** per-metric bar charts across models; `demeaned_pearson` vs `avg_rank` scatter across models (the
  `sc_type_benchmark` style), through the runner pattern.
- **Depends on:** C2 (re-tune the M5b-affected models; resolves C!1); C1 done 2026-09-30. E1 may inform whether
  benchmark models also get composite-weight search or stay MSE-only.
- **Open decisions:** sources beyond the two directions (`SC_r2t`, `SC+SC_r2t`); trial budget per model (D2); loss
  policy (MSE-only vs E1-informed); which `CovProjector` / `NodalMLP` variants count as separate entries; further
  requirements for modular cross-model comparisons (to be added).

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

**Krakencoder `FC → SC` results (for memory):** `krakencoder/example_data/mydata_kraken_seed{seed}_source_{parc}.FC.mat`
(seeds 0–9 present for Glasser and 4S456Parcels), key `predicted_alltypes["FCcorr_{parc}_hpf"]["SCifod2act_{parc}_volnorm"]`
(every file holds all four input → output types; the SC-source files are `…_source_{parc}.SC.mat` with outer key
`SCifod2act_{parc}_volnorm`). Checked 2026-09-30 (Glasser, seed 0): 957 subjects in `HCP_Base` order (per-subject r
with our SC 0.915), already on our SC scale (same mean 0.0533; linear fit slope 1.006, intercept 0.000), test demeaned
r ≈ 0.116. The `mydata_kraken_demeaned*` files are other variants, not the per-seed benchmark files.

### E3 — Composite-loss magnitude tuning for final models   (slug: tbd) · status: outline · owner: —

- **Question:** for models where E1's landscape shows a useful direction, what term magnitudes should a final model
  use? E1's grid is coarse (0.1 / 0.5 / 1.0) and holds each model's Stage 1 hyperparameters fixed.
- **Design (outline):** a dense sweep along E1's promising directions, or Optuna over the scaled weights
  (`loss_weight_*` with the E1 fixed scales), with `lr` / `l2_reg` re-tuned jointly; selection on `val_demeaned_r`.
  Only models with established signal (E1 instances, not E0's degenerate models).
- **Depends on:** E1 results.

## 5. Backlog (not scheduled)

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
- **Default normalization:** after E1, decide whether fixed reference scales replace `auto`/`ema` as the repo default.

## 6. Change log

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
| 2026-09-30 | E1 prereqs done: slug → `composite_loss/linear_backbone`; E1.1 (fixed scales + monitor-only terms) built and verified on local branch `e1-loss-scale-monitor`, merges when E0 closes; Stage 1 config and packed launcher added; E1.3/E1.4 runner design and dynamics figures specified; D3 (compute envelope, autonomous execution). |
| 2026-09-30 | E1.1 merged (`75b7c11`) after E0 closed; regression 45/45 and E1.1 checks 23/23 on `main`; checks tracked as `scripts/sbatch/checks/loss_regression.py` and `composite_loss/checks/check_e1_loss.py`. |
| 2026-09-30 | E1 becomes composite-loss protocol v1: shared versioned grid `composite_loss/grid.yml` (8-cell factorial at w = 0.5 + 8 dose points), batch 64 (D4), Stage 1 at 24 trials; replicability instance `composite_loss/pca_pls_learnable` added; E3 (magnitude tuning) outlined. |
| 2026-09-30 | E1 restructured: one experiment folder `scripts/experiments/composite_loss/` (protocol write-up, `grid.yml`, `checks/`, instances `linear_backbone/`, `pca_pls_learnable/`); Stage 1 resubmitted on the new paths: linear seeds 0–2 `18899811`, 3–4 `18899801`; learnable 0–4 `18899802`. The first linear submission (`18899142`, seeds 0–2) failed when the move ran before its search-space read (`FileNotFoundError`, ~11 GPU-min lost): jobs re-read the config after Ray start-up. |
| 2026-09-30 | E1.3–E1.5 code ready (`loss_grid.py`, `protocol.py`, `report.py`, launchers, `check_protocol.py`). C!4 (single runs ignored the tuned batch size) found; fix + `extra_callbacks` on branch `e1-callbacks-batchsize` (C7, merges after Stage 1). Trial-index parsing fixed in the `multimodel_scfc` audit script. |
| 2026-09-30 | C7 merged (`cfc1c32`, regression 45/45): batch-size fix (C!4), `extra_callbacks`, and **Ray sized to the SLURM CPU allocation** — five Stage 1 tasks had hung because Ray pre-started 128 workers (all node cores) that never registered; cancelled. E1 order: linear backbone first (seeds 1/2/4 resubmitted `18902227`), learnable paused at seeds 0–2 as the reproducibility target. |

Last updated at: 2026-09-30 EDT
