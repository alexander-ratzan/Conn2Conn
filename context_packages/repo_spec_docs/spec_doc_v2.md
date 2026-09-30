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
| E0 | Nodal models benchmark and architecture check (`nodal_models_benchmark`) | in progress (E0.1–E0.3, E0.6 done; E0.5 pilot running; results preliminary) | E0.5 gate | agent:infra |
| E1 | Composite-loss trade-off on the linear backbone (`composite_loss_tradeoff`) | planned | C3 | agent:modeling |
| E2 | Cross-model benchmark (`model_benchmark`, working name) | outline | C1, C2 | — |
| C1 | `torch_geometric` missing from `kraken_env` | done 2026-09-30 (via C6) | — | agent:infra |
| C2 | Re-tune the M5b-affected sweeps | planned (within E2) | E2 | — |
| C3 | Confirm `loss_signature` in Tune-trial W&B configs | done | — | agent:infra |
| C4 | `latent_masked_test` notebook fixes | planned | — | — |
| C6 | Environment: two `kraken_env` stacks; jobs import from `~/.local` | done 2026-09-30 (C6.1–C6.4); C6.5 open; archive awaiting deletion | C6.5: user | agent:infra |
| C!1 | v1:M5b — sampled L1/L2 not applied in past sweeps | open | resolved by C2 | — |
| C!2 | v1:M1b — `ema` runs with `neidist` ≤ 0 during warmup | open | — | — |
| C!3 | `CrossModal_linear_backbone` z-scored latents are PCA-space | open | — | — |
| D1 | One job environment for all experiments and runs | decided | — | user |
| D2 | Tuning budget: pilot first, pack small models, scale on evidence | decided | — | user |

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

### Caveats (`C!`)

| ID | Caveat | Affected artifacts | Resolved by |
|---|---|---|---|
| C!1 | v1:M5b — every sweep before 2026-09-23 of `CrossModal_PCA_PLS_learnable`, `CrossModal_PCA_PLS_CovProjector` and `Sarwar2020MLP` trained with the YAML default regularization (L2 = 1e-4 for the PCA/PLS models, none for Sarwar); W&B logged the sampled `l1_reg`/`l2_reg`, which were not applied. Results are valid as default-regularization results. | `sc_type_benchmark` (`PCA_PLS_learnable` rows); `cov_projector_benchmark` (`PCA_PLS_learnable`, all projector rows, `Sarwar2020MLP`) | C2 |
| C!2 | v1:M1b — any `ema` run whose `neidist` reached ≤ 0 during warmup had that term inflated ~10⁸-fold (includes the old `LatentAttnMasked` default composite). Which past runs were hit is not determined. | past `ema` composite runs, mainly `LatentAttnMasked` | re-run or audit if those results are reused |
| C!3 | `CrossModal_linear_backbone(zscore_pca_scores=True)` returns PCA-space latents from `predict_target_latents` (`LatentAttnMasked` returned z-space). Edge outputs are unchanged. | latent losses and latent diagnostics under z-scoring | informational; stays open while z-scored latents are in use |

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

### E0 — Nodal models benchmark and architecture check   (slug: `nodal_models_benchmark`) · status: in progress (results preliminary) · owner: agent:infra

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
- **Preliminary results (2026-09-30, test split, seeds 0–3; write-up `nodal_models_benchmark.md`):** every nodal model
  is at the null. Probes: demeaned r 0.011–0.018, avg rank 0.50–0.54; NodalMLP MLP decoder / NodalGNN default / Chen
  0.016–0.023; null 0.013 / 0.518; MSE-only `PCA_PLS_learnable` 0.078 / 0.686 on the same seeds (all rows verified MSE-only per run; `run.py` enforces it). Probes cannot be ranked at
  n = 3–4 (bilinear seed 0 not re-run). Tune-trial val scores are set mainly by the split (trials on one seed often
  tie), so importance is computed on within-seed-centred scores — near-uniform, no hyperparameter moves the probes off
  the null.
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
  - **E0.5 — NodalGNN pilot** · running (`18887602`, submitted 2026-09-30). `nodal_models_benchmark/pilot/`:
    `NodalGNN_mse_pilot.yml` (`NodalGNN.yml` without the loss-weight / loss-scale searches, so MSE-only; `max_epochs`
    ≤ 750) and `tune_nodalgnn_pilot.sh` (array index = seed); seed 0 × 12 trials, 4 packed per GPU, reuse on (E0.6). **Gate:** extend to seeds 1–3
    only if the seed-0 best val demeaned r ≥ 0.045 (the null's seed-0 val 0.025 + 0.02 — val is only comparable
    within a seed); otherwise record NodalGNN as a negative result. Budget: 1 job, 8 h limit (old 750-epoch default runs took ~47 min
    each on a whole GPU; 3 packed waves expected ~3–4 h); +3 jobs (`--array=1-3`) if gated in.
  - **E0.4 — Close-out** · planned. Add the pilot's NodalGNN rows (a `nodal_gnn` variant in `config.yml`, selected
    like the others), `--rescrape`; write-up `nodal_models_benchmark.md` (status closed, SMT revisit
    pointer); experiments-index row; retire the notebook. Accept: write-up cites caveats by ID.
- **Depends on:** nothing open (C6, C1 and E0.6 done 2026-09-30).

### E1 — Composite-loss trade-off on the linear backbone   (slug: `composite_loss_tradeoff`) · status: planned · owner: agent:modeling

- **Question:** with MSE fixed at weight 1, how do `varmatch`, `correye` and `neidist` trade test `demeaned_pearson`
  against `avg_rank` for a simple, strong linear probe? Where does the trade-off frontier lie across about 16 weight
  combinations?
- **Design:**
  - Model `CrossModal_linear_backbone`, `SC → FC` (Glasser), `batch_size` fixed at 128 (identity terms are batch-dependent).
  - Stage 1 learns the structure and regularization under MSE only. Stage 2 runs a fixed weight grid with the Stage 1
    hyperparameters held fixed, under **fixed reference scales (no EMA)**.
  - Seeds 0–4 in both stages.

**Fixed reference scales (definition).** MSE stays raw (scale 1), so Stage 1's `l2_reg` stays calibrated against the loss.
Every other term is rescaled to MSE's magnitude at the Stage 1 reference model:
`term_t / c_t` with `c_t = s_t / s_mse`, where `s_t` is the mean `|raw_t|` over training batches (batch 128, eval mode) of
the Stage 1 model, averaged over seeds. A weight `w_t` then means "w × MSE's size at a good MSE-only solution". The
constants `c_t` are recorded in the experiment `config.yml` and are identical across all Stage 2 runs.

#### E1.1 — Fixed per-term scale in `CompositeLoss`
- **Changes:** optional term kwarg `scale` (default 1): the term contributes `weight · raw / scale`. `*_loss_raw_*` stays
  unscaled; `*_loss_term_*` / `*_loss_weighted_*` include the scale. Reachable from Tune as `loss_kwarg_<term>__scale`.
  `scale` must be > 0. Validation rejects `scale` combined with `ema`/`auto` for a multi-term loss, so two
  normalizations can't be stacked by accident.
- **Accept:** `scale: 1` is bit-identical to the current composite (regression harness over all YAMLs); `weight · raw / scale`
  holds exactly on random tensors, gradients included; the signature is unchanged (scales are recorded in the experiment
  config, not the signature).

#### E1.2 — Stage 1: MSE-only tune
- **Changes:**
  - `scripts/experiments/composite_loss_tradeoff/` skeleton (`composite_loss_tradeoff.md`, `config.yml`) and index row.
  - Launcher `scripts/sbatch/CrossModal_linear_backbone/tune_array_linear_backbone_mse_seeds.sh`, based on the
    `_learnable` array template: `--array=0-4` (seeds), Optuna, `--num_samples 32`, `--report_best_after_tune`.
  - Search: the `CrossModal_linear_backbone.yml` structure and regularization keys (`n_components_pca_source`,
    `zscore_pca_scores`, `l2_reg`, `l1_reg`, `lr`, `max_epochs`), loss weights pinned at 0 (MSE only).
- **Selection rule:** take each seed's best trial; choose categorical keys by majority and `lr` / `l2_reg` by geometric
  median; then run that single consensus config on seeds 0–4. Accept it if its mean `val_demeaned_r` is within one
  standard error of the mean of the per-seed bests; otherwise use the best per-seed config with the highest mean over seeds.
- **Accept:** the consensus config and its 5-seed val/test metrics are recorded in the experiment doc. C3 is checked
  on these trial runs.
- **Budget:** 5 tasks × ≤2 h ≈ ≤10 GPU-h, plus the consensus rerun (5 short runs).

#### E1.3 — Reference scales
- **Changes:** a small script in the experiment folder loads the Stage 1 consensus model per seed and computes `s_t`
  for `mse`, `varmatch`, `correye`, `neidist` over training batches. It writes `c_t` (mean and across-seed spread) into
  `config.yml`.
- **Accept:** `c_t` recorded with spread. `neidist`'s sign at the reference model is noted, since its scaled term is signed.

#### E1.4 — Stage 2: weight grid
- **Grid (16 combinations; weights on the scaled terms, MSE = 1):**

  | Group | Combinations | Count |
  |---|---|---|
  | baseline | MSE only | 1 |
  | single term | `varmatch`, `correye`, `neidist` each at {0.1, 0.5, 1.0} | 9 |
  | pairs | each pair at 0.5 / 0.5 | 3 |
  | all three | all at {0.1, 0.5, 1.0} | 3 |

- **Changes:** `config.yml` lists the 16 combinations. Launcher
  `scripts/sbatch/CrossModal_linear_backbone/run_array_linear_backbone_loss_grid.sh`: `--array=0-15`, one combination
  per task, looping seeds 0–4 as direct prod runs (no tune) with the Stage 1 config, `loss_normalize: none` and the
  E1.3 scales. W&B tags `composite_loss_tradeoff`, `composite_loss_tradeoff:stage2`, `combo:<id>`.
- **Accept:** 80 runs complete; every run's `loss_signature` matches its combination; MSE-only reproduces the E1.2
  consensus metrics per seed.
- **Budget:** 80 short runs ≈ ≤8 GPU-h.

#### E1.5 — Analysis and figures
- **Changes:** `run.py` following the runner pattern: `tables/seed_records.csv` (combination × seed) and
  `tables/combo_summary.csv` (mean, SE). Figures:
  - `figures/metric_scatter__demeaned_pearson__vs__avg_rank__SC.png`: one point per combination (mean over seeds,
    SE bars), MSE-only highlighted, same style as `sc_type_benchmark`.
  - `figures/tradeoff_interactive.html`: the same scatter, self-contained. Hover and click on a point show the four
    weights, `loss_signature`, mean ± SE and the per-seed values.
  - Optional: single-term response curves (weight → each metric).
- **Accept:** rendering is deterministic from the tracked tables; `composite_loss_tradeoff.md` records the question,
  design, how to run, W&B ids, results, observations and caveats.

- **Depends on:** v1 (done); C3 (done; E1.2's online sweep re-confirms it).
- **Decisions (defaults, confirm at review):** seeds 0–4 in both stages; plotted metrics on test with selection on val;
  the grid above.

### E2 — Cross-model benchmark   (slug: `model_benchmark`, working name) · status: outline · owner: —

- **Question:** on equal footing, how do all tunable models compare on test `demeaned_pearson`, `avg_rank` and the other
  standard metrics?
- **Design (outline):** standard CV-style tuning per model (Optuna over each model's YAML `search_space`, selection on
  `val_demeaned_r`, best-trial report), seeds 0–9. This is **not** the staged design of E1. Closed-form and precomputed
  baselines are included without tuning.
- **Outputs:** per-metric bar charts across models; `demeaned_pearson` vs `avg_rank` scatter across models (the
  `sc_type_benchmark` style), through the runner pattern.
- **Depends on:** C1 (PyG for `Chen2024GCN` / `NodalGNN`), C2 (re-tune the M5b-affected models; resolves C!1). E1 may inform whether
  benchmark models also get composite-weight search or stay MSE-only.
- **Open decisions:** sources (`SC` only vs `SC`, `SC_r2t`, `SC+SC_r2t`); trial budget per model; loss policy (MSE-only
  vs E1-informed); which `CovProjector` / `NodalMLP` variants count as separate entries.

## 5. Backlog (not scheduled)

From v1 §6, v1 §8.6, the unrun parts of v1 M10, and E1 follow-ups:
- **Editable install** (`pyproject.toml` + `pip install -e .`, needs a one-time `:rw` overlay mount) and `conn2conn/` namespacing.
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

Last updated at: 2026-09-29 EDT
