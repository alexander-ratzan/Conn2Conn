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
| E0 | Architecture check: NodalMLP probe-decoder close-out (`nodal_mlp_probe_decoders`) | in progress (E0.1 done; E0.2 running; E0.5 planned) | code freeze during E0.2; E0.5 on C6, C1 | agent:infra |
| E1 | Composite-loss trade-off on the linear backbone (`composite_loss_tradeoff`) | planned | C3 | agent:modeling |
| E2 | Cross-model benchmark (`model_benchmark`, working name) | outline | C1, C2 | — |
| C1 | `torch_geometric` missing from `kraken_env` | blocked (on C6) | C6 | agent:infra |
| C2 | Re-tune the M5b-affected sweeps | planned (within E2) | E2 | — |
| C3 | Confirm `loss_signature` in Tune-trial W&B configs | done | — | agent:infra |
| C4 | `latent_masked_test` notebook fixes | planned | — | — |
| C6 | Environment: two `kraken_env` stacks; jobs import from `~/.local` | planned (after E0.2) | E0.2 done; overlay not mounted elsewhere | agent:infra |
| C!1 | v1:M5b — sampled L1/L2 not applied in past sweeps | open | resolved by C2 | — |
| C!2 | v1:M1b — `ema` runs with `neidist` ≤ 0 during warmup | open | — | — |
| C!3 | `CrossModal_linear_backbone` z-scored latents are PCA-space | open | — | — |
| D1 | One job environment for all experiments and runs | decided | — | user |

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
| C1 | `torch_geometric` missing from the `kraken_env` overlay (torch 2.9.0); `Chen2024GCN` / `NodalGNN` cannot import, and their launchers fail. It was importable when those models ran (Mar/Apr 2026; both import it unconditionally) and has since disappeared; the overlay never had it. | E0.5, E2 (those two models) | Install `torch_geometric==2.8.0.post1` into the overlay as part of C6 (dry run: 10 new packages, no existing package changed); then `sbatch --array=0 scripts/sbatch/checks/verify_modeling_track_array.sh` (`dev_runs`) must train both. |
| C2 | Re-tune the sweeps affected by v1:M5b (`CrossModal_PCA_PLS_learnable`, `CrossModal_PCA_PLS_CovProjector`, `Sarwar2020MLP`) | E2 | Re-tune within E2; resolves C!1. |
| C3 | Confirm Tune-trial W&B configs carry `loss_signature` (v1 8.7 check failed) | — | **Done 2026-09-29.** The failure was a false negative: wandb 0.25 offline runs write no `files/config.yaml`; the config is inside `run-*.wandb`. Both trials of the 8.7 tune run (`results/ray_checkpoints/CrossModal_linear_backbone_tune_1790207949/*/wandb/offline-run-*/run-*.wandb`) contain `loss_signature` = `mse+0.25*correye+0.5*neidist`. The check in `scripts/sbatch/checks/verify_modeling_track.py` should read `run-*.wandb` (or an online run) instead. E1.2's online sweep can re-confirm at no cost. |
| C4 | `scripts/notebooks/model_testing/latent_masked_test.ipynb`: cell 6 reads `residual_linear.weight` (absent in `attention_only`); cell 4 sets `l2_reg` twice | — | Fix when that notebook is next used. |
| C6 | **Environment divergence** (found 2026-09-29). (1) Two stacks share the overlay: `source /ext3/env.sh` (all 59 launchers) gives torch 2.9.0+cu128 from the overlay **plus `~/.local`**; `source /scratch/asr655/envs/activate_env.sh kraken_env` gives torch 2.11.0+cu130 from `/scratch/asr655/envs/kraken_env/pylibs` (5.1 GB, installed 2026-04-08) and hides `~/.local`. Interactive sessions and jobs can therefore run different torch / Ray / Lightning. (2) Jobs import ray 2.54.1, lightning 2.6.1, wandb 0.25.1, optuna 4.8.0, torchmetrics, pyarrow from `~/.local` (38 packages, 520 MB, 14k files, installed 2026-04-08). Home is at 25.4k / 30k files; `~/.local` is the only large home dir not symlinked to `/scratch`. **Root cause:** `pip install` in a session with the overlay mounted `:ro` silently falls back to a user install in `~/.local`, and `/ext3/env.sh` puts `~/.local` on every job's `sys.path`. | E0.5, E2, every run | See **C6 plan** below. |

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
- **C6.1 Snapshot:** `pip freeze` of the launcher stack (overlay + `~/.local`) saved as the reference "job version";
  `~/.local` file list + checksums; temporary backup `cp --sparse=always` of the overlay (deleted after C6.4).
- **C6.2 Consolidate into the overlay** (`singularity exec --fakeroot --overlay …:rw` with the launchers' image
  `cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif`): install the exact `~/.local` versions (`--no-deps`, from the snapshot) and
  `torch_geometric==2.8.0.post1` into `/ext3/miniforge3` site-packages, existing overlay packages pinned by a constraints
  file. Guards: `PYTHONNOUSERSITE=1`, `PIP_USER=0`, `PIP_CACHE_DIR` on `/scratch`, `unset PYTHONPATH`. Then `pip check`.
- **C6.3 Close the leak:** append to the overlay's `/ext3/env.sh`: `export PYTHONNOUSERSITE=1` and `export PIP_USER=0`
  (all launchers pick it up; none edited). Also drop its stray `PYTHONPATH=<bin dirs>` line. Optional safety net for
  every environment: `~/.config/pip/pip.conf` with `[install] user = false`.
- **C6.4 Verify:** the launcher stack's `pip freeze` equals the C6.1 snapshot plus exactly the PyG packages; `~/.local`
  absent from `sys.path`; `~/.local` file list unchanged; `main.py --help`; `verify_modeling_track` `dev_runs` trains
  every model family including `Chen2024GCN` / `NodalGNN` (closes C1). Then delete the backup copy.
- **C6.5 Align interactive use (user):** point `activate_env.sh` at the same setup as `/ext3/env.sh`, or retire `pylibs`;
  give Jupyter kernel specs (`~/.local/share/jupyter/kernels`) the same two variables.
- **Not in scope yet:** removing packages from `~/.local`. `vformer_env` / `main_env` may still import from it; check
  those projects first.

### Decisions (`D`)

| ID | Decision | Rationale |
|---|---|---|
| D1 | **One job environment for all experiments and runs:** the launchers' stack (`/ext3/env.sh` in `kraken_env`), consolidated into the overlay with user site-packages disabled (C6). Notebooks and interactive sessions use the same stack. | Every recorded result came from the launcher stack; one stack makes interactive checks reproduce in jobs. |

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

### E0 — Architecture check: NodalMLP probe-decoder close-out   (slug: `nodal_mlp_probe_decoders`) · status: in progress · owner: agent:infra

Last check of the v1 architecture (moved launchers, results tooling, runner pattern) on a real, small experiment
before E1. It closes out the never-run `scripts/notebooks/results_scrape/nodal_decoder_importance.ipynb`.

- **Question:** how much of the SC→FC mapping can simple, interpretable edge decoders recover from per-region NodalMLP
  embeddings (`dot`, `bilinear`, `linear_beta`, and the connectome-harmonic `spectral` encoder), and which
  hyperparameters matter for each? Recorded as **closed** so it can be revisited from the SMT angle in GeneEx2Conn.
- **Design:**
  - `NodalMLP` configs `NodalMLP_{dot,bilinear,linear_beta,spectral}.yml`, `SC → FC`, loss MSE-only (v1 D4), seeds 0–3
    (the launchers' `--array=0-3`; stated per §2), selection on `val_demeaned_r`.
  - Evidence: Tune trials created ≥ 2026-04-27 (the NodalMLP schema change), 32 per variant × seed; best-trial reports
    for test metrics. A run belongs to a variant only if its config matches that YAML's options; one best-trial run per
    variant × seed (max `val_demeaned_r`). April trials logged `reg`, which is treated as `l2_reg` (equal per v1 8.7).
  - Reference rows from existing W&B records (no new compute): NodalMLP with the MLP decoder (pre-schema runs),
    `NodalGNN` (untuned prod runs), `Chen2024GCN` (tuned), null `CrossModalPCA` SC and `CrossModal_PCA_PLS_learnable`.
- **Steps:**
  - **E0.1 — Launcher and entrypoint checks (no GPU).** Changes: none. Accept: `bash -n` on all launchers;
    `sbatch --test-only` on all 58 (directives and resources); `main.py --help` in `kraken_env`. `Chen2024GCN` /
    `NodalGNN` launchers still fail at runtime until C1.
    Result (2026-09-29, done): 59 launchers (58 + `checks/verify_modeling_track_array.sh`): `bash -n` 59/59;
    `sbatch --test-only` 59/59 accepted (no jobs queued), plus the two E0.2 submissions with `--array=0|3
    --time=8:00:00 --requeue`. `main.py --help` exits 0 in the launcher image (`cuda12.8.1…sif` + `/ext3/env.sh`):
    torch 2.9.0+cu128, ray 2.54.1, lightning 2.6.1, wandb 0.25.1, optuna 4.8.0; `torch_geometric` absent (C1).
  - **E0.2 — Fill the two missing seeds.** Changes: resubmit the unchanged launchers with overrides:
    `sbatch --array=0 --time=8:00:00 --requeue scripts/sbatch/NodalMLP/tune_array_nodalmlp_bilinear_seeds.sh`
    (seed 0 hit the 4 h limit) and `--array=3 … tune_array_nodalmlp_spectral_seeds.sh` (seed 3 was killed by a signal).
    **Code freeze** on `main.py`, `models/` and `models/configs/NodalMLP_*` from submission until both finish
    (jobs import live code at start and per trial). Accept: logs show `Tune finished: 32 trial(s)` and a best-trial
    report; a new `best_trial_report` run per variant × seed in W&B; trial configs carry `loss_signature` (re-confirms C3).
    Submitted 2026-09-29 against `04ae6ee`: job `18814359` (bilinear, seed 0), job `18814360` (spectral, seed 3).
    Bilinear `18814359` stalled after trial 2 failed at 15:00 (`Trainable runner reuse requires reset_config() to be
    implemented and return True`, from `reuse_actors=True` in `main.py`'s `TuneConfig`); Tune then scheduled nothing.
    Cancelled after 41 min and resubmitted unchanged as `18816010`; the same code and package versions ran all April
    trials without this error. If it recurs, set `reuse_actors=False` after the freeze.
  - **E0.3 — Experiment build.** Changes: `scripts/experiments/nodal_mlp_probe_decoders/` on the runner pattern:
    `config.yml` (variants, cutoff, metric, reference rows), `run.py`, tracked `records.json` (trial params + metrics,
    best-trial records); tables `importance` (fANOVA, hparam × variant), `trial_summary`, `test_summary`,
    `seed_records`; PNGs: importance heatmap, per-variant val distributions, test-metric bars with reference rows;
    new generic `FIGURE_TYPES` entries for the heatmap and distributions. Optional self-contained HTML parallel-
    coordinates view (replaces the notebook's plotly plot). Accept: dry run on current data; rendering deterministic
    from `records.json`.
  - **E0.5 — Graph-model expansion (after C6 and C1).** Changes: tune `NodalGNN` on seeds 0–3 with
    `scripts/sbatch/NodalGNN/tune_array_nodalgnn_SC_seeds.sh` (it has only untuned default runs so far); keep
    `Chen2024GCN` as a reference row from its tuned March sweeps, or re-tune for parity (decide at review). Add them as
    rows in the E0 tables and figures. Accept: 4/4 seeds per added model. Budget: 4 jobs, approved separately.
  - **E0.4 — Close-out.** Changes: `--rescrape`; write-up `nodal_mlp_probe_decoders.md` (status closed, revisit pointer
    to GeneEx2Conn / SMT); experiments-index row; retire the notebook; record E0.1–E0.2 results here. Accept: 4/4 seeds
    per variant; write-up cites caveats by ID.
- **Budget:** 2 GPU jobs × ≤8 h (parallel), E0.2 only.
- **Depends on:** code freeze during E0.2; nothing else (reference rows need no C1).
- **Decisions (defaults, confirm at review):** 4 seeds (not extended to 10); reference rows included; parallel-
  coordinates HTML optional. No W&B experiment tag on the reruns: `main.py` has no CLI flag for extra tags, and adding
  one would break the freeze; runs are identified by variant filters and date.
- **Known from existing data:** best-trial test `demeaned_pearson` ≈ 0.00–0.03 across variants (null 0.012; linear
  family ≈ 0.09); best tune val ≈ 0.04–0.05.

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

Last updated at: 2026-09-29 EDT
