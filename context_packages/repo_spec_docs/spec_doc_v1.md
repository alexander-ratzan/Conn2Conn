# Repo Spec v1 — `scripts/` Build-Out (Stage 1 Code Refactor)

**Status:** Stage 1 complete (T1–T8 done); modeling track §8 ready for execution (M1–M11) · **Started:** 2026-09-23 · **Scope:** Conn2Conn repo layout, `scripts/` umbrella
**Tracks:** stepwise to-dos, deliverables, and acceptance criteria for moving all non-library code
(results tooling, notebooks, experiments, SLURM launchers) under a single `scripts/` folder.

---

## 1. Purpose

Separate the repo into three clear roles:

| Role | Location | Tracked in git |
|---|---|---|
| Core library + entrypoint | `main.py`, `data/`, `models/`, `krakencoder/` | yes (krakencoder vendored, ignored) |
| Everything that *uses* the library | `scripts/` | yes |
| Generated artifacts | `results/` | no (except two `local_results/` reports, see §3) |

Goals:
- A tidy repo root with one umbrella for non-library code.
- `results/` holds output only; no code lives there.
- Notebooks, experiments, and launchers can move within the repo without breaking imports or paths.
- Side experiments get a consistent home and a ledger entry, instead of ad-hoc notebooks.

## 2. Target layout

```
Conn2Conn/
├── main.py                        # entrypoint (unchanged)
├── data/  models/  krakencoder/   # core library (unchanged)
├── scripts/
│   ├── __init__.py                # makes `scripts.results_utils` a regular package import
│   ├── results_utils/             # W&B/Ray scraping → tables → figures        [DONE]
│   │   ├── __init__.py            #   notebook-facing API + reload()
│   │   ├── records.py             #   paths, W&B consts, display vocab, fetch → RunRecord, cache, local enrichment
│   │   ├── tables.py              #   status/metric/covtype/SC-type/cov_dl tables
│   │   ├── plots.py               #   source bars, model scatter, cov_dl plots
│   │   ├── local_results.py       #   results/local_results/ loaders + plots
│   │   ├── runner.py              #   shared plumbing for config-driven experiment scripts (post-v1)
│   │   └── optuna_importance.py   #   python -m scripts.results_utils.optuna_importance
│   ├── notebooks/                 # interactive: EDA, model overviews/testing, results scraping   [T2]
│   │   ├── EDA/
│   │   ├── kraken/
│   │   ├── model_overviews/
│   │   ├── model_testing/
│   │   └── results_scrape/        #   + nodal_decoder_importance.ipynb
│   ├── experiments/<name>/        # self-contained side experiments: code, launchers, small outputs  [T5]
│   └── sbatch/<Model>/            # core model tuning grids                                      [T4]
├── results/                       # generated artifacts only (bulky experiment outputs → results/experiments/<name>/)
└── context_packages/
    └── repo_spec_docs/            # this document
```

## 3. Decisions and conventions

| Decision | Choice | Rationale |
|---|---|---|
| Umbrella folder | `scripts/` | Tidy root; all non-library code in one place. |
| Results tooling name | `scripts/results_utils/` | Avoids a `scripts/results` vs `results/` name clash; content is scrapers + tables + plots. |
| Module granularity | stage-based modules (records → tables → plots) + `local_results`, `optuna_importance`, and (post-v1) `runner` for shared experiment-script plumbing | Limit file bloat; one-way dependencies `runner → plots → tables → records`. |
| Backward-compat shims | none | Repo convention: callers are updated instead. |
| Notebook imports | walk-up bootstrap in each notebook's first cell (below) | Survives moves; no install step. Editable install deferred (§6). |
| Notebook API surface | import from package level (`from scripts.results_utils import ...`) | Reshuffling files inside `results_utils/` never touches notebooks. |
| `results/local_results/` reports | left as-is (Krakencoder, test_structured_loss_model tracked via `.gitignore` exceptions) | Revisit later. |
| `scripts/notebooks/kraken/` | stays a notebook folder | Not converted to an experiment. |
| Experiment documentation | write-up inside each experiment folder, named after it (`<slug>/<slug>.md`, not `README.md`), or the notebook itself when self-documenting; outer index `scripts/experiments/experiments_index.md` (changed 2026-09-23 from a separate `context_packages/experiment_ledger/`) | One place per experiment: code, config, results snapshot, and write-up move together. |
| Experiment folder naming | plain descriptive slug, e.g. `linear_backbone_geodesic` | Keep it simple. |
| Notebook checkpoints | `.ipynb_checkpoints/` deleted at the repo root and under notebooks; the ones inside vendored `krakencoder/` and artifact `results/` are left alone | Untracked editor clutter; already gitignored. |

Standard notebook bootstrap (already applied to all notebooks):

```python
import sys
from pathlib import Path

REPO_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "main.py").exists())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
```

Repo paths inside notebooks are written as `REPO_ROOT / "results/..."`, never cwd-relative or absolute `/scratch/...`.
Known limitation: raises a bare `StopIteration` if the kernel cwd is outside the repo.

## 4. Completed groundwork

| # | Change | Commit |
|---|---|---|
| G1 | Removed stale `context_packages/main.py` and `hcp_dataset.py` copies (strict older subsets of live code; untracked) | — (untracked delete) |
| G2 | Results scripts moved into tracked package; `.gitignore` fixed (old `results/` rule blocked all re-includes) | `487bb12` |
| G3 | `results_scraper.py` (2.1k lines) split into `records` / `tables` / `plots`; function bodies verbatim; notebook + optuna imports fixed | `f5a4253` |
| G4 | `test_structured_loss_model` local report tracked | `5c5304e` |
| G5 | `context_packages/` reorganization committed (latent-attention docs → `modeling/`) | `9ef6565` |
| G6 | Package moved to `scripts/results_utils/`; `results/__init__.py` removed | `e6ce354` |
| G7 | Walk-up bootstrap in all 32 notebooks; cwd-relative and absolute repo paths anchored on `REPO_ROOT` | `5690551` |

## 5. Stepwise plan

Each step is one commit unless noted. Verify before committing; never run notebooks or compute on the login node.

### T1 — Author this spec doc  ✅ done
- **Deliverable:** `context_packages/repo_spec_docs/spec_doc_v1.md`.
- **Accept:** reviewed; committed.

### T2 — Move notebooks → `scripts/notebooks/`  ✅ done
- **Changes:**
  - `git mv notebooks scripts/notebooks`
  - `nodal_decoder_importance.ipynb` → `scripts/notebooks/results_scrape/`
  - delete all untracked `.ipynb_checkpoints/` dirs: under `notebooks/` and the stray repo-root `.ipynb_checkpoints/` (holds `main-checkpoint.py` + old notebook checkpoints)
  - `quick_experiments/` moves along for now; emptied in T5
- **Accept:** git records pure renames; bootstrap resolves `REPO_ROOT` from the new depth; no `.ipynb_checkpoints/` at the repo root or under `scripts/notebooks/`; no remaining `notebooks/` references outside docs.
- **Result:** `506e008`; 32 renames, ~400 MB of checkpoints removed.

### T3 — Fix stale `models.*` imports (13 notebooks)  ✅ done
Notebooks still imported modules removed in the `models/` refactor. Mapping applied (verified per symbol against current definitions):

| Old import | New location |
|---|---|
| `models.config` (`build_model`, `load_config`, `get_default_config`, ...) | `models.registry` |
| `models.loss.train_model` | `models.train.trainer` |
| `models.loss.get_target_train_mean` (and other loss terms) | `models.train.loss` |
| `models.models.predict_from_loader` | `models.utils` |
| `models.lightning_module` | `models.train.lightning_module` |
| `models.conditional_gaussian` | `models.architectures.latent_attention.conditional_gaussian` |
| `models.latent_attn_masked` | `models.architectures.latent_attention.latent_attn_masked` |
| `models.eval_utils` | `models.eval.eval_utils` |
| `models.FC_distance` | `models.eval.fc_distance` (`distance_top1_accuracy`, `distance_avg_rank` → `models.eval.metrics`) |
| `from models.eval import Evaluator` | `models.eval.evaluator` (package `__init__` re-exports nothing) |
| `from models import CrossModalPCA, ...` | `models.architectures.crossmodal_pca_pls` |

Affected: `kraken/kraken_eval`, `kraken/track_krakencoder_model`,
`model_overviews/{conditional_gaussian_multimodal, latent_masked_attention_softmax, latent_masked_linear_residual, latent_masked_transformer, test_latent_attn_masked}_overview`,
`model_testing/{test_conditional_gaussian_model, test_conditional_gaussian_raw_edges_model, test_loss_linear_model, test_proj_model}`,
`quick_experiments/linear_backbone_geodesic`, plus `model_testing/test_VAE` (package-level `from models import ...`).
- **Rules:** explicit imports are rewritten name-by-name; star imports become explicit imports of only the names the notebook uses (dropped when none are used); bare `import models.X` / `importlib.reload(models.X)` lines point at the new modules the notebook draws from.
- **Accept:** every `models.*` / `data.*` / `scripts.*` import in every notebook resolves to an existing module and symbol (static check + import-only run of each notebook's import cell in `kraken_env`, overlay mounted `:ro`).
- **Result:** 108 import lines rewritten in 13 notebooks; static check and `kraken_env` import run pass for all 32 notebooks. Notebook bodies were not executed.

### T4 — Move SLURM launchers → `scripts/sbatch/`  ✅ done
- **Changes:** `git mv sbatch scripts/sbatch`; update the two PCA/PLS overview notebooks that build `REPO_ROOT / "sbatch/..."` paths.
- **Safe because:** launchers use absolute `#SBATCH --output/--error` paths and `cd ${CONN2CONN_DIR}` before `python main.py`; SLURM copies scripts at submission, so queued jobs are unaffected.
- **Accept:** no `sbatch/` path references outside docs; `bash -n` passes on every launcher; submission command becomes `sbatch scripts/sbatch/<Model>/<script>.sh`.
- **Result:** 58 launchers renamed; all set an absolute `CONN2CONN_DIR` (nothing script-relative); 10 notebook paths updated and all resolve; `bash -n` clean.

### T5 — Create `scripts/experiments/`  ✅ done
- **Changes:**
  - `git mv scripts/notebooks/quick_experiments/linear_backbone_geodesic.ipynb scripts/experiments/linear_backbone_geodesic/` (after T3 fixes its imports); remove `scripts/notebooks/quick_experiments/`
  - no READMEs: an experiment folder holds only its code, launchers, and small outputs
  - bulky outputs go to `results/experiments/<name>/` (already ignored by `results/*`)
- **Accept:** `scripts/experiments/linear_backbone_geodesic/` exists; `quick_experiments/` gone; notebook bootstrap still resolves from the new depth.
- **Result:** notebook renamed into `scripts/experiments/linear_backbone_geodesic/`; its `RESULTS_ROOT` (previously only created, never written) now points at `results/experiments/linear_backbone_geodesic/`.

### T6 — Experiment ledger entry  ✅ done
- **Changes:** add a ledger entry for `linear_backbone_geodesic` in `context_packages/experiment_ledger/`, following the existing ledger style: what it tested, where the code lives (`scripts/experiments/linear_backbone_geodesic/`), how to run, W&B tags / `ray_tune_id`s if any, status and outcome (filled from the notebook's contents; unknowns marked as such, not guessed).
- **Accept:** every folder under `scripts/experiments/` is covered by a ledger entry.
- **Result:** `context_packages/experiment_ledger/linear_backbone_geodesic.md` (question, recorded setup, how to run, recorded test-split results, observations, caveats).

### T7 — `.gitignore` review  ✅ done
- **Changes:** confirm `results/*` covers `results/experiments/`; confirm all of `scripts/**` (code and small experiment outputs) is tracked; keep the two `local_results/` exceptions.
- **Accept:** `git check-ignore -v` matrix over representative paths matches §1.
- **Result:** no `.gitignore` change needed; an 18-path `git check-ignore --no-index` matrix matches §1 and nothing under `scripts/` is ignored. Global rules also keep `checkpoints/`, `lightning_logs/`, `wandb/` inside experiment folders out of git. **Gotcha:** the global `*_context/` rule would ignore an experiment folder named `*_context` — avoid that suffix under `scripts/`.

### T8 — Documentation pass (end of session)  ✅ done
- **Changes:** README.md and CONTEXT.md — repo layout, `scripts/results_utils` API (replacing `results/results_scraper.py` references), notebook paths, `sbatch scripts/sbatch/...` commands, `context_packages/` layout, notebook bootstrap convention; update `Last updated at` signatures. Check `Conn2ConnWorkspace/.cursor/rules.md` for stale paths.
- **Accept:** no doc references to `results/results_scraper.py`, `results/scripts/`, top-level `notebooks/`, or top-level `sbatch/`.
- **Result:** README.md and CONTEXT.md rewritten for the `scripts/` layout (results_utils API by module, notebook bootstrap + experiment/ledger conventions, `scripts/sbatch` commands, corrected `results/` folder roles, new gotchas #12–13); old paths remain only as history in CONTEXT Recent Changes; every backticked path resolves. Workspace `rules.md` and `/scratch/asr655/CLAUDE.md` had no stale paths.

## 6. Deferred / out of scope for v1

- **Editable install** (`pyproject.toml` + `pip install -e .`) to remove per-notebook bootstraps; needs a one-time `:rw` overlay mount. Longer term, namespace under `conn2conn/` to avoid generic top-level names (`data`, `models`, `scripts`).
- **Artifact cleanup:** `results/ray_results/` (118 GB), `results/ray_tmp/`, dangling `results/wandb/` symlinks, `results/logs/` retention policy.
- **Launcher/config manifest layer** to replace copied per-variant sbatch scripts and YAMLs (CONTEXT.md gotcha #11).
- **Shared constants** between `main.py` and `scripts/results_utils/records.py` (W&B project/entity, results paths are currently duplicated).
- **Untracked reference code** in `context_packages/modeling/*_context/` (`.py`, `.csv` are ignored; only `.md` is tracked).

## 7. Open questions

None open. Resolved 2026-09-23: delete all `.ipynb_checkpoints/` (T2); keep `scripts/notebooks/kraken/` as notebooks; no experiment READMEs, ledger is the record, plain slug naming (T5/T6).

## 8. Modeling track (M-steps)

**Status:** ready for execution (M1–M11); decisions on the former open items are in 8.5 · **Drafted:** 2026-09-23
**Goal:** every learned model that predicts FC edges trains through one composite loss path. Its term weights and L1/L2 strengths are searchable under the same keys for every model. The linear backbone is separated from `LatentAttnMasked`. Loss and regularization code that duplicates this path is removed.
Same rules as §5: one commit per step unless noted; verify before committing; no training or long compute on the login node (checks that need a GPU or data go through SLURM); callers are updated, with no backward-compatibility shims.

### 8.1 Scope

Composite terms act on FC edge predictions (`y_pred` vs `y`). They therefore apply to every learned model trained through `CrossModalLightningModule` whose output is FC edges.

| Model | Loss today | After this track |
|---|---|---|
| `LatentAttnMasked` | `composite` (4 terms, weight 1, `ema`) | composite, MSE-only default; the four terms are searched (8.5 D1) |
| linear backbone (`LatentAttnMasked`, `residual_mode: none`) | notebook overrides | `CrossModal_linear_backbone`, a thin subclass of `CrossModal_PCA_PLS_learnable` (M7), with composite |
| `CrossModal_PCA_PLS_learnable` (+2 source variants) | `mse` | composite |
| `CrossModal_PCA_PLS_CovProjector` (8 YAMLs) | `mse` | composite |
| `Chen2024GCN`, `NodalGNN` | `mse` | composite |
| `NodalMLP` (7 YAMLs) | `mse` | composite, MSE-only; identity terms withheld until it shows learning signal (8.5 D4) |
| `Sarwar2020MLP` | `sarwar_mse_corr` | composite `[mse, pairwise_corr]` (M4) |
| `CrossModalVAE` | `vae` | composite `[mse, kld]` (M4) |
| `MaskedLatentPretrainer`, `MaskedMLPPretrainer` | `latent_*` | unchanged (latent objective; see 8.6) |
| closed-form / precomputed models | — | unchanged (no training) |

### 8.2 Decisions

| Decision | Choice | Rationale |
|---|---|---|
| One loss path | All in-scope models use `loss_type: composite`; the `mse`, `weighted_mse`, `sarwar_mse_corr` and `vae` loss types are retired. The `latent_*` loss types stay. | Consistency across models going forward. |
| Default = plain MSE | Default `loss_terms` has only `mse` active, with `loss_normalize: auto` (below). | Computes exactly `F.mse_loss` (same values, same gradients), so hyperparameters tuned in earlier MSE sweeps and past results stay comparable. |
| `loss_normalize: auto` (new default) | `none` when only one term has nonzero weight; `ema` otherwise. `ema` and `none` can still be set explicitly. | An MSE-only run stays exact; a weighted run is normalized automatically. |
| Weight anchor | `mse` weight fixed at 1.0; only the other term weights are searched. | Removes the overall scale, which `lr` / reg already cover; weights read as "relative to MSE". |
| Weight 0 | Drops the term (not computed, not logged). | Lets one search switch a term off. |
| Searchable weights | Flat trainer keys `loss_weight_<term>`. | Tune only reaches top-level keys; flat scalars work with Optuna and become filterable W&B config columns. |
| Searchable term kwargs | Flat trainer keys `loss_kwarg_<term>__<name>` (e.g. `loss_kwarg_pairwise_corr__corr_target`). | Sarwar's `corr_target` is searched today. |
| Weighted-MSE scaling | Not kept as a separate mode. `weighted_mse(α)` is exactly composite `[mse: α, demeaned_mse: 1−α]` with `ema` and `loss_scale_warmup_steps: 1`: the first training step sets the scale, then it freezes. | No YAML, launcher or notebook uses `weighted_mse`. |
| KLD | `kld` term; used with `loss_normalize: none`, so its weight *is* β. Raises if the model returns no `mu`/`logvar`. | Keeps VAE-type losses; the old `VAELoss` silently dropped KLD when `mu` was missing. |
| Loss signature | Flat config key `loss_signature`, e.g. `"mse"` or `"mse+0.5*neidist+0.25*correye"`, logged to W&B and excluded from model constructors. | Groups old (`loss_type: mse`) and new runs by what they actually optimized. |
| Selection metric | `val_demeaned_r` (current Tune default and best-trial selector). | Independent of loss scale, so trials with different weights or normalization are comparable. |
| Regularization keys | Every learned model takes `l1_reg` and `l2_reg`; the scalar `reg` is removed. Which parameters are penalized stays model-specific (`_reg_params()`); strength goes through `compute_reg_loss(params, l1_l2_tuple)`. Models tied to one style set the other to 0 and don't search it. | `build_model` already maps `l1_reg`/`l2_reg` → `l1_l2_tuple` (`registry.py:198`); the PCA/PLS family, VAE and Sarwar already use it. |
| `Chen2024GCN` reg exception | Keeps its paper-faithful `l2_reg · ‖W‖₂` (plain, not squared, norm on the two edge-MLP layers); `l1_reg > 0` raises. | Preserves the replication. Documented as an exception. |
| Linear backbone | Folded into the PCA/PLS family as `CrossModal_linear_backbone`, a thin subclass of `CrossModal_PCA_PLS_learnable` that pins the backbone flags (8.5 D2). It has its own model name, YAML, W&B tag and results-table rows. Any missing behavior is added as flags on `_learnable`. `LatentAttnMasked` reuses the same linear latent map for `linear_residual`. | One model family for PCA-space linear maps; removes unused modules; the backbone becomes sweepable. |

**Default search ranges** (M3/M8 YAMLs), searched under `ema`, where each term starts at about 1:

```yaml
loss_weight_neidist:  {type: choice, values: [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]}
loss_weight_correye:  {type: choice, values: [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]}
loss_weight_varmatch: {type: choice, values: [0.0, 0.1, 0.25, 0.5, 1.0]}
loss_scale_warmup_steps: {type: choice, values: [20, 50, 100]}
loss_scale_ema_decay:    {type: choice, values: [0.9, 0.95, 0.99]}
l2_reg: {type: loguniform, lower: 1.0e-7, upper: 1.0e-3}
l1_reg: {type: choice, values: [0.0, 1.0e-7, 1.0e-6, 1.0e-5]}   # omitted for L2-only models
```

To move an `ema` result to `none`, don't hand-tune separate ranges. Convert with `w_none = w_ema × ref_mse / ref_term`, using the frozen scales each run logs as `*_loss_ref_*`. Raw term scales differ by orders of magnitude, and `correye` grows with batch size.

### 8.3 Stepwise plan

#### M1 — Fix stale loss configs in notebooks  ✅ done
- **Changes:** `scripts/notebooks/model_testing/test_loss_linear_model.ipynb` cell 6: `"balanced_composite"` → `"composite"`. Check every notebook's `loss_type` literals against `create_loss_fn`.
- **Accept:** a static check finds no loss type outside the accepted set.
- **Result:** `e094f3d`. 4 live `balanced_composite` settings, in `test_loss_linear_model` and `linear_backbone_geodesic_metrics`, changed to `composite`, plus one markdown mention. `joint_edge_latent_mse_scaled` appears only in commented-out lines and was left alone.

#### M1b — EMA scale for signed composite terms  ✅ done
- **Bug:** `neidist = d_self − d_other` goes negative once predictions are identifiable. `CompositeLoss._maybe_update_scales` clamped the *signed* value to `≥1e-8`, so a negative `neidist` during warmup pinned its scale at `1e-8`. The term was then inflated about 10⁸-fold, dominating the loss and its gradients. Reproduced: noise 0.1 on random data gave total = −7.96×10⁸.
- **Fix:** the EMA scale tracks `|raw|` (floor `1e-8`). Normalized terms keep their sign and start at about ±1.
- **Accept (met):**
  - The failing case gives normalized `neidist` ≈ −1.01 and total 1.0.
  - A `neidist` crossing zero during warmup stays finite (max |total| 12.3).
  - With only non-negative terms (`mse`, `varmatch`, `correye`), outputs are identical before and after.
- **Note:** scales freeze after warmup, so a term that grows much larger after warmup (e.g. `neidist` as predictions improve) can reach a normalized magnitude well above 1. This is inherent to frozen-scale normalization; M10 watches `*_loss_ref_*` against the raw terms.
- **Impact on past runs:** any `ema` run whose `neidist` reached ≤ 0 during warmup was affected. That includes the `LatentAttnMasked` default composite. Which past runs were hit is not determined here.

#### M2 — Single loss-config path + `loss_signature`  ✅ done
- **Changes:**
  - Add `resolve_loss_config(trainer_cfg) -> dict` in `models/train/loss.py`. It collects the `loss_*` keys, applies the M3 flat overrides, resolves `loss_normalize: auto`, validates term names, and returns the resolved config plus `loss_signature`.
  - Replace the nine hand-threaded `loss_*` arguments in `main.py` (`_run_learned_single` ×2, the Tune trainable), `models/train/trainer.py::train_model` and `CrossModalLightningModule` with one `loss_cfg` dict, consumed by `create_loss_fn(**loss_cfg)`.
  - Log `loss_signature` as a flat config key in prod, Tune-trial and best-trial runs.
- **Accept:** for every YAML in `models/configs/`, the built loss module (class and parameters) is unchanged. Logged hparam keys are unchanged apart from the added `loss_signature`. `bash -n`/import checks pass.
- **Result:**
  - `models/train/loss.py` gains `resolve_loss_config` (defaults, validation, `loss_signature`), `LOSS_CONFIG_DEFAULTS` and `EDGE_/LATENT_LOSS_TYPES`. `create_loss_fn(loss_cfg, base)` takes the resolved dict.
  - `CrossModalLightningModule(..., loss_cfg=...)` and `train_model(..., loss_cfg=...)` replace the nine separate arguments. The Lightning module imports `LATENT_LOSS_TYPES` instead of keeping its own copy.
  - `main.py` resolves once per call site (`_run_learned_single`, the Tune trainable). The Tune W&B callback adds a per-trial `loss_signature`; prod and best-trial runs get it through the module hparams.
  - Notebook callers updated: `kraken_eval`, `crossmodal_pca_pls_learnable_overview`, `test_VAE`.
  - Check in `kraken_env` (old code loaded from `HEAD`): 45 cases (every learned YAML default plus each `loss_type` search choice) give the same class, identical outputs over 31 train/eval steps, and identical hparams apart from the added `loss_signature`.
  - Invalid configs (`balanced_composite`, unknown term, composite without terms) now fail at config time. `main.py --help` and the imports pass.

#### M3 — Searchable weights and term kwargs  ✅ done
- **Changes:**
  - `models/registry.py::_flat_to_nested` routes the `loss_weight_*` and `loss_kwarg_*` prefixes to `trainer` (`TRAINER_KEYS` is exact-match today).
  - `resolve_loss_config` applies the overrides from 8.2:
    - A weight or kwarg for a term not in `loss_terms` raises.
    - Weight 0 drops the term.
    - `loss_weight_mse` in a search space warns.
- **Accept:** a unit check round-trips `search_space_to_tune` → sample → `_flat_to_nested` → `resolve_loss_config` for a sample space, with and without Optuna. The best-trial rerun rebuilds the sampled weights. `loss_signature` matches the active terms.
- **Result:**
  - `models/registry.py` owns `LOSS_WEIGHT_PREFIX` / `LOSS_KWARG_PREFIX` and `is_trainer_key()`. `main._flat_to_nested` routes by `is_trainer_key`, so `loss_weight_*` / `loss_kwarg_*` land in `trainer` and never reach model constructors.
  - `resolve_loss_config` applies the overrides:
    - weight 0 drops the term; an unlisted term, a malformed kwarg key or an all-zero set raises; `loss_weight_mse` warns;
    - weights are ignored for non-composite `loss_type`s, since `loss_type` is searched alongside them;
    - `loss_terms` is rewritten only when an override or a drop applies, so logged terms keep their original form otherwise.
  - `loss_normalize: auto` (`none` for one active term, `ema` otherwise) is implemented here. The default stays `ema` until M8 flips the YAML defaults.
  - Checks in `kraken_env`: 18 unit checks; 200 random Tune samples and 50 Optuna suggestions round-trip (routing, weights, drops, normalize); regression against `HEAD`: 45/45 configs identical.

#### M4 — Fold legacy losses into composite terms  ✅ done
- **Changes:**
  - Add three terms to `CompositeLoss`:
    - `demeaned_mse`: target train mean from `base`, which is already passed to `create_loss_fn`.
    - `pairwise_corr`: `|mean_pair_corr(y_pred) − corr_target|`, kwarg `corr_target`.
    - `kld`: consumes the `mu`/`logvar` the Lightning module already passes; raises if absent.
  - Delete `WeightedMSELoss`, `SarwarMSECorrLoss` and `VAELoss` (`MSELoss` and `loss_type: mse` are retired in M8, once every caller has moved), and the `loss_alpha` / `loss_beta` / `loss_corr_target` / `loss_corr_weight` keys in `TRAINER_KEYS`, `main.py`, the trainer and the Lightning module.
  - Migrate the YAMLs:
    - **`Sarwar2020MLP.yml`:** terms `[mse: 1, pairwise_corr: 1e-3 {corr_target: 0.4}]`, `none`. Search `loss_weight_pairwise_corr` loguniform [1e-6, 1e-3] and `loss_kwarg_pairwise_corr__corr_target` uniform [0.3, 0.9], the current ranges.
    - **`CrossModalVAE.yml`:** terms `[mse: 1, kld: 1.0]`, `none`. Search `loss_weight_kld` uniform [0.1, 2.0], the current `loss_beta` range. Drop the unused `loss_alpha`.
- **Accept:** on random tensors, the new composite equals each old loss within 1e-6:
  - `sarwar_mse_corr` equals `[mse, pairwise_corr]` under `none`.
  - `vae` equals `[mse, kld]` under `none`.
  - `weighted_mse(α)` equals `[mse: α, demeaned_mse: 1−α]` under `ema` with warmup 1, including after the first step.
  - No references to `weighted_mse` (non-latent), `sarwar_mse_corr` or `vae` loss types remain in `models/`, `main.py`, `scripts/` or the YAMLs.
- **Result:**
  - New `CompositeLoss` terms: `demeaned_mse` (target mean via `base`), `pairwise_corr` (kwarg `corr_target`), `kld` (uses `mu`/`logvar`; raises without them). `WeightedMSELoss`, `SarwarMSECorrLoss` and `VAELoss`, their loss types, and the `loss_alpha` / `loss_beta` / `loss_corr_*` keys are removed from `loss.py`, `TRAINER_KEYS` and the Lightning hparams.
  - YAMLs: `Sarwar2020MLP` → `[mse, pairwise_corr 1e-3 {corr_target 0.4}]`, `none`; search keys `loss_weight_pairwise_corr` / `loss_kwarg_pairwise_corr__corr_target`. `CrossModalVAE` → `[mse, kld 1.0]`, `none`; search key `loss_weight_kld`. Dead `loss_alpha`/`loss_beta` dropped from `CrossModal_PCA_PLS_CovProjector_SC+SC_r2t.yml`; once removed from `TRAINER_KEYS` they would have been routed to the model constructor.
  - Notebooks: `test_sarwar2020_model` cells 2–3 migrated.
  - **Pre-existing bug fixed in `test_VAE`:** cells 22/24/26 passed `loss_fn='vae', beta=…, lr=…, epochs=…` to the `CrossModalVAE` constructor, which swallows them via `**kwargs`. The following `train_model` calls passed nothing, so those runs trained with plain MSE, no KLD, and `lr=1e-4` (cell 26 intended 1e-3). `lr`, `max_epochs` and a `[mse, kld: β]` loss now go to `train_model`.
  - Checks in `kraken_env`, old classes from `HEAD`: **bit-exact (max |diff| 0), including gradients**, over 12 train steps plus eval, for Sarwar (4 points in the search range), VAE (4 β), and weighted MSE (3 α, with and without eval-before-train) as `[mse: α, demeaned_mse: 1−α]` under `ema` with warmup 1. The Sarwar/VAE YAML defaults and remapped search keys are bit-exact too. Retired types are rejected. The regression over the other 43 configs is identical.

#### M5 — Unify L1/L2 regularization keys  ✅ code done · model-level check in the verification array (M8)
- **Changes:**
  - Models on scalar `reg` switch to `l1_l2_tuple` via `compute_reg_loss(self._reg_params(), self.l1_l2_tuple)`, keeping their current parameter selection: `LatentAttnMasked`, `NodalGNN`, `NodalMLP`, `MaskedLatentPretrainer`, `MaskedMLPPretrainer`.
  - `Chen2024GCN`: `l2_reg` drives its existing plain-norm penalty; `l1_reg > 0` raises (8.2).
  - YAMLs: `reg` → `l2_reg` in `default` and `search_space`, same values and ranges. Add `l1_reg: 0.0` defaults. Search `l1_reg` only where it is searched today: the PCA/PLS family and `Sarwar2020MLP` (8.5 D3). Every other model is L2-only (`l1_reg: 0`, not searched).
- **Accept:**
  - With `l2_reg` equal to the old `reg` and `l1_reg: 0`, each migrated model's `get_reg_loss()` matches the old value on the same weights (a one-batch check, CPU where possible, SLURM otherwise).
  - No scalar `reg` keys remain in `models/configs/`.
  - The scraper is unaffected (it reads neither key).
- **Result (code):**
  - `LatentAttnMasked`, `NodalGNN`, `NodalMLP`, `MaskedLatentPretrainer` and `MaskedMLPPretrainer` take `l1_l2_tuple` (default `(0, 1e-4)`, same as the old `reg` default). They pass it straight to `compute_reg_loss`; parameter selection is unchanged.
  - `Chen2024GCN` keeps its plain-norm penalty driven by `l2_reg` and raises on `l1_reg > 0`.
  - 15 YAMLs: `reg: X` → `l1_reg: 0.0` + `l2_reg: X`; search `reg` → `l2_reg` with the same ranges. No `l1_reg` search added (D3).
  - 40 notebook config keys in 14 notebooks: `"reg"` → `"l2_reg"`.
  - `cov_projector_benchmark/records.json` keeps `reg`: it is recorded W&B history.
  - Static check (`kraken_env`): for every learned YAML, default and search-space model keys (after `build_model`'s `l1_reg`/`l2_reg` → `l1_l2_tuple` mapping) are all accepted by the class constructor, with no keys silently swallowed by `**kwargs`.
- **Environment finding:** `torch_geometric` is not installed in the current `kraken_env` overlay (torch 2.9.0), so `Chen2024GCN` / `NodalGNN` cannot import, and their launchers would fail. Their model-level checks are skipped until PyG is reinstalled; that needs a `:rw` overlay mount by the user.

#### M5b — Sampled L1/L2 silently discarded in sweeps  ✅ done
- **Bug (predates this track):** `build_model` did `kwargs.setdefault("l1_l2_tuple", (l1_reg, l2_reg))`. A Tune trial merges the sampled `l1_reg`/`l2_reg` over a default that carries `l1_l2_tuple`, so the default tuple always won.
- **Affected sweeps:** every sweep of `CrossModal_PCA_PLS_learnable` (all 3 source variants), `CrossModal_PCA_PLS_CovProjector` and `Sarwar2020MLP`.
  - All trials trained with the YAML default: L2 = 1e-4 for the PCA/PLS models, no regularization for Sarwar. W&B nonetheless logged the sampled `l1_reg`/`l2_reg`.
  - Reproduced by replaying the Tune flow (`default_flat` + sample → `_flat_to_nested` → `build_model`) with a stub model.
  - Separately, `CrossModalVAE.yml` and `CrossModal_PCA_PLS_CovProjector_SC+SC_r2t.yml` declared their L1/L2 search as an `l1_l2_tuple` of type `quniform`. `search_space_to_tune` silently dropped unsupported types, so those models never searched L1/L2 either.
- **Fix:**
  - `build_model`: explicit `l1_reg`/`l2_reg` override `l1_l2_tuple`.
  - 13 YAML defaults: `l1_l2_tuple: [a, b]` → `l1_reg: a` / `l2_reg: b`. The two `quniform` entries became the grid they describe (0–1e-3, step 1e-4) as `l1_reg`/`l2_reg` choices.
  - `search_space_to_tune` now raises on unsupported types.
  - `test_proj_model` cell 9 override converted as well; left as it was, the YAML defaults would now override it.
- **Accept (met):** the replay now delivers the sampled values for all three models. The config↔constructor check has 0 failures. Every search type in the YAMLs is supported (choice 254, loguniform 52, uniform 15, grid 2).
- **Impact:** past tuned results for these models reflect the default regularization, not the searched values. Treat their logged `l1_reg`/`l2_reg` as not applied.

#### M6 — Per-term losses in Tune trial runs  ✅ code done · live 2-trial check in the verification array (M8)
- **Changes:** Tune trial W&B runs currently receive only the 6 metrics in `tune_metrics` (`main.py`); the Tune trainer runs with `logger=False`. Add `train/val_loss_raw_*`, `val_loss_weighted_*` and `val_loss_ref_*` for the active terms, built from the resolved `loss_cfg`.
- **Accept:** a short SLURM tune run (2 trials, few epochs) shows per-term curves on the trials' W&B runs.
- **Result (code):**
  - `structured_loss_metric_names(loss_cfg, phases, kinds)` in `lightning_module.py` sits next to the logging it mirrors, and returns names only for active composite terms.
  - The Tune trainable adds `train_loss_raw_*` and `val_loss_{raw,weighted,ref}_*` to `tune_metrics`. Ray only warns on a missing metric (checked in ray 2.54.1), so nothing can break a trial.
  - CPU check: a 2-epoch Lightning fit with `mse + 0.5·neidist` logs every generated name.

#### M7 — Fold the linear backbone into the PCA/PLS family as `CrossModal_linear_backbone`  ✅ done (real-data check in the verification array)
- **M7a — Equivalence map.**
  - Target: `LatentAttnMasked(residual_mode="none")` ≡ `CrossModal_PCA_PLS_learnable(learn_encoder=False, learn_decoder=False, random_init=True, dropout=0, n_components_pca_source=n_components_pca_target=k)`, plus the flags added here.
  - Known gaps to close with flags on `_learnable`:
    - `mid_bias` (the backbone's latent map has a bias; `W_mid` has none).
    - `zscore_pca_scores` (latent z-scoring before and after the map).
    - Init: the backbone uses `nn.Linear` init; `_learnable` uses `random_init`. Match or document.
  - `_reg_params`: the backbone regularizes the map only; `_learnable` includes the encoders and decoder. Frozen parameters are skipped by `compute_reg_loss`, so these should match; verify.
  - Test: copy weights across and compare outputs on one batch (tolerance 1e-5).
- **M7b — Build.**
  - Add the flags to `_learnable`. Defaults reproduce the current `_learnable` exactly, so existing YAMLs and results are unchanged.
  - Add `class CrossModal_linear_backbone(CrossModal_PCA_PLS_learnable)` in `models/architectures/crossmodal_pca_pls.py`. It pins `learn_encoder=False`, `learn_mid=True`, `learn_decoder=False`, `random_init=True`, `dropout=0`, `mid_bias=True`, and ties `n_components_pca_target` to `n_components_pca_source`. Register it in `build_model`.
  - Add `models/configs/CrossModal_linear_backbone.yml` (source `SC`):
    ```yaml
    default:
      model: {n_components_pca_source: 128, zscore_pca_scores: false, l1_reg: 0.0, l2_reg: 1.0e-6}
      trainer: {lr: 3.0e-4, max_epochs: 150, batch_size: 128,
                loss_type: composite, loss_normalize: auto,
                loss_terms: [{name: mse, weight: 1.0}, {name: varmatch, weight: 0.0},
                             {name: correye, weight: 0.0}, {name: neidist, weight: 0.0}]}
    search_space:
      n_components_pca_source: {type: choice, values: [64, 128, 256]}
      zscore_pca_scores:       {type: choice, values: [false, true]}
      l2_reg: {type: loguniform, lower: 1.0e-7, upper: 1.0e-3}
      l1_reg: {type: choice, values: [0.0, 1.0e-7, 1.0e-6, 1.0e-5]}
      lr:     {type: loguniform, lower: 1.0e-4, upper: 3.0e-3}
      max_epochs: {type: choice, values: [100, 150, 250]}
      loss_weight_neidist:  {type: choice, values: [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]}
      loss_weight_correye:  {type: choice, values: [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]}
      loss_weight_varmatch: {type: choice, values: [0.0, 0.1, 0.25, 0.5, 1.0]}
      loss_scale_warmup_steps: {type: choice, values: [20, 50, 100]}
      loss_scale_ema_decay:    {type: choice, values: [0.9, 0.95, 0.99]}
    ```
    `batch_size` is fixed, not searched: `correye`/`neidist` depend on batch size, and M10 tests that separately. The ranges span the values the backbone notebooks used (128/256 components, reg 1e-6–1e-7, lr 1e-4–3e-4, 100–250 epochs).
- **M7c — Remove the overlap from `LatentAttnMasked`.**
  - `residual_mode: none` is removed; the value raises, and it is dropped from the `LatentAttnMasked.yml` search space.
  - Token-embedding, attention and readout modules are built only when the attention branch is active.
  - `_reg_params()` returns only parameters that are used.
  - `linear_residual` keeps its learned linear backbone; sharing the exact module with `_learnable` is optional and not required for acceptance.
- **M7d — Update callers** to `Sim(model_name="CrossModal_linear_backbone", ...)`:
  - `scripts/notebooks/model_overviews/linear_backbone_overview.ipynb`
  - `scripts/notebooks/model_testing/test_loss_linear_model.ipynb`
  - `scripts/experiments/linear_backbone_geodesic/`
- **Accept:**
  - With copied weights, `CrossModal_linear_backbone` reproduces `LatentAttnMasked(residual_mode="none")` (checked before M7c removes the mode).
  - Existing `_learnable` YAMLs build identical models (parameter shapes, `requires_grad`, one-batch output with fixed seed).
  - `LatentAttnMasked` builds and runs a forward pass for each remaining `residual_mode`.
  - The notebook import check passes.
- **Result:**
  - `CrossModal_PCA_PLS_learnable` gains `mid_bias` and `zscore_pca_scores`; defaults reproduce the old model exactly. The PLS fit is skipped when `W_mid` is random and learned (its result was unused). The duplicated `target_latent_encoder` / `latent_loss_weights` buffer registration is removed.
  - `CrossModal_linear_backbone` (thin subclass) pins the backbone flags, ties the target latent size to the source, and rejects pinned keys. It is registered in `build_model`, with `models/configs/CrossModal_linear_backbone.yml` as in M7b.
  - `LatentAttnMasked`: `residual_mode: none` raises with a pointer to the new model. Its dead branches (`predict_target_latents`, `inspect_attention_state`, the reg guards) are removed; `none` is dropped from the YAML search.
  - Notebooks migrated (JSON-valid, code parses, not executed): `linear_backbone_overview` (stale outputs cleared; the Conditional Gaussian comparison helper now takes `base` explicitly and plots `W_mid.T`), `test_loss_linear_model`, `latent_masked_test` cell 3 (`Sim` default → `attention_only`; its run already used it), and `linear_backbone_geodesic_metrics`. The last keeps its recorded outputs plus a record note, since the notebook is the experiment record. `experiments_index.md` updated.
  - Cross-tree checks on a synthetic base (`kraken_env`, CPU):
    - **M7a** `LatentAttnMasked(none)` → `CrossModal_linear_backbone` with copied weights: edge outputs **bit-identical** at k = 16, 32 and with z-scoring; only `W_mid` and `mid_bias` trainable.
    - Existing `_learnable` configs: identical state, outputs, reg and trainable sets (3 configs).
    - **M5** reg equality for `LatentAttnMasked` (pls/linear residual), `MaskedMLPPretrainer`, `MaskedLatentPretrainer`.
    - Remaining residual modes run a forward pass.
  - **Documented difference:** with `zscore_pca_scores: true`, `predict_target_latents` now returns PCA-space target latents; `LatentAttnMasked` returned z-space latents. Edge outputs are unchanged; latent losses and latent diagnostics under z-scoring are measured in PCA space.
- **Notebook issues noticed, left for the user:**
  - `latent_masked_test` cell 6 reads `residual_linear.weight`, which is absent in the `attention_only` mode its run uses.
  - Cell 4 sets `"l2_reg"` twice (0.25, then 1e-7); this predates M5.

#### M8 — Wire composite into every in-scope YAML  ✅ code done · real-data dev runs in the verification array
- **Changes:** for each model in 8.1:
  - `trainer.loss_type: composite` and `loss_normalize: auto`.
  - `loss_terms`: `mse: 1.0` plus the candidate terms `varmatch`, `correye` and `neidist` at weight 0. For `NodalMLP`: `mse` only, with no weight search (8.5 D4).
  - `search_space`: `loss_weight_*` for the candidates (8.2 ranges). `loss_type` search choices drop `mse`; the latent choices stay where present.
  - `LatentAttnMasked.yml`: default moves to MSE-only; the four terms are searched (8.5 D1).
  - Notebook config overrides that set `"loss_type": "mse"` (e.g. `linear_backbone_overview` cell 5) → composite MSE-only.
  - Then delete `MSELoss` and the `mse` branch of `create_loss_fn`.
- **Accept:**
  - Every in-scope YAML resolves to the MSE-only default: `loss_signature: "mse"`, and the loss equals `F.mse_loss` exactly.
  - No `loss_type: mse` remains in YAMLs, launchers or notebook sources (saved cell outputs excluded).
- **Result (code):**
  - `loss.py`: `composite` is the only edge-space loss type. `loss_type` defaults to `composite`, `loss_terms` defaults to `["mse"]`, `loss_normalize` defaults to `auto`. `MSELoss` is deleted; `loss_type: mse` raises with a pointer to composite.
  - 21 YAMLs:
    - `_learnable` (3), `CovProjector` (8), `Chen2024GCN`, `NodalGNN`: `[mse 1, varmatch 0, correye 0, neidist 0]`, `auto`, with the 8.2 weight / warmup / decay search added.
    - `NodalMLP` (7): `[mse]` only, no weight search (D4).
    - `LatentAttnMasked`: default moved to MSE-only (D1); weights searched; `loss_type` choices `mse` → `composite`.
  - **Sarwar2020MLP and CrossModalVAE keep their own objectives under `none`, with no identity-term candidates.** Adding `correye`/`neidist` would switch them to `ema` and change what their paper weight / β mean.
  - Notebooks: 21 live `"loss_type": "mse"` overrides in 13 notebooks → `"composite"`, which resolves to plain MSE.
  - Checks (`kraken_env`):
    - all 20 migrated YAMLs resolve to `loss_signature: "mse"` and are **bit-exact, gradients included**, against the old MSE loss built from the old YAMLs;
    - `loss_type: mse` is rejected; an empty trainer config gives plain MSE;
    - 40 Optuna samples on each of `_learnable`, `CrossModal_linear_backbone`, `LatentAttnMasked` and `Chen2024GCN` resolve correctly, and no `loss_*` key reaches a model constructor;
    - config↔constructor check: 0 failures.
- **Verification array:** `scripts/sbatch/checks/verify_modeling_track_array.sh` with `verify_modeling_track.py`, one task per index:
  - task 0, `dev_runs` (M8): a 2-epoch dev run per model family; checks the signature, per-term logging, `val_loss == Σ weighted terms + reg`, and finite test metrics;
  - task 1, `cross_tree` (M7a + M5 on real data, old code via `git archive` of `f74cc0e` / `5286526`);
  - task 2, `tune` (M6: 2-trial Tune, W&B offline).

  Reports go to `results/logs/verify_modeling_track_<task>.json`. `Chen2024GCN`/`NodalGNN` are skipped until `torch_geometric` is reinstalled.
  - One short dev run per model family, via SLURM, logs `*_loss_raw_mse` with train/val curves matching the previous `mse` runs at the same seed. Differences are limited to GPU nondeterminism.

#### M9 — Launchers and the weight-sweep experiment
- **Changes:**
  - Launchers `tune_array_linear_backbone_loss_weights_seeds.sh` and `run_array_linear_backbone_loss_checks_seeds.sh` under `scripts/sbatch/CrossModal_linear_backbone/`, copied from the existing `_learnable` array template; the loss settings come from the YAML.
  - `scripts/experiments/composite_loss_weights/` with `composite_loss_weights.md` (question, grid, how to run, W&B `ray_tune_id`s, outcome) and an entry in `scripts/experiments/experiments_index.md`.
- **Accept:** `bash -n` passes; the experiment doc and index entry exist.

#### M10 — Interaction tests
Run on `CrossModal_linear_backbone` (M7), source `SC`, selecting on `val_demeaned_r`. Resources follow the closest existing template, `scripts/sbatch/CrossModal_PCA_PLS_learnable/tune_array_pca_pls_learnable_SC_SCr2t_SCpSCr2t_seeds.sh`: 2 h, 64 GB, 1 GPU, Optuna, `MAX_CONCURRENT_TRIALS=1`, one array task per seed.
- **Stage A — weight × reg sweep** (`tune_array_linear_backbone_loss_weights_seeds.sh`):
  - `--array=0-2` (seeds 0–2), `--num_samples 32`.
  - Optuna over `loss_weight_{neidist,correye,varmatch}`, `l2_reg`, `lr`, under `ema`.
  - A full grid is too large (6×6×5 weights × reg ≈ 540 trials per seed); Optuna with 32 trials covers it.
  - Budget: 3 jobs × ≤2 h ≈ **≤6 GPU-h**.
- **Stage B — robustness of the Stage A winner** (`run_array_linear_backbone_loss_checks_seeds.sh`):
  - Direct prod runs, no tune: seeds 0–2 × {`ema`, `none` via the `*_loss_ref_*` conversion} × `batch_size` {64, 128} = 12 tasks, `--array=0-11`, ~30 min each.
  - Budget **≈6 GPU-h**.
  - Also checks that `*_loss_ref_*` stays flat after warmup.
- **Stage C — only if Stage A beats MSE-only on `val_demeaned_r` across seeds:** extend Stage A to seeds 0–9 (`--array=0-9`, ≈20 GPU-h) for the reported comparison, matching the 10-seed convention of the other arrays.
- **Total before Stage C:** ≈12 GPU-h, smaller than one existing 30-task `_learnable` array.
- **Accept:** each stage is submitted only after you approve it. Results, with W&B `ray_tune_id`s, are recorded in `composite_loss_weights.md`, along with a stated conclusion on whether `l2_reg` ranges must be set per normalize mode.

#### M11 — Docs
- **Changes:** `CONTEXT.md`:
  - Config System Notes: composite-only loss path, `loss_normalize: auto`, `loss_weight_*` / `loss_kwarg_*`, `loss_signature`, the `l1_reg`/`l2_reg` convention and the Chen exception.
  - Model Inventory: the separated linear backbone.
  - High-Risk Gotchas: reg interacts with the loss scale; `correye`/`neidist` depend on batch size; `kld` needs `mu`/`logvar`.

  Also `README.md` (models table, CLI/loss notes). Update `Last updated at` in both.
- **Accept:** no doc references to `balanced_composite`, `weighted_mse` (non-latent), `sarwar_mse_corr`, `loss_type: vae`/`mse`, the scalar `reg` key, or `residual_mode: none`.

### 8.4 Order and dependencies

M1 → M1b → M2 → M3 → M4 → M5 → M5b → M6 run in order; each is small and needs no data or GPU except the M6 check.
M7 can run any time after M2.
M8 needs M3–M5 and M7b. M9 needs M7b. M10 needs M8–M9 and approval for each stage. M11 comes last.

### 8.5 Decisions on the former open items (2026-09-23)

| # | Item | Decision |
|---|---|---|
| D1 | `LatentAttnMasked` default loss | MSE-only default, like every other model; the four terms are searched. |
| D2 | Linear backbone home | PCA/PLS family (`CrossModal_PCA_PLS_learnable` configuration) as the default. A standalone class only if M7a finds a gap that can't be closed with flags; report it if so. |
| D3 | `l1_reg` search | Only where searched today (PCA/PLS family, `Sarwar2020MLP`); L2-only elsewhere. |
| D4 | Identity terms (`correye`/`neidist`) for `NodalMLP` | Withheld until `NodalMLP` shows learning signal; MSE-only for now. |
| D5 | M10 compute | Staged budget in M10 (≈12 GPU-h before the optional Stage C); each stage approved before submission. |

Still open: none. The M7a result is reported, but it doesn't need a decision unless a gap can't be closed with flags.

### 8.6 Deferred (modeling)

- **Latent-space terms inside composite** (`latent_mse` / `latent_weighted_mse` as mixable terms for models exposing `encode_target_latents`). The `latent_*` loss types stay as they are for now.
- **YAML variant duplication** (8 `CovProjector`, 7 `NodalMLP`) and copied launchers: the §6 manifest item.

## 9. Change log

| Date | Change |
|---|---|
| 2026-09-23 | v1 drafted: groundwork G1–G7 recorded; plan T1–T8 defined. |
| 2026-09-23 | Open questions resolved; T2, T5, T6 simplified (no experiment READMEs; ledger is the single record). |
| 2026-09-23 | T2–T7 executed: `506e008`, `64dbc62`, `08a917c`, `9ed9d7f`, `fa88722`, T7 (no change). Remaining: T8 docs pass. |
| 2026-09-23 | T8 docs pass done; artifact cleanup: `results/wandb/` removed, `results/ray_tmp/` sessions before 2026-04-01 pruned (131 April sessions kept for debugging). Stage 1 complete. |
| 2026-09-23 | Post-v1: `scrape_covtype_results.ipynb` → `scripts/experiments/cov_projector_benchmark/` (both tables reproduced exactly); shared runner plumbing in `scripts/results_utils/runner.py`. |
| 2026-09-23 | Post-v1: experiment runners write `tables/`, `figures/`, `manifest.json` into the experiment folder. Figures are PNG-only (300 dpi, per the figure-making skill) and tracked; per-figure CSVs dropped as redundant with `tables/seed_records.csv`. |
| 2026-09-23 | Post-v1: `context_packages/experiment_ledger/` folded into experiment folders (`<slug>/<slug>.md`) + `scripts/experiments/experiments_index.md`; Adel summary → `scripts/experiments/adel_summer_2026/`; `linear_backbone_geodesic` documented by its notebook. T6's ledger file no longer exists. |
| 2026-09-23 | Modeling track §8 (M1–M11): composite-only loss path with searchable `loss_weight_*`, legacy losses folded into composite terms, unified `l1_reg`/`l2_reg`, linear backbone folded into the PCA/PLS family; decisions D1–D5; latent terms deferred. |
| 2026-09-23 | M1 done (`e094f3d`); M1b added and done: EMA scale uses `|raw|` so signed terms (`neidist`) no longer blow up. |
| 2026-09-23 | M2 done: single `loss_cfg` path via `resolve_loss_config`; `loss_signature` logged to W&B. |
| 2026-09-23 | Linear backbone named `CrossModal_linear_backbone` (thin `_learnable` subclass); its YAML and search space added to M7b. |
| 2026-09-23 | M3 done: searchable `loss_weight_*` / `loss_kwarg_*`; `loss_normalize: auto`. |
| 2026-09-23 | M4 done: legacy losses folded into composite terms (bit-exact); `test_VAE` KLD/lr bug fixed. |
| 2026-09-23 | M5 code done: `l1_reg`/`l2_reg` for every learned model; found `torch_geometric` missing from `kraken_env`. |
| 2026-09-23 | M6 code done: per-term composite losses reported to Tune. |
| 2026-09-23 | M5b: fixed sampled `l1_reg`/`l2_reg` being discarded in `_learnable`/`CovProjector`/`Sarwar` sweeps; YAML tuples → `l1_reg`/`l2_reg`; unsupported search types now raise. |
| 2026-09-23 | M7 done: `CrossModal_linear_backbone`; `LatentAttnMasked` `none` mode removed; bit-identical equivalence on a synthetic base. |
| 2026-09-23 | M8 code done: composite is the only edge-space loss (bit-exact MSE default); verification array added. |

Last updated at: 2026-09-23 EDT
