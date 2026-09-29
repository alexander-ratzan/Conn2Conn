# Repo Spec v1 — `scripts/` Build-Out (Stage 1 Code Refactor)

**Status:** complete (T1–T8 done) · **Started:** 2026-09-23 · **Scope:** Conn2Conn repo layout, `scripts/` umbrella
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

## 8. Change log

| Date | Change |
|---|---|
| 2026-09-23 | v1 drafted: groundwork G1–G7 recorded; plan T1–T8 defined. |
| 2026-09-23 | Open questions resolved; T2, T5, T6 simplified (no experiment READMEs; ledger is the single record). |
| 2026-09-23 | T2–T7 executed: `506e008`, `64dbc62`, `08a917c`, `9ed9d7f`, `fa88722`, T7 (no change). Remaining: T8 docs pass. |
| 2026-09-23 | T8 docs pass done; artifact cleanup: `results/wandb/` removed, `results/ray_tmp/` sessions before 2026-04-01 pruned (131 April sessions kept for debugging). Stage 1 complete. |
| 2026-09-23 | Post-v1: `scrape_covtype_results.ipynb` → `scripts/experiments/cov_projector_benchmark/` (both tables reproduced exactly); shared runner plumbing in `scripts/results_utils/runner.py`. |
| 2026-09-23 | Post-v1: experiment runners write `tables/`, `figures/`, `manifest.json` into the experiment folder. Figures are PNG-only (300 dpi, per the figure-making skill) and tracked; per-figure CSVs dropped as redundant with `tables/seed_records.csv`. |
| 2026-09-23 | Post-v1: `context_packages/experiment_ledger/` folded into experiment folders (`<slug>/<slug>.md`) + `scripts/experiments/experiments_index.md`; Adel summary → `scripts/experiments/adel_summer_2026/`; `linear_backbone_geodesic` documented by its notebook. T6's ledger file no longer exists. |

Last updated at: 2026-09-23 14:25 EDT
