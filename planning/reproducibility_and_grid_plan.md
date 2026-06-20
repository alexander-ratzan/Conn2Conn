# Reproducibility And Grid Plan

## Purpose

Create a clean, reproducible grid suite for the Conn2Conn FC-to-SC methods paper after
the current `notebooks-FC_to_SC-experimental/` workspace is organized.

The suite should rerun the core claims from scratch with:

- 10 seeds
- 2 parcellations (Glasser, 4S456Parcels — only 2 available; see STOP gate)
- deterministic estimators
- all reconstruction metrics
- downstream behavior/cognition metrics
- leak checks
- W&B tracking
- local CSV mirrors for every run

## Core Design Rule

Do not collapse input sets and estimators.

There are two orthogonal axes:

1. **Input / feature set**: what gets fed into the estimator.
2. **Estimator**: the mathematical mapping from input to output.

`bv+demo` is an input set, not a model. It must be evaluated with the same estimators as
connectome inputs whenever the comparison is meant to be apples-to-apples.

## Axis 1 — Input / Feature Sets

The input axis is claim-driven, not combinatorial. Every row must answer a named claim.

### Reconstruction Table

Output space: connectome target, usually `FC` or `SC`.

Rows:

| Input set | Target(s) | Claim |
|---|---|---|
| `FC` | `SC` | FC->SC side of the asymmetry |
| `SC` | `FC` | SC->FC side of the asymmetry |
| `bv` | `SC`, `FC` | anatomy axis / modality dissociation |
| `demo` | `SC`, `FC` | demographic axis / modality dissociation |
| `bv+demo` | `SC`, `FC` | full subject-info baseline |
| `connectome+bv+demo` | `SC`, `FC` | whether connectome adds over subject-info |
| `FC` | `FC` | within-modality FC oracle ceiling |
| `SC` | `SC` | within-modality SC oracle ceiling |

Notes:

- `FC->FC` and `SC->SC` are oracle/reference rows, not claims of cross-modal prediction.
- `connectome+bv+demo` means:
  - `FC+bv+demo -> SC`
  - `SC+bv+demo -> FC`

Explicitly cut:

```text
FC+bv
SC+bv
SC+bv+demo+SC
```

`bv` without `demo` is useful only as the anatomy-axis probe. The full baseline is
`bv+demo`.

### Downstream Table

Output space: behavior/cognition target.

Targets:

```text
CogTotal
CogFluid
CogCryst
sex
age
```

Rows:

| Input set | Claim |
|---|---|
| `bv+demo` | baseline / bar to beat |
| `obs_FC` | FC is the cognition workhorse |
| `obs_SC` | SC underperforms the baseline |
| `obs_FC+obs_SC` | whether observed fusion helps |
| `pred_SC` | utility of SC imputed from FC |
| `pred_FC` | utility of FC imputed from SC |
| `connectome+bv+demo` | whether connectome adds downstream over subject-info |

Notes:

- `pred_SC` and `pred_FC` should come from the reconstruction grid or a clearly paired
  imputation step.
- Keep downstream separate from reconstruction. Do not merge the tables.

## Axis 2 — Estimators

The three estimators are locked:

| Estimator | Role |
|---|---|
| PCA -> PLS | protagonist; deterministic closed-form cross-modal baseline |
| BayesianRidge | cheap second linear estimator; regularized per-component check |
| KernelRidge (RBF) | nonlinear deterministic estimator; closes linear-artifact objection |

These span:

- linear projection
- linear regularized regression
- nonlinear kernel regression

All are deterministic and sweepable.

## Explicitly Not In This Grid

Do not add:

- Krakencoder
- learnable / end-to-end trained PLS
- MLP
- CovProjector

Reasons:

- Krakencoder is prior art and external/heavy; cite it, do not make it a grid row.
- Learnable PLS reintroduces training nondeterminism and weakens the closed-form thesis.
- MLP is replaced by KernelRidge for the nonlinear check.
- CovProjector answers a performance question, while this grid answers a robustness /
  baseline / ceiling question.

## Other Grid Axes

Required:

```text
10 seeds
2 parcellations  (Glasser 360, 4S456Parcels 456) — RESOLVED, see below
task_type in {reconstruction, downstream, leak_check}
```

> ✅ **STOP GATE RESOLVED (2026-06, parcellation count): only 2 parcellations are
> available, and FC is the hard cap.** Verified across (a) local atlas labels, (b) the
> precomputed SC/FC/node-feature caches, and (c) the **raw HCP source**: the xcpd
> functional output contains only `seg-Glasser` and `seg-4S456Parcels`. Cross-modal work
> needs matching FC+SC at the same parcellation, so FC caps the grid at **2**. A third
> parcellation would require re-running xcpd on the fMRI with an added atlas
> (Schaefer/Gordon/another 4S resolution) — a multi-day processing job, out of scope.
> **Decision: lock the grid at 2 parcellations** (Glasser 360 = classical anatomical;
> 4S456 = 456-region multi-resolution AtlasPack). Frame the claim as "replicates across
> two parcellations," which already kills the single-atlas-artifact objection (the noise
> module confirmed clean cross-parcellation replication). Revisit only if a third atlas is
> ever processed through xcpd.

## Metrics

### Reconstruction Metrics

Log all six:

```text
mse
r2
pearson
demeaned_pearson
top1_acc
avg_rank
```

Where applicable, also compute ratios or paired differences for the asymmetry summaries.

### Downstream Metrics

For behavior/cognition:

```text
pearson
spearman
r2
lift_over_bvdemo
residualized_score_against_bvdemo
```

For sex/age leak checks:

```text
sex_accuracy_or_auc
age_pearson
age_r2
```

Leak guardrails should flag unexpectedly high demographic predictability from derived
inputs.

## KernelRidge Hyperparameter Discipline

Keep the sub-grid modest. A small 3 x 3 over bandwidth/gamma and alpha is enough.

The goal is not exhaustive tuning. The goal is to show robustness or flatness across
reasonable deterministic choices.

## W&B Logging

Every run should log:

```text
seed
parcellation
estimator
input_set
source
target
task_type
metrics/*
artifact_paths
git_commit
data_load_mode
config_hash
```

Suggested W&B project:

```text
conn2conn-fc-to-sc-reproduction
```

Suggested tags:

```text
reproduction_2026_06
reconstruction
downstream
leak_check
parcellation:<name>
estimator:<name>
input:<input_set>
seed:<seed>
```

Every W&B run should also write a local CSV mirror so the repo remains analyzable without
W&B.

**Decision: run W&B OFFLINE.** No API key is configured on Torch (`wandb` 0.26.1 is
installed, no netrc). Use `WANDB_MODE=offline` — runs log to a local `wandb/` dir and can be
`wandb sync`-ed later if/when a key is attached. The **CSV mirrors are the source of truth**;
W&B is a convenience layer, never a dependency. The grid must produce identical CSVs whether
or not W&B is ever synced.

## Proposed Directory

After cleanup, create:

```text
notebooks-FC_to_SC-experimental/05_reproduction/
  README.md
  configs/
    grid.yml
  scripts/
    run_reconstruction_grid.py
    run_downstream_grid.py
    run_leak_checks.py
    summarize_reconstruction.py
    summarize_downstream.py
  outputs/
  reports/
    reproduction_findings.md
```

## Low-dim input handling (lesson from noise-module E)

Narrow inputs (`bv` 16-dim, `demo` ~9-dim, `bv+demo` ~26-dim) **cannot** go through a fixed
PCA(256). Every estimator call must cap components to the input width:
`k_src = min(256, X_train.shape[1])`, `k_pls = min(64, k_src)`. Bake this into the runner so
the baseline rows don't crash (they did once in the noise module's cross-modal step).

## Pipeline Ordering & Reconstruction→Downstream Handoff Contract

**This is a pipeline with a hard ordering, not two parallel grids.** The downstream table's
`pred_SC` / `pred_FC` rows are **products of the reconstruction grid**, not loadable cache
inputs. Downstream cannot run until reconstruction has written its imputed connectomes. Three
failure modes if the seam stays implicit: (1) run-order failure (downstream finds no
artifacts), (2) train/test inconsistency (imputations generated differently for train vs
test → subtle leak), (3) leak reintroduction (derived inputs that re-add bv+demo predict
sex/age trivially — the leak already caught once). The contract makes the handoff explicit
and assertion-guarded.

**Ordering (hard gate):** `run_reconstruction_grid.py` → `run_downstream_grid.py` →
`run_leak_checks.py` → `summarize_*`. Downstream asserts reconstruction artifacts exist for
the matching `(parcellation, seed)` and **fails loudly** otherwise.

**Imputation generation rule (consistency-critical):** imputation estimator is **fixed to
PCA→PLS** (NOT the swept estimator — the imputed connectome is a fixed derived input, not a
swept object). Train and test generated the *same way*:
- `pred_SC_test  = pca_pls_predict(FC_train, FC_test,  SC_train)`
- `pred_SC_train = pca_pls_predict(FC_train, FC_train, SC_train)`  (naive in-sample; optimism constant across rows, does not bias the comparison)
- `pred_FC_test  = pca_pls_predict(SC_train, SC_test,  FC_train)`
- `pred_FC_train = pca_pls_predict(SC_train, SC_train, FC_train)`

**Artifacts reconstruction MUST write (per `parcellation × seed`):**

| Artifact | Contents | Shape | Generated by |
|---|---|---|---|
| `pred_SC_test` | SC imputed from FC, test | (n_test, n_edges) | `pca_pls_predict(FC_tr, FC_te, SC_tr)` |
| `pred_SC_train` | SC imputed from FC, train | (n_train, n_edges) | `pca_pls_predict(FC_tr, FC_tr, SC_tr)` |
| `pred_FC_test` | FC imputed from SC, test | (n_test, n_edges) | `pca_pls_predict(SC_tr, SC_te, FC_tr)` |
| `pred_FC_train` | FC imputed from SC, train | (n_train, n_edges) | `pca_pls_predict(SC_tr, SC_tr, FC_tr)` |
| `recon_per_subject` | per-subject achieved, all 6 metrics, both directions | (n_subjects, n_metrics) | reconstruction scoring |
| `split_index` | train/test subject IDs for this seed | index | `seed` |

**Handoff keys / invariants:**

| Field | Rule |
|---|---|
| key | `(parcellation, seed)` on every artifact |
| split | re-derived from `seed`; downstream asserts loaded artifact's subject index == its own split |
| imputation estimator | fixed PCA→PLS (not the swept estimator) |
| train generation | same call as test (`X_tr → X_tr`), naive in-sample, constant across rows |
| format | per-subject rows preserved; **no seed-mean collapse at write time** (aggregation is in summarize) |
| provenance | `git_commit` / `config_hash` logged on every artifact |

**Per-subject emission:** reconstruction writes per-subject achieved scores for every metric
(not seed-means) so the noise-module per-subject panels regenerate from the grid for free.

**Leak guardrail (runs in downstream/leak-check, after derived inputs exist):** any downstream
input predicting `sex` > 0.99 balanced-acc or `age` > 0.85 Pearson **hard-fails** the run.
Derived inputs may only *remove* bv+demo (residualization) or stay raw — never add the bv+demo
OLS term back. `combined_pred_*` inputs that legitimately contain bv+demo are **exempt but
flagged** "contains subject-info, interpret cognition columns only."

**Run order with gates:**

| Step | Runner | Hard precondition | Produces |
|---|---|---|---|
| 1 | `run_reconstruction_grid.py` | connectome cache present | recon metrics + 6 handoff artifacts |
| 2 | `run_downstream_grid.py` | **assert** 4 `pred_*` + `split_index` exist for `(parc, seed)` | downstream scores incl. imputation rows |
| 3 | `run_leak_checks.py` | downstream inputs materialized | leak verdict; hard-fail on sex/age over threshold |
| 4 | `summarize_*` | steps 1–3 complete | seed-mean aggregation; metric-ordered reports |

## Metric Reporting Order (summaries lead with load-bearing numbers)

| Task | Primary | Secondary | Reported-but-caveated |
|---|---|---|---|
| reconstruction | **demeaned_r** | avg_rank | top1_acc (noisy); mse/r2/pearson (mean-dominated; r2<0 cross-modal expected) |
| downstream | **lift_over_bvdemo + paired permutation p** | residualized_vs_bvdemo | raw pearson/spearman; sex/age (leak-check only, not results) |

(Marginal CIs mislead for "beats baseline" — use the **paired** permutation test.)

## Implementation Order

1. Verify available parcellations and cache paths. ✅ (2: Glasser, 4S456 — STOP gate resolved)
2. Write `grid.yml` with explicit input-set rows and estimator rows (+ low-dim caps).
3. Implement reconstruction runner — writes metrics + the 6 handoff artifacts, per-subject.
4. Implement downstream runner — **asserts** the handoff artifacts + split match before running.
5. Implement leak checks — hard-fail thresholds; `combined_pred_*` exemption+flag.
6. Add offline W&B logging + local CSV mirrors (CSV is source of truth).
7. Run a one-seed / one-parcellation **smoke test** via sbatch (never login node).
8. Launch full grid on Torch only after smoke test passes.
9. Summarize into `reports/reproduction_findings.md` (metric-reporting-order above).

## Governing Principle

For any future addition:

```text
State the claim it answers in one sentence, or do not add it.
```

Estimators earn their place by the robustness question they close. Inputs earn their
place by the finding they isolate.

