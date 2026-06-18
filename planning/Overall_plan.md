# Conn2Conn Reorganization And Reproducibility Plan

## Purpose

This plan has three phases:

1. Preserve and sync the current scientific state.
2. Physically reorganize `notebooks-FC_to_SC-experimental/` so the evidence is navigable.
3. Build a clean reproducibility/grid-search layer that reruns the core claims from scratch.

All work in this plan should happen locally on the laptop first. Torch/HPC is used only as
the compute target later, not as the place to do file reorganization.

## Phase 1 — Preserve Current State

Before moving files, commit and push the current evidence state.

Goals:

- Add/track meaningful HPC-derived artifacts that were synced back locally.
- Avoid tracking scheduler clutter such as `DONE_*.sentinel` and SLURM `.out` files unless a
  specific output log is itself part of the evidence trail.
- Keep `.claude/scheduled_tasks.lock` out of the scientific cleanup commit.
- Establish a clean git baseline before reorganizing paths.

Expected commit:

```text
sync: preserve HPC-derived FC-to-SC analysis artifacts
```

After this commit is pushed, it is acceptable to reorganize with normal local `mv` / `cp`
commands. There is no requirement to use `git mv`; git can detect renames after the fact.

## Phase 2 — Organize Existing FC-To-SC Workspace

The target is a cleaner physical layout under `notebooks-FC_to_SC-experimental/`, while
preserving all evidence and updating relative paths.

The reorganization should separate:

- main paper story
- mechanism / PC analyses
- sanity checks
- tractography representation tests
- nonlinear / residual / scaling checks
- future reproducibility suite

This phase is both file cleanup and intellectual cleanup. Findings should be labeled as
confirmatory, exploratory, stale, superseded, or needing rerun.

See [cleanup_plan.md](cleanup_plan.md).

## Phase 3 — Reproducibility And Grid Layer

After the current workspace is clean, create a new reproducible grid suite that reruns the
core claims from scratch with W&B tracking and local CSV mirrors.

The locked design:

- Three deterministic estimators:
  - PCA -> PLS
  - BayesianRidge
  - KernelRidge (RBF)
- `bv+demo` is an input/feature set, not a model.
- Reconstruction and downstream are separate claim-driven tables.
- No Krakencoder, no learnable PLS, no MLP, no CovProjector in this grid.
- 10 seeds x 2 parcellations (Glasser, 4S456Parcels — only 2 available, FC-capped) x reconstruction/downstream/leak-check tasks.
- All reconstruction metrics plus downstream cognition/behavior metrics.

See [reproducibility_and_grid_plan.md](reproducibility_and_grid_plan.md).

## Working Rule

Every added experiment row must answer a named claim in one sentence. If it does not,
it is clutter and should not enter the grid.

