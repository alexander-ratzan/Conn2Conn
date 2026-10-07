# E2: Joint task-FC → SC — do the task-FC blocks add to rest FC?

**Spec:** v3 E2 (E2.1 raw, E2.2 predictive residual), infrastructure v3 I5.
**Code:** this folder: `config.yml` (grid, versions, seeds), `grid.py` (CPU grid runner: one cohort per seed, every
`k_rest × k_task × k_sc` cell, one PLS fit truncated to every `n_pls`, metrics as the Evaluator), `launch_grid.sh`
(CPU array, seed × version), `run.py` (selection, tables, figures), `checks/` (`bench_pls_grid` feasibility,
`check_i5` acceptance, `fingerprint_fc2sc` before / after I5). Library: `FC_<task>` source modalities
(`data/hcp_dataset.py`), `CrossModal_PCA_PLS` (raw, per-block dict), `CrossModal_PCA_PLS_TaskResid` and
`fit_rest_residualizer` (`models/architectures/crossmodal_pca_pls.py`).
**Outputs:** `runs/<version>/seed<S>.csv` (every configuration, val + test metrics, block VIP shares);
`results/Glasser/{raw,predictive_resid}/` (selected configs, summary, heatmap, weave) and `results/Glasser/compare/`
(ablation curve, dimensionality, paired Δ, block contribution, selected `k_sc`).
**Status:** infrastructure built 2026-10-07; grid not yet run.

---

## 1. Question

For FC → SC, does adding the seven task-FC blocks to rest FC improve SC prediction over rest alone, and does that
hold for the part of each subject's task FC that their own rest FC cannot predict?

## 2. Setup

- Closed-form multi-block PCA-PLS: one train-split PCA per condition (rest + 7 tasks), scores concatenated (each
  column standardized), PLS to the SC PCs, decoded through the SC PCA.
- **raw** (E2.1): task block = PCA of the task FC. **predictive residual** (E2.2): per task, a ridge fit on the
  train split (penalty by GCV) predicts the subject's task FC from their own `k_rest` rest scores; the block is the PCA
  of the prediction error.
- Grid: `k_rest` 64–512, `k_task` 0 (rest only) – 64, `k_sc` 16–256, `n_pls` 4–128 (spec v3 E2 has the reasons).
  Selection per seed on val demeaned r **within each `k_task` slice**; test reported.
- FC → SC, Glasser, E0's matched 917-subject cohort and splits, seeds 0–4. CPU only.

## 3. Results

(pending)

## 4. Caveats

- Task blocks differ in scan length, so a task's added value is partly data quantity.

Last updated at: 2026-10-07 EDT
