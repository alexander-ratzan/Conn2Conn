# Notebook architecture — `crossmodal_pca_pls_closed_form_overview.ipynb`

Cell-by-cell map of the FC→SC experimental notebook
(`notebooks-FC_to_SC-experimental/model_overviews/crossmodal_pca_pls_closed_form_overview.ipynb`)
as of end-of-session 2026-05-26.

Companion to [`session_log_2026-05-26.md`](session_log_2026-05-26.md) and
[`metrics_glossary.md`](metrics_glossary.md).

VS Code doesn't show cell numbers, so cells are identified by their first line.

---

## High-level structure

The notebook has 33 cells in 5 layers:

```
Layer 1: Setup (cells 0-12)              [project model demos + direction toggle]
Layer 2: Phase 0 (cells 13-19)           [SC->SC oracle, 10-seed FC->SC robustness]
Layer 3: Anatomy stress test (cells 20-23) [brain-vol baseline + residualization]
Layer 4: Uncertainty + variance (cells 24-28) [bootstrap, permutation, partition]
Layer 5: Controlled comparisons (cells 29-32) [2x2 + cross-step summary]
```

The newest work (full-panel retrofit added today) lives in Layer 3 and Layer 5.

---

## Layer 1 — Setup (cells 0-12, mostly unchanged)

| first line of cell | what it does |
|---|---|
| `# EXPERIMENTAL — FC → SC direction` | Markdown header for the notebook |
| `# Cross-modal PCA/PLS closed-form overview` | Markdown intro |
| `import importlib` | **Cell 2** — imports, reloads, **direction toggle** (`SOURCE = "FC", TARGET = "SC"`), parcellation/seed config, helper functions for showing yaml + sbatch refs |
| `## Production references and shared local data` | Markdown |
| `data_sim = Sim(` | **Cell 4** — builds the `data_sim` object: train/val/test loaders, model name `CrossModal_PCA_PLS`, parcellation Glasser, seed 0. **Source of truth for all data.** |
| `## 1. CrossModalPCA` ... | Three model demo blocks (cells 5-12). Run the three project closed-form models (`CrossModalPCA`, `CrossModal_PLS_SVD`, `CrossModal_PCA_PLS`) end-to-end via `Sim._evaluate_model` and show their `test_metrics["base_metrics"]` in a summary DataFrame at the bottom. **These cells produce `pca_run`, `pls_svd_run`, `pca_pls_run`** — used in the summary cell. |

Key globals after Layer 1: `data_sim`, `pca_run`, `pls_svd_run`, `pca_pls_run`, `closed_form_summary`.

---

## Layer 2 — Phase 0 oracle + 10-seed robustness (cells 13-19)

| first line of cell | what it does |
|---|---|
| `# ===================== STEP 1: SC -> SC ORACLE (ceiling) =====================` | **Cell 13** — self-contained: rebuilds Sim with `SOURCE=TARGET="SC"` and runs `CrossModalPCA` to measure the SC→SC self-prediction ceiling. Result: demeaned 0.647, top1 1.0, avg_rank 1.0. This is the **upper bound** on what any SC-prediction model could hit. |
| (empty cell) | Cell 14, no-op spacer |
| `# ============= STEP 2a..2e: 10-SEED ROBUSTNESS =============` | **Cells 15-19** — run FC→SC PCA_PLS across seeds 0..9 (broken into 5 chunks of 2 seeds each for memory headroom). Final summary in Step 2e prints `demeaned 0.1341 ± 0.0054` across seeds. **Key globals after**: `_seed_results` dict (sometimes lost on kernel restart). |

Key globals after Layer 2: `_seed_results` (if not GC'd), `_SC_train_eval_pred` etc.

---

## Layer 3 — Anatomy stress test (cells 20-23, ALL TOUCHED TODAY)

This is where the day's main work lives.

### Cell 20 — `# ============= HELPER: FULL 6-METRIC PANEL =============` (NEW today)

Defines `_full_panel_eval(y_pred, y_true, target_train_mean_vec)` — computes all
6 metrics (mse, r2, pearson, demeaned_pearson, top1_acc, avg_rank) for any
prediction. Also defines `_fmt_panel(panel, label)` for one-line printing.

**End-of-day patch:** fixed `demeaned_pearson` to use the project's cosine
convention (cosine of (y − train_mean) per subject), not row-Pearson on demeaned
data. See [`metrics_glossary.md`](metrics_glossary.md) for the gotcha.

Must run before any of cells 21–23, 29, 32.

### Cell 21 — `# ============= STEP 3: BRAIN-VOLUME-ONLY BASELINE (confound check) =============`

**What it does:** fits OLS from 16 FreeSurfer brain-volume features (intracranial
vol, GM, WM, etc.) to SC edges. Pure anatomy, no FC. Tests "is FC→SC just
predicting brain anatomy?"

**Key globals after run:**
- `_base` — HCP_Base data object (used by all downstream cells)
- `_train_idx, _test_idx` — partition indices
- `_X_train, _X_test` — brain-volume features (z-scored)
- `_Y_train, _Y_test` — SC edge vectors (target)
- `_reg` — fitted LinearRegression
- `_Y_pred_test` — brain-vol → SC predictions on test
- `_panel_step3` — full 6-metric panel (added today)

**Result:** `brain-vol → SC demeaned = 0.1670` — *above* FC→SC's 0.134, the
"MAJOR FINDING" that triggered the anatomy investigation. **But** new identifiability
metrics show top1=0.103 < FC→SC's 0.154 — anatomy wins on demeaned, loses on top1.

### Cell 22 — `# ============= STEP 3.5: Does FC predict SC ABOVE brain anatomy? =============`

**What it does:**
1. Re-fits brain-vol → SC OLS (same as Step 3).
2. Computes `SC_residual = SC − brain_vol_prediction` (the part of SC that
   anatomy can't explain).
3. Manually rebuilds the `CrossModal_PCA_PLS` pipeline (PCA→PLS→inverse-PCA)
   and runs FC → SC_residual.
4. Sanity check: also runs FC → SC raw (should reproduce project's ~0.134).
5. Full-panel evaluates all three predictions (added today).

**Key globals after run:**
- `_X_brainvol_train, _X_brainvol_test` — brain-vol features (used by cell 29)
- `_SC_train, _SC_test` — raw SC edges
- `_FC_train, _FC_test` — raw FC edges
- `_SC_brainvol_train, _SC_brainvol_test` — brain-vol → SC predictions
- `_SC_resid_train, _SC_resid_test` — SC residuals after anatomy
- `_SC_test_pred_raw` — FC → SC raw (sanity)
- `_SC_resid_test_pred` — FC → SC_residual (main test)
- `_panel_35_bv, _panel_35_raw, _panel_35_res` — three full panels (added today)

**End-of-day patch:** PLSRegression now uses `max_iter=2000` (was default 500,
hit convergence warning).

**Result:** `FC → SC_residual demeaned = 0.0778`. So FC carries ~0.078 of
real connectivity signal about SC above anatomy. Sanity check: `FC → SC raw =
0.1322` (matches project's 0.134 ✓).

### Cell 23 — `# ============= STEP 3.6: SYMMETRIC -- SC -> FC after removing anatomy =============`

**What it does:** mirror of cell 22 with directions flipped. Brain-vol → FC,
SC → FC_residual.

**Key globals after run:**
- `_FC_bv_train, _FC_bv_test` — brain-vol → FC predictions
- `_FC_resid_train, _FC_resid_test` — FC residuals after anatomy (used by cell 29)
- `_FC_test_pred_raw` — SC → FC raw (sanity)
- `_FC_resid_test_pred` — SC → FC_residual (main test)
- `_panel_36_bv, _panel_36_raw, _panel_36_res` — three full panels (added today)

**End-of-day patch:** same `max_iter=2000` fix.

**Result:** `brain-vol → FC demeaned = 0.0467` (vs SC's 0.167 — SC is 3.6× more
anatomy-driven). `SC → FC_residual = 0.0561`. **Asymmetry survives anatomy
control: FC→SC_resid (0.078) / SC→FC_resid (0.056) = 1.39×.**

---

## Layer 4 — Variance partition + uncertainty (cells 24-28, unchanged today)

| first line of cell | what it does |
|---|---|
| `# ============= STEP 4.1: BOOTSTRAP CIs ON RESIDUAL DEMEANED-R =============` | Cell 24 — bootstrap CIs on the residual demeaned-r point estimates. Single-metric. |
| `# ============= STEP 4.2: PERMUTATION TEST FOR FC->SC_residual SIGNIFICANCE =============` | Cell 25 — permutation null on demeaned-r. |
| `# ============= STEP 4.3: BAYESIAN VARIANCE DECOMPOSITION =============` | Cell 26 — Cohen-style variance partition (anatomy vs FC vs full) in PCA-latent space using BayesianRidge. **Target = SC, K_TGT=64.** Gave Unique-FC = 0.0205. |
| `# ============= STEP 4.4: SYMMETRIC BAYESIAN VARIANCE DECOMPOSITION (TARGET = FC) =============` | Cell 27 — same with target = FC. Gave Unique-SC = 0.0196. Ratio 1.05× → triggered the (later-walked-back) "asymmetry is symmetric" framing. |
| `# ============= STEP 5: RECONCILE 1.39x vs 1.05x DISCREPANCY =============` | Cell 28 — **the broken reconciliation cell.** Conflated K_TGT change with model class change in Exp 4. PLS hit convergence warnings. Gave bogus 18.24× ratio. **Kept for history but don't trust outputs.** Superseded by cell 29 (Step 5.1). |

---

## Layer 5 — Controlled comparisons (cells 29-32, NEW/REWRITTEN today)

### Cell 29 — `# ============= STEP 5.1: CONTROLLED 2x2 AT K_TGT=256 (FULL-PANEL RETROFIT) =============` (REWRITTEN today)

**The controlled 2×2 stress test.** Fixes Step 5's flaws:
- PLS uses `max_iter=2000` and `n_components=64` (not over-parametrized)
- K_TGT held FIXED at 256 across all 4 combinations
- K_SRC held FIXED at 256
- Residual and partition each use their natural PCA basis

**4 combinations to isolate model class vs framework:**
| | residual + edge | partition + latent |
|---|---|---|
| **PLS** | A | B |
| **BR** | C | D |

**Today's rewrite** added full 6-metric panels for A/C (edge-space) and latent
pearson alongside R² for B/D.

**Key globals after run:**
- `panel_A_FC_SC, panel_A_SC_FC` — PLS residual full panels
- `panel_C_FC_SC, panel_C_SC_FC` — BR residual full panels
- `_pred_A_FC_SC, _pred_A_SC_FC` — PLS residual prediction matrices (used by bootstrap)
- `_pred_C_FC_SC, _pred_C_SC_FC` — BR residual prediction matrices (used by bootstrap)
- `B_sc, B_fc, D_sc, D_fc` — partition framework R² + latent pearson dicts

**Result (recalibrated from yesterday's wrong reading):**
- A (PLS, residual): ratio 1.39× demeaned, 5.33× top1, 1.25× avg_rank
- C (BR, residual): ratio 1.42× demeaned, 1.83× top1, 1.21× avg_rank
- B (PLS, partition): ratio 2.06× R² (noise-dominated)
- D (BR, partition): ratio 0.80× R² (REVERSE asymmetry at K_TGT=256 vs 1.05× at K_TGT=64)

**Robust headline:** residual+edge gives ~1.4× across model class, model-class
invariant. Partition+latent is K_TGT-sensitive and unstable.

### Cell 30 — short markdown summary cell of `pca_run, pls_svd_run, pca_pls_run` (legacy)

### Cell 31 — empty spacer

### Cell 32 — `# ============= STEP 5.2: CROSS-STEP FULL-PANEL SUMMARY =============` (NEW today)

**Pulls all `_panel_*` variables** from earlier cells into one consolidated
pandas table. Produces:
1. **Full-panel table** (10 rows × 7 columns) — every prediction × every metric
2. **Asymmetry ratio table** — FC→SC / SC→FC ratios for each metric, for each
   model/framework combo
3. **Anatomy-vs-FC table** on identifiability triad — the key new comparison

Read `phase0_snapshots.md` for the latest interpretation of these tables.

---

## Variable dependency graph

What you need in memory to run each cell from scratch (cold kernel):

```
Cell 4 (data_sim) ─┐
                   ├── Cell 13 (Step 1, SC->SC oracle)
                   ├── Cells 15-19 (Steps 2a-e, 10-seed)
                   ├── Cell 21 (Step 3, brain-vol -> SC) ──┐
                   ├── Cell 22 (Step 3.5, FC -> SC_resid) ─┤── Cell 29 (Step 5.1)
                   └── Cell 23 (Step 3.6, SC -> FC_resid) ─┤
                                                            └── Cell 32 (Step 5.2)
Cell 20 (helper) ── must run before any cell that calls _full_panel_eval
                    (i.e., cells 21, 22, 23, 29, 32 after today's retrofit)
```

**Minimum cells to refresh after today's fixes (warm kernel):**
20, 21, 22, 23, 29, 32 in order. ~10–15 minutes total.

**Cold kernel:** add cells 2, 4 at the start. ~12–18 minutes total.

---

## Common pitfalls

1. **Cell 20 must run first.** Cells 21-23, 29, 32 all call `_full_panel_eval`
   and will `NameError` if you skip it.

2. **Cells 22, 23 require `_base` from cell 21** (or rebuild Sim themselves if
   `_base` not in dir). If you restart kernel, run 21 before 22/23.

3. **Cell 29 requires `_SC_resid_*, _FC_resid_*, _X_brainvol_*` from cells 22, 23.**
   The cell has an assertion to catch this.

4. **Cell 32 requires `_panel_step3, _panel_35_*, _panel_36_*, panel_A_*, panel_C_*`
   from cells 21, 22, 23, 29.** The cell has assertions to catch missing vars.

5. **PLS convergence warnings** are common in cells 22, 23 (used to use default
   max_iter=500). End-of-day patch added max_iter=2000 — warnings should now
   be absent. If you still see them, the patch didn't take or you're running
   stale cells.

6. **`_panel_*` variables are recomputed every time you re-run the cell** — they
   don't persist if you re-run a downstream cell without the upstream. This is
   the main reason cells 21-23 are listed in the refresh order.

---

## Notebook size & editing notes

The notebook is large (over 25K tokens for several cells), past the standard
`Read` tool's max. To edit:

1. **Read individual cells via Python:**
   ```python
   import json
   with open(NB_PATH) as f: nb = json.load(f)
   print(''.join(nb['cells'][29]['source']))
   ```

2. **Edit via direct JSON manipulation** (see today's `/tmp/retrofit_notebook.py`
   approach — write a small Python script that loads the JSON, modifies the
   source string, writes it back). Don't try to use `NotebookEdit` for
   wholesale rewrites — it'll fail on "file modified since read."

3. **Append to a cell** without rewriting it: read the source, concatenate the
   new code, write it back. This is what we did for the full-panel blocks in
   cells 21-23.
