# Imputation Objective Functions — PLAN

**Status:** PLAN ONLY — not executed. Awaiting go-ahead.
**Date:** 2026-06-26 · **Scope:** Glasser only (4S456 deferred), reuse the BR-only harness.

---

## 1. Goal & framing

We measured that the **imputation estimator's objective** sets where a predicted connectome lands on
the F7 reconstruct↔identify tradeoff (BR = per-PC MSE shrinkage → great reconstruction/cognition, bad
identity; PLS = covariance → balanced). Now build estimators that deliberately optimize the **other
two corners** and test whether each wins its own axis and loses the others (a frontier, not a free lunch):

| estimator | objective | recon (demeaned-r) | identity (sib AUC) | cognition (lift) |
|---|---|---|---|---|
| `bayesian_ridge` | per-component MSE + evidence shrink | **high** | low | mid |
| `pca_pls` | max FC↔SC covariance | mid | **high** | mid |
| **Obj1 (this plan)** | max between-participant difference | low? | **highest?** | low? |
| **Obj2 (this plan)** | max cognition biomarker | low? | low? | **highest?** |

**Hypothesis:** the bolded diagonal lights up → "what a predicted connectome is good for is set by the
imputer's objective; you can't win all three." F7 generalized into a design principle.

## 2. Where it plugs in (reuse everything)

Each new objective is just a new imputation estimator with the existing interface
`fn(Z_train, Z_test, Y_train) -> pred` in PCA-latent space (source PCA → map → target → inverse-PCA).
It then flows through the harness we already built:
- **Reconstruction** axis: `full_panel_eval` (demeaned_pearson, avg_rank, top1).
- **Cognition** axis: save `pred_*` artifacts → `br_imputation/run_br_unit.py`-style downstream (lift).
- **Identity** axis: `br_family/run_br_family.py`-style sibling AUC.
- **Mechanism**: `br_imputation/probe_shrinkage.py` (per-PC amplitude/recovery) for any new estimator.

So the only new code is the estimator `fn`s; evaluation is the existing three harnesses pointed at the
new `pred_*`.

Notation: source FC train `X` (n×p), target SC train `Y` (n×q); source latents `Z=PCA_k(X)` (n×256);
target latents `W=(Y−μ)U_yᵀ` (n×256); base map `M: Z→Ŵ`; reconstruct `Ŷ = Ŵ U_y + μ`.

---

## 3. Objective 1 — maximize between-participant difference (identity)

### 1A — Per-PC amplitude restoration (cheap, closed-form; FIRST)
Fit a base regressor (BR or PLS), then restore each predicted target-PC's between-subject spread to
the true train amplitude, **gated by recovery reliability** (don't amplify noise):
```
g_k = corr_k * std(W_train[:,k]) / std(Ŵ_train_OOF[:,k])      # corr_k = OOF recovery corr of PC k
Ŵ_test[:,k] <- g_k * Ŵ_test[:,k]
```
- Directly inverts the tail-collapse we measured (BR amp 0.04 → toward 1.0) ONLY where direction is
  reliably recovered. ~10 lines on top of BR/PLS.
- **Honesty:** `g_k` must use **out-of-fold** train predictions (k-fold within train) or it's
  optimistic. The OOF is the only real cost.
- **Predict:** sibling AUC ↑ (recovers discriminative tail), reconstruction demeaned-r ~flat or ↓
  (amplitude restoration can add variance without improving cosine).

### 1B — Contrastive / fingerprint loss (the true objective; gradient, SECOND)
Train a linear map `M` in latent space with InfoNCE on demeaned cosine `s`:
```
L = -Σ_i log [ exp(s(Ŷ_i, Y_i)/τ) / Σ_j exp(s(Ŷ_i, Y_j)/τ) ]
```
- Directly optimizes identifiability (avg_rank/top1) — pushes each subject's prediction away from all
  others. Linear M on 256-dim, n=683 → seconds to train (torch or sklearn+autograd).
- **Predict:** highest sibling AUC of all estimators; lowest reconstruction.

### 1C — Rayleigh quotient (closed-form middle ground; optional)
Linear `M` maximizing between-subject scatter of predictions s.t. a reconstruction constraint →
generalized eigenproblem. No gradient descent; less directly tied to the cosine metric than 1B.

**Pick:** 1A first (instant read), 1B as the identity-optimal estimator.

---

## 4. Objective 2 — maximize the cognition biomarker (cognition-supervised)

The imputer sees **train cognition** `c` and builds `pred_SC` to be cognition-predictive.

### 2C — Cognition-weighted PC reconstruction (cheapest; FIRST)
Weight each target SC-PC's reconstruction by its train cognition-predictiveness:
```
β_k from  cog ~ SC-PC_k  (OOF on train);   w_k ∝ β_k²
impute with BR/PLS but scale the per-PC target by w_k (spend fidelity on cognition-bearing modes)
```
Closed-form on top of the existing per-PC estimators.

### 2A — Supervised target basis (clean; SECOND)
Replace the unsupervised SC-PCA with **PLS(SC_train, c_train)** → cognition-aligned SC directions `V`;
impute `FC → V-latents → inverse`. The imputed connectome lives in the cognition-relevant SC subspace.

### 2B — Multi-task joint loss (frontier knob; gradient, THIRD)
```
L = ||Ŵ − W||²  +  λ · ||c − g(Ŵ)||²        # g = linear cognition readout
```
Sweep `λ` → trace the reconstruct↔cognition frontier directly.

**⚠️ Honesty (critical):** the imputer uses **train** cognition; eval stays on **held-out test**
cognition (no leak). BUT imputer + downstream model double-dip the same train cognition signal →
**cross-fit / OOF** the imputer's cognition-learning, else train `pred_SC` is optimistically loaded.
And from F5 the SC↔cognition signal is weak → this can beat the *reconstruction-objective* imputation
but **won't beat `obs_FC`**. Frame the question as "does a cognition objective recover more cognition
than a reconstruction objective," not "does imputation finally beat FC."

**Pick:** 2C first (cheap), 2A second; 2B only if we want the λ-frontier.

---

## 5. Evaluation (the 3-axis scorecard)

For each new estimator, on Glasser × 10 seeds, fill the row:
- **Reconstruction:** FC→SC `demeaned_pearson`, `avg_rank`, `top1` (reuse `full_panel_eval`).
- **Identity:** sibling AUC of `pred_SC_resid_bvdemo` (reuse `br_family`).
- **Cognition:** `pred_SC` + `pred_SC+bv+demo` lift on Cog{Total,Fluid,Cryst} (reuse `br_imputation`).
- **Mechanism:** per-PC amplitude/recovery (reuse `probe_shrinkage.py`) — does Obj1 restore the tail?

Compare against the BR and PLS rows we already have. Success = the diagonal (each objective wins its axis).

## 6. Build order
1. **1A** (amplitude-restore) + **2C** (cognition-weighted) — both closed-form, ~a day, confirm directions move.
2. Gate: if 1A ↑ identity and 2C ↑ cognition vs BR/PLS, proceed.
3. **1B** (contrastive) + **2A/2B** (supervised/multi-task) — gradient versions for the clean frontier.
4. Optional: λ-sweep (2B) to draw the reconstruct↔cognition curve; 4S456 replication.

## 7. Isolation / outputs
New module `reproduction/obj_functions/` (mirrors `br_imputation/`): own `outputs/`, own `pred_*`
artifacts, own CSVs. Spine + br_imputation + br_family untouched. Estimators import `_grid_common`.

## 8. Caveats / risks
- **OOF everywhere** for the supervised gains (1A gains, 2C weights, 2A/2B) — the only honest way.
- Gradient estimators (1B/2B) add a torch dependency in the imputer — keep the map **linear** first
  (closed-form-ish, interpretable) before any MLP.
- The FC ceiling stands: none of these will beat `obs_FC` for cognition; the comparison is
  objective-vs-objective, not vs-FC.
- Glasser only; 4S456 deferred.

## 9. Open questions (confirm before build)
1. Base regressor for 1A / per-PC for 2C: BR or PLS as the backbone? (suggest BR — it's where the
   shrinkage is, so amplitude-restore has the most to fix.)
2. Cognition target for 2C/2A: single (CogCryst, strongest) or all three jointly? (suggest CogCryst first.)
3. OOF folds within train: 5-fold? (suggest 5.)
4. Identity metric to optimize in 1B: sibling AUC needs family pairs (only in test) → optimize the
   **self-identifiability** surrogate (rank/top1 of pred_i vs true_i) on train, then evaluate sibling
   AUC on test. Confirm that surrogate is acceptable.

## Decisions locked
- (none yet — this is the draft; fill on go-ahead)
