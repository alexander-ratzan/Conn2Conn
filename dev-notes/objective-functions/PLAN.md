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

### Notation (used in §3–§4 — every symbol below)

| symbol | shape | meaning |
|---|---|---|
| $n$ | scalar | number of training subjects (≈ 683) |
| $p,\;q$ | scalar | number of FC edges / SC edges (64,620 each on Glasser) |
| $X$ | $n\times p$ | training **FC** edge matrix — the source we predict *from* |
| $Y$ | $n\times q$ | training **SC** edge matrix — the target we **impute** |
| $\mu$ | $q$ | training group-mean SC connectome (the "average brain") |
| $U_y$ | $256\times q$ | target **PCA loadings** — orthonormal SC "shape" directions (PCs) |
| $Z$ | $n\times 256$ | **source latents**: FC compressed to 256 PCA scores |
| $W$ | $n\times 256$ | **target latents**: each subject's SC deviation on the PCs. $W_{i,k}$ = how strongly subject $i$ expresses SC-PC $k$ |
| $M$ | map | the learned **map** from source to target latents: $\hat W = M(Z)$ |
| $\hat W$ | $n\times 256$ | **predicted** target latents |
| $\hat Y$ | $n\times q$ | reconstructed SC connectome $=\hat W\,U_y+\mu$ |
| $c$ | $n$ | training **cognition** score per subject (e.g. CogCryst) |
| $s(a,b)$ | scalar | **demeaned cosine** similarity between connectomes $a,b$ — the identity metric, $s(a,b)=\dfrac{(a-\mu)\cdot(b-\mu)}{\lVert a-\mu\rVert\,\lVert b-\mu\rVert}$ |

---

## 3. Objective 1 — maximize between-participant difference (identity)

**Intuition:** make predicted connectomes *spread out and distinguishable* across people instead of all
collapsing toward the average brain (which is exactly what BR's shrinkage does). A good identity
imputation lets you tell subjects — and families — apart.

### 1A — Per-PC amplitude restoration (cheap closed-form; FIRST)
**What it does:** after a base regressor predicts the SC latents, re-inflate each PC's between-subject
spread back to the true amount — but only for the PCs it predicts *reliably*, so we restore signal, not
noise. Applied per target PC $k$:

$$\hat W^{\text{test}}_{:,k}\;\leftarrow\;g_k\cdot \hat W^{\text{test}}_{:,k},
\qquad
g_k \;=\; \underbrace{r_k}_{\text{reliability}}\;\cdot\;
\underbrace{\frac{\operatorname{std}_i\!\big(W_{i,k}\big)}{\operatorname{std}_i\!\big(\hat W^{\text{OOF}}_{i,k}\big)}}_{\text{amplitude gap}}$$

where:
- $\hat W^{\text{test}}_{:,k}$ — predicted scores of all **test** subjects on SC-PC $k$ (the vector we rescale).
- $\operatorname{std}_i(\cdot)$ — standard deviation **across subjects** = the between-participant spread.
- $\operatorname{std}_i(W_{i,k})$ — spread of the **true** scores on PC $k$ (how much real people differ on that mode).
- $\hat W^{\text{OOF}}_{i,k}$ — the **out-of-fold** predicted score for subject $i$ (predicted by a model trained on *other* folds); using OOF, not in-sample, keeps the gain honest.
- $r_k$ — across-subject correlation between predicted and true scores on PC $k$ (how reliably the model recovers that mode). Multiplying by $r_k$ means *only restore amplitude where the direction is trustworthy; leave noisy PCs shrunk.*

**Why:** directly inverts the tail-collapse we measured (BR amplitude 0.04 → back toward the true spread),
reliability-gated. ~10 lines on a BR/PLS backbone. **Predict:** sibling AUC ↑; reconstruction demeaned-r
flat or slightly ↓.

### 1B — Contrastive / fingerprint loss (the true objective; gradient, SECOND)
**What it does:** train the map so each subject's predicted connectome is closest to *their own* true
connectome and far from everyone else's — i.e. directly optimize "can you fingerprint people." Minimize
over the map $M$:

$$\mathcal{L}\;=\;-\sum_{i=1}^{n}\;\log\frac{\exp\!\big(s(\hat Y_i,\,Y_i)/\tau\big)}
{\displaystyle\sum_{j=1}^{n}\exp\!\big(s(\hat Y_i,\,Y_j)/\tau\big)}$$

where:
- $\hat Y_i$ — predicted SC connectome for subject $i$ (from the map $M$, via inverse-PCA).
- $Y_j$ — the **true** SC connectome of subject $j$ (every subject is a candidate to match against).
- $s(\cdot,\cdot)$ — demeaned cosine similarity (defined in the Notation table) — the identity metric.
- $\tau$ — temperature; smaller $\tau$ = sharper "must match the exact subject" pressure.
- The fraction = subject $i$'s prediction similarity to **its own** true connectome ÷ similarity to **everyone**. Maximizing it pushes each subject's prediction *away* from the others — literally "maximize the difference between participants," in the discriminative sense.

Linear $M$ on 256-dim latents, $n=683$ → trains in seconds. **Predict:** highest sibling AUC; lowest
reconstruction.

### 1C — Rayleigh quotient (closed-form middle ground) — ❌ DEFERRED (skip; circle back if needed)
**What it does:** find the linear map whose predictions carry the **most between-subject variance** while
still reconstructing SC — a variance-maximizing compromise:

$$M^\star=\arg\max_{M}\;\frac{\operatorname{tr}\!\big(\operatorname{Cov}_{\text{between-subj}}(MZ)\big)}{\big\lVert W-MZ\big\rVert_F^{2}}$$

where the **numerator** is the spread of predictions across subjects and the **denominator** is
reconstruction error. Solvable as a generalized eigenproblem (no gradient descent), but less directly
tied to the cosine identity metric than 1B.

**Pick:** 1A first (instant read), 1B as the identity-optimal estimator.

---

## 4. Objective 2 — maximize the cognition biomarker (cognition-supervised)

**Intuition:** instead of imputing the most *faithful* SC, impute the SC that best *predicts cognition*.
The imputer is allowed to see **training** cognition $c$; test cognition is never touched.

### 2C — Cognition-weighted reconstruction (cheapest; FIRST)
**What it does:** make the imputer spend its effort on the SC modes that carry cognition, and neglect the
cognition-irrelevant ones. First, on train, regress cognition on the SC latents to score each mode's
relevance, then fit the map with a **weighted** reconstruction loss:

$$c\;\approx\;\sum_{k=1}^{256}\beta_k\,W_{:,k}
\;\;\Longrightarrow\;\;w_k\propto\beta_k^{2},
\qquad
M=\arg\min_{M}\;\sum_{k=1}^{256} w_k\,\big\lVert W_{:,k}-(MZ)_{:,k}\big\rVert^{2}$$

where:
- $\beta_k$ — regression weight of SC-PC $k$ when predicting cognition $c$ on train (fit **OOF**). Large $|\beta_k|$ = that SC mode matters for cognition.
- $w_k\propto\beta_k^{2}$ — per-mode importance weight; cognition-relevant PCs get more weight.
- $W_{:,k}$ vs $(MZ)_{:,k}$ — true vs predicted score of all subjects on PC $k$.
- Net effect: the map is pushed to nail the cognition-bearing PCs and may sacrifice the rest.

Closed-form on the per-component backbone. **Predict:** cognition lift ↑ vs BR/PLS; reconstruction ↓.

### 2A — Supervised target basis (clean; SECOND)
**What it does:** swap the unsupervised SC-PCA basis for one built to capture cognition, then impute into it.

$$V=\operatorname{PLS}\big(Y,\,c\big)\;\;(\text{SC directions of maximal covariance with cognition}),
\qquad W'=(Y-\mu)\,V^{\top}$$

where $V$ are the cognition-aligned SC directions (replacing $U_y$) and $W'$ the cognition-aligned target
latents. Impute $FC\to W'$ and reconstruct with $V$; the imputed connectome lives in the cognition-relevant
SC subspace.

### 2B — Multi-task joint loss (the frontier knob; gradient, THIRD)
**What it does:** train one map that both reconstructs SC and predicts cognition, with a dial $\lambda$
trading the two:

$$\mathcal{L}=\underbrace{\big\lVert W-MZ\big\rVert_F^{2}}_{\text{reconstruct SC}}
\;+\;\lambda\underbrace{\big\lVert c-g(MZ)\big\rVert^{2}}_{\text{predict cognition}}$$

where:
- $g(\cdot)$ — a linear readout from predicted latents $MZ$ to a cognition estimate.
- $\lambda$ — the tradeoff dial: $\lambda=0$ is pure reconstruction (≈ BR); large $\lambda$ is cognition-first. **Sweeping $\lambda$ traces the reconstruct↔cognition frontier directly.**

**⚠️ Honesty (critical):** the imputer uses **train** cognition only; eval stays on **held-out test**
cognition (no leak). BUT imputer + downstream model double-dip the same train cognition signal →
**cross-fit / OOF** the imputer's cognition-learning, else train `pred_SC` is optimistically loaded.
And from F5 the SC↔cognition signal is weak → these beat the *reconstruction-objective* imputation but
**won't beat `obs_FC`**. Frame the question as "does a cognition objective recover more cognition than a
reconstruction objective," not "does imputation finally beat FC."

**Pick:** 2C first (cheap), 2A second; 2B only for the λ-frontier.

---

## 5. Evaluation (the 3-axis scorecard)

For each new estimator, on **Glasser × 5 seeds** (0–4), fill the row:
- **Reconstruction:** FC→SC `demeaned_pearson`, `avg_rank`, `top1` (reuse `full_panel_eval`).
- **Identity:** sibling AUC of `pred_SC_resid_bvdemo` (reuse `br_family`).
- **Cognition:** `pred_SC` + `pred_SC+bv+demo` lift on Cog{Total,Fluid,Cryst} (reuse `br_imputation`),
  led by **CogCryst** (the supervised target) with Total/Fluid as transfer checks.
- **Mechanism:** per-PC amplitude/recovery (reuse `probe_shrinkage.py`) — does Obj1 restore the tail?

Compare against the BR and PLS rows we already have (re-aggregate those on the same 5 seeds for a fair
row-to-row comparison). Success = the diagonal (each objective wins its axis).

## 6. Build order
1. **Phase 1 — closed-form (FIRST):** **1A** (amplitude-restore) + **2C** (cognition-weighted), BR
   backbone, 5-fold OOF, CogCryst-resid target. Confirm directions move.
2. Gate: if 1A ↑ identity and 2C ↑ cognition vs BR/PLS, proceed.
3. **Phase 2 — gradient/supervised (ALL THREE Obj2 worth trying — they're genuinely different):**
   **1B** (contrastive), **2A** (supervised basis), **2B** (multi-task). 1C deferred.
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

## 9. Decisions (resolved 2026-06-26)
1. **Backbone = BayesianRidge** (where the shrinkage is → amplitude-restore has the most to fix). ✓
2. **Cognition target = CogCryst, supervised on its bv+demo *residual*** (genuine non-demographic
   biomarker; strongest connectome signal; thesis-aligned). **Evaluate on all three** raw targets via
   `lift_over_bvdemo` — CogCryst = matched, CogTotal/CogFluid = transfer/generalization checks. **Scalar**
   for Phase 1 (3-vector is a later extension). Cheap side-check in 2C: also run **raw-CogCryst**
   supervision so we can see whether residualizing was necessary.
   - *Why CogCryst:* CogFluid is too weak to optimize against (a null would be uninterpretable); CogCryst
     has F4-confirmed real connectome signal beyond demographics, so it's the fairest test of whether a
     cognition objective can do anything. Residualizing guards against the objective just chasing the
     demographic confound (CogCryst is the most demographically loaded).
3. **OOF = 5-fold** within train. ✓
4. **1B identity target = self-identifiability surrogate** (rank/top1 of pred_i vs true_i) on train,
   evaluate sibling AUC on test. ✓ Accepted as *not perfectly* matched to the family objective — can
   swap the 1B objective to a family-aware loss later if warranted.

## Decisions locked
- **Seeds:** Glasser × **5** (seeds 0–4); re-aggregate BR/PLS reference rows on the same 5 for fairness.
- **Objective 1:** build **1A** + **1B**; **1C deferred** (skip, circle back if needed).
- **Objective 2:** build **all three** (2C, 2A, 2B) — they're genuinely different approaches.
- **Backbone** BR · **5-fold OOF** · **target** CogCryst-resid (eval all three) · **1B** self-ID surrogate.
- **Scope:** Glasser only (4S456 deferred); isolated `reproduction/obj_functions/`; reuse existing harnesses.
