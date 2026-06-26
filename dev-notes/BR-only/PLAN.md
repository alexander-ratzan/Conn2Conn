# BR-only Imputation + Downstream Run — PLAN

**Status:** PLAN ONLY — not executed. Awaiting explicit go-ahead before any code/run.
**Owner:** Adel · **Date:** 2026-06-26 · **Scope:** Glasser only (4S456 deferred).

---

## 1. Goal (one sentence)

Re-run the imputation→downstream pipeline with **BayesianRidge in both estimator slots**
(impute the connectomes with BR, predict cognition with BR), over an **expanded 18-input set**,
saved to its **own isolated output location**, to see whether the *strongest* version of this
pipeline changes the F5 picture ("imputed connectomes don't transfer downstream").

No hypothesis is being forced. This is a **best-foot-forward / SOTA-for-this-pipeline** run:
BR is the strongest reconstructor (owns the demeaned-r oracle: FC→FC 0.67, SC→SC 0.65) *and* the
best downstream cognition estimator. So "BR-impute + BR-downstream" is the strongest imputation-
transfer test this pipeline can run. Both outcomes are informative:
- pred_* still neutral/harmful → F5 hardens ("even the best imputer doesn't transfer").
- pred_* lifts cognition where PLS didn't → a real new positive result worth chasing.

---

## 2. What changes vs. the current grid (the two estimator slots)

The pipeline has **two independent estimator slots**. The current spine grid uses different
estimators in each; this run sets both to BR.

| Slot | Current spine grid | This run |
|---|---|---|
| **Imputation** — creates saved `pred_SC`/`pred_FC` (`make_handoff_artifacts.py`) | fixed capped **PCA→PLS** | **capped BayesianRidge** (`capped_bayesian_ridge`) |
| **Downstream** — predicts cognition from an input (`run_downstream_grid.py`) | swept (pca_pls, BR, kernel_ridge ×9) | **BayesianRidge only** (`bayesian_ridge_scalar`) |

Everything else (frozen splits, BP-1 block scaling, BP-2 subject_id join, metric panel, leak
guardrail logic) is **reused verbatim** from `_grid_common.py` so the math stays identical to the
rest of the project.

---

## 3. Inputs — 18 total (10 carried over + 8 new)

All `pred_*` below are **BR-imputed** (the slot change). Block inputs go through the existing
per-block-latent path (`_block_latents`, BP-1). `bv` = FreeSurfer brain volumes (z); `demo` =
age_z + sex_oh + race_eth_oh; `bv+demo` = both. (Confirmed: `demo` literally contains sex/age —
this drives the leak classes below.)

### Carried over (10) — for a self-contained file (baseline + obs references in-file)
1. `bv+demo` — **baseline** (the bar; lift is computed over this)
2. `obs_FC`
3. `obs_SC`
4. `obs_FC+obs_SC`
5. `pred_SC`  ← BR-imputed
6. `pred_FC`  ← BR-imputed
7. `obs_FC+bv+demo`
8. `obs_SC+bv+demo`
9. `pred_SC+bv+demo`  ← BR-imputed
10. `pred_FC+bv+demo` ← BR-imputed

### New (8)
| # | Input | Block composition | Leak class (sex/age target) |
|---|---|---|---|
| 11 | `pred_FC+bv` | [pred_FC, bv] | contains-bv → flag (subject-info) |
| 12 | `pred_SC+bv` | [pred_SC, bv] | contains-bv → flag |
| 13 | `obs_SC+pred_FC` | [obs_SC, pred_FC] | connectome-only → EXPECTED_SIGNAL |
| 14 | `obs_FC+pred_SC` | [obs_FC, pred_SC] | connectome-only → EXPECTED_SIGNAL |
| 15 | `obs_FC+pred_SC+bv` | [obs_FC, pred_SC, bv] | contains-bv → flag |
| 16 | `obs_FC+pred_SC+bv+demo` | [obs_FC, pred_SC, bv, demo] | contains-demo → EXEMPT_FLAGGED |
| 17 | `obs_SC+pred_FC+bv+demo` | [obs_SC, pred_FC, bv, demo] | contains-demo → EXEMPT_FLAGGED |
| 18 | `everything` | [obs_FC, obs_SC, pred_FC, pred_SC, bv, demo] | contains-demo → EXEMPT_FLAGGED |

**Interpretation notes**
- Demo-containing combos (9,10,16,17,18): `lift_over_bvdemo` isolates the *connectome's marginal
  contribution* (demo is in both input and baseline) → the "does the connectome add over
  subject-info" question.
- `+bv` combos (11,12,15): lift = connectome + brain-volume vs. the full bv+demo baseline.
- 13/14 (obs+pred, opposite modalities): test whether an imputed connectome adds anything *on top
  of the observed other modality*.
- 18 (`everything`): the kitchen-sink ceiling for this input family.

---

## 4. Targets — 5 (unchanged from the spine grid)

- **Cognition (results):** `CogTotal`, `CogFluid`, `CogCryst` (NIH Toolbox composites).
- **Leak-checks (not findings):** `sex`, `age`.

---

## 5. Metrics (lead with the lift)

Per cell, BR-scalar predictions scored by (`_grid_common` helpers, reused):
- **`lift_over_bvdemo`** — PRIMARY: input pearson − bv+demo-baseline pearson (same target/seed,
  BR). For `sex`: balanced-accuracy lift.
- **`lift_perm_p`** — paired sign-flip permutation p for the lift.
- **`residualized_pearson`** — cognition after removing the bv+demo OLS term (cognition only).
- `pearson` / `spearman` / `r2` / `n_eval`; `balanced_acc` for `sex`.

Aggregate as **mean ± std across the 10 seeds** (pilot = seed-0 only).

---

## 6. Known, deliberate choice: in-sample train imputation (NOT a leak)

Recorded here so it is on the record as intentional, not a bug.

- The **train/test split is fully seed-based and frozen** (BP-2). Test subjects are held out; the
  imputer is fit on **train only** and never sees test. `pred_*_test` is a genuine out-of-sample
  prediction. **No test-set / test-label leak.**
- The **train-side** imputed connectomes are made **in-sample**
  (`pred_SC_train = impute(FC_tr, FC_tr, SC_tr)`): the imputer that produced subject *i*'s train
  connectome had subject *i*'s true target in its fitting set. This makes training inputs slightly
  "cleaner" than test inputs — a train/test **input-distribution mismatch**, not a leak.
- We **keep in-sample** for this run (one-variable swap vs. the PLS spine; apples-to-apples). BR is
  a regularized Bayesian model, so in-sample optimism is mild.
- **Future option (not in scope):** out-of-fold (OOF) train imputation closes this gap; orthogonal
  change, can be a later pass. The choice is identical for PLS and BR, so it does not affect the
  BR-vs-PLS comparison either way.

---

## 7. Isolation — own spot, existing results untouched

New self-contained module. Nothing here writes to the spine grid's CSVs or `artifacts/`.

```
reproduction/br_imputation/
  run_br_pilot.py                 # BR-impute pred_* → save → BR-downstream over the 18 inputs → leak verdict
  README.md                       # what this run is + pointer back to this PLAN + the §6 note
  outputs/
    downstream_br.csv             # ← THE result file (own spot)
    leak_verdict_br.csv
    artifacts/Glasser/seed{N}/    # BR-imputed pred_{SC,FC}_{train,test}.npy + subject_ids_{train,test}.npy
  configs/
    expected_cells_br.csv         # completeness manifest for this sub-grid
```

The spine grid's `reproduction/outputs/{reconstruction,downstream,leak_verdict}.csv` and
`reproduction/outputs/artifacts/` are **never opened for writing**.

---

## 8. Execution plan (when greenlit)

1. **Pilot — Glasser, seed 0.** Run `run_br_pilot.py --seeds 0`. Sanity: BR imputation completes,
   18 inputs build, `downstream_br.csv` has 18×5 = 90 rows (BR-only), leak verdict has 0 LEAK_FAIL.
   Eyeball the four `pred_*` lifts on CogCryst vs the spine grid's PLS-imputed numbers
   (PLS: pred_FC −0.135, pred_SC −0.010) to see direction of the effect.
2. **Decision gate.** If wiring is clean, fan out.
3. **Full — Glasser × 10 seeds.** `run_br_pilot.py --seeds 0..9` → 900 downstream rows. Verify
   completeness against `expected_cells_br.csv`; run leak verdict on merged.
4. **4S456 — deferred** (separate later pass if results warrant).

**Cost note:** BR imputation is the expensive part (per seed: 4 imputations × 256 per-component
BayesianRidge fits = ~1024 fits). Glasser is the fast parcellation; BR-only downstream is light
(90 cells/seed). Pilot first to confirm before spending the 10-seed fan-out.

---

## 9. Leak handling for the new inputs

Extend the guardrail bookkeeping (in the new module, not the shared file) so the new inputs are
classified correctly for the sex/age targets:
- **`contains_bvdemo` (→ EXEMPT_FLAGGED if over threshold):** any input containing `demo`
  → #1,7,8,9,10,16,17,18. (demo holds sex/age outright, so these trivially predict the leak
  targets — expected, interpret cognition columns only.)
- **contains-bv flag:** inputs with `bv` but not `demo` → #11,12,15. Brain volume correlates with
  sex but won't hit balanced_acc > 0.99 alone; expected verdict `ok`, flagged as subject-info.
- **connectome-only (→ EXPECTED_SIGNAL if over threshold):** #2,3,4,5,6,13,14. Raw/imputed
  connectomes encoding sex/age is real biology, not a leak.
- **Hard rule unchanged:** a demographic-free, non-connectome input over threshold = `LEAK_FAIL`
  (none expected here).

---

## 10. Completeness

`expected_cells_br.csv` enumerates 18 inputs × 5 targets × {bayesian_ridge} × {Glasser} × seeds,
generated from the same constants the runner uses (no drift). A verifier hard-fails on any
missing/non-finite cell, mirroring the spine grid's `verify_completeness.py`.

---

## 11. Implementation sketch (file-by-file, for the build step — not done yet)

- **`run_br_pilot.py`**
  - `BR_INPUTS` = the 18-name list (§3); `build_br_input(sp, hand, name)` extends
    `run_downstream_grid.build_downstream_input` to cover the 8 new block names (#11–18) using
    `sp["bv_train"]/sp["demo_train"]` and `hand["pred_*"]`.
  - Imputation: reuse `capped_bayesian_ridge` for `pred_{SC,FC}_{train(in-sample),test}`; save to
    `outputs/artifacts/Glasser/seed{N}/` with `subject_ids_*` (BP-2 join contract).
  - Downstream: loop targets × inputs, estimator fixed to `bayesian_ridge_scalar`; compute
    lift/perm-p/residualized; append to `outputs/downstream_br.csv`.
  - Leak: classify per §9, write `outputs/leak_verdict_br.csv`.
- **`README.md`** — one-paragraph what/why + links to this PLAN and the §6 note.
- **No edits** to `_grid_common.py`, `run_downstream_grid.py`, `make_handoff_artifacts.py`, or any
  spine output. (If a helper genuinely must be shared, import it read-only; do not modify.)

---

## 12. Open items / risks

- BR demeaned-r is only *slightly* above PLS; downstream lift is noisier — the cognition numbers
  may barely move. That flat result is itself the answer ("imputation quality isn't the
  bottleneck").
- The 6-way `everything` block (#18) is the widest per-block latent concat — confirm it builds
  within memory on Glasser (should be fine; 6 × 256 latents).
- Keep `random_state=0` everywhere so the pilot is bit-reproducible.

---

## Decisions locked (this session)
1. **Downstream estimator = BayesianRidge only.** ✓
2. **Carry over the original 10 inputs** (self-contained file) + 8 new = **18**. ✓
3. **In-sample train imputation stays**; OOF noted as a deliberate, known choice (§6). ✓
4. **Glasser only**, seed-0 pilot → Glasser × 10 seeds. 4S456 deferred. ✓
5. **Isolated outputs** under `reproduction/br_imputation/` — spine grid untouched. ✓
