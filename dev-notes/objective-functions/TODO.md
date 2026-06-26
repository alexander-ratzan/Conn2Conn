# Objective Functions — TODO / STATUS

Companion to [PLAN.md](PLAN.md). Two new imputation objectives: **Obj1 = maximize between-participant
difference (identity)**, **Obj2 = maximize cognition biomarker**. Goal: test whether each wins its own
axis on the 3-axis scorecard (reconstruction / identity / cognition) — the F7 frontier.

**Overall status:** 🟡 PLAN FINALIZED + decisions locked — ready to build Phase 1 (1A + 2C). Nothing built yet.

Legend: ⬜ todo · 🟡 in progress · ✅ done · ⏸️ blocked/waiting · ❌ dropped

_Last updated: 2026-06-26 (decisions locked: 5 seeds, BR backbone, CogCryst-resid target, 5-fold OOF, 1A+1B skip 1C, all of 2C/2A/2B)_

---

## Phase 0 — Align ✅
- ✅ Draft + finalize PLAN.md (objectives, rendered math, eval, build order, caveats)
- ✅ Decisions locked (PLAN §9): BR backbone · CogCryst-resid target (eval all 3) · 5-fold OOF ·
      1B self-ID surrogate · **Glasser × 5 seeds** · 1A+1B (skip 1C) · all of 2C/2A/2B
- ✅ **GO-AHEAD given** — build the cheap closed-form cuts

## Phase 1 — Closed-form first cuts (Glasser × 5; confirm directions move)
- ⬜ `reproduction/obj_functions/` skeleton (mirror `br_imputation/`)
- ⬜ **1A** per-PC amplitude restoration — BR backbone, reliability-gated `g_k`, 5-fold OOF
- ⬜ **2C** cognition-weighted reconstruction — OOF `β_k²` from CogCryst-resid; +raw-CogCryst side-check
- ⬜ Save `pred_*` artifacts; run the 3-axis scorecard (recon panel + br_imputation + br_family + probe),
      re-aggregating BR/PLS reference rows on the same 5 seeds
- ⬜ **Gate:** does 1A ↑ sibling AUC and 2C ↑ cognition lift (CogCryst) vs BR/PLS?

## Phase 2 — Gradient / supervised versions (all 3 Obj2 are different → all worth trying)
- ⏸️ **1B** contrastive InfoNCE linear map (self-ID surrogate; identity-optimal)
- ⏸️ **2A** supervised target basis PLS(SC, CogCryst-resid)
- ⏸️ **2B** multi-task joint loss (+ optional λ-sweep → reconstruct↔cognition curve)
- ❌ **1C** Rayleigh quotient — deferred

## Phase 3 — Synthesis
- ⏸️ Fill the 3-axis scorecard (BR / PLS / Obj1 / Obj2); does the diagonal light up?
- ⏸️ Findings note + figure (the triangle/frontier); fold into MASTER_FINDINGS if clean

## Deferred / out of scope
- ❌ 1C Rayleigh quotient (skip; circle back if needed)
- ❌ 4S456 (later)
- ❌ MLP/nonlinear maps (keep linear first)
- ❌ 3-vector (all-3 cognition) supervision — later extension if single-target promising

## Notes
- OOF/cross-fit is mandatory for every supervised gain (1A, 2C, 2A, 2B) — the only honest eval.
- None of these beat `obs_FC` for cognition; the comparison is objective-vs-objective.
- Reuse harnesses verbatim; only the estimator `fn` is new.
