# Objective Functions — TODO / STATUS

Companion to [PLAN.md](PLAN.md). Two new imputation objectives: **Obj1 = maximize between-participant
difference (identity)**, **Obj2 = maximize cognition biomarker**. Goal: test whether each wins its own
axis on the 3-axis scorecard (reconstruction / identity / cognition) — the F7 frontier.

**Overall status:** 🟡 PLAN DRAFTED — awaiting go-ahead + answers to §9 open questions. Nothing built.

Legend: ⬜ todo · 🟡 in progress · ✅ done · ⏸️ blocked/waiting · ❌ dropped

_Last updated: 2026-06-26 (plan drafted)_

---

## Phase 0 — Align
- ✅ Draft PLAN.md (objectives, formulations, eval, build order, caveats)
- ⬜ Answer §9 open questions (backbone BR vs PLS; cognition target; OOF folds; 1B surrogate)
- ⬜ **GO-AHEAD to build the cheap closed-form cuts**

## Phase 1 — Closed-form first cuts (confirm directions move)
- ⬜ `reproduction/obj_functions/` skeleton (mirror `br_imputation/`)
- ⬜ **1A** per-PC amplitude restoration (reliability-gated, OOF g_k) on a BR backbone
- ⬜ **2C** cognition-weighted PC reconstruction (OOF β_k² weights)
- ⬜ Save `pred_*` artifacts; run the 3-axis scorecard (reuse recon panel + br_imputation + br_family + probe)
- ⬜ **Gate:** does 1A ↑ sibling AUC and 2C ↑ cognition lift vs BR/PLS?

## Phase 2 — Gradient / supervised versions (the clean frontier)
- ⏸️ **1B** contrastive InfoNCE linear map (identity-optimal)
- ⏸️ **2A** supervised target basis PLS(SC, cog) / **2B** multi-task joint loss
- ⏸️ Optional **2B λ-sweep** → reconstruct↔cognition curve

## Phase 3 — Synthesis
- ⏸️ Fill the 3-axis scorecard (BR / PLS / Obj1 / Obj2); does the diagonal light up?
- ⏸️ Findings note + figure (the triangle/frontier); fold into MASTER_FINDINGS if clean

## Deferred / out of scope
- ❌ 4S456 (later)
- ❌ MLP/nonlinear maps (keep linear first)

## Notes
- OOF/cross-fit is mandatory for every supervised gain (1A, 2C, 2A, 2B) — the only honest eval.
- None of these beat `obs_FC` for cognition; the comparison is objective-vs-objective.
- Reuse harnesses verbatim; only the estimator `fn` is new.
