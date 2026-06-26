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

## Phase 1 — Closed-form first cuts (Glasser × 5; confirm directions move) 🟡
- ✅ `reproduction/obj_functions/` built (committed ddb0094): `_obj_estimators.py`, `run_obj_unit.py`
      (3 axes/seed), `finalize_obj.py` (scorecard), sbatch/submit/README
- ✅ **1A** per-PC amplitude restoration — BR backbone, reliability-gated `g_k`, 5-fold OOF
- ✅ **2C** cognition-weighted reconstruction — OOF `β_k²` from CogCryst-resid + raw-CogCryst side-check
- ✅ 3-axis scorecard built into the runner (recon `full_panel_eval` + identity sib-AUC via `_fm_common`
      + cognition lift via `bayesian_ridge_scalar`); BR/PLS computed inline on the same 5 seeds
- ✅ Pilot seed 0 (job 11860548): BR/PLS validated **byte-for-byte** (recon dr 0.163/0.132;
      CogCryst lift BR +0.215 / PLS +0.121 = prior runs). obj1a slow (5-fold OOF = long pole).
- ✅ Full run done (seeds 0–4, finalize 11868624). scorecard.csv pulled + committed.
- ✅ **GATE RESULT: diagonal did NOT light up — clean NEGATIVE.**
      - obj1a (identity) FAILED: sib_AUC 0.752 < BR 0.771 ≪ PLS 0.812 (reliability gate can't restore tail)
      - obj2c (cognition resid) FAILED: +0.023 < BR +0.037, recon collapsed (0.044)
      - obj2c_raw (cognition raw): +0.047 vs BR +0.037 = noise-level flicker, costs recon+identity
      - BR still owns reconstruction, PLS still owns identity → F7 frontier is structural, not a cheap knob
- ✅ FINDINGS.md written. Phase 1 = clean negative.

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
