# Latent-Direct — Findings (Glasser × 3 seeds)

**Question.** Does classifying on the task-tuned **latent** (skip the `inverse-PCA → re-PCA`
"glow-up" of the imputation→downstream pipeline) (a) recover signal the round-trip was eating, and
(b) let a custom objective finally beat the BR/PLS baselines on its own axis?

**Architecture.** `source → PCA → [objective] → latent → BayesianRidge directly on the latent`.
Objectives reused **verbatim** from `obj_functions` (BR, PLS, obj1a=identity, obj2c/obj2c_raw=cognition),
stopping before `inverse_transform`. Arms: `FC2SC`, `SC2FC` (cross-modal), `FCSC` (observed both).
Source: `outputs/scorecard_latent.csv`. Lead metrics: CogCryst lift over bv+demo; sibling AUC.

## Scorecard (mean over 3 seeds, Glasser)
| arm | objective | sib_AUC | CogCryst | CogTotal | CogFluid |
|---|---|---|---|---|---|
| FC2SC | BR | 0.767 | +0.082 | +0.021 | −0.020 |
| FC2SC | PLS | **0.808** | +0.034 | −0.014 | −0.074 |
| FC2SC | obj1a (identity) | 0.748 | +0.055 | +0.009 | −0.029 |
| FC2SC | obj2c (cog) | 0.709 | +0.075 | −0.018 | −0.086 |
| FC2SC | obj2c_raw (cog) | 0.698 | **+0.115** | +0.030 | −0.016 |
| SC2FC | BR | 0.732 | −0.074 | −0.065 | −0.055 |
| SC2FC | PLS | 0.741 | −0.117 | −0.101 | −0.078 |
| FCSC | plain (obs both) | — | **+0.162** | +0.103 | +0.059 |
| FCSC | obj2c | — | −0.080 | −0.131 | −0.165 |
| FCSC | obj2c_raw | — | −0.054 | −0.131 | −0.202 |

Reference (observed, 10-seed spine/br_family): obs_FC CogCryst **+0.133**, obs_SC **−0.101**;
obs_FC sib_AUC 0.823, obs_SC **0.863**.

## Result 1 — the round-trip WAS attenuating cognition (the pipeline win) ✅
Δ (latent-direct − round-trip `obj_functions`), FC2SC:

| objective | Δ sib_AUC (identity) | Δ CogCryst (cognition) |
|---|---|---|
| BR | −0.004 | **+0.044** |
| PLS | −0.005 | **+0.037** |
| obj1a | −0.004 | **+0.037** |
| obj2c | +0.021 | **+0.052** |
| obj2c_raw | +0.014 | **+0.069** |

- **Cognition: every objective improves +0.037…+0.069** when you classify on the latent instead of
  round-tripping through a reconstructed connectome. The inverse-PCA upscale re-orders the basis and
  BR re-shrinks in it, eroding the downstream signal.
- **Identity: Δ ≈ 0** (all within noise) — as predicted: PCA components are orthonormal, so
  `inverse_transform` is an **isometry** on centered latents; cosine-based identity is invariant. The
  round-trip cannot touch identity.
- **Mechanism, pinned:** round-trip = rotation → harmless for cosine/identity, distorting for the
  cognition regression. Consistent with the br-probe shrinkage story, now on the downstream side.

## Result 2 — the OBJECTIVES are still a dead end (no new utility) ❌
This was the actual question, and the answer is the same negative as `obj_functions`:

- **Identity objective FAILED:** `obj1a` (0.748) is *worse* than plain BR (0.767) and far below the
  PLS baseline (**0.808**). PLS — a baseline — remains the best identifier.
- **Cognition objective:** `obj2c_raw` (+0.115) beats plain BR (+0.082) by **+0.033** — the one real
  "objective > baseline" — but (i) it's within the per-seed scatter (±0.08, 3 seeds), (ii) it pays the
  **worst identity** (sib_AUC 0.698), and (iii) it never reaches **observed FC** (+0.133). A move
  *along* the F7 frontier, not a Pareto gain.
- **No manufactured connectome beats its observed scan on the matched axis:** real SC still wins
  identity (0.863 > best reconstructed 0.808); real FC still wins cognition (0.133 > 0.115).
- **The cognition objective even HURTS when the input is already observed** (`FCSC` obj2c −0.080 vs
  plain +0.162) — OOF cognition-reweighting overfits real signal. It's a cross-modal/imputation tool
  at best, not a universal one.

## Result 3 — F5 mechanism revised (the genuinely useful takeaway)
"Imputed connectomes don't transfer cognition" (F5) was **substantially a pipeline artifact**.
Reconstructed-SC cognition climbs −0.010 (PLS round-trip, spine) → +0.045 (BR round-trip) →
**+0.082 (BR latent-direct) → +0.115 (obj2c_raw latent-direct)** ≈ 86% of the observed-FC ceiling.
**Honest qualifier:** `FC→SC`-reconstructed SC *is* FC information, so this is "the latent preserves
FC's cognition signal where the round-trip discarded it," **not** "SC newly predicts cognition." The
`SC2FC` arm stays negative (−0.074…−0.117) — you cannot recover cognition from SC-derived FC.

## Verdict: PARKED (dead end on the objectives; one keeper on the pipeline)
- **Objectives: closed negative.** You cannot cheaply re-aim a connectome at a third objective — the
  reconstruct↔identify frontier is structural (now confirmed in latent space too). Extends F7c.
- **Keeper:** *classify on the latent, skip the inverse-PCA upscale* — a strict cognition improvement
  (+0.04–0.07) at zero identity cost, and it shows F5's "no transfer" was partly the round-trip, not
  a fundamental limit. Worth one sentence in the paper's F5 discussion.
- **Caveats:** Glasser × **3 seeds**, ±0.08 per-seed variance; Δ vs the 5-seed round-trip (direction
  monotone across all 5 objectives, magnitudes preliminary). Not scaled to 10 seeds / 4S456 by choice
  — the negative on the objectives is clear enough to close.
