# Noise Sanity-Check Roadmap

## Purpose

Quantify how much of the connectome (FC, SC) is **reproducible signal** vs **measurement
noise vs real day-to-day state change**, so we can contextualize the prediction ceiling
(MASTER_FINDINGS F10): is FC→SC / SC→FC capped because the signal isn't there, or because
the *measurement* is noisy? The key tool is the **reliability ceiling** — you cannot
predict a connectome better than it predicts itself.

Scope of this doc: everything we'd ideally do (Part 1), then exactly what is doable with
**HCP-YA, our current cache** (Part 2), with a **compute-cost estimate per check** (Part 3).

Reference data scale (HCP-YA, current cache):
- ~957 subjects
- FC: **2×2 design per subject** — `run-1/run-2` (session, ~1 day apart) × `LR/RL`
  (direction, minutes apart) = 4 Pearson FC matrices, for **both** parcellations
  (Glasser 360→64,620 edges; 4S456 456→103,740 edges). **No BOLD timeseries.**
- SC: **1 matrix per subject** (SIFT2 connectome) + r2t. **No tractogram, no retest.**

---

## PART 1 — Everything we WANT to do (the ideal menu)

Each check: what it measures · data it needs · what it tells us.

### A. Reliability ceilings (native metric: `full_panel` — demeaned_r, pearson, top1_acc, avg_rank, mse, r2)
Treat one real measurement as the "prediction" of another; the agreement is the ceiling
any predictor could reach, in our exact metrics.
- **A1 within-session** (run minutes apart) · needs 2 within-session measurements · ceiling excluding day-to-day state.
- **A2 between-session** (~1 day) · needs 2 sessions · the **trait ceiling** — correct denominator for cross-modal trait prediction.
- **A3 between-visit** (months) · needs a months-apart rescan · ceiling including biological drift.
- **A4 between-scanner / site** · needs multi-site rescan · ceiling including hardware.

### B. Variance-components decomposition (generalizability theory)
Partition each edge's variance into named sources.
- **B1 session × direction (2×2)** · needs the 4-cell FC design · splits variance into
  σ²_subject (trait/signal) / σ²_session (state) / σ²_direction (within-session+distortion) / σ²_residual (noise).
- **B2 + visit + scanner factors** · needs months/multi-site rescans · adds drift + hardware components.

### C. Bootstrap estimation noise (single scan)
Resample the underlying data and recompute the connectome; the wobble = pure estimation noise.
- **C1 FC timeframe bootstrap** · needs BOLD timeseries · pure sampling noise from finite TRs.
- **C2 SC streamline bootstrap** · needs the tractogram (.tck) · pure sampling noise from finite streamlines (the one way to get SC reliability from a single scan).

### D. Analytic sampling-noise model
- **D1** · needs #TRs (FC) / #streamlines (SC) · the irreducible statistical floor via
  `var(r)≈(1−r²)²/(N−1)` (FC) or Poisson on counts (SC). No rescan needed; model-dependent.

### E. Cross-modal disattenuation (% of reliability ceiling captured)
- **E1 SC→FC ÷ FC ceiling** · needs FC reliability · "of reproducible FC, what fraction does SC explain?"
- **E2 FC→SC ÷ SC ceiling** · needs SC reliability · the symmetric statement.

### F. Identifiability statistics
- **F1 fingerprinting** (top1 self-match across the two measurements) · needs 2 measurements · the "scans match" aggregate reliability (reconciles modest per-edge r with high whole-connectome reproducibility).
- **F2 discriminability / I2C2** · needs ≥2 measurements · multivariate reliability scalar (better than averaging per-edge ICC).

### G. SC reliability (the blocker)
- **G1 SC test-retest** · needs repeat dMRI · direct SC noise.
- **G2 SC streamline bootstrap** (= C2) · needs tractogram · single-scan SC noise.
- **G3 literature ICC plug-in** · needs nothing · external bound / sensitivity band only.

---

## PART 2 — What we can ACTUALLY do with HCP-YA (current cache)

| Check | Doable on HCP-YA? | Why |
|---|---|---|
| A1 within-session ceiling (LR↔RL) | ✅ yes | 4-cell FC design has LR/RL; caveat = phase-encode distortion confound |
| A2 between-session ceiling (REST1↔REST2) | ✅ yes | run-1/run-2 present; **already partially done** (per-edge r=0.45) |
| A3 between-visit (months) | ❌ no | no months-apart rescan in cache |
| A4 between-scanner | ❌ no | single scanner |
| B1 variance decomposition (session×direction 2×2) | ✅ yes | exactly the 4-cell FC design |
| B2 + visit/scanner factors | ❌ no | needs A3/A4 data |
| C1 FC timeframe bootstrap | ❌ no | **no BOLD timeseries** (inputs are precomputed Pearson relmats) |
| C2 SC streamline bootstrap | ❌ no | **no tractogram** (only the connectome matrix) |
| D1 analytic sampling-noise | ⚠️ partial | TR count knowable (~1200/run) for a rough FC floor; SC counts unavailable |
| E1 SC→FC ÷ FC ceiling | ✅ yes | needs only FC reliability (have it) — **the headline** |
| E2 FC→SC ÷ SC ceiling | ❌ no | needs SC reliability (literature plug-in only) |
| F1 fingerprinting (top1) | ✅ yes | from the 2 FC measurements |
| F2 discriminability / I2C2 | ✅ yes | from the FC measurements |
| G1/G2 SC reliability | ❌ no | no retest, no streamlines |
| G3 SC literature plug-in | ✅ caveated | external value, sensitivity band only |

**Bottom line:** the entire **FC** noise story is doable (ceiling at 2 intervals,
variance decomposition, fingerprinting, cross-modal disattenuation). The entire **SC**
noise story is **blocked** — literature plug-in only — until a test-retest dMRI dataset
(the friend ask) or the HCP retest release is ingested.

**One prep step required:** the FC loader currently averages LR+RL into a single session
matrix. To get A1/B1 we must re-cache the **4 cells separately** (REST1-LR, REST1-RL,
REST2-LR, REST2-RL), both parcellations. Small loader change; I/O-bound, one-time.

---

## PART 3 — Compute cost per doable check (HCP-YA scale)

Everything feasible here is **vectorized linear algebra on connectome matrices, not model
fitting** — so it's cheap. No SLURM grid needed; this is notebook/single-node scale. Sizes:
FC cell = 957×64,620 (Glasser) ≈ 247 MB float32; 4 cells ≈ 1 GB; 4S456 ≈ 1.6× that.

| Check | Compute | Where | RAM | Wall-clock |
|---|---|---|---|---|
| **Prep: 4-cell FC re-cache** | read 4 TSVs × 957 subj × 2 parc ≈ 7.7k files, parallel | **Torch** (TSVs live there) | low | **~10–20 min** (I/O-bound, one-time) |
| A1 within-session ceiling (LR↔RL) | per-subject demeaned_r (vectorized) + one 957×957 similarity matmul for top1/avg_rank | laptop | ~1 GB | **<2 min** /parc |
| A2 between-session ceiling (REST1↔REST2) | same as A1 | laptop | ~1 GB | **<2 min** /parc |
| B1 variance decomposition (2×2 per edge) | vectorized variance components over (957×4×64,620); chunk edges if RAM tight | laptop | ~2–4 GB | **~2–5 min** /parc |
| E1 SC→FC ÷ FC ceiling | ratios of already-computed panel numbers | laptop | trivial | **seconds** |
| F1 fingerprinting (top1) | 957×957 cross-similarity (one matmul: (957×E)·(E×957)) | laptop | ~1 GB | **<1 min** /parc |
| F2 discriminability / I2C2 | within/between-subject distance over 957² pairs | laptop | ~1 GB | **~1–3 min** /parc |
| D1 analytic FC sampling floor | closed-form per edge from TR count | laptop | trivial | **seconds** |
| G3 SC literature plug-in | apply a scalar/banded ICC | laptop | trivial | **seconds** |

**Total for the full feasible FC suite:** one ~10–20 min Torch re-cache (once) + **~15–25
min of laptop compute** for both parcellations combined. No GPU, no SLURM, no multi-seed
loops (reliability is a fixed-data measurement, not a seeded fit).

Contrast with what's *not* costable here because it's data-blocked, not compute-blocked:
C1/C2 (bootstrap), A3/A4/B2 (longer intervals), E2/G1/G2 (SC reliability) — these need
**new data**, not more compute.

---

## What this buys us

A defensible, cheap answer to "how much of the scan is noise" for **FC**, in our native
metric, that directly sharpens F10: we can state SC→FC as a fraction of the FC reliability
ceiling and decompose FC variance into trait / state / noise. The **SC** half stays an
explicit open item gated on test-retest dMRI — flagged, not faked.

> ⚠️ **STOP — ASK FOR HUMAN INPUT (SC reliability):** every SC-noise check (C2, E2, G1,
> G2) requires data we do not have (repeat dMRI or tractograms). Do **not** substitute a
> literature ICC silently into a headline number — use it only as a clearly-labeled
> sensitivity band, and revisit once a test-retest dMRI dataset is sourced.
