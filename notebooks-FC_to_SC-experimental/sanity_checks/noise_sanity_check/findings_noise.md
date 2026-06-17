# FC noise sanity check — findings

**Run**: HCP-YA, SLURM jobs 11034024 (build+A+B+F) + 11034554 (E re-run after low-dim fix).
4-cell FC design (run-1/run-2 × LR/RL), both parcellations, ~1018 FC subjects
(reliability) / 957 canonical (cross-modal). All values from `outputs/*.csv`.

## TL;DR

FC is **mostly noise at the single-edge level but highly reliable as a whole connectome**,
and SC predicts only a small fraction of even the *reproducible* FC signal. This is the
physical counterpart to MASTER_FINDINGS F10 (statistical saturation): the FC→SC / SC→FC
prediction gap is part genuine cross-modal independence, part FC being edge-noisy.

- **FC reliability ceiling** (between-session, Glasser): demeaned_r **0.49**, fingerprint
  top1 **0.93**, avg_rank **0.99**.
- **FC edge variance**: **30% trait / 4% day-to-day state / 2% within-session / 64%
  noise**; the averaged connectome we actually use has reliability **G ≈ 0.59**.
- **SC→FC captures ~17%** of the reproducible FC signal (demeaned_r 0.085 of ceiling
  0.49) — and only **~5%** of the fingerprinting ceiling.
- **Parcellation-robust** (4S456 nearly identical).
- **SC noise itself remains UNMEASURED** — no test-retest dMRI in HCP-YA.

## A. Reliability ceiling (native metric)

| parc | comparison | demeaned_r | pearson | top1 (fingerprint) | avg_rank |
|---|---|---|---|---|---|
| Glasser | within-session (LR↔RL) | 0.364 | 0.710 | 0.78 | 0.968 |
| Glasser | **between-session (REST1↔REST2)** | **0.491** | 0.813 | **0.933** | 0.992 |
| 4S456 | within-session | 0.337 | 0.670 | 0.78 | 0.969 |
| 4S456 | between-session | 0.457 | 0.783 | 0.936 | 0.993 |

**Note (important):** within-session (LR↔RL) is *lower* than between-session — the opposite
of "shorter interval = more reliable." Cause = the **phase-encode distortion confound**:
LR and RL have opposite distortions, so single-direction connectomes disagree more, while
each session (LR+RL averaged) is distortion-cancelled and cleaner. So **between-session
0.49 is the valid ceiling**; the LR↔RL rung is contaminated and is NOT a clean
short-interval estimate.

## B. Variance decomposition (G-theory 2×2; individual-difference fractions, sum to 1)

| parc | trait (signal) | state (day) | within-session | **noise** | G (avg connectome) |
|---|---|---|---|---|---|
| Glasser | **0.301** | 0.035 | 0.021 | **0.643** | 0.585 |
| 4S456 | 0.261 | 0.035 | 0.021 | **0.683** | 0.519 |

At the single-edge level **~64–68% of between-subject variance is measurement noise**,
only ~26–30% is stable trait, and day-to-day state is small (~3–4%). The 4-cell average
(REST1+REST2, LR+RL) lifts reliability to G≈0.52–0.59. Per-edge components saved in
`outputs/b_variance_components_{parc}.npz`.

## F. Whole-connectome reliability

| parc | fingerprint top1 | discriminability |
|---|---|---|
| Glasser | 0.927 | 0.998 |
| 4S456 | 0.934 | 0.999 |

**The reconciliation** (the original puzzle: "but scans match across sessions"): per-edge
~64% noise (B) yet whole-connectome 93% identifiable / 0.998 discriminable. Both true —
the individual signal is **distributed**, reliable in aggregate even when each edge is
mostly noise.

## E. Cross-modal disattenuation — "% of reproducible FC captured"

ceiling = REST1↔REST2 (demeaned_r 0.491). Achieved = source→FC, PCA→PLS, 10-seed median.

| source | metric | achieved | ceiling | fraction of ceiling |
|---|---|---|---|---|
| **SC→FC** | demeaned_r | 0.085 | 0.491 | **0.17** |
| SC→FC | top1 (fingerprint) | 0.051 | 0.933 | **0.05** |
| SC→FC | avg_rank | 0.713 | 0.992 | 0.72 |
| bv+demo→FC | demeaned_r | 0.098 | 0.491 | 0.20 |
| bv+demo→FC | top1 | 0.026 | 0.933 | 0.03 |

Reads:
- **SC captures only ~17% of the reproducible FC signal** (demeaned_r) and **~5% of the
  fingerprinting ceiling** — so SC→FC is far from the reliability ceiling: the gap is
  genuine cross-modal independence, not just FC noise. (If SC→FC were near the ceiling we'd
  blame noise; it isn't, so most of the unexplained FC is reliable-but-structurally-
  unpredictable.)
- **bv+demo→FC (0.20) ≥ SC→FC (0.17)** even after disattenuation — consistent with the
  project's baseline finding: cheap subject confounds match/beat the structural connectome.
- `pearson` fraction >1.0 is an artifact (raw pearson is dominated by the shared population
  mean and is uninformative here — use demeaned_r); not reported as meaningful.

## G + H. Per-subject reliability + does prediction track it?

**G — per-subject reliability is heterogeneous and a stable trait.** The 0.49 ceiling is a
mean; per-subject between-session reliability spans ~0 to 0.78 (Glasser: mean 0.489,
median 0.494, std **0.117**, skew **−0.58**, non-normal p=5e-15), with a low tail (~2% of
subjects < 0.2; one at −0.002 = pure noise). It's a stable subject property
(within-vs-between per-subject ρ=0.41) — consistent with motion/compliance being
person-level. (`outputs/g_per_subject_*`, `g_reliability_hist.png`.)

**H — SC→FC is NOT noise-limited per subject; the gap is uniform independence.**
- A subject's SC→FC prediction quality is **uncorrelated** with their own FC reliability
  ceiling: Pearson **r=0.01 (p=0.72)**, Spearman ≈0. (bv+demo→FC weakly tracks it,
  r=0.15, p=2e-5.) So cleaner-FC subjects are **not** more predictable from SC.
- **Reliability-filtering does not sharpen SC→FC** — it makes the fraction *worse*:
  | filter | n | SC achieved | SC ceiling | SC fraction |
  |---|---|---|---|---|
  | all | 857 | 0.083 | 0.491 | **0.169** |
  | drop rel<0.2 | 841 | 0.083 | 0.498 | 0.167 |
  | drop bottom 10% | 771 | 0.083 | 0.517 | 0.160 |
  | keep top 50% | 429 | 0.080 | 0.582 | **0.137** |
  Keeping only high-reliability subjects raises the ceiling (0.49→0.58) but SC's achieved
  stays flat (~0.08), so the fraction drops. The noisy tail was never the bottleneck.
- (bv+demo→FC holds ~0.20–0.21 across all filters.)
- **Conclusion**: at the per-subject level too, the SC→FC gap is *genuine cross-modal
  independence*, not per-subject measurement noise — if it were noise, cleaner subjects
  would be more predictable and filtering would help; neither happens. Reinforces E.
- (`outputs/h_per_subject_achieved_vs_ceiling.csv`, `h_reliability_filtered_summary.csv`,
  `h_correlations.csv`, `h_achieved_vs_ceiling_scatter.png`.)

## What this resolves and what stays open

- **Resolves**: how much of FC is noise (edge-level ~64%; whole-connectome reliable), and
  that SC→FC is *not* noise-ceiling-limited — it leaves reliable FC signal unexplained, so
  the cross-modal gap is real independence. Sharpens F10 with a physical denominator.
- **Open (data-blocked)**: SC's own noise floor, and FC→SC disattenuated by SC reliability
  — both need test-retest dMRI (not in HCP-YA). ⚠️ literature plug-in only until sourced.

## Caveats
- Within-session rung confounded by phase-encode distortion (above).
- Reliability computed on ~1018 FC subjects; SC→FC achieved on the 957 canonical set
  (reliability is a per-subject property, stable across the subset).
- Between-session interval is ~1 day, same scanner (HCP-YA REST1/REST2) — the optimistic
  end; a months-apart/multi-site retest would show lower reliability (more noise).
- Cognition not involved here — this is FC↔FC reliability + SC→FC; the cognition ceiling
  is covered separately (F4/F5/F10).

## Files
- `outputs/a_reliability_ceiling.csv`, `b_variance_decomposition.csv`,
  `b_variance_components_{Glasser,4S456Parcels}.npz`, `f_discriminability.csv`,
  `e_crossmodal_disattenuation.csv`, `noise_synthesis.csv`
- scripts: `build_fc_cells.py`, `a_`/`b_`/`e_`/`f_`, `synthesize_noise.py`,
  `_noise_common.py`, `run_all.sbatch`, `run_e_synth.sbatch`
- roadmap: `planning/roadmap/noise-sanity-check.md`
