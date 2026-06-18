# FC noise sanity check — findings

**Run**: HCP-YA, SLURM jobs 11034024 (build+A+B+F) + 11034554 (E re-run after low-dim fix).
4-cell FC design (run-1/run-2 × LR/RL), both parcellations, ~1018 FC subjects
(reliability) / 957 canonical (cross-modal). All values from `outputs/*.csv`.

## TL;DR

FC is **mostly noise at the single-edge level but highly reliable as a whole connectome**,
and SC predicts only a small fraction of even the *reproducible* FC signal — and that
shortfall is **not explained by FC measurement noise** (proven per-subject). This is the
physical counterpart to MASTER_FINDINGS F10 (statistical saturation): the SC→FC gap is
demonstrably not an FC-noise artifact; genuine cross-modal independence is the strong
interpretation, pending the one data-blocked piece (SC's own reliability — needs
test-retest dMRI). **CLOSED, filed as F10-supporting (MASTER A6).**

- **FC reliability ceiling** (between-session, Glasser): demeaned_r **0.49**, fingerprint
  top1 **0.93**, avg_rank **0.99**.
- **FC edge variance**: **30% trait / 4% day-to-day state / 2% within-session / 64%
  noise**; the averaged connectome we actually use has reliability **G ≈ 0.59**.
- **SC→FC captures ~17%** of the reproducible FC signal (demeaned_r 0.085 of ceiling
  0.49) — and only **~5%** of the fingerprinting ceiling.
- **Parcellation-robust** (4S456 nearly identical).
- **SC noise itself remains UNMEASURED** — no test-retest dMRI in HCP-YA.

## The honest bottom line on "how much is noise"

There isn't one number, and that's not a failure to find a straight answer — the question
is genuinely **level-dependent**. The straight answer is the set:

1. **A single FC measurement is ~64% noise** (per-edge individual-difference variance).
2. **Your averaged usable connectome is ~41% noise** (reliability G ≈ 0.59).
3. **The reproducible individual signal you can actually predict tops out at demeaned-r
   0.49** (the between-session ceiling), and it is **heterogeneous across people (0 to
   0.78)**.
4. **Of that reproducible 0.49, SC explains ~17%, flat across subjects** (independent of
   each subject's own reliability).

(Caveat that rides #4: this is the FC-side accounting; SC's *own* noise floor is unmeasured
— so "SC explains 17% of reproducible FC" is exact, while "the other 83% is signal SC
doesn't contain" is the strong interpretation, pending SC test-retest.)

> ### #**REVIEW AND QUESTION** — is "ceiling" valid per-subject, or only at the population level?
>
> *(Flagged for review — logic below seems sound but wants a second look before it hardens
> any per-subject claim. It does NOT affect the population headline or the flatness result;
> it sharpens how we're allowed to phrase the per-subject material.)*
>
> **The disattenuation logic (Spearman 1904) is a *population* theorem**, not a per-subject
> law: the correlation between two variables is bounded by the geometric mean of their
> reliabilities **in expectation, over a population.** A single subject's "reliability"
> (correlation between their two scans) is **one noisy number from a single pair of
> measurements** — large standard error. So is their achieved prediction. Comparing two
> noisy single-subject estimates, some will land achieved > reliability **purely by sampling
> noise** — especially low-reliability subjects, whose reliability estimate sits near zero
> and is easy to exceed by chance. (The −0.002 subject isn't truly perfectly unpredictable;
> −0.002 is noise around some small true value, and a noisy achieved score can exceed it.)
>
> **Precise statement:** the ceiling is a **valid population bound but an invalid
> per-subject bound** — per-subject reliability is a single noisy estimate, not a true
> per-person limit. The theorem holds on average, not pointwise; treating it pointwise
> produces the contradiction we saw (achieved > "ceiling" for some subjects).
>
> **Corrected framing (threads both concerns):**
> - **Population level — keep "ceiling".** "Across subjects, individual FC reproduces at
>   mean demeaned-r 0.49; SC→FC captures ~17% of that." The disattenuation is legitimate
>   here and is not exceeded on average. → stays in the main text (bottom-line #3, E).
> - **Per-subject level — do NOT call it a ceiling.** Call it **"per-subject reliability"**
>   and describe its *distribution* (0–0.78, ρ=0.41, a stable trait). Frame the H result as
>   **"SC's prediction is *uncorrelated* with subject reliability"** (a relationship between
>   two measured quantities) — **not** "SC stays below each subject's ceiling," which is the
>   framing that breaks.
>
> **Net:** the H conclusion stands (flat r=0.01 = SC prediction is reliability-independent),
> and it's arguably *cleaner* under this framing. The thing to fix downstream: the
> per-subject `fraction_of_ceiling` column in `h_per_subject_achieved_vs_ceiling.csv`
> implicitly treats per-subject reliability as an individual bound — report it as a
> descriptive ratio at most, and lean on the **correlation/flatness** statement (and the
> bv+demo contrast) for the actual claim, not on pointwise "fraction of ceiling."
>
> ---
>
> ### Two ceilings are different objects (and we should report both, labeled)
>
> There are **two distinct ceilings**; conflating them is the deeper source of the tension.
>
> - **Ceiling A — data / reproducibility ceiling** (`FC_day1 ↔ FC_day2`). *How reproducible
>   is the target itself?* Model-free, a property of the **data**. "Individual FC only agrees
>   with itself at **0.49**, so no predictor can exceed the reproducible signal." Valid at
>   population level, breaks per-subject (above).
> - **Ceiling B — model / oracle ceiling** (`FC→FC`, `SC→SC` through the *same* PCA→PLS
>   pipeline). *How well can THIS model class predict the target from a perfect same-modality
>   copy?* A property of **model + data together**. "Even predicting FC from FC, PCA→PLS only
>   reaches X — the architecture has a representational limit." (`SC→SC` oracle ≈ **0.647**
>   from earlier work — to recompute consistently in the grid; `FC→FC` = **[to compute]**.)
>
> **Ceiling B fixes both earlier complaints:**
> - **Not exceeded per-subject** — it's a within-modality prediction run through the *same
>   pipeline / estimator / CV / metric* as the cross-modal predictions; an apples-to-apples
>   upper reference, no "two noisy scans" comparison that can flip.
> - **Available for SC** — `SC→SC` is just a prediction task, no test-retest needed. The
>   SC-side data gap that blocks Ceiling A **disappears** for B, so both directions get a
>   consistent ceiling.
>
> **But B is a *looser* bound than A** (the "???" catch): `FC→FC` can exploit
> **session-specific signal that wouldn't replicate** in a fresh scan, so B can sit *above*
> A. B measures "max ability of the model to reproduce *this* connectome," not "max
> recoverable *individual trait*." **B cannot substitute for A** for the biological fraction.
>
> **How to report — a two-rung reference, each labeled for what it bounds:**
> - **SC→FC achieved: 0.085.**
> - **Ceiling B (model oracle), both directions:** `FC→FC` = [to compute], `SC→SC` ≈ 0.647 →
>   "within-modality is the architecture's best case; cross-modal loses this much." Clean,
>   consistent, available both directions, never exceeded per-subject → the **model-capacity /
>   cross-modal-loss** story.
> - **Ceiling A (data reproducibility): 0.49**, FC-side only, heterogeneous → "of the
>   *reproducible* signal, SC gets ~17%." The **biological** fraction, with the per-subject
>   caveat + SC-side gap flagged.
>
> **Honesty sentence to carry:** *"The within-modality oracle (FC→FC) is a model-capacity
> reference and exceeds the cross-session reproducibility limit, because it can fit
> session-specific signal that does not replicate; we therefore disattenuate biological
> claims against the reproducibility ceiling (A) and use the oracle (B) only to quantify
> cross-modal vs within-modality loss."*

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
- **Why flat r=0.01 is the strong outcome (not a weak/null one):** per-subject achieved
  prediction is *mechanically* bounded by reliability (you can't predict noise), so the
  default expectation was a **positive** slope by construction. We got flat. SC captures a
  fixed ~0.08 whether a subject's FC is reliable (0.78) or near-noise (0.0) — which *rules
  out* the boring ceiling-effect explanation. The filtering result corroborates by going
  the "wrong" way: dropping unreliable subjects *lowers* the captured fraction
  (0.169→0.137) because the ceiling rises while achieved stays pinned. Ceiling moves,
  achieved doesn't = SC has a fixed, modest grip on FC unrelated to FC's measurement quality.
- **The bv+demo contrast is what makes it a clean dissociation (not an artifact):** the
  analysis *does* detect a real reliability effect when one exists — bv+demo→FC mildly
  tracks reliability (r=0.15). That SC's slope is flat while the baseline's isn't means the
  flatness is not a methodological artifact (it would have hit both).
- **Conclusion (with the caveat riding it):** the airtight claim is **the SC→FC gap is not
  FC-measurement-noise and not per-subject reliability** — decisively, at the per-subject
  level. The reading "SC doesn't *contain* that part of FC" is the strong **interpretation**,
  but is not fully separable from "SC contains it but measures it too noisily, *uniformly*
  across subjects" — a uniform SC noise floor would also produce a flat line. Distinguishing
  those needs SC test-retest (data-blocked). So: 95% of the way to genuine independence; the
  last 5% is the SC-reliability hole. Reinforces E.
- (`outputs/h_per_subject_achieved_vs_ceiling.csv`, `h_reliability_filtered_summary.csv`,
  `h_correlations.csv`, `h_achieved_vs_ceiling_scatter.png`.)

## Status: CLOSED (supporting item for F10)

This module has done its pre-grid job: it confirms the cross-modal ceiling is not an
FC-measurement-noise artifact, in our native metric, per-subject. **Filed as F10-supporting
in MASTER_FINDINGS (Appendix A6). No further per-subject reliability analysis is warranted
here** — the finding is extracted; the discipline now is to run the reproducibility grid
this de-risked, not to chase the per-subject rabbit hole.

## What this resolves and what stays open

- **Resolves (airtight)**: how much of FC is noise (edge-level ~64%; whole-connectome
  reliable), and that the SC→FC gap is **not FC-measurement-noise and not per-subject
  reliability** — decisively, per-subject (flat r=0.01 vs the baseline's r=0.15).
  Sharpens F10 with a physical denominator.
- **Interpretation (strong, pending data)**: the natural reading is "SC doesn't *contain*
  that part of FC" — but a *uniform* SC noise floor would also produce the flat line, so
  this last step is not fully separable from FC-side evidence alone.
- **Open (data-blocked)**: SC's own noise floor, and FC→SC disattenuated by SC reliability
  — both need test-retest dMRI (not in HCP-YA). ⚠️ literature plug-in only until sourced.
  This is the 5% gap between "not-noise-on-the-FC-side" (proven) and "genuine independence"
  (interpretation).

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
