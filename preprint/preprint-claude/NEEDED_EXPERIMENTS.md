# Holes, Fixes, and Needed Experiments

A reviewer-style audit of the preprint. Part 1 lists what was **fixed now** from the
existing `reproduction/outputs/*.csv` (no new data). Part 2 lists what **needs new data or
re-running the pipeline**, each with a concrete design and a feasibility rating.

Feasibility key: **A** = cheap, existing data already loaded by the grid (hours);
**B** = needs re-running the existing pipeline with a new input/control (1–2 days);
**C** = needs new derived data from raw HCP files we already have access to (days–week);
**D** = needs data HCP-YA does not contain, or an external cohort (weeks / blocked).

---

## Part 1 — Fixed now (see `ROBUSTNESS_ANALYSIS.md`, figure SX5)

| # | Hole | Fix delivered | Outcome |
|---|---|---|---|
| 2 | Downstream lifts not multiplicity-corrected; no CIs | BH-FDR (whole-grid + restricted affirmative family) + 95% seed CIs on every cell | **Materially changes the positive claim.** obs FC CogCryst (uncorrected p≈0.03) does **not** survive whole-grid FDR (q≈0.24); the strongest cell `obs_FC+bv+demo`→CogCryst survives restricted-family FDR (q=0.027 Glasser / 0.048 4S456). Report restricted-family-corrected, with CIs, not bare p<0.05. |
| 6 | Negatives stated as p>0.05, no power | TOST equivalence (±0.02 band) + minimum-detectable-effect | Median MDE ≈ 0.056 at 80% power, n=10 seeds. SC/imputed cells are not merely non-significant — they sit **below** the null band (genuinely negative). Design is well-powered for ~0.1 lifts, underpowered below ~0.03. |
| 11 | Oracle/cross-modal estimator mismatch (BR vs PLS) | Recomputed "fraction of model ceiling" with a consistent PCA→PLS oracle | SC→FC reaches ~20% and FC→SC ~37% of the same-estimator within-modality ceiling; the large cross-modal loss is estimator-independent. |
| 3 (partial) | "Imputation harmful" could be a high-dim input artifact | Compared equally high-dim pred_SC vs pred_FC | Harm is **direction-specific** (pred_FC ~2–3× more harmful than pred_SC at matched dimensionality), so not pure conditioning. Definitive control still needs the pipeline (see E3 below). |

**The most important consequence:** the paper's single affirmative claim should be softened
and reframed as restricted-family-corrected with CIs. The negative claims are, if anything,
*strengthened* by the equivalence framing.

---

## Part 2 — Needs new data or pipeline runs

### E1 · Split-half SC reliability (the highest-value missing check) — feasibility **C**

**Why.** Claim 1's strong reading ("genuine cross-modal independence") rests entirely on
FC-side accounting; the supplement concedes a uniform SC noise floor would mimic the flat
per-subject result, and frames SC reliability as fully data-blocked pending test-retest
dMRI. It is only *partly* blocked: SC reliability can be estimated **within a single
acquisition by split-half**.

**Design.**
1. For each subject, split the tractogram into two independent halves — either by streamline
   bootstrap (random 50/50 split of streamlines) or, better, by **diffusion sub-sampling**
   (split gradient directions / shells, run tractography on each half).
2. Build SC$_a$, SC$_b$ per subject for both parcellations.
3. Whole-connectome reliability = the SC$_a$↔SC$_b$ `demeaned_pearson` distribution
   (mirror of the FC `a_reliability_ceiling` analysis); also per-edge ICC and a G-theory
   split.
4. **Disattenuate FC→SC** against this SC ceiling, exactly as SC→FC was disattenuated
   against the FC ceiling (S2.3). Recompute the "% of reproducible signal captured" for the
   *forward* direction.

**Payoff.** Converts "we can't measure SC reliability" into "here is a (optimistic)
lower-bound ceiling," closes most of the acknowledged 5% gap, and lets the asymmetry be
stated as reliability-corrected in *both* directions — which would make the directional
claim much harder to attack.

**Caveats.** Split-half within one scan is an *upper* bound on reliability (shares
session/registration noise), so it under-estimates true test-retest noise — state it as a
lower bound on the *gap*. Requires the raw dMRI + tractography pipeline, not just the
connectome matrices; that is the cost driver.

### E2 · Head-size / volume normalization of SC — feasibility **B**

**Why.** Brain volume mechanically inflates streamline counts, so the bv→SC arm of the
"double dissociation" (§3) may be a tractography-construction artifact, not biology.

**Design.** Rebuild SC under (a) ICV normalization, (b) total-streamline-count
normalization, (c) per-subject demeaning, and re-run the bv→SC and demo→FC reconstruction
rows. If the bv→SC edge over demo→SC collapses under normalization, reframe the dissociation.

**Feasibility.** The connectomes and bv features are in hand; this is a re-derivation +
re-run of a handful of grid rows, not a new acquisition.

### E3 · Imputation-harm controls (finish #3) — feasibility **B**

**Why.** To prove pred_FC's sub-baseline harm is "structured misinformation," not input
geometry, even after the direction-specific argument in Part 1.

**Design.** Add three control inputs to the downstream grid for cognition:
(a) the **group-mean** connectome (same for all subjects),
(b) a **phase-randomized / spectrum-matched** surrogate of pred_FC,
(c) a **random Gaussian** connectome with pred_FC's covariance.
If these controls do *not* depress cognition below baseline while pred_FC does, the harm is
genuinely in the imputed content. Cheap: same pipeline, three new input rows × targets ×
seeds × parcellations.

### E4 · Baseline decomposition: what carries `bv+demo`? — feasibility **A/B**

**Why.** CogCryst is heavily education/SES-driven. If `demo` already contains
education/income, "beat bv+demo" is a much stronger bar than age+sex — and the reader can't
currently tell. Conversely, FC's CogCryst lift might be FC re-encoding education-correlated
structure.

**Design.** (A) Report the feature composition of `bv` and `demo` and each block's solo
cognition prediction. (B) Add a `bv+demo+education` baseline and re-test the FC lift over
*that*. Feasibility A if the feature table exists in the grid config; B if education must be
re-pulled from the HCP restricted data.

### E5 · Family-block permutation for the AUC nulls — feasibility **B**

**Why.** Sibling-pair AUCs use 1,254 non-independent pairs (each subject in many pairs), so
the permutation null/CIs are likely over-confident; and obs_SC MZ-AUC≈0.999 warrants an
explicit no-identity-leakage check.

**Design.** Re-run the family-mechanism permutation **shuffling family labels at the family
level**, not the pair level; report family-block CIs. Add a leakage probe confirming the
same subject never appears on both sides of a pair. Needs the pair/family-ID arrays (present
in `family_mechanism`), so a re-run of the permutation, not new data.

### E6 · A single clean held-out test (de-pseudo-replicate the seeds) — feasibility **A/B**

**Why.** The 10 "seeds" are family-aware resplits of one cohort with overlapping training
sets, so seed-level std and "p≤0.001 across seeds" overstate precision. Useful to show the
headline numbers survive one genuinely held-out split.

**Design.** Carve a locked test set once (family-disjoint from all training), report the
headline FC→SC ratio and the obs_FC CogCryst lift on it with a single honest CI. Mostly a
re-run / re-split.

### E7 · Alternative individual-deviation metrics — feasibility **A**

**Why.** Everything hinges on `demeaned_pearson`. Show the asymmetry and the utility wall
are not metric-specific.

**Design.** Recompute the headline rows under distance correlation, top-k edge-overlap, and
cosine on residualized connectomes. Cheap if predictions are cached; otherwise a light
re-run.

### E8 · External-cohort replication — feasibility **D**

**Why.** Single cohort (HCP-YA). The biggest generality gap; reviewers will ask.

**Design.** Replicate the two load-bearing results (FC→SC>SC→FC ratio; the bv+demo cognition
wall) in HCP-Aging, ABCD, or a UK Biobank subset. Expensive (new ingestion + harmonization)
but the single highest-impact addition for generality.

### E10 · pred_FC vs bv+demo information overlap ("do those two match?") — feasibility **B**

**Why.** A natural worry: `pred_FC` (FC imputed from SC) and `bv+demo` might carry the same
subject information, so `pred_FC+bv+demo` would be a redundant/collinear input rather than a
real test. Two facts already bound this:
1. **By construction they cannot be circular.** `pred_FC = capped_pca_pls(SC_tr, SC_te,
   FC_tr)` (`make_handoff_artifacts.py`) — it is a pure function of SC; `bv+demo` is never an
   input to the imputation. No bv+demo information is baked in.
2. **Empirically they do not match.** If `pred_FC` were a redundant copy of `bv+demo`, the
   regressor would ignore it and `pred_FC+bv+demo ≈ bv+demo`. Instead, for Glasser CogCryst,
   `bv+demo` = 0.354, `pred_FC+bv+demo` = 0.314 (**below** baseline), while the
   identically-constructed `pred_SC+bv+demo` = 0.415 (**above**). Redundancy predicts no
   change; the observed drop means `pred_FC` injects non-redundant, actively harmful variance,
   and the opposite sign for `pred_SC` rules out a generic collinearity artifact.

**Residual question worth one direct measurement.** `pred_FC` is low-rank (≤64-dim PLS
projection of SC), and part of that subject-varying content *could* still align with `bv+demo`
(both predict FC). The arrays differ but the information could partially overlap.

**Design (cheap, once artifacts are pulled).** With `outputs/artifacts/{parc}/seed*/pred_FC_*.npy`
and the split `bvdemo` arrays: (a) regress each `pred_FC` PLS component on `bv+demo` and report
the variance of `pred_FC` explained by `bv+demo` (R²); (b) canonical correlation between
`pred_FC` and `bv+demo`; (c) re-run the cognition row using `pred_FC` **residualized on
bv+demo** as the added block — if the harm persists on the residual, it is genuinely SC-derived
misinformation, not bv+demo overlap. Needs the .npy artifacts (cluster-generated, not on disk
locally) + a short script.

### E9 · Gold-standard SC ICC via HCP retest dMRI — feasibility **D**

**Why.** The PC-mechanism localization (S7) currently partials a strength+distance+FC-rel
proxy, not a true per-edge SC ICC. The proxy explains 41% of |PC3|; a real ICC would settle
whether the residual visual/DAN localization is biological.

**Design.** Stand up the HCP test-retest dMRI release (~45 subjects scanned twice), compute
per-edge SC ICC, and partial *that* out of |PC3|. Out of scope for the current data; flagged
as the rigorous future version of S7.

---

## Priority ordering

1. **E1 (split-half SC reliability)** — converts the one acknowledged interpretive gap into
   a measured bound. Highest scientific value. (C)
2. **E3 + E2** — finish the imputation-harm and head-size controls; both directly harden or
   honestly soften quotable sentences. (B)
3. **E4** — baseline composition; determines how strong "beat bv+demo" really is. (A/B)
4. **E5, E6, E7** — statistical-rigor re-runs; cheap credibility. (A/B)
5. **E8, E9** — generality and gold-standard reliability; high impact, high cost. (D)

The throughline: the negative claims are now extremely well-defended (S1–S8 + the equivalence
analysis). The remaining work is concentrated on (i) the SC-side reliability gap, (ii) the
two construction confounds a connectomics reviewer reaches for first (head size, baseline
composition), and (iii) the fragility of the one positive claim — all of which the items
above target directly.
