# Adel's Summer 2026 Work — Ledger & Pointer Document

**Contributor:** ans9868 (Adel)
**Period:** 2026-05-25 → 2026-06-26 (142 commits)
**Where the work lives now:** branch [`adel-temp`](../../.git) — not on `main`
**Commit range:** `6ec3482` (first commit) .. `98b5821` (branch tip, includes 3 parked
follow-on commits plus a consolidation commit pulling in infra scripts from the transient
`adel-temp-torch` sync branch)
**Rolled off `main` on:** 2026-09-21, back to `f7e9fbf` (Alex's last commit, 2026-05-20)

This document exists because `adel-temp` carries ~235MB / 867 changed files of largely
self-contained side research that doesn't belong permanently on `main`. It is the pointer
so that work isn't lost or forgotten — anyone (human or agent) who needs to know what
happened over the summer, or wants to pull a specific piece back into `main`, should start
here.

---

## Table of Contents

1. [What this was](#1-what-this-was)
2. [Where things live on `adel-temp`](#2-where-things-live-on-adel-temp)
3. [Findings ledger (F1–F10, C1/C2)](#3-findings-ledger-f1f10-c1c2)
4. [Sanity-check discipline (why the findings are trustworthy)](#4-sanity-check-discipline)
5. [The reproduction grid](#5-the-reproduction-grid)
6. [Bloat inventory (why this isn't on `main`)](#6-bloat-inventory-why-this-isnt-on-main)
7. [Branch/commit hygiene notes](#7-branchcommit-hygiene-notes)
8. [What might be worth pulling into `main`](#8-what-might-be-worth-pulling-into-main)
9. [How to resume this work](#9-how-to-resume-this-work)

---

## 1. What this was

This was **not incremental development on Conn2Conn's existing SC→FC benchmarking
framework**. It's a self-contained research arc that reused Conn2Conn's data-loading
infrastructure (`main.py`, `models/`, `data/`) without modifying it, and pursued a new
question:

> Cross-modal connectome prediction is **directional** (FC→SC ≫ SC→FC), **linearly
> saturated** (bigger models/more data don't help), and **FC is the cognition ceiling**
> (SC and imputed connectomes add nothing or hurt) — yet predicted connectomes still carry
> **heritable family signal**, though you cannot optimize one connectome to both
> reconstruct and identify.

The pivot to this direction (FC→SC, away from the repo's default SC→FC framing) was
**directed by Alex**, not unsanctioned scope drift — see `slack-msg.md` on `adel-temp`:
"I think you exhausted SC→FC-from-rest and proved its ceiling, which justifies pivoting to
FC→SC + multi-task... I honestly think your results here are already publishable."

The intended output was a **methods/redirect paper** (not a leaderboard paper): its
contribution is (1) a required `bv+demo` (brain-volume + demographics) baseline the field
should report and usually doesn't, and (2) a rigorous demonstration that the cognition
ceiling from connectome data is structural, not a modeling or sample-size limit — see
`preprint/STRUCTURE.md` for the drafted paper shape.

Two working titles floated (`preprint/STRUCTURE.md`):
- *"What Cross-Modal Connectome Prediction Does and Does Not Buy"*
- *"Connectome Translation Is Reproducible but Not Sufficient: Baselines, Ceilings, and
  Utility in FC-SC Prediction"*

## 2. Where things live on `adel-temp`

All paths below are relative to repo root, on the `adel-temp` branch (`git show
adel-temp:<path>` or `git checkout adel-temp -- <path>` to pull a specific file/dir).

| Directory | Size | What it is |
|---|---|---|
| `notebooks-FC_to_SC-experimental/` | 133 MB | The **exploratory layer** — original ad-hoc research (mostly Glasser parcellation, single-seed runs) where findings F1–F10 were first discovered. Start with `MASTER_FINDINGS.md`. |
| `dev-notes/` | 50 MB | Session handoffs and context dumps. **Start here for orientation**: `PROJECT-HANDOFF-2026-06-22.md` is the single best "what is this project" document Adel wrote. Also contains ~40MB of reference-paper PDFs and LaTeX build junk (see §6). |
| `preprint/` | 45 MB | Manuscript-in-progress. Canonical source: `manuscript.md`, `supplement.md`, `STRUCTURE.md` (paper map), `README.md`. Four redundant rendered-draft trees (`preprint-claude/`, `preprint-claudev2/`, `preprint-codex/`, `preprint-codexv2/`) sit alongside these — see §6. |
| `reproduction/` | 6.2 MB | The **confirmatory layer** — a deterministic, parameterized re-run of the exploratory findings across both parcellations (Glasser, 4S456Parcels) × 10 frozen seeds, with completeness and leak-check guardrails. The cleanest, most disciplined part of the whole arc. Start with `README.md` and `HANDOFF_FOR_ANALYSIS.md`. |
| `planning/` | 62 KB | `Overall_plan.md`, `cleanup_plan.md` (an unexecuted plan to reorganize `notebooks-FC_to_SC-experimental/` — Adel himself flagged the disorganization), `reproducibility_and_grid_plan_{theory,todo,runlog}.md`, `family_mechanism_grid_plan.md`. |
| `slack-msg.md` | — | Record of Alex↔Adel Slack messages — the SC data-processing sign-off and the FC→SC pivot directive. Useful provenance, not research output. |
| `output/pdf/Updated_Results_Section.pdf` | — | A single stray output PDF, unclear if superseded by `preprint/`. |
| `.claude/` | — | Claude Code session config (`settings.local.json`, `scheduled_tasks.lock`) from Adel's own agent-assisted workflow. Not project content. |

## 3. Findings ledger (F1–F10, C1/C2)

Full detail and CSV/MD evidence pointers: `notebooks-FC_to_SC-experimental/MASTER_FINDINGS.md`
on `adel-temp`. Each entry below follows **Research question → Experimental setup → Main
takeaways**, in the paper's logical order (baseline → asymmetry → ceiling → imputation
utility → family signal/tradeoff → exploratory mechanism). Status tags follow
`MASTER_FINDINGS.md`'s own confidence grading — several magnitudes are flagged there as
needing bootstrap CIs before any write-up relies on them; that caveat is preserved per
entry rather than repeated.

All findings share the same substrate: HCP-YA, Glasser parcellation (360 regions, 64,620
edges) unless noted as replicated on 4S456Parcels (456 regions), ~683 train / ~195 test
subjects, family-aware splits, primary metric `demeaned_pearson` (per-subject cosine of
connectome minus training group-mean — raw Pearson is population-mean-dominated and not
used for claims).

---

### C1 — The `bv+demo` baseline requirement
**Status:** confirmatory (primary contribution #1) · **Evidence:** `model_overviews/results/downstream_prediction_phase2/{aggregate,perm_vs_bvdemo}.csv`, `model_overviews/results/FINDINGS.md`

- **Research question:** How much of any connectome→cognition prediction is actually just
  recoverable from cheap subject-level information (brain volume, age, sex, race/ethnicity)
  that has nothing to do with the connectome itself?
- **Experimental setup:** Train a T1-derived brain-volume + demographics (`bv+demo`)
  regressor for cognition (CogCryst/CogFluid/CogTotal) as a baseline, then compare it
  against predictors using observed FC, observed SC, or combinations, using paired
  permutation tests (not overlapping-CI heuristics) to test whether each modality clears
  the baseline.
- **Main takeaways:** `bv+demo` alone reaches CogCryst r=0.353 — already close to what
  many connectome studies report as their headline number. Adding observed FC lifts this
  to 0.434 (the only real gain); adding observed SC *drops* it to 0.258, below the free
  baseline. Recommendation adopted as a reporting standard for the rest of the project:
  **any SC↔FC cognition-prediction study should report the `bv+demo` baseline explicitly.**

### F1 — The FC→SC ≫ SC→FC asymmetry
**Status:** confirmatory, but an established effect (Krakencoder), not a novel result — used as scaffolding · **Evidence:** `model_overviews/results/{notebook_snapshot/phase1/section2_asymmetry.csv, Exp7_stringent_ratio_10_seed/*, Step11_grand_10seed/aggregate_ratios.csv}`

- **Research question:** Is cross-modal connectome prediction symmetric, or is one
  direction (FC predicting SC) genuinely easier than the other (SC predicting FC) — and if
  so, is that a real directional-information effect or an artifact of one modality being
  noisier as a *target*?
- **Experimental setup:** Closed-form PCA→PLS prediction in both directions, evaluated with
  `demeaned_pearson` across 10 seeds and multiple bases (raw, anatomy/demographics-stripped
  residuals), plus a within-modality oracle (FC→FC, SC→SC) to test whether SC is simply a
  "noisier target" than FC.
- **Main takeaways:** FC→SC (≈0.134) beats SC→FC (≈0.085) by ~1.5×, survives
  anatomy+demographics stripping (~1.4×), holds across all 6 evaluation metrics, and
  reproduces Krakencoder's published ratio (0.16/0.09). Critically, at the oracle level
  FC→FC and SC→FC are equally self-predictable (~1.0), so the asymmetry is **not** a
  target-reliability artifact — it's genuine directional information. Robustness confirmed
  independently across reduction method (A1: full PLS, learned PCA, JL projections),
  estimator choice (A2: PLS vs BayesianRidge vs KernelRidge), and PCA/PLS dimensionality
  (A3).

### F2 — Modality dissociation (SC↔anatomy, FC↔demographics)
**Status:** confirmatory · **Evidence:** `model_overviews/results/notebook_snapshot/phase1/section1_baselines.csv`

- **Research question:** What predicts each modality best when using only non-connectome
  subject information — brain anatomy (volume) or demographics?
- **Experimental setup:** Predict SC and FC separately from brain-volume-only and
  demographics-only inputs, compared across 10 seeds.
- **Main takeaways:** SC is anatomy-driven (bv→SC 0.162, anatomy wins over demographics);
  FC is demographics-driven (demo→FC 0.114, demographics wins over anatomy) — a clean,
  10/10-seed double dissociation on the same data. Used later as part of the F1 robustness
  argument (ruling out that the asymmetry is just anatomy leaking into both sides equally).

### C2 — The cognition ceiling is structural (F5 + F9 + F10 as one argument)
**Status:** confirmatory core — "the confirmatory core... well-powered negatives across four axes" · **Evidence:** listed per sub-finding below

This is presented in `MASTER_FINDINGS.md` as three findings that must be read together,
each closing a different escape route from the same question.

- **Research question:** Given that FC beats the `bv+demo` baseline for cognition (C1) and
  SC does not, is that gap fixable — by richer structural features, a bigger/nonlinear
  model, or more training data — or is it structural to this data regime?
- **Experimental setup (three orthogonal probes):**
  - *F5 (imputation transfer):* does a *predicted* connectome (SC imputed from FC, or vice
    versa) carry the source modality's downstream utility, or does it lose it in the
    round-trip?
  - *F9 (richer structural features):* does replacing simple SC streamline counts with
    named-bundle tractography (region-to-target, "r2t") features improve FC prediction or
    cognition?
  - *F10 (nonlinearity / scale):* four independent null tests — nonlinear model classes
    (KernelRidge, HistGradientBoosting), a residual-boost design (hand the nonlinear model
    the correct linear answer for free and see if it still can't do better), a cross-modal
    "sink" model combining FC×SC×r2t interactions, and a data-scaling curve from n=100 to
    n=683.
- **Main takeaways:** All three close their respective escape routes. **F5:** predicted SC
  (from FC) actually beats real SC for cognition (1.25–1.45×), but predicted FC (from SC)
  loses 40–50% of observed FC's utility — a connectome carries its *source* modality's
  information, imputation doesn't rescue the SC-underperforms problem. **F9:** r2t
  tractography features predict FC *worse* than plain streamline counts (0.049 vs 0.085
  demeaned-r) and add nothing marginal over SC (Δ=−0.001, p=0.98) — streamline-count SC is
  already a sufficient statistic; better structural features don't close the gap. **F10:**
  all four nonlinear/scaling probes are null — the data-scaling curve is flat from
  n=100→683, so this isn't a sample-size-limited effect either. Combined: **you can't
  recover FC's advantage over SC with better features, a bigger model, or more data in this
  regime.** Explicitly framed as a *redirect*, not a dead stop — the open question is
  whether the same ceiling holds in clinical, longitudinal, or much-larger-n cohorts (see
  `MASTER_FINDINGS.md` "Where the signal might actually live").

### F3 — Imputation utility asymmetry
**Status:** high confidence on direction; derived ratios lack CIs · **Evidence:** `model_overviews/results/downstream_prediction_phase2/aggregate.csv`

- **Research question:** If you impute one connectome modality from the other, does the
  ~1.5× reconstruction asymmetry (F1) carry over proportionally into downstream cognitive
  utility, or does it amplify/shrink?
- **Experimental setup:** Compare cognition prediction using observed vs. predicted SC and
  FC (`obs_SC`, `pred_SC`, `obs_FC`, `pred_FC`) as inputs, same estimator/seed pairing.
- **Main takeaways:** Predicted SC (from FC) *beats* real SC for cognition (1.25–1.45×);
  predicted FC (from SC) *loses* 40–50% of real FC's utility. The reconstruction asymmetry
  (~1.5×) becomes a **2.06–2.54× downstream-utility asymmetry** — imputation is directional
  in a stronger sense than raw reconstruction accuracy suggests. Flagged: the 2.06–2.54×
  ratios are derived (pred/obs) point estimates without propagated bootstrap CIs.

### F4 — FC's cognition signal is real; SC's is mostly demographics
**Status:** high confidence on direction; survival-fraction percentages are derived ratios without CIs · **Evidence:** `model_overviews/results/downstream_prediction_phase2/{aggregate,perm_vs_bvdemo}.csv`

- **Research question:** Of whatever cognition signal FC and SC each carry, how much
  survives once the `bv+demo` confound is explicitly removed?
- **Experimental setup:** Residualize each modality's cognition prediction against the
  `bv+demo` baseline and measure the surviving fraction, per cognition target (Total,
  Fluid, Crystallized).
- **Main takeaways:** FC retains 60% / 60% / 73% of its raw signal after `bv+demo` removal
  (Total/Fluid/Cryst); SC retains only 27% / 18% / 31% — most of SC's apparent cognition
  signal was demographic confound to begin with, while FC's is largely independent of it.

### F6 — Predicted connectomes carry heritable family signal
**Status:** medium-high — AUCs carry permutation CIs, but the underlying table is from an older run flagged for regeneration; the reported gap (+0.247) is a derived difference without a propagated CI · **Evidence:** `model_overviews/results/family_structure_phase2/aggregate_auc.csv`

- **Research question:** Even though predicted connectomes don't reliably help cognition
  prediction (F3/F5), do they still carry meaningful biological signal — specifically,
  family/genetic structure?
- **Experimental setup:** Train a sibling-vs-stranger classifier (AUC) on different
  variants of predicted SC (raw, residualized against `bv+demo`, combined-objective) and
  compare against a `bv+demo`-only baseline classifier.
- **Main takeaways:** `pred_SC_resid_bvdemo` (predicted SC with the `bv+demo` component
  removed) separates siblings from unrelated pairs at AUC **0.810**, far above the
  `bv+demo`-only baseline (0.563) — a +0.247 gap. Predicted connectomes are not useless;
  they carry family-specific structural signal beyond shared anatomy, even where they don't
  help cognition prediction.

### F7 — The predictor/identifier tradeoff
**Status:** high — confirmed not a bug via follow-up perturbation checks · **Evidence:** `model_overviews/results/family_structure_phase2/aggregate_auc.csv`, `local_results/further_exploration/depth2_robustness_sensitivity/combined_followup.csv`

- **Research question:** Can a single predicted connectome be simultaneously good at
  reconstruction/cognition (F6's use case) *and* good at family identification, or do these
  objectives compete?
- **Experimental setup:** Compare sibling-identification AUC for `combined_pred_SC` (the
  variant optimized for reconstruction/cognition, i.e. still containing `bv+demo` content)
  against `pred_SC_resid_bvdemo` (the F6 variant, with `bv+demo` content explicitly
  removed), plus a 4-perturbation robustness follow-up to rule out a bug.
- **Main takeaways:** `combined_pred_SC` collapses to chance on sibling separation (AUC
  0.505, n.s.) — the `bv+demo` content that helps reconstruction and cognition *drowns out*
  the within-family signal that made F6 work. Confirmed structural (not an artifact) via
  the 4-perturbation follow-up. **You cannot reconstruct and discriminate from one loss
  objective simultaneously** — this becomes the motivating question for F7b/F7c/F7d.

### F7b — The tradeoff's mechanism: estimator-level shrinkage (BR vs PLS)
**Status:** high — Glasser × 10 seeds with bootstrap CIs, mechanism measured on identical splits; 4S456 replication deferred · **Evidence:** `../reproduction/br_family/outputs/family_auc_br.csv`, `../reproduction/br_imputation/outputs/{probe_shrinkage,downstream_br}.csv`

- **Research question:** Does the reconstruct/identify tradeoff (F7) depend on which
  closed-form estimator does the imputation, and if so, what's the actual mechanism?
- **Experimental setup:** Swap the imputation estimator from PLS to BayesianRidge (BR) —
  the stronger reconstructor — and re-measure both reconstruction quality and sibling-AUC
  identification, then directly measure per-PC amplitude retention and direction-recovery
  correlation down the target-component spectrum for both estimators.
- **Main takeaways:** BR reconstructs better (FC→SC demeaned-r 0.166 vs PLS's 0.136) but is
  a *worse* identifier (`pred_SC_resid_bvdemo` sibling AUC drops to 0.763 vs PLS's 0.810).
  The measured mechanism: BR's evidence-tuned shrinkage flattens the low-variance target-PC
  tail toward the group mean, retaining only 0.182× of the true individual-deviation
  amplitude vs PLS's 0.306× (per-PC amplitude collapses 0.384→0.044 for BR vs a nearly-flat
  0.519→0.382 for PLS — ~9× more tail amplitude retained by PLS). **The same shrinkage that
  wins the high-variance bulk (reconstruction/cognition) erases the low-variance
  idiosyncratic tail that fingerprints families** — the estimator's objective is one
  concrete knob on the F7 tradeoff, not just an abstract one.

### F7c — Custom objectives can't move the tradeoff (negative result)
**Status:** high confidence — clean negative, BR/PLS baselines reproduce prior runs byte-for-byte; Glasser × 5 only, two objective families untried and parked · **Evidence:** `../reproduction/obj_functions/outputs/{scorecard,diag_ungated}.csv`

- **Research question:** If the tradeoff is genuinely structural (not just an estimator
  artifact), can a custom-designed loss function re-aim the frontier — e.g. explicitly
  maximize identity-preservation or cognition-relevance instead of raw reconstruction?
- **Experimental setup:** Built and ran four custom imputation objectives on a BR backbone
  (Glasser × 5 seeds): an identity-maximizing objective (gated and ungated per-PC amplitude
  restoration) and two cognition-maximizing objectives (per-PC cognition weighting, raw and
  `bv+demo`-residualized target), scored simultaneously on reconstruction, sibling-AUC, and
  CogCryst lift.
- **Main takeaways:** **None of the four custom objectives beat the existing BR/PLS
  baselines on the axis they were built for.** BR remained the reconstruction champion
  (0.163), PLS the identity champion (sib AUC 0.812); all four custom objectives landed at
  or below them. A follow-up ablation pinned the reason: scaling BR's low-variance tail
  amplitude recovers identity only up to BR's own level (0.774, never PLS's 0.812) while
  *cratering* reconstruction (0.163→0.108) — because PLS's identity advantage comes from
  *directionally correct* tail predictions (real covariance structure), not just more tail
  amplitude; BR's tail is closer to directional noise (corr≈0.04) that can't be scaled into
  a fingerprint. **The reconstruct/identify tradeoff is a wall, not a tunable knob** —
  reinforces the redirect framing (stop engineering past a wall that doesn't move).

### F7d — Latent-direct: the round-trip was quietly costing cognition signal
**Status:** medium — clean negative on the objectives question (consistent across all 5 objectives); the pipeline-artifact finding is Glasser × 3 seeds only, not seed-matched against the round-trip baseline · **Evidence:** `../reproduction/latent_direct/outputs/scorecard_latent.csv`

- **Research question:** F7c's objectives were evaluated on connectomes reconstructed via
  an inverse-PCA round-trip (latent → back to full connectome space). Does skipping that
  round-trip — classifying directly on the task-tuned latent — change the negative verdict,
  or was the round-trip itself distorting the comparison?
- **Experimental setup:** Re-ran the F7c objectives classifying directly on the latent
  (no inverse-PCA → re-PCA step) on Glasser × 3 seeds, across FC2SC/SC2FC/FCSC arms.
- **Main takeaways:** Two separable results. (1) **The objectives are still a dead end**
  — confirms F7c: the identity objective (sib AUC 0.748) still loses to plain BR (0.767)
  and PLS (0.808); the best cognition objective beats plain BR by only +0.033, within
  seed-to-seed noise, and never approaches observed FC's ceiling. (2) **But the
  inverse-PCA round-trip itself was silently costing cognition signal** — skipping it
  recovers +0.037…+0.069 CogCryst across every objective tested, with ~zero change to
  sibling-AUC (an isometry: PCA components are orthonormal, so the round-trip only distorts
  the *cognition regression basis*, not identity). This means F5's "imputed connectomes
  don't transfer cognition" finding was **substantially a round-trip artifact, not a
  ceiling on the information itself** — reconstructed-SC cognition climbs from −0.010
  (round-trip) to +0.115 (latent-direct), ≈86% of the observed-FC ceiling. Flagged
  reporting fix for the paper: classify on the latent, not a round-tripped connectome.

### F8 — Exploratory: a fragile spectral mechanism (SC-PC3)
**Status:** medium, explicitly downgraded pending 4S456 replication — "one parcellation away from being a possible atlas artifact"; already produced one false positive · **Evidence:** `further_exploration/{depth1_spectral_mechanism,depth1.1_pc_stability_and_confounds}/*.csv`, `figures/output_glassbrain/pc3_glassbrain.png`

- **Research question:** Does the FC→SC asymmetry (F1) localize to a specific,
  interpretable structural mechanism — and if so, is that mechanism itself heritable?
- **Experimental setup:** Decompose SC into principal components; test each component for
  (a) stability across 10 seeds, (b) FC-predictability, (c) heritability via sibling/MZ/DZ
  AUC, (d) spatial/anatomical localization, with an explicit cross-seed alignment step
  built specifically to catch single-seed false positives.
- **Main takeaways:** A specific low-variance SC mode (PC3 in Glasser, migrates to PC4 in
  4S456) is stable across seeds (cos 0.89), modestly FC-predictable (R²≈0.22), weakly
  heritable (sibling AUC≈0.58, barely above chance; MZ 0.71 > DZ 0.60 > sibling 0.58
  gradient — the right ordering for a genetic signal), and spatially concentrated in
  visual/dorsal-attention cortex (62% of localization energy in 1% of edges, 8–12×
  enrichment). **Explicitly hedged**: this is small-effect, hypothesis-generating work, and
  the same pipeline already produced one false positive — an earlier "PC2 carries it,
  R²=0.26" result that did not replicate across seeds (median R²=0.017) and was only caught
  by the cross-seed alignment check. 4S456Parcels replication (needed to rule out a
  Glasser-specific atlas artifact) was never run.

### Reliability ceilings (Ceiling A / Ceiling B / A6, closed)
**Status:** A6 explicitly marked "closed" in `MASTER_FINDINGS.md` · **Evidence:** `sanity_checks/noise_sanity_check/{findings_noise.md, outputs/*.csv}`

- **Research question:** Is the FC→SC / SC→FC gap (C2) actually just an artifact of FC
  being a noisy measurement — i.e. would a perfectly clean SC→FC predictor look fine if FC
  itself weren't noisy?
- **Experimental setup:** Measure FC's own cross-session (day1↔day2) reproducibility
  (Ceiling A, the data noise floor) against the within-modality model oracle (Ceiling B,
  FC→FC / SC→SC), then directly test whether a subject's SC→FC prediction quality
  correlates with that subject's own FC reliability.
- **Main takeaways:** Ceiling A (cross-session reproducibility) ≈ 0.49; Ceiling B (model
  oracle) ≈ 0.63–0.67 — the correct ordering (models can't exceed the reproducibility
  floor, and don't). Decisively, a subject's SC→FC prediction quality is **uncorrelated**
  with their own FC reliability (r=0.01, vs. the `bv+demo` baseline's r=0.15), and
  reliability-filtering doesn't improve achieved accuracy — so **the SC→FC gap is not an
  FC-measurement-noise artifact.** Marked closed; the one remaining open item (SC-side
  test-retest reliability) is data-blocked, not run.

### Summary: what's concrete vs. what remains open

**Concrete (confirmatory, hold up across seeds/parcellations/estimators/robustness
checks):** the `bv+demo` baseline requirement (C1); the FC→SC ≫ SC→FC directional
asymmetry (F1) and its anatomy/demographics dissociation (F2); the structural,
non-fixable cognition ceiling — FC is the wall, richer structural features and
bigger/nonlinear/data-scaled models don't move it (C2/F5/F9/F10); that the SC→FC gap is
not an FC-measurement-noise artifact (Ceiling A/B/A6, closed); and the reconstruct/identify
tradeoff being a genuine wall rather than a tunable knob — four custom-built loss
objectives all failed to move it (F6/F7/F7b/F7c). **One finding needs re-checking before
being treated as settled**: F7d showed the inverse-PCA round-trip used throughout the
project was itself quietly destroying cognition signal in imputed connectomes, which means
F5/C2's "imputation doesn't transfer cognition" framing is **partly a pipeline artifact,
not a pure ceiling** — this should be the first thing revisited if this work resumes, since
it could soften one of the project's central negative claims. **Open / exploratory (do not
anchor on these without more work):** the F8 spectral mechanism — a specific SC component
that's FC-predictable and weakly heritable — is hedged as hypothesis-generating, already
produced one false positive on a neighboring component, and its required
second-parcellation (4S456) replication was never run; several supporting-finding
magnitudes (F3's utility ratios, F4's survival fractions, F6's AUC gap) have directionally
solid results but lack propagated bootstrap confidence intervals; and the manuscript itself
(`preprint/`) is an unfinished draft, not a submitted or peer-reviewed product.

## 4. Sanity-check discipline

This is worth preserving as a template even if the specific findings aren't ported:
frozen train/val/test splits (derived once, asserted everywhere, versioned in
`reproduction/splits/`), per-block feature scaling to avoid a connectome-PCA block
swamping small demographic features, a completeness manifest that hard-fails on missing
cells, and a demographic-leak guardrail (`ok` / `EXPECTED_SIGNAL` / `EXEMPT_FLAGGED` /
`LEAK_FAIL`) — the full 13,640-cell grid produced zero `LEAK_FAIL`.

## 5. The reproduction grid

`reproduction/` on `adel-temp` is a deterministic, parameterized re-run of F1–F5 and
Ceiling B (within-modality oracle) across **2 parcellations × 10 frozen seeds × 3
estimators** (pca_pls, bayesian_ridge, kernel_ridge 3×3 sweep = 11 estimator-variants),
13,640 cells, all verified complete and finite. CSV outputs
(`reproduction/outputs/{reconstruction,downstream,leak_verdict}.csv`) are the source of
truth; W&B (`conn2conn-fc-to-sc-reproduction` project) and Markdown reports are downstream
views. Separately, `reproduction/family_mechanism/`, `reproduction/br_family/`,
`reproduction/br_imputation/`, and `reproduction/obj_functions/` extend the grid to
F6/F7/F7b/F7c/F7d.

This is the one part of the summer's work that reads as reusable infrastructure rather
than one-off exploration — see §8.

## 6. Bloat inventory (why this isn't on `main`)

142 commits, 867 files changed, ~195,700 insertions / 66 deletions — almost purely
additive, essentially none of Conn2Conn's existing `main.py`/`models/`/`data/` was touched.
Roughly 235MB of new content across:

| Source | Size | Issue |
|---|---|---|
| `dev-notes/research-papers/*.pdf` | ~40 MB | Reference-paper PDFs committed directly to git; belong in a reference manager, not version control. |
| `dev-notes/grid_models*.{aux,fdb_latexmk,fls,log,out,synctex.gz}` | small but numerous | LaTeX build artifacts — should never be committed (regenerable, machine-specific). Same pattern repeats in `preprint/`. |
| `preprint/preprint-{claude,claudev2,codex,codexv2}/` | ~44 MB | Four redundant rendered-draft trees (LaTeX sources + ~196 PNG figures + ~151 PDFs including slide decks) layered on top of the actual canonical source (`preprint/manuscript.md` + `supplement.md`). |
| `notebooks-FC_to_SC-experimental/model_overviews/crossmodal_pca_pls_closed_form_overview.ipynb` | 83 MB | Single notebook, almost certainly carrying uncleared cell outputs (every edit re-diffs the full blob). |
| `notebooks-FC_to_SC-experimental/EDA/track_test_retest_model.ipynb` | 28 MB | Same issue. |
| `notebooks-FC_to_SC-experimental/model_overviews/TEMP-BACKUP-results/` | — | A directory literally named "TEMP-BACKUP" was committed. |

`reproduction/` (6.2MB, almost entirely `.py`/`.sbatch`/reasonably-sized CSVs) is the
exception — clean and disciplined throughout.

Note: removing this from `main`'s working tree (the rollback done here) does **not**
shrink `main`'s `.git` history — the ~235MB of objects were already pushed and fetched
into every clone's object database before the rollback. Reclaiming that would require a
history rewrite (`git filter-repo` or similar) across every clone, which is a separate,
more disruptive decision — not attempted here.

## 7. Branch/commit hygiene notes

- All commits in this repo (both Alex's and Adel's) are authored under a shared HPC
  account's git identity (`Alexander Ratzan <asr655@...>` for nearly everything, one
  `ans9868@...` pair on the now-absorbed `adel-temp-torch` commits) — `git log --author`
  cannot reliably distinguish contributors here. Attribution was intentionally left as
  committed (not rewritten) so Adel remains visible as a contributor in the repo's
  history, per Alex's preference.
- `origin/adel-temp-torch` was a transient, auto-generated git sync branch (the
  Torch↔laptop bridge described in `dev-notes/PROJECT-HANDOFF-2026-06-22.md` §5) frozen at
  Adel's first commit. Its only reusable content — 3 local HPC infra scripts
  (`apptainer.sh`, `conda_activate.sh`, `kraken_env.local.yml`, pinned to
  `/scratch/ans9868/...` paths) — was pulled into `adel-temp` (commit `98b5821`). The rest
  of that branch (a stray `test_notebook.ipynb`, a mid-edit notebook autosave) was left
  behind as disposable sync noise.
- `origin/adel-temp` (before consolidation) was 3 commits ahead of `main`'s old tip
  (`ce8d599`): a parked dead-end experiment (`latent_direct`, F7d — see §3). These are
  now folded into the consolidated `adel-temp`.

## 8. What might be worth pulling into `main`

Not yet decided — flagged here as candidates for a follow-up review, roughly in order of
how load-bearing / reusable they look:

1. **`reproduction/`'s methodology** (frozen-split discipline, leak-check guardrail,
   completeness-manifest pattern) — could generalize to Conn2Conn's existing SC→FC model
   family even without adopting the FC→SC findings themselves.
2. **The `bv+demo` baseline concept (C1)** — a cheap, principled baseline any future
   Conn2Conn cognition-prediction work should report.
3. Everything else (preprint, notebooks, dev-notes) — parked on `adel-temp`, revisit only
   if/when this research direction is actively resumed.

## 9. How to resume this work

```bash
git fetch origin
git checkout adel-temp   # or: git checkout -b resume-fc-sc origin/adel-temp
```

Read in this order: `dev-notes/PROJECT-HANDOFF-2026-06-22.md` →
`notebooks-FC_to_SC-experimental/MASTER_FINDINGS.md` → `reproduction/README.md` →
`reproduction/HANDOFF_FOR_ANALYSIS.md` → `preprint/STRUCTURE.md`.

---

*Generated 2026-09-21 as part of rolling Adel's summer 2026 work off `main` and onto
`adel-temp`. This document lives on `main`; the work it describes does not.*
