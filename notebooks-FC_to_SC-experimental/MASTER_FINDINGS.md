# Conn2Conn — methods/redirect paper argument + evidence index

HCP-YA (Glasser, 64,620 edges, ~683 train / ~195 test, family-aware splits, closed-form
PCA→PLS unless noted). Every finding lists its exact repo-relative `.md`/`.csv` evidence
(paths relative to `notebooks-FC_to_SC-experimental/`). Branch `adel-temp`.

---

## What this paper is (read this first)

**This is a methods / redirect paper, not a discovery paper.** Its contribution is two
things, and the document is organized around them:

1. **A required baseline the field is failing to use.** A brain-volume + demographics
   (bv+demo) predictor matches or beats the connectome for cognition and accounts for most
   apparent cross-modal signal. Every SC↔FC prediction study should report it. This is a
   forcing-function / checkpoint contribution (extends "beat the group mean" and "report
   effect size" to **"beat subject-level confounds"**).
2. **A rigorous demonstration that the cognition ceiling is structural** — not a method
   limit and not a sample-size limit — which *redirects* effort rather than just stopping
   it.

Everything else is supporting. The **FC→SC > SC→FC asymmetry is an established effect**
(Krakencoder); we characterize it more rigorously and use it as **scaffolding**, not as a
novel result. The **PC3 mechanism is exploratory and hedged** — small effects, one
parcellation from being a possible atlas artifact, and it already produced one false
positive (PC2). It is a hypothesis-generating bonus, not a headline.

The value here is **counterfactual**: saving the field wasted effort and installing a
checkpoint, plus a map of where the signal might actually live. We proved a *specific*
wall (healthy young adults, normal-range cognition, cross-sectional), not a universal one.

---

## PART 1 — THE CONTRIBUTION (the spine)

### C1 / F5 — The bv+demo baseline (primary contribution)
A T1-derived brain-volume + demographics predictor is the bar the connectome must clear,
and usually doesn't. Cognition (CogCryst): **T1→bv+demo r=0.353** (cheap, ~universal);
**+resting fMRI (obs FC) 0.434** (the only real gain); **+diffusion (obs SC) 0.258 —
below the free baseline.** Recommendation: acquire fMRI, skip diffusion *for cognition*;
**report a bv+demo baseline in any SC↔FC prediction study.** FC's gain over the baseline
is significant by paired permutation test (the correct test; CI-overlap is the wrong one).
- **Evidence (CSV)**:
  - `model_overviews/results/downstream_prediction_phase2/aggregate.csv` — bv+demo / obs_FC / obs_SC per target, mean + ci_lo/ci_hi (raw scores ARE CI'd)
  - `model_overviews/results/downstream_prediction_phase2/perm_vs_bvdemo.csv` — p_vs_bvdemo, delta_mean, p_fdr (the "beats baseline" test; also the sex/age leak diagnostics)
- **Evidence (MD)**: `model_overviews/results/FINDINGS.md`
- *Confidence: high (direction + significance). The lift magnitudes (obs_FC − bv+demo) are
  derived differences without bootstrap CIs — attach before anchoring.*

### C2 — The structural ceiling: FC is the wall, and nothing gets past it (F5 + F9 + F10 as one argument)
These three close each other's escape routes and must be read as a block, not separately:
- **FC is the ceiling** (F5): the only modality above the bv+demo floor; SC and anatomy
  fall below it.
- **Richer structure doesn't help** (F9): named-bundle tractography (r2t) predicts FC
  *worse* than plain streamline counts (0.049 vs 0.085 demeaned-r), adds nothing marginal
  over SC (Δ=−0.001, p=0.98), and carries no cognition above the floor; count-SC is a
  sufficient statistic. → you can't recover the gap with better structural features.
- **Nonlinearity and more data won't help** (F10): four orthogonal nonlinear tests are
  null — model class (KernelRidge/HGB), residual-boost (hand the model the linear answer
  free), cross-modal sink (FC×SC×r2t interactions), and a **data-scaling curve** (the
  nonlinear gap is flat from n=100→683, so the ceiling is structural, **not** sample-size).
  → you can't recover the gap with a bigger model or a bigger n in this regime.
- **Why it's a redirect, not a dead-stop**: the signal is *dominated by confounds in this
  regime* (healthy young adults, normal-range cognition, cross-sectional). The open
  question — explicitly part of the contribution — is whether it survives in **clinical
  populations, longitudinal designs, or large-n cohorts**, and the bv+demo protocol (C1)
  is the checkpoint to carry there. See "Where the signal might live" below.
- **Evidence (CSV) — F9 tractography**:
  - `tractography_predict/e1_source_rep_results.csv` (rep → FC, all 6 metrics, 10 seeds),
    `e2_asymmetry_summary.csv`, `e3_marginal_summary.csv`,
    `e4_r2t_pc_stability.csv` / `e4_sc_pc3_to_r2t_pc_projection.csv` / `e4_r2t_top_bundles_per_mode.csv`,
    `e5_downstream_results.csv` / `e5_downstream_summary.csv`, `tractography_synthesis.csv`
- **Evidence (CSV) — F10 saturation**:
  - N1–N3 `non-linear-sanity-check/n{1,2,3}_*_summary.csv` (+ `*_results.csv`), `nonlinear_synthesis.csv`
  - N4 residual-boost `n4_{cog,recon}_summary.csv` (+ results), `residual_synthesis.csv`
  - N5 cross-modal sink `n5_{cog,recon}_summary.csv` (+ results)
  - N6 data-scaling `n6_scaling_summary.csv` / `n6_scaling_results.csv`, `scaling_synthesis.csv`
- **Evidence (MD)**: `tractography_predict/findings.md`, `findings_in_depth.md`;
  `non-linear-sanity-check/findings_nonlinear.md`, `findings_residual.md`, `findings_scaling.md`, `DESIGN_residual_learning.md`
- *Confidence: high. This is the confirmatory core — well-powered negatives across four axes.*

### Where the signal might actually live (forward-pointing contribution)
The wall is regime-specific. Candidate regimes where structural→cognition signal could
survive the bv+demo baseline, each a concrete next study using the C1 protocol:
- **Clinical / lesioned populations** — where SC varies far beyond the healthy normal range.
- **Longitudinal / developmental designs** — within-subject change may carry signal that
  cross-sectional between-subject variance (dominated by stable anatomy) does not.
- **Large-n cohorts (ABCD, UK Biobank)** — the data-scaling curve (F10/N6) shows the
  *linear* model still improving with n; a faint nonlinear gap can't be ruled out at n≈10⁴,
  though our data shows convergence toward zero, not growth. If chased, the move is a
  bigger cohort, never a bigger model.
- In all three: **report the bv+demo baseline** so the field can tell real structural
  signal from recycled confounds.

---

## PART 2 — SUPPORTING FINDINGS (scaffolding + science)

### F1 — The FC→SC > SC→FC asymmetry (established effect; scaffolding, not a result)
**This is Krakencoder's finding, reproduced with a simpler deterministic model — owned as
established, not claimed as novel.** FC→SC ≈ 0.134 vs SC→FC ≈ 0.085 (~1.5×); survives
anatomy+demo stripping (~1.4×); holds on all 6 metrics (avg_rank 1.25× most stable);
clears the 1.15 gate 9–10/10 seeds; reproduces Krakencoder's 0.16/0.09. We use it as a
characterized, robust substrate for the baseline/ceiling argument.
- **Evidence (CSV)**:
  - `model_overviews/results/notebook_snapshot/phase1/section2_asymmetry.csv` (single-seed, all bases)
  - `model_overviews/results/Exp7_stringent_ratio_10_seed/aggregate_full_panel_ratios.csv` (10-seed, all metrics, CIs + n_above_1.15) + `per_seed.csv` + `full_panel_seed_{0..9}.csv`
  - `model_overviews/results/Step11_grand_10seed/aggregate_ratios.csv` (10-seed across every basis, t/p vs 1.00 & 1.15, CIs)
- **Evidence (MD)**: `model_overviews/results/FINDINGS.md`
- *Confidence: high. Ratios carry CIs. Robustness: Appendix A1–A3.*

### F2 — Modality dissociation
SC is anatomy-driven (bv→SC 0.162, anatomy wins); FC is demographics-driven (demo→FC
0.114, demo wins). Same data, opposite drivers, 10/10 seeds.
- **Evidence (CSV)**: `model_overviews/results/notebook_snapshot/phase1/section1_baselines.csv`
- **Evidence (MD)**: `model_overviews/results/FINDINGS.md`
- *Confidence: high.*

### F3 — Imputation inherits the source modality (utility asymmetry)
pred_SC-from-FC beats real SC for cognition (1.25–1.45×); pred_FC-from-SC loses 40–50%.
The ~1.5× reconstruction asymmetry becomes **2.06–2.54× downstream-utility asymmetry**. A
predicted connectome carries its *source* modality's information — imputing the other
modality is not worth it for cognition.
- **Evidence (CSV)**: `model_overviews/results/downstream_prediction_phase2/aggregate.csv` (obs_/pred_ SC/FC per target)
- **Evidence (MD)**: `model_overviews/results/FINDINGS.md`
- *Confidence: high direction. ⚠ The 2.06–2.54× utility ratios are DERIVED (pred/obs) and
  currently have NO CIs — attach bootstrap CIs before using as a headline number.*

### F4 — FC's cognition signal is real; SC's is mostly demographics
Fraction surviving bv+demo removal: obs_FC 60/60/73% vs obs_SC 27/18/31% (Total/Fluid/Cryst).
- **Evidence (CSV)**: `model_overviews/results/downstream_prediction_phase2/aggregate.csv` (+ `perm_vs_bvdemo.csv`)
- **Evidence (MD)**: `model_overviews/results/FINDINGS.md`
- *Confidence: high direction. ⚠ The survival-fraction percentages are derived ratios
  without CIs.*

### F6 — Predicted connectomes carry heritable family signal
pred_SC_resid_bvdemo separates siblings from strangers at AUC **0.810** vs bv+demo baseline
0.563 (Δ +0.247) — family-specific wiring beyond shared anatomy.
- **Evidence (CSV)**: `model_overviews/results/family_structure_phase2/aggregate_auc.csv` (per variant × relation, AUC + auc_lo/auc_hi + p_perm + p_fdr)
- **Evidence (MD)**: `model_overviews/results/FINDINGS.md`
- *Confidence: medium-high. AUCs carry permutation CIs in the CSV, BUT this table is from
  an OLDER run (close-out #1: regenerate from the consistent pass). The +0.247 gap is a
  derived difference without a propagated CI.*

### F7 — The predictor / identifier tradeoff
combined_pred_SC (optimized for reconstruction) wins cognition but collapses to chance on
sibling separation (AUC 0.505): bv+demo content dominates the reconstruction objective and
drowns the within-family signal. Reconstruct OR discriminate, not both from one loss.
- **Evidence (CSV)**: `model_overviews/results/family_structure_phase2/aggregate_auc.csv` (combined_pred_SC 0.505 vs pred_SC_resid_bvdemo 0.810); `model_overviews/results/local_results/further_exploration/depth2_robustness_sensitivity/combined_followup.csv` (4-perturbation check: structural, not a bug)
- **Evidence (MD)**: `model_overviews/results/FINDINGS.md`; `sanity_checks/preprocessing_check/findings.md`
- *Confidence: high (confirmed not-a-bug).*

### F7b — The reconstruct/identify tradeoff is set by the estimator's objective (BR vs PLS), with a measured mechanism
Swapping the imputation estimator from PLS to **BayesianRidge** (the stronger reconstructor)
sharpens F7 *at the estimator level*. BR reconstructs better (FC→SC demeaned-r **0.166 vs PLS
0.136**; marginally higher cognition lift) but is the **worse identifier**: sibling AUC for
`pred_SC_resid_bvdemo` drops to **0.763 (BR) vs 0.810 (PLS)**, and BR is lower on every imputed
variant. F6 still replicates under BR (0.763 ≫ demographic baseline 0.563, p<1e-4) and F7 still
holds (`combined_pred_SC` 0.505, n.s.).
**Mechanism (measured, Glasser ×10):** BR's evidence-tuned per-component shrinkage flattens the
low-variance target-PC tail toward the group mean — it retains only **0.182×** of the true
individual-deviation amplitude vs PLS's **0.306×**, and per-PC amplitude collapses **0.384 → 0.044**
down the spectrum vs PLS's nearly-flat **0.519 → 0.382** (~9× more tail amplitude for PLS;
direction-recovery corr crosses over at ~PC 50). The same shrinkage that wins the high-variance bulk
(reconstruction / cognition) erases the low-variance idiosyncratic tail that fingerprints families.
*You cannot optimize one connectome to both reconstruct and identify — and the estimator's objective
is one knob on that tradeoff.*
- **Evidence (CSV)**: `../reproduction/br_family/outputs/family_auc_br.csv` (BR sibling AUC + CI + perm + FDR); `../reproduction/br_imputation/outputs/probe_shrinkage.csv` (per-PC amplitude + recovery, BR vs PLS); `../reproduction/br_imputation/outputs/downstream_br.csv` (cognition, 18 inputs × 10 seeds)
- **Evidence (MD)**: `../reproduction/br_family/FINDINGS.md`; `../reproduction/br_imputation/FINDINGS.md`
- *Confidence: high. Estimator-independent variants reproduce the PLS run to float tolerance (wiring validated); Glasser × 10 with bootstrap CIs + permutation p; mechanism measured on identical splits. Glasser only (4S456 deferred).*

---

## PART 3 — EXPLORATORY / HYPOTHESIS-GENERATING (clearly hedged)

### F8 — Mechanism: FC predicts a visual / dorsal-attention structural mode (SC-PC3)
**This is exploratory and small. Do not write it as settled.** The asymmetry partly
localizes to SC's third principal mode (~1.5% variance): stable across seeds (cos 0.89),
FC-predictable (**R²≈0.22 — modest**), heritable beyond demographics (**AUC sibling 0.58 —
barely above the 0.5 chance line**; MZ 0.71 > DZ 0.60 > sibling 0.58 gradient), spatially
concentrated (62% energy in 1% of edges), within-hemisphere visual/DAN (enrichment
11.7×/8.2×/4.9×). Distributed across PC3–PC5. PC1 is a 89% sex+BV confound (reported as a
methods caveat: don't interpret your top SC PC as heritable).
- **These are small effects** (R²≈0.22, sibling AUC≈0.58) and **the analysis already
  produced one false positive**: the earlier "PC2 carries it, R²=0.26" did NOT replicate
  (median R²=0.017 across 10 seeds) — a single-seed artifact nearly written up. The story
  moved to PC3 only after cross-seed alignment.
- **Real exposure**: F8 is **one parcellation away from being a possible "Glasser
  artifact."** Until the 4S456Parcels replication runs, treat as unconfirmed.
- **Evidence (CSV)**:
  - `model_overviews/results/local_results/further_exploration/depth1_spectral_mechanism/synthesis_cross_tab.csv`, `per_pc_heritability.csv`, `per_pc_fc_predictability.csv`, `pc2_top_edges.csv`, `sc_pc_loadings.npy`, `sc_pca_mean.npy`
  - `…/depth1.1_pc_stability_and_confounds/stability_aligned_to_seed0.csv` (cross-seed alignment; PC2 false-positive caught here), `pc_confound_r2.csv` (PC1 89%, PC3 ≈0), `pc3_enrichment_agg.csv`, `pc3_enrichment_per_seed.csv`, `pc3_localization_per_seed.csv`, `pc3_top50_edges_seed0_labeled.csv`
  - `further_exploration/pc4_pc5_results/pc{4,5}_{stability_aligned,confound,enrichment_agg,localization_per_seed,top30_edges}.csv`
  - figure: `further_exploration/figures/output_glassbrain/pc3_glassbrain.png` (+ `pc3_top200_edges_labeled.csv`)
- **Evidence (MD)**: `further_exploration/README.md`
- *Confidence: **medium, pending 4S456 replication** (downgraded from medium-high). All
  mechanism point-estimates (R², AUCs, enrichments) are cross-seed medians WITHOUT
  bootstrap CIs.*

---

## PART 4 — HONESTY PASS

### Exploratory vs confirmatory (which findings carry weight)
- **Confirmatory (paper can rest on these)**: C1/F5 baseline, C2/F9/F10 ceiling, F1
  asymmetry, F2 dissociation. Well-powered, pre-registered-in-spirit, robust across
  seeds/metrics/estimators/reductions.
- **Exploratory / hypothesis-generating (cannot anchor the paper)**: F8 mechanism (already
  produced the PC2 false positive), and the *magnitudes* (not directions) of F3/F4/F6.

### Effect sizes are small (state plainly)
Cognition prediction is r≈0.45 (research-grade, not diagnostic-grade). The mechanism is
R²≈0.22 with sibling AUC≈0.58 (barely above chance). The headline asymmetry is ~1.5×. None
of this is a biomarker; the contribution is the baseline + the ceiling, not performance.

### Numbers that still need CIs before anchoring claims
Have CIs in their CSVs: F1 asymmetry ratios (Exp7/Step11), F5 raw cognition scores, F6
family AUCs. **Currently CI-less (derived/point estimates — attach bootstrap before
write-up)**: F3 utility ratios (2.06–2.54×), F4 survival fractions (60/60/73%, 27/18/31%),
F5 lifts (obs_FC − bv+demo), F6 +0.247 gap, all F8 mechanism point-estimates.

### Limitations (dedicated block)
- **Single cohort** (HCP-YA) — no external replication.
- **Non-representative sample** — healthy young adults, higher-education-skewed.
- **Cross-sectional** — no within-subject/longitudinal signal.
- **Normal-range cognition only** — restricted outcome variance.
- **Single parcellation** (Glasser) — F8 especially exposed.
- **n≈683 train / 195 test** — the ceiling is shown *in this regime*; not proven universal.

---

## PART 5 — ROBUSTNESS APPENDIX (with evidence)

### A1 — Reduction-axis robustness (defends F1)
5 reductions: no-reduction full PLS 1.81×, learned PCA 1.62×, three JL 1.39–1.55×; all
p≤0.001. PCA does not inject the effect.
- **CSV**: `sanity_checks/preprocessing_check/reduction_axis_summary.csv`, `reduction_axis_synthesis.csv`, `method_a_results.csv`, `method_b_results.csv`, `method_c_results.csv`
- **MD**: `sanity_checks/preprocessing_check/findings.md`, `README.md`

### A2 — Estimator robustness (defends F1)
PLS 1.56× vs BayesianRidge 1.75×; KernelRidge ≈ PLS (F10/N2).
- **CSV**: `model_overviews/results/notebook_snapshot/phase1/exp5_br_robustness.csv`; `tractography_predict/e2_asymmetry_summary.csv`; `non-linear-sanity-check/n2_reconstruction_summary.csv`

### A3 — K_PCA / K_PLS sensitivity (defends F1)
Stable 1.5–1.9× over K_PCA{64,128,256}; inflation only at K=512 via weaker SC→FC denominator.
- **CSV**: `…/depth2_robustness_sensitivity/k_sweep.csv`, `estimator_comparison.csv`

### A4 — PC mechanism verification (defends/hedges F8)
PC2-disappearance caught (R²=0.26 → 0.017); PCs 1–5 stable, 6+ noise; PC1 = 89% sex+BV.
- **CSV**: `…/depth1.1_pc_stability_and_confounds/stability_aligned_to_seed0.csv`, `pc_confound_r2.csv`; `…/depth1_spectral_mechanism/synthesis_cross_tab.csv`

### A5 — PC3 reliability (defends F8)
Visual/DAN localization survives partialling edge strength+distance (12.0→13.3×) and an
independent FC scan-rescan reliability proxy (joint R² 41.2% vs 41.0%).
- **CSV**: `sanity_checks/tract_check/enrichment_residual_top200.csv`, `retest_icc_results/enrichment_residual_with_fc_reliability.csv`, `retest_icc_results/fc_reliability_summary.csv`
- **MD**: `sanity_checks/tract_check/findings.md`, `README.md`

### A6 — FC noise / reliability ceiling (defends C2 / F10) — **CLOSED**
The cross-modal ceiling is **not an FC-measurement-noise artifact** — shown in our native
metric, per-subject. FC between-session reliability ceiling (Glasser) = demeaned_r **0.49**
(fingerprint top1 0.93); FC edge variance is **~30% trait / ~64% noise** yet the whole
connectome is 93% identifiable (distributed signal). **SC→FC captures ~17% of the
reproducible FC signal**, and — decisively — a subject's SC→FC quality is **uncorrelated
with their own FC reliability** (Pearson r=**0.01**, vs the bv+demo baseline's r=0.15), and
reliability-filtering does **not** sharpen it (fraction 0.169→0.137 as the ceiling rises but
achieved stays pinned at ~0.08). So the gap is **not FC-noise and not per-subject
reliability** (airtight); "SC doesn't *contain* it" is the strong interpretation, pending
SC test-retest (a uniform SC noise floor would also produce the flat line). Per-subject
reliability is heterogeneous (~0–0.78, std 0.12, left-skewed) and a stable subject trait
(within-vs-between ρ=0.41). **Status: closed — done its pre-grid job; SC-side reliability
remains the one data-blocked open item.**
- **CSV**: `sanity_checks/noise_sanity_check/outputs/` — `a_reliability_ceiling.csv`,
  `b_variance_decomposition.csv`, `f_discriminability.csv`, `e_crossmodal_disattenuation.csv`,
  `g_per_subject_summary.csv`, `h_per_subject_achieved_vs_ceiling.csv`,
  `h_reliability_filtered_summary.csv`, `h_correlations.csv`, `noise_synthesis.csv`
  (+ figures `g_reliability_hist.png`, `h_achieved_vs_ceiling_scatter.png`)
- **MD**: `sanity_checks/noise_sanity_check/findings_noise.md`, `README.md`;
  roadmap `planning/roadmap/noise-sanity-check.md`

### Negatives kept / deliberately not chased
SC carries no non-demographic cognition signal (F4); cognition ceiling is structural (C2);
CNN/autoencoder not chased (wrong inductive bias for non-grid edges); rigorous HCP-retest
ICC not run (proxy passed); genomics/clinical claims out of scope (heritability here is
family-structure inference, not molecular).

---

## PART 6 — HARDENING / CLOSE-OUT AGENDA
1. **Regenerate every table from one consistent run** — F6 family CSVs are from an older
   pass (`family_structure_phase2/aggregate_auc.csv`). The one real correctness debt.
2. **Bootstrap CIs** on the CI-less headline numbers listed in the Honesty pass.
3. **Parcellation replication (4S456Parcels)** of F1/F2/F5/F8 — kills the "Glasser
   artifact" exposure; *required* before F8 can move above "medium, pending replication."
4. **Leak guardrail** — auto-fail any downstream input predicting sex>0.99 / age>0.85.
5. **KRR ✅ done; MLP optional** (argued against).
6. **WandB + dataset parameterization → port a second HCP cohort → auto-regenerate figures.**

---

## MAP OF THE WORK
| Area | Directory | Key docs |
|---|---|---|
| Main notebook, Phase 0/1/2 (F1–F7, baseline C1) | `model_overviews/` | `results/FINDINGS.md`, `results/NEXT_STEPS.md` |
| Mechanism PC3/4/5 (F8, exploratory) | `further_exploration/` | `README.md` |
| Reduction robustness (A1), combined-pred (F7) | `sanity_checks/preprocessing_check/` | `findings.md` |
| PC3 reliability (A5) | `sanity_checks/tract_check/` | `findings.md` |
| Tractography r2t (F9, ceiling) | `tractography_predict/` | `findings.md`, `findings_in_depth.md` |
| Nonlinear/residual/sink/scaling (F10, ceiling) | `non-linear-sanity-check/` | `findings_nonlinear.md`, `findings_residual.md`, `findings_scaling.md` |
| FC noise / reliability ceiling (A6, C2/F10-supporting, **closed**) | `sanity_checks/noise_sanity_check/` | `findings_noise.md` |

*All quantitative values trace to the cited CSVs (executed notebooks / SLURM runs).
Confirmatory claims (baseline, ceiling, asymmetry) are robust; the mechanism is
exploratory and pending parcellation replication; flagged magnitudes need bootstrap CIs.*
