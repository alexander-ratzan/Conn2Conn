# Supplementary Material — Sanity Checks and Reproducibility

**What Cross-Modal Connectome Prediction Does and Does Not Buy**

This supplement documents, in full, the sanity-check discipline behind the main paper. The
purpose is not to pad the results with extra numbers but to show that each headline claim
was actively attacked from the directions a skeptical reviewer would choose, and survived.
Every section corresponds to a self-contained check that lives in the project tree under
`notebooks-FC_to_SC-experimental/`; the source artifacts are named at the end of each
section. All values are reproduced from the committed `*.csv` / `findings*.md` outputs of
those checks.

The main paper makes three structural claims:

1. **Directional reconstruction.** FC predicts SC better than SC predicts FC
   (≈1.6–1.7× in `demeaned_pearson`), stable across seeds and parcellations.
2. **No automatic utility.** A cheap `bv+demo` baseline predicts cognition; only observed
   FC beats it; observed SC and imputed connectomes do not, and `pred_FC` is actively
   harmful.
3. **Objective-dependent identity signal.** Predicted SC carries heritable family signal,
   but only under objectives that preserve it.

Each could fail for a mundane reason. The checks below close those reasons one by one.
A one-line index:

| § | Check | Attacks the explanation | Verdict |
|---|---|---|---|
| S1 | Reduction-axis robustness | "the asymmetry is a PCA artifact" | closed |
| S2 | FC noise & reliability ceilings | "SC→FC fails because FC is noisy" | closed (FC-side) |
| S3 | Nonlinear model class | "a bigger/nonlinear model would find it" | closed |
| S4 | Residual-boost & multimodal sink | "the linear model left signal on the table" | closed |
| S5 | Sample-size scaling | "more subjects would unlock it" | closed (to n≈683) |
| S6 | Richer structural representation | "streamline counts are too crude" | closed |
| S7 | PC mechanism localization | "the family mode is a reliability artifact" | survives partialling |
| S8 | Completeness, leaks, reproducibility | "the grid is silently wrong" | clean |

A recurring distinction runs through several sections: two different **ceilings**.
*Ceiling A* (data reproducibility) is how well the target agrees with a repeat measurement
of itself; it bounds the recoverable biological signal but is only valid at the population
level. *Ceiling B* (model/oracle) is how well the same estimator predicts the target from a
perfect same-modality copy (`FC→FC`, `SC→SC`); it is available for both modalities and
never exceeded per subject, but it is looser because it can exploit session-specific signal
that would not replicate. We disattenuate biological claims against Ceiling A and use
Ceiling B only to quantify cross-modal-vs-within-modality loss.

---

## S1. Reduction-Axis Robustness — the asymmetry is a property of the data

**Concern.** The headline FC→SC > SC→FC asymmetry was computed with one preprocessing
pipeline: PCA(256) on the source, PLS(64) on the latents, inverse-PCA on the target.
A reviewer will ask whether the asymmetry is a property of the brain or of that learned
linear reduction.

**Design.** A three-point ladder of reduction strategies, each run over 10 frozen seeds in
both directions:

- **A. PCA→PLS→PCA** — learned linear reduction on both ends (the main model).
- **B. Full PLS** — `PLSRegression` directly on the raw 64,620-edge vectors, no reduction.
  If A and B agree, the reduction is neither hiding nor injecting signal.
- **C. JL→PLS→PCA** — the input PCA is replaced by a data-blind Johnson–Lindenstrauss
  random projection (three variants: Gaussian dense, sparse `density='auto'` ≈ 1/√p, and
  sparse `density=1/3`). If A and C agree, the learned FC basis is not privileged; any
  256-dim linear projection captures the cross-modal signal.

**Result.** All five strategies give FC→SC > SC→FC at median ratio > 1.39× with Wilcoxon
`p ≤ 0.001` against the null ratio of 1.0 across 10 seeds.

| Method (n=10 seeds) | median FC→SC dp | median SC→FC dp | median ratio | min | max | p vs 1 |
|---|---:|---:|---:|---:|---:|---:|
| PCA→PLS→PCA (main) | 0.1355 | 0.0847 | **1.621×** | 1.32 | 1.86 | 0.001 |
| Full PLS (raw 64,620-dim) | 0.1382 | 0.0762 | **1.813×** | 1.49 | 2.11 | 0.001 |
| JL Gaussian dense | 0.0837 | 0.0610 | **1.397×** | 1.19 | 2.25 | 0.001 |
| JL sparse-auto (≈1/√p) | 0.0841 | 0.0592 | **1.389×** | 1.11 | 1.90 | 0.001 |
| JL sparse-1/3 (Achlioptas) | 0.0846 | 0.0535 | **1.554×** | 1.24 | 1.78 | 0.001 |

**Reading.**
- **Full PLS (1.81×) ≥ baseline (1.62×).** Removing reduction entirely gives the *same
  direction* and a *slightly stronger* magnitude. The PCA preprocessing, if anything,
  modestly attenuates the asymmetry rather than creating it.
- **All JL variants (1.39–1.55×).** A data-blind random projection of FC gives essentially
  the same asymmetry as the learned PCA. JL projections lower the *absolute* `demeaned_pearson`
  (they lose PCA's variance concentration) but preserve the *ratio*, which is the load-bearing
  quantity.
- The magnitude spread (1.39–1.81×) is wider than the conservative ±0.15× the synthesizer
  initially encoded, but it lies between known-equivalent reductions and the **direction is
  unambiguous**.

> **Reviewer-proof sentence.** "The FC→SC asymmetry is robust to the choice of input
> reduction. Across 10 seeds the demeaned-pearson ratio FC→SC / SC→FC was 1.62× with the
> main PCA(256)→PLS(64)→inverse-PCA pipeline, 1.81× with no reduction at all (PLSRegression
> on the raw 64,620-edge vectors), and 1.40×, 1.39×, and 1.55× with Johnson–Lindenstrauss
> random projections. All five methods reject the null ratio of 1.0 at Wilcoxon `p ≤ 0.001`."

**Operational notes (kept honest).** The full-PLS method OOM-killed on the first attempt
because `PLSRegression` materializes a `coef_` of shape (64620, 64620) ≈ 33 GB at fit time;
the fix bypasses `coef_` at predict time by computing
`(X−x̄)/x_std @ x_rotations_ @ y_loadings_.Tᵀ · y_std + ȳ` directly, and adds a per-seed cache
so cancellations resume cleanly. A separate pandas gotcha (empty `jl_variant` cells read as
NaN, which `groupby` silently drops) had briefly hidden methods A and B from the printed
summary; the underlying CSVs were always complete. The table above is post-fix.

*Source:* `sanity_checks/preprocessing_check/` (`findings.md`, `reduction_axis_summary.csv`,
`reduction_axis_synthesis.csv`, `method_{a,b,c}_results.csv`). **See figures S5, SX-none.**

---

## S2. FC Measurement Noise and Reliability Ceilings — the SC→FC gap is not FC noise

**Concern.** SC→FC is weak. Maybe that is only because FC is a noisy target, not because SC
and FC are genuinely cross-modally independent.

**Design.** HCP-YA includes repeated FC (REST1/REST2 × LR/RL), so FC-side reliability can be
measured directly: a reliability ceiling, a variance decomposition, whole-connectome
fingerprinting, cross-modal disattenuation, and per-subject reliability-vs-achieved
analysis. SC's own reliability cannot be measured (no test-retest dMRI in HCP-YA) and is the
one acknowledged data-blocked piece.

### S2.1 FC is mostly edge-noise but reliable as a whole connectome

Variance decomposition (G-theory 2×2, individual-difference fractions summing to 1):

| Parcellation | trait (signal) | state (day) | within-session | noise | G (avg connectome) |
|---|---:|---:|---:|---:|---:|
| Glasser | 0.301 | 0.035 | 0.021 | **0.643** | 0.585 |
| 4S456Parcels | 0.261 | 0.035 | 0.021 | **0.683** | 0.519 |

At the single-edge level ~64–68% of between-subject variance is measurement noise. Yet the
**whole connectome is 93% identifiable** (fingerprint top1 0.927 Glasser / 0.934 4S456;
discriminability ≈ 0.998). The reconciliation: individual signal is *distributed* — each edge
is noisy, but the multivariate pattern is stable.

### S2.2 The FC reproducibility ceiling

| Parcellation | comparison | demeaned_r | pearson | top1 | avg_rank |
|---|---|---:|---:|---:|---:|
| Glasser | within-session (LR/RL) | 0.364 | 0.710 | 0.78 | 0.968 |
| Glasser | **between-session (REST1/REST2)** | **0.491** | 0.813 | 0.933 | 0.992 |
| 4S456 | within-session (LR/RL) | 0.337 | 0.670 | 0.78 | 0.969 |
| 4S456 | between-session (REST1/REST2) | 0.457 | 0.783 | 0.936 | 0.993 |

The within-session rung is *lower* than between-session — the opposite of "shorter interval =
more reliable" — because LR and RL carry opposite phase-encode distortions, so single-direction
connectomes disagree more. The distortion-cancelled between-session value (0.491) is the valid
ceiling.

### S2.3 SC→FC captures only ~17% of the reproducible FC signal

Disattenuating against Ceiling A (0.491):

| source → FC | metric | achieved | ceiling | fraction |
|---|---|---:|---:|---:|
| **SC→FC** | demeaned_r | 0.085 | 0.491 | **0.17** |
| SC→FC | top1 (fingerprint) | 0.051 | 0.933 | 0.05 |
| SC→FC | avg_rank | 0.713 | 0.992 | 0.72 |
| bv+demo→FC | demeaned_r | 0.098 | 0.491 | 0.20 |
| bv+demo→FC | top1 | 0.026 | 0.933 | 0.03 |

SC captures ~17% of reproducible FC by `demeaned_r` and only ~5% of the fingerprinting
ceiling — far from the reliability ceiling, so the gap is genuine independence rather than FC
noise. Notably `bv+demo→FC` (0.20) ≥ `SC→FC` (0.17) even after disattenuation: cheap subject
confounds match or beat the structural connectome.

### S2.4 Per-subject reliability does not explain SC→FC

Per-subject between-session FC reliability is heterogeneous (Glasser mean 0.489, median 0.494,
std 0.117, range ≈ 0 to 0.78; ~2% of subjects < 0.2) and is a stable subject trait
(within-vs-between ρ = 0.41). If SC→FC were noise-limited, cleaner subjects would be more
predictable from SC. They are not:

- A subject's SC→FC quality is **uncorrelated** with their FC reliability: Pearson **r = 0.01
  (p = 0.72)**. The baseline `bv+demo→FC` *does* weakly track reliability (r = 0.15, p = 2e-5),
  which proves the analysis can detect a reliability effect when one exists — the SC flatness
  is not a methodological artifact.
- Reliability-filtering makes the captured fraction *worse*, not better:

| filter | n | SC achieved | SC ceiling | fraction |
|---|---:|---:|---:|---:|
| all | 857 | 0.083 | 0.491 | 0.169 |
| drop rel < 0.2 | 841 | 0.083 | 0.498 | 0.167 |
| drop bottom 10% | 771 | 0.083 | 0.517 | 0.160 |
| keep top 50% | 429 | 0.080 | 0.582 | 0.137 |

Keeping only high-reliability subjects raises the ceiling (0.49 → 0.58) while SC's achieved
stays pinned at ~0.08, so the fraction falls. The noisy tail was never the bottleneck.

Because per-subject reliability is one noisy estimate from a single pair of scans, we report
it as a *distribution* and lean on the **correlation/flatness** statement (and the `bv+demo`
contrast) rather than a pointwise "fraction of each subject's ceiling."

**Bottom line.** The SC→FC shortfall is decisively **not** an FC-measurement-noise or
per-subject-reliability artifact. The strong reading — "SC does not *contain* that part of
FC" — is interpretation, because a *uniform* SC noise floor would also produce a flat line;
distinguishing the two needs SC test-retest dMRI, which HCP-YA lacks. This is the only
acknowledged 5% gap between "not-FC-noise" (proven) and "genuine independence" (interpretation).

*Source:* `sanity_checks/noise_sanity_check/` (`findings_noise.md`, `outputs/*.csv`).
**See figures S6, SX1.**

---

## S3. Nonlinear Model Class — a bigger model does not find hidden signal

**Concern.** The ceiling might be a linear-model failure; a nonlinear estimator could recover
signal that PLS/BayesianRidge miss.

**Design.** Estimators swapped (HistGradientBoosting and KernelRidge-RBF) with the
representation and splits held byte-identical to the linear runs. Three probes: nonlinear
cognition (N1), nonlinear reconstruction + asymmetry (N2), nonlinear marginal contribution of
bundle features over counts (N3).

**An estimator-trust check first.** FC carries a real linear cognition lift (+0.078). A
trustworthy nonlinear probe must preserve it:

| estimator | FC pearson | floor | FC lift |
|---|---:|---:|---:|
| linear (BayesianRidge) | 0.451 | 0.372 | **+0.078** ✓ |
| KernelRidge | 0.338 | 0.282 | **+0.056** ✓ |
| HGB | 0.254 | 0.355 | **−0.101** ✗ |

KernelRidge preserves FC's signal, so its nulls are meaningful. **HGB destroys even FC's
signal at n ≈ 683** (overfits / cannot model the smooth FC→cognition map), so its nulls are
estimator-weakness, not evidence — we trust KernelRidge throughout.

**N1 — nonlinear cognition.** No tractography representation clears the `bv+demo` floor under
any estimator; the closest is KernelRidge on SC/crystallized at −0.029 (still below floor).

**N2 — nonlinear reconstruction & asymmetry.** KernelRidge vs linear PLS, FC→X `demeaned_pearson`:

| direction | PLS | KR | Δ |
|---|---:|---:|---:|
| FC→SC | 0.1355 | 0.1334 | −0.0021 |
| FC→r2t | 0.1113 | 0.1133 | +0.0020 |
| FC→r2t_corr | 0.0582 | 0.0605 | +0.0023 |

All |Δ| ≤ 0.0023 — nonlinear neither helps nor hurts. The asymmetry ratio is essentially
identical under linear and kernel models (SC 1.62× → 1.64×; r2t 2.26× → 2.27×).

**N3 — nonlinear marginal.** Paired Δ(SC+r2t − SC)→FC = −0.0049 under KernelRidge (Wilcoxon
p = 0.998 in the wrong direction): bundle features add nothing over counts even under a
nonlinear combined model.

All three decision rules return null (cognition unlock ≥ +0.03: not met; reconstruction gain
≥ +0.02: max +0.0023; marginal gain ≥ +0.02: −0.0049). **The findings are model-class robust.**

*Source:* `non-linear-sanity-check/` (`findings_nonlinear.md`, `n1`–`n3` summaries). **See figure S7.**

---

## S4. Residual-Boost and Multimodal Sink — the linear model left nothing on the table

**Concern.** Even if a plain nonlinear model finds nothing, perhaps the linear model already
"used up" the easy variance and a nonlinear model tasked specifically with the *residual*
would find structure.

**Design (the maximally-sensitive probe).** Architecture A hands the nonlinear model the
out-of-fold linear prediction for free (no leakage) and makes its loss the improvement *above*
that template. N4 does this per single modality; N5 builds a multimodal sink
`[FC‖SC‖r2t‖bv‖demo]` so a kernel can exploit any cross-modal conjunction.

**N4 cognition — null.** Δ(final − template) was ±0.008, mixed sign; every tractography rep
stayed 0.13–0.25 below the floor; FC stayed above.

**N4 reconstruction — one tiny real signal.** The residual-boost beats the PLS template at
p = 0.001 in every direction, but by only ≈ +0.002–0.006 `demeaned_pearson`:

| direction | template | final | Δ | p |
|---|---:|---:|---:|---:|
| FC→SC | 0.1355 | 0.1405 | +0.0052 | 0.001 |
| SC→FC | 0.0847 | 0.0866 | +0.0023 | 0.001 |
| FC→r2t | 0.1113 | 0.1169 | +0.0058 | 0.001 |
| r2t→FC | 0.0493 | 0.0526 | +0.0041 | 0.001 |
| FC→r2t_corr | 0.0582 | 0.0623 | +0.0050 | 0.001 |
| r2t_corr→FC | 0.0362 | 0.0402 | +0.0039 | 0.001 |

A genuine but trivial nonlinear sliver in connectome↔connectome mapping — ~4× below the +0.02
"matters" threshold, and it washes out under the multimodal sink (below).

**N5 cognition — null, and combining hurts.**

| target | bv+demo floor | FC alone | sink_linear | sink_residual |
|---|---:|---:|---:|---:|
| CogTotal | 0.373 | **0.436** | 0.401 | 0.381 |
| CogFluid | 0.298 | **0.306** | 0.301 | 0.286 |
| CogCrystal | 0.349 | **0.452** | 0.414 | 0.415 |

`sink_linear` sits 0.05–0.08 *below* FC alone: adding SC/tractography/demographics to FC
*dilutes* cognition prediction. The nonlinear cross-modal residual does not help (slightly
hurts). **No FC×tractography interaction exists** — the most plausible route for tractography
to matter is empty.

**N5 reconstruction — null.** Sink residual − sink linear = −0.0023 (p = 0.935). The tiny
N4recon sliver does not survive.

A clean null from OOF residual-boost means `y − ŷ_linear` is noise with respect to the source:
not "the model could not find it" but "there is no learnable structure left at this n."

*Source:* `non-linear-sanity-check/` (`findings_residual.md`, `DESIGN_residual_learning.md`,
`n4`/`n5` summaries). **See figure S7.**

---

## S5. Sample-Size Scaling — more data does not unlock nonlinearity

**Concern.** The linear ceiling might be a sample-size ceiling that a larger cohort would
break.

**Design.** Learning curve at n = 100 / 200 / 400 / full (≈683), test fixed and full at every
n; the question is whether the nonlinear gap (residual-boost `final` − linear `template`) grows
with n. Growing → data-limited (bigger cohort indicated). Flat/negative → structural ceiling.

**Cognition (CogTotal).**

| n | linear | final | gap | p(gap>0) |
|---|---:|---:|---:|---:|
| 100 | 0.302 | 0.290 | −0.0087 | 1.00 |
| 200 | 0.327 | 0.324 | −0.0152 | 0.99 |
| 400 | 0.394 | 0.380 | −0.0119 | 0.999 |
| ~683 | 0.391 | 0.375 | −0.0128 | 0.99 |

Gap-vs-n Spearman ρ = +0.05 (p = 0.77) → flat; the gap is ≤ 0 at every n while the *linear*
score climbs 0.30 → 0.39. More data helps the linear model; nonlinearity adds nothing at any
size.

**Reconstruction (sink→FC).** The gap is negative at every n and merely creeps toward zero
from below (−0.0054 → −0.0009); the apparently positive trend (ρ = 0.52) is the KernelRidge
residual's overfitting penalty vanishing as data grows, asymptoting *at* zero, never reaching
+0.005, never going positive. (The auto-verdict initially mis-flagged this as "data-limited"
on `ρ>0 & p<0.05` alone; the rule was corrected to require `gap@max_n > 0.005` — a positive
slope on a negative gap is a shrinking penalty, not emerging signal.)

**The null is now robust to four independent axes:** model class (S3), the residual-boost
head-start (S4 single-modality), cross-modal interactions (S4 sink), and sample size (S5).
We tested to n ≈ 683; we cannot exclude that n ≈ 10⁴ surfaces a tiny positive gap, but the
data converges *to* zero, not above it — the indicated move would be a bigger cohort, not a
bigger model.

*Source:* `non-linear-sanity-check/` (`findings_scaling.md`, `n6_scaling_summary.csv`).
**See figure S7.**

---

## S6. Richer Structural Representations — counts are a sufficient statistic

**Concern.** Parcellated streamline counts may be too crude; a richer named-bundle
tractography representation (`r2t`) might recover the missing structural signal.

**Design.** Six experiments (E1–E5 + retest), 10 seeds, Glasser, family-aware, all metrics
reported. Representations: count-SC, bundle-r2t, bundle-similarity (`r2t_corr`), combined
(per-block PCA to avoid scale domination), and a `r2t→synthetic-FC` substitution chain.

**E1 — predicting FC from each representation (median dp):**

| rep | demeaned_pearson |
|---|---:|
| SC (counts) | **0.0847** |
| kitchen-sink [SC‖r2t‖bv‖demo] | 0.0942 |
| SC+r2t | 0.0791 |
| r2t (bundle) | 0.0493 |
| r2t_corr (bundle-similarity) | 0.0362 |

Count-SC beats bundle-r2t on every metric; the bundle profile is a *lossier* FC predictor.
The kitchen-sink edge over SC (0.094 vs 0.085) comes from `bv+demo`, not `r2t` (E3).

**E2 — asymmetry across all six metrics.** FC→SC > SC→FC on all six metrics, every one
p = 0.001 — the strongest multi-metric statement of the headline asymmetry. The bundle
representation *amplifies* the asymmetry on shape/identifiability metrics (dp 2.26× vs SC's
1.62×) but fails on raw MSE, an honest metric-dependence.

**E3 — marginal contribution of r2t over SC.** Median Δ(SC+r2t − SC) = **−0.0013** (one-sided
p = 0.98 not an improvement; two-sided p = 0.049 marginally *worse*). Count-SC is a sufficient
statistic.

**E5 — downstream cognition (decisive), CogCryst pearson:**

| rep | CogCryst | lift over bv+demo |
|---|---:|---:|
| **FC** | **0.451** | **+0.105** ✓ |
| bv+demo (floor) | 0.346 | 0 |
| SC | 0.267 | −0.079 |
| r2t→synthFC | 0.180 | −0.166 |
| r2t | 0.164 | −0.183 |
| SC+r2t | 0.160 | −0.187 |
| r2t_corr | 0.103 | −0.243 |

**FC is the only representation above the demographic floor.** Every tractography
representation falls below it, and the substitution chain `r2t → synthetic-FC → cognition`
does **not** recover FC's cognition signal — synthetic FC from tractography is not a useful
cognitive biomarker.

**E4 — does the family/dorsal-stream SC mode exist in bundles?** The SC mode does not map
cleanly to any single r2t mode (best Spearman of subject scores −0.32; everything else
|ρ| < 0.23). The 66-tract bundle atlas does not carve cortex finely enough to express that
intra-parietal/occipital edge pattern — it is a count-SC phenomenon.

*Source:* `tractography_predict/` (`findings.md`, `findings_in_depth.md`, `e1`–`e5` summaries,
`tractography_synthesis.csv`). **See figures S8, SX2.**

---

## S7. Mechanism-Mode Localization and Reliability Partialling

**Framing.** The mechanism analysis is **exploratory**. The defensible claim is "a
property-selected, low-variance, FC-predictable SC mode carries weak family signal," not
"PC3 is the mechanism." The component index migrates across parcellations: the mode aligns
with **PC3 in Glasser** and **PC4 in 4S456Parcels** (Figure SX3). Effects are small
(FC→PC R² ≈ 0.22–0.24; sibling AUC ≈ 0.58). **PC1 is a confound** — strongly tied to sex and
brain volume (confound R² ≈ 0.89–0.93) — and must not be read as biological mechanism.

**Concern.** The selected mode's visual / dorsal-attention (DAN) localization could simply
reflect higher tractography reliability in posterior, short, high-strength connections.

**Design.** Partial edge strength and inter-region distance out of |PC3| and recompute Yeo-7
network enrichment on the residual.

- `Spearman(|PC3|, strength)` = +0.89 (R² 0.40); `Spearman(|PC3|, distance)` = −0.50 (R² 0.13).
  Jointly, strength + distance explain **41%** of |PC3| magnitude — the confound is real and
  non-trivial.
- After partialling, the visual/DAN localization **survives, and visual–visual intensifies**:

| network pair | raw enrichment | residualized | Δ |
|---|---:|---:|---:|
| visual ‖ visual | 11.97× | **13.32×** | +1.35 |
| dorsal attention ‖ dorsal attention | 7.73× | **5.73×** | −2.00 |
| dorsal attention ‖ visual | 4.49× | **4.99×** | +0.50 |

Visual–visual goes *up* after partialling — reliability under-predicts how strongly the mode
emphasizes visual cortex. DAN–DAN gives back about a quarter (7.7× → 5.7×) but stays ≈ 5–6×
chance. An independent FC scan-rescan cross-check adds essentially nothing beyond the
strength+distance proxy (joint R²(|PC3| ~ strength+distance+FC-reliability) = 41.2% vs 41.0%),
and the partialled localization is identical (visual 13.3×, DAN 5.7×, DAN-visual 5.0×).

> **Reviewer-proof sentence.** "The selected mode's visual/DAN localization is not attributable
> to higher reconstruction reliability in posterior short connections: after partialling
> streamline density and inter-region distance from |PC3| (together 41% of its magnitude
> variance), within-network enrichments remain at 13.3× (visual–visual; higher than
> unadjusted), 5.7× (DAN–DAN), and 5.0× (DAN–visual)."

**Honest caveat.** This uses a strength+distance (and FC-reliability) proxy, not a
gold-standard per-edge SC ICC, which would require the HCP retest dMRI release. The proxy is
the standard field fallback and tends to agree with retest reliability on which edges are
noisy; the partialling itself was done at seed 0 (the mode is cross-seed stable, median
|cos| ≈ 0.89). The 41% R² is high and is reported, not buried: reliability *matters*, and the
surviving signal is the other half.

*Source:* `sanity_checks/tract_check/` (`findings.md`, `enrichment_residual_top200.csv`,
`retest_icc_results/`), `reproduction/family_mechanism/outputs/f8_*.csv`,
`further_exploration/pc4_pc5_results/`. **See figures S10, SX3, SX4.**

---

## S8. Completeness, Leak Guardrails, and Operational Reproducibility

**Concern.** A large grid can be silently wrong: missing cells, non-finite metrics, hidden
leakage, or estimator confusion.

**Completeness contract.** Expected cells are generated from the grid configuration *before*
running, and merged outputs are checked against that manifest. The spine grid contains
**13,640 cells** (2,640 reconstruction + 11,000 downstream); the merged output is complete
(13,640/13,640), with zero non-finite values in load-bearing metrics and zero hard leak
failures.

**Leak verdicts** separate expected biological signal from implementation leakage:

| Verdict | Meaning | Count |
|---|---|---:|
| `ok` | no threshold violation | 3,344 |
| `EXPECTED_SIGNAL` | raw connectome predicts sex/age — biological, not a leak | 4 |
| `EXEMPT_FLAGGED` | input contains `bv+demo`; cognition rows interpretable, sex/age expected | 1,052 |
| `LEAK_FAIL` | non-exempt derived input exceeds threshold | **0** |

Only the combined observed FC+SC input crossed the sex threshold (expected biological signal).
No demographic-free derived input triggered `LEAK_FAIL`.

**Estimator-reporting rule.** Reconstruction leads with `demeaned_pearson` (raw Pearson is
dominated by the population-mean connectome). Downstream scalar targets are reported with
**BayesianRidge**: a resolved discrepancy showed the older report's cognition/oracle numbers
matched BayesianRidge exactly, while an analyst had accidentally read PCA→PLS rows, which are
ill-conditioned for low-signal scalar regression (e.g. Glasser SC→SC oracle `demeaned_pearson`
0.376 under PCA→PLS vs 0.648 under BayesianRidge; Glasser `bv+demo`→CogTotal pearson 0.129 vs
0.359). Every reported number names its estimator.

**Two-layer reproducibility.** Exploratory notebooks discovered the findings; the
`reproduction/` suite imports the same shared setup, freezes splits, reruns confirmatory grids
on both parcellations × 10 seeds, writes CSV mirrors, and verifies completeness. **CSV outputs
are the source of truth; W&B is a replay layer, not a dependency**, so offline or failed syncs
cannot change the analysis. Family/mechanism ports are validated against the notebooks with
exact or near-exact regression tests.

*Source:* `reproduction/` (`gen_expected_cells.py`, `configs/expected_cells.csv`,
`verify_completeness.py`, `outputs/leak_verdict.csv`),
`reproduction/exploration/DISCREPANCY_RESOLUTION.md`,
`dev-notes/PROJECT-HANDOFF-2026-06-22.md`. **See figure S4.**

---

## Summary — every escape route, closed

| Mundane explanation for the negative/ directional results | Check | Outcome |
|---|---|---|
| The asymmetry is a PCA reduction artifact | S1 | ratio > 1 for raw-PLS, learned PCA, and 3 JL projections; p ≤ 0.001 |
| SC→FC fails because FC is a noisy target | S2 | SC→FC reaches only 17% of FC's reliability ceiling; per-subject r = 0.01 |
| A bigger / nonlinear model would find the signal | S3 | KernelRidge (validated on FC) finds nothing; HGB nulls discounted |
| The linear model left residual signal unused | S4 | OOF residual-boost null for cognition; +0.005 sliver for recon washes out |
| More subjects would unlock it | S5 | nonlinear gap flat/≤0 from n=100 to n≈683 |
| Streamline counts are too crude | S6 | r2t predicts FC worse, adds nothing marginal, no cognition signal |
| The family mode is a tractography-reliability artifact | S7 | survives strength+distance+reliability partialling (41% R²); visual–visual intensifies |
| The grid is silently incomplete or leaking | S8 | 13,640/13,640 cells, 0 non-finite, 0 LEAK_FAIL |

The narrowed, defensible claim: **in healthy young adults, under cross-sectional normal-range
cognition, the missing structural utility is not recovered by obvious changes in dimensionality
reduction, model class, structural representation, or sample size — and the SC→FC shortfall is
not an FC-noise artifact.** The remaining open piece is SC's own reliability, which is
data-blocked pending test-retest dMRI.
