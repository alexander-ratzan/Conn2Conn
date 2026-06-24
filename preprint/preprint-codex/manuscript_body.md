## Abstract

Cross-modal connectome prediction is often treated as a route from one brain measurement to
another: if structural connectivity (SC) can predict functional connectivity (FC), or FC can predict
SC, then the predicted connectome may stand in for the missing modality. We tested this assumption
in HCP Young Adults using two parcellations, 10 frozen family-aware splits, deterministic estimators,
and a grid that separates reconstruction from downstream cognitive utility. Cross-modal prediction
was real and directional: FC predicted individual deviations in SC better than SC predicted FC
(1.61x in Glasser and 1.68x in 4S456Parcels with PCA->PLS). This asymmetry was not explained by
within-modality reliability, because FC and SC were similarly self-predictable under oracle
same-modality controls. However, reconstruction did not imply downstream utility. A cheap
brain-volume plus demographics baseline (`bv+demo`) predicted cognition substantially, observed FC
provided the only consistent lift over this baseline, observed SC underperformed it, and imputed
connectomes did not transfer FC's cognitive signal through SC. In particular, FC imputed from SC was
not merely unhelpful but consistently below the `bv+demo` baseline. Additional sanity checks ruled
out several easy explanations: the ceiling was not a PCA artifact, nonlinear models and richer
tractography features did not recover the missing signal, scaling trends did not suggest a growing
nonlinear gap, and the SC->FC shortfall was not explained by FC measurement noise. Predicted
connectomes did carry family signal, but optimizing reconstruction and optimizing identification
selected different information. These results support a reporting standard for connectome
translation studies: report individual-deviation reconstruction metrics, always include a cheap
subject-information baseline for downstream claims, and separate reconstruction accuracy from
utility.

## 1. Introduction

Structural and functional connectomes describe the same brain from different measurement
perspectives. This has made FC-SC translation an attractive goal: infer the missing modality,
compare modalities in a common space, and use the predicted connectome for behavior, cognition, or
individual identification. A successful reconstruction number can make this goal look solved.

But connectome translation has two distinct questions. The first is reconstruction: can one modality
recover individual deviations in the other? The second is utility: does the recovered connectome
carry information that matters for the target analysis? A model can succeed at the first and fail at
the second. In neuroimaging, the gap is especially important because many downstream outcomes are
already predictable from cheap subject-level information such as age, sex, and anatomy-derived brain
volumes.

This paper treats FC-SC prediction as a methods problem rather than a leaderboard problem. We ask
what cross-modal prediction proves, what it fails to prove, and which baselines should be required
before a predicted connectome is treated as useful. The central checkpoint is a brain-volume plus
demographics baseline (`bv+demo`). For downstream cognition, beating the group mean is too weak;
the connectome must beat subject information that is already available in most imaging studies.

We make four contributions. First, we reproduce and characterize the known FC->SC > SC->FC
asymmetry under a deterministic grid across two parcellations and 10 frozen splits. Second, we show
that the apparent value of translation changes once cognition is evaluated against `bv+demo`:
observed FC helps, SC does not, and imputed connectomes do not rescue the gap. Third, we close
several escape routes with sanity checks: reduction choice, nonlinear models, richer structural
representations, sample-size trends, and FC reliability do not explain away the ceiling in this
healthy young adult regime. Fourth, we show that predicted connectomes are not globally useless:
they carry family signal, but reconstruction and identification pull on different information.

## 2. Data and Evaluation Design

We used HCP Young Adults subjects with both resting-state FC and diffusion-derived SC available,
plus demographics and anatomy-derived brain-volume features. Connectomes were represented as
upper-triangle edge vectors. Analyses were run on two parcellations: Glasser
(360 regions; 64,620 edges) and 4S456Parcels (456 regions; 103,740 edges). Splits were frozen once
for 10 seeds and reused across reconstruction, downstream prediction, imputation, leak checks, and
family analyses. Family-aware splitting kept siblings and twins on the same side of the split.

The primary reconstruction metric was `demeaned_pearson`: the per-subject correlation/cosine after
subtracting the training-set group mean connectome from both the prediction and target. This is the
load-bearing metric because raw Pearson is dominated by the shared population mean structure of the
connectome. We also tracked fingerprinting metrics (`avg_rank`, `top1_acc`) and conventional
regression metrics, but these are secondary or diagnostic.

For reconstruction, the main closed-form estimator was PCA(256)->PLS(64)->inverse-PCA. We also
ran BayesianRidge and KernelRidge controls. For downstream scalar targets, BayesianRidge is the
reportable estimator: PCA->PLS and KernelRidge scalar rows were retained diagnostically but are
ill-conditioned and not numerically stable for low-signal scalar regression. Every reported number
therefore names its estimator.

The grid separates two axes that are often conflated:

- The input set: FC, SC, brain volumes (`bv`), demographics (`demo`), `bv+demo`, observed
  connectomes plus `bv+demo`, and imputed connectomes.
- The estimator: the mathematical mapping from input to output.

For reconstruction, we evaluate FC->SC, SC->FC, subject-information baselines, connectome plus
subject-information inputs, and same-modality oracle controls. For downstream cognition, we ask
whether observed or imputed connectomes improve prediction of CogTotal, CogFluid, and CogCryst over
`bv+demo`. Sex and age are evaluated as leak checks, not as scientific outcomes.

**Figure 1. Evaluation spine.** Inputs, targets, frozen splits, reconstruction branch, downstream
branch, and sanity-check branch.

**Table 1. Dataset and parcellations.** HCP-YA cohort, split sizes, atlas nodes/edges, and metrics.

**Table 2. Task grid and claims.** Reader-facing map of each input->target row and the claim it
tests.

## 3. Cross-Modal Prediction Is Directional

FC predicted SC substantially better than SC predicted FC. With PCA->PLS, the mean
`demeaned_pearson` across seeds was 0.136 for FC->SC and 0.084 for SC->FC in Glasser, a 1.61x
ratio. In 4S456Parcels, FC->SC reached 0.146 and SC->FC 0.087, a 1.68x ratio. The direction was
stable across seeds and parcellations.

This asymmetry is not simply because FC or SC is a noisier target. Same-modality oracle controls
showed that FC and SC were comparably self-predictable. With BayesianRidge, the FC->FC oracle was
0.673 in Glasser and 0.632 in 4S456Parcels; the SC->SC oracle was 0.648 and 0.622 respectively.
Thus the target modalities have similar within-modality ceilings, while the cross-modal directions
differ strongly.

The baselines also showed a clean double dissociation. Brain-volume features were better predictors
of SC than demographics were, whereas demographics were more than twice as predictive of FC as brain
volumes were. This crossover matters because it shows that "subject information" is not one generic
confound. Anatomy and demographics align differently with structure and function.

The finer 4S456Parcels atlas added a useful nuance. Every input->SC row improved relative to
Glasser, while input->FC rows were slightly lower. The larger parcellation appears to expose more
predictable structural detail rather than merely adding noise.

**Figure 2. Directional translation.** FC->SC vs SC->FC reconstruction, seed-level ratios,
within-modality oracle controls, and the anatomy/demographics double dissociation.

## 4. Reconstruction Does Not Imply Cognitive Utility

The downstream grid changed the interpretation of cross-modal prediction. The `bv+demo` baseline
itself predicted cognition: with BayesianRidge, CogTotal was 0.359 in both parcellations, CogFluid
0.283, and CogCryst 0.354. This is the bar a connectome must clear.

Observed FC was the only connectome input that consistently improved over this bar. In Glasser,
observed FC predicted CogCryst at 0.487, lifting 0.133 over `bv+demo` with median paired-permutation
`p=0.0307`. In 4S456Parcels, observed FC predicted CogCryst at 0.476, lifting 0.122 over baseline
with `p=0.0407`. The strongest rows were observed FC plus `bv+demo`, especially for CogCryst
(0.516 in Glasser, lift 0.162; 0.500 in 4S456Parcels, lift 0.146).

Observed SC did not clear the same baseline. In Glasser, observed SC was below `bv+demo` for
CogTotal, CogFluid, and CogCryst, with lifts of -0.111, -0.120, and -0.101. In 4S456Parcels, the
corresponding lifts were -0.103, -0.118, and -0.095.

Imputation did not transfer FC's utility through SC. SC imputed from FC was roughly neutral to
slightly below baseline. FC imputed from SC was consistently harmful: in Glasser, `pred_FC` lifted
CogTotal, CogFluid, and CogCryst by -0.113, -0.097, and -0.135. In 4S456Parcels, the corresponding
lifts were -0.088, -0.072, and -0.109. The important point is not merely that imputation failed to
help. It injected structured information that made the cognition predictor worse than the cheap
subject-information baseline.

These results separate reconstruction from utility. A predicted connectome can be measurable,
replicable, and still fail the downstream task it was supposed to enable.

**Figure 3. Utility checkpoint.** Cognition prediction and lift over `bv+demo` for `bv+demo`,
observed FC, observed SC, observed FC+SC, imputed SC, and imputed FC.

## 5. The Ceiling Is Structural in This Regime

The negative downstream result could have several mundane explanations. The grid and supplementary
checks test these explicitly.

The supplement expands this section into an audit trail rather than a grab bag of controls:
expected-cell completeness, leakage checks, reconstruction-to-utility decoupling, estimator
robustness, nonlinear and residualized models, richer tractography features, FC-noise accounting,
family-mechanism localization, preprocessing-axis stress tests, and a triage table separating
questions closed by existing data from experiments that require new measurements.

First, the cross-modal asymmetry is not an artifact of PCA reduction. In a reduction-axis sanity
check, the FC->SC / SC->FC ratio remained greater than 1.0 across learned PCA, full PLS on raw
64,620-edge vectors, and three Johnson-Lindenstrauss random projections. Median ratios ranged from
1.39x to 1.81x across methods, with Wilcoxon `p <= 0.001` against a ratio of 1.0 for all methods.

Second, the ceiling is not recovered by richer models. KernelRidge and other nonlinear checks did
not reveal hidden signal beyond the linear baselines; the KernelRidge 3x3 hyperparameter sweep was
nearly degenerate, with within-seed spreads around 0.004-0.005 in `demeaned_pearson`.

Third, richer structural representations did not help. Named-bundle tractography features predicted
FC worse than streamline-count SC, added no marginal value over count-SC, and did not improve
cognition above the subject-information floor.

Fourth, the SC->FC shortfall is not explained by FC measurement noise. FC was noisy at the
single-edge level but reliable as a whole connectome: the between-session FC reliability ceiling in
Glasser was `demeaned_r=0.491` with fingerprint `top1=0.933`. SC->FC captured only about 17% of
that reproducible FC signal. Moreover, a subject's SC->FC prediction quality was essentially
uncorrelated with that subject's FC reliability (`r=0.01`), and filtering to high-reliability
subjects raised the ceiling without improving SC->FC prediction. The measured gap is therefore not
an FC-side noise artifact.

These checks do not prove that SC can never predict cognition. They make a narrower and more useful
claim: in healthy young adults, under cross-sectional normal-range cognition, the missing utility is
not recovered by obvious changes in dimensionality reduction, model class, structural
representation, or sample size within this regime.

**Figure 4. Closed escape routes.** Oracle-ceiling fraction, reduction-axis robustness, nonlinear
nulls, richer-structure nulls, and FC reliability/noise accounting.

## 6. Predicted Connectomes Carry Family Signal but Objectives Diverge

The failure to improve cognition does not mean predicted connectomes are empty. Predicted SC from FC
carried family structure. In both parcellations, `pred_SC_resid_bvdemo` separated siblings from
unrelated subjects with AUC around 0.81, far above the `bv+demo` baseline around 0.56. This shows
that cross-modal prediction can preserve individualized, heritable signal not reducible to the
subject-information baseline.

However, the signal depends on the objective. A combined predictor optimized for reconstruction
collapsed to chance for sibling separation, around 0.50 AUC, even though it performed well under
reconstruction-style objectives. The practical lesson is that one predicted connectome cannot be
assumed to serve every purpose. Reconstruction, cognition, and identification select different
components of the data.

An exploratory mechanism analysis localized part of the family signal to a small,
FC-predictable, low-variance SC mode. The exact component index was parcellation-dependent: the
mode aligned with PC3 in Glasser and PC4 in 4S456Parcels. PC1 was dominated by sex and brain-volume
confounding and should not be over-interpreted as biological mechanism. Because the mechanism effect
is small and component labels migrate across atlases, we treat it as hypothesis-generating. The
robust claim is the objective tradeoff, not a settled named-component mechanism.

**Figure 5. Signal changes with objective.** Family AUCs for `bv+demo`, raw predicted SC,
residualized predicted SC, and combined predicted SC; reconstruction/identification tradeoff; and
property-selected mechanism mode.

## 7. Discussion

Cross-modal connectome prediction works, but it does not mean what it is often taken to mean. FC
predicts SC better than SC predicts FC, and this directional structure is stable across atlases,
seeds, and reduction checks. Yet downstream cognition tells a stricter story: observed FC helps,
SC does not clear a cheap subject-information baseline, and imputed connectomes do not transfer
utility from one modality to the other.

The main reporting implication is simple. Connectome translation papers should report:

1. Individual-deviation reconstruction metrics such as `demeaned_pearson`, not only raw
   population-mean-dominated correlations.
2. Same-modality oracle references, so cross-modal performance is placed against a model-capacity
   ceiling.
3. A cheap subject-information baseline such as `bv+demo` for downstream behavior or cognition.
4. Separate evaluations for reconstruction, downstream prediction, and identification.
5. Split manifests, leak checks, and completeness checks for any large grid.

The study also narrows where structural signal might still matter. The wall shown here is
regime-specific: healthy young adults, cross-sectional data, and normal-range cognition. Clinical
or lesioned populations, longitudinal designs, developmental change, or very large cohorts could
expose structural signals that are weak or confounded here. The point is not to stop asking those
questions. It is to carry the right checkpoint into them: does the connectome beat subject-level
information?

## 8. Limitations

This is a single-cohort study in HCP Young Adults. The sample is healthy, young, and not
representative of clinical or lifespan variation. The outcomes are normal-range cognitive measures,
not diagnostic endpoints. SC test-retest data were not available in HCP-YA, so FC-side reliability
could be measured directly but SC-side reliability remains data-blocked. The F8 mechanism analysis
is exploratory: the property replicates better than the component label, and its effect size is
small. Finally, the negative results apply to this regime and these data; they should be treated as
a strong checkpoint rather than a universal impossibility theorem.

## 9. Conclusion

FC-SC translation is reproducible, directional, and scientifically informative. But reconstruction
accuracy alone is not enough. In HCP-YA, observed FC is the useful cognitive modality, SC and
imputed connectomes do not beat cheap subject information for cognition, and predicted connectomes
carry family signal only under objectives that preserve it. The field should therefore treat
cross-modal prediction as a set of separable claims: reconstructability, downstream utility, and
identity signal. Each needs its own baseline.
