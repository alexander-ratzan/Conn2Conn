# Modeling directions for identifiable structure–function mapping

**Status:** draft v0.1 for discussion (2026-10-15) · **Scope:** the next round of modeling for subject-level SC ↔ FC
mapping · **Sources:** handwritten notes in this folder (`FM+tx_integrate/`), cited as `FM p<n>` (*FC-SC flow
match*), `MY p<n>` (*Myelination retreat 3*) and `ST p<n>` (*Spatiotemporal integration*).

## Contents

**Main text**

- [Summary](#summary)
- [1. Introduction](#1-introduction)
  - [1.1 The problem](#11-the-problem)
  - [1.2 What we know so far](#12-what-we-know-so-far)
  - [1.3 Three lenses](#13-three-lenses)
- [2. Data and notation](#2-data-and-notation)
  - [2.1 Indices](#21-indices)
  - [2.2 Connectomes](#22-connectomes)
  - [2.3 Covariates and geometry](#23-covariates-and-geometry)
  - [2.4 Transcriptomes](#24-transcriptomes)
  - [2.5 Latent codes](#25-latent-codes)
  - [2.6 Sample axes](#26-sample-axes)
- [3. A common formulation](#3-a-common-formulation)
  - [3.1 The three lenses](#31-the-three-lenses)
  - [3.2 Identifiability](#32-identifiability)
  - [3.3 Evaluation contract](#33-evaluation-contract)
- [4. Direction I: what is shared across views?](#4-direction-i-what-is-shared-across-views)
  - [4.1 Question](#41-question)
  - [4.2 Formulation](#42-formulation)
  - [4.3 Baselines and nulls](#43-baselines-and-nulls)
  - [4.4 What a result would mean](#44-what-a-result-would-mean)
  - [4.5 Risks and open questions](#45-risks-and-open-questions)
- [5. Direction II: what is the distribution of function given structure?](#5-direction-ii-what-is-the-distribution-of-function-given-structure)
  - [5.1 Question](#51-question)
  - [5.2 Formulation](#52-formulation)
  - [5.3 Baselines and nulls](#53-baselines-and-nulls)
  - [5.4 What a result would mean](#54-what-a-result-would-mean)
  - [5.5 Risks and open questions](#55-risks-and-open-questions)
- [6. Direction III: what does molecular architecture explain?](#6-direction-iii-what-does-molecular-architecture-explain)
  - [6.1 Question, and the constant-input principle](#61-question-and-the-constant-input-principle)
  - [6.2 Regional priors from the group atlas](#62-regional-priors-from-the-group-atlas)
  - [6.3 Individualized transcriptomes from an unpaired cohort](#63-individualized-transcriptomes-from-an-unpaired-cohort)
  - [6.4 Precursor: individual-level bilinear integration (TX-6)](#64-precursor-individual-level-bilinear-integration-tx-6)
  - [6.5 What a result would mean](#65-what-a-result-would-mean)
  - [6.6 Risks and open questions](#66-risks-and-open-questions)
- [7. Direction IV: which inductive biases capture individual coupling?](#7-direction-iv-which-inductive-biases-capture-individual-coupling)
  - [7.1 Question](#71-question)
  - [7.2 Edge-wise and graph models](#72-edge-wise-and-graph-models)
  - [7.3 Timeseries models](#73-timeseries-models)
  - [7.4 Baselines and nulls](#74-baselines-and-nulls)
  - [7.5 What a result would mean](#75-what-a-result-would-mean)
  - [7.6 Risks and open questions](#76-risks-and-open-questions)
- [8. Cross-cutting considerations](#8-cross-cutting-considerations)
  - [8.1 Dimensionality](#81-dimensionality)
  - [8.2 Geometry and the choice of SC target](#82-geometry-and-the-choice-of-sc-target)
  - [8.3 Scan length and reliability](#83-scan-length-and-reliability)
  - [8.4 Spatial autocorrelation](#84-spatial-autocorrelation)
  - [8.5 Unpaired-cohort bias](#85-unpaired-cohort-bias)
- [9. Synthesis](#9-synthesis)
  - [9.1 Overview](#91-overview)
  - [9.2 Dependencies](#92-dependencies)
  - [9.3 Suggested order](#93-suggested-order)
- [10. Open questions for collaborators](#10-open-questions-for-collaborators)

**Appendices**

- [Appendix A. Notation](#appendix-a-notation)
  - [A.1 Symbols](#a1-symbols)
  - [A.2 Mapping from the handwritten notes](#a2-mapping-from-the-handwritten-notes)
- [Appendix B. Derivations](#appendix-b-derivations)
  - [B.1 PLS and CCA as one family](#b1-pls-and-cca-as-one-family)
  - [B.2 MAXVAR closed form](#b2-maxvar-closed-form)
  - [B.3 Flow matching](#b3-flow-matching)
  - [B.4 The assignment step](#b4-the-assignment-step)
- [Appendix C. Relation to the codebase](#appendix-c-relation-to-the-codebase)
- [Appendix D. Source notes and errata](#appendix-d-source-notes-and-errata)
  - [D.1 Where each note is used](#d1-where-each-note-is-used)
  - [D.2 Errata found while formalizing](#d2-errata-found-while-formalizing)
- [Appendix E. References](#appendix-e-references)

Each model carries a short stable label (`MV-2`, `FM-2`, `TX-4`, `AR-2`, …) so that experiment plans and other
documents can cite it. Labels are never reused.

---

## Summary

We want models that map between a person's structural connectome (SC) and functional connectome (FC) well enough to
tell individuals apart, and that say something about *why* the two are coupled. Our linear baselines reach modest
subject-level accuracy, and three findings constrain what should come next. Adding more FC conditions to a linear model
does not help. Much of what makes an individual's SC distinctive is parcel geometry, which FC cannot see. MSE-trained
predictions shrink toward the group mean, so accuracy and identifiability pull against each other.

This document organizes four directions around four questions:

- **I. What is shared across views?** Multiview latent models (CCA family, shared + view-specific factors) that
  separate the structure–function "backbone" common to all conditions from condition-specific variation.
- **II. What is the distribution of function given structure?** Generative translation with flow matching, which
  replaces a point prediction by a conditional distribution and gives uncertainty for free.
- **III. What does molecular architecture explain?** Transcriptomic priors: group-level gene expression as a regional
  prior, and individualized expression maps from an unpaired donor cohort linked to our subjects by an assignment.
- **IV. Which inductive biases capture individual coupling?** Edge-wise and graph models conditioned on the subject,
  and models that read FC timeseries directly.

All four share one notation (§2), one formal backbone of three modeling lenses and one definition of identifiability
with its nulls (§3).

---

## 1. Introduction

### 1.1 The problem

Structural connectivity constrains functional connectivity, but the constraint is loose. At the group level, SC
explains a sizeable fraction of FC; at the level of individuals, the subject-specific part of either connectome is
small next to the shared group pattern and to measurement noise. A mapping that reproduces the group mean scores well
on edge-wise accuracy and says nothing about individuals. We therefore evaluate every model on **identifiability**:
does the prediction for subject $i$ resemble subject $i$'s measured connectome more than other subjects'
connectomes (§3.2)?

We work in both directions, $\mathrm{SC} \to \mathrm{FC}$ and $\mathrm{FC} \to \mathrm{SC}$, on the Human
Connectome Project Young Adult cohort (about 1,000 subjects aged 22–35; rest and seven task conditions; diffusion
tractography), mainly in the 360-region Glasser parcellation, with a 456-region replication.

### 1.2 What we know so far

These findings come from our benchmark of linear, latent and graph models and from follow-up ablations. Each one
constrains the directions below.

1. **Linear latent models set a modest bar.** PCA followed by PLS reaches a test demeaned correlation of about 0.16
   for $\mathrm{FC} \to \mathrm{SC}$ and 0.07–0.10 for $\mathrm{SC} \to \mathrm{FC}$ (Glasser, rest FC). Nonlinear
   and graph models trained on full connectomes have not beaten it; several graph models stay at the null.
2. **Rest FC is the most informative condition, and scan length explains much of the order.** Across eight FC
   conditions, both directions and both parcellations, rest is best and the condition order replicates. The amount of
   data per condition tracks it closely (correlation 0.93–0.97 with log scan time).
3. **More conditions do not help a linear model.** Concatenating per-condition PCA blocks and fitting one PLS to SC
   lowers test accuracy for every task block added, even when task blocks keep only the part of task FC that the
   subject's own rest FC cannot predict. PLS gives the task blocks about twice their share of weight, and that weight
   does not generalize. At the same time, the tasks have real and reproducible group-level signatures above rest: the
   information is there, but not in a form a concatenated linear model can use.
4. **Subject-specific SC is dominated by parcel geometry.** A per-edge model using only each subject's own parcel
   volumes (plus inter-regional distance) predicts held-out SC better than FC plus all FreeSurfer covariates
   (demeaned r 0.26 vs 0.22) and identifies 99–100% of test subjects; it gives nothing for FC. This holds for
   volume-normalized and raw streamline counts alike, so it is not only a normalization artifact. FC cannot access
   this signal, which caps what $\mathrm{FC} \to \mathrm{SC}$ can show.
5. **MSE shrinks predictions toward the mean.** Adding loss terms that reward between-subject spread or contrastive
   separation improves rank and top-1 identifiability at some cost in demeaned correlation. Point prediction under a
   squared loss is the wrong objective when individual differences are the target.

Taken together: the subject-specific information that is accessible across modalities is small and partly geometric;
more inputs of the same kind do not help linear models; and point prediction is a poor fit for a loosely constrained,
noisy mapping. The four directions respond to these in different ways (§9).

### 1.3 Three lenses

The directions differ in what they claim, so we classify each model by the lens it is evaluated under (formal
definitions in §3.1):

- **Predictive:** a point estimate $\hat y = f(x, c)$, scored on held-out subjects.
- **Generative:** a conditional distribution $q(y \mid x, c)$, scored as a distribution.
- **Explanatory:** whether a factor $h$ (genes, geometry, a condition) accounts for variance beyond a matched null.

A model can be used under more than one lens; the lens fixes the claim and the null it must beat.

---

## 2. Data and notation

The notation below is used throughout. Appendix A lists every symbol and maps the symbols of the handwritten notes to
it (several collide there).

### 2.1 Indices

| Index | Range | Meaning |
|---|---|---|
| $i$ | $1, \dots, n$ | subjects (connectome cohort) |
| $j$ | $1, \dots, M$ | transcriptome donors (a separate, unpaired cohort; §2.4) |
| $u, v$ | $1, \dots, r$ | brain regions (parcels) |
| $e$ | $1, \dots, d$ | edges, $d = r(r-1)/2$ |
| $c$ | $1, \dots, C$ | FC conditions (rest and tasks) |
| $t$ | $1, \dots, T_c$ | fMRI frames in condition $c$ |
| $g$ | $1, \dots, G$ | genes |
| $\tau$ | $[0, 1]$ | flow time (Direction II) |

For Glasser, $r = 360$ and $d = 64{,}620$; a training split holds about 660 subjects, so every edge-space problem
has $d \gg n$.

### 2.2 Connectomes

- **Timeseries:** $X_i^{(c)} \in \mathbb{R}^{r \times T_c}$, regional BOLD signals of subject $i$ in condition $c$.
- **FC:** $F_i^{(c)} = \operatorname{corr}\big(X_i^{(c)}\big) \in [-1, 1]^{r \times r}$ (optionally Fisher-transformed).
  Rest is $c = 1$; we write $F_i$ when the condition is clear.
- **SC:** $S_i \in \mathbb{R}_{\ge 0}^{r \times r}$, streamline-based structural connectivity (log-transformed; with or
  without normalization by parcel volume).
- **Edge vectors:** $f_i^{(c)} = \operatorname{vech}\big(F_i^{(c)}\big) \in \mathbb{R}^d$ and
  $s_i = \operatorname{vech}(S_i) \in \mathbb{R}^d$, the strict upper triangle. Stacking subjects gives
  $\mathbf{F}^{(c)}, \mathbf{S} \in \mathbb{R}^{n \times d}$.
- **Group means and deviations:** $\bar f^{(c)} = \tfrac{1}{n}\sum_i f_i^{(c)}$ over the training split, and
  $\delta f_i^{(c)} = f_i^{(c)} - \bar f^{(c)}$; likewise $\bar s$, $\delta s_i$. Identifiability lives entirely in the
  deviations.
- **Generic direction:** when a statement holds for both directions, $x_i$ is the source and $y_i$ the target
  ($x_i = s_i, y_i = f_i$ for $\mathrm{SC} \to \mathrm{FC}$ and the reverse for $\mathrm{FC} \to \mathrm{SC}$).

### 2.3 Covariates and geometry

- **Covariates:** $a_i \in \mathbb{R}^p$ (age, sex, and other demographics; FreeSurfer morphometry when stated).
- **Parcel volumes:** $\nu_i \in \mathbb{R}_{> 0}^r$, subject $i$'s own parcel volumes.
- **Distance:** $D \in \mathbb{R}_{\ge 0}^{r \times r}$, inter-regional Euclidean distance between parcel centroids
  (group template; a subject-specific $D_i$ where needed).
- **Region coordinates:** $p_u \in \mathbb{R}^3$, parcel centroid of region $u$.

Geometry is a first-class object here rather than a nuisance covariate, because of finding 4 in §1.2: any claim about
individual SC has to be made relative to it (§3.2, §8.2).

### 2.4 Transcriptomes

Two kinds of expression data enter, and they must be kept apart:

- **Group-level atlas:** $\bar E \in \mathbb{R}^{r \times G}$, regional expression from a post-mortem atlas, mapped to
  the parcellation. It is the same for every subject. Its row $\bar e_u \in \mathbb{R}^G$ is region $u$'s expression
  profile.
- **Individualized maps:** $E_j \in \mathbb{R}^{r \times G}$ for donors $j = 1, \dots, M$ of a separate cohort, imputed
  by a generative model that places each donor's sparse tissue samples into the atlas's latent frame (in active
  development; currently a few hundred donors aged 20–79). Donors carry their own covariates $b_j$. **No donor is a
  connectome subject:** donor maps and connectomes are linked only through an assignment (§6.3).

Gene expression is high-dimensional and spatially smooth, so it usually enters through a low-dimensional code
$z^E_u \in \mathbb{R}^{k_E}$ (PCA of $\bar E$ across regions, or the generative model's latent space).

### 2.5 Latent codes

Most models work on low-dimensional codes rather than raw edges. For a view $m$ (SC, or FC in one condition) with edge
vectors $x_i^m$, a PCA fit on the training split gives a mean $\mu_m \in \mathbb{R}^d$, an orthonormal basis
$V_m \in \mathbb{R}^{d \times k_m}$ and codes

$$
z_i^m = V_m^\top \big(x_i^m - \mu_m\big) \in \mathbb{R}^{k_m},
\qquad
\hat x_i^m = \mu_m + V_m z_i^m .
$$

We write $z_i^S$ for SC and $z_i^{(c)}$ for FC in condition $c$, and $Z_m \in \mathbb{R}^{n \times k_m}$ for stacked
codes. Learned encoders $\phi_m$ replace PCA where noted. The view set is
$\mathcal{M} = \lbrace S, F^{(1)}, \dots, F^{(C)} \rbrace$; with $C = 10$ FC conditions there are
$\lvert \mathcal{M} \rvert (\lvert \mathcal{M} \rvert - 1) = 110$ ordered translations $m \to m'$ (`FM p1`).

### 2.6 Sample axes

Every model in this document treats some unit as its training sample. Which unit it is decides what an input can
contribute:

| Sample axis | Unit | Count (Glasser) | Typical models |
|---|---|---|---|
| subject | $i$ | $n \approx 10^3$ | PCA–PLS, CCA family, flow matching on codes |
| region | $u$ | $r = 360$ | SC–transcriptome PLS, regional maps |
| edge | $(u, v)$ | $d \approx 6.5 \times 10^4$ | group-level edge predictors, bilinear gene models |
| subject × edge | $(i, u, v)$ | $n d \approx 6 \times 10^7$ | edge-wise subject-conditioned models, SMT |
| subject × window | $(i, w)$ | $n \times$ windows | timeseries models |

An input that is constant along the sample axis cannot explain variation along it. This simple fact drives the design
of Direction III (§6.1).

---

## 3. A common formulation

### 3.1 The three lenses

**Predictive.** A map $f_\theta : (x, c) \mapsto \hat y$ fit by minimizing an empirical risk on training subjects,

$$
\hat\theta = \arg\min_\theta \frac{1}{n_{\mathrm{tr}}} \sum_{i \in \mathrm{tr}} \ell\big(y_i, f_\theta(x_i, c)\big),
$$

with $\ell$ the squared error or a composite loss. Under squared error the minimizer is the conditional mean
$\mathbb{E}[y \mid x, c]$, which is pulled toward $\bar y$ whenever $x$ is only weakly informative. This is finding 5
of §1.2 in formal terms.

**Generative.** A conditional distribution $q_\theta(y \mid x, c)$ fit to approximate $p(y \mid x, c)$. A point
prediction is a summary of it, usually the mean of $K$ samples, and the spread of the samples is a per-subject
uncertainty. Generative models are scored as distributions (§3.3), not only through their means.

**Explanatory.** A factor $h$ (a gene map, a geometry term, a condition) explains variance if a model that uses $h$ beats
the same model with $h$ replaced by a draw $\tilde h$ from a null family $\mathcal{H}_0$ that keeps $h$'s nuisance
structure (spatial autocorrelation, marginal distribution, degrees of freedom):

$$
\Delta(h) = \mathrm{score}(f; h) - \mathrm{score}(f; \tilde h),
\qquad
\tilde h \sim \mathcal{H}_0 .
$$

The claim is always relative to $\mathcal{H}_0$: a gene map that beats a constant but not a spatially permuted map
explains nothing gene-specific.

### 3.2 Identifiability

For test subjects $i, i' \in \mathrm{te}$, let $\delta \hat y_i = \hat y_i - \bar y$ and $\delta y_{i'} = y_{i'} - \bar y$ be
prediction and target deviations from the training mean. The **identification matrix** is

$$
R_{i i'} = \operatorname{corr}\big(\delta \hat y_i, \delta y_{i'}\big), \qquad i, i' \in \mathrm{te} .
$$

- **Demeaned correlation:** the mean of the diagonal, $\tfrac{1}{n_{\mathrm{te}}} \sum_i R_{ii}$.
- **Top-1 identification:** the fraction of subjects with $R_{ii} > R_{ii'}$ for all $i' \neq i$.
- **Average rank:** the mean percentile rank of $R_{ii}$ within row $i$ (0.5 at chance, 1 when every subject is
  ranked first).

**Geometry-adjusted identifiability.** Let $g_\psi(\nu_i, D)$ be the geometry-only predictor of finding 4 (fit per edge
on training subjects). A model $f$ adds to geometry if combining them helps,

$$
\Delta_{\mathrm{geo}}(f) = \mathrm{score}\big(f \oplus g_\psi\big) - \mathrm{score}\big(g_\psi\big),
$$

where $f \oplus g_\psi$ is a model given both inputs (or the geometry model fit to the residual of $f$). Equivalently,
scores can be computed on geometry residuals $y_i - g_\psi(\nu_i, D)$. For $\mathrm{FC} \to \mathrm{SC}$ this is the
quantity of interest; comparisons between models with the same inputs remain valid without it.

### 3.3 Evaluation contract

Every direction is scored against the same contract.

**Noise ceilings.** Test–retest reliability bounds what any model can reach: the identification matrix between two
sessions of the same subjects gives an upper bound for each metric.

**Null hierarchy.** Each claim must beat the nulls relevant to it:

| Null | What it controls for | Required for |
|---|---|---|
| N0 group mean, $\hat y_i = \bar y$ | the shared pattern (demeaned r = 0, chance identification) | every model |
| N1 geometry, $g_\psi(\nu_i, D)$ | parcel size and distance | every SC-target claim |
| N2 covariates only, $f(a_i)$ | demographics and morphometry | models that see covariates |
| N3 shuffled pairing (train $x$ permuted across subjects) | the model learning only marginals | generative and deep models |
| N4 spatial nulls (spin-permuted regional maps) | spatial autocorrelation | every regional or gene claim |
| N5 matched-capacity assignment (optimize over spun donor maps) | selection over many candidates | assignment models (§6.3) |

**Distributional scores** (generative lens). For $K$ samples $y_i^{(1)}, \dots, y_i^{(K)} \sim q_\theta(\cdot \mid x_i, c)$:

- the energy distance between the sample set and held-out targets, pooled over subjects;
- **calibration:** coverage of the empirical within-subject spread (e.g. across sessions or windows; §7.3) by the
  predicted spread;
- **within- vs between-subject spread:** samples for one subject should vary less than samples across subjects, or
  they carry no individual information;
- **sample identifiability:** the §3.2 metrics computed on single samples as well as on sample means.

**Leakage rules.** Every data-dependent transform (PCA, scalers, residualizers, assignments) is fit on training
subjects only; splits respect family structure; and **no inference step may use the target** (this matters for
assignment models, §6.3).

---

## 4. Direction I: what is shared across views?

### 4.1 Question

Is there a structure–function "backbone" common to all FC conditions and SC, and how much of each view is
view-specific? A concatenated linear model (finding 3) treats every block's variance as potential signal and overfits
the task blocks. A multiview model separates the shared part from the specific parts before predicting.

### 4.2 Formulation

Let the views be $m \in \mathcal{M}$ with centred codes $Z_m \in \mathbb{R}^{n \times k_m}$ (§2.5), covariance blocks
$\Sigma_{m m'} = \tfrac{1}{n} Z_m^\top Z_{m'}$, and projection vectors $w_m \in \mathbb{R}^{k_m}$. Write
$\Sigma^\lambda_{mm} = (1 - \lambda)\Sigma_{mm} + \lambda I$ for a regularized within-view covariance.

**MV-1 Multiblock PLS** (`MY p2`, `FM p4`). Concatenate the FC blocks, $Z_F = [Z_{(1)}, \dots, Z_{(C)}]$, and fit PLS to
$Z_S$. This is the tested baseline of finding 3. A block-weighted variant scales each block (by its leading singular
value, or with a group penalty per block) so the fit can switch noisy blocks off:

$$
\min_{B} \; \big\lVert Z_S - \textstyle\sum_c Z_{(c)} B_c \big\rVert_F^2 + \sum_c \lambda_c \lVert B_c \rVert_F .
$$

**MV-2 Multiview CCA** (`MY p4–5`). Three classical generalizations of CCA to more than two views:

- **SUMCOR** maximizes the summed pairwise covariance of the projections:

$$
\max_{\lbrace w_m \rbrace} \sum_{m \neq m'} w_m^\top \Sigma_{m m'} w_{m'}
\quad \text{s.t.} \quad w_m^\top \Sigma^\lambda_{mm} w_m = 1 \;\; \forall m .
$$

- **MAXVAR** (generalized CCA) finds a consensus $\Gamma \in \mathbb{R}^{n \times k}$ that every view approximates:

$$
\min_{\Gamma, \lbrace W_m \rbrace} \sum_{m} \big\lVert \Gamma - Z_m W_m \big\rVert_F^2
\quad \text{s.t.} \quad \Gamma^\top \Gamma = I_k .
$$

  Its solution is closed-form: $\Gamma$ holds the top $k$ eigenvectors of $\sum_m Z_m (Z_m^\top Z_m + \lambda I)^{-1} Z_m^\top$
  (Appendix B.2).

- **Least-squares CCA** minimizes pairwise disagreement, $\sum_{m < m'} \lVert Z_m w_m - Z_{m'} w_{m'} \rVert^2$, under an
  average-variance constraint; it has a PLS-like solution.

With $\lambda = 1$ the constraints become $\lVert w_m \rVert = 1$ and the problem reduces to multiview PLS; $\lambda = 0$
gives classical CCA. Regularization moves between them (Appendix B.1).

**MV-3 Tensor CCA** (`MY p5–6`). Pairwise objectives ignore higher-order dependence. Tensor CCA maximizes the
high-order canonical correlation of all views at once,

$$
\max_{\lbrace w_m \rbrace} \; \rho = \frac{1}{n} \sum_{i=1}^n \prod_{m \in \mathcal{M}} \big(w_m^\top z_i^m\big)
\quad \text{s.t.} \quad w_m^\top \Sigma^\lambda_{mm} w_m = 1 ,
$$

which is a rank-1 decomposition of the whitened covariance tensor
$\mathcal{C} = \tfrac{1}{n} \sum_i z_i^{1} \circ z_i^{2} \circ \dots \circ z_i^{\lvert\mathcal{M}\rvert}$ (Luo et al., 2015).
The tensor has $\prod_m k_m$ entries, so it is feasible only for a few views or small $k_m$.

**MV-4 Shared + view-specific factor models.** A generative reading of the same idea:

$$
z_i^m = A_m \gamma_i + B_m \eta_i^m + \varepsilon_i^m,
\qquad
\gamma_i \sim \mathcal{N}(0, I), \;\; \eta_i^m \sim \mathcal{N}(0, I), \;\; \varepsilon_i^m \sim \mathcal{N}(0, \Psi_m),
$$

with a shared latent $\gamma_i$ (the backbone) and view-specific latents $\eta_i^m$. Probabilistic CCA (Bach and
Jordan, 2005) is the two-view case; group factor analysis and JIVE-type decompositions generalize it. Prediction across
views goes through the shared factor: $\hat z^S_i = A_S\, \mathbb{E}[\gamma_i \mid z_i^{(1)}, \dots, z_i^{(C)}]$.

**MV-5 Linear-Gaussian conditional** (`FM p3`: "a linear baseline for flow matching"). If source and target codes are
jointly Gaussian (as under MV-4), the conditional distribution is available in closed form:

$$
p\big(z^y \mid z^x\big) = \mathcal{N}\Big(\mu_y + \Sigma_{yx} \Sigma_{xx}^{-1} (z^x - \mu_x), \;\;
\Sigma_{yy} - \Sigma_{yx} \Sigma_{xx}^{-1} \Sigma_{xy}\Big).
$$

This is the exact linear counterpart of conditional flow matching (FM-2): it gives a mean, a full predictive
covariance and samples. With regularized $\Sigma_{xx}$ and conditioning on several views it is the "multiview CCA with
a predictive module" of the notes.

### 4.3 Baselines and nulls

MV-1 (unweighted) is the reference. N0, N3 for all; N1 for SC targets.

### 4.4 What a result would mean

- A shared subspace whose dimension and identifiability are stable across conditions would be a direct, interpretable
  answer to "what part of FC is structurally constrained". Its loadings are edge maps that can be read anatomically.
- If MV-4 predicts SC better than MV-1, finding 3 was a failure of concatenation, not an absence of signal.
- The view-specific parts measure condition-dependent coupling, linking to the reproducible task signatures.
- If nothing beats MV-1, the linear multiview route is closed, and Directions II and IV must carry the argument.

### 4.5 Risks and open questions

- Many views with $n \approx 660$ make MV-3 and unregularized MV-2 fragile; work on PCA codes with shrinkage.
- Conditions differ in scan length, so the shared factor may weight long scans by reliability rather than biology.
- Should SC be one of the views (symmetric multiview), or the target of a shared FC factor (asymmetric)? The first
  answers "what is shared", the second "what predicts SC".

---

## 5. Direction II: what is the distribution of function given structure?

### 5.1 Question

A given SC is compatible with many FCs: brain state, arousal, task and measurement noise all change FC without changing
structure. Two subjects with similar SC can have different FC (`FM p1`). A deterministic map $\hat f = f(s, c)$ can at
best return the conditional mean, which is shrunk toward the group (finding 5). Modeling $p(F \mid S, c)$ instead
answers two questions a point predictor cannot: how much FC variability does structure leave open, and is a subject's
own FC typical under the distribution their SC implies?

### 5.2 Formulation

All models work on codes (§2.5). For $\mathrm{SC} \to \mathrm{FC}$ the source is $z^S_i$ and the target
$z^{(c)}_i$; condition $c$ enters through a learned embedding $e(c)$, and flow time through an embedding
$\gamma(\tau)$.

**FM-0 Conditional regression baseline** (`FM p2`). An MLP $\hat z^{(c)}_i = f_\theta\big(z^S_i, e(c)\big)$ trained
with squared error. It extends the per-condition linear models to one shared nonlinear model across conditions and is
the reference every flow model must beat.

**FM-1 Paired deterministic bridge** (`FM p1–2`, `MY p3`). Interpolate between a subject's own source and target,
$z_\tau = (1 - \tau)\, z^S_i + \tau\, z^{(c)}_i$ with $\tau \sim \mathcal{U}(0, 1)$, and regress the velocity:

$$
\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{i, c, \tau} \Big[ \big\lVert v_\theta\big(z_\tau, \tau, e(c)\big) - \big(z^{(c)}_i - z^S_i\big) \big\rVert_2^2 \Big].
$$

At inference, set $z(0) = z^S_i$ and integrate $\mathrm{d}z / \mathrm{d}\tau = v_\theta(z, \tau, e(c))$ to $\tau = 1$.
Source and target codes must live in a common space (equal $k$, or a learned shared encoder).

Two properties set what FM-1 can and cannot add over FM-0 (Appendix B.3):

- One Euler step of size 1 from $\tau = 0$ returns
  $z^S_i + \mathbb{E}[z^{(c)} - z^S \mid z^S = z^S_i] = \mathbb{E}[z^{(c)} \mid z^S_i]$, the regression solution of FM-0.
- Integrated fully, the learned flow transports the *distribution* of sources onto the distribution of targets
  (marginal preservation; Liu et al., 2022). Its outputs are not shrunk, but it is still a deterministic map: it cannot
  express conditional spread, and where interpolation paths cross it re-pairs sources and targets, so subject pairing is
  not guaranteed to survive.

FM-1 is therefore best understood as a de-shrinking deterministic translator, not a conditional generative model.

**FM-2 Conditional flow matching from noise** (`FM p2–3`). Start from noise and condition on the source:
$z_0 = \epsilon \sim \mathcal{N}(0, I)$, $z_1 = z^{(c)}_i$, $z_\tau = (1 - \tau)\,\epsilon + \tau\, z^{(c)}_i$, and

$$
\mathcal{L}_{\mathrm{CFM}} = \mathbb{E}_{i, c, \tau, \epsilon} \Big[ \big\lVert v_\theta\big(z_\tau, \tau, z^S_i, e(c)\big) - \big(z^{(c)}_i - \epsilon\big) \big\rVert_2^2 \Big].
$$

At inference, draw $\epsilon$, integrate $\mathrm{d}z / \mathrm{d}\tau = v_\theta(z, \tau, z^S_i, e(c))$ from 0 to 1, and
repeat $K$ times. The sample mean is a point estimate and the sample covariance a per-subject uncertainty. By the
conditional flow matching theorem (Lipman et al., 2023) the minimizer generates $p(z^{(c)} \mid z^S, c)$ when the
model class is rich enough.

The practical difficulty is that each subject contributes one target per condition, so the model must learn a
conditional distribution from one sample per conditioning value, relying on smoothness in $z^S$. Two ways to get more
targets per subject: the second rest session, and windows of the timeseries (§7.3).

**FM-3 Any-to-any translation** (`FM p1`). Replace the fixed source by modality and state tokens $(m, m')$ and learn
one field $v_\theta\big(z_\tau, \tau, z^m_i, e(m), e(m')\big)$ for all ordered pairs in $\mathcal{M}$, including
$\mathrm{FC} \to \mathrm{FC}$ across conditions (110 translations per subject). Sharing one model across translations
is a form of multi-task regularization; the per-pair error matrix is itself a result (which views predict which).

**FM-4 Coupling and path variants.** Minibatch optimal-transport couplings (Tong et al., 2024) straighten paths and
reduce integration steps; rectified flow (Liu et al., 2022) iterates the coupling; stochastic interpolants and
Schrödinger bridges add noise along the path so a paired bridge can represent conditional spread. FM-1 with a
noise-augmented path is the natural middle ground between FM-1 and FM-2.

### 5.3 Baselines and nulls

- Point predictions: FM-0 and the linear models; MV-5 as the linear generative reference.
- Distributions: MV-5's Gaussian conditional; N3 (shuffled pairing) to show that samples depend on the subject's own
  source.

### 5.4 What a result would mean

- If FM-2's sample means match FM-0 while its spread is calibrated against within-subject variability, we have a model
  of how much FC structure leaves undetermined, per subject and per condition.
- If samples are identifiable (§3.3) but means are not more accurate than FM-0, the gain is in representing
  individuality without shrinkage, which matters for downstream uses (simulation, augmentation).
- If FM-2 collapses to the marginal (samples ignore $z^S$), the conditional signal is too weak for a generative model
  at this sample size; MV-5 then sets the achievable bar.
- The FM-3 error matrix gives a full map of cross-view predictability (rest ↔ task ↔ SC).

### 5.5 Risks and open questions

- Sample size: about 660 training subjects per condition; generative models in 64–256-dimensional code space may
  overfit. Shared conditioning across conditions (FM-3) helps.
- Code space: PCA codes are linear and discard detail; a learned autoencoder adds a second model to validate.
- Scoring generative models on identifiability needs care: sample means discard the spread, single samples are noisy.
- Is the right target FC itself, or the FC deviation $\delta f$? Generating deviations removes the shared pattern the
  model would otherwise spend capacity on.

---

## 6. Direction III: what does molecular architecture explain?

### 6.1 Question, and the constant-input principle

Regional gene expression is a plausible substrate for both structural wiring and functional coupling. The question is
what it can explain about *individual* connectomes, given that the dense atlas is group-level and the individualized
maps come from different people.

**TX-0 Constant-input principle.** If an input $h$ is the same for every subject, then on the subject axis any model
$f(x_i, h)$ is just a function $f_h(x_i)$ of the subject's own data: $h$ can change the function class (an inductive bias)
but cannot add subject-specific information. Formally, $p(y_i \mid x_i, h) = p(y_i \mid x_i)$ when $h$ does not vary with
$i$. Consequences:

1. The group atlas $\bar E$ can only act as a **regional prior** (§6.2): it varies across regions and edges, so it can
   share statistical strength between edges or explain the group-mean connectome, but it cannot by itself improve
   identifiability.
2. Individualized maps $E_j$ can carry subject information only through how donors are assigned to subjects (§6.3). If
   the assignment $\pi(i)$ is a function of the subject's own source data and covariates, $\pi(i) = h(x_i, a_i)$, then
   $E_{\pi(i)}$ is a function of $(x_i, a_i)$ too, and by the data-processing inequality

$$
I\big(y_i ; E_{\pi(i)} \mid x_i, a_i\big) = 0 .
$$

   Any predictive gain from assigned donor maps is then a gain in inductive bias (the donor pool acts as a structured
   prior), not new information. This is why the assignment models are explanatory first (`MY p11`).
3. "Atlas fill" (giving every subject $\bar E$; `MY p8`) is an ablation that should change nothing at the subject level.
   Doing it confirms the pipeline, not a hypothesis.

### 6.2 Regional priors from the group atlas

**TX-1 Region-axis SC–transcriptome PLS** (`FM p4`). On the region axis, let $\bar S \in \mathbb{R}^{r \times r}$ hold the
group connectivity profiles (row $u$ is region $u$'s mean connectivity). PLS between $\bar S$ and $\bar E$ finds gene
combinations whose regional pattern covaries with connectivity:

$$
\max_{w_S, w_E} \operatorname{cov}\big(\bar S w_S, \bar E w_E\big)
\quad \text{s.t.} \quad \lVert w_S \rVert = \lVert w_E \rVert = 1 .
$$

The resulting gene scores $\bar E w_E \in \mathbb{R}^r$ are regional maps that can be passed to subject-level models as
fixed node features. With $r = 360$ samples and strong spatial autocorrelation, every component needs N4 (spin) nulls.

**TX-1b Bilinear connectivity–transcriptome model.** On the edge axis, model the group connectome as a bilinear form
of regional expression,

$$
\bar F \approx \bar Z^E\, \mathcal{O}\, (\bar Z^E)^\top ,
$$

with $\bar Z^E \in \mathbb{R}^{r \times k_E}$ the gene codes and $\mathcal{O} \in \mathbb{R}^{k_E \times k_E}$ a learned
"gene-pair rule" matrix (Kovács et al., 2020). $\mathcal{O}$ is directly interpretable: entry $(a, b)$ says how strongly
co-expression of gene programs $a$ and $b$ predicts connectivity. The precursor model of §6.4 is this form applied per
individual.

**TX-2 Gene-informed subject models.** Pass $z^E_u$ as node features to the subject-conditioned architectures of
Direction IV (AR-2, AR-3). By TX-0, any gain is an inductive-bias gain; the test is whether it helps identifiability
beyond the same architecture with spin-permuted gene features (N4).

### 6.3 Individualized transcriptomes from an unpaired cohort

The donors $j = 1, \dots, M$ have individualized maps $E_j$ but no connectomes; our subjects have connectomes but no
expression. An assignment $\Pi \in \lbrace 0, 1 \rbrace^{n \times M}$ links them, with $\Pi_{ij} = 1$ when donor $j$ is
assigned to subject $i$ (each subject gets one donor; a donor may serve several subjects up to a capacity $\kappa$). The
subject's assigned map is $\tilde E_i = \sum_j \Pi_{ij} E_j$.

**TX-3 Covariate-matched assignment** (`MY p8–9`). Fix $\Pi$ by matching covariates, a linear assignment problem

$$
\Pi^{\mathrm{cov}} = \arg\min_{\Pi} \sum_{i, j} \Pi_{ij} \lVert a_i - b_j \rVert_2^2
\quad \text{s.t.} \quad \sum_j \Pi_{ij} = 1, \;\; \sum_i \Pi_{ij} \le \kappa ,
$$

then fit $f_\theta\big(x_i, \tilde E_i\big)$. By TX-0 this adds no information beyond $a_i$; it tests whether
covariate-structured expression is a better prior than the covariates themselves.

**TX-4 Jointly optimized assignment** (`MY p9–10`). Learn $\theta$ and $\Pi$ together,

$$
\min_{\theta, \Pi} \sum_{i=1}^{n} \Big\lVert y_i - f_\theta\Big(x_i, \textstyle\sum_j \Pi_{ij} E_j\Big) \Big\rVert_2^2 ,
$$

by alternation: fit $\theta$ at the current assignment; compute the cost $K_{ij} = \lVert y_i - f_\theta(x_i, E_j) \rVert^2$;
update $\Pi = \arg\min_\Pi \operatorname{tr}(\Pi^\top K)$ by the Hungarian algorithm (or a capacitated variant).
Initialize from TX-3.

The step that needs care is inference. The notes fix $\theta$ and re-solve $\Pi$ for test subjects (`MY p10`), but
$K_{ij}$ uses the test target $y_i$, so this is not a prediction. It is a **matching score**: how well the best donor
explains each held-out subject. That is a legitimate explanatory quantity, but choosing the best of $M$ donors always
improves fit, so it must be compared with the same optimization over null donor maps (N5: spin-permuted $E_j$, or donors
with shuffled regions). The explanatory claim is $\Delta$ in §3.1 with $\mathcal{H}_0$ = N5.

**TX-5 Amortized and soft assignment** (`MY p11–12`). Two ways to make TX-4 predictive and smoother:

- **Amortized:** a side model $h_\phi(x_i, a_i)$ predicts the assignment (a distribution over donors) from source-side
  data only, trained to match the TX-4 assignments on training subjects. This is a valid predictor, bounded by TX-0.
- **Optimal transport:** replace the permutation by a transport plan $\Pi \in \mathbb{R}_{\ge 0}^{n \times M}$ with
  fixed marginals and an entropy term, solved by Sinkhorn iterations. Soft plans make the objective differentiable in
  $\theta$ and $\Pi$ jointly and let each subject draw on several donors.

### 6.4 Precursor: individual-level bilinear integration (TX-6)

The earlier spatiotemporal-integration sketch (`ST p1–3`) is the same assignment idea at the level of whole
connectomes, with donors whose maps cover different subsets of regions. In current notation, donor $j$'s map
$E_j \in \mathbb{R}^{r_j \times G}$ covers $r_j$ regions with mask $\mathcal{R}_j$, and

$$
\min_{\Pi, \mathcal{O}} \sum_{i, j} \Pi_{ij} \Big[ \frac{1}{r_j^2} \big\lVert E_j\, \mathcal{O}\, E_j^\top - (F_i)_{\mathcal{R}_j} \big\rVert_F^2
+ \lambda \lVert a_i - b_j \rVert_2^2 \Big],
$$

with $\Pi$ a transport plan between subjects and donors, $(F_i)_{\mathcal{R}_j}$ the connectome restricted to donor $j$'s
regions, and $1 / r_j^2$ normalizing for the number of regions. It alternates a closed-form update of $\mathcal{O}$
(a linear regression on vectorized gene-pair products) with an assignment update. It is kept as a precursor: TX-1b is
its group-level form and TX-4 its subject-level form.

### 6.5 What a result would mean

- TX-1 / TX-1b components that survive spin nulls identify gene programs aligned with connectivity gradients
  (interpretability; `MY p8`).
- A TX-4 matching score above N5 shows that individualized expression with realistic variability carries structure
  aligned with individual connectome variation, the outcome the notes target (`MY p11`).
- A TX-5 predictive gain above covariates alone would show that donor expression is a useful structured prior, though
  by TX-0 not a new information source.

### 6.6 Risks and open questions

- Cohort mismatch: the donor cohort spans ages 20–79, our subjects 22–35; matching should restrict or weight by age.
- Parcellation alignment: donor maps are defined on an atlas parcellation that must be resampled to the connectome
  parcellation, and atlas coverage is uneven (e.g. hemispheric sampling).
- The individualized maps are model outputs. Any finding inherits the imputation model's assumptions; results should
  be stable across its versions.
- Capacity $\kappa$ and $M \ll n$: with a few hundred donors and about a thousand subjects, each donor serves several
  subjects, which limits how individual the assignment can be.

---

## 7. Direction IV: which inductive biases capture individual coupling?

### 7.1 Question

Models trained on whole connectomes (subject axis) have $n \approx 660$ samples in $10^4$–$10^5$ dimensions, and the
graph models we tried stayed at the null. Moving to the subject × edge axis multiplies the number of training samples by
$d$, and lets architectural priors (locality, symmetry, region identity) do work that sample size cannot. Which priors
help the subject-specific part of the mapping?

### 7.2 Edge-wise and graph models

**AR-1 Edge-wise models** (`MY p7`). Predict each edge from the two regions' connectivity profiles:

$$
\hat y_{i, uv} = f_\theta\big(S_{i, u:}, S_{i, v:}\big) , \qquad f_\theta \in \lbrace \text{MLP}, \text{SMT}, \text{GNN} \rbrace ,
$$

trained over all subjects and edges. Two design rules: $f_\theta$ must be symmetric in $(u, v)$ (pool the two region
embeddings symmetrically), and the target should be the deviation $\delta y_{i, uv}$ so the model cannot score by
reproducing $\bar y$.

**AR-2 SMT with a subject-conditioned CLS token** (`FM p5`, `MY p7`). The Spatiomolecular Transformer (SMT) encodes each
region as a sequence of gene-expression tokens; a shared ("dual") transformer encoder is applied to regions $u$ and $v$,
a CLS token summarizes each region, and an MLP head maps the pair to the edge value. Originally it predicts the
group-level connectome from $\bar E$ alone (edge axis). For individuals, the CLS token is conditioned on the subject:

$$
\hat y_{i, uv} = f_\theta\Big(\mathrm{SMT}\big(\mathrm{CLS}_{i,u} \oplus \mathrm{tok}(\bar e_u)\big), \;
\mathrm{SMT}\big(\mathrm{CLS}_{i,v} \oplus \mathrm{tok}(\bar e_v)\big)\Big) ,
$$

with candidate CLS inputs (`FM p5`): the subject's SC code $z^S_i$; region coordinates $p_u$; or a two-layer spatial
graph network over region $u$'s neighbourhood with $S_{i, u:}$ as features. By TX-0, only CLS inputs that vary across
subjects ($z^S_i$, $S_{i, u:}$) can make predictions subject-specific; coordinates and gene tokens are priors.

**AR-3 Graph networks with molecular node features** (`FM p5`). A two-layer GNN or masked graph attention network on the
subject's SC graph (edges $S_i$) with gene codes $z^E_u$ as node features, read out per edge for FC. Subject information
enters through the graph topology and weights; genes give nodes an identity beyond their position.

**AR-4 Models of the deviation from the mean.** Graph and nodal models applied to $\delta S_i$ and $\delta F_i$ (or to
low-rank factors of them), rather than to full connectomes, so that the shared pattern does not dominate the loss.
This reframes the graph models that stayed at the null.

### 7.3 Timeseries models

**AR-5 Region tokens from timeseries** (`MY p7`). Read FC's raw material instead of FC. For a window
$X^{(c)}_{i, u}[t : t + L]$ of length $L$, build one token per region,

$$
h_{i, u} = \phi\big(X^{(c)}_{i, u}[t : t + L]\big) + \alpha_{i, u} \odot z^E_u + e(c) ,
$$

with $\phi$ a temporal encoder, $z^E_u$ the region's gene code and $\alpha_{i, u} \in \mathbb{R}^{k_E}$ a subject- and
region-specific gate that up- or down-weights gene programs. Self-attention across the $r$ tokens,
$\mathrm{softmax}\big(Q K^\top / \sqrt{d_k}\big) V$, mixes regions; a pooled output predicts the SC code $z^S_i$, decoded through
the SC PCA basis. The gate must be computed from the subject's own data, $\alpha_{i, u} = g_\psi(h_{i, :})$, so that it
exists for test subjects. This is the "natural direction" for SC prediction in the notes: timeseries to SC.

**AR-6 Windows as samples.** Short windows give many noisy FC estimates per subject, $F_i^{(c)}(w)$ for windows $w$.
They serve three purposes:

- augmentation for AR-5 and for any FC-input model;
- multiple targets per subject for the generative models of Direction II (FM-2), which otherwise see one target per
  condition;
- an empirical within-subject distribution $p\big(F^{(c)}(w) \mid i\big)$, against which a generative model's spread can
  be calibrated at matched window length (§3.3).

### 7.4 Baselines and nulls

The linear models and FM-0 on the same inputs; N0–N3; N4 for any gene-feature claim (spin-permuted $z^E$); for
timeseries models, the same architecture on FC (no temporal information) to show that time adds something.

### 7.5 What a result would mean

- AR-1/AR-4 beating the subject-axis models would show that the bottleneck was sample size, not the mapping.
- AR-2/AR-3 beating spin-permuted gene features would show that molecular identity is a useful prior for individual
  coupling (an inductive-bias result under TX-0).
- AR-5 beating FC-input models would show that FC discards information about SC that the timeseries keep.

### 7.6 Risks and open questions

- Compute: the subject × edge axis has about $6 \times 10^7$ samples per epoch for Glasser; subsample edges per batch.
- Leakage across edges: edges of one subject are not independent; splits must stay at the subject level.
- Window length trades the number of samples against the noise of each FC estimate.

---

## 8. Cross-cutting considerations

### 8.1 Dimensionality

Edge spaces have $d \approx 6.5 \times 10^4$ dimensions against $n \approx 660$ training subjects. Predicting a
high-dimensional target from a high-dimensional source overfits easily (`MY p8`). The remedies used here are: low-rank
codes (§2.5), shrinkage and PLS-style regularization, sharing one model across conditions (FM-3) and moving to the
subject × edge axis (Direction IV).

### 8.2 Geometry and the choice of SC target

Parcel size dominates the subject-specific part of SC (finding 4), with or without volume normalization. Three ways to
handle it, not mutually exclusive:

1. report every SC-target result with geometry-adjusted identifiability (§3.2);
2. predict a geometry-residualized target $s_i - g_\psi(\nu_i, D)$;
3. make $\mathrm{SC} \to \mathrm{FC}$ the primary direction, since FC is not geometry-dominated in the same way.

Which of these is primary is an open choice (§10).

### 8.3 Scan length and reliability

Conditions differ in scan length (rest about four times longer than any task), and accuracy tracks it. Any comparison
between conditions, and any multiview weight, is partly a reliability comparison. A length-matched control (rest
truncated to a task's frame count) separates the two.

### 8.4 Spatial autocorrelation

Regional maps (gene scores, model loadings, regional accuracy) are spatially smooth, so naive permutation nulls are too
permissive. Spin tests (Alexander-Bloch et al., 2018) or variogram-matched surrogates are required for every regional
claim (N4).

### 8.5 Unpaired-cohort bias

Assignment models (§6.3) inherit the donor cohort's composition (age, sex, tissue sampling) and the imputation model's
assumptions. Results should be checked under alternative matchings and model versions.

---

## 9. Synthesis

### 9.1 Overview

| Label | Model | Lens | Direction | Sample axis | New data needed | Risk | A positive result shows |
|---|---|---|---|---|---|---|---|
| MV-1 | multiblock PLS (+ block weights) | predictive | FC → SC | subject | — | low | concatenation can use task FC once weighted |
| MV-2 | multiview CCA (SUMCOR, MAXVAR, LS-CCA) | explanatory | symmetric | subject | — | low | a shared SC–FC backbone across conditions |
| MV-3 | tensor CCA | explanatory | symmetric | subject | — | medium | higher-order shared structure |
| MV-4 | shared + specific factors | predictive / explanatory | both | subject | — | medium | shared vs condition-specific coupling |
| MV-5 | linear-Gaussian conditional | generative | both | subject | — | low | linear bar for conditional generation |
| FM-0 | conditional MLP | predictive | both | subject | — | low | nonlinear multi-condition gain |
| FM-1 | paired deterministic bridge | predictive | both | subject | — | medium | de-shrunk deterministic translation |
| FM-2 | conditional flow matching | generative | both | subject | windows (optional) | high | per-subject distribution of FC given SC |
| FM-3 | any-to-any translation | generative | all pairs | subject | — | high | full cross-view predictability map |
| TX-1 | region-axis SC–gene PLS, bilinear model | explanatory | — | region / edge | group atlas | low | gene programs aligned with connectivity |
| TX-2 | gene node features | predictive | both | subject × edge | group atlas | medium | molecular prior helps individual coupling |
| TX-3 | covariate-matched donors | predictive | both | subject | donor maps | medium | donor expression as a structured prior |
| TX-4 | optimized assignment | explanatory | both | subject | donor maps | high | individual expression aligns with connectomes |
| TX-5 | amortized / OT assignment | predictive | both | subject | donor maps | high | a usable predictive version of TX-4 |
| AR-1 | edge-wise models | predictive | SC → FC | subject × edge | — | medium | sample size was the bottleneck |
| AR-2 | SMT with subject CLS | predictive | SC → FC | subject × edge | group atlas | medium | molecular tokens + subject context |
| AR-3 | GNN / masked GAT + gene nodes | predictive | SC → FC | subject × edge | group atlas | medium | graph prior on the subject's SC |
| AR-4 | deviation-space graph models | predictive | both | subject | — | medium | graph models work once the mean is removed |
| AR-5 | timeseries region tokens | predictive | FC → SC | subject × window | timeseries | high | timeseries carry SC information FC discards |
| AR-6 | windows as samples | (enabler) | — | subject × window | timeseries | low | augmentation and calibration targets |

### 9.2 Dependencies

- MV-1 (done) → MV-2, MV-4 → MV-5. MV-5 is the linear reference for FM-2.
- FM-0 → FM-1 → FM-2 → FM-3. FM-2 benefits from AR-6 (more targets per subject).
- TX-0 frames all of Direction III. TX-1 needs only the group atlas; TX-3 → TX-4 → TX-5 need the donor maps.
- AR-1 → AR-2, AR-3 (adding molecular features); AR-6 → AR-5.
- Geometry adjustment (§3.2) applies to every SC-target result.

### 9.3 Suggested order

A first round could take one low-risk entry point per direction, each answering its question at small cost: MV-2/MV-4
(is there a shared backbone?), FM-0 plus MV-5 (the deterministic and generative bars), TX-1 (gene programs aligned with
connectivity, needs only the atlas), and AR-1 on deviations (does the subject × edge axis help?). The high-risk items
(FM-2, TX-4, AR-5) then start from established baselines. Experiment plans set the actual priorities.

---

## 10. Open questions for collaborators

1. **Primary target.** Given finding 4, should $\mathrm{SC} \to \mathrm{FC}$ be the main direction, or should SC targets
   be geometry-residualized?
2. **What counts as success for a generative model** at this sample size: calibrated spread, identifiable samples, or
   sample means that beat point predictors?
3. **Direction III's claim.** Is the goal interpretability (gene programs aligned with connectivity) or predictive
   gain? TX-0 says the second is bounded; the first is open.
4. **Donor–subject matching.** Which covariates or connectome features should drive the assignment, and how should the
   age-range mismatch be handled?
5. **Timeseries scope.** Rest only, or all conditions? Which window lengths?
6. **Shared encoders.** Should all directions share one learned code space (e.g. a multiview autoencoder), so that
   flow, multiview and timeseries models are directly comparable?

---

## Appendix A. Notation

### A.1 Symbols

| Symbol | Meaning |
|---|---|
| $n, M$ | numbers of connectome subjects and transcriptome donors |
| $r, d, C, G$ | numbers of regions, edges ($d = r(r-1)/2$), FC conditions, genes |
| $X_i^{(c)}$ | regional timeseries, $r \times T_c$ |
| $F_i^{(c)}, S_i$ | FC in condition $c$, SC ($r \times r$) |
| $f_i^{(c)}, s_i$ | edge vectors ($\operatorname{vech}$) |
| $\mathbf{F}^{(c)}, \mathbf{S}$ | stacked edge vectors, $n \times d$ |
| $\bar f, \bar s$; $\delta f_i, \delta s_i$ | training means; deviations from them |
| $x_i, y_i$ | generic source and target |
| $a_i, b_j$ | subject and donor covariates |
| $\nu_i, D, p_u$ | parcel volumes, distance matrix, region centroids |
| $\bar E, \bar e_u$ | group expression atlas ($r \times G$), region $u$'s profile |
| $E_j, \tilde E_i$ | donor $j$'s individualized map; map assigned to subject $i$ |
| $z_i^m, Z_m, V_m, \mu_m$ | PCA code, stacked codes, basis, mean of view $m$ |
| $z_i^S, z_i^{(c)}, z^E_u$ | SC code, FC code in condition $c$, gene code of region $u$ |
| $\mathcal{M}$ | set of views (SC and FC conditions) |
| $\Sigma_{mm'}, \Sigma^\lambda_{mm}$ | cross-covariance of codes; regularized within-view covariance |
| $w_m, W_m, \Gamma$ | projection vectors / matrices, multiview consensus |
| $\gamma_i, \eta_i^m$ | shared and view-specific latent factors |
| $\tau, z_\tau, v_\theta, \epsilon$ | flow time, interpolant, velocity field, noise |
| $e(\cdot), \gamma(\tau)$ | condition / modality embedding, time embedding |
| $\Pi, \kappa, K$ | assignment (or transport plan), donor capacity, assignment cost |
| $\mathcal{O}$ | gene-pair rule matrix of the bilinear model |
| $R$ | identification matrix |
| $g_\psi$ | geometry-only predictor |

### A.2 Mapping from the handwritten notes

| Notes | Where | Meaning | Here |
|---|---|---|---|
| $T$ | `FM p1` | timeseries | $X_i^{(c)}$ |
| $F, S$ | `FM` | FC, SC | $F_i^{(c)}, S_i$ |
| $X^{FC}$; $Y^{FC}, Y^{SC}$ | `MY p1` | timeseries; FC and SC | $X_i^{(c)}$; $F_i^{(c)}, S_i$ (stacked: $\mathbf{F}^{(c)}, \mathbf{S}$) |
| $X^G$, $G$ | `MY p1`, `FM p4` | group expression | $\bar E$ |
| $\tilde X^G$, $g_i$ | `MY p8`, `ST p1` | individualized / donor expression | $E_j$ |
| $\mathcal{D}_i$ | `MY p1` | a subject's data tuple | (not used) |
| $P, Q$ | `MY p2` | PLS loadings | $w_m$, $W_m$ |
| $T, U$ | `MY p2` | PLS scores | $Z_m w_m$ |
| $G$ | `MY p4` | MAXVAR consensus | $\Gamma$ |
| $P$ | `MY p9`, `ST p2` | assignment matrix | $\Pi$ |
| $C$ | `MY p5` | covariance tensor | $\mathcal{C}$ |
| $C$ | `MY p10` | assignment cost | $K$ |
| $C$ (count), $c$ | throughout | conditions | $C$, $c$ |
| $t$ | `FM p1–3` | flow time | $\tau$ |
| $t_i$ | `ST p2` | covariate vector | $a_i, b_j$ |
| $c_j$ | `ST p1` | a connectome | $F_i$ (subjects indexed by $i$) |
| $m_i$ | `ST p2` | region mask | $\mathcal{R}_j$ |
| $\beta$ | `ST p2` | gene-pair matrix | $\mathcal{O}$ |
| $\alpha_{i, r}$ | `MY p7` | gene-latent gate | $\alpha_{i, u}$ |
| $N$ | `MY`, `ST` | subjects | $n$ |
| $i, j$ as regions | `FM p5`, `MY p7` | edge endpoints | $u, v$ |

---

## Appendix B. Derivations

### B.1 PLS and CCA as one family

For two centred code matrices $Z_x, Z_y$, both PLS and CCA solve

$$
\max_{w_x, w_y} \; w_x^\top \Sigma_{xy} w_y
\quad \text{s.t.} \quad w_x^\top \Sigma^{\lambda_x}_{xx} w_x = 1, \;\; w_y^\top \Sigma^{\lambda_y}_{yy} w_y = 1 .
$$

The Lagrangian conditions give the generalized eigenproblem

$$
\begin{pmatrix} 0 & \Sigma_{xy} \\ \Sigma_{yx} & 0 \end{pmatrix}
\begin{pmatrix} w_x \\ w_y \end{pmatrix}
= \rho
\begin{pmatrix} \Sigma^{\lambda_x}_{xx} & 0 \\ 0 & \Sigma^{\lambda_y}_{yy} \end{pmatrix}
\begin{pmatrix} w_x \\ w_y \end{pmatrix} .
$$

With $\lambda_x = \lambda_y = 1$ the right-hand matrix is the identity and the solution is the SVD of $\Sigma_{xy}$
(PLS-SVD); with $\lambda = 0$ it is classical CCA. Successive components come from the next eigenpairs (CCA, PLS-SVD)
or from deflation (PLS regression).

### B.2 MAXVAR closed form

For fixed $\Gamma$, each $W_m$ is a ridge regression, $W_m = (Z_m^\top Z_m + \lambda I)^{-1} Z_m^\top \Gamma$. Substituting
back, the objective becomes $\sum_m \lVert (I - H_m) \Gamma \rVert_F^2$ with the hat matrix
$H_m = Z_m (Z_m^\top Z_m + \lambda I)^{-1} Z_m^\top$. Under $\Gamma^\top \Gamma = I$, minimizing it is maximizing
$\operatorname{tr}\big(\Gamma^\top \textstyle\sum_m H_m\, \Gamma\big)$ (exactly for $\lambda = 0$, where $H_m$ is a projection), so
$\Gamma$ holds the top $k$ eigenvectors of $\sum_m H_m$ (Carroll, 1968; Kettenring, 1971).

### B.3 Flow matching

**Conditional flow matching.** For an interpolant $z_\tau = (1 - \tau) z_0 + \tau z_1$ with $(z_0, z_1)$ drawn from a
coupling, the regression target $z_1 - z_0$ has conditional expectation

$$
v^\star(z, \tau) = \mathbb{E}\big[z_1 - z_0 \mid z_\tau = z\big],
$$

and minimizing $\mathbb{E} \lVert v_\theta(z_\tau, \tau) - (z_1 - z_0) \rVert^2$ recovers $v^\star$. The ODE
$\mathrm{d}z / \mathrm{d}\tau = v^\star(z, \tau)$ started from the law of $z_0$ has, at every $\tau$, the law of $z_\tau$
(Lipman et al., 2023; Liu et al., 2022). With $z_0 = \epsilon$ independent of the conditioning variable and
$v^\star$ conditioned on it, the endpoint law is the conditional target law (FM-2).

**Paired bridge (FM-1).** With $z_0 = x$ and $z_1 = y$ drawn as subject pairs:

- at $\tau = 0$, $z_\tau = x$, so $v^\star(x, 0) = \mathbb{E}[y \mid x] - x$ and one Euler step of size 1 gives
  $\mathbb{E}[y \mid x]$, the squared-error regression solution;
- the full ODE maps the law of $x$ onto the law of $y$ (marginal preservation) through a deterministic, invertible
  map. It cannot represent $p(y \mid x)$ with nonzero spread, and where paths cross, $v^\star$ averages over pairs and
  the induced pairing differs from the data pairing.

### B.4 The assignment step

For fixed $\theta$, $\min_\Pi \operatorname{tr}(\Pi^\top K)$ over one-to-one assignments is a linear assignment problem,
solved exactly in $O(n^3)$ by the Hungarian algorithm. With capacities $\kappa$ it is a transportation problem (a
min-cost flow). Adding an entropy term, $\min_\Pi \operatorname{tr}(\Pi^\top K) - \varepsilon H(\Pi)$ with fixed marginals,
gives the Sinkhorn solution $\Pi = \operatorname{diag}(\boldsymbol{a})\, \exp(-K / \varepsilon)\, \operatorname{diag}(\boldsymbol{b})$,
differentiable in $K$ and hence in $\theta$ (Cuturi, 2013).

---

## Appendix C. Relation to the codebase

The only place this document refers to the repository. Experiment plans live in
[`repo_spec_docs/`](../repo_spec_docs/).

| Item | Status | Where |
|---|---|---|
| PCA–PLS, PLS-SVD, conditional Gaussian (closed form) | implemented | `models/architectures/crossmodal_pca_pls.py`, `crossmodal_conditional_gaussian.py` |
| Multiblock PCA–PLS over FC conditions (MV-1), predictive-residual variant | tested (no gain) | `scripts/experiments/joint_taskfc2sc/` |
| FC conditions as model inputs, per-condition caches | implemented | `data/hcp_dataset.py` |
| Condition comparison in both directions and parcellations | done | `scripts/experiments/task_bidirectional_bench/` |
| Geometry-only per-edge model ($g_\psi$) and edge-wise evaluation | implemented | `data/data_viz.py`, `scripts/experiments/covariate_ablation/node_size.py` |
| Composite losses (between-subject spread, contrastive terms) | implemented | `scripts/experiments/composite_loss/` |
| Graph and nodal models (GCN, nodal GNN / MLP) | implemented, at the null | `models/architectures/` |
| SMT | implemented in GeneEx2Conn (group level) | `GeneEx2Conn/models/smt*.py` |
| Individualized donor expression maps | in development | Seq2GeneEx `gtex_gp` (separate project) |
| Flow matching, multiview CCA, assignment models, timeseries models | not started | — |

---

## Appendix D. Source notes and errata

### D.1 Where each note is used

| Note | Content | Section |
|---|---|---|
| `FM p1` | data, all-FC-to-SC flow loss, modality / state tokens, distributional framing | §5.1, FM-1, FM-3 |
| `FM p2` | noise-to-target flow, MLP baseline, deterministic flow | FM-0, FM-1, FM-2 |
| `FM p3` | conditional flow matching, sampling for uncertainty, linear multiview baseline | FM-2, MV-5 |
| `FM p4` | multiview PCA + PLS, SC–transcriptome PLS | MV-1, TX-1 |
| `FM p5` | GNN with gene nodes, SMT with conditional CLS | AR-2, AR-3 |
| `MY p1` | data definitions | §2 |
| `MY p2–3` | PLS, multi-task concatenation, MSE and flow variants | MV-1, FM-1 |
| `MY p4–6` | SUMCOR, MAXVAR, least-squares and tensor CCA | MV-2, MV-3 |
| `MY p7` | edge-wise models, timeseries tokens with gene gates | AR-1, AR-5 |
| `MY p8–12` | atlas fill, covariate-matched and optimized assignment, nulls, amortized assignment | §6.1, TX-3 to TX-5 |
| `ST p1–3` | individual-level bilinear integration with assignment | TX-6 |

### D.2 Errata found while formalizing

- `FM p1`: the flow loss uses an unsquared norm; flow matching regresses with the squared norm.
- `FM p1` vs `FM p2`, `MY p3`: the bridge runs FC → SC on p1 and SC → FC on p2; here each use states its direction.
- `MY p4`: with $Y_c \in \mathbb{R}^{N \times d}$ (subjects × edges), SUMCOR terms are $w_c^\top Y_c^\top Y_{c'} w_{c'}$ and the
  constraint $w_c^\top Y_c^\top Y_c w_c = 1$; MAXVAR's residual is $\Gamma - Y_c W_c$ (the note transposes $Y_c$). The sum
  runs over pairs $c \neq c'$.
- `MY p5`: the least-squares CCA constraint averages per-view variances, $\tfrac{1}{C} \sum_c w_c^\top Y_c^\top Y_c w_c = 1$.
- `MY p9`: the assignment is written $N \times N$, but donors and subjects are different cohorts of different sizes;
  it is $n \times M$ with capacities.
- `MY p10`: re-solving the assignment for test subjects uses their target; this is a matching score, not a prediction
  (§6.3).
- `ST p2`: the constraints $\sum_i P_{ij} \le 1/N$ with $P$ binary cannot both hold; the intended object is a transport
  plan with marginal constraints.

---

## Appendix E. References

- Alexander-Bloch, A. F., et al. (2018). On testing for spatial correspondence between maps of human brain structure
  and function. *NeuroImage*.
- Bach, F. R., and Jordan, M. I. (2005). A probabilistic interpretation of canonical correlation analysis. Technical
  report, UC Berkeley.
- Carroll, J. D. (1968). Generalization of canonical correlation analysis to three or more sets of variables.
  *Proceedings of the APA*.
- Cuturi, M. (2013). Sinkhorn distances: lightspeed computation of optimal transport. *NeurIPS*.
- Hawrylycz, M. J., et al. (2012). An anatomically comprehensive atlas of the adult human brain transcriptome.
  *Nature*.
- Kettenring, J. R. (1971). Canonical analysis of several sets of variables. *Biometrika*.
- Kovács, I. A., Barabási, D. L., and Barabási, A.-L. (2020). Uncovering the genetic blueprint of the *C. elegans*
  nervous system. *PNAS*.
- Lipman, Y., et al. (2023). Flow matching for generative modeling. *ICLR*.
- Liu, X., Gong, C., and Liu, Q. (2022). Flow straight and fast: learning to generate and transfer data with rectified
  flow. *arXiv:2209.03003*.
- Luo, Y., et al. (2015). Tensor canonical correlation analysis for multi-view dimension reduction. *IEEE TKDE*.
- Tong, A., et al. (2024). Improving and generalizing flow-based generative models with minibatch optimal transport.
  *TMLR*.
- Jamison, K. W., et al. (2024). Release the Krakencoder: a unified brain connectome translation and fusion tool.
  *bioRxiv*.
- Sarwar, T., Tian, Y., Yeo, B. T. T., Ramamohanarao, K., and Zalesky, A. (2021). Structure–function coupling in the
  human connectome: a machine learning approach. *NeuroImage*.

References are to be checked and completed (SMT and imputation-model citations to be added by the authors).
