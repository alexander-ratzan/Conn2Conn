# Repo Spec v4

**Purpose:** v4 work: modeling directions beyond the linear baselines, drawn from
[`modeling_directions.md`](../modeling/modeling_directions.md) (the technical reference; model labels such as FM-1 and
MV-5 refer to it). First item: joint multidirectional translation across SC and every FC condition with one shared
nonlinear backbone, as a regression model and as a deterministic flow (E1); a CCA-latent benchmark that sets how E1 is
framed (E2); a conditional generative follow-up (E3).
**Status:** draft (v3 stays active until it is closed; open v3 items move here at closure) · **Started:** 2026-10-20 ·
**Predecessor:** [`spec_doc_v3.md`](spec_doc_v3.md) · **Format:** [`spec_conventions.md`](spec_conventions.md) ·
**Modeling reference:** [`modeling_directions.md`](../modeling/modeling_directions.md)

**Contents:** [Status](#status) · [1. Conventions](#1-conventions) · [2. Experiments](#2-experiments) ·
[3. Infrastructure](#3-infrastructure) · [4. Carried over from v3](#4-carried-over-from-v3) ·
[5. Backlog](#5-backlog-not-scheduled) · [6. Change log](#6-change-log)

## Status

| ID | Title | Status | Depends on | Owner |
|---|---|---|---|---|
| E1 | Joint multidirectional translation: one backbone with view tokens, regression vs deterministic flow (FM-0 + tokens, FM-3 deterministic) | in progress (E1.0 done; I1 built, G0 CPU checks pass; pilot E1.3 waits for E2) | I1, E2 | agent:modeling |
| E2 | CCA-latent bidirectional benchmark: multiview CCA latent of FC as the encoding for FC → SC and the target for SC → FC | done 2026-10-20 | I2, D4 | agent:modeling |
| E3 | Conditional generative translation: noise-start flow (FM-2) and noisy bridge (FM-4) with view tokens | outline | E1 gate G3 | — |
| I1 | Joint translation infrastructure: pooled FC basis, (subject, source, target) data module, token model, integrator, runner | in progress | — | agent:modeling |
| I2 | Multiview CCA library module (MAXVAR fit / encode / decode), shared by E1 and E2 | done 2026-10-20 | — | agent:modeling |
| D1 | Views and the true rest target | decided 2026-10-20 | — | user |
| D2 | Code space: pooled FC basis with equal-variance (total-norm) block scaling, SC basis, $k$ grid | decided 2026-10-20 | E1.0 | user |
| D3 | E1 compute envelope and autonomous execution | decided 2026-10-20 | — | user |
| D4 | E2 autonomous execution | decided 2026-10-20 | — | user |

---

## 1. Conventions

Inherited from v3 §1 unchanged: experiment homes, reported metrics (test Pearson r, demeaned r, average rank, top-1),
seeds 0–4, val selection with test reported, `sbatch` compute, PNG figures; v2:D1–D6 and v3:D4 stay in force. Glasser
first; 4S456Parcels is a later replication with the same configs. **Benchmark reference for SC pairs:** v3:E0
(`scripts/experiments/task_bidirectional_bench/results/Glasser/{fc2sc,sc2fc}/tables/`), which uses the same matched
917-subject cohort and per-seed splits, so E1 differences are paired by seed and subject.

**Notation (from `modeling_directions.md` §2):** views $m \in \mathcal{M}$; code $z^m_i \in \mathbb{R}^k$ of subject
$i$ in view $m$; view embedding $e(m)$; flow time $\tau \in [0, 1]$.

### Decisions (`D`)

| ID | Decision | Rationale |
|---|---|---|
| D1 | **Views (user 2026-10-20):** main model on 9 views, $\mathcal{M}$ = {SC, `rest`, 7 tasks} (72 ordered pairs). `rest` (4 runs, xcp-d combined relmat) is the true rest view and the rest target, as in every earlier result. Arm E1.5c adds `rest_S1` and `rest_S2` (per-session mean of the LR and RL run matrices) with a nesting mask: pairs between `rest` and either session are excluded from training and evaluation, because a session is half of `rest`'s data. S1 ↔ S2 is then the in-model test–retest translation. | Continuity with v3:E0 / E2 and v2:E2.2; the mask prevents near-identity leakage; the sessions add a ceiling and the two-sample targets E2 needs. |
| D2 | **Decided (user 2026-10-20):** one FC basis from a PCA of the stacked train-split FC of all FC views, each block centered on its own train mean (condition means stored and added back at decoding by the target token) and divided by its total (Frobenius) norm, so every view carries the same total variance (standard multiblock / consensus PCA block scaling, Westerhuis et al. 1998; the first-singular-value alternative is multiple factor analysis, Escofier & Pagès; E1.0c: total-norm scaling is better for rest and for cross-view identification); SC keeps its own train PCA (also divided by its total norm); codes share one dimension $k \in$ {64, 128, 256, 512}, bounded above by the SC rank (658; E1.0b). Arms (E1.5): per-condition bases (as v3:E0; A3 only, since E1.0 shows their codes are not comparable across views), and an SC basis rotated onto the FC basis (orthogonal Procrustes on train pairs). | The bridge interpolates codes directly, so all views need comparable coordinates; per-condition centering keeps condition offsets out of the flow, so the model works on subject deviations (the quantity demeaned metrics score). |
| D3 | **Decided (user 2026-10-20):** E1.1–E1.6 run autonomously; cap 25 GPU-h (estimate 15–18); per-job time limit = pilot time per trial × 1.5. Stop and report on a failed gate (G0, G1, G3), > 20 % task errors after one retry, a log / config mismatch, a code change outside I1's files, or the cap. Commits local (no push). Relaxing a threshold needs the user. | Small models in code space; explicit exits as v3:D1–D3. |
| D4 | **Decided (user 2026-10-20):** E2 runs autonomously through E2.0–E2.3, CPU only, cap 15 CPU-node-hours. Stop and report on a failed gate (G0, G1), > 20 % task errors after one retry, a log / config mismatch, a code change outside I2 and E2's folder (plus the E1 import switch), or the cap. Commits local (no push). Relaxing a threshold needs the user. The E1 A3 / A4 pilot (E1.3) waits for E2's result. | E2 is closed form and cheap; it decides E1's framing, so it runs first. |

## 2. Experiments

### E1 — Joint multidirectional translation   (slug: `joint_flow_matching`) · status: in progress · owner: agent:modeling

- **Question:** does one shared nonlinear backbone, trained on every ordered pair of views with source and target
  tokens, translate between SC and FC conditions better than one model per pair? Does integrating it as a
  deterministic flow (paired bridge) add individual information beyond the regression it reduces to at one step?
- **Why now:** v3 found that no linear per-pair or multi-block model gets more out of task FC than rest alone (v3:E0,
  v3:E2), and that short task scans are the weak conditions. A shared backbone is the one route left that can borrow
  strength across conditions; the bridge is the cheapest flow-matching model and the base for the generative E3.

#### E1.A Hypotheses and decision rules

All readouts are on the test split, per ordered pair, paired by seed. **Support rule (every H):** the seed-mean Δ is
above 2 SE over the 5 seeds and has the same sign in at least 4 of 5 seeds. Pair groups are averaged within a seed
before the rule is applied.

| ID | Hypothesis | Comparison | Primary metric | Supported if | Deliverable |
|---|---|---|---|---|---|
| H1a | Sharing helps the weak conditions | L4 − L1 (shared vs per-pair, both linear) and A3 − A0 (joint model vs the bench), over the 14 task ↔ SC pairs | demeaned r | support rule for each comparison, reported separately; the per-task Δ vs log scan volume (Spearman over 7 tasks × 2 directions) is reported, not gated | `h1_task_sc.csv`, `fig_h1_delta_vs_scantime.png` |
| H1b | A nonlinear shared model beats a linear shared model | A3 − L4, all 72 pairs and the 14 task ↔ SC pairs | demeaned r | support rule | `h1_task_sc.csv` |
| H2 | Joint training does not hurt rest | A3 − A0 on rest → SC and SC → rest | demeaned r | non-inferior: seed-mean Δ > −0.01 with the lower 2-SE bound above −0.02 | `h2_rest.csv` |
| H3 | Integration adds individual information | A4 integrated − A4 one-step; A4 integrated − L2 and − L3; all 72 pairs | average rank (demeaned r and top-1 reported) | support rule vs all three | `h3_integration.csv`, `fig_h3_steps.png` |
| H4 | Many FC sources beat one for SC | multi-source − rest-only SC prediction, for A3 and A4 (uniform average of the 8 FC → SC predictions in code space) and for L4 (average of the 8 FC encodings of the shared latent) | demeaned r | support rule; read next to v3:E2 (linear multi-block regression: no gain) | `h4_multisource.csv`, `fig_h4_multisource.png` |
| H5 | Map of cross-view predictability | all arms, 9 × 9 pairs, with identity and ceiling references | demeaned r, average rank | descriptive (no rule); share of FC → FC pairs where A3 beats identity | `h5_matrix_<arm>.csv`, `fig_h5_matrix.png` |

Identification metrics are unchanged by uniform rescaling of a subject's deviation, so H3 can only be met by a
direction-dependent change; the variance-matched ridge and the Gaussian OT map are the linear controls for that.

#### E1.B Design

**Data.**

| Item | Setting |
|---|---|
| Cohort and splits | v3:E0 matched cohort (917 Glasser subjects), `HCP_Base(fc_conditions=[rest, 7 tasks], expose_fc_sessions=True)`, seeds 0–4 (seed 0: 659 / 74 / 184) |
| Views (D1) | SC, `rest`, emotion, gambling, language, motor, relational, social, wm: 72 ordered pairs (16 SC ↔ FC, 56 FC → FC) |
| Edge vectors | upper triangle, 64,620 edges; SC log1p, `invnodevol` normalized; FC Pearson correlation as cached (both as v3:E0) |
| Leakage rules | means, bases, block scales, closed-form fits and model weights fit on train subjects only; val for early stopping and selection; test scored once per selected model |

**Code space (D2).** For each view $m$, train mean $\mu_m$ and block scale $s_m$ (total Frobenius norm of the
train-centered block, so each view enters with equal total variance; E1.0c). FC views share one orthonormal basis $W_{\mathrm{FC}} \in \mathbb{R}^{E \times k}$, the top-$k$
right singular vectors of the stacked blocks $(X^{(m)} - \mu_m) / s_m$; SC has its own $W_{\mathrm{SC}}$. Codes and
decoding:

$$
z^m_i = W_m^\top \frac{x^m_i - \mu_m}{s_m}, \qquad \hat x^{m'}_i = \mu_{m'} + s_{m'}\, W_{m'}\, \hat z^{m'}_i .
$$

$k \in$ {64, 128, 256, 512}, searched; the same $k$ for every view, so source and target codes have one dimension. The upper
bound is set by SC: its basis has at most $n_{\mathrm{train}} - 1$ = 658 components (E1.0b). The bases are computed once
per seed at $k$ = 512 and sliced.

**Models.**

| Arm | Model | Scope | Detail |
|---|---|---|---|
| A0 | per-pair PCA-PLS learnable | 16 SC ↔ FC pairs | v3:E0 `seed_records.csv` (per-condition bases, 16-trial tunes); not rerun |
| A1 | closed-form linear references L1–L4 (below) | 72 pairs, CPU | $k$ = the seed's A3 choice |
| A2 | per-pair MLP (FM-0) | rest → SC, SC → rest, wm → SC, SC → wm | A3 backbone without tokens; own search |
| A3 | joint MLP with tokens (FM-0 + tokens) | 72 pairs | residual form $\hat z^{m'} = z^m + f_\theta\big(z^m, e_{\mathrm{src}}(m), e_{\mathrm{tgt}}(m')\big)$ |
| A4 | joint paired bridge (deterministic FM-3) | 72 pairs | $z_\tau = (1 - \tau)\, z^m_i + \tau\, z^{m'}_i$, $\tau \sim \mathcal{U}(0, 1)$; loss $\lVert v_\theta(z_\tau, \gamma(\tau), e_{\mathrm{src}}(m), e_{\mathrm{tgt}}(m')) - (z^{m'}_i - z^m_i) \rVert^2$; scored at one Euler step and integrated with RK4 at 4, 16 and 64 steps |

**Linear references (arm A1).** All closed form, fit on train subjects in the same code space, hyperparameters chosen
on val:

| Ref | Model | What it controls |
|---|---|---|
| L1 | per-pair ridge, $\hat z^{m'} = z^m B_{mm'}$ (penalty by generalized cross-validation on train) | the per-pair linear conditional mean, shrunk toward the group |
| L2 | L1 with each output dimension rescaled to the target's train SD | de-shrinkage alone (H3) |
| L3 | Gaussian OT map $T(z) = \mu_{m'} + A (z - \mu_m)$ between the train source and target code distributions (Ledoit–Wolf covariances); **uses no subject pairing** | matching distributions alone, which a fully integrated flow also does (H3) |
| L4 | multiview CCA translator (MAXVAR, MV-2): shared train subject scores $G \in \mathbb{R}^{n \times d}$, $G^\top G = I$, and per-view encoders $A_m$ minimizing $\sum_m \lVert G - Z_m A_m \rVert^2 + \lambda \lVert A_m \rVert^2$ over all 9 views; $G$ = top-$d$ eigenvectors of $\sum_m Z_m (Z_m^\top Z_m + \lambda I)^{-1} Z_m^\top$, $A_m = (Z_m^\top Z_m + \lambda I)^{-1} Z_m^\top G$; ridge decoders $D_m$ from $G$ to $Z_m$. Translation $\hat z^{m'} = z^m A_m D_{m'}$; several sources: average their encodings. $d$ ∈ {16, 32, 64, 128}, $\lambda$ on a log grid | the shared linear model, the reference for the joint backbone (H1b) |

Shared backbone (A2–A4): input projection, then $L$ residual blocks (LayerNorm, linear to width $w$, SiLU, dropout,
linear), then a linear output of size $k$. Separate source and target token tables of dimension $d_e$; $\gamma$ is a
64-dimensional sinusoidal embedding of $\tau$. A3 and A4 at one Euler step have the same functional form, which is
what gate G3 checks.

**References per pair:** the target-view train mean (N0; demeaned r = 0 by construction, so it anchors Pearson r);
for FC → FC, identity, $\hat x^{m'}_i = \mu_{m'} + (x^m_i - \mu_m)$; the **reconstruction ceiling**, the target's own
$k$-dimensional reconstruction $\mu_{m'} + s_{m'} W_{m'} z^{m'}_i$; and for A4 a shuffled-pairing null (N3; seed 0, the
selected configuration retrained with train subjects permuted between source and target).

**Training (A2–A4).**

| Setting | Value |
|---|---|
| Objective | code-space squared error, mean over dimensions |
| Batches | 512 (subject, source, target) triples; an epoch covers every train subject × every pair once (47,448 triples at 659 subjects), shuffled |
| Optimizer | AdamW, cosine decay to 0 over at most 200 epochs, gradient clipping at norm 1 |
| Early stopping | val code loss over all pairs, patience 20 epochs; the best-val-loss weights are kept |

**Tuning and selection.**

| Setting | Value |
|---|---|
| Search | Optuna TPE (seeded), 24 trials per arm × seed (A2: per pair), median pruning on val loss from epoch 20 |
| Space | learning rate log-uniform 1e-4 to 3e-3; weight decay log-uniform 1e-6 to 1e-1; $w$ ∈ {256, 512, 1024}; $L$ ∈ {2, 3, 4}; dropout 0 to 0.4; $d_e$ ∈ {16, 32, 64}; $k$ ∈ {64, 128, 256, 512} |
| Selection metric | mean val demeaned r over the arm's pairs, on decoded edges; A4 tuned at RK4 16 steps, and the reported step count chosen afterwards on val |
| Edge rule | none on $k$ (bounded by SC rank); a winner at the edge of $w$ or $L$ in ≥ 3 seeds is reported, not extended |

The joint arms are selected on one pooled objective, while A0 and A2 are selected per pair: a selection advantage for
the per-pair references.

**Scoring.** `models/eval/metrics.py` on decoded test edges: Pearson r, demeaned r (against the target view's train
mean), average rank and top-1 (chance 1/184 at seed 0), and the spread ratio (norm of the predicted deviation over the
norm of the true deviation, from `compute_prediction_norm_ratio`).

#### E1.C Steps, gates and outputs

| Step | What | Compute | Gate (stop and report if it fails) | Output |
|---|---|---|---|---|
| E1.0 | Pre-checks: pooled vs own variance, cross-view identification, spectra | CPU | — | done 2026-10-20 (E1.0 results below) |
| E1.1 | I1 build and its acceptance checks | CPU, 1 short GPU job | **G0:** I1 accept criteria | `checks/` logs |
| E1.2 | Synthetic gates: 9 simulated views from a shared latent through view-specific linear maps plus noise ($n$ = 659, $k$ = 128), so every conditional mean is known | 1 GPU, < 1 h | **G1:** A3 and A4 one-step predictions reach R² ≥ 0.9 against the true conditional mean on every pair (token routing); A4 with shuffled pairing has demeaned r ≤ 0.02 | `checks/synthetic/`; **G1 failed 2026-10-20** (job 20459033): A3 min R² 0.865, A4 one-step 0.712 vs ridge oracle 0.957; G1b passed (0.001). Stopped per D3 |
| E1.3 | Pilot, seed 0: A1, A3, A4 full tunes | 2 GPU jobs | **G3:** (a) A3 and A4 one-step test predictions correlate ≥ 0.95 edge-wise, mean over pairs; (b) A3 rest → SC val demeaned r ≥ A0 seed-0 val (0.154) − 0.02; (c) every pair's metrics finite; (d) selected trials stop before epoch 200 | pilot records; measured time per trial, which sets the job time limits (× 1.5) |
| E1.4 | Main: A3 and A4 seeds 1–4, A1 seeds 0–4, A2 (4 pairs) seeds 0–4 | GPU arrays, CPU | — | `records.csv`, `hyperparams.csv` |
| E1.5 | Arms on each seed's selected E1.4 configuration (retrained, no new search): (a) per-condition bases, A3; (b) SC basis rotated onto the FC basis by orthogonal Procrustes on train pairs, A3 and A4; (c) views + `rest_S1`, `rest_S2` with the nesting mask (D1), A3 and A4 | 1 GPU array | — | `arms.csv` |
| E1.6 | Analysis: H1–H5, ceilings, identity, N3 (seed 0), tables, figures, write-up | CPU | — | below |
| E1.7 | Optional, after E1.6: identity-aware loss. A3 and A4 retrained at each seed's selected configuration with MSE + demeaned corr-eye (v3 composite term, fixed v2 reference scale; weights {0.1, 0.3, 1}), no new search | 1 GPU array | — | `identity_loss.csv`; Δ vs E1.4 on all metrics |

**Deliverables** (`scripts/experiments/joint_flow_matching/results/Glasser/`):

- `tables/`: `records.csv` (arm × seed × pair × metric, val and test), `hyperparams.csv`, `ceilings.csv`,
  `h1_task_sc.csv`, `h2_rest.csv`, `h3_integration.csv`, `h4_multisource.csv`, `h5_matrix_<arm>.csv`, `arms.csv`,
  and `summary.md` (one row per hypothesis: Δ ± SE, seeds positive, supported or not).
- `figures/`:
  1. `fig_sc_pairs.png`: the 16 SC ↔ FC pairs, A0–A4, 4 metrics, with the ceiling marked;
  2. `fig_h1_delta_vs_scantime.png`;
  3. `fig_h3_steps.png`: average rank and demeaned r vs integration steps (one-step to 64), L2 and L3 as lines;
  4. `fig_h4_multisource.png`;
  5. `fig_h5_matrix.png`: 9 × 9 demeaned r for L1, L4, A3, A4, and A3 − L4;
  6. `fig_arms.png`: E1.5 Δ vs the E1.4 configuration.
- Write-up `scripts/experiments/joint_flow_matching/joint_flow_matching.md`; the `experiments_index.md` row; the spec
  result line.

#### E1.D Outcomes and what follows

| Outcome | Reading | Next |
|---|---|---|
| H1a or H1b supported, H3 not | sharing (linear or nonlinear) is what helps; the flow adds nothing as a point predictor | E3 on the joint backbone with a noise start (FM-2 + tokens) |
| H3 supported | integration adds individual structure the regression shrinks away | E3 on the bridge with path noise (FM-4) |
| H1a, H1b and H3 not supported | in code space at this sample size, per-pair linear maps are sufficient | E3 only for the distributional question, against MV-5 |
| H2 fails | joint training interferes with rest | a pair-weighting arm before E3 |
| H4 supported | first gain from task FC for SC (v3:E2 found none linearly) | regional follow-up (v3:E2 follow-up 2) |

- **Accept:** G0, G1 and G3 pass; every arm × pair × seed scored on test next to its ceiling; H1a, H1b, H3 and H4 decided by the
  support rule and H5 delivered; the deliverables above; A0 read from v3:E0 without rerunning.
- **Budget / autonomy:** D3. Estimate 15–18 GPU-h: about 5 min per trial (pilot measures it), 24 trials per arm ×
  seed with 4 trials packed per GPU ≈ 1 GPU-h per arm × seed for A3 and A4 (10 GPU-h), A2 ≈ 2, E1.5 ≈ 3, E1.2 ≤ 1.
- **Caveats:** C!2 for every FC → SC pair (v3:C!2); task scan length is confounded with condition (v3:E0); the code
  space keeps only part of each subject's individuality (E1.0 finding 4), which the ceiling makes visible; A0 uses
  per-condition bases and 16-trial tunes, so A3 − A0 includes the basis change (E1.5a isolates it).

#### E1.0 results — pre-checks (2026-10-20)

Scripts: [`checks/pooled_basis_variance.py`](../../scripts/experiments/joint_flow_matching/checks/pooled_basis_variance.py),
[`checks/basis_spectrum.py`](../../scripts/experiments/joint_flow_matching/checks/basis_spectrum.py) (CPU, seeds 0–1,
Glasser, 917-subject cohort; CSVs in `checks/pooled_basis_variance/`). Held-out = val + test subjects (258), projected
with train means and bases.

**E1.0a: variance kept on held-out subjects, pooled basis / own basis** (pooled basis over `rest` + 7 tasks with
first-singular-value block scaling, superseded by total-norm scaling in E1.0c; mean of 2 seeds):

| View | k = 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| `rest` | 0.79 | 0.87 | 0.92 | 0.94 | 0.97 |
| wm | 0.97 | 1.03 | 1.08 | 1.14 | 1.18 |
| language | 0.83 | 0.92 | 0.99 | 1.04 | 1.10 |
| social | 0.89 | 0.96 | 1.02 | 1.08 | 1.13 |
| motor, gambling, relational, emotion | 0.94–1.03 | 1.01–1.10 | 1.08–1.18 | 1.15–1.25 | 1.22–1.32 |

1. **Every task keeps more held-out variance in the pooled basis than in its own** from k ≈ 32 up (up to +32 %).
   A per-task PCA fit on 659 short scans overfits its train split; the pooled basis is estimated from 8 × as many
   rows.
2. **`rest` pays a small price:** it loses 6–9 % of its held-out variance at k = 64–128 (3 % at 256). Block scaling
   reduces rest's loss by about a fifth to a third (unscaled: 0.73 / 0.83 / 0.88 / 0.92 / 0.94) and changes the tasks by
   ≤ 0.03. Pooling the two sessions instead of `rest` (arm E1.5c) gives rest 0.95–1.00 at k ≥ 64.
3. **Per-condition codes are not comparable across views:** identifying a held-out subject's task FC from their rest
   FC by correlating codes is at chance in own-basis codes (k = 64: top-1 0.000–0.006, average rank 0.45–0.56), and
   well above chance in pooled codes (top-1 0.03–0.09, average rank 0.72–0.81; S1 → S2 0.53 / 0.94). The paired bridge
   (A4) therefore needs the pooled basis; the per-condition arm (E1.5a) is run for A3 only.
4. **Much of the cross-state fingerprint lies beyond 64 components:** in edge space the same identification reaches
   top-1 0.53–0.82 (average rank 0.97–0.99; S1 → S2 0.95). Two consequences: (a) the k-dimensional code bounds what any
   E1 model can recover, so a per-view reconstruction ceiling is reported next to every result; (b) for FC → FC pairs
   the identity reference (copy the subject's source deviation) is a strong bar.

**E1.0b: spectra** (mean of 2 seeds; train variance share, and variance captured on held-out subjects):

| Basis | Max components | k for 95 % of train variance | Held-out, k = 128 | k = 256 | k = 512 | all components |
|---|---|---|---|---|---|---|
| own, `rest` | 658 | 523 | 0.41 | 0.47 | 0.51 | 0.53 |
| own, each task | 658 | 571–583 | 0.17–0.25 | 0.20–0.29 | 0.23–0.32 | 0.24–0.34 |
| own, SC | 658 | 499 | 0.33 | 0.42 | 0.53 | 0.56 |
| pooled FC (block-scaled): `rest` / tasks | 5,264 | 3,738 | 0.39 / 0.21–0.28 | 0.45 / 0.26–0.34 | 0.52 / 0.33–0.41 | 0.74 / 0.54–0.62 |

5. **Maximum components:** a per-view PCA on 659 train subjects has at most 658 (train subjects − 1); the pooled
   basis has 8 × 658 = 5,264 (seed 1: 657 and 5,256).
6. **95 % of train variance needs nearly every component** (499–583 of 658 per view; 3,738 pooled): beyond the first
   few hundred components the spectrum is flat (subject-specific variance and noise), so a train-variance threshold
   does not pick a useful $k$.
7. **Held-out variance is the useful guide.** In own bases it saturates: $k$ = 512 keeps 94–97 % of what all 658
   components capture. The pooled basis keeps gaining past 512 (rest 0.52 → 0.74 at full rank), because its span is
   8 × wider; $k$ > 512 is out of reach while SC (658 components) must share the code dimension, so it is noted for E3,
   not used in E1.
8. **$k$ grid {64, 128, 256, 512} (user):** 512 keeps ≥ 0.94 of each own basis's capturable
   held-out variance and stays below the SC rank.

**E1.0c: how much each view contributes to the pooled spectrum** (script
[`checks/pooled_block_contributions.py`](../../scripts/experiments/joint_flow_matching/checks/pooled_block_contributions.py);
share of the pooled train variance carried by `rest`, mean of 2 seeds; an even share over 8 views is 0.125):

| Block scaling | Total variance | Top 10 components | Top 128 | Top 256 | Top 512 |
|---|---|---|---|---|---|
| none | 0.028 | 0.041 | 0.038 | 0.034 | 0.030 |
| first singular value (MFA) | 0.047 | 0.073 | 0.067 | 0.061 | 0.055 |
| total norm | 0.125 | 0.220 | 0.189 | 0.176 | 0.161 |

9. **Unscaled, rest is 3 % of the pooled variance.** Between-subject FC variance grows as scan length falls (emotion,
   352 volumes, is 22 %; wm, 810, 8 %; rest, 4,800, 3 %), so the pooled spectrum is set by the noisiest views.
10. **First-singular-value scaling only partly fixes this** (rest 5–7 % of the leading components): rest's spectrum
    is more concentrated, so its first singular value is large relative to its total, and dividing by it keeps rest
    small.
11. **Total-norm scaling gives rest an even total share and more than its share of the leading components**
    (22 % of the top 10, 18 % of the top 256): rest's variance is structured and concentrates early, while the tasks'
    is spread out. With it, rest keeps 0.97 / 0.99 / 1.02 of its own-basis held-out variance at k = 64 / 128 / 256 (vs
    0.92 / 0.94 / 0.97), the tasks change by ≤ 0.03, and rest → task identification in pooled codes improves for
    every task (k = 64 top-1 0.05–0.10 vs 0.04–0.09; average rank 0.75–0.82 vs 0.73–0.81). **D2 now proposes
    total-norm scaling.**

**E1.0d: what the leading pooled components are** (script
[`checks/pooled_pc_interpretation.py`](../../scripts/experiments/joint_flow_matching/checks/pooled_pc_interpretation.py);
total-norm basis, seed 0; held-out subjects for scores; Yeo-7 network blocks):

| PC | Pattern (Yeo-7 blocks) | Corr. with group-mean FC | `rest` share | Held-out rest ↔ task score r | Seed stability |
|---|---|---|---|---|---|
| 1 | stronger within default mode, within dorsal attention and visual, dorsal attention ↔ visual / ventral attention; more negative default mode ↔ dorsal attention / visual / ventral attention | 0.63 | 0.28 | 0.16–0.25 | 1.00 |
| 2 | somatosensory ↔ visual and ↔ dorsal attention vs default mode ↔ somatosensory | 0.21 | 0.17 | −0.03 to 0.14 | 0.99 |
| 3 | visual and dorsal attention ↔ visual vs default mode ↔ visual, dorsal attention ↔ frontoparietal | 0.05 | 0.26 | −0.09 to 0.18 | 0.99 |
| 4 | within somatosensory and within frontoparietal vs frontoparietal ↔ somatosensory | 0.40 | 0.19 | 0.13–0.32 | 0.98 |
| 5 | ventral attention ↔ somatosensory vs somatosensory / ventral attention ↔ visual | 0.29 | 0.31 | 0.11–0.24 | 0.99 |

12. **PC1 is the contrast of the canonical network architecture,** not a global-strength axis: loadings are half
    positive, half negative; a high score means stronger within-network coupling and stronger default mode ↔
    task-positive anticorrelation, an amplified copy of the group-mean FC pattern (r 0.63). Its link to a subject's mean
    FC is weak (r 0.28 in rest, −0.19 to 0.25 in tasks).
13. **The leading axes are reproducible but only weakly trait-like:** loadings are nearly identical across seeds
    (|r| ≥ 0.98), yet a held-out subject's rest score predicts their task score at r ≤ 0.32, and near zero for PCs 2–3 (one task reaches 0.18).
    Most of the leading variance is state- or session-specific, which no translation model can predict; the
    individual fingerprint is spread over many components (finding 4).
14. **Rest is over-represented in the leading components** (17–31 % of PCs 1–5 vs 12.5 % even), as E1.0c predicts.
15. Age correlates at |r| ≤ 0.10. Head motion is not in the loaded metadata, so a nuisance contribution to these axes
    is untested (xcp-d framewise displacement could be added).

### E2 — CCA-latent bidirectional benchmark   (slug: `cca_latent_bench`) · status: done 2026-10-20 · owner: agent:modeling

- **Question:** is a multiview CCA latent of FC (the part of a subject's FC shared across conditions) a better encoding
  for structure–function prediction than per-condition PCA codes, in both directions? FC → SC: does encoding FC
  through the latent, and fusing conditions in it, predict SC better? SC → FC: does SC predict the state-invariant
  functional latent, and does decoding that prediction beat predicting a single FC scan?
- **Why now (before the E1 pilot):** the E1.0 notebook
  (`scripts/notebooks/EDA/FC_encoded_conditions_analysis.ipynb`, seed 0) found that a held-out subject's rest encoding
  identifies their task encoding at top-1 0.46 in a 32-dimensional CCA latent vs 0.06 in 32 pooled-PCA codes, and that
  an 8-condition latent consensus with a ridge readout matches the tuned bench rest FC → SC model (0.147 vs 0.150
  demeaned r) while v3:E2 found no gain from task FC with multi-block regression. E2 tests this formally; its outcome
  sets how E1 is framed (E2.D).

#### E2.A Hypotheses and decision rules

Test split, paired by seed (5 seeds); support rule as E1 (seed-mean Δ above 2 SE, same sign in ≥ 4 of 5 seeds).

| ID | Hypothesis | Comparison | Primary metric |
|---|---|---|---|
| H1 | The CCA encoding is a better FC input than PCA codes | FC → SC: rest CCA encoding − rest PCA codes, same readout | demeaned r |
| H2 | Fusing conditions in the latent helps | FC → SC: 8-condition consensus − rest encoding; reported: 7-task consensus vs rest encoding, learned fusion (concatenated encodings) vs consensus | demeaned r |
| H3 | Latent fusion reaches the tuned bench | FC → SC: consensus − v3:E0 rest (PCA-PLS learnable) | demeaned r (non-inferiority margin 0.01; superiority by the support rule) |
| H4 | SC predicts the state-invariant functional latent | SC → FC: SC → consensus latent → decoded to each FC condition, − v3:E0 SC → FC for that condition and − direct SC → condition ridge (same codes) | demeaned r per condition, averaged over the 8 conditions |
| H5 | Which latent dimensions SC carries | SC → latent R² per dimension and latent identification (predicted vs true consensus encoding) | descriptive |

Average rank and top-1 are reported for every comparison; FC → SC rank metrics are read with v3:C!2.

#### E2.B Design

| Item | Setting |
|---|---|
| Data | v3:E0 cohort and splits (917 Glasser subjects, seeds 0–4), E1's per-seed cache (equal-variance pooled FC basis, SC basis; D2) |
| Latent | MAXVAR multiview CCA over the 8 FC conditions (never SC), fit on train codes: $G^\top G = I$, encoders $A_m$ with $Z_m A_m \approx G$, decoders $D_m$ ($G \to Z_m$). Encoding $g^m_i = z^m_i A_m$; consensus $\bar g_i$ = mean of the 8 encodings (7-task consensus: mean of the tasks) |
| Latent selection (SC-blind) | per seed, $(k_{FC}, d, \lambda) \in$ {128, 256, 512} × {8, 16, 32, 64, 128} × {0.01, 0.1, 1, 10} chosen on val by mean cross-condition identification (average rank of a subject's encoding from condition $a$ among all val subjects' encodings from condition $b$, over the 56 ordered FC pairs). The same latent serves both directions; it never sees SC, so the target in SC → FC is not chosen for being easy to predict |
| FC → SC inputs | rest PCA codes ($k$ ∈ {64, 128, 256, 512}); rest encoding; 7-task consensus; 8-condition consensus; concatenated 8 encodings (learned fusion); each single task's encoding (per-condition rows, next to v3:E0) |
| SC → FC | input SC codes ($k_{SC}$ ∈ {64, 128, 256, 512}); targets: consensus latent (decoded to each condition through $D_m$) and each condition's own encoding (decoded through $D_m$); direct SC → condition-code ridge as the no-latent reference |
| Readouts | ridge (penalty by generalized cross-validation on train) and PLS (`PLSRegression`, components ∈ {2, 4, 8, 16, 32}, chosen on val); target code dimension ($k_{SC}$ for FC → SC) chosen on val |
| Scoring | decoded edges, v3:E0 metrics (`models/eval/metrics.py`) on the 184 test subjects; latent R² and latent identification for SC → FC; secondary FC → SC rows with $(d, \lambda)$ chosen on val SC demeaned r (SC-tuned latent) |
| Ceilings | SC: its own $k_{SC}$ reconstruction; SC → FC through the latent: decoding the subject's **true** consensus encoding to each condition (the most any latent prediction can give) |
| References | v3:E0 `seed_records.csv` (fc2sc, sc2fc) per seed and condition; E1 linear arms L1 / L4 at matched $k$ |

#### E2.C Steps, gates and outputs

| Step | What | Compute | Gate (stop and report if it fails) |
|---|---|---|---|
| E2.0 | Move MAXVAR to `models/architectures/multiview_cca.py` (I2); E1 imports it | CPU | **G0:** E1 `checks/check_infra.py` ALL_OK; E1 linear arms seed 0, k = 64, L4 rows reproduce the stored records exactly |
| E2.1 | Seed-0 pilot of the full grid | 1 CPU job | **G1:** every row finite; rest PCA codes → SC ridge equals E1 L1 at the same $k$ (to 1e-6); selected latent's val cross-condition average rank > the pooled-PCA codes' at the same dimension |
| E2.2 | Seeds 0–4 | CPU array | — |
| E2.3 | Analysis: H1–H5, tables, figures, write-up | CPU | — |

**Deliverables** (`scripts/experiments/cca_latent_bench/results/Glasser/`): `records/seed<S>.csv`;
`tables/summary.md` (one row per hypothesis: Δ ± SE, seeds positive, supported), `fc2sc.csv`, `sc2fc.csv`,
`latent_selection.csv`; figures `fig_fc2sc_inputs.png` (inputs × readouts, bench line), `fig_sc2fc_conditions.png`
(per condition: bench, direct ridge, via latent, latent ceiling), `fig_latent_r2.png` (SC → latent R² per dimension),
`fig_latent_selection.png`; write-up `cca_latent_bench.md`; `experiments_index.md` row; spec result line.

#### E2.D Outcomes and what follows for E1

| Outcome | Reading | E1 |
|---|---|---|
| H1–H3 supported | the cross-condition latent is the better FC encoding for SC | add an arm with the CCA encoding (or consensus) as the A3 / A4 source for FC → SC pairs; frame the flow between the FC latent and SC |
| H4 supported | structure predicts the trait part of function better than a single scan | frame SC → FC as SC → latent → condition, with the condition-specific remainder as the generative part (E3) |
| H1 supported, H2 not | the encoding helps, fusion does not | single-condition CCA encodings as inputs; no multi-source arm |
| None supported | the notebook signal was seed-0 noise or selection | E1 proceeds as planned |

- **Accept:** G0 and G1 pass; 5 seeds complete; H1–H4 decided by the rule, H5 delivered; deliverables written. **Met.**
- **Result:** [`cca_latent_bench.md`](../../scripts/experiments/cca_latent_bench/cca_latent_bench.md) (test, 5 seeds, ridge).
  - **H2, H4 supported; H1 reversed; H3 not supported.** The SC-blind latent ($k_{FC}$ 256, $d$ 32, $\lambda$ 0.1 in
    every seed) loses rest-specific SC information: rest encoding 0.131 vs rest PCA codes 0.141 (H1 −0.010, 0 / 5).
    Fusion helps within the latent (8-condition consensus +0.0075 over the rest encoding, 5 / 5; 7-task consensus
    +0.0045, 4 / 5; learned weighting adds nothing) but stays below the bench rest model (0.139 vs 0.148).
  - **Every task, encoded through the latent, beats its own v3:E0 FC → SC row** (+0.008 WM to +0.050 language, 5 / 5
    each): the shared latent rescues short task scans.
  - **SC → consensus latent → condition beats v3:E0 SC → FC on all 8 conditions** (mean +0.020, 5 / 5) and a direct
    SC → condition ridge (+0.006, 5 / 5); far from the latent ceiling (rest 0.105 vs 0.304).
  - Caveat: FC → SC chose the smallest SC dimension offered (64) in 151 / 170 rows; extending to {8, 16, 32} and a
    hybrid input (rest codes + task consensus) are the first follow-ups. ≈ 0.6 CPU-node-hours (jobs 20484872–74).
- **Budget / autonomy:** D4. CPU only; estimate ≈ 1 CPU-node-hour per seed.
- **Caveats:** v3:C!2 for FC → SC; the latent decoders are linear, so SC → FC via the latent is bounded by the latent
  ceiling; v3:E0 rows used tuned learnable models and per-condition bases, E2 closed-form readouts on pooled codes.

### E3 — Conditional generative translation · status: outline · owner: —

- **Question:** given a subject's source view, what is the distribution of their target view, and is its spread
  subject-specific beyond a linear-Gaussian conditional (MV-5)?
- **Sketch:** the E1 backbone with a noise start (FM-2) or noise along the bridge (FM-4); `rest_S1` / `rest_S2` as two
  targets per subject; scored by held-out negative log-likelihood against `CrossModal_ConditionalGaussian` (MV-5),
  energy distance, calibration of predicted spread against the S1–S2 difference, and identification of single samples
  and sample means (`modeling_directions.md` §3.3).
- **Depends on:** E1 G3 passed and the E1 infrastructure; designed after E1 reports.

## 3. Infrastructure

### I1 — Joint translation infrastructure · status: in progress · owner: agent:modeling

- **Goal:** run E1 (and later E3) from library code, on the v3:E0 cohort, splits and metrics.
- **Changes** (new files only; no file read by queued or running jobs is edited):
  1. `models/architectures/crossmodal_flow_matching.py`: `CodeSpace` (per-view train mean, block scale, pooled FC /
     SC bases, encode / decode, slicing to $k$); `TokenBackbone` (residual MLP with source / target tokens and an
     optional time embedding), used as A2 (no tokens), A3 (residual regression) and A4 (velocity field);
     `integrate` (fixed-step Euler and RK4 in torch).
  2. `scripts/experiments/joint_flow_matching/`:
     - `config.yml`: views, nesting mask, arms, search space, seeds, budget, gate thresholds;
     - `tune.py`: one arm × seed (× pair for A2) per job: builds `HCP_Base` and the code space, runs the Optuna
       search with packed trials, keeps the selected trial's weights, scores them once on test, writes one JSON record;
     - `linear_arms.py`: L1–L4 (CPU, all pairs and seeds) and the identity / ceiling references;
     - `submit.py` / `launch.sh`: GPU arrays (one task per arm × seed), logs in `results/logs/`;
     - `run.py`: records → tables and figures of E1.C;
     - `checks/`: E1.0 pre-checks (done), `check_infra.py` (G0), `synthetic.py` (G1), the G3 pilot check;
     - write-up `joint_flow_matching.md`.
- **Accept (G0):**
  - cohort and per-seed splits identical to v3:E0 (subject IDs and partition lists);
  - decoding the encoded train and test edges reproduces the $k$-dimensional PCA reconstruction (max abs. diff
    < 1e-5), and the pooled basis is orthonormal;
  - nothing fit on val / test (means, scales, bases, A1 fits, models): asserted from the index sets;
  - RK4 and Euler are exact on a constant velocity field, and RK4 matches the closed-form solution of a linear field
    to < 1e-6 at 64 steps;
  - the L3 Gaussian OT map sends the train source code mean and covariance to the target's (relative error < 1e-6);
  - L4: $G$ is orthonormal, and with two views and $\lambda \to 0$ its scores reproduce two-view CCA (canonical
    correlations match `sklearn` CCA to 1e-6);
  - one full A3 trial on real data runs end to end and writes a valid record.

### I2 — Multiview CCA library module · status: done 2026-10-20 · owner: agent:modeling

- **Goal:** one implementation of MAXVAR multiview CCA and the closed-form ridge readout, imported by E1 (L4) and E2.
- **Changes:** new `models/architectures/multiview_cca.py` (`maxvar` fit returning encoders, decoders and shared
  scores; `ridge_gcv`); `scripts/experiments/joint_flow_matching/linear_arms.py` imports them instead of defining
  them (no queued job reads that file).
- **Accept:** E2 gate G0. **Met** (E1 checks ALL_OK; 726 / 726 linear-arm rows reproduced exactly).

## 4. Carried over from v3

Filled when v3 is closed. Expected: v3:E3 (composite-loss benchmark, in progress) as a carry-over; v3:C!2 (FC → SC
identifiability dominated by parcel geometry) as a standing caveat; v3 backlog items.

## 5. Backlog (not scheduled)

- Further directions from `modeling_directions.md` §9 (multiview CCA, transcriptome integration, graph models,
  timeseries tokens), added as items once scoped.
- E1 replication on 4S456Parcels.

## 6. Change log

| Date | Change |
|---|---|
| 2026-10-20 | v4 drafted (v3 still active): E1 joint multidirectional translation (regression vs deterministic flow), E2 outline (generative), I1; D1 (9 views, `rest` as the rest target, sessions as an arm with a nesting mask), D2 and D3 proposed. E1.0 pre-check done (CPU): pooled block-scaled basis keeps more held-out task variance than per-task bases and 92–97 % of rest's at k ≥ 64; per-condition codes are not comparable across views; much of the cross-state fingerprint lies beyond 64 components. |
| 2026-10-20 | D1 decided (user). E1 protocol written out (E1.A hypotheses and support rule, E1.B design, E1.C steps / gates / deliverables, E1.D outcomes); steps renumbered E1.0–E1.6. E1.0b spectra: 658 components per view (5,264 pooled); 95 % of train variance needs 499–583 per view, so $k$ is set from held-out variance: grid {128, 256, 512} (user: lower bound 128). D3 proposal revised (25 GPU-h). |
| 2026-10-20 | E1.0c: rest carries 3 % of the unscaled pooled FC variance (5 % with first-singular-value scaling); total-norm block scaling gives it 18 % of the top 256 components, keeps ≥ 0.97 of its own-basis held-out variance at k ≥ 64 and improves cross-view identification; D2 switched to total-norm scaling. |
| 2026-10-20 | E1.0d: leading pooled PCs are reproducible network-contrast axes (PC1 = amplified canonical architecture, default mode ↔ task-positive segregation), only weakly consistent within subject across conditions (r ≤ 0.32). Linear references written out as L1–L4 (arm A1), with a multiview CCA (MAXVAR) translator as the shared linear model; H1 split into H1a (sharing) and H1b (nonlinear vs linear shared); block-scaling citations in D2. |
| 2026-10-20 | E1.7 (optional identity-aware loss step) added. |
| 2026-10-20 | D2 decided (user): equal-variance (total-norm) block scaling, $k$ ∈ {64, 128, 256, 512}. D3 decided (user): autonomous, 25 GPU-h cap. E1 and I1 in progress (agent:modeling). |
| 2026-10-20 | E2 (CCA-latent bidirectional benchmark) and I2 added; D4 decided (user: autonomous, CPU); generative outline renumbered E3; E1 pilot waits for E2. E1: G0 CPU checks ALL_OK (35), linear arms L1–L4 done for 5 seeds. |
| 2026-10-20 | E1.2 gate G1 failed (synthetic): A3 / A4 one-step R² against the true conditional mean 0.865 / 0.712 (< 0.9); G1b passed. E1 stopped at the gate per D3 (already on hold for E2). Smoke trials: A3 13–44 s, A4 18–33 s per trial on one GPU. |
| 2026-10-20 | E2 done (autonomous, CPU): H2 and H4 supported, H1 reversed, H3 not; tasks through the latent beat every v3:E0 task row for FC → SC; SC → latent → FC beats v3:E0 on all conditions. I2 done. Follow-ups: SC dimension grid edge, hybrid input. E1 still stopped at its synthetic gate. |

Last updated at: 2026-10-20 EDT
