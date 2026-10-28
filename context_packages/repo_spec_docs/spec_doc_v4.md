# Repo Spec v4

**Purpose:** v4 work: modeling directions beyond the linear baselines, drawn from
[`modeling_directions.md`](../modeling/FM+tx_integrate/modeling_directions.md) (the technical reference; model labels such as FM-1 and
MV-5 refer to it). Items: a CCA-latent benchmark that found a shared functional latent across FC conditions (E1);
linear, nonlinear and flow models that predict each condition's latent encoding from SC, including a conditional
flow that generates a subject's encodings (E2); a generative follow-up with more samples per subject (E3, closed:
no gain); full cross-condition translation with one conditional model and SC as a view (E4, outline).
**Status:** active (since 2026-10-26; v3 closed 2026-10-21) · **Started:** 2026-10-20 ·
**Predecessor:** [`spec_doc_v3.md`](spec_doc_v3.md) · **Format:** [`spec_conventions.md`](spec_conventions.md) ·
**Modeling reference:** [`modeling_directions.md`](../modeling/FM+tx_integrate/modeling_directions.md)

**Contents:** [Status](#status) · [1. Conventions](#1-conventions) · [2. Experiments](#2-experiments) ·
[3. Infrastructure](#3-infrastructure) · [4. Carried over from v3](#4-carried-over-from-v3) ·
[5. Backlog](#5-backlog-not-scheduled) · [6. Change log](#6-change-log)

## Status

| ID | Title | Status | Depends on | Owner |
|---|---|---|---|---|
| E1 | CCA-latent bidirectional benchmark: multiview CCA latent of FC as the encoding for FC → SC and the target for SC → FC | done 2026-10-20 | I2, D4 | agent:modeling |
| E2 | SC → functional latent: E2.1 conditional generation of each condition's CCA encoding from SC (linear, nonlinear, linear-Gaussian, flow); E2.2 superseded by E4 | done 2026-10-26 (E2.1; E2.2 superseded by E4 2026-10-28) | I1, E1, D7 | agent:modeling |
| E3 | Generative follow-ups to E2.1: path noise, rest sessions as repeats, FC timeseries windows as more samples per subject | done 2026-10-27 (windows: no gain, `windowed_fc_gain`; path noise dropped; rest sessions and activity flow moved to E4) | E2.1 | agent:modeling |
| E4 | Full cross-condition translation: conditional flows from noise, nonlinear and linear (SC + source condition → any target condition; DeepSets interface, pairwise first), against identity, block-PCA ridge (± SC), 9-view multiview CCA with SC and linear-Gaussian baselines | planned | E2.1, E2.0, I1, E3, D8 | agent:modeling |
| I1 | Joint translation infrastructure: pooled FC basis, (subject, source, target) data module, token model, integrator, runner | in progress | — | agent:modeling |
| I2 | Multiview CCA library module (MAXVAR fit / encode / decode), shared by E2 and E1 | done 2026-10-20 | — | agent:modeling |
| D1 | Views and the true rest target | decided 2026-10-20 | — | user |
| D2 | Code space: pooled FC basis with equal-variance (total-norm) block scaling, SC basis, $k$ grid | decided 2026-10-20 | E2.0 | user |
| D3 | E2 compute envelope and autonomous execution | decided 2026-10-20 | — | user |
| D4 | E1 autonomous execution | decided 2026-10-20 | — | user |
| D5 | E2 reframed around the consensus latent; training fix for the synthetic gate; identity-loss arm | decided 2026-10-21 | E1 | user |
| D6 | Spec and repository housekeeping | decided 2026-10-21; v4 active 2026-10-26 | — | user |
| D7 | E2.1 as conditional generation of each condition's encoding (8 targets per subject) | decided 2026-10-26 | E1 | user |
| D8 | E4 design (nonlinear and linear flows from noise, DeepSets interface trained pairwise first, baselines reviewed first; activity flow deferred) and execution | decided 2026-10-28 (autonomous through E4.4, 15 GPU-h) | — | user |
| D10 | E4 code size fixed by rule after E4.0c; optimisability pre-checks before the flow search | decided 2026-10-28 | E4.0c | user |
| D9 | Validation selection on average rank for E4.0c onward | decided 2026-10-28 | — | user |
| C!1 | (from v3:C!2) Subject-specific SC is dominated by parcel size, which FC cannot see | standing caveat | — | — |

---

## 1. Conventions

Inherited from v3 §1 unchanged: experiment homes, reported metrics (test Pearson r, demeaned r, average rank, top-1),
seeds 0–4, val selection with test reported, `sbatch` compute, PNG figures; v2:D1–D6 and v3:D4 stay in force. Glasser
first; 4S456Parcels is a later replication with the same configs. **Benchmark reference for SC pairs:** v3:E0
(`scripts/experiments/task_bidirectional_bench/results/Glasser/{fc2sc,sc2fc}/tables/`), which uses the same matched
917-subject cohort and per-seed splits, so E2 differences are paired by seed and subject.

**Notation (from `modeling_directions.md` §2):** views $m \in \mathcal{M}$; code $z^m_i \in \mathbb{R}^k$ of subject
$i$ in view $m$; view embedding $e(m)$; flow time $\tau \in [0, 1]$.

### Decisions (`D`)

| ID | Decision | Rationale |
|---|---|---|
| D1 | **Views (user 2026-10-20):** main model on 9 views, $\mathcal{M}$ = {SC, `rest`, 7 tasks} (72 ordered pairs). `rest` (4 runs, xcp-d combined relmat) is the true rest view and the rest target, as in every earlier result. For E2.2 and E3, a sessions variant adds `rest_S1` and `rest_S2` (per-session mean of the LR and RL run matrices) with a nesting mask: pairs between `rest` and either session are excluded from training and evaluation, because a session is half of `rest`'s data. S1 ↔ S2 is then the in-model test–retest translation. | Continuity with v3:E0 / v3:E2 and v2:E2.2; the mask prevents near-identity leakage; the sessions add a ceiling and the two-sample targets E3 needs. |
| D2 | **Decided (user 2026-10-20):** one FC basis from a PCA of the stacked train-split FC of all FC views, each block centered on its own train mean (condition means stored and added back at decoding by the target token) and divided by its total (Frobenius) norm, so every view carries the same total variance (standard multiblock / consensus PCA block scaling, Westerhuis et al. 1998; the first-singular-value alternative is multiple factor analysis, Escofier & Pagès; E2.0c: total-norm scaling is better for rest and for cross-view identification); SC keeps its own train PCA (also divided by its total norm); codes share one dimension $k \in$ {64, 128, 256, 512}, bounded above by the SC rank (658; E2.0b). (The first E2 design's basis ablations, per-condition bases and an SC basis rotated onto the FC basis, apply to Part 2 only.) | The bridge interpolates codes directly, so all views need comparable coordinates; per-condition centering keeps condition offsets out of the flow, so the model works on subject deviations (the quantity demeaned metrics score). |
| D3 | **Decided (user 2026-10-20; revised by D5 / D7: E2.1 cap 10 GPU-h, E2.2 needs the user's go):** E2 runs autonomously; cap 25 GPU-h (estimate 15–18); per-job time limit = pilot time per trial × 1.5. Stop and report on a failed gate (G0, G1, G3), > 20 % task errors after one retry, a log / config mismatch, a code change outside I1's files, or the cap. Commits local (no push). Relaxing a threshold needs the user. | Small models in code space; explicit exits as v3:D1–D3. |
| D4 | **Decided (user 2026-10-20):** E1 runs autonomously through E1.0–E1.3, CPU only, cap 15 CPU-node-hours. Stop and report on a failed gate (G0, G1), > 20 % task errors after one retry, a log / config mismatch, a code change outside I2 and E1's folder (plus the E2 import switch), or the cap. Commits local (no push). Relaxing a threshold needs the user. The first-design E2 pilot (A3 / A4) waited for E1's result. | E1 is closed form and cheap; it decides E2's framing, so it runs first. |
| D5 | **Decided (user 2026-10-21; target refined by D7):** E2 is reframed as SC ↔ consensus latent (Part 1, the core), with the multi-condition FC → FC model kept as an optional Part 2. The failed synthetic gate is fixed in training (flow time drawn at 0 with probability $p_0$, longer training), not by lowering the threshold. An identity-aware loss (demeaned corr-eye on the latent) is an arm of Part 1. E3 stays an outline until E2 reports. D3 is revised: Part 1 autonomous, cap 10 GPU-h; Part 2 needs the user's go. | E1: the latent is where SC → FC gains and it is a smaller, cleaner problem; the gate should hold. |
| D6 | **Decided (user 2026-10-21):** v3 is closed (and v4 made active) once v3:E3's queued jobs finish; `scripts/notebooks/EDA/FC_encoded_conditions_analysis.ipynb` is committed as is; commits stay local until the user reviews them. | Avoid editing a spec with running work; keep the record reviewable. |
| D7 | **Decided (user 2026-10-26):** E2.1 trains on every condition's encoding in E1's latent (8 targets per subject, condition token) instead of the consensus alone, so the problem keeps within-subject multiplicity; the main model is a conditional flow from noise with token dropout (generates with or without a condition); a linear-Gaussian model (ridge mean, one shared residual covariance) is the distributional reference; the ridge-start bridge and the nonlinear corrector stay as deterministic references. Primary evaluation: energy distance to each subject's observed encodings, flow vs linear-Gaussian (H3). The first pass is a baseline signal of whether the models learn, linearly and nonlinearly. | A consensus-only target gives one sample per subject, too close to a single paired regression; the eight encodings are observed samples of each subject's distribution. |
| D8 | **Decided (user 2026-10-28):** E4's flows are trained from noise (no linear anchor, no CCA in the model; the weak linear signal stays in the training target rather than being residualised out), in two forms, nonlinear and linear, against the closed-form linear-Gaussian to separate nonlinearity from fitting effects; DeepSets conditioning interface ($\phi$ modular, linear first) trained pairwise first, multi-source as a retrain; baselines B0–B3 run and reviewed before the flows (E4.1); activity flow and ablations are follow-ups. Autonomous from E4.0c through E4.4 (user 2026-10-28), cap 15 GPU-h; stop and report on a failed gate (G1, G2, optimisability checks), > 20 % task errors after one retry, the cap, or a code change outside E4's folder (beyond importing E2.1's model classes); commits local. | Optimisation capacity first: the pairwise-trained DeepSets form equals a pairwise model and extends to multi-source without a new architecture. |
| D9 | **Decided (user 2026-10-28):** from E4.0c onward, model and hyperparameter selection on validation uses **average rank** (identification), not demeaned r: ridge penalty and SC code size, CCA $(d, \lambda)$ (grid extended to $d$ = 128, 256), flow hyperparameters and sampling temperature. Demeaned r stays reported. E4.0 / k = 512 results (selected on demeaned r / GCV) are kept as they are. | Individual FC deviations agree only weakly across conditions (r 0.06–0.19) yet identify subjects (average rank 0.88–0.99); MSE-like selection shrinks predictions toward the mean and discards that individual signal (E4.0: ridge and CCA trail copying on identification, and do not use components 257–512). |
| D10 | **Decided (user 2026-10-28):** the flows' code size is fixed after E4.0c by this rule, on validation data only. (1) E4.0c refits B1 and B2 with D9 selection at $k$ = 256, 512 and full rank (per-condition bases). (2) Score per $k$: mean val average rank over the 56 single-source cells, best of B1 / B2. (3) Choose the smallest $k$ in {256, 512} within 0.005 of the better of the two. (4) Full rank is diagnostic: if it beats the chosen size by more than 0.01, flag it (signal beyond the shared basis) and proceed. (5) The same $k$ for source and target codes; the SC code size stays in the flow search. Before the search, **optimisability checks** on FM-nl and FM-lin: loss decreases and gradients are finite; each overfits a small training subset; FM-lin on synthetic linear-Gaussian data recovers the closed-form linear-Gaussian (energy within 2 SE); then G1. | Fix the representation before the flows, from evidence the flows do not see; make sure a flow failure is not an optimisation failure. |

## 2. Experiments

### E1 — CCA-latent bidirectional benchmark   (slug: `cca_latent_bench`) · status: done 2026-10-20 · owner: agent:modeling

- **Question:** is a multiview CCA latent of FC (the part of a subject's FC shared across conditions) a better encoding
  for structure–function prediction than per-condition PCA codes, in both directions? FC → SC: does encoding FC
  through the latent, and fusing conditions in it, predict SC better? SC → FC: does SC predict the state-invariant
  functional latent, and does decoding that prediction beat predicting a single FC scan?
- **Why now (before the E2 pilot):** the E2.0 notebook
  (`scripts/notebooks/EDA/FC_encoded_conditions_analysis.ipynb`, seed 0) found that a held-out subject's rest encoding
  identifies their task encoding at top-1 0.46 in a 32-dimensional CCA latent vs 0.06 in 32 pooled-PCA codes, and that
  an 8-condition latent consensus with a ridge readout matches the tuned bench rest FC → SC model (0.147 vs 0.150
  demeaned r) while v3:E2 found no gain from task FC with multi-block regression. E1 tests this formally; its outcome
  sets how E2 is framed (E1.D).

#### E1.A Hypotheses and decision rules

Test split, paired by seed (5 seeds); support rule as everywhere in v4 (seed-mean Δ above 2 SE, same sign in ≥ 4 of 5 seeds).

| ID | Hypothesis | Comparison | Primary metric |
|---|---|---|---|
| H1 | The CCA encoding is a better FC input than PCA codes | FC → SC: rest CCA encoding − rest PCA codes, same readout | demeaned r |
| H2 | Fusing conditions in the latent helps | FC → SC: 8-condition consensus − rest encoding; reported: 7-task consensus vs rest encoding, learned fusion (concatenated encodings) vs consensus | demeaned r |
| H3 | Latent fusion reaches the tuned bench | FC → SC: consensus − v3:E0 rest (PCA-PLS learnable) | demeaned r (non-inferiority margin 0.01; superiority by the support rule) |
| H4 | SC predicts the state-invariant functional latent | SC → FC: SC → consensus latent → decoded to each FC condition, − v3:E0 SC → FC for that condition and − direct SC → condition ridge (same codes) | demeaned r per condition, averaged over the 8 conditions |
| H5 | Which latent dimensions SC carries | SC → latent R² per dimension and latent identification (predicted vs true consensus encoding) | descriptive |

Average rank and top-1 are reported for every comparison; FC → SC rank metrics are read with v3:C!2.

#### E1.B Design

| Item | Setting |
|---|---|
| Data | v3:E0 cohort and splits (917 Glasser subjects, seeds 0–4), E2's per-seed cache (equal-variance pooled FC basis, SC basis; D2) |
| Latent | MAXVAR multiview CCA over the 8 FC conditions (never SC), fit on train codes: $G^\top G = I$, encoders $A_m$ with $Z_m A_m \approx G$, decoders $D_m$ ($G \to Z_m$). Encoding $g^m_i = z^m_i A_m$; consensus $\bar g_i$ = mean of the 8 encodings (7-task consensus: mean of the tasks) |
| Latent selection (SC-blind) | per seed, $(k_{FC}, d, \lambda) \in$ {128, 256, 512} × {8, 16, 32, 64, 128} × {0.01, 0.1, 1, 10} chosen on val by mean cross-condition identification (average rank of a subject's encoding from condition $a$ among all val subjects' encodings from condition $b$, over the 56 ordered FC pairs). The same latent serves both directions; it never sees SC, so the target in SC → FC is not chosen for being easy to predict |
| FC → SC inputs | rest PCA codes ($k$ ∈ {64, 128, 256, 512}); rest encoding; 7-task consensus; 8-condition consensus; concatenated 8 encodings (learned fusion); each single task's encoding (per-condition rows, next to v3:E0) |
| SC → FC | input SC codes ($k_{SC}$ ∈ {64, 128, 256, 512}); targets: consensus latent (decoded to each condition through $D_m$) and each condition's own encoding (decoded through $D_m$); direct SC → condition-code ridge as the no-latent reference |
| Readouts | ridge (penalty by generalized cross-validation on train) and PLS (`PLSRegression`, components ∈ {2, 4, 8, 16, 32}, chosen on val); target code dimension ($k_{SC}$ for FC → SC) chosen on val |
| Scoring | decoded edges, v3:E0 metrics (`models/eval/metrics.py`) on the 184 test subjects; latent R² and latent identification for SC → FC; secondary FC → SC rows with $(d, \lambda)$ chosen on val SC demeaned r (SC-tuned latent) |
| Ceilings | SC: its own $k_{SC}$ reconstruction; SC → FC through the latent: decoding the subject's **true** consensus encoding to each condition (the most any latent prediction can give) |
| References | v3:E0 `seed_records.csv` (fc2sc, sc2fc) per seed and condition; E2 linear arms L1 / L4 at matched $k$ |

#### E1.C Steps, gates and outputs

| Step | What | Compute | Gate (stop and report if it fails) |
|---|---|---|---|
| E1.0 | Move MAXVAR to `models/architectures/multiview_cca.py` (I2); E2 imports it | CPU | **G0:** E2 `checks/check_infra.py` ALL_OK; E2 linear arms seed 0, k = 64, L4 rows reproduce the stored records exactly |
| E1.1 | Seed-0 pilot of the full grid | 1 CPU job | **G1:** every row finite; rest PCA codes → SC ridge equals E2 L1 at the same $k$ (to 1e-6); selected latent's val cross-condition average rank > the pooled-PCA codes' at the same dimension |
| E1.2 | Seeds 0–4 | CPU array | — |
| E1.3 | Analysis: H1–H5, tables, figures, write-up | CPU | — |

**Deliverables** (`scripts/experiments/cca_latent_bench/`, Glasser; layout of 2026-10-27): `results/seed<S>.csv`;
`tables/summary.md` (one row per hypothesis: Δ ± SE, seeds positive, supported), `fc2sc.csv`, `sc2fc.csv`,
`latent_selection.csv`; figures `fig_fc2sc_inputs.png` (inputs × readouts, bench line), `fig_sc2fc_conditions.png`
(per condition: bench, direct ridge, via latent, latent ceiling), `fig_latent_r2.png` (SC → latent R² per dimension),
`fig_latent_selection.png`; write-up `cca_latent_bench.md`; `experiments_index.md` row; spec result line.

#### E1.D Outcomes and what follows for E2

| Outcome | Reading | E2 |
|---|---|---|
| H1–H3 supported | the cross-condition latent is the better FC encoding for SC | add an arm with the CCA encoding (or consensus) as the A3 / A4 source for FC → SC pairs; frame the flow between the FC latent and SC |
| H4 supported | structure predicts the trait part of function better than a single scan | frame SC → FC as SC → latent → condition, with the condition-specific remainder as the generative part (E3) |
| H1 supported, H2 not | the encoding helps, fusion does not | single-condition CCA encodings as inputs; no multi-source arm |
| None supported | the notebook signal was seed-0 noise or selection | E2 proceeds as planned |

- **Accept:** G0 and G1 pass; 5 seeds complete; H1–H4 decided by the rule, H5 delivered; deliverables written. **Met.**
- **Potential next steps (noted, not scheduled; user 2026-10-21):** SC code sizes {8, 16, 32} (FC → SC chose the
  smallest offered, 64); a combined FC → SC input (rest PCA codes + task consensus); a leave-the-target-out ceiling for
  SC → FC.
- **Result:** [`cca_latent_bench.md`](../../scripts/experiments/cca_latent_bench/cca_latent_bench.md) (test, 5 seeds, ridge).
  - **H2, H4 supported; H1 reversed; H3 not supported.** The SC-blind latent ($k_{FC}$ 256, $d$ 32, $\lambda$ 0.1 in
    every seed) loses rest-specific SC information: rest encoding 0.131 vs rest PCA codes 0.141 (H1 −0.010, 0 / 5).
    Fusion helps within the latent (8-condition consensus +0.0075 over the rest encoding, 5 / 5; 7-task consensus
    +0.0045, 4 / 5; learned weighting adds nothing) but stays below the bench rest model (0.139 vs 0.148).
  - **Every task, encoded through the latent, beats its own v3:E0 FC → SC row** (+0.008 WM to +0.050 language, 5 / 5
    each): the shared latent rescues short task scans.
  - **SC → consensus latent → condition beats v3:E0 SC → FC on all 8 conditions** (mean +0.020, 5 / 5) and a direct
    SC → condition ridge (+0.006, 5 / 5); far from the latent ceiling (rest 0.105 vs 0.304).
  - Caveat: FC → SC chose the smallest SC dimension offered (64) in 151 / 170 rows (potential next steps below).
    ≈ 0.6 CPU-node-hours (jobs 20484872–74).
- **Budget / autonomy:** D4. CPU only; estimate ≈ 1 CPU-node-hour per seed.
- **Caveats:** v3:C!2 for FC → SC; the latent decoders are linear, so SC → FC via the latent is bounded by the latent
  ceiling; v3:E0 rows used tuned learnable models and per-condition bases, E1 closed-form readouts on pooled codes.

### E2 — SC → functional latent: linear, nonlinear and flow models   (slug: `joint_flow_matching`) · status: done 2026-10-26 · owner: agent:modeling

**In one paragraph.** E1 found that a shared functional latent, learned by multiview CCA across rest and the seven
tasks without SC, is a better target for structure than any single FC scan. Every subject has **eight encodings** in
that latent, one per condition; their mean is the subject's consensus. E2 asks whether SC can predict these
encodings, first as point predictions (linear and nonlinear) and then as a **distribution**: given a subject's SC,
can a conditional flow generate the cloud of that subject's condition encodings, with the right location, spread and
individuality? Because each subject contributes eight targets instead of one, the problem keeps the within-subject
multiplicity that a consensus-only setup would discard. The first pass is a baseline signal: can these models learn
the mapping at all, linearly and nonlinearly, at 659 training subjects?

**History.** The first E2 design (2026-10-20) trained one network over all 72 view pairs in block PCA codes. After E1
it was reframed around the consensus latent (2026-10-21, D5), and then (2026-10-26, D7) around each condition's
encoding, so the flow is a conditional generative model with several observed samples per subject rather than a
single paired map. The 72-pair FC → FC model is kept as optional E2.2. Work done so far carries over (end of this
section).

#### E2.1 — Conditional generation of each condition's latent encoding from SC (the core)

**Data.** E1's SC-blind latent, refit per seed exactly as E1 chose it (block FC PCA, 256 codes per condition; 32
dimensions; λ = 0.1). For held-out and training subjects alike, each condition's encoding $g^m_i$ is that condition's
FC codes passed through its fixed CCA encoder; nothing about the latent is trained in E2. Because every condition was
centered on its own group mean before encoding, a subject's eight encodings scatter around their consensus by the
subject's condition-specific variation plus noise, not by the shared task differences. Input: SC's own PCA codes
(64–512, chosen on val). Decoding to condition $m$ uses that condition's fixed CCA decoder.

**Models.** All except the Gaussian are point models unless marked; the condition enters as a token.

| Model | What it predicts | Role |
|---|---|---|
| Ridge → consensus | the consensus from SC (E1's model) | linear point reference for the consensus; must reproduce E1 exactly (gate G0b) |
| Ridge → each condition | each condition's encoding from SC (one ridge per condition) | linear point reference per condition |
| Linear-Gaussian | ridge mean per condition, with one residual covariance shared by all subjects (the spread does not depend on SC) | **linear distributional reference** (MV-5 style); the bar the flow must clear |
| Nonlinear corrector | ridge prediction + a learned correction from SC and the condition token, trained on all 8 targets per subject | nonlinear point model |
| Corrector + identity loss | the same, trained with squared error + demeaned corr-eye (`compute_correye_dm_loss`, imported) | identity-aware nonlinear point model |
| Ridge-start bridge | a deterministic flow from the ridge prediction to the encoding, with the condition token | deterministic flow reference |
| **Conditional flow (residual)** | **samples** of the residual around the ridge mean, $g^m - \mathrm{ridge}_m(\mathrm{SC})$, starting from noise, conditioned on SC and on the condition token, with the ridge mean added back; the token is dropped for 20 % of training examples. The subject's cloud is built as the Gaussian reference's: each sample picks a condition and draws that condition's residual | **main model** (residual form adopted 2026-10-26 after the first gate run) |

The conditional flow therefore answers two questions: with a token, $p(g^m \mid \mathrm{SC}, m)$, comparable to the
point models; without one, $p(g \mid \mathrm{SC})$, the distribution of the subject's condition encodings, for which
every subject has eight observed samples. All networks are the `TokenBackbone` (`models/architectures/
crossmodal_flow_matching.py`) with a new output-size argument (input 64–512 SC codes, output 32); the flows add the
flow-time embedding. Small networks (width 128–512, 1–3 residual blocks); Optuna, 32 trials per model and seed,
selected on val (point models on mean decoded demeaned r; flows on val energy distance, below).

**After the first E2.1 gate run (2026-10-26, job 21010434).** The conditional flow from noise recovered each
simulated subject's spread (r 0.70) but placed its clouds less precisely than ridge (sample-mean R² 0.95 vs 0.99), so
the Gaussian reference, centred on ridge means, won on energy distance; the simulation's between-condition scatter
also swamped the spread. Fixes (user 2026-10-26; thresholds unchanged): the flow models the residual around the
ridge mean (above), early-stops on a smoothed val loss (patience 200), and the simulation is made spread-dominated.
The original 9-view gate (joint MLP 0.865, flow one-step 0.818 after the fix) belongs to E2.2.

**Training fix from the first failed synthetic gate** (applies to the ridge-start bridge): flow time is set to 0 with
probability $p_0 \in$ {0, 0.25, 0.5} and drawn uniformly otherwise, and training runs longer (up to 1,000 epochs,
patience 100). The conditional flow from noise is trained with the standard straight path from noise to target.

**Evaluation** (184 test subjects; 64 samples per subject for the flows):

| ID | Question | Metric | Compared with | Supported if |
|---|---|---|---|---|
| H1 | Can SC → encodings be learned nonlinearly beyond the linear map? | decoded demeaned r per condition, mean of 8 | corrector − ridge per condition | support rule |
| H2 | Does the identity loss make predictions more individual? | decoded average rank, mean of 8; demeaned r may not drop by more than 0.005 | corrector + identity loss − corrector | support rule and margin |
| **H3 (primary)** | Does the flow generate each subject's cloud of encodings better than a linear-Gaussian model? | energy distance between a subject's generated samples (no token) and their 8 observed encodings, averaged over subjects (lower is better) | linear-Gaussian − flow | support rule |
| H4 | Are the generated clouds subject-specific? | set-to-set identification: 184 × 184 energy distances (generated cloud of subject $i$ vs observed encodings of subject $j$); top-1 and average rank | flow − linear-Gaussian; flow trained with SC shuffled across subjects must sit at chance | support rule |
| H5 | Does the predicted spread track each subject's real spread? | correlation across subjects between predicted and observed within-subject variance (the Gaussian has none by construction); ratio of predicted to observed variance; coverage of the observed encodings | flow vs 0 | correlation above 0 by the support rule |
| H6 | Does the flow keep point accuracy? | sample mean (with token) decoded per condition, demeaned r, mean of 8 | flow − corrector; non-inferiority margin 0.005 | margin |
| H7 | (secondary) Does a nonlinear model improve latent → SC? | SC demeaned r; inputs: consensus and each task's encoding | corrector − ridge | support rule |

**Support rule** (as everywhere in v4): the seed-mean difference is above 2 standard errors and has the same sign in
at least 4 of 5 seeds. Every result is also reported per condition, next to the v3:E0 rows, E1's ridge and the latent
ceiling. What a positive first pass looks like: the corrector at or above ridge (H1), and the flow beating the
Gaussian on energy distance and set identification (H3, H4) while keeping point accuracy (H6).

**Caveats.** The eight encodings are not independent draws: each comes from a different condition with its own
systematic subject × condition effects, so "spread" mixes task-specific individuality with noise; the token mode
separates them. Rest has a true repeat (two sessions), used as a calibration check for one condition. The latent
ceiling is mildly optimistic (the consensus includes the target condition).

**Steps, gates and outputs.**

| Step | What | Compute | Gate (stop and report if it fails) |
|---|---|---|---|
| E2.1a | Latent-target cache per seed (E1's latent, all subjects' 8 encodings, decoders); the runner | CPU | **G0b:** ridge → consensus reproduces E1's SC → FC rows to 1e-6  **passed 2026-10-26** (max diff ≤ 1e-16, 5 seeds) |
| E2.1b | Synthetic gate shaped like E2.1: simulated subjects whose 8 encodings share most of their mean structure and scatter with a strongly input-dependent spread (the original 9-view gate moved to E2.2) | 1 GPU, < 1 h | **G1:** point models' R² ≥ 0.9 against the true conditional mean; shuffled pairing collapses (demeaned r ≤ 0.02); on the distributional simulation the flow recovers the per-subject spread (correlation ≥ 0.5) and beats the Gaussian on energy distance |
| E2.1c | Pilot, seed 0: all models | 1 GPU | **G3:** corrector val within 0.005 of ridge or better; flow sample means correlate ≥ 0.9 with the corrector's predictions; metrics finite; selected trials stop before the epoch cap |
| E2.1d | Seeds 1–4 | GPU array | — |
| E2.1e | Analysis, figures, write-up | CPU | — |

**E2.1 result (2026-10-26, 5 seeds, test;** [`joint_flow_matching.md`](../../scripts/experiments/joint_flow_matching/joint_flow_matching.md)**):**
- **SC → latent is linear here.** H1 holds statistically but is negligible (corrector − ridge +0.0003 demeaned r,
  4 / 5 seeds); the identity loss (H2) and both flows add nothing to point accuracy (H6: residual flow −0.002,
  non-inferior).
- **SC predicts where a subject sits, not how variable they are.** H3 (primary) is not supported: the linear-Gaussian
  matches the residual flow's clouds slightly better (energy 0.0462 vs 0.0469, 5 / 5 seeds). Set identification
  0.84–0.85 for both vs 0.50 for the full null (H4 not supported); predicted spread does not track a subject's spread
  across conditions or between rest sessions (H5 not supported; the spread-only null matches the flow).
- **The only nonlinear gain is latent → SC** (H7, +0.004, 5 / 5; consensus → SC 0.142).
- Reading: for SC → FC the limit is information, not model class; next are E2.2 (FC → FC, SC as an added view) and
  more samples per subject (E3). ≈ 2 GPU-h.

**E2.1 addendum — can a flow learn SC → latent without a ridge anchor?** (user 2026-10-26; status: done 2026-10-26). Every
real-data flow in E2.1 was anchored on ridge (residual flow, ridge-start flow), and E2.1's conclusions were drawn in
linear PCA / CCA coordinates. The addendum fits two **un-anchored** flows, from noise straight to each condition's
encoding, conditioned on SC and the condition token (20 % token dropout): (a) a **linear velocity field**, linear in
the state, the SC codes and the condition with flow-time-varying coefficients (the form of the exact flow for
linear-Gaussian data, so it should match the linear-Gaussian reference if training works); (b) the **nonlinear** MLP
velocity field. Each gets a full null (SC shuffled), a synthetic check (reported, not a gate), and 5 seeds, scored
like E2.1 (point demeaned r against ridge; energy distance, set identification and spread against the
linear-Gaussian). The question: does a flow-matching setup learn at all here, linear or nonlinear, without being given
the linear mean? Output: `results/Glasser/e21/tables/addendum.md`, `figures/fig_addendum_free_flows.png`. Jobs:
synthetic `21033481`; runs `21033482`–`87`; analysis `21033489`. Representation caveat recorded for future flow work:
PCA / CCA coordinates are linear and, for CCA, chosen for linear cross-view agreement, so they tilt toward linear
answers; later flow avenues should not anchor on ridge and should consider learned nonlinear encoders and more
samples per subject. **Result:** flow matching does learn SC → latent without an anchor (point signal 94 % of ridge with a
linear velocity, 91 % nonlinear; cloud identification 0.84 / 0.82 vs 0.50 null), but less efficiently than the closed
form (both trail ridge and the linear-Gaussian in every seed, by 0.003–0.005 demeaned r and 0.002 energy). On the
synthetic data the nonlinear flow captures input-dependent spread (energy 0.030 vs 0.511); on real data it does not
beat the linear flow, so the real relation is linear-Gaussian in these coordinates at this sample size.

**Deliverables** (`scripts/experiments/joint_flow_matching/results/Glasser/`): per-seed records; `summary.md` (one
row per hypothesis: difference ± SE, seeds positive, supported); figures with input → output labels: (1) SC → each
condition, decoded, for every model, with v3:E0 and the ceiling; (2) energy distance and set identification for the
flow, the Gaussian and the shuffled null; (3) predicted vs observed spread across subjects; (4) a few held-out
subjects' generated clouds next to their 8 observed encodings in the first latent dimensions (the E1 viewer's layout);
(5) latent → SC. Write-up `joint_flow_matching.md`, experiments index row, spec result line.

**What each outcome means.**

| Outcome | Reading | Next |
|---|---|---|
| H1 supported | SC → latent has nonlinear structure | nonlinear models carry forward |
| H1 not supported | linear is enough at 659 subjects | keep ridge as the mean model; the flow's value is in the spread |
| H3 and H4 supported | the flow learns subject-specific distributions | E3: path noise, sessions as repeats, timeseries windows as more samples |
| H3 supported, H5 not | the flow gets the shape but not who is more variable | the spread is population-level; more samples per subject needed (E3) |
| H3 not supported | at this sample size a Gaussian with one shared covariance is as good | more samples per subject before more model (sessions, timeseries windows) |

- **Budget / autonomy:** D3 (revised by D5 and D7): E2.1 runs autonomously, cap 10 GPU-h (estimate ≤ 3; trials take
  seconds); E2.2 needs the user's go.

#### E2.2 — Cross-condition FC translation as a generative model, with SC as an added view · status: superseded (by E4, 2026-10-28)

Superseded by **E4** (user, 2026-10-28). The 2026-10-26 draft (single / multi-source translation, corrector and residual flow
around the best linear map, H1–H6) is in git history (`bad618a`).

#### E2 work done so far

- Pre-checks E2.0a–d (results below): the block PCA with equal-variance scaling, its spectrum and components.
- Linear references for all 72 view pairs, 5 seeds × 4 code sizes (`results/Glasser/records/linear/`).
- Gate G0 (code space, decoding, integrators, transport map, CCA): 35 / 35 checks passed.
- First synthetic gate (job 20459033): failed (MLP 0.865, flow one-step 0.712 against 0.9; ridge 0.957); shuffled
  pairing collapsed as required. Smoke trials on real data: 13–44 s each. Addressed by the training fix above.

#### E2.0 results — pre-checks (2026-10-20)

Scripts: [`checks/pooled_basis_variance.py`](../../scripts/experiments/joint_flow_matching/checks/pooled_basis_variance.py),
[`checks/basis_spectrum.py`](../../scripts/experiments/joint_flow_matching/checks/basis_spectrum.py) (CPU, seeds 0–1,
Glasser, 917-subject cohort; CSVs in `checks/pooled_basis_variance/`). Held-out = val + test subjects (258), projected
with train means and bases.

**E2.0a: variance kept on held-out subjects, pooled basis / own basis** (pooled basis over `rest` + 7 tasks with
first-singular-value block scaling, superseded by total-norm scaling in E2.0c; mean of 2 seeds):

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
   ≤ 0.03. Pooling the two sessions instead of `rest` (sessions variant) gives rest 0.95–1.00 at k ≥ 64.
3. **Per-condition codes are not comparable across views:** identifying a held-out subject's task FC from their rest
   FC by correlating codes is at chance in own-basis codes (k = 64: top-1 0.000–0.006, average rank 0.45–0.56), and
   well above chance in pooled codes (top-1 0.03–0.09, average rank 0.72–0.81; S1 → S2 0.53 / 0.94). The paired bridge
   (A4) therefore needs the pooled basis; the per-condition ablation (E2.2) applies to A3 only.
4. **Much of the cross-state fingerprint lies beyond 64 components:** in edge space the same identification reaches
   top-1 0.53–0.82 (average rank 0.97–0.99; S1 → S2 0.95). Two consequences: (a) the k-dimensional code bounds what any
   E2 model can recover, so a per-view reconstruction ceiling is reported next to every result; (b) for FC → FC pairs
   the identity reference (copy the subject's source deviation) is a strong bar.

**E2.0b: spectra** (mean of 2 seeds; train variance share, and variance captured on held-out subjects):

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
   not used in E2.
8. **$k$ grid {64, 128, 256, 512} (user):** 512 keeps ≥ 0.94 of each own basis's capturable
   held-out variance and stays below the SC rank.

**E2.0c: how much each view contributes to the pooled spectrum** (script
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

**E2.0d: what the leading pooled components are** (script
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
14. **Rest is over-represented in the leading components** (17–31 % of PCs 1–5 vs 12.5 % even), as E2.0c predicts.
15. Age correlates at |r| ≤ 0.10. Head motion is not in the loaded metadata, so a nuisance contribution to these axes
    is untested (xcp-d framewise displacement could be added).

### E3 — Generative follow-ups · status: done 2026-10-27 · owner: agent:modeling

- **Question:** once E2.1 shows whether a conditional flow learns subject-specific distributions from eight condition
  encodings, does it improve with a stochastic path (noise along the bridge) and with more samples per subject?
- **Sketch (2026-10-26):** (1) path noise on the ridge-start bridge (FM-4); (2) the two rest sessions as true repeats
  (calibration of within-condition spread); (3) FC timeseries windows as many samples per subject (AR-6).
- **Result (closed 2026-10-27, user):**
  - (3) **No gain** (`scripts/experiments/windowed_fc_gain/`, commits `0dad2c0`, `2fc963d`, `a47955f`). FC from halves /
    quarters of every run (both phase-encoding directions) is the subject's target plus within-run noise, not an
    independent retest (gates 1–6). Retraining E2.1's ridge, linear-Gaussian and flows (residual, linear and nonlinear
    from noise; CCA and block PCA) on full + part samples, with a sample-type token, or on parts only: no model gains;
    the residual flow gets worse in every arm (average rank −0.007 to −0.024); the linear baselines stay best (4.5 GPU-h).
    Built on the timeseries cache (`Conn2Conn_data/timeseries`, `data/timeseries.py`).
  - (1) **Dropped:** E2.1's ridge-start bridge did not beat ridge (demeaned r 0.0458 vs 0.0490); a stochastic path is
    not where the signal is.
  - (2) **Moved to E4:** the rest session 1 ↔ 2 pair is E4's test–retest ceiling and spread calibration check.
  - The activity-flow baseline noted for E3 (2026-10-26, E3.C) is E4's mechanistic benchmark.

### E4 — Full cross-condition translation: one conditional flow for all condition → condition maps, with SC   (slug: `cross_condition_translation`) · status: planned · owner: agent:modeling

**In one paragraph.** E2.1 found that SC predicts each FC condition's encoding only linearly and weakly (demeaned r
≈ 0.05), and E3 (`windowed_fc_gain`) that extra samples from the same scan do not help. Function predicts function
far better: a subject's rest encoding finds their task encodings at top-1 0.46 (E1), and raw-edge cross-condition
identification reaches 0.53–0.82 against ≈ 0.09 from SC (E2.0). E4 trains **one conditional generative model** that,
given a subject's SC and their FC in another condition (or none), generates their FC in any target condition, and asks
whether it beats linear translation as a point predictor and as a distribution, whether that is about nonlinearity or
about how the model is fit, and whether SC adds anything once some FC is known. Uses: imputing missing task scans,
generating task FC from rest.

**Data and representation.** Matched cohort (917 Glasser subjects), seeds 0–4, E2.1's splits. Targets and sources:
each condition's **block FC PCA codes** ($k$ = 256, E2's equal-variance pooled basis, D2: one basis for all FC
conditions, each condition centred on its own train mean), decoded to edges for scoring; SC: its own PCA codes (64–512,
chosen on val). Full targets only (E3: part samples do not help).

**Cells** (per seed; scored on the test subjects):

| Setting | Source → target | Cells | Models |
|---|---|---|---|
| Single source | one FC condition → another | 56 (8 × 7) | all |
| SC only | SC → each condition (no FC input) | 8 | all |
| (each FC cell with and without SC) | | × 2 | all |
| Multi-source | the other 7 conditions → the held-out target | 8 | baselines (first pass); FM after the multi-source retrain |
| Rest retest ceiling | rest session 1 ↔ session 2 (identity, no model) | 2 | reference |

**Flow models (from noise; first pass trained pairwise).** Both generate the target condition's codes from Gaussian
noise by flow matching (E2.1's recipe); no linear anchor, no CCA, no initialisation from ridge (coefficients and
weights start from their default / zero init).
- **Conditioning interface (DeepSets form):** each available source is embedded as $h = \phi(z_{src}) + e(m_{src})$
  ($\phi$ shared across conditions, possible because all FC conditions share the D2 basis; $e$ a learned condition
  embedding); the embeddings are mean-pooled (empty set: a learned "no FC" vector); the pooled vector, SC codes (zeros
  + a flag when dropped) and the target-condition embedding condition the velocity field. $\phi$ is modular: linear
  (first pass) or a small MLP.
- **FM-nl:** nonlinear velocity field (E2.1's `TokenBackbone`, a residual MLP).
- **FM-lin:** velocity linear in the state, the conditioning and the embeddings, with time-varying coefficients (E2.1's
  `LinearFlowVelocity`, source and target terms separate). With linear $\phi$ and mean pooling it stays exactly a
  linear-Gaussian model, so FM-lin vs B3 isolates the cost of fitting by flow matching, and FM-nl vs FM-lin isolates
  nonlinearity.
- **Pairwise training (first pass):** source set of size 0 (SC only) with probability 0.2, size 1 (one random other
  condition) with 0.8; SC dropped with probability 0.2 when a source is present. With a set of one, mean pooling is the
  identity, so the model is a pairwise translator ($\phi$ linear: a re-parameterisation of a pairwise MLP). An epoch
  draws every training subject once with all 8 targets and a random source per target (E2.1's step count).
- **Multi-source (follow-up, no architecture change):** retrain (or fine-tune from the pairwise weights) with random set
  sizes 1–7, which adds the other-7 → 1 cells and arbitrary subsets. A pairwise-trained model is not evaluated on
  pooled sets it never saw.
- Sampling: point estimate = mean of 256 samples; distribution = 64 samples; starting-noise temperature selected on val
  from {0.25, 0.5, 1}. Search: Optuna on seed 0 (32 trials per model; E2.1's space, cap 3000 epochs, no stopping
  before epoch 50), selected on mean val decoded demeaned r over the cells; seeds 1–4 at the selected settings.

**Baselines** (closed form, CPU; run and reviewed before the flows):

| ID | Model | Role |
|---|---|---|
| B0 | Identity: the source's deviation from its train mean copied into the target (edge space, as E2.0); multi-source: mean of the sources' deviations | the bar every FC → FC model must clear |
| B1 | Block-PCA ridge per cell (E2.0's L1: GCV penalty): source codes → target codes; multi-source: the sources' codes concatenated; **B1+SC**: SC codes appended | linear point reference with the flows' inputs |
| B2 | 9-view multiview CCA (E2.0's L4: MAXVAR over the 8 FC conditions + SC, $(d, \lambda)$ on mean val demeaned r): encode each available view into the shared latent, **average the encodings**, decode to the target. Because every view is encoded into the same latent, averaging handles one source, several (other 7 → held-out target), SC alone, or SC + FC with no new fit | linear model for any input subset; the closed-form counterpart of the pooled flow |
| B3 | Linear-Gaussian: B1 / B1+SC mean + the cell's residual covariance (cross-fitted train residuals) | distributional reference; same model class as FM-lin |

**Hypotheses** (test; support rule as everywhere in v4; per cell and as the mean over each setting's cells):

| ID | Question | Metric | Comparison |
|---|---|---|---|
| H1 | Do the flows translate better than linear maps? | decoded demeaned r, average rank, top-1 | FM-nl and FM-lin − best of B1 / B2 (same inputs) |
| **H2 (primary)** | Is a flow's predictive distribution better? | **energy score** of the observed target under generated samples (proper for one observation), code space | B3 − FM-nl, B3 − FM-lin |
| H3 | Linearity or optimisation? | H1 / H2 metrics | FM-nl − FM-lin (nonlinearity); FM-lin − B3 (fitting by flow matching vs closed form) |
| H4 | Does predicted uncertainty track error? | correlation across subjects of predicted spread with realised error; 90 % coverage | FM vs 0 (B3's spread is constant per cell) |
| H5 | Does SC add once FC is known? | H1 / H2 metrics | (SC + FC) − FC for B1, B2 and the flows |
| H6 | How often does translation beat copying? | share of single-source cells beating B0 on demeaned r and on average rank | each model vs B0 |

**Steps and gates.**

| Step | What | Compute | Gate |
|---|---|---|---|
| E4.0 | Baselines B0–B3 for all cells (incl. multi-source), seeds 0–4; ceilings; heatmaps (done; also $k$ = 512) | CPU | **G0:** B1 (no SC) reproduces E2.0's L1 and B2 (no SC, single source) its L4 at $k$ = 256 (to 1e-6) |
| E4.0c | B1 / B2 re-selected on val average rank (D9; CCA $d$ up to 256) at $k$ = 256, 512 and full rank (CCA without PCA truncation); code size fixed by D10 | CPU | D10 rule |
| E4.1 | Review of the baselines with the user (done 2026-10-28; go given: D8, D9, D10) | — | user |
| E4.2 | Flow code (DeepSets interface, FM-nl, FM-lin) + optimisability checks (D10) + synthetic check | CPU / GPU | **G1:** on simulated data with a planted nonlinear FC → FC map and a linear SC path, FM-nl beats B1 on the nonlinear cells; FM-lin matches B3 (energy within 2 SE) |
| E4.3 | Seed-0 search and pilot | GPU | **G2:** metrics finite; selected trials not at the epoch cap; SC-only cells not below E2.1's block-PCA flows from noise beyond 2 SE |
| E4.4 | Seeds 1–4, analysis, write-up | GPU / CPU | — |

**Deliverables.** Heatmaps per model and metric class (point: demeaned r, average rank, top-1; distribution: energy
score, coverage, spread–error correlation): rows = source (8 conditions, "other 7" for the baselines, "SC only"),
columns = target, with and without SC; difference heatmaps (flows − best linear, FM-nl − FM-lin, FM-lin − B3,
(SC + FC) − FC, model − identity); setting means with the rest retest and own-reconstruction ceilings. Folder
`scripts/experiments/cross_condition_translation/` (standard layout), write-up, spec result.

**What each outcome means.**

| Outcome | Reading | Next |
|---|---|---|
| H1 or H2 supported for FM-nl, and FM-nl > FM-lin | nonlinear translation structure the linear maps miss | FM-nl as the imputation / generation tool; multi-source retrain |
| FM-lin ≈ FM-nl ≈ B3 | the flows only recover the linear-Gaussian solution | B2 / B3 as the tools |
| FM-lin < B3 | fitting by flow matching costs accuracy at this n | optimisation work before any nonlinear claim |
| H5 supported | SC carries information beyond FC | keep SC as a view |
| nothing beats B0 | translation is mostly copying the subject's own pattern | report; the subject fingerprint dominates |

**Follow-ups (not scheduled):** multi-source retrain of the flows (above); nonlinear $\phi$; activity-flow benchmark
(needs per-subject task activations: HCP EVs and GLM betas; formulation from the Cole lab papers); joint-training
ablation; residual-flow variant; the CCA-latent route; rest sessions as training views (D1 nesting mask).

- **Depends on:** E2.1 (models and training recipe), E2.0 (L1 / L4), I1, E3 (no augmentation).
- **Budget / autonomy:** D8.

## 3. Infrastructure

### I1 — Joint translation infrastructure · status: in progress · owner: agent:modeling

- **Goal:** run E2 (and later E3) from library code, on the v3:E0 cohort, splits and metrics.
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
     - `run.py`: records → tables and figures of E2.C;
     - `checks/`: E2.0 pre-checks (done), `check_infra.py` (G0), `synthetic.py` (G1), the G3 pilot check;
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

- **Goal:** one implementation of MAXVAR multiview CCA and the closed-form ridge readout, imported by E2 (L4) and E1.
- **Changes:** new `models/architectures/multiview_cca.py` (`maxvar` fit returning encoders, decoders and shared
  scores; `ridge_gcv`); `scripts/experiments/joint_flow_matching/linear_arms.py` imports them instead of defining
  them (no queued job reads that file).
- **Accept:** E1 gate G0. **Met** (E2 checks ALL_OK; 726 / 726 linear-arm rows reproduced exactly).

## 4. Carried over from v3

v3 closed 2026-10-21 (all experiments done). Carried forward:

| ID | Item | Status |
|---|---|---|
| C!1 | v3:C!2: subject-specific SC is dominated by parcel size (v3:E1.5); FC cannot see it, so FC → SC demeaned r and top-1 are capped by geometry. Comparisons between FC-only models stay valid. | standing caveat (v3:D4: SC stays a target) |
| — | v3:E3 "then": the final reoptimized benchmark with the composite-loss winners as the model set | backlog |
| — | v3:I3: graph / nodal models on top of the mean, incl. FC → SC variants | backlog |
| — | `scripts/notebooks/EDA/FC_matrix_analysis.ipynb` (v3:E4 cells + user edits) uncommitted | open (user) |

## 5. Backlog (not scheduled)

- Further directions from `modeling_directions.md` §9 (multiview CCA variants, transcriptome integration, graph models,
  timeseries tokens), added as items once scoped.
- E2 and E1 replication on 4S456Parcels.
- From v3: final reoptimized benchmark (composite-loss winners); graph / nodal models on top of the mean (absorbs
  v3:I3); editable install and `conn2conn/` namespacing; artifact cleanup and retention (v3 audit 2026-10-06, Ray
  storage pruning script added 2026-10-20); shared constants between `main.py` and `scripts/results_utils/records.py`;
  untracked reference code in `context_packages/modeling/*_context/`; latent-space terms inside the composite loss;
  EMA diagnostics for the composite reference scales.

## 6. Change log

| Date | Change |
|---|---|
| 2026-10-20 | v4 drafted (v3 still active): E2 joint multidirectional translation (regression vs deterministic flow), E1 outline (generative), I1; D1 (9 views, `rest` as the rest target, sessions as an arm with a nesting mask), D2 and D3 proposed. E2.0 pre-check done (CPU): pooled block-scaled basis keeps more held-out task variance than per-task bases and 92–97 % of rest's at k ≥ 64; per-condition codes are not comparable across views; much of the cross-state fingerprint lies beyond 64 components. |
| 2026-10-20 | D1 decided (user). E2 protocol written out (E2.A hypotheses and support rule, E2.B design, E2.C steps / gates / deliverables, E2.D outcomes); steps renumbered E2.0–E2.6. E2.0b spectra: 658 components per view (5,264 pooled); 95 % of train variance needs 499–583 per view, so $k$ is set from held-out variance: grid {128, 256, 512} (user: lower bound 128). D3 proposal revised (25 GPU-h). |
| 2026-10-20 | E2.0c: rest carries 3 % of the unscaled pooled FC variance (5 % with first-singular-value scaling); total-norm block scaling gives it 18 % of the top 256 components, keeps ≥ 0.97 of its own-basis held-out variance at k ≥ 64 and improves cross-view identification; D2 switched to total-norm scaling. |
| 2026-10-20 | E2.0d: leading pooled PCs are reproducible network-contrast axes (PC1 = amplified canonical architecture, default mode ↔ task-positive segregation), only weakly consistent within subject across conditions (r ≤ 0.32). Linear references written out as L1–L4 (arm A1), with a multiview CCA (MAXVAR) translator as the shared linear model; H1 split into H1a (sharing) and H1b (nonlinear vs linear shared); block-scaling citations in D2. |
| 2026-10-20 | E2.7 (optional identity-aware loss step) added. |
| 2026-10-20 | D2 decided (user): equal-variance (total-norm) block scaling, $k$ ∈ {64, 128, 256, 512}. D3 decided (user): autonomous, 25 GPU-h cap. E2 and I1 in progress (agent:modeling). |
| 2026-10-20 | E1 (CCA-latent bidirectional benchmark) and I2 added; D4 decided (user: autonomous, CPU); generative outline renumbered E3; E2 pilot waits for E1. E2: G0 CPU checks ALL_OK (35), linear arms L1–L4 done for 5 seeds. |
| 2026-10-20 | E2.2 gate G1 failed (synthetic): A3 / A4 one-step R² against the true conditional mean 0.865 / 0.712 (< 0.9); G1b passed. E2 stopped at the gate per D3 (already on hold for E1). Smoke trials: A3 13–44 s, A4 18–33 s per trial on one GPU. |
| 2026-10-20 | E1 done (autonomous, CPU): H2 and H4 supported, H1 reversed, H3 not; tasks through the latent beat every v3:E0 task row for FC → SC; SC → latent → FC beats v3:E0 on all conditions. I2 done. Follow-ups: SC dimension grid edge, hybrid input. E2 still stopped at its synthetic gate. |
| 2026-10-21 | E2 rewritten around the consensus latent (D5): Part 1 SC ↔ latent with MLP, identity-aware loss and flow (H1–H4), training fix for the synthetic gate, optional Part 2 FC → FC; D3 revised (Part 1 cap 10 GPU-h). D6 housekeeping. E1 follow-ups recorded as potential next steps. |
| 2026-10-21 | E2 Part 1 MLP set as a nonlinear corrector of ridge (user): ridge prediction + learned correction, starting at the ridge solution; direct MLP as a secondary arm. |
| 2026-10-26 | v4 active (v3 closed 2026-10-21). E2 revised (D7): E2.1 conditional generation of each condition's CCA encoding from SC, 8 targets per subject; models ridge, linear-Gaussian, nonlinear corrector (± identity loss), ridge-start bridge, conditional flow with token dropout (main); H1–H7, primary H3 (energy distance, flow vs linear-Gaussian); distributional synthetic gate added. Former Part 2 → optional E2.2. E3 narrowed to generative follow-ups. v3 caveat and backlog carried over (C!1). |
| 2026-10-26 | **Renumbered (user):** the finished CCA-latent benchmark is now **E1** (was E2) and comes first; the SC → functional-latent experiment is now **E2** (was E1; its sub-items E2.0–E2.2, was E1.0–E1.2), leading into E3. Commit messages and job names before this date use the old IDs (e.g. `5032017` "E2 CCA-latent benchmark", job names `e2_cca_bench`, `e1_linear`, `e1_gpu_checks`). Earlier change-log rows are rewritten with the new IDs. |
| 2026-10-26 | E2.1 built (`2c3e7d7`, `28a1898`): latent cache + linear references, G0b passed (5 seeds); runner, synthetic gates, G3 check, analysis. Chain submitted: gates `21010434` → seed-0 pilot `21010435` → G3 `21010436` → seeds 1–4 `21010437`–`40` → analysis `21010569`, each `afterok`. `TokenBackbone` gained an optional conditioning input (SC codes). |
| 2026-10-26 | E2.1b gate failed (job 21010434: flow energy distance above the Gaussian's; flow spread r 0.70 passed; corrector / bridge 0.99 passed); chain cancelled. Fixes (user): residual conditional flow around the ridge mean, smoothed early stopping, spread-dominated simulation, 9-view gate moved to E2.2 (`be3d501`). Gate rerun: job 21016054. |
| 2026-10-26 | E2.1b gate **passed** (job 21016453) after the residual-flow change and a corrected null (job 21016054 failed only because the shuffled null kept ridge means fit on real pairs; the null now breaks the pairing for the whole model, `e133a2e`): corrector R² 0.982 (ridge 0.983), bridge one step 0.913, flow spread r 0.89, energy 0.0015 vs Gaussian 0.51, shuffled null −0.025. Pilot waits for the user's go. |
| 2026-10-26 | Diagnostics added before the pilot (user): a spread-only null (SC shuffled for the residual flow only, real ridge means; tests whether subject-specific spread comes from SC) and a rest test–retest check (each subject's session 1 vs 2 encodings in the latent; the correlation with the flow's predicted rest spread is the readout, since predicted spread also contains what SC does not explain). Chain: pilot `21020036` → G3 `21020037` → seeds 1–4 `21020038`–`41` → analysis `21020042` (`64a7f21`). |
| 2026-10-26 | E2.2 redrafted (user): cross-condition FC translation as a generative model in block FC PCA codes (single-source, multi-source, SC added, SC only), identity / linear / linear-Gaussian / corrector / residual flow, H1–H6 with the energy score as the primary single-observation distributional metric; runs on the user's go after E2.1. G3 failed on the seed-0 pilot (flow means from 16 samples too noisy: r 0.68 with the corrector); flow point estimate now from 256 samples; chain resubmitted (flow seed 0 `21029998` → G3 `21029999` → seeds `21030000`–`03` → analysis `21030004`). |
| 2026-10-26 | E2.1 done (5 seeds): SC → latent linear (H1 negligible), no SC-dependent spread (H3–H5 not supported), linear-Gaussian is enough; latent → SC corrector +0.004 (H7). Write-up `joint_flow_matching.md`. |
| 2026-10-26 | E2.1 addendum added and running (user): un-anchored flows from noise, linear and nonlinear velocity, with full nulls (`13e111e`). |
| 2026-10-26 | E2.1 addendum done: un-anchored flows learn (94 % / 91 % of ridge's point signal), less efficiently than the closed form; no nonlinear gain on real data, clear gain on synthetic. |
| 2026-10-27 | Flows: 3000-epoch cap and no early stopping or best-weight pick before epoch 50 (user); seed-0 notebooks re-baked (targets shown next to predictions, native subject IDs, temperature sweep). Temperature sweep (seed 0): on block PCA the starting noise dominates single samples (demeaned r 0.006 at T 1 vs 0.031 at T 0); the CCA route is insensitive. Sampling temperature per route set to block PCA 0.25, CCA 1.0 (user); block-PCA flows re-evaluated from saved weights at T 0.25 (`_longT025`). |
| 2026-10-27 | Longer training (3000 cap, min 50) does not help: CCA flows get worse (nonlinear overfits early), block-PCA flows unchanged. Block-PCA linear flow at T 0.25 beats ridge on average rank (+0.030, 5 / 5) and top-1 (+0.012, 5 / 5), trails on demeaned r (−0.004); lower temperature costs energy (calibration). Write-up §8. |

| 2026-10-28 | Housekeeping (user): E2 done (E2.1); E2.2 superseded by E4 (full cross-condition translation, one masked conditional model with SC as a view; outline, drafting). E3 closed: timeseries windows give no gain (`windowed_fc_gain`, gates 1–7), path noise dropped, rest sessions and the activity-flow baseline moved to E4. |

| 2026-10-28 | E4 written (user): one conditional flow from noise for every condition → condition cell (single source, other-7, SC only; with and without SC), benchmarks identity / block-PCA ridge ± SC / 9-view MAXVAR with SC / linear-Gaussian, H1–H5, gates G0–G2; activity flow and ablations as follow-ups; D8 pending (budget, go). E2.2's draft reduced to a pointer. |

| 2026-10-28 | E4 revised (user): nonlinear and linear flows from noise vs the linear-Gaussian (linearity vs fitting), DeepSets conditioning interface trained pairwise first (multi-source as a retrain), B2 = E2.0's 9-view MAXVAR with averaged encodings for any input subset, baselines (E4.0) reviewed before the flows (E4.1). |

| 2026-10-28 | E4.0 done (CPU; jobs 21225137, 21227332): G0 passed (1e-16). Identity (copy the source pattern) is the bar: ties the best linear model on demeaned r (0.107), beats all on identification (average rank 0.845, top-1 0.25); copying within 256 codes drops top-1 to 0.13, so much of its edge is outside the code space; averaging the other 7 conditions helps (identity 0.208, CCA 0.154; ridge on concatenated sources overfits); SC adds nothing once FC is known; B3 calibrated (coverage 0.92). E4.1 review pending. |

| 2026-10-28 | E4.0 at k = 512: copying in codes recovers (top-1 0.13 → 0.17), ridge / CCA do not. D9 decided (user): selection on val average rank from E4.0c on. EDA `fc_deviation_similarity.py`: within-subject agreement of deviations 0.06–0.19, identification 0.88–0.99; split-half reliability dropped (LR / RL runs are not parallel halves). |

| 2026-10-28 | D8 decided (user): autonomous from E4.0c through E4.4, 15 GPU-h. D10 decided: code size fixed by rule after E4.0c (val average rank, smallest of 256 / 512 within 0.005; full rank diagnostic); optimisability checks before the flow search; both FM-nl and FM-lin. E4.0c added. |

Last updated at: 2026-10-28 EDT
