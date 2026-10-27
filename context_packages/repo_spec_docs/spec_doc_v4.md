# Repo Spec v4

**Purpose:** v4 work: modeling directions beyond the linear baselines, drawn from
[`modeling_directions.md`](../modeling/FM+tx_integrate/modeling_directions.md) (the technical reference; model labels such as FM-1 and
MV-5 refer to it). Items: a CCA-latent benchmark that found a shared functional latent across FC conditions (E1);
linear, nonlinear and flow models that predict each condition's latent encoding from SC, including a conditional
flow that generates a subject's encodings (E2); a generative follow-up with more samples per subject (E3).
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
| E2 | SC → functional latent: E2.1 conditional generation of each condition's CCA encoding from SC (linear, nonlinear, linear-Gaussian, flow); E2.2 cross-condition FC translation as a generative model, SC as an added view (planned) | in progress (E2.1 done 2026-10-26; E2.2 planned) | I1, E1, D7 | agent:modeling |
| E3 | Generative follow-ups to E2.1: path noise, rest sessions as repeats, FC timeseries windows as more samples per subject | outline | E2.1 | — |
| I1 | Joint translation infrastructure: pooled FC basis, (subject, source, target) data module, token model, integrator, runner | in progress | — | agent:modeling |
| I2 | Multiview CCA library module (MAXVAR fit / encode / decode), shared by E2 and E1 | done 2026-10-20 | — | agent:modeling |
| D1 | Views and the true rest target | decided 2026-10-20 | — | user |
| D2 | Code space: pooled FC basis with equal-variance (total-norm) block scaling, SC basis, $k$ grid | decided 2026-10-20 | E2.0 | user |
| D3 | E2 compute envelope and autonomous execution | decided 2026-10-20 | — | user |
| D4 | E1 autonomous execution | decided 2026-10-20 | — | user |
| D5 | E2 reframed around the consensus latent; training fix for the synthetic gate; identity-loss arm | decided 2026-10-21 | E1 | user |
| D6 | Spec and repository housekeeping | decided 2026-10-21; v4 active 2026-10-26 | — | user |
| D7 | E2.1 as conditional generation of each condition's encoding (8 targets per subject) | decided 2026-10-26 | E1 | user |
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

### E2 — SC → functional latent: linear, nonlinear and flow models   (slug: `joint_flow_matching`) · status: in progress · owner: agent:modeling

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

#### E2.2 — Cross-condition FC translation as a generative model, with SC as an added view (planned)

**In one paragraph.** E2.1 asks what structure says about function; its seed-0 pilot suggests the answer is a linear
mean and little else at 659 subjects. Function says far more about function: a subject's rest encoding finds their
task encodings at top-1 0.46 in E1's latent, and raw-edge identification across conditions reaches 0.53–0.82, against
about 0.09 from SC. E2.2 asks whether a generative model translating **between FC conditions** captures that
signal better than linear maps: given a subject's FC in one or more conditions, generate their FC in another, with a
mean and a calibrated spread. SC enters as one more conditioning view, which turns "does SC predict FC?" into the
sharper "does SC add anything once some FC is known?". Practical uses: imputing missing task scans and generating a
subject's task signatures from rest.

**Representation.** Targets are each condition's **block FC PCA codes** (E2.0's equal-variance pooled basis, $k$ =
256), decoded to edges for scoring. They keep the condition-specific individuality that E1 showed the 32-dimensional
shared latent discards (E1: rest's own codes beat its latent encoding for SC). Sources are the same codes, optionally
plus E1's latent encodings and SC codes as conditioning.

**Translations.**

| Setting | Source → target | Count per seed |
|---|---|---|
| Single source | one FC condition → another (FC → FC pairs) | 56 ordered pairs |
| Multi-source | the other 7 conditions → the held-out target (leave the target out) | 8 targets |
| SC added | SC + single or multi-source FC → target | as above |
| SC only (reference) | SC → each condition (E2.1's question, in block codes) | 8 |

**Models** (the E2.1 recipe, which passed its synthetic gate):

| Model | What it is | Role |
|---|---|---|
| Identity | copy the subject's source deviation into the target condition | the bar any FC → FC model must clear (strong; E2.0 finding 4) |
| Linear references | per-pair ridge (L1) and the multiview CCA translator (L4), both already computed for all pairs (E2.0); multi-source CCA (average of the sources' encodings, decoded to the target) | linear point references; the best linear model per pair is the mean anchor below |
| Linear-Gaussian | linear mean + one residual covariance per target condition | distributional reference |
| Nonlinear corrector | linear mean + a learned correction, one shared network with source and target condition tokens (the old A3, now anchored on the linear mean) | nonlinear point model |
| **Residual conditional flow** | samples of the residual around the linear mean, from noise, conditioned on the source codes and both condition tokens (and SC codes in the SC-added setting) | **main model** |

**Evaluation.** Each subject has one observed FC per target condition (two for rest, via its sessions), so the
distributional scores are ones that work with a single observation:

| ID | Question | Metric | Compared with |
|---|---|---|---|
| H1 | Is there nonlinear structure in FC → FC beyond the best linear map? | decoded demeaned r, mean over pairs (also task-target pairs alone) | corrector − best linear reference |
| H2 | Do several source conditions beat the best single one? | the same, task targets | multi-source − best single source (usually rest) |
| **H3 (primary)** | Does the flow's predictive distribution beat a linear-Gaussian one? | **energy score** of the observed target under the generated samples (a proper score for one observation; lower is better), in code space | linear-Gaussian − flow |
| H4 | Does predicted uncertainty track actual error? | correlation across subjects between the flow's predicted spread and its realized error; coverage of the observed target | flow vs 0; Gaussian has constant spread |
| H5 | Does SC add anything once FC is known? | H1 and H3 metrics with SC added vs without | (SC + FC) − FC |
| H6 | How often does translation beat copying? | share of the 56 pairs where the model beats identity on demeaned r and on average rank | model vs identity |

References and ceilings: each target's own $k$-dimensional reconstruction; the rest session 1 ↔ session 2
translation (with the D1 nesting mask) as the test–retest ceiling for rest. Same support rule as everywhere in v4.

**Gates.** G0: the linear references reproduce E2.0's stored L1 / L4 rows exactly. G1: the original 9-view synthetic
gate, rerun with the corrector and residual-flow forms (it failed for the un-anchored models; thresholds unchanged),
plus a single-observation version of the distributional checks. G3: seed-0 pilot (corrector at least as good as the
best linear reference; flow means agree with the corrector; metrics finite; no selected trial at the epoch cap).

**Deliverables.** The 9 × 9 cross-condition matrix (demeaned r, average rank) for identity, linear, corrector and
flow; multi-source and SC-added panels; energy scores and spread–error calibration; write-up section; spec result.

**What each outcome means.**

| Outcome | Reading | Next |
|---|---|---|
| H3 and H4 supported | the flow models subject-specific uncertainty in FC → FC | it becomes the cross-task generative tool; E3 adds timeseries windows |
| H1 or H2 supported, H3 not | gains are in the mean (nonlinear or multi-source), not the distribution | use the corrector / multi-source model as the translation tool |
| H5 supported | SC carries information beyond FC | keep SC as a view in E3 |
| nothing beats identity / linear | at this sample size cross-condition FC is linearly translatable | linear multiview tools for imputation; generative work waits for more samples per subject |

- **Depends on:** E2.1's result (the recipe, and whether the flow behaves on real data); E2.0's linear references
  (done); the infrastructure in `tune.py` / `e21_run.py`.
- **Budget / autonomy:** to be decided when E2.1 reports (estimate ≤ 10 GPU-h: one shared network per seed covers all
  pairs). Runs on the user's go.

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

### E3 — Generative follow-ups · status: outline · owner: —

- **Question:** once E2.1 shows whether a conditional flow learns subject-specific distributions from eight condition
  encodings, does it improve with a stochastic path (noise along the bridge) and with more samples per subject?
- **Sketch:** (1) path noise on the ridge-start bridge (FM-4); (2) the two rest sessions as true repeats (calibration
  of within-condition spread); (3) FC timeseries windows as many samples per subject (needs timeseries caches,
  `modeling_directions.md` AR-6); scored as E2.1 (energy distance, set identification, spread calibration) and by
  held-out likelihood against `CrossModal_ConditionalGaussian` (MV-5).
- **Depends on:** E2.1's result; designed after it reports.

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

Last updated at: 2026-10-27 EDT
