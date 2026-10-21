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
| E1 | Structure ↔ functional-latent translation: SC → consensus latent (decoded to each FC condition) with MLP, identity-aware loss and flow; optional multi-condition FC → FC (Part 2) | planned (revised 2026-10-21 after E2) | I1, E2, D5 | agent:modeling |
| E2 | CCA-latent bidirectional benchmark: multiview CCA latent of FC as the encoding for FC → SC and the target for SC → FC | done 2026-10-20 | I2, D4 | agent:modeling |
| E3 | Conditional generative translation: noise-start flow (FM-2) or noisy bridge (FM-4), built on E1's SC → latent model | outline | E1 | — |
| I1 | Joint translation infrastructure: pooled FC basis, (subject, source, target) data module, token model, integrator, runner | in progress | — | agent:modeling |
| I2 | Multiview CCA library module (MAXVAR fit / encode / decode), shared by E1 and E2 | done 2026-10-20 | — | agent:modeling |
| D1 | Views and the true rest target | decided 2026-10-20 | — | user |
| D2 | Code space: pooled FC basis with equal-variance (total-norm) block scaling, SC basis, $k$ grid | decided 2026-10-20 | E1.0 | user |
| D3 | E1 compute envelope and autonomous execution | decided 2026-10-20 | — | user |
| D4 | E2 autonomous execution | decided 2026-10-20 | — | user |
| D5 | E1 reframed around the consensus latent; training fix for the synthetic gate; identity-loss arm | decided 2026-10-21 | E2 | user |
| D6 | Spec and repository housekeeping | decided 2026-10-21 | — | user |

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
| D1 | **Views (user 2026-10-20):** main model on 9 views, $\mathcal{M}$ = {SC, `rest`, 7 tasks} (72 ordered pairs). `rest` (4 runs, xcp-d combined relmat) is the true rest view and the rest target, as in every earlier result. For E1 Part 2 and E3, a sessions variant adds `rest_S1` and `rest_S2` (per-session mean of the LR and RL run matrices) with a nesting mask: pairs between `rest` and either session are excluded from training and evaluation, because a session is half of `rest`'s data. S1 ↔ S2 is then the in-model test–retest translation. | Continuity with v3:E0 / E2 and v2:E2.2; the mask prevents near-identity leakage; the sessions add a ceiling and the two-sample targets E3 needs. |
| D2 | **Decided (user 2026-10-20):** one FC basis from a PCA of the stacked train-split FC of all FC views, each block centered on its own train mean (condition means stored and added back at decoding by the target token) and divided by its total (Frobenius) norm, so every view carries the same total variance (standard multiblock / consensus PCA block scaling, Westerhuis et al. 1998; the first-singular-value alternative is multiple factor analysis, Escofier & Pagès; E1.0c: total-norm scaling is better for rest and for cross-view identification); SC keeps its own train PCA (also divided by its total norm); codes share one dimension $k \in$ {64, 128, 256, 512}, bounded above by the SC rank (658; E1.0b). (The first E1 design's basis ablations, per-condition bases and an SC basis rotated onto the FC basis, apply to Part 2 only.) | The bridge interpolates codes directly, so all views need comparable coordinates; per-condition centering keeps condition offsets out of the flow, so the model works on subject deviations (the quantity demeaned metrics score). |
| D3 | **Decided (user 2026-10-20; revised by D5 on 2026-10-21: Part 1 cap 10 GPU-h, Part 2 needs the user's go):** E1 runs autonomously; cap 25 GPU-h (estimate 15–18); per-job time limit = pilot time per trial × 1.5. Stop and report on a failed gate (G0, G1, G3), > 20 % task errors after one retry, a log / config mismatch, a code change outside I1's files, or the cap. Commits local (no push). Relaxing a threshold needs the user. | Small models in code space; explicit exits as v3:D1–D3. |
| D4 | **Decided (user 2026-10-20):** E2 runs autonomously through E2.0–E2.3, CPU only, cap 15 CPU-node-hours. Stop and report on a failed gate (G0, G1), > 20 % task errors after one retry, a log / config mismatch, a code change outside I2 and E2's folder (plus the E1 import switch), or the cap. Commits local (no push). Relaxing a threshold needs the user. The E1 A3 / A4 pilot (E1.3) waits for E2's result. | E2 is closed form and cheap; it decides E1's framing, so it runs first. |
| D5 | **Decided (user 2026-10-21):** E1 is reframed as SC ↔ consensus latent (Part 1, the core), with the multi-condition FC → FC model kept as an optional Part 2. The failed synthetic gate is fixed in training (flow time drawn at 0 with probability $p_0$, longer training), not by lowering the threshold. An identity-aware loss (demeaned corr-eye on the latent) is an arm of Part 1. E3 stays an outline until E1 reports. D3 is revised: Part 1 autonomous, cap 10 GPU-h; Part 2 needs the user's go. | E2: the latent is where SC → FC gains and it is a smaller, cleaner problem; the gate should hold. |
| D6 | **Decided (user 2026-10-21):** v3 is closed (and v4 made active) once v3:E3's queued jobs finish; `scripts/notebooks/EDA/FC_encoded_conditions_analysis.ipynb` is committed as is; commits stay local until the user reviews them. | Avoid editing a spec with running work; keep the record reviewable. |

## 2. Experiments

### E1 — Structure ↔ functional-latent translation: nonlinear and flow models   (slug: `joint_flow_matching`) · status: planned (revised 2026-10-21) · owner: agent:modeling

**In one paragraph.** E2 showed that a subject's *consensus functional latent* (the part of their FC shared across
rest and the seven tasks, learned without SC) is a better thing to predict from structure than any single FC scan:
predicting it from SC and decoding it into each condition beats predicting that condition directly, for all eight
conditions. But E2 used only linear maps, and the gap to what the latent can hold is large (rest: 0.105 predicted vs
0.304 when the true latent is decoded). E1 asks whether nonlinear models, an identity-aware loss and a deterministic
flow close part of that gap. The latent becomes the shared functional target: one model predicts it from SC, and fixed
decoders translate it into each task's FC.

**What changed from the first E1 design (2026-10-20) and why.** The first design trained one network over all 72
view pairs in the block PCA codes. E2 moved the centre of gravity to the latent, so the core of E1 is now the single
SC ↔ latent translation (Part 1). The original multi-condition FC → FC model is kept as an optional Part 2. Work
already done carries over unchanged: the pre-checks (E1.0, below), the code space and its checks (I1, gate G0
passed), and the linear references for all 72 pairs (A1: L1–L4, 5 seeds).

#### E1.A Questions and hypotheses

| ID | Question | Comparison (test, paired by seed) | Primary metric | Supported if |
|---|---|---|---|---|
| H1 | Does a nonlinear model predict the latent from SC better than ridge? | MLP − ridge (E2's SC PCA → consensus latent model) | demeaned r of the decoded FC, averaged over the 8 conditions | support rule |
| H2 | Does an identity-aware loss make the predicted latent more subject-specific? | MLP + identity loss − MLP | average rank of the decoded FC (mean of 8 conditions); demeaned r must not drop by more than 0.005 | support rule, plus the non-inferiority margin |
| H3 | Does a flow add individual detail beyond regression? | flow, integrated − flow, one step; and − ridge | average rank (demeaned r and top-1 reported) | support rule against both |
| H4 | (secondary) Does a nonlinear model improve the reverse, latent → SC? | MLP − ridge, inputs: 8-condition consensus and each task's encoding | SC demeaned r | support rule |

**Support rule** (as everywhere in v4): the seed-mean difference is above 2 standard errors and has the same sign in
at least 4 of 5 seeds. Every result is also reported per condition, with latent R² and latent identification
(predicted vs true latent among the 184 test subjects), next to the v3:E0 rows, E2's ridge and the latent ceiling.

#### E1.B Design (Part 1: SC ↔ latent)

**The target.** The SC-blind consensus latent from E2, refit per seed exactly as E2 chose it (block FC PCA, 256 codes
per condition; 32 dimensions; λ = 0.1). It never sees SC, so the target is not tuned to be easy to predict. Decoding
to condition $m$ uses that condition's fixed CCA decoder; nothing about the decoders is trained in E1.

**The input.** SC's own PCA codes (equal-variance scaled, as in E2), with the number of codes (64–512) chosen on val.

**The models.**

| Model | What it does | Detail |
|---|---|---|
| Ridge (reference) | linear map SC codes → latent | E2's model, re-scored in E1's pipeline (must reproduce E2's numbers exactly; gate G0b) |
| MLP | nonlinear regression SC codes → latent, squared error | residual MLP, small (width 128–512, 1–3 blocks), dropout and weight decay searched |
| MLP + identity loss | the same, trained with squared error + demeaned corr-eye (`compute_correye_dm_loss` in `models/train/loss.py`, imported, not modified) on the latent predictions | loss weight ∈ {0.1, 0.3, 1}; selected on val average rank, demeaned r reported |
| Flow (paired bridge) | starts from the ridge prediction and learns a velocity field that carries it to the true latent; one Euler step is a learned correction of the ridge prediction, full integration transports the distribution | start points on train are **cross-fitted** ridge predictions (5 folds), so they look like test-time predictions; scored at 1 step and at 4 / 16 / 64 RK4 steps |

**Training fix (from the failed synthetic gate).** The first gate failed because the flow was trained with flow time
drawn uniformly, so very little training sees the start point that a one-step prediction uses, and the networks were
under-trained at this sample size. E1 now (a) draws flow time as 0 with probability $p_0$ and uniformly otherwise,
$p_0 \in$ {0, 0.25, 0.5} searched, which directly supervises the one-step prediction without removing the transport
part; (b) trains longer (up to 1,000 epochs, patience 100, batch 64). The gate thresholds are unchanged.

**Selection and search.** Optuna, 32 trials per model and seed; each model selected on val (MLP and flow: mean
decoded demeaned r over the 8 conditions; identity-loss model: mean decoded average rank); test scored once.

**Part 1b (reverse, H4).** The same MLP, inputs = the 8-condition consensus and each task's latent encoding, output =
SC codes; ridge from E2 as the reference. Rest FC is left out on purpose: E2 showed the 32-dimensional latent loses
rest's SC information, and rest's own PCA codes already do better.

#### E1.C Part 2 (optional): multi-condition FC → FC translation

The original idea: one network with source and target condition tokens translating between the 8 FC conditions in
the block PCA codes, as a regression (A3) and as a flow (A4), against the linear references already computed
(L1–L4, identity, ceilings). It runs only after Part 1 reports and with the user's go. Its infrastructure exists
(`tune.py`); it needs the same training fix and a rerun of the original synthetic gate.

#### E1.D Steps, gates and outputs

| Step | What | Compute | Gate (stop and report if it fails) |
|---|---|---|---|
| E1.0 | Pre-checks; code space; linear references L1–L4 | CPU | done 2026-10-20 |
| E1.1 | Latent-target cache per seed (E2's latent, encodings, decoders); the latent runner | CPU | **G0b:** ridge through the new pipeline reproduces E2's SC → FC rows to 1e-6 |
| E1.2 | Synthetic gates with the training fix: the original 9-view gate rerun unchanged, plus a single-pair version shaped like Part 1 (input 256 → latent 32) | 1 GPU, < 1 h | **G1:** MLP and flow one-step R² ≥ 0.9 against the true conditional mean; shuffled pairing collapses (demeaned r ≤ 0.02) |
| E1.3 | Pilot, seed 0: all Part 1 models | 1 GPU | **G3:** MLP val within 0.005 of ridge or better; flow one-step predictions correlate ≥ 0.9 with the MLP's; all metrics finite; selected trials stop before the epoch cap |
| E1.4 | Seeds 1–4, Part 1 and 1b | GPU array | — |
| E1.5 | Analysis and write-up | CPU | — |
| E1.6 | Part 2, if approved | GPU | its own pilot gate |

**Deliverables** (`scripts/experiments/joint_flow_matching/results/Glasser/`): per-seed records; `summary.md` with
one row per hypothesis (difference ± SE, seeds positive, supported or not); figures in the E2 style with
input → output labels: (1) SC → FC per condition for ridge, MLP, MLP + identity loss and flow, with the v3:E0 rows
and the latent ceiling; (2) average rank and demeaned r against flow integration steps; (3) latent R² per dimension
for each model; (4) latent → SC for each input. Write-up `joint_flow_matching.md`, experiments index row, spec result
line.

#### E1.E What each outcome means

| Outcome | Reading | Next |
|---|---|---|
| H1 supported | the SC → latent map is nonlinear | the nonlinear model becomes the base for E3 |
| H1 not supported | linear is enough at 659 subjects | E3 builds on ridge; consider more data or a structural prior |
| H2 supported | identity-aware training makes predictions more individual at no accuracy cost | carry the loss into E3 and Part 2 |
| H3 supported | transport adds individual detail regression shrinks away | E3 on the flow (adding noise to the path) |
| H3 not supported | the flow is a regression in disguise here | E3 starts from noise instead (FM-2) |

- **Budget / autonomy:** D3 (revised 2026-10-21): Part 1 runs autonomously, cap 10 GPU-h (estimate ≤ 3: trials take
  seconds); Part 2 needs the user's go.
- **Caveats:** the latent ceiling is mildly optimistic (the consensus includes the target condition, 1/8 of it);
  the decoders are linear, so even a perfect latent prediction is bounded by the ceiling; FC → SC is read with v3:C!2.

#### E1 work done so far (first design, 2026-10-20)

- Pre-checks E1.0a–d (results below): the block PCA with equal-variance scaling, its spectrum and components.
- Linear references for all 72 view pairs, 5 seeds × 4 code sizes (`results/Glasser/records/linear/`).
- Gate G0 (code space, decoding, integrators, transport map, CCA): 35 / 35 checks passed.
- First synthetic gate (job 20459033): failed (MLP 0.865, flow one-step 0.712 against 0.9; ridge 0.957); shuffled
  pairing collapsed as required. Smoke trials on real data: 13–44 s each. Fixed by the training change in E1.B.

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
   ≤ 0.03. Pooling the two sessions instead of `rest` (sessions variant) gives rest 0.95–1.00 at k ≥ 64.
3. **Per-condition codes are not comparable across views:** identifying a held-out subject's task FC from their rest
   FC by correlating codes is at chance in own-basis codes (k = 64: top-1 0.000–0.006, average rank 0.45–0.56), and
   well above chance in pooled codes (top-1 0.03–0.09, average rank 0.72–0.81; S1 → S2 0.53 / 0.94). The paired bridge
   (A4) therefore needs the pooled basis; the per-condition ablation (Part 2) applies to A3 only.
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
| 2026-10-21 | E1 rewritten around the consensus latent (D5): Part 1 SC ↔ latent with MLP, identity-aware loss and flow (H1–H4), training fix for the synthetic gate, optional Part 2 FC → FC; D3 revised (Part 1 cap 10 GPU-h). D6 housekeeping. E2 follow-ups recorded as potential next steps. |

Last updated at: 2026-10-21 EDT
