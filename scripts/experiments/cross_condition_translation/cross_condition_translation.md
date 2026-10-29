# cross_condition_translation — full cross-condition translation (spec v4 E4)

**Question.** Given a subject's SC and their FC in another condition (or none), can one conditional generative model
predict their FC in any target condition better than linear and copy baselines, as a point and as a distribution, and
does SC add anything once FC is known? Spec: `context_packages/repo_spec_docs/spec_doc_v4.md` E4 (design D8).

**Setup.** Matched cohort (917 Glasser subjects), seeds 0–4, E2.1's splits (659 / 74 / 184 at seed 0). Sources and
targets: each condition's block FC PCA codes (k = 256, E2's equal-variance pooled basis), decoded to edges for
scoring; SC codes 64–512. Cells per seed: 56 single-source, 8 "other 7 → held-out", 8 SC-only; FC cells with and
without SC. Point metrics on decoded edges (demeaned r, average rank, top-1; E2's `score`); distribution metrics in
code space (energy score, 90 % coverage).

**Code.** `config.yml`; `e4_common.py` (cells, E2 data access, energy metrics, per-k folders); `run_baselines.py` (B0–B3,
ceilings, G0; one seed; `--k`); `run_identity_codes.py` (identity in code space); `analyze_baselines.py`; `launch/run_baselines.sh`,
`launch/analyze_baselines.sh`. E4.0c: `run_e40c.py`, `analyze_e40c.py`. Flows: `fm_common.py` (data, FM-nl, FM-lin,
FM-lin shared, training, sampling, scoring), `run_flows.py` (search / final / dist), `run_pooled_ridge.py` (control),
`analyze_flows.py`; checks in `checks/`; launchers `launch/run_flows_{search,final}.sh` (`MODELS=` selects models).

## E4.0 — baselines (2026-10-28; jobs 21225137, 21227332; CPU, 12–24 min per seed)

**Models.** B0 identity (copy the source's edge deviation; multi-source: mean of the sources'); B0-codes (the same in
256 codes); B1 ridge per cell (E2.0 L1; +SC appends SC codes; multi-source concatenates the sources); B2 9-view
multiview CCA (E2.0 L4: MAXVAR over SC + 8 FC conditions; average the available views' encodings, decode the target);
B3 linear-Gaussian (B1 mean + per-cell residual covariance).

**G0 passed** (all 5 seeds): B1 and B2 without SC reproduce E2.0's L1 / L4 rows at k = 256 to 1e-16.

**Results** (k = 256; test, mean over each setting's cells, mean ± SE over 5 seeds; `tables/k256/baselines_summary.md`,
`tables/k256/baselines_settings.csv`, per cell `tables/k256/baselines_cells.csv`):

| Model | Single source: demeaned r / avg rank / top-1 | Other 7 → held-out | SC only |
|---|---|---|---|
| B0 identity, edges | 0.107 / 0.845 / 0.25 | 0.208 / 0.996 / 0.89 | – |
| B0 identity, 256 codes | 0.095 / 0.780 / 0.13 | 0.177 / 0.964 / 0.56 | – |
| B1 ridge | 0.097 / 0.767 / 0.10 | 0.100 / 0.765 / 0.12 | – |
| B1 ridge + SC | 0.091 / 0.767 / 0.08 | 0.110 / 0.797 / 0.15 | 0.044 / 0.662 / 0.02 |
| B2 multiview CCA | 0.107 / 0.799 / 0.10 | 0.154 / 0.900 / 0.24 | – |
| B2 + SC | 0.104 / 0.806 / 0.10 | 0.153 / 0.900 / 0.23 | 0.047 / 0.658 / 0.02 |
| Ceilings | own 256-code reconstruction 0.556 / 1.000 / 1.00; rest session 1 ↔ 2 copy 0.494 / 0.989 | | |

B3 (distribution, code space): energy score 0.015 (single source), 0.017–0.019 (other 7), 0.016 (SC only); 90 %
coverage 0.92 throughout (calibrated).

**Findings.**
1. **Copying is the bar to beat.** Identity (copy the source's own pattern) ties the best linear model on demeaned r
   and beats every model on identification (average rank 0.845 vs ≤ 0.81; top-1 0.25 vs ≤ 0.10). Ridge beats it in
   13 / 56 single-source cells, mostly with rest as the target.
2. **Much of identity's lead is outside the 256 codes.** Copying in code space drops top-1 from 0.25 to 0.13 (other 7:
   0.89 → 0.56). Within code space, ridge ties the copy on demeaned r (and wins 22 / 56 cells) but trails on
   identification; CCA is slightly ahead on demeaned r and average rank. The 256-code representation caps how
   individual any code-space model (including the flows) can be.
3. **Averaging sources helps, if done by averaging.** Mean of the other 7 conditions: identity 0.208 (rest target
   0.34), codes 0.177, CCA's averaged encodings 0.154; ridge on 7 concatenated sources (1,792 inputs) overfits and is
   no better than one source (0.100). The CCA latent loses individuality relative to averaging raw codes.
4. **Rest is the easy target; rest and wm are the best sources.** Every source predicts rest at 0.13–0.23; task
   targets sit at 0.05–0.15 (emotion lowest). Best source per target (ridge): rest for 5 tasks, wm for rest, gambling,
   relational.
5. **SC adds nothing once FC is known**, and alone predicts little (0.044–0.047, E2.1's level). Appending SC slightly
   hurts single-source ridge (more inputs at n = 659).
6. **The linear-Gaussian is calibrated** (coverage 0.92): the flows' distributional bar is energy score ≈ 0.015.

Figures (`figures/k256/`): `fig_baselines_demeaned_pearson.png`, `fig_baselines_avg_rank.png`, `fig_baselines_top1_acc.png`
(heatmaps; columns = copy / ridge / multiview CCA, top row without SC, bottom row codes / + SC),
`fig_baselines_differences.png` (each model − its reference: ridge and CCA vs the copy in codes, + SC vs without; red =
better), `fig_baselines_distribution.png`, `fig_baselines_settings.png`. Results, tables and figures sit in per-code-size
subfolders (`k256/`, `k512/`).

### k = 512 (2026-10-28; jobs 21228675, 21231505; `tables/k512/`, `figures/k512/`)

G0 passed (1e-16). Single source, k 256 → 512 (demeaned r / average rank / top-1):

| Model | k = 256 | k = 512 |
|---|---|---|
| Copy, codes | 0.095 / 0.780 / 0.13 | 0.101 / 0.808 / 0.17 |
| B1 ridge | 0.097 / 0.767 / 0.10 | 0.094 / 0.772 / 0.10 |
| B2 multiview CCA | 0.107 / 0.799 / 0.10 | 0.107 / 0.803 / 0.10 |
| Own reconstruction (ceiling) | 0.556 | 0.612 |

Other 7: copy in codes 0.177 / 0.964 / 0.56 → 0.191 / 0.979 / 0.69; ridge 0.100 → 0.118 (top-1 0.12 → 0.20); CCA unchanged
(0.156 / 0.905 / 0.24). CCA still selects d = 32 (4 / 5 seeds). **Components 257–512 carry individual signal** (the
code-space copy recovers about half of the gap to the edge copy), **but neither ridge nor CCA uses it**: their
single-source scores do not move. The limit is in the models (shrinkage, the small CCA latent chosen on demeaned r),
not only in the truncation. Next check (E4.0c): CCA on full-rank codes (no PCA truncation).

Related EDA (2026-10-28): `scripts/notebooks/EDA/FC_deviation_similarity.py` (figures in
`scripts/notebooks/EDA/figures/fc_conditions/deviation_similarity/`): a subject's FC deviations from each condition's
mean agree only weakly across conditions (within-subject r 0.06–0.19; rest–task 0.13–0.19), yet that agreement is
individual (between-subject r ≈ 0; average rank 0.88–0.99). Split-half reliability is not estimated: LR / RL task runs
differ in phase-encoding direction, stimuli and block order, so they are not parallel halves.

## E4.0c — selection on val average rank and the code size (2026-10-28; `tables/e40c/`, `figures/e40c/`)

B1 / B2 re-selected on val average rank (D9) at k = 256, 512 and full rank (CCA without PCA truncation). D10 rule
(smallest k within 0.005 of the better val score): **k = 256** (0.812; 512: 0.799; full rank 0.803, no flag). CCA
selects d = 32 at k = 256 in every seed. Full-rank codes do not help either linear model.

## E4.2 — flow checks (2026-10-28; `checks/`, `figures/checks/`)

`checks/check_fm.py` (O1–O3, G1; records `check_fm*.json`): both flows train (O1) and memorise 32 subjects (O2: FM-nl
0.96, FM-lin 0.98); FM-nl beats ridge on a planted nonlinear map in 56 / 56 cells (G1). **O3 failed**: on
linear-Gaussian synthetic data FM-lin's energy score is 0.55 worse than the closed-form Gaussian.
`checks/diagnose_o3.py` splits the gap: it comes from the **mean** (+0.58), not the spread (−0.04); a per-target
low-rank state term (added to FM-lin) changed nothing. Flows estimate the conditional mean less efficiently than
regression at n = 659 (as E2.1 §7).

`checks/loss_curves.py` (`fig_loss_curves.png`, real data, seed 0, 3000 epochs): both flows overfit early (val loss
minimum: FM-lin ≈ epoch 50, FM-nl ≈ 200) and the val-loss minimum coincides with the val average-rank peak (FM-lin 0.777,
FM-nl 0.800; ridge 0.780, CCA 0.812). A 300-epoch floor briefly used after O3 cost FM-lin 0.01; the floor is back at 50.

`checks/shared_flow_test.py` (`fig_shared_flow_test.png`): E2.1's linear flow form (fixed time basis; **one input map
for every (source, target) pair**, pair identity only through a bias; `fm_shared`, 1.0 M parameters) reaches val average
rank 0.909 on seed 0 (ridge 0.788, CCA 0.824); with one map per target (`fm_shared_tgt`, 4.4 M) 0.826. `fm_shared` was
benchmarked (user: if it passes, run end to end). Control added with it: **B1-pooled**, the same sharing in closed form
(one ridge for all 56 pairs, `run_pooled_ridge.py`; B1-pooled-tgt: one per target). Its penalty is irrelevant (val
average rank 0.9204 from α = 0 to 10⁻³).

## E4.3–E4.4 — search and final runs (2026-10-29; jobs 21251796, 21318307–09, 21320724)

FM-nl: 32-trial search (18 completed), selected width 256, depth 1, dropout 0.18, T 0.25. `fm_shared`: 32-trial search
(11 completed), T 0.25. Seeds 0–4 retrained at the selected settings (FM-nl with the 50-epoch floor; its search ran with
300). Distributions scored at T = 1 from the saved weights (`run_flows.py dist`): the selected T (chosen for point
accuracy) narrows the samples (coverage 0 at T 0.25). FM-lin (per-pair encoders) was not benchmarked (superseded by
`fm_shared`). Analysis `analyze_flows.py` → `tables/flows/k256/`, `figures/flows/k256/`.

Test, seeds 0–4, single source without SC (56 cells; mean ± SE over seeds):

| Model | Demeaned r | Average rank | Top-1 | Energy (T 1) | Coverage 90 |
|---|---|---|---|---|---|
| B0 copy (edges) | 0.107 | 0.843 | 0.25 | – | – |
| B0 copy (codes) | 0.095 ± 0.001 | 0.780 ± 0.003 | 0.13 | – | – |
| B1 ridge per pair | 0.099 ± 0.001 | 0.784 ± 0.002 | 0.11 | – | – |
| B2 multiview CCA | 0.105 ± 0.002 | 0.815 ± 0.002 | 0.12 | – | – |
| B3 linear-Gaussian | – | – | – | 0.0154 | 0.93 |
| B1-pooled-tgt (one ridge per target) | 0.119 ± 0.001 | 0.843 ± 0.002 | 0.18 | – | – |
| **B1-pooled (one ridge for all pairs)** | **0.124 ± 0.001** | **0.916 ± 0.001** | **0.32** | – | – |
| FM-nl | 0.097 ± 0.002 | 0.795 ± 0.001 | 0.11 | 0.0155 | 0.78 |
| **FM-lin shared (`fm_shared`)** | 0.119 ± 0.001 | 0.911 ± 0.001 | 0.31 | **0.0152** | 0.88 |

Hypotheses (support rule; `tables/flows/k256/summary.md`):
- **H1 (flows vs linear maps):** FM-lin shared beats B1 and B2 on every point metric (average rank +0.127 / +0.096,
  top-1 +0.20 / +0.19, 5 / 5). FM-nl beats B1 on average rank (+0.012) and top-1 but not B2 (−0.019).
- **H1c (control): the gain is the sharing, not the flow.** B1-pooled beats FM-lin shared on every point metric in
  every seed (average rank −0.006, top-1 −0.019, demeaned r −0.004). Fitting the shared map by flow matching costs a
  little, as E2.1 found for per-pair maps.
- **H2 (distribution):** FM-lin shared beats B3's energy score by 0.0002 (5 / 5; ~1 %, B3 has the per-pair mean); FM-nl
  ties B3. Both flows under-cover (90 % coverage 0.88 / 0.78 vs B3 0.93).
- **H3:** FM-nl − FM-lin shared: −0.115 average rank; nonlinearity without sharing does not compete with sharing.
- **H4:** FM-nl's predicted spread tracks its error (r 0.10, 5 / 5); FM-lin shared's does not (as a linear-Gaussian
  must not).
- **H5 (SC given FC):** adds nothing for B1, B2 or FM-lin shared; FM-nl gains a little (average rank +0.010, 5 / 5).
- **H6 (vs copying):** B1-pooled and FM-lin shared are the **first models to beat the edge copy**: in 92 % / 90 % of
  cells on average rank and 85 % / 83 % on demeaned r (B1 13 % / 26 %, B2 27 % / 52 %, FM-nl 12 % / 33 %).
- **SC only:** the flows trail or tie ridge / CCA (FM-lin shared 0.673 average rank, FM-nl 0.628; B1 0.661).

**Gate G2 (as written) failed on two counts:** `fm_shared`'s selected weights sit at the 3000-epoch cap (still
improving slowly, though B1-pooled bounds the linear class), and its SC-only cells trail E2.1's block-PCA linear flow at
T 0.25 (average rank 0.673 vs 0.692; E4 trains SC-only on 20 % of draws). Results reported regardless (user: run end to
end).

**Reading.** Translation between conditions is best done with **one map shared by every condition pair**: least
squares then regresses each condition's codes onto the mean of the subject's other conditions, i.e. it predicts the
subject's cross-condition consensus (a denoised fingerprint), and it does so better than copying the source pattern.
Per-pair models (ridge, CCA's decoders, FM-nl) fit condition-specific noise from 659 subjects. Flow matching recovers
the shared map (FM-lin shared) but not more, and the nonlinear flow does not find structure that sharing misses.

Figures (`figures/flows/k256/`): `fig_flows_settings.png`, `fig_flows_cells_{avg_rank, demeaned_pearson, top1_acc}.png`,
`fig_flows_differences.png`; checks in `figures/checks/`.

## Next (proposed, not run)

- Pooled maps as the standard linear reference for FC → FC (and other-7 / SC cells: one map per input type).
- FM-nl with a shared map (nonlinear correction on top of the pooled map), the only route left for a flow gain.
- Multi-source retrain (spec follow-up) starting from the shared form.

Last updated: 2026-10-29
