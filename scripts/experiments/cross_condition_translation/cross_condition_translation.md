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
`launch/analyze_baselines.sh`.

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

Related EDA (2026-10-28): `scripts/notebooks/EDA/fc_deviation_similarity.py` — individual FC deviations are weakly
reliable within a task (LR vs RL run r 0.045–0.15; full-length reliability 0.09–0.26; rest 0.66), so cross-condition
agreement is low (0.06–0.19) but, corrected for that unreliability, about half of the individual pattern is shared
(disattenuated 0.41–0.77, median 0.51).

## Next

E4.1: review with the user before the flows (E4.2–E4.4).

Last updated: 2026-10-28
