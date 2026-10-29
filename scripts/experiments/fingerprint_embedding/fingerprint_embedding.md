# fingerprint_embedding — is E4's pooled map a useful subject embedding? (spec v4 E6)

**Question.** E4's best cross-condition translator is one map shared by all FC condition pairs, which turns out to be a
reliability-weighted fingerprint filter (`../cross_condition_translation/pooled_map.md`). Does this embedding hold up
across days, is it driven by confounds, does it carry behaviour, and is the fingerprint a better target for SC than any
single FC condition or the CCA route?

**Setup.** As E4: matched cohort (917 Glasser subjects), family-aware E2.1 splits, seeds 0–4, $k$ = 256 pooled block-PCA
codes. The map $B$ is fit on the train subjects' 8 conditions (combined rest + 7 tasks), never on the rest sessions.
Test split reported (mean ± SE over seeds); "paired" = seed-paired difference, support rule as in v4.

- **Embedding:** a subject's codes in one condition times $B$. **Fingerprint:** the subject's mean codes over the 8
  conditions. **Fingerprint embedding:** the fingerprint times $B$.
- $B$ matches its closed form $B = \tfrac{8}{7}W - \tfrac{1}{7}I$, $W = (8S_f + S_d)^{-1} 8S_f$, to $10^{-15}$ in every
  seed; 55–56 of the 256 code directions have fingerprint share $w > 0.5$ (`figures/fig_w_spectrum.png`).

**Code.** `fe_common.py` (map, closed form, helpers), `build_confounds.py` (proxies), `run_checks.py` (one seed, all four
checks), `analyze.py`, `launch/run_checks.sh`. Jobs 21344818 (confounds), 21345170 (checks). Tables `tables/`
(`summary.md`), figures `figures/`.

## 1. Across days (rest session 1 ↔ session 2)

| Prediction of the other session | Average rank | Top-1 | Demeaned $r$ |
|---|---|---|---|
| copy (codes) | 0.967 | 0.79 | 0.374 |
| copy (edges) | 0.989 | 0.95 | 0.494 |
| embedding | **0.994** | 0.89 | 0.294 |

The weights learned across conditions transfer across days: in the same 256 codes the embedding raises top-1 from 0.79
to 0.89 (+0.10, 5 / 5 seeds) and average rank to the highest value (+0.005 over raw edges, 4 / 5, not supported). Raw
edges still find the exact subject more often (top-1 0.95): rest carries rest-specific individual structure that a map
fit to be shared with the tasks discards, which also lowers demeaned $r$.

## 2. Confounds

No framewise displacement in our data; proxies (`build_confounds.py`): DVARS and global-signal SD from the rest
timeseries, mean rest FC, intracranial volume, total parcel volume.

| Representation | R² explained by the proxies (test) |
|---|---|
| rest codes | 0.143 |
| rest embedding | **0.048** |
| fingerprint | 0.039 |
| fingerprint embedding | 0.054 |

| Cross-condition identification (56 pairs, code space) | Average rank | Top-1 |
|---|---|---|
| codes | 0.832 | 0.15 |
| codes, proxies removed | 0.830 | 0.15 |
| embedding | 0.965 | 0.54 |
| embedding, proxies removed | 0.962 | 0.53 |

The embedding is less tied to the proxies than the rest codes (R² 0.14 → 0.05), and removing them barely changes its
identification. Within the limits of proxies, identifiability is not carried by motion or head size. (Embedding vs
embedding identifies better than E4's prediction vs raw target: top-1 0.54 vs 0.32.)

## 3. Behaviour

Ridge (penalty on val) to HCP cognition, test $r$:

| Target | rest edges | rest codes | rest embedding | fingerprint | fingerprint embedding |
|---|---|---|---|---|---|
| Total composite | 0.47 | 0.42 | 0.37 | **0.50** | 0.49 |
| Fluid composite | 0.30 | 0.28 | 0.25 | **0.39** | 0.36 |
| Crystallised composite | **0.51** | 0.48 | 0.41 | 0.49 | 0.46 |
| PMAT | 0.35 | 0.29 | 0.28 | **0.39** | 0.37 |

- **The embedding does not carry more behaviour.** The rest embedding is below rest codes on every score, and the
  fingerprint embedding is below the plain fingerprint on every score (−0.01 to −0.03; 0 / 5 seeds for crystallised
  and PMAT). Identifiable directions and behaviour-relevant directions are not the same.
- **Averaging over conditions helps somewhat** (fingerprint vs rest edges: fluid +0.09, PMAT +0.03, total +0.03; 3–4 / 5
  seeds, none supported at 5 seeds given the seed spread). This is mostly more scan time per subject (8 conditions vs
  rest), in line with task FC aiding behaviour prediction.

## 4. SC → fingerprint

Ridge from SC codes (size and penalty on val average rank against measured rest FC) with three training targets, plus
E4's SC → condition ridge (B1) and multiview CCA (B2):

| SC → … | Own target R² | vs measured rest FC: average rank / top-1 / demeaned $r$ | vs all 8 conditions: average rank / top-1 / demeaned $r$ |
|---|---|---|---|
| rest FC (ridge) | 0.010 | 0.726 / 0.041 / 0.091 | 0.662 / 0.025 / 0.038 |
| each condition (E4 ridge) | – | 0.726 / 0.041 / 0.092 | 0.661 / 0.018 / 0.044 |
| each condition (E4 multiview CCA) | – | 0.719 / 0.056 / 0.097 | 0.661 / 0.029 / 0.046 |
| fingerprint | 0.021 | 0.800 / 0.061 / 0.089 | 0.722 / 0.037 / 0.048 |
| **fingerprint embedding** | **0.073** | **0.822 / 0.067 / 0.096** | **0.744 / 0.043 / 0.052** |

- **SC predicts the fingerprint far better than any single condition:** 7× the explained variance of rest FC codes
  (R² 0.073 vs 0.010).
- **Training SC on the fingerprint makes its predictions of measured rest FC more individual than training on rest
  itself:** average rank +0.097 (5 / 5), top-1 +0.026 (5 / 5); demeaned $r$ unchanged (+0.005, 3 / 5). The same holds
  against every condition (0.744 vs 0.661) and beats the CCA route (0.719).
- **Reading:** SC's subject-specific information is aligned with the stable, condition-invariant part of FC; a single
  condition's FC buries it under condition-specific variance. The gain is in identification, not in edge-level accuracy.

Figures: `fig_retest.png`, `fig_confounds.png`, `fig_behaviour.png`, `fig_sc_targets.png`, `fig_w_spectrum.png`.

## Takeaways

1. The pooled map is a real, reasonably clean fingerprint filter: it transfers across days and is not explained by the
   available motion / head-size proxies.
2. It is not a better behavioural representation; what is reliable across conditions is not what predicts cognition.
3. It is a better target for SC: SC → fingerprint is the most individual SC → FC route so far (average rank vs rest FC
   0.82 vs 0.72 for every earlier route).

**Caveats.** Motion only through proxies (no FD); behaviour on 5 family-aware splits of ~180 test subjects (wide
spread); the fingerprint uses 8 conditions, so it has more scan time than any single condition.

## Next (proposed)

- SC → fingerprint as the target for the SC → FC models (E2.1's flows, nonlinear correctors) and for SC → FC
  identification claims.
- Within-subject state structure: predict the condition-specific part $d^t$ (FC minus the fingerprint) as the real
  cross-condition translation problem (E4 metrics do not reward it).

Last updated: 2026-10-29
