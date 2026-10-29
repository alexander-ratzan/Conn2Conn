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
checks), `analyze.py`, `r2_by_direction.py`, `launch/run_checks.sh`. Jobs 21344818 (confounds), 21347437 (checks),
21347847 (R² by direction). Tables `tables/`
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

## 4. SC → rest FC through five target spaces

Every route takes a test subject's **SC only** and outputs **predicted rest FC edges**; task FC enters only through the
training targets (train subjects). Settings for every route are chosen on val average rank against measured rest FC.
Diagram: `figures/fig_routes_schematic.png`.

| Route | Group | What SC is trained to predict | Decoded to rest FC by |
|---|---|---|---|
| 1 Rest codes | single condition | the subject's rest FC (block-PCA codes), ridge | block-PCA decoder |
| 2 CCA, SC as a view | CCA latents | nothing regressed: CCA over SC + 8 FC conditions encodes SC into its shared latent | CCA rest decoder |
| 3 CCA consensus (E1) | CCA latents | mean of the subject's 8 CCA encodings (CCA over the 8 FC conditions, no SC), ridge | CCA rest decoder |
| 4 Mean fingerprint | block-PCA fingerprints | mean of the subject's 8 conditions' block-PCA codes, ridge | block-PCA decoder |
| 5 Fingerprint embedding | block-PCA fingerprints | the mean fingerprint × pooled map (reliability-weighted), ridge | block-PCA decoder |

Two evaluations (`figures/fig_sc_routes.png`):

| Route | Stage 1: vs own target, average rank / top-1 | Stage 2: vs measured rest FC, average rank / top-1 / demeaned $r$ | Ceiling (true target decoded): average rank / demeaned $r$ |
|---|---|---|---|
| 1 Rest codes | 0.720 / 0.038 | 0.726 / 0.041 / 0.091 | 1.000 / 0.678 |
| 2 CCA, SC as a view | 0.802 / 0.042 | 0.732 / 0.036 / 0.100 | 0.996 / 0.459 |
| 3 CCA consensus | 0.891 / 0.129 | 0.749 / 0.037 / 0.101 | 0.985 / 0.351 |
| 4 Mean fingerprint | 0.825 / 0.096 | 0.800 / 0.061 / 0.089 | 1.000 / 0.421 |
| **5 Fingerprint embedding** | 0.852 / 0.122 | **0.822 / 0.067** / 0.096 | 0.998 / 0.339 |

(Route 3 at E1's fixed latent, d 32, λ 0.1: stage 2 0.741 / 0.028 / 0.100. E4's SC-only ridge cell reproduces route 1.)

- **Stage 1 (pairing):** SC pairs most strongly with the fingerprint-type targets (routes 3–5) and least with rest FC
  itself. Stage 1 is not a fair ranking: each route has its own target.
- **Stage 2 (the common readout):** the block-PCA fingerprints make SC's predictions of measured rest FC the most
  individual: route 5 average rank 0.822 vs 0.726 for route 1 (+0.097, 5 / 5 seeds) and 0.749 for the CCA consensus.
  Demeaned $r$ is the same for every route (0.09–0.10): the gain is identification, not edge accuracy. The same order
  holds against every condition (`fig_sc_routes_conditions.png`; route 5 0.744 vs 0.662 averaged over the 8).
- **Ceilings:** every true target identifies the subject's rest FC almost perfectly (0.97–1.00), so the routes differ in
  how much of their target SC can predict, not in what the target could deliver. The CCA consensus pairs best with SC
  in its own space (0.89) but loses most of it when decoded to rest FC (0.75); the block-PCA fingerprints keep it.

**Why: SC predicts the fingerprint directions of FC** (`figures/fig_r2_by_direction.png`). Rotating the codes into the
pooled map's directions and computing SC's held-out R² per direction: in directions where the fingerprint dominates
(w > 0.5) SC explains R² ≈ 0.03 (up to 0.08 in the most fingerprint-like bin), in condition-specific directions
(w < 0.2) R² ≈ 0. This holds **when SC is trained on plain rest FC** (route 1), so it is a property of SC–FC coupling,
not of the training target. The fingerprint routes win because their target contains only the part SC can predict;
a rest target adds condition-specific variance SC cannot reach. (Each route's R² on its own target: route 1 0.010,
3 0.019, 4 0.021, 5 0.073; not comparable across routes.)

Figures (`figures/`): `fig_routes_schematic`, `fig_sc_routes` (main), `fig_sc_routes_conditions`, `fig_r2_by_direction`,
`fig_retest`, `fig_confounds`, `fig_behaviour`, `fig_w_spectrum`.

## Takeaways

1. The pooled map is a real, reasonably clean fingerprint filter: it transfers across days and is not explained by the
   available motion / head-size proxies.
2. It is not a better behavioural representation; what is reliable across conditions is not what predicts cognition.
3. It is a better target for SC: training SC on the fingerprint embedding gives the most individual predictions of
   measured rest FC so far (average rank 0.822 vs 0.726 trained on rest, 0.732 / 0.749 for the CCA routes), because SC
   only predicts the fingerprint directions of FC, whatever it is trained on.

**Caveats.** Motion only through proxies (no FD); behaviour on 5 family-aware splits of ~180 test subjects (wide
spread); the fingerprint uses 8 conditions, so it has more scan time than any single condition.

## Next (proposed)

- SC → fingerprint as the target for the SC → FC models (E2.1's flows, nonlinear correctors) and for SC → FC
  identification claims.
- Within-subject state structure: predict the condition-specific part $d^t$ (FC minus the fingerprint) as the real
  cross-condition translation problem (E4 metrics do not reward it).

Last updated: 2026-10-29
