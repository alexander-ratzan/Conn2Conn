# Tractography-predict — findings

**Run**: 6 experiments in parallel (`:ro` overlay), 10 seeds each, Glasser, family-aware
splits. Jobs 10688656 (e1, 16 cpu) + 10689496–500 (e2–retest, 4 cpu under the 32-core
per-user QOS cap). All metrics reported, not just demeaned_pearson.

## TL;DR

**Richer tractography does not help.** The named-bundle representation (`r2t`) predicts
FC *worse* than raw streamline counts, adds nothing marginal on top of counts, and
carries no downstream cognition signal above demographics. Meanwhile the FC↔SC
asymmetry is robust across all 6 metrics and *amplifies* in the bundle representation.
Downstream, **FC remains the only modality that beats the bv+demo cognition floor** —
no structural or tractography representation (count, bundle, bundle-similarity, or
synthetic-FC-from-tractography) clears that bar.

This is a clean negative result for the "parcellated counts lose information that
richer tractography would recover" hypothesis, and a strong positive for FC's unique
downstream value.

## E1 — predicting FC from each structural representation (median, 10 seeds)

| rep | demeaned_pearson | pearson | top1_acc | avg_rank | mse | r2 |
|---|---|---|---|---|---|---|
| **SC** (count baseline) | **0.0847** | 0.8292 | 0.0513 | 0.7128 | 0.0143 | −0.057 |
| kitchen_sink [SC‖r2t‖bv‖demo] | 0.0942 | 0.8288 | 0.0487 | 0.7228 | 0.0143 | −0.057 |
| SC_r2t [SC‖r2t] | 0.0791 | 0.8272 | 0.0359 | 0.7020 | 0.0145 | −0.064 |
| r2t (bundle) | 0.0493 | 0.8153 | 0.0205 | 0.6090 | 0.0154 | −0.127 |
| r2t_corr (bundle-similarity) | 0.0362 | 0.8231 | 0.0179 | 0.6070 | 0.0148 | −0.091 |

- **Count-SC beats bundle-r2t** for FC prediction on every metric. The bundle profile is
  a *lossier* FC predictor than raw counts.
- **kitchen_sink edges SC** (0.0942 vs 0.0847 demeaned_pearson; best avg_rank) — but the
  gain is ~11% and comes from bv+demo, not r2t (see E3).
- Combined-rep rows use **per-block PCA** (PCA each block to its own 256-dim latent, then
  concat) to fix a scale-domination bug where the naive single-PCA on concatenated
  features was dominated entirely by r2t's magnitude, making SC/bv/demo invisible.

## E2 — FC↔SC asymmetry across representations, ALL 6 metrics

median FC-wins (ratio>1, or r2 as difference>0, means FC→X beats X→FC):

| rep | demeaned_pearson | pearson | top1_acc | avg_rank | mse | r2(diff) |
|---|---|---|---|---|---|---|
| **SC** | +1.62× ✓ | +1.10× ✓ | +2.56× ✓ | +1.23× ✓ | +2.59× ✓ | +0.036 ✓ |
| **r2t** | +2.26× ✓ | +1.04× ✓ | +1.50× ✓ | +1.19× ✓ | 0.00 ✗ | +0.45 ✓ |
| r2t_corr | +1.63× ✓ | 0.98× ✗ | +1.83× ~ | +1.18× ✓ | 0.85× ✗ | −6.9 ✗ |

(✓ = Wilcoxon one-sided p<0.05 that FC wins; all 10 seeds.)

- **SC asymmetry is bullet-proof: FC→SC > SC→FC on all 6 metrics, every one p=0.001.**
  This is the strongest multi-metric statement of the project's headline asymmetry.
- **r2t amplifies the asymmetry on the "shape/identifiability" metrics**
  (demeaned_pearson 2.26× vs SC's 1.62×) but FAILS on mse (ratio 0.00) — i.e. FC→r2t and
  r2t→FC have similar raw error, the asymmetry lives in the demeaned/rank structure, not
  raw magnitude. Honest nuance: the r2t asymmetry is metric-dependent.
- r2t_corr is mixed (fails pearson/mse/r2); not a clean asymmetry carrier.

## E3 — marginal contribution of r2t over SC (paired Δ, 10 seeds)

- median Δ (SC_r2t − SC) demeaned_pearson = **−0.0013** (essentially zero / slightly
  negative); one-sided Wilcoxon p(greater)=0.98 (NOT an improvement), two-sided p=0.049
  (SC_r2t is marginally *worse*).
- **Verdict: count-SC is a sufficient statistic for cross-modal prediction.** Adding the
  bundle representation does not add FC-predictive signal — it slightly hurts (extra
  latents, no new information).

## E5 — downstream cognition (the real metric)

Median test Pearson predicting NIH-Toolbox composites, and lift over the bv+demo floor:

| rep | CogTotal | CogFluid | CogCrystal | lift over bv+demo (Total) |
|---|---|---|---|---|
| **FC** | **0.451** | **0.342** | **0.451** | **+0.078** ✓ |
| bv+demo (floor) | 0.372 | 0.309 | 0.346 | 0 |
| SC | 0.261 | 0.180 | 0.267 | **−0.111** |
| r2t→synthFC (substitution) | 0.202 | 0.197 | 0.180 | −0.170 |
| r2t | 0.196 | 0.130 | 0.164 | −0.176 |
| SC_r2t | 0.196 | 0.135 | 0.160 | −0.177 |
| r2t_corr | 0.151 | 0.157 | 0.103 | −0.221 |

**This is the decisive result:**
- **FC is the ONLY representation that beats the bv+demo floor** (+0.078 total, +0.105
  crystallized, +0.033 fluid). After residualizing demographics, FC retains 0.37
  crystallized / 0.22 fluid; every structural rep collapses to ~0–0.08.
- **Every tractography representation falls BELOW the demographic floor** — count-SC,
  bundle-r2t, bundle-similarity, and the combined rep all carry *less* cognition signal
  than age+sex+brain-volume alone.
- **The substitution chain fails**: `r2t → synthetic-FC → cognition` (raw 0.18–0.20)
  does NOT recover FC's cognition signal. Synthetic FC generated from tractography is
  not a useful cognitive biomarker — the cognition-predictive structure of real FC is
  not reconstructable from tractography.

This extends the project's prior "FC carries non-demographic cognition signal, SC
doesn't" finding: **richer tractography does not rescue SC**, and you cannot launder
tractography into FC's cognition signal via cross-modal prediction.

## E4 — does SC-PC3 (the dorsal-stream backbone) exist in the bundle representation?

- r2t PC modes 1–3 are stable across seeds (median |cos| 0.99 / 0.98 / 0.95); modes 4+
  degrade.
- **SC-PC3 does NOT map cleanly to any single r2t mode** (best Spearman of subject scores
  = −0.32 with r2t-mode 10; everything else |ρ|<0.23). The dorsal visual-stream / DAN
  structural backbone (Depth 1.1) is a **count-SC phenomenon**, not cleanly present in
  the named-bundle decomposition. The bundle atlas (66 tracts) doesn't carve the cortex
  finely enough to express that intra-parietal/occipital edge pattern.

## Retest reliability (FC scan-rescan, independent cross-check)

- FC reliability adds essentially nothing beyond the strength+distance proxy: joint
  R²(|PC3| ~ strength+distance+FC-reliability) = **41.2%** vs strength+distance-only
  41.0%.
- PC3 visual/DAN localization survives the 3-way partial: visual||visual **13.3×**,
  DAN||DAN **5.7×**, DAN||visual **5.0×** — identical to the strength+distance result.
- Independent confirmation that the PC3 localization is not a reliability artifact.
  (Caveat unchanged: FC reliability ≠ SC reliability; gold-standard SC ICC would need
  the HCP retest dMRI release.)

## Bottom line for the writeup

1. **The asymmetry is real and multi-metric** — FC→SC beats SC→FC on all 6 metrics at
   p=0.001 across 10 seeds. Strongest statement yet.
2. **Tractography richness is a dead end** — the bundle representation predicts FC worse
   than counts, adds nothing marginal, carries no cognition signal, and doesn't contain
   the PC3 backbone. The parcellated-count "information loss" hypothesis is falsified for
   this data: counts are a sufficient statistic.
3. **FC's downstream uniqueness is confirmed and sharpened** — FC is the only modality
   above the demographic cognition floor; synthetic-FC-from-tractography does not recover
   it. Cognition prediction is an FC story, full stop.

## Files

- `e1_source_rep_results.csv`, `e2_asymmetry_results.csv` + `_summary.csv`,
  `e3_marginal_results.csv` + `_summary.csv`, `e4_*` (3 CSVs),
  `e5_downstream_results.csv` + `_summary.csv`
- `tractography_synthesis.csv`, `synthesize_tractography_output.txt`
- `../sanity_checks/tract_check/retest_icc_results/` (FC reliability + 3-proxy enrichment)
- scripts: `e1`–`e5`, `synthesize_tractography.py`, `_tract_setup.py`,
  `run_one.sbatch` (parameterized, `:ro` overlay for parallel runs)
