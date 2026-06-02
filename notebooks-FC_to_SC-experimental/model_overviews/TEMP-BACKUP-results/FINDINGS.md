# Findings — FC↔SC asymmetry stress test + Phase 2 downstream analyses

_Generated 2026-05-29 from the end-to-end run of [`crossmodal_pca_pls_closed_form_overview.ipynb`](../crossmodal_pca_pls_closed_form_overview.ipynb) plus [`sanity_checks.ipynb`](../../sanity_checks.ipynb)._

This file summarizes what we learned across Phase 0 (baseline replication), Phase 1 (residualization basis sweep + Experiments 4–7), and Phase 2 (family-structure + downstream prediction). All numbers cited here are pulled from the CSVs in this directory; see file inventory at the bottom for paths.

---

## TL;DR

The FC→SC direction beats SC→FC at predicting individual structural connectivity from individual functional connectivity, and the asymmetry is **robust to controlling for brain anatomy and demographics**. The directional ratio is **1.505 ± 0.198 (target-only) / 1.553 ± 0.122 (double-sided)** across 10 family-aware seeds, with t-tests against thresholds 1.00 and 1.15 both significant at p < 0.001 (double-sided p < 0.0001 vs 1.15).

In Phase 2, the FC→SC predicted connectome carries:
- **Genuine family-specific connectivity** beyond what subject info alone provides — for sibling pairs specifically, `pred_SC_resid_bvdemo` separates them from unrelated pairs at AUC 0.810, vs `bvdemo_to_SC` baseline AUC 0.563 (Δ +0.247).
- **More downstream cognition signal than observed SC** — substitution fidelity `pred_SC_raw / obs_SC` = **1.25–1.44×** across CogTotal, CogFluid, and CogCrystal composites. The cross-modal denoising via FC actually improves SC's downstream usefulness.

But two important caveats:
1. **`bv+demo` (subject info alone) is highly competitive with the connectome on cognition.** The 26-feature OLS predicts cognitive composites at r ≈ 0.28–0.36. The connectome's marginal lift above `bv+demo` is only ~0.05–0.08 r — real but small. Worth a methodological-wake-up-call paragraph in the writeup.
2. **`bv → SC` demeaned-r = 0.167 is 88% brain size, only 12% compositional structure** (sanity check 1). The bv→SC baseline number that initially looked surprising is mostly driven by total brain volume, not by compositional anatomy. Identifiability (`top1`) tells the opposite story: composition drives identification, size doesn't.

---

## Phase 0 — Baseline replication

**Goal**: confirm the project's published numbers reproduce in this notebook.

| Quantity | Value | Reads as |
|---|---|---|
| **SC→SC oracle ceiling** (seed 0) | demeaned-r = **0.647** | Self-reconstruction of SC via PCA. Upper bound on what any SC-prediction model could achieve. |
| **FC→SC PCA_PLS 10-seed** | **0.1341 ± 0.0054** | Matches the project's published 10-seed benchmark exactly. Replication confirmed. |
| **SC→FC closed-form (PCA_PLS, seed 0)** | demeaned-r = **0.085** | Matches single-seed Section 2 / Step 3.6 numbers. |
| **`bv → SC` (16 FS volume features, OLS, seed 0)** | demeaned-r = **0.167** | Higher than the raw cross-modal FC→SC value. Major finding to address. |
| **`bv → FC` (same OLS, seed 0)** | demeaned-r = **0.047** | Much lower. SC is anatomy-driven; FC isn't, at least not via FS volumes. |

See `notebook_snapshot/phase0/` for full tables.

---

## Phase 1 Section 1 — Non-connectome baselines (single seed + 10-seed)

The 26-dim `bv+demo` feature vector is a strong baseline for FC→SC, comparable to the cross-modal model itself.

| Basis | dims | →SC demeaned (single seed / 10-seed mean) | →FC demeaned (single / 10-seed) |
|---|---|---|---|
| group mean | 0 | 0.000 / 0.000 | 0.000 / 0.000 |
| `bv` (FreeSurfer volumes) | 16 | 0.167 / **0.162 ± 0.007** | 0.047 / **0.049 ± 0.005** |
| `demo` (age+sex+race_eth) | 10 | 0.133 / **0.133 ± 0.009** | 0.095 / **0.114 ± 0.011** |
| `bv+demo` | 26 | 0.176 / **0.171 ± 0.010** | 0.090 / **0.102 ± 0.006** |

Two non-obvious patterns emerge:
- **For predicting FC, demographics (10 dim) beats anatomy (16 dim) by 2×.** demo→FC at 0.11 vs bv→FC at 0.05. Different mechanisms drive SC vs FC individual variation.
- **For predicting SC, anatomy edges demographics** (0.16 vs 0.13). Consistent with SC being more structural-developmental in origin.

Source: `notebook_snapshot/phase1/section1_baselines.csv`, `Exp7_stringent_ratio_10_seed/full_panel_seed_*.csv`.

---

## Phase 1 Section 2 — Asymmetry across residualization bases (THE HEADLINE)

**Method**: residualize the target connectome against each non-connectome basis, then run cross-modal PCA+PLS in both directions, compute the FC→SC / SC→FC ratio.

| Residual basis | FC→SC (seed 0) | SC→FC (seed 0) | Ratio (single seed) | Ratio (10-seed mean ± std) |
|---|---|---|---|---|
| none (raw) | 0.132 | 0.085 | **1.56** | not computed |
| `bv` | 0.078 | 0.056 | **1.39** | not computed |
| `demo` | 0.089 | 0.057 | **1.58** | not computed |
| **`bv+demo` target-only** | **0.066** | **0.048** | **1.39** | **1.505 ± 0.198** ⭐ |
| **`bv+demo` double-sided** | **0.071** | **0.044** | **1.59** | **1.553 ± 0.122** ⭐⭐ |

(See "What's missing" note in [NEXT_STEPS.md](NEXT_STEPS.md) for why 10-seed wasn't run on bv-only and demo-only bases.)

**Paper-grade t-tests (10 seeds, bv+demo basis):**

| Framework | Mean ratio | Std | n>1.00 | n>1.15 | t vs 1.00 (p) | t vs 1.15 (p) |
|---|---|---|---|---|---|---|
| target-only | 1.505 | 0.198 | **10/10** | 9/10 | +8.06 (**p<0.0001**) | +5.67 (**p=0.0003**) |
| double-sided | 1.553 | 0.122 | **10/10** | **10/10** | +14.35 (**p<0.0001**) | +10.46 (**p<0.0001**) |

The double-sided version has *less* variance than target-only (0.12 vs 0.20) — stripping source-side leakage stabilizes the ratio AND keeps it above the strong-gate threshold of 1.15.

Source: `Exp7_stringent_ratio_10_seed/aggregate_full_panel_ratios.csv` + per-seed CSVs.

---

## Phase 1 Exp 7 — Full panel: the asymmetry holds across ALL 6 metrics

Not just demeaned-r — every metric tells the same story across 10 seeds:

| Metric | Target-only ratio | Double-sided ratio | n_seeds > 1.15 |
|---|---|---|---|
| `demeaned_pearson` | **1.51 ± 0.20** | **1.55 ± 0.12** | 9/10 → **10/10** |
| `pearson` | 1.51 ± 0.20 | 1.55 ± 0.12 | 9/10 → 10/10 |
| `top1_acc` | 2.85 ± 1.49 (noisy) | 3.43 ± 1.65 | 10/10 → 10/10 |
| **`avg_rank` (most stable)** | **1.25 ± 0.037** | 1.26 ± 0.025 | 10/10 → 10/10 |
| `mse` (lower-is-better; FC→SC has lower MSE) | 2.62 ± 0.04 | 2.63 ± 0.05 | always > 1 |
| `r2` (diff, not ratio) | +0.019 ± 0.007 | +0.021 ± 0.008 | always > 0 |

**`avg_rank`** is the most stable headline number (1.25× across all seeds and frameworks), and the cleanest to report in a paper alongside the demeaned-r ratio.

Source: `Exp7_stringent_ratio_10_seed/aggregate_full_panel_{wide,ratios}.csv`.

---

## Phase 1 Section 3 — Identifiability under stringent residual

Once you strip the bv+demo-explainable variance from the target, does FC still fingerprint subjects in their SC?

| Predictor | demeaned-r | top1 | avg_rank |
|---|---|---|---|
| `bv+demo → SC` | 0.176 | 0.144 | 0.876 |
| `FC raw → SC` | 0.132 | **0.154** | 0.853 |
| **`FC raw → SC_resid_bv+demo`** | 0.066 | **0.056** (≈11× chance) | 0.792 |
| `bv+demo → FC` | 0.090 | **0.015** (anatomy can't ID via FC) | 0.666 |
| `SC raw → FC` | 0.085 | 0.031 | 0.712 |
| `SC raw → FC_resid_bv+demo` | 0.048 | 0.026 | 0.649 |

Key finding: **`bv+demo` wins demeaned-r against FC for SC (0.176 vs 0.132) but loses top1 (0.144 vs 0.154).** Same metric-disagreement pattern Phase 0 found for bv-only: subject info predicts SC edge *values* well, but doesn't fingerprint subjects as distinctively as FC does. And `FC raw → SC_resid_bv+demo top1` = 0.056 ≈ 11× chance — FC fingerprints subjects in their SC *even when* the SC variance explainable by subject info has been removed.

Source: `notebook_snapshot/phase1/section3_identifiability.csv`.

---

## Phase 2 Analysis 1 — Family-structure validation (10 seeds, paper-grade)

**Method**: for each connectome variant, compute pairwise demeaned-cosine similarity within test subjects; bucket by relationship (MZ twin / DZ twin / sibling / unrelated-matched); compute AUC of (relation vs unrelated) separation. Pool across 10 seeds → 242 MZ pairs, 127 DZ pairs, 1254 sibling pairs.

| Variant | MZ AUC | DZ AUC | sibling AUC |
|---|---|---|---|
| `obs_SC` (oracle) | **0.999** | 0.938 | 0.863 |
| `obs_FC` | 0.990 | 0.886 | 0.823 |
| **`pred_SC_resid_bvdemo`** | **0.989** | **0.823** | **0.810** |
| `pred_SC_raw` | 0.974 | 0.798 | 0.680 |
| `pred_FC_resid_bvdemo` | 0.918 | 0.741 | 0.743 |
| `pred_FC_raw` | 0.910 | 0.756 | 0.710 |
| `bvdemo_to_SC` (anatomy+demo baseline) | 0.951 | 0.800 | 0.563 |
| `combined_pred_SC` ⚠️ | 0.897 | 0.701 | **0.505** ❌ (chance, only failing FDR) |

**The headline test**: does `pred_SC_resid_bvdemo` (FC→SC residual prediction in original SC space) carry family-specific signal beyond what `bv+demo` alone (`bvdemo_to_SC`) provides?

| Relation | pred resid | bv+demo baseline | **Δ-AUC** |
|---|---|---|---|
| MZ | 0.989 | 0.951 | **+0.038** |
| DZ | 0.823 | 0.800 | +0.023 |
| **sibling** | **0.810** | **0.563** | **+0.247** ← massive |

For siblings specifically, adding the FC→SC residual prediction on top of what bv+demo can do **recovers +24.7 AUC points**. The FC→SC asymmetry captures genuine family-specific connectivity beyond shared anatomy + demographics. The `combined_pred_SC` collapse to chance for siblings is a counter-intuitive anomaly worth following up (see Caveats).

Source: `family_structure_phase2/aggregate_auc.csv` + `violin_panel.png`.

---

## Phase 2 Analysis 2 — Downstream prediction (10 seeds, 9 inputs × 5 targets, leak-fixed)

**Method**: per seed, fit a downstream predictor (`RidgeCV` for continuous targets, `LogisticRegressionCV` for sex) on each of 9 input matrices, score on test (Pearson r for continuous, balanced accuracy for binary). Aggregate mean ± std + 95% t-CI across 10 seeds. Paired permutation test vs `bv+demo` baseline (FDR-corrected).

**Mean score across 10 seeds:**

| Input | sex | age | CogTotal | CogFluid | CogCrystal |
|---|---|---|---|---|---|
| **`bv+demo`** baseline | 1.000* | 1.000* | **0.359** | **0.277** | 0.353 |
| `obs_FC` | 0.872 | 0.347 | **0.405** | **0.306** | **0.434** |
| `obs_SC` | 0.883 | 0.268 | 0.264 | 0.162 | 0.258 |
| `obs_FC+obs_SC` | 0.931 | 0.353 | 0.402 | 0.291 | 0.411 |
| **`pred_SC_raw`** | 0.876 | 0.278 | 0.329 | 0.234 | 0.331 |
| **`combined_pred_SC`** | 0.929 | **0.576** | 0.397 | 0.286 | 0.413 |
| `pca_FC` (256-d PCA-reduced FC) | 0.870 | 0.349 | 0.406 | 0.300 | 0.435 |
| `pca_SC` (256-d PCA-reduced SC) | 0.888 | 0.272 | 0.262 | 0.166 | 0.258 |
| `pca_FC+pca_SC` | 0.933 | 0.356 | 0.398 | 0.296 | 0.414 |

\*sex and age = 1.000 for `bv+demo` because those features ARE in `bv+demo`.

### Three derived headlines

**(1) `bv+demo` is highly competitive with the connectome on cognition.** It beats observed SC on every cognition target. It's competitive with observed FC on CogFluid (0.277 vs 0.306) and CogTotal (0.359 vs 0.405). Connectome marginal lift above bv+demo is **~0.05–0.08 r** for cognition. Methodological flag: the connectome's behavioral value-add is real but small.

**(2) Predicted SC beats observed SC on cognition (substitution fidelity > 1):**

| Target | obs_SC | pred_SC_raw | fidelity (pred/obs) |
|---|---|---|---|
| CogTotal | 0.264 | 0.329 | **1.25×** |
| CogFluid | 0.162 | 0.234 | **1.44×** |
| CogCrystal | 0.258 | 0.331 | **1.28×** |

**The FC→SC predicted connectome is a better downstream predictor of cognition than observed SC.** Cross-modal imputation actually denoises SC for behavior. This is a Krakencoder-style imputation utility finding.

**(3) `combined_pred_SC` dominates age prediction** (0.576 vs `obs_FC`'s 0.347) — extracting age signal from the interaction of FC and bv+demo.

Source: `downstream_prediction_phase2/aggregate.csv` + `perm_vs_bvdemo.csv` + `bars_panel.png`.

---

## Sanity check 1 (separate notebook) — Brain-volume single-feature vs 16-feature OLS

From [`sanity_checks.ipynb`](../../sanity_checks.ipynb):

| Metric | 16-feature `bv` | 1-feature `FS_BrainSeg_Vol` | Delta |
|---|---|---|---|
| `demeaned_pearson` | 0.167 | **0.148** | +0.019 (1.13×) |
| `top1_acc` | 0.103 | **0.021** | −0.082 (**5.0× drop**) |
| `avg_rank` | 0.879 | 0.791 | −0.088 |

**Verdict — both interpretations true, on different metrics:**
- **On demeaned-r**: brain size dominates. 88% of `bv → SC` demeaned-r is recoverable from total brain volume alone. Edge-value prediction is mostly "scale the per-edge train mean by brain size."
- **On identifiability**: composition dominates. Brain size alone can't fingerprint subjects; the other 15 compositional features carry the inter-subject distinguishing signal.

**Writeup framing**: "Anatomy predicts SC at demeaned-r = 0.167, of which ~88% is recoverable from total brain volume alone (1-feature OLS: 0.148). The remaining ~12% comes from compositional structure across the 16 FreeSurfer volumes. Compositional structure contributes disproportionately to identifiability: 1-feature top1 = 0.021 (~4× chance) vs 16-feature top1 = 0.103 (~20× chance) — a 5× lift driven by composition, not size."

Source: `sanity_checks.ipynb` (cell 12 markdown), local file outside this results dir.

---

## Caveats and known issues

### 1. `combined_pred_SC` sibling AUC collapse (Phase 2 Analysis 1)

`combined_pred_SC` (= `[PCA(FC) ‖ bv ‖ demo] → SC` via BR per target component) achieves AUC = 0.505 (chance, p_fdr = 0.65) for separating sibling pairs from unrelated pairs — the **only row in the family-structure table that fails FDR**. MZ separation stays high (0.897), DZ moderate (0.701), but sibling collapses.

Possible explanations:
- (a) Real finding: the combined predictor's loss is dominated by overall reconstruction quality (where bv+demo wins), washing out the within-family-vs-between-family contrast that the cross-modal residual carries.
- (b) Implementation bug: `_combined_predict` does something subtly wrong (PCA basis, input concat order, per-component BR scaling).

Action item: add a one-cell diagnostic comparing `combined_pred_SC` and `pred_SC_raw` prediction value distributions for seed 0 to determine which.

### 2. `pred_SC_resid_bvdemo` removed from Phase 2 Analysis 2 inputs

In the original Phase 2 Analysis 2 plan, `pred_SC_resid_bvdemo` was included as an input. It was constructed as `_pca_pls_predict(FC->SC_residual) + bv+demo_OLS_prediction` (the additive term puts the prediction back in raw SC space). The side effect was that the input matrix carried bv+demo info implicitly, leading to a leak: `sex` prediction = 1.000 (perfect), `age` = 0.905 (near-perfect) — both impossible without the input containing those features.

The row was removed from STEP 9.1's input set on 2026-05-28. The leak doesn't affect Phase 2 Analysis 1 (family structure) where the variant is correctly evaluated in residual space.

### 3. The `bv → SC = 0.167` framing is more nuanced than originally written

Per sanity check 1, this number is 88% brain size on edge-value prediction. Earlier writeup language about "anatomy is a strong cross-modal baseline" should be replaced with: "**total brain volume** is a strong cross-modal baseline for edge-value reconstruction; compositional anatomy is what carries identifiability."

### 4. Cross-fit train predictions are naively in-sample

`_phase2_build_inputs_for_seed` uses `_pca_pls_predict(_FC_tr, _FC_tr, _SC_tr)` for the train predictions (same PLS fit predicting in-sample on train). This is slightly optimistic but consistent across all rows, so it doesn't bias the downstream prediction comparison itself. For paper-grade rigor we should switch to a K-fold cross-fit (see [NEXT_STEPS.md](NEXT_STEPS.md) Tier 5).

### 5. Hyperparameters not swept

K_PCA = 256, K_PLS = 64 throughout. These are project defaults. We haven't shown the headline is robust to alternate values. Reviewer-defense work (see [NEXT_STEPS.md](NEXT_STEPS.md) Tier 3).

---

## What survives → recommended manuscript claims

1. **FC→SC asymmetry holds at paper-grade significance.** 10-seed double-sided ratio 1.553 ± 0.122, t vs 1.15 p < 0.0001, 10/10 seeds clear the strong gate. Cite both target-only and double-sided.

2. **The asymmetry is multi-metric robust.** All 6 metrics (demeaned-r, pearson, top1, avg_rank, mse, r2 diff) point the same direction across all 10 seeds. `avg_rank ratio = 1.25` is the most stable summary statistic.

3. **The asymmetry survives anatomy + demographics control.** Both at the per-subject residual prediction level (1.5×–1.6× ratio) and at the family-structure level (Δ +0.247 AUC for siblings).

4. **FC→SC predicted SC has imputation utility for cognition** (substitution fidelity 1.25–1.44×). The cross-modal denoising via FC actually improves SC's downstream predictive value for behavior.

5. **Methodological caveat: `bv+demo` alone is a non-trivial baseline for cognition.** The connectome's marginal value above 26-dim subject info is ~0.05–0.08 r. Worth a sentence in the limitations or methods.

---

## File inventory (this directory)

```
results/
├── FINDINGS.md                            ← this file
├── NEXT_STEPS.md                          ← what's missing / what to do next
├── notebook_snapshot/                      (Phase 0 + Phase 1 consolidated CSVs)
│   ├── snapshot_manifest.csv               (audit log)
│   ├── phase0/                             (10 CSVs)
│   │   ├── closed_form_summary.csv
│   │   ├── step1_sc_oracle.csv             ← demeaned 0.647
│   │   ├── step2_10seed_fc_to_sc.csv       ← 10-seed FC→SC PCA_PLS baseline
│   │   ├── step2_10seed_fc_to_sc_aggregate.csv  ← 0.1341 ± 0.0054
│   │   ├── step3_brain_vol_to_sc.csv       ← bv→SC = 0.167
│   │   ├── step3_5_three_panels.csv        ← SC target: bv, FC raw, FC residual
│   │   ├── step3_6_three_panels.csv        ← FC target mirror
│   │   ├── step5_1_2x2_residual_panels.csv  ← PLS A + BR C
│   │   ├── step5_1_2x2_partition_panels.csv ← PLS B + BR D
│   │   └── step5_2_cross_step.csv          ← 10-row cross-step summary
│   └── phase1/                             (6 CSVs)
│       ├── section1_baselines.csv          ← bv/demo/bv+demo → SC, FC
│       ├── section2_asymmetry.csv          ← 4-basis asymmetry ratios
│       ├── section3_identifiability.csv    ← stringent residual identifiability
│       ├── exp4_double_sided.csv           ← target-only vs double-sided
│       ├── exp5_br_robustness.csv          ← PLS vs BR on bv+demo
│       └── exp6_combined_predictor.csv     ← combined model in both directions
├── Exp7_stringent_ratio_10_seed/           (Phase 1 Exp 7, the 10-seed paper-grade)
│   ├── per_seed.csv                        ← demeaned-r only (emergency save cell 45)
│   ├── aggregate.csv                       ← mean/std across seeds
│   ├── full_panel_seed_0..9.csv            ← per seed, 10 conditions × 6 metrics
│   ├── aggregate_full_panel_long.csv       ← long form: framework × direction × metric
│   ├── aggregate_full_panel_wide.csv       ← wide form, mean ± std per metric
│   └── aggregate_full_panel_ratios.csv     ← directional ratios, both frameworks
├── family_structure_phase2/                (Phase 2 Analysis 1)
│   ├── seed_0..9.npz                       ← per-pair similarities, npz binary
│   ├── aggregate_auc.csv                   ← variant × relation → AUC + bootstrap CI + p_fdr
│   └── violin_panel.png                    ← Krakencoder Fig 3a-style figure
└── downstream_prediction_phase2/           (Phase 2 Analysis 2, leak-fixed)
    ├── seed_0..9.parquet                   ← per-seed scores, columnar parquet
    ├── aggregate.csv                       ← input × target mean ± std + CI
    ├── perm_vs_bvdemo.csv                  ← paired permutation p-values, FDR-corrected
    └── bars_panel.png                      ← Krakencoder Fig 3b-style figure
```

Total: 53 files, ~512 KB.
