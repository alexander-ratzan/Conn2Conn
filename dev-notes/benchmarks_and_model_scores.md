# Benchmarks & model scores

A reference for (1) every evaluation metric the project uses — what it measures, why it exists, how to read it — and (2) how each model scores on each one.

Metric implementations: [`models/eval/metrics.py`](../models/eval/metrics.py), plotting/aggregation in [`models/eval/evaluator_viz_markdown.py`](../models/eval/evaluator_viz_markdown.py).

---

# Part 1 — The benchmark suite

All metrics are computed on **held-out test subjects** (n=195 of the 957, family-aware split). Predictions are FC edge-vectors (upper triangle); "true"/eFC is the empirical FC. Read every model **between the null floor and the oracle ceiling** (Part 3).

## 1. Raw Pearson correlation
- **Code:** `compute_basic_regression_metrics` → mean of `diag(compute_corr_matrix(true, pred))` (`metrics.py:182`).
- **What:** average per-subject correlation between predicted and true FC edge-vectors.
- **Gotcha:** dominated by the **group-average FC** — every subject's FC looks ~85% like everyone else's, so a model that just reproduces the mean scores ~0.83. *Looks impressive, says little.* Never use it alone.

## 2. R²
- **Code:** `r2_score(...).mean()`.
- **What:** fraction of per-edge variance explained vs. predicting each edge's mean.
- **Gotcha:** here it is **near zero or negative**. Negative R² = the model does *worse than predicting each edge's training mean* on held-out subjects — i.e. the individual edge-level signal is essentially unpredictable from SC.

## 3. Demeaned Pearson — THE HEADLINE
- **Code:** `compute_demeaned_pearson_r(pred, true, target_train_mean)` (`metrics.py:29`): subtract the **training** group-mean from both, then correlate.
- **What:** correlation of the **subject-specific deviations** only. This is the honest individual-signal metric and the number the whole project lives or dies by.
- **Note:** uses the *train* mean (not test) to avoid leakage.

## 4. Average rank percentile
- **Code:** `corr_avg_rank` (`metrics.py:96`).
- **What:** for each subject, where their true FC ranks (by similarity) among all predictions. **1.0 = perfect, 0.5 = chance.** A soft identifiability measure.

## 5. Top-1 accuracy
- **Code:** `corr_topn_accuracy(topn=1)` (`metrics.py:81`).
- **What:** fraction of subjects whose true FC is the single best match to their prediction. Greedy and per-row (one prediction can be the best match for several subjects). Chance ≈ 1/195 ≈ 0.005.

## 6. Identifiability (intra vs inter, Cohen's d, t-test)
- **Code:** `compute_identifiability` (`metrics.py:142`).
- **What:** `r_intra` = corr(pred_i, true_i); `r_inter` = mean corr(pred_i, true_{j≠i}). A one-sample t-test on `d = r_intra − r_inter` gives a p-value; **Cohen's d** is the effect size of "you look more like yourself than others."
- **How to read:** computed both raw and demeaned. The *demeaned* Cohen's d is the meaningful one (raw is inflated by the shared group mean). This is the Zalesky-style test — the field's reality is small effect sizes.

## 7. Hungarian matching (one-to-one)
- **Code:** `hungarian_matching` (`metrics.py:200`), `scipy.optimize.linear_sum_assignment`.
- **What:** the **optimal one-to-one assignment** of predictions to subjects that maximizes total similarity; accuracy = fraction assigned to themselves. Stricter than top-1 because it forces a *bijection* (each prediction used exactly once). Answers "if I have N scans and N identities, can I line them all up?"
- Reported raw and demeaned.

## 8. Hungarian sample-size sweep (+ nulls + FDR stars)
- **Code:** `plot_hungarian_sample_size_analysis` (`evaluator_viz_markdown.py:1644`).
- **What:** repeats Hungarian matching (M=2500 random subsets) for increasing cohort sizes n. Plots mean accuracy vs n for three lines:
  - **pFC** — real predictions vs true.
  - **Null (noise)** — predictions replaced by `train_mean + Gaussian·train_std` (zero individual info).
  - **Null (permute)** — real similarity matrix with columns permuted (empirical chance).
- A `*` is drawn at each n where pFC beats **both** nulls (two-sample t-test, Benjamini–Hochberg FDR-corrected, α=0.05).
- **How to read:** chance ≈ 1/n, so all lines fall as n grows; what matters is the *gap* between pFC and the nulls, and how far out the stars persist.

## 9. Reference bounds (always cite alongside any model)
- **Null floor:** `CrossModalPCA` with no cross-map / mean+noise / permute → what zero individual signal looks like.
- **Oracle ceilings:**
  - **FC→FC self-reconstruction** (PCA autoencoder, FC in → FC out): demeaned ≈ **0.69**, top-1 = **1.0**. The most a perfect within-modality model achieves.
  - **Test–retest** (predict session 2 FC from session 1 FC): demeaned ≈ **0.50**, top-1 ≈ 0.96 *(single-run smoke estimate)*. The biological reliability ceiling — no SC→FC model can exceed how reproducible FC is with itself.

---

# Part 2 — How each model scores

**Provenance:** numbers below are **10-seed aggregates (mean ± std)** scraped from W&B, best trial per (model, seed) selected by `val_demeaned_r`, evaluated on the test split (n=195), Glasser parcellation, SC→FC. Sources: [`scrape_SCtype_results.ipynb`](../notebooks/results_scrape/scrape_SCtype_results.ipynb) and [`scrape_covtype_results.ipynb`](../notebooks/results_scrape/scrape_covtype_results.ipynb). Krakencoder benchmark detail anchored to [`results/local_results/Krakencoder_precomputed/final/metrics_final.json`](../results/local_results/Krakencoder_precomputed/final/metrics_final.json).

## 2.1 Main comparison — all models, SC input (10 seeds)

Sorted by demeaned Pearson (the headline), best first:

| Model | Demeaned r | Raw r | MSE | Avg rank | Top-1 |
|---|---|---|---|---|---|
| **CovProjector + demographics** | **0.1031 ± 0.0103** | 0.8353 ± 0.0057 | 0.0137 | 0.7252 | 0.0369 |
| CovProjector + fs_all_demo | 0.0967 ± 0.0098 | 0.8345 ± 0.0047 | 0.0138 | 0.7073 | 0.0461 |
| CovProjector + fs_volumes | 0.0941 ± 0.0107 | 0.8350 ± 0.0049 | 0.0137 | 0.7186 | 0.0390 |
| CovProjector + fs_all | 0.0941 ± 0.0132 | 0.8348 ± 0.0051 | 0.0137 | 0.7107 | 0.0384 |
| CrossModal_PCA_PLS_learnable | 0.0924 ± 0.0272 | 0.8365 ± 0.0051 | 0.0136 | 0.6915 | 0.0333 |
| CrossModal_PCA_PLS (closed-form) | 0.0904 ± 0.0101 | 0.8344 ± 0.0065 | 0.0138 | 0.7037 | 0.0338 |
| **Krakencoder** (precomputed) | 0.0850 ± 0.0045 | 0.8325 ± 0.0040 | 0.0139 | **0.7485** | **0.0554** |
| CrossModal_PLS_SVD | 0.0727 ± 0.0139 | 0.8228 ± 0.0111 | 0.0148 | 0.6525 | 0.0354 |
| Sarwar2020 MLP | 0.0636 ± 0.0118 | 0.8204 ± 0.0129 | 0.0153 | 0.6376 | 0.0302 |
| Chen2024 GCN | 0.0186 ± 0.0051 | 0.7487 ± 0.0181 | 0.0199 | 0.5688 | 0.0063 |
| CrossModalPCA (no cross-map) | 0.0116 ± 0.0080 | 0.8177 ± 0.0104 | 0.0151 | 0.5220 | 0.0097 |
| Nodal GNN | 0.0080 ± 0.0154 | 0.7704 ± 0.0055 | 0.0188 | 0.5490 | 0.0113 |

## 2.2 Input source type (10 seeds, demeaned r)

Does the SC representation matter? (`SC` = SIFT2-weighted + inverse-node-volume streamline connectome — see [sc_reconstruction.md](sc_reconstruction.md); `SC_r2t` = region-to-tract correlation; `FC` = functional input *oracle*.)

| Model | SC | SC_r2t | SC+SC_r2t | FC (oracle) |
|---|---|---|---|---|
| CrossModalPCA | 0.0116 | 0.0032 | — | **0.6914** |
| CrossModal_PLS_SVD | 0.0727 | 0.0385 | — | — |
| CrossModal_PCA_PLS | 0.0904 | 0.0476 | 0.0786 | — |
| CrossModal_PCA_PLS_learnable | 0.0924 | 0.0333 | 0.0827 | — |
| Krakencoder | 0.0850 | — | — | — |

Takeaways: **SC beats SC_r2t**; combining them does not beat SC alone; the **FC-input oracle (0.69)** dwarfs every SC-input result — the ceiling is set by modality, not model. (Many cells are MISSING — only `CrossModalPCA` has the FC oracle, Krakencoder only ran on SC.)

## 2.3 Identifiability / Cohen's d (single-run notebook values)

⚠️ These are **single-run** values from individual `model_testing` notebooks (not 10-seed aggregates), and Cohen's d is **not** in the scrape tables. Use as indicative, not definitive. All are *demeaned*.

| Model | r_intra | r_inter | Cohen's d | p |
|---|---|---|---|---|
| Krakencoder (canonical JSON) | 0.0829 | 0.0003 | 0.800 | 1.1e-22 |
| Conditional Gaussian (PCA-latent) | — | — | 0.76 | 4.5e-21 |
| Conditional Gaussian (raw edges) | — | — | 0.65 | 1.8e-16 |
| CovProjector + demo | 0.1006 | 0.0019 | 0.55 | 6.4e-13 |
| LatentAttnMasked (attn-only) | 0.0736 | 0.0006 | 0.50 | 4.9e-11 |
| Sarwar2020 MLP | 0.0686 | 0.0101 | 0.36 | 1.2e-06 |

## 2.4 Hungarian matching — Krakencoder, demeaned (canonical)

Single-subset (n=195): pFC **0.051**, null(noise) 0.010, null(permute) 0.005.

Sample-size sweep (demeaned), **8/8 sizes significant vs both nulls**:

| n | 2 | 12 | 22 | 32 | 42 | 52 | 62 | 72 |
|---|---|---|---|---|---|---|---|---|
| pFC | 0.807 | 0.322 | 0.234 | 0.183 | 0.157 | 0.137 | 0.125 | 0.114 |
| Null(noise) | 0.528 | 0.106 | 0.063 | 0.048 | 0.038 | 0.033 | 0.029 | 0.026 |
| Null(permute) | 0.493 | 0.085 | 0.044 | 0.031 | 0.023 | 0.020 | 0.016 | 0.014 |

## 2.5 Reference bounds

| Bound | Demeaned r | Raw r | Avg rank | Top-1 | Notes |
|---|---|---|---|---|---|
| FC→FC self-recon (oracle) | ~0.69 | ~0.92 | 1.00 | 1.00 | `CrossModalPCA` w/ FC input, 10-seed |
| Test–retest (oracle) | ~0.50 | ~0.82 | ~0.99 | ~0.96 | single-run smoke estimate |
| Null (mean+noise / permute) | ~0.00 | n/a | ~0.50 | ~0.005 | chance floor |

---

# Part 3 — Reading the results

## The one pattern
Every SC→FC model — closed-form, learnable, VAE, latent attention/transformer, GNN, MLP, Krakencoder — converges to **raw r ≈ 0.82–0.84, demeaned r ≈ 0.07–0.10, R² ≈ 0**. Architecture barely moves the headline. The two things that *do* move it:
1. **Modality** — FC input (oracle) jumps demeaned to ~0.69; SC caps out ~0.10.
2. **Covariates** — adding demographics nudges 0.092 → **0.103** (the current best).

This matches the Zalesky/Smolders literature: **the individual-signal bottleneck is information content, not model capacity.** The `masked_mlp_pretraining` diagnostic independently concluded the plateau is *structural* (PCA decorrelation), not architectural.

## Per-model "why"
- **CovProjector + demo (best, 0.103):** the only model that injects *new information* (demographics) rather than reshaping SC — consistent with the "information, not architecture" thesis.
- **PCA_PLS family (0.090–0.092):** the strong, simple closed-form/learnable bridge; learnable barely beats closed-form. The workhorse baseline.
- **Krakencoder (0.085):** best **identifiability** (avg_rank 0.749, top-1 0.055) despite mid-pack demeaned r — its contrastive latent optimizes for individuation. Lowest variance across seeds (±0.0045).
- **PLS_SVD (0.073) / Sarwar MLP (0.064):** middle tier; the MLP wastes capacity on the group mean (trained on raw MSE, not residuals — see [masked_latent_pretrainer.md](masked_latent_pretrainer.md) on demeaning).
- **Chen GCN (0.019) / Nodal GNN (0.008) / plain PCA (0.012):** bottom. GNNs underperform the linear family here; plain `CrossModalPCA` has no cross-map so it carries almost no individual signal (near the null floor).

## Caveats
- **Identifiability/Cohen's d are single-run** notebook values (§2.3), not 10-seed; treat as indicative.
- **Many source-type cells are MISSING** (§2.2): the FC oracle exists only for `CrossModalPCA`; Krakencoder ran only on SC.
- **NodalGNN** results exist in the scrape but its dev notebook crashed (CUDA OOM); the 0.008 demeaned is from completed seeds.
- Per-notebook smoke-test numbers (single seed) often differ slightly from these 10-seed aggregates — prefer the tables here for any citation.
