# Phase 0 snapshots — closed-form baselines, both directions

Fixed reference numbers to diff future runs against. Closed-form models are deterministic given the same data + seed, so these should reproduce exactly.

**Setup:** Glasser (64,620 edges), `shuffle_seed=0`, `data_load_mode=precomputed`, CPU, default YAML configs. 683 train / 79 val / 195 test. Run on Torch in the conda env (py 3.12), via `notebooks/model_overviews/crossmodal_pca_pls_closed_form_overview.ipynb` (toggle `SOURCE, TARGET`).

---

## SC → FC (confirmed reproduced 2026-05-25)

| model | mse | r2 | pearson | avg_rank | top1_acc | demeaned_pearson |
|---|---|---|---|---|---|---|
| CrossModalPCA | 0.015551 | -0.174628 | 0.811684 | 0.537278 | 0.000000 | 0.010411 |
| CrossModal_PLS_SVD | 0.014902 | -0.132517 | 0.819806 | 0.688889 | 0.035897 | 0.079649 |
| CrossModal_PCA_PLS | 0.013715 | -0.041234 | 0.833907 | 0.704379 | 0.051282 | 0.085543 |

## FC → SC (seed 0, 2026-05-25 — first run)

| model | mse | r2 | pearson | avg_rank | top1_acc | demeaned_pearson |
|---|---|---|---|---|---|---|
| CrossModalPCA | 0.011505 | -0.971422 | 0.834551 | 0.516055 | 0.005128 | 0.009206 |
| CrossModal_PLS_SVD | 0.005771 | -0.073301 | 0.909100 | 0.859776 | 0.123077 | 0.131753 |
| CrossModal_PCA_PLS | 0.005479 | -0.009883 | 0.913923 | 0.864142 | 0.153846 | **0.137029** |

---

## Headline comparison (seed 0)

| metric (PCA_PLS) | SC→FC | FC→SC | change |
|---|---|---|---|
| demeaned_pearson | 0.0855 | 0.1370 | **+60%** |
| pearson | 0.834 | 0.914 | +0.08 |
| top1_acc | 0.051 | 0.154 | **3.0×** |
| avg_rank | 0.704 | 0.864 | +0.16 |

For cross-map models (PLS_SVD, PCA_PLS), **FC→SC beats SC→FC on every metric**. FC→SC PCA_PLS demeaned (0.137) also exceeds the best *SC→FC-with-covariates* result (0.103, from the 10-seed scrape). Consistent with the hypothesis: FC = richer input, SC = more reliable target.

**Not a leakage artifact:** `CrossModalPCA` (no learned cross-map, just reuses source coords) FAILS in FC→SC (demeaned 0.009, r² −0.97). If `x`/`y` were the same modality it would be near-perfect — so the gains come only from the learned PLS map.

---

## Open checks before trusting the FC→SC win

- [x] **SC→SC oracle** (the ceiling, seed 0) = **demeaned 0.647** (pearson 0.951, top1 1.0, avg_rank 1.0; CrossModalPCA SC→SC). So FC→SC's 0.137 captures **~21%** of recoverable individual SC → **~4.7× headroom**. Compare SC→FC: 0.086 of FC→FC's 0.69 ≈ 12.5%. FC→SC is *not* near-saturated. (Ceiling at the config's PCA k; true ceiling may be slightly higher with more components.)
- [x] **10-seed robustness** (confirmed 2026-05-26): FC→SC CrossModal_PCA_PLS over seeds 0..9, **demeaned = 0.1341 ± 0.0054**, top1 = 0.1138 ± 0.0181. Per-seed demeaned: [0.137, 0.131, 0.123, 0.134, 0.134, 0.143, 0.141, 0.136, 0.133, 0.130]. Tighter std than expected; seed-0 (0.137) was not lucky — population mean is 0.134. Vs SC→FC PCA_PLS (0.0904 ± 0.0101): gap = 0.044, ~3.8 pooled-σ separation, distributions don't overlap at ±1σ. The 10-seed result confirms FC→SC clears the SC→FC wall robustly, not from one lucky seed.
- [x] **Geometry / brain-size confound** (checked 2026-05-26, **MAJOR FINDING**): **brain-volume-only baseline demeaned = 0.1670** (raw pearson 0.9155) — *ABOVE* FC→SC's 0.1341. A 16-feature FreeSurfer volume predictor (IntraCranial, BrainSeg, cortical/subcortical GM, total WM, etc.) with no functional info beats FC→SC by ~25% on demeaned-r. Implication: most or all of FC→SC's 0.134 plausibly reflects brain anatomy, not individual *connectivity*. The FC→SC > SC→FC asymmetry is still real (0.134 vs 0.090) but the cleanest interpretation is now: *SC's individual variation is more anatomy-driven than FC's*, so a simple anatomy predictor catches a lot of SC's residual. Step 3.5 (residualization, done 2026-05-26): **FC → SC_residual demeaned = 0.0778** (sanity check FC→SC raw via manual pipeline = 0.1322, close to project's 0.1341 ✓). So FC adds 0.078 of *real connectivity signal* above anatomy — modest but non-zero. ~41% of FC→SC's 0.134 was brain-anatomy confound; 59% is genuine FC-specific signal. **Headline reframed**: not "+59% over SC→FC via connectivity," but "FC carries ~0.08 of connectivity signal about SC orthogonal to brain anatomy." Step 3.6 (symmetric, done 2026-05-26): **brain-vol → FC demeaned = 0.0467** (vs brain-vol → SC = 0.1670 — SC is ~3.6× more anatomy-driven than FC, biologically obvious in hindsight). **SC → FC_residual demeaned = 0.0561** (sanity: manual SC→FC raw = 0.0849 vs project 0.0904 ✓). So in the anatomy-controlled comparison: **FC→SC_residual (0.0778) > SC→FC_residual (0.0561), ratio 1.39×** — the directional asymmetry SURVIVES anatomy control. Down from the raw 1.48× lift, but the pivot's mechanistic premise holds. **Final headline**: FC→SC has a real, anatomy-orthogonal individual-connectivity advantage of ~40% over SC→FC.
- [ ] **SC-target null floor:** group-average SC baseline (should be ~0 demeaned).

Update this file as each check completes.
